"""Agent detection using Python AST parsing with a regex fall back for non python code.

Detects explicit agent class definitions and instantiations in Python source code.
Includes framework-specific pattern matching for known agent frameworks.
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, cast

from src.detectors.agents.structured_agents import StructuredAgentDetector
from src.detectors.keywords import KeywordsConfig, load_keywords

_DEFAULT_KEYWORDS_PATH = Path(__file__).parent.parent.parent / "config" / "keywords.json"

logger = logging.getLogger(__name__)


class AgentDetector:
    """Detect and count explicit agent instances in repository files using AST parsing.

    Detection strategy:
        - Python files: AST parsing to find class definitions ending with Agent,
          instantiation calls like Agent(...) or SomeAgent(...), and framework-specific
          patterns (ChatOpenAI, ConversableAgent, Task, Crew, initialise_agent, etc.).
        - Non-Python files: Simple regex-based detection for Agent-like tokens.

    The detector is framework-agnostic but includes specific patterns for popular frameworks.
    """

    _cached_keywords: KeywordsConfig | None = None
    _DETECTION_PRIORITY = {
        "class": 100,
        "inheritance": 95,
        "baseagent_instantiation": 90,
        "call": 85,
        "framework": 80,
        "composition": 75,
        "agent_composition": 70,
        "imported_agent": 65,
        "agentic_method": 60,
        "llm_tools": 55,
        "autonomous": 50,
        "structured": 40,
        "text": 10,
    }

    @staticmethod
    def _normalise_agent_name(name: str | None) -> str:
        if not name:
            return ""
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def __init__(self, keywords_path: str | None = None) -> None:
        """Initialise the agent detector.

        Args:
            keywords_path: Optional path to keywords configuration file. If None, uses default.

        Raises:
            ValueError: If keywords_path is provided but is empty or whitespace only.
        """
        if keywords_path is not None and not keywords_path.strip():
            msg = "keywords_path cannot be empty or whitespace"
            raise ValueError(msg)

        if AgentDetector._cached_keywords is None or (keywords_path is not None):
            AgentDetector._cached_keywords = load_keywords(keywords_path or str(_DEFAULT_KEYWORDS_PATH))
        keywords: KeywordsConfig = AgentDetector._cached_keywords
        self.structured_detector = StructuredAgentDetector()

        self._framework_patterns = keywords.get("agent_instantiation_patterns", [])
        self._agent_base_classes = keywords.get("agent_base_classes", [])
        self._framework_modules = frozenset(keywords.get("framework_modules", []))
        self._strong_agentic_methods = frozenset(keywords.get("strong_agentic_methods", []))
        self._weak_agentic_methods = frozenset(keywords.get("weak_agentic_methods", []))
        self._skip_methods = frozenset(keywords.get("skip_methods", []))
        self._llm_parameter_names = frozenset(keywords.get("llm_parameter_names", []))
        self._agent_parameter_names = frozenset(keywords.get("agent_parameter_names", []))
        self._tools_parameter_names = frozenset(keywords.get("tools_parameter_names", []))
        self._llm_call_patterns = frozenset(keywords.get("llm_call_patterns", []))
        self._llm_provider_methods = frozenset(keywords.get("llm_provider_methods", []))
        self._llm_provider_modules = frozenset(keywords.get("llm_provider_modules", []))
        self._generic_role_names = frozenset(keywords.get("generic_role_names", []))
        self._setup_method_names = frozenset(keywords.get("setup_method_names", []))

        settings = keywords.get("settings", {}) if isinstance(keywords, dict) else {}
        self._max_generic_name_length = (
            settings.get("max_generic_name_length", 20) if isinstance(settings, dict) else 20
        )
        self._max_agent_name_length = settings.get("max_agent_name_length", 20) if isinstance(settings, dict) else 20
        self._user_defined_agent_factories = frozenset({"_agent", "agent_factory"})
        self._generic_class_re = re.compile(r"class\s+(\w+Agent)\b")
        self._word_agent_re = re.compile(r"\b(\w+Agent)\b")
        self._agent_pattern = re.compile(r"Agent\d*$")
        self._yaml_available = True
        try:
            import yaml  # noqa: F401
        except ImportError:
            self._yaml_available = False
            debug_msg = "PyYAML not installed, YAML agent detection disabled"
            logger.debug(debug_msg)

    @staticmethod
    def _extract_base_class_names(bases: list[Any]) -> set[str]:
        """Extract base class names from ClassDef.bases.

        Args:
            bases: List of base node expressions from ClassDef.bases.

        Returns:
            Set of base class names found.
        """
        base_names: set[str] = set()
        for base in bases:
            if isinstance(base, ast.Name):
                base_names.add(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.add(base.attr)
        return base_names

    def _inherits_from_agent_base(self, base_names: set[str], tree: ast.AST) -> bool:
        """Check if class inherits from known agent base classes.

        Args:
            base_names: Set of direct base class names.
            tree: The parsed AST tree for module-level lookup.

        Returns:
            True if any base class matches known agent base classes.
        """
        for base_name in base_names:
            if base_name in self._agent_base_classes:
                return True
            module = self._get_import_module(tree, base_name)
            if module and any(agent_base in base_name for agent_base in self._agent_base_classes):
                return True
        return False

    @staticmethod
    def _extract_callee_name(node: ast.Call) -> str | None:
        """Extract a sensible callee name from a Call node's func.

        Args:
            node: An ast.Call node to extract the callee name from.

        Returns:
            The function or method name being called, or None if extraction fails.

        Handles simple names and attributes. For nested calls like
        `factory().create_agent()` the final attribute name (`create_agent`)
        will be returned which is the most useful heuristic here.
        """
        fn = getattr(node, "func", None)
        if fn is None:
            return None
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return fn.attr
        return None

    def _is_agentish_symbol(self, name: str | None, module_name: str | None = None) -> bool:
        """Heuristic: does this symbol plausibly refer to an agent construct?"""
        if not name:
            return False

        if self._looks_agent_like(name):
            return True

        if module_name and ("agent" in module_name.lower() or "agents" in module_name.lower()):
            return True

        return name in {"Agent", "BaseAgent", "ParallelAgent", "SequentialAgent"}

    def _is_llm_provider_method(self, fn_name: str, framework_imports: set[str]) -> bool:
        """Check if a method name is from an LLM provider client.

        Args:
            fn_name: The method name being called.
            framework_imports: Set of imported framework modules.

        Returns:
            True if the method is from an LLM provider client that indicates agent usage.
        """
        for framework_import in framework_imports:
            for module in self._llm_provider_modules:
                if module in framework_import:
                    return fn_name in self._llm_provider_methods

        return False

    @staticmethod
    def _extract_class_from_method_call(node: ast.Call) -> str | None:
        """Extract class name from class method calls like ClassName.method().

        Args:
            node: An ast.Call node.

        Returns:
            Class name if it's a class method call, None otherwise.
        """
        fn = getattr(node, "func", None)
        if fn and isinstance(fn, ast.Attribute):
            value = getattr(fn, "value", None)
            if isinstance(value, ast.Name):
                return value.id
        return None

    def _matches_framework_pattern(self, fn_name: str, module_name: str | None = None) -> bool:
        """Check if a function/class name matches known framework patterns.

        Args:
            fn_name: Name of the function or class being called.
            module_name: Optional module name for imports.

        Returns:
            True if matches any framework pattern.
        """
        if isinstance(self._framework_patterns, list):
            return fn_name in self._framework_patterns
        if not isinstance(self._framework_patterns, dict):
            return False
        for _framework, patterns in self._framework_patterns.items():
            if self._check_pattern_match(patterns, fn_name, module_name):
                return True
        return False

    def _check_pattern_match(self, patterns: object, fn_name: str, module_name: str | None) -> bool:
        """Check if patterns match the function or module name.

        Args:
            patterns: Pattern to check against.
            fn_name: Name of the function or class being called.
            module_name: Optional module name for imports.

        Returns:
            True if patterns match.
        """
        if isinstance(patterns, list):
            return fn_name in patterns
        if isinstance(patterns, dict):
            return self._check_dict_patterns(patterns, fn_name, module_name)
        if isinstance(patterns, str):
            return fn_name == patterns or (module_name is not None and module_name == patterns)
        return False

    @staticmethod
    def _check_dict_patterns(patterns: dict, fn_name: str, module_name: str | None) -> bool:
        """Check if dictionary patterns match the function or module name.

        Args:
            patterns: Dictionary of patterns.
            fn_name: Name of the function or class being called.
            module_name: Optional module name for imports.

        Returns:
            True if patterns match.
        """
        if fn_name in patterns.get("names", []):
            return True
        if module_name and module_name in patterns.get("modules", []):
            return True
        return False

    @staticmethod
    def _check_alias_match(alias: ast.alias, name: str) -> bool:
        """Check if an import alias matches the target name.

        Args:
            alias: The import alias to check.
            name: The name to match against.

        Returns:
            True if alias matches the name.
        """
        return alias.name == name or alias.asname == name

    @staticmethod
    def _find_matching_alias(
        aliases: list[ast.alias],
        name: str,
    ) -> ast.alias | None:
        """Find a matching import alias."""
        for alias in aliases:
            if AgentDetector._check_alias_match(alias, name):
                return alias

        return None

    @staticmethod
    def _get_import_module(tree: ast.AST, name: str) -> str | None:
        """Extract module name for an imported name from the AST tree.

        Args:
            tree: The parsed AST tree.
            name: The name to look up in imports.

        Returns:
            Module name if found, None otherwise.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                alias = AgentDetector._find_matching_alias(
                    getattr(node, "names", []),
                    name,
                )
                if alias:
                    return getattr(node, "module", None)

            elif isinstance(node, ast.Import):
                alias = AgentDetector._find_matching_alias(
                    getattr(node, "names", []),
                    name,
                )
                if alias:
                    return alias.name

        return None

    def _detect_imported_agents(self, tree: ast.AST) -> set[str]:
        """Find all agent-like imports in a file.

        Returns: Set of detected agent names
        """
        imported_agents = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_path = node.module or ""

                if "agent" in module_path.lower() or any(
                    pattern in module_path.lower() for pattern in ["multiagent", "agents"]
                ):
                    for alias in getattr(node, "names", []):
                        imported_name = alias.name
                        final_name = alias.asname or imported_name

                        if self._looks_agent_like(final_name):
                            imported_agents.add(final_name)

            if isinstance(node, ast.Import):
                for alias in getattr(node, "names", []):
                    module_name = alias.name
                    final_name = alias.asname or module_name.split(".")[-1]

                    if "agent" in module_name.lower() and self._looks_agent_like(final_name):
                        imported_agents.add(final_name)

        return imported_agents

    @staticmethod
    def _looks_agent_like(name: str) -> bool:
        """Check if a name resembles an agent.

        Suffix-based check reduces generic matches like agent_session_id while
        keeping common agent identifiers.
        """
        lower = name.lower()
        return lower.endswith(("agent", "_agent"))

    def _detect_in_nested_structures(self, tree: ast.AST) -> list[tuple[str, str]]:
        """Find agents assigned to nested data structures (dicts, lists, etc.)

        Returns: List of (container_path, agent_name) tuples
        """
        nested_agents = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for value in node.values:
                    if isinstance(value, ast.Call):
                        agent_name = self._extract_callee_name(value)
                        if agent_name and self._looks_agent_like(agent_name):
                            nested_agents.append(("dict_value", agent_name))

            if isinstance(node, ast.List):
                for element in node.elts:
                    if isinstance(element, ast.Call):
                        agent_name = self._extract_callee_name(element)
                        if agent_name and self._looks_agent_like(agent_name):
                            nested_agents.append(("list_element", agent_name))

        return nested_agents

    def _detect_agents_from_type_hints(self, tree: ast.AST) -> set[str]:
        """Find agents referenced in type annotations.

        For example, in:
            class OrchestratorAgent(BaseAgent):
                def __init__(self, ipai_agent: IPAIAgent, guidance_agent: Agent):

        This detects IPAIAgent and Agent.
        """
        agents_from_hints = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.annotation:
                        type_name = self._extract_annotation_name(arg.annotation)
                        if type_name and self._looks_agent_like(type_name):
                            agents_from_hints.add(type_name)

            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        type_name = self._extract_annotation_name(item.annotation)
                        if type_name and self._looks_agent_like(type_name):
                            agents_from_hints.add(type_name)

        return agents_from_hints

    def _extract_annotation_name(self, annotation: ast.expr) -> str | None:
        """Extract type name from annotation."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return annotation.attr
        elif isinstance(annotation, ast.Subscript):
            return self._extract_annotation_name(annotation.value)
        return None

    def _detect_agents_via_regex(self, text: str) -> list[str]:
        """Detect agent-like tokens in arbitrary text using regex heuristics.

        Args:
            text: Text content to search for agent-like patterns.

        Returns:
            List of unique matched token strings (e.g., "MyAgent", "Agent") in first-seen order.
        """
        if not text:
            return []

        seen: set[str] = set()
        names: list[str] = []

        for line in text.splitlines():
            for m in self._word_agent_re.finditer(line):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    names.append(name)

            for m in self._generic_class_re.finditer(line):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    names.append(name)

        return names

    @staticmethod
    def _extract_call_kwargs(node: ast.Call) -> dict[str, str]:
        """Extract keyword argument names and their string representations.

        Args:
            node: An ast.Call node.

        Returns:
            Dictionary mapping kwarg names to their string representations.
            If ast.unparse is unavailable, the argument name is used as a placeholder.
        """
        kwargs: dict[str, str] = {}
        for keyword in getattr(node, "keywords", []):
            arg_name = getattr(keyword, "arg", None)
            if arg_name:
                if hasattr(ast, "unparse"):
                    kwargs[arg_name] = cast(str, ast.unparse(keyword.value))
                else:
                    kwargs[arg_name] = arg_name
        return kwargs

    @staticmethod
    def _extract_name_from_call(node: ast.Call) -> str | None:
        """Extract the 'name' parameter value from an agent instantiation call.

        Args:
            node: An ast.Call node.

        Returns:
            String value of the 'name' parameter if present, None otherwise.
        """
        for keyword in getattr(node, "keywords", []):
            arg_name = getattr(keyword, "arg", None)
            if arg_name == "name":
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return cast(str, value.value)
                if isinstance(value, ast.Str):
                    return cast(str, value.s)
        return None

    def _detect_llm_composition(self, node: ast.Call) -> bool:
        """Check if a call uses LLM or agent composition patterns.

        Detects calls with llm=, model=, agents=, or similar parameters that indicate
        an LLM instance or agent collection is being passed to an agent-like constructor.

        Args:
            node: An ast.Call node.

        Returns:
            True if LLM or agent composition pattern detected.
        """
        kwargs = self._extract_call_kwargs(node)
        return bool((self._llm_parameter_names | self._agent_parameter_names) & set(kwargs.keys()))

    def _get_framework_imports(self, tree: ast.AST) -> set[str]:
        """Extract framework imports known for agents.

        Args:
            tree: The parsed AST tree.

        Returns:
            Set of imported framework modules related to agents.
        """
        imported_frameworks: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = getattr(node, "module", "")
                if module and any(module.startswith(fw) or fw.startswith(module) for fw in self._framework_modules):
                    imported_frameworks.add(module)
            elif isinstance(node, ast.Import):
                for alias in getattr(node, "names", []):
                    if any(alias.name.startswith(fw) or fw.startswith(alias.name) for fw in self._framework_modules):
                        imported_frameworks.add(alias.name)

        return imported_frameworks

    def _is_agent_composition_call(
        self, fn_name: str, node: ast.Call, framework_imports: set[str], tree: ast.AST
    ) -> bool:
        """Check if a call represents agent composition (LLM + constructor pattern).

        Args:
            fn_name: Name of the function/class being called.
            node: The ast.Call node.
            framework_imports: Set of framework modules imported.
            tree: The parsed AST tree.

        Returns:
            True if the call matches agent composition patterns.
        """
        fn_module = self._get_import_module(tree, fn_name)
        if fn_module and any(fn_module.startswith(fw) for fw in framework_imports):
            return True

        if self._detect_llm_composition(node) and framework_imports:
            return True

        return False

    def _has_agentic_methods(self, class_node: ast.ClassDef) -> bool:
        """Check if a class has agentic method names.

        Args:
            class_node: An ast.ClassDef node.

        Returns:
            True if the class contains methods with agentic names.
        """
        found_strong = False
        found_weak = False
        class_name = getattr(class_node, "name", "")

        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = getattr(item, "name", "")
                if method_name.startswith("__") and method_name.endswith("__"):
                    continue
                clean_name = method_name.lstrip("_")
                if clean_name in self._skip_methods:
                    continue
                if clean_name in self._strong_agentic_methods:
                    found_strong = True
                if clean_name in self._weak_agentic_methods:
                    found_weak = True

        if found_strong:
            return True

        if found_weak:
            name_lower = class_name.lower()
            if "agent" in name_lower or "assistant" in name_lower or "bot" in name_lower:
                return True

            has_llm, _ = self._check_class_attributes(class_node, check_llm=True, check_tools=False)
            if has_llm:
                return True

        return False

    def _detect_baseagent_subclass_instantiations(
        self, tree: ast.AST, seen: set[tuple[Any, ...]]
    ) -> list[dict[str, Any]]:
        """Detect instantiations of custom BaseAgent subclasses with strong confidence.

        Targets the pattern:
            class MyCustomAgent(BaseAgent):
                def __init__(self, name: str, ...):
                    super().__init__(name=name, ...)

            my_agent = MyCustomAgent(name="...")

        Only detects if:
        - Class inherits from BaseAgent

        Args:
            tree: The parsed AST tree.
            seen: Set of already seen (type, name, lineno) tuples to avoid duplicates.

        Returns:
            List of location dictionaries for custom agent instantiations.
        """
        locations: list[dict[str, Any]] = []

        baseagent_subclasses: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            name = getattr(node, "name", "")
            bases = getattr(node, "bases", [])
            base_names = self._extract_base_class_names(bases)

            if "BaseAgent" in base_names:
                baseagent_subclasses.add(name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn_name = self._extract_callee_name(node)
            if not fn_name or fn_name not in baseagent_subclasses:
                continue

            lineno = getattr(node, "lineno", None)

            self._add_location_if_new(seen, locations, "baseagent_instantiation", fn_name, lineno)

        return locations

    def _detect_agent_assignments_from_modules(self, tree: ast.AST, seen: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """Detect agent variable assignments imported from other modules.

        Detects assignments where:
        - Variable name contains "agent" (case-insensitive)
        - RHS is either an imported agent name or an agent-like call

        Args:
            tree: The parsed AST tree.
            seen: Set of already seen (type, name, lineno) tuples to avoid duplicates.

        Returns:
            List of location dictionaries for imported agents used in composition.
        """
        locations: list[dict[str, Any]] = []

        imported_agent_names: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = getattr(node, "module", "")
                if not module or "agent" not in module.lower():
                    continue

                for alias in getattr(node, "names", []):
                    name = alias.asname or alias.name
                    if name and self._looks_agent_like(name):
                        imported_agent_names[name] = module

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue

            for target in getattr(node, "targets", []):
                if not isinstance(target, ast.Name):
                    continue

                target_name = target.id
                if not self._looks_agent_like(target_name):
                    continue

                if isinstance(node.value, ast.Name):
                    rhs_name = node.value.id
                    if rhs_name in imported_agent_names:
                        lineno = getattr(node, "lineno", None)
                        self._add_location_if_new(seen, locations, "imported_agent", target_name, lineno)

                elif isinstance(node.value, ast.Call):
                    fn_name = self._extract_callee_name(node.value)
                    if fn_name and self._looks_agent_like(fn_name):
                        lineno = getattr(node, "lineno", None)
                        self._add_location_if_new(seen, locations, "imported_agent", target_name, lineno)

        return locations

    def _detect_composed_agent_calls(self, tree: ast.AST, seen: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """Detect agent composition patterns with strong confidence.

        Only detects if:
        - Constructor has a 'name' parameter (agent idiom)
        - Has at least one parameter containing "agent" (composition pattern)
        - RHS is a capitalised call (likely a class)

        Args:
            tree: The parsed AST tree.
            seen: Set of already seen (type, name, lineno) tuples to avoid duplicates.

        Returns:
            List of location dictionaries for composed agent instantiations.
        """
        locations: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn_name = self._extract_callee_name(node)
            fn_module = self._get_import_module(tree, fn_name) if fn_name else None
            if not fn_name or not fn_name[0].isupper():
                continue

            kwargs = self._extract_call_kwargs(node)

            if "name" not in kwargs:
                continue

            agent_params = [k for k in kwargs.keys() if k.lower() in {"agent", "agents"}]
            if not agent_params:
                continue

            if not self._is_agentish_symbol(fn_name, fn_module):
                continue

            lineno = getattr(node, "lineno", None)

            self._add_location_if_new(seen, locations, "agent_composition", fn_name, lineno)

        return locations

    def _check_assignment_for_params(
        self, node: ast.Assign | ast.AnnAssign, check_llm: bool, check_tools: bool
    ) -> tuple[bool, bool]:
        """Check if an assignment contains LLM or tools parameters.

        Args:
            node: An assignment node (Assign or AnnAssign).
            check_llm: Whether to check for LLM-related parameters.
            check_tools: Whether to check for tools-related parameters.

        Returns:
            Tuple of (has_llm, has_tools) booleans.
        """
        has_llm = False
        has_tools = False

        targets = getattr(node, "targets", []) if isinstance(node, ast.Assign) else [getattr(node, "target", None)]

        for target in targets:
            if target is None:
                continue

            attr_name = self._extract_attribute_name(target)
            if not attr_name:
                continue

            if check_llm and attr_name in self._llm_parameter_names:
                has_llm = True
            if check_tools and attr_name in self._tools_parameter_names:
                has_tools = True

        return has_llm, has_tools

    @staticmethod
    def _extract_attribute_name(target: ast.expr) -> str | None:
        """Extract attribute or variable name from an AST target node.

        Args:
            target: An AST expression node (Attribute or Name).

        Returns:
            The attribute or variable name, or None if not extractable.
        """
        if isinstance(target, ast.Attribute):
            return getattr(target, "attr", None)
        if isinstance(target, ast.Name):
            return getattr(target, "id", None)
        return None

    def _check_function_args_for_params(
        self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, check_llm: bool, check_tools: bool
    ) -> tuple[bool, bool]:
        """Check if a function has LLM or tools parameters.

        Args:
            func_node: A function definition node.
            check_llm: Whether to check for LLM-related parameters.
            check_tools: Whether to check for tools-related parameters.

        Returns:
            Tuple of (has_llm, has_tools) booleans.
        """
        args = getattr(func_node, "args", None)
        if not args:
            return False, False

        has_llm = False
        has_tools = False

        for arg in getattr(args, "args", []):
            arg_name = getattr(arg, "arg", None)
            if not arg_name:
                continue

            if check_llm and arg_name in self._llm_parameter_names:
                has_llm = True
            if check_tools and arg_name in self._tools_parameter_names:
                has_tools = True

        return has_llm, has_tools

    def _check_function_body_for_params(
        self, func_node: ast.FunctionDef | ast.AsyncFunctionDef, check_llm: bool, check_tools: bool
    ) -> tuple[bool, bool]:
        """Check function body for dictionary get() calls with LLM/tools keys.

        Args:
            func_node: A function definition node.
            check_llm: Whether to check for LLM-related parameters.
            check_tools: Whether to check for tools-related parameters.

        Returns:
            Tuple of (has_llm, has_tools) booleans.
        """
        has_llm = False
        has_tools = False

        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue

            func = getattr(node, "func", None)
            if not isinstance(func, ast.Attribute) or getattr(func, "attr", None) != "get":
                continue

            args = getattr(node, "args", [])
            if not args or not isinstance(args[0], ast.Constant):
                continue

            arg_value = getattr(args[0], "value", None)
            if check_llm and arg_value in self._llm_parameter_names:
                has_llm = True
            if check_tools and arg_value in self._tools_parameter_names:
                has_tools = True

        return has_llm, has_tools

    def _check_class_attributes(
        self, class_node: ast.ClassDef, check_llm: bool = True, check_tools: bool = False
    ) -> tuple[bool, bool]:
        """Check if a class has LLM and/or tools attributes.

        Args:
            class_node: An ast.ClassDef node.
            check_llm: Whether to check for LLM-related attributes.
            check_tools: Whether to check for tools-related attributes.

        Returns:
            Tuple of (has_llm, has_tools) booleans.
        """
        has_llm = False
        has_tools = False

        for item in class_node.body:
            if isinstance(item, (ast.Assign, ast.AnnAssign)):
                llm_found, tools_found = self._check_assignment_for_params(item, check_llm, check_tools)
                has_llm = has_llm or llm_found
                has_tools = has_tools or tools_found

            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                llm_found, tools_found = self._check_method_item(item, check_llm, check_tools)
                has_llm = has_llm or llm_found
                has_tools = has_tools or tools_found

            elif check_tools and has_llm and isinstance(item, ast.ClassDef):
                tools_found = self._check_nested_class_for_tools(item)
                has_tools = has_tools or tools_found

        return has_llm, has_tools

    def _check_method_item(
        self, item: ast.FunctionDef | ast.AsyncFunctionDef, check_llm: bool, check_tools: bool
    ) -> tuple[bool, bool]:
        """Check a method for LLM/tools parameters.

        Args:
            item: Method node to check.
            check_llm: Whether to check for LLM parameters.
            check_tools: Whether to check for tools parameters.

        Returns:
            Tuple of (has_llm, has_tools) booleans.
        """
        has_llm = False
        has_tools = False
        method_name = getattr(item, "name", "")

        if method_name in self._setup_method_names:
            llm_found, tools_found = self._check_function_args_for_params(item, check_llm, check_tools)
            has_llm = has_llm or llm_found
            has_tools = has_tools or tools_found

        if check_tools:
            llm_found, tools_found = self._check_function_body_for_params(item, check_llm, check_tools)
            has_llm = has_llm or llm_found
            has_tools = has_tools or tools_found

        if check_llm and method_name in {"run", "execute", "invoke", "call", "stream"}:
            llm_found, _ = self._check_function_args_for_params(item, check_llm, False)
            has_llm = has_llm or llm_found

        return has_llm, has_tools

    def _check_nested_class_for_tools(self, nested_class: ast.ClassDef) -> bool:
        """Check nested class setup methods for tools parameters.

        Args:
            nested_class: Nested class definition to check.

        Returns:
            True if tools parameters found.
        """
        for nested_item in nested_class.body:
            if not isinstance(nested_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if getattr(nested_item, "name", "") in self._setup_method_names:
                _, tools_found = self._check_function_args_for_params(nested_item, False, True)
                if tools_found:
                    return True

        return False

    def _has_autonomous_loop(self, class_node: ast.ClassDef) -> bool:
        """Check if a class has autonomous loops with LLM calls.

        Args:
            class_node: An ast.ClassDef node.

        Returns:
            True if the class contains loops with LLM call patterns.
        """
        for item in getattr(class_node, "body", []):
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if self._method_has_loop_with_llm(item):
                return True

        return False

    def _method_has_loop_with_llm(self, method_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a method contains a loop with LLM calls.

        Args:
            method_node: A function or async function definition node.

        Returns:
            True if the method contains loops with LLM call patterns.
        """
        for node in ast.walk(method_node):
            if not isinstance(node, (ast.While, ast.For, ast.AsyncFor)):
                continue

            if self._loop_contains_llm_call(node):
                return True

        return False

    def _loop_contains_llm_call(self, loop_node: ast.While | ast.For | ast.AsyncFor) -> bool:
        """Check if a loop contains LLM calls.

        Args:
            loop_node: A loop node (While, For, or AsyncFor).

        Returns:
            True if the loop contains any LLM call patterns.
        """
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Call):
                callee_name = self._extract_callee_name(node)
                if callee_name and callee_name in self._llm_call_patterns:
                    return True

        return False

    def _extract_user_defined_agent_factories(self, tree: ast.AST) -> set[str]:
        """Extract user-defined agent factory function names from AST.

        Args:
            tree: The parsed AST tree.

        Returns:
            Set of user-defined function names ending with '_agent' or 'agent_factory'.
        """
        factories: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = getattr(node, "name", "")
                if any(name.endswith(suffix) for suffix in self._user_defined_agent_factories):
                    factories.add(name)
        return factories

    def _should_detect_as_agent_class(self, node: ast.ClassDef, tree: ast.AST) -> bool:
        """Determine if a class definition should be detected as an agent.

        Args:
            node: ClassDef node to check.
            tree: Full AST tree for context.

        Returns:
            True if class should be detected as an agent.
        """
        name = getattr(node, "name", "")

        if self._agent_pattern.search(name):
            return True

        bases = getattr(node, "bases", [])
        base_names = self._extract_base_class_names(bases)
        if self._inherits_from_agent_base(base_names, tree):
            return True

        if self._has_agentic_methods(node):
            return True

        has_llm, has_tools = self._check_class_attributes(node, check_llm=True, check_tools=True)
        if has_llm and has_tools:
            return True

        if self._has_autonomous_loop(node):
            return True

        return False

    def _detect_agent_classes(self, tree: ast.AST) -> set[str]:
        """Detect agent class definitions in an AST tree.

        Args:
            tree: The parsed AST tree.

        Returns:
            Set of detected agent class names.
        """
        agent_names: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            if self._should_detect_as_agent_class(node, tree):
                agent_names.add(getattr(node, "name", ""))

        return agent_names

    def _should_add_agent_call(
        self,
        fn_name: str,
        class_name: str | None,
        node: ast.Call,
        framework_imports: set[str],
        user_defined_functions: set[str],
        tree: ast.AST,
    ) -> bool:
        """Determine if a function call should be detected as an agent instantiation.

        Args:
            fn_name: Function name being called.
            class_name: Class name if it's a class method call.
            node: The Call node.
            framework_imports: Set of framework modules imported.
            user_defined_functions: Set of user-defined function names.
            tree: The parsed AST tree.

        Returns:
            True if call should be detected as agent instantiation.
        """
        if class_name and (self._agent_pattern.search(class_name) or self._matches_framework_pattern(class_name)):
            return True

        if not fn_name:
            return False

        if self._agent_pattern.search(fn_name):
            if fn_name in user_defined_functions:
                return False
            return True

        if self._matches_framework_pattern(fn_name):
            if fn_name.endswith("_agent") or fn_name == "agent_factory":
                if fn_name in user_defined_functions:
                    return False
                return True

            if fn_name in self._generic_role_names and len(fn_name) < self._max_generic_name_length:
                return True

            if fn_name not in self._generic_role_names:
                return True

        if fn_name in user_defined_functions:
            return False

        if "Agent" in fn_name and (self._detect_llm_composition(node) or len(fn_name) <= self._max_agent_name_length):
            return True

        if self._detect_llm_composition(node):
            return True

        if self._is_llm_provider_method(fn_name, framework_imports):
            return True

        fn_module = self._get_import_module(tree, fn_name)
        if not self._is_agentish_symbol(fn_name, fn_module):
            return False

        return False

    def _detect_agent_calls(
        self, tree: ast.AST, framework_imports: set[str], user_defined_functions: set[str]
    ) -> set[str]:
        """Detect agent instantiation calls in an AST tree.

        Args:
            tree: The parsed AST tree.
            framework_imports: Set of framework modules imported.
            user_defined_functions: Set of user-defined function names to exclude.

        Returns:
            Set of detected agent names from calls.
        """
        agent_names: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn_name = self._extract_callee_name(node)
            class_name = self._extract_class_from_method_call(node)

            if fn_name and self._should_add_agent_call(
                fn_name, class_name, node, framework_imports, user_defined_functions, tree
            ):
                name_to_add = (
                    class_name
                    if class_name
                    and (self._agent_pattern.search(class_name) or self._matches_framework_pattern(class_name))
                    else fn_name
                )
                if name_to_add:
                    agent_names.add(name_to_add)

        return agent_names

    def count_agents_in_text(self, text: str) -> int:
        """Count agent class definitions and instantiations in source code.

        Args:
            text: Python source code text to analyse.

        Returns:
            Total number of agents detected.
        """
        if not text:
            return 0

        dedented = textwrap.dedent(text)
        if not dedented.strip():
            return 0
        try:
            tree = ast.parse(dedented)
        except SyntaxError as exc:
            logger.debug("Syntax error parsing text, falling back to regex: %s", exc)
            matches = self._detect_agents_via_regex(dedented)
            return len(set(matches))

        framework_imports = self._get_framework_imports(tree)
        user_defined_functions = self._extract_user_defined_agent_factories(tree)
        agent_names = self._detect_agent_classes(tree)
        agent_names.update(self._detect_agent_calls(tree, framework_imports, user_defined_functions))

        return len(agent_names)

    def count_agents_in_file(self, path: str | Path) -> int:
        """Count agents in a file by extension. Uses AST for .py, regex for others.

        Args:
            path: Path to file to analyse.

        Returns:
            Count of detected agents in the file.
        """
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            msg = f"Unable to read file: {file_path}"
            logger.warning(msg)
            return 0

        if file_path.suffix == ".py":
            return self.count_agents_in_text(text)
        return len(set(self._detect_agents_via_regex(text)))

    def get_agent_locations_in_file(self, path: str | Path) -> list[dict[str, Any]]:
        """Get agent locations for a file. Uses AST for .py, regex for others.

        Args:
            path: Path to file to analyse.

        Returns:
            List of detected agent locations with line numbers and detection types.
        """
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            msg = f"Unable to read file: {file_path}"
            logger.warning(msg)
            return []

        if file_path.suffix == ".py":
            return self.get_agent_locations(text)
        locations: list[dict[str, Any]] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            for m in self._word_agent_re.finditer(line):
                locations.append({"line": idx, "name": m.group(1), "detection_type": "regex_line"})
        return locations

    def _detect_agent_class_locations(self, tree: ast.AST, seen: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
        """Detect agent class definition locations in an AST tree.

        Args:
            tree: The parsed AST tree.
            seen: Set of already seen (type, name, lineno) tuples to avoid duplicates.

        Returns:
            List of location dictionaries for agent classes.
        """
        locations: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            name = getattr(node, "name", "")
            lineno = getattr(node, "lineno", None)

            if self._agent_pattern.search(name):
                if self._add_location_if_new(seen, locations, "class", name, lineno):
                    continue

            bases = getattr(node, "bases", [])
            base_names = self._extract_base_class_names(bases)
            if self._inherits_from_agent_base(base_names, tree):
                if self._add_location_if_new(seen, locations, "inheritance", name, lineno):
                    continue

            if self._has_agentic_methods(node):
                if self._add_location_if_new(seen, locations, "agentic_method", name, lineno):
                    continue

            has_llm, has_tools = self._check_class_attributes(node, check_llm=True, check_tools=True)
            if has_llm and has_tools:
                if self._add_location_if_new(seen, locations, "llm_tools", name, lineno):
                    continue

            if self._has_autonomous_loop(node):
                self._add_location_if_new(seen, locations, "autonomous", name, lineno)

        return locations

    @staticmethod
    def _add_location_if_new(
        seen: set[tuple[Any, ...]],
        locations: list[dict[str, Any]],
        detection_type: str,
        name: str,
        lineno: int | None,
    ) -> bool:
        """Add a location if not already seen.

        Args:
            seen: Set of seen (type, name, lineno) tuples.
            locations: List to append location to.
            detection_type: Type of detection (e.g., "class", "inheritance").
            name: Name of the detected agent.
            lineno: Line number of detection.

        Returns:
            True if location was added (not a duplicate).
        """
        key = (detection_type, name, lineno)
        if key in seen:
            return False

        seen.add(key)
        locations.append({"line": lineno, "name": name, "detection_type": detection_type})
        return True

    def get_detection_priority(self, detection_type: str | None) -> int:
        if not detection_type:
            return 0
        if detection_type.startswith("structured_"):
            return self._DETECTION_PRIORITY.get("structured", 40)
        return self._DETECTION_PRIORITY.get(detection_type, 50)

    def _deduplicate_locations(self, locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse duplicate hits on the same line keeping the most specific detection.

        Strategy:
        - Group by line number.
        - Within a line, drop generic names ("Agent") if a more specific name is present.
        - Pick the highest-priority detection; if priority ties, keep the longer name (more specific).
        """
        by_line: dict[int | None, list[dict[str, Any]]] = {}
        for loc in locations:
            line = loc.get("line")
            by_line.setdefault(line, []).append(loc)

        deduped: list[dict[str, Any]] = []
        for _line, locs in by_line.items():
            specific: list[dict[str, Any]] = []
            generic: list[dict[str, Any]] = []

            for loc in locs:
                norm = self._normalise_agent_name(loc.get("name"))
                if norm == "agent":
                    generic.append(loc)
                else:
                    specific.append(loc)

            candidates = specific if specific else generic

            best_loc: dict[str, Any] | None = None
            best_priority = -1
            best_name_len = -1

            for loc in candidates:
                priority = self.get_detection_priority(loc.get("detection_type"))
                name_len = len(loc.get("name") or "")

                if priority > best_priority or (priority == best_priority and name_len > best_name_len):
                    best_priority = priority
                    best_name_len = name_len
                    best_loc = loc

            if best_loc:
                deduped.append(best_loc)

        def _line_key(item: dict[str, Any]) -> int:
            v = item.get("line")
            return v if isinstance(v, int) else -1

        deduped.sort(key=_line_key)
        return deduped

    def _detect_agent_call_locations(
        self,
        tree: ast.AST,
        framework_imports: set[str],
        seen: set[tuple[Any, ...]],
        user_defined_functions: set[str],
        assignment_targets: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect agent instantiation call locations in an AST tree.

        Args:
            tree: The parsed AST tree.
            framework_imports: Set of framework modules imported.
            seen: Set of already seen (type, name, lineno) tuples to avoid duplicates.
            user_defined_functions: Set of user-defined function names to exclude from detection.
            assignment_targets: Optional mapping of call node ids to assignment target names.

        Returns:
            List of location dictionaries for agent calls.
        """
        locations: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn_name = self._extract_callee_name(node)
            if not fn_name:
                continue

            if fn_name in user_defined_functions:
                continue

            lineno = getattr(node, "lineno", None)

            extracted_name = self._extract_name_from_call(node)

            target_name = None
            if assignment_targets is not None:
                target_name = assignment_targets.get(id(node))

            agent_name = extracted_name or target_name or fn_name

            if self._agent_pattern.search(fn_name):
                self._add_location_if_new(seen, locations, "call", agent_name, lineno)
                continue

            if self._matches_framework_pattern(fn_name):
                self._add_location_if_new(seen, locations, "framework", agent_name, lineno)
                continue

            if self._detect_llm_composition(node):
                if self._is_agent_composition_call(fn_name, node, framework_imports, tree):
                    self._add_location_if_new(seen, locations, "composition", agent_name, lineno)

        return locations

    def get_agent_locations(self, text: str) -> list[dict[str, Any]]:
        """Get simple list of agent locations (line numbers and names)."""
        if not text:
            return []

        dedented = textwrap.dedent(text)
        if not dedented.strip():
            return []

        locations: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        try:
            tree = ast.parse(dedented)
        except SyntaxError as exc:
            logger.debug("Syntax error parsing text, falling back to regex: %s", exc)
            for idx, line in enumerate(dedented.splitlines(), start=1):
                for m in self._word_agent_re.finditer(line):
                    name = m.group(1)
                    key = ("text", name, idx)
                    if key not in seen:
                        seen.add(key)
                        locations.append({"line": idx, "name": name, "detection_type": "regex_fallback"})
            return locations

        framework_imports = self._get_framework_imports(tree)

        locations.extend(self._detect_agent_class_locations(tree, seen))
        user_defined_functions = self._extract_user_defined_agent_factories(tree)
        assignment_targets: dict[int, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = getattr(node, "value", None)
                if isinstance(value, ast.Call):
                    target_names = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
                    if target_names:
                        assignment_targets[id(value)] = target_names[0]

            if isinstance(node, ast.AnnAssign):
                value = getattr(node, "value", None)
                target = getattr(node, "target", None)
                if isinstance(value, ast.Call) and isinstance(target, ast.Name):
                    assignment_targets[id(value)] = target.id

        locations.extend(
            self._detect_agent_call_locations(
                tree,
                framework_imports,
                seen,
                user_defined_functions,
                assignment_targets,
            )
        )
        locations.extend(self._detect_baseagent_subclass_instantiations(tree, seen))
        locations.extend(self._detect_agent_assignments_from_modules(tree, seen))
        locations.extend(self._detect_composed_agent_calls(tree, seen))

        return self._deduplicate_locations(locations)

    def get_structured_agent_locations(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Detect agents in YAML/JSON configuration files and Bru HTTP requests.

        Args:
            content: File content as string.
            file_path: Path to file to determine file type (YAML, JSON, Bru, or unsupported).

        Returns:
            List of detected agent definitions with detection type, confidence, and format.
        """
        path_lower = file_path.lower()

        if path_lower.endswith((".yaml", ".yml")):
            if not self._yaml_available:
                logger.debug("Skipping YAML agent detection because PyYAML is not installed")
                return []
            detections = self.structured_detector.detect_in_yaml(content)
        elif path_lower.endswith(".json"):
            detections = self.structured_detector.detect_in_json(content)
        elif path_lower.endswith(".bru"):
            detections = self.structured_detector.detect_in_bru(content)
        else:
            return []

        locations = []
        for detection in detections:
            locations.append(
                {
                    "line": 1,
                    "name": detection["name"],
                    "detection_type": f"structured_{detection['detection_type']}",
                    "confidence": detection["confidence"],
                    "format": detection["format"],
                }
            )

        return locations
