"""Conservative JavaScript/TypeScript agent factory detection.

Detects agent construction only when a known factory is imported from a
trusted LangChain/LangGraph module and then invoked. This avoids counting
mere imports, comments, string examples, or unrelated local functions named
like agent factories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class _NamedImport:
    module: str
    canonical_name: str
    local_name: str


class JavaScriptAgentDetector:
    """Detect explicit agent factory invocations in JavaScript and TypeScript."""

    SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".js", ".jsx", ".ts", ".tsx")

    _TRUSTED_FACTORIES: Final[dict[str, frozenset[str]]] = {
        "langchain": frozenset({"createAgent"}),
        "@langchain/langgraph/prebuilt": frozenset({"createReactAgent"}),
        "@langchain/langgraph/prebuilts": frozenset({"createReactAgent"}),
    }

    _ES_NAMED_IMPORT_RE = re.compile(
        r'import\s*\{(?P<names>[^}]+)\}\s*from\s*["\'](?P<module>[^"\']+)["\']',
        re.MULTILINE | re.DOTALL,
    )
    _ES_NAMESPACE_IMPORT_RE = re.compile(
        r'import\s+\*\s+as\s+(?P<alias>[A-Za-z_$][\w$]*)\s+from\s*["\'](?P<module>[^"\']+)["\']',
        re.MULTILINE,
    )
    _CJS_NAMED_IMPORT_RE = re.compile(
        r'(?:const|let|var)\s*\{(?P<names>[^}]+)\}\s*=\s*require\(\s*["\'](?P<module>[^"\']+)["\']\s*\)',
        re.MULTILINE | re.DOTALL,
    )
    _CJS_NAMESPACE_IMPORT_RE = re.compile(
        r'(?:const|let|var)\s+(?P<alias>[A-Za-z_$][\w$]*)\s*=\s*require\(\s*["\'](?P<module>[^"\']+)["\']\s*\)',
        re.MULTILINE,
    )

    def get_agent_locations(self, text: str) -> list[dict[str, object]]:
        """Return trusted agent factory call locations."""
        if not text:
            return []

        code = self._strip_comments_preserving_newlines(text)
        named_imports = self._extract_named_imports(code)
        namespace_imports = self._extract_namespace_imports(code)
        call_code = self._mask_string_literals_preserving_newlines(code)

        locations: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()

        for imported in named_imports:
            for match in re.finditer(rf"\b{re.escape(imported.local_name)}\s*\(", call_code):
                line = self._line_number(call_code, match.start())
                key = (imported.canonical_name, line)
                if key in seen:
                    continue
                seen.add(key)
                locations.append(
                    {
                        "line": line,
                        "name": imported.canonical_name,
                        "detection_type": "javascript_framework_factory",
                    }
                )

        for module, alias in namespace_imports:
            for canonical_name in self._TRUSTED_FACTORIES.get(module, frozenset()):
                pattern = rf"\b{re.escape(alias)}\s*\.\s*{re.escape(canonical_name)}\s*\("
                for match in re.finditer(pattern, call_code):
                    line = self._line_number(call_code, match.start())
                    key = (canonical_name, line)
                    if key in seen:
                        continue
                    seen.add(key)
                    locations.append(
                        {
                            "line": line,
                            "name": canonical_name,
                            "detection_type": "javascript_framework_factory",
                        }
                    )

        return sorted(
            locations,
            key=lambda item: (
                item["line"] if isinstance(item["line"], int) else 0,
                str(item["name"]),
            ),
        )

    def get_framework_imports(self, text: str) -> set[str]:
        """Return LangChain/LangGraph module imports found in JS/TS source."""
        if not text:
            return set()

        code = self._strip_comments_preserving_newlines(text)
        modules = {
            match.group("module")
            for regex in (
                self._ES_NAMED_IMPORT_RE,
                self._ES_NAMESPACE_IMPORT_RE,
                self._CJS_NAMED_IMPORT_RE,
                self._CJS_NAMESPACE_IMPORT_RE,
            )
            for match in regex.finditer(code)
        }
        return {module for module in modules if module == "langchain" or module.startswith("@langchain/")}

    def _extract_named_imports(self, text: str) -> list[_NamedImport]:
        imports: list[_NamedImport] = []
        for regex in (self._ES_NAMED_IMPORT_RE, self._CJS_NAMED_IMPORT_RE):
            for match in regex.finditer(text):
                module = match.group("module")
                trusted = self._TRUSTED_FACTORIES.get(module)
                if not trusted:
                    continue
                for raw_name in match.group("names").split(","):
                    parsed = self._parse_named_binding(raw_name)
                    if parsed is None:
                        continue
                    canonical_name, local_name = parsed
                    if canonical_name in trusted:
                        imports.append(
                            _NamedImport(
                                module=module,
                                canonical_name=canonical_name,
                                local_name=local_name,
                            )
                        )
        return imports

    def _extract_namespace_imports(self, text: str) -> list[tuple[str, str]]:
        imports: list[tuple[str, str]] = []
        for regex in (self._ES_NAMESPACE_IMPORT_RE, self._CJS_NAMESPACE_IMPORT_RE):
            for match in regex.finditer(text):
                module = match.group("module")
                if module in self._TRUSTED_FACTORIES:
                    imports.append((module, match.group("alias")))
        return imports

    @staticmethod
    def _parse_named_binding(raw_name: str) -> tuple[str, str] | None:
        binding = raw_name.strip()
        if not binding:
            return None

        as_parts = re.split(r"\s+as\s+", binding, maxsplit=1)
        if len(as_parts) == 2:
            canonical_name, local_name = (part.strip() for part in as_parts)
        elif ":" in binding:
            canonical_name, local_name = (part.strip() for part in binding.split(":", 1))
        else:
            canonical_name = binding
            local_name = binding

        identifier_re = r"^[A-Za-z_$][\w$]*$"
        if not re.match(identifier_re, canonical_name) or not re.match(identifier_re, local_name):
            return None
        return canonical_name, local_name

    @staticmethod
    def _line_number(text: str, offset: int) -> int:
        return text.count("\n", 0, offset) + 1

    @staticmethod
    def _mask_string_literals_preserving_newlines(text: str) -> str:
        """Mask quoted literals so examples inside strings are not treated as calls.

        Template literals are intentionally masked wholesale. That trades a rare
        false negative for `${...}` expressions for a lower false-positive rate,
        which is the preferred bias for governance-oriented scanning.
        """
        out: list[str] = []
        i = 0
        quote = ""

        while i < len(text):
            char = text[i]

            if not quote:
                if char in {'"', "'", "`"}:
                    quote = char
                    out.append(" ")
                else:
                    out.append(char)
                i += 1
                continue

            if char == "\\" and i + 1 < len(text):
                out.append(" ")
                escaped = text[i + 1]
                out.append("\n" if escaped == "\n" else " ")
                i += 2
                continue

            if char == quote:
                quote = ""
                out.append(" ")
            else:
                out.append("\n" if char == "\n" else " ")
            i += 1

        return "".join(out)

    @staticmethod
    def _strip_comments_preserving_newlines(text: str) -> str:
        """Remove JS comments while preserving line numbers and quoted strings."""
        out: list[str] = []
        i = 0
        state = "code"
        quote = ""

        while i < len(text):
            char = text[i]
            next_char = text[i + 1] if i + 1 < len(text) else ""

            if state == "code":
                if char in {'"', "'", "`"}:
                    quote = char
                    state = "string"
                    out.append(char)
                    i += 1
                    continue
                if char == "/" and next_char == "/":
                    out.extend((" ", " "))
                    i += 2
                    state = "line_comment"
                    continue
                if char == "/" and next_char == "*":
                    out.extend((" ", " "))
                    i += 2
                    state = "block_comment"
                    continue
                out.append(char)
                i += 1
                continue

            if state == "string":
                out.append(char)
                if char == "\\" and i + 1 < len(text):
                    out.append(text[i + 1])
                    i += 2
                    continue
                if char == quote:
                    state = "code"
                    quote = ""
                i += 1
                continue

            if state == "line_comment":
                if char == "\n":
                    out.append("\n")
                    state = "code"
                else:
                    out.append(" ")
                i += 1
                continue

            if state == "block_comment":
                if char == "*" and next_char == "/":
                    out.extend((" ", " "))
                    i += 2
                    state = "code"
                    continue
                out.append("\n" if char == "\n" else " ")
                i += 1

        return "".join(out)
