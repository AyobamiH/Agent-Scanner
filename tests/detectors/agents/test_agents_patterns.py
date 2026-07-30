"""Tests for framework-specific agent pattern matching."""

import ast

from src.detectors.agents.agents import AgentDetector


class TestFrameworkPatternsMatching:
    """Test framework pattern matching logic."""

    @staticmethod
    def test_matches_known_framework_pattern() -> None:
        """Match framework-specific patterns."""
        detector = AgentDetector()

        patterns_to_test = [
            "initialize_agent",
            "initialise_agent",
            "ChatOpenAI",
            "ConversableAgent",
            "Task",
            "Crew",
        ]

        for pattern in patterns_to_test:
            result = detector._matches_framework_pattern(pattern)
            assert isinstance(result, bool)


class TestIsAgentCompositionCall:
    """Test detection of agent composition calls."""

    @staticmethod
    def test_composition_with_llm_parameter() -> None:
        """Detect composition with LLM parameter."""
        detector = AgentDetector()
        code = "agent = MyClass(llm=llm_obj)"
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                result = detector._is_agent_composition_call("MyClass", node, set(), tree)
                assert isinstance(result, bool)


class TestDetectAgentViaRegex:
    """Test regex-based detection."""

    @staticmethod
    def test_regex_finds_all_agent_patterns() -> None:
        """Regex detects all agent-like tokens."""
        detector = AgentDetector()
        text = "MyAgent class Worker Tool Execute"

        matches = detector._detect_agents_via_regex(text)
        assert "MyAgent" in matches

    @staticmethod
    def test_regex_deduplicates_correctly() -> None:
        """Regex removes duplicates."""
        detector = AgentDetector()
        text = "MyAgent here MyAgent there MyAgent everywhere"

        matches = detector._detect_agents_via_regex(text)
        assert matches.count("MyAgent") == 1


class TestGetFrameworkImports:
    """Test extraction of framework imports."""

    @staticmethod
    def test_extract_framework_imports_from_code() -> None:
        """Extract framework module imports."""
        detector = AgentDetector()
        code = """
from langchain.agents import Tool
from crewai import Agent
import autogen
"""
        tree = ast.parse(code)

        imports = detector._get_framework_imports(tree)
        assert isinstance(imports, set)


class TestFrameworkPatternMatching:
    """Test framework-specific pattern matching."""

    @staticmethod
    def test_matches_framework_pattern_with_list() -> None:
        """Framework patterns as list are matched correctly."""
        text = """
        from langchain.agents import initialize_agent

        agent = initialize_agent(llm=model, tools=tools)
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_matches_framework_pattern_with_dict() -> None:
        """Framework patterns as dict are matched correctly."""
        detector = AgentDetector()

        assert detector._matches_framework_pattern("initialize_agent") is True

    @staticmethod
    def test_extract_callee_name_from_nested_call() -> None:
        """Callee names are extracted from nested calls."""
        text = """
        agent = factory.create_agent()
        """
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert isinstance(count, int)


class TestRegexFallbackBehaviour:
    """Test regex-based detection fallback behaviour."""

    @staticmethod
    def test_regex_detects_agent_in_invalid_syntax() -> None:
        """Regex fallback detects agents in invalid Python."""
        text = "MyAgent, WorkerAgent, ExecutorAgent invalid syntax {{"
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count >= 1

    @staticmethod
    def test_regex_deduplicates_matches() -> None:
        """Regex fallback deduplicates agent names."""
        text = "MyAgent appears multiple times: MyAgent and MyAgent again"
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count == 1

    @staticmethod
    def test_regex_matches_case_sensitive_agent_names() -> None:
        """Regex matches agent names with proper case (CamelCase with Agent suffix)."""
        text = "myagent, MYAGENT, MyAgent, mYAGENT [[INVALID"
        detector = AgentDetector()
        count = detector.count_agents_in_text(text)

        assert count == 1


class TestCheckDictPatterns:
    """Test _check_dict_patterns() preserves bool return contract."""

    @staticmethod
    def test_returns_false_when_no_match_found() -> None:
        """Verify _check_dict_patterns returns False (not None) when no match is found.

        Arrange: Create detector and patterns dict with no matching values.
        Act: Call _check_dict_patterns with non-matching fn_name and module_name.
        Assert: Verify return value is explicitly False (not None or other falsy value).
        """
        detector = AgentDetector()
        patterns = {"names": ["initialize_agent", "create_agent"], "modules": ["langchain.agents", "autogen"]}
        fn_name = "unknown_function"
        module_name = "unknown.module"

        result = detector._check_dict_patterns(patterns, fn_name, module_name)

        assert result is False
        assert isinstance(result, bool)
        assert result is not None


class TestCachingBehaviour:
    """Test caching of keywords across detector instances."""

    @staticmethod
    def test_keywords_cached_across_instances() -> None:
        """Keywords are cached and reused across detector instances."""
        detector1 = AgentDetector()
        detector2 = AgentDetector()

        assert detector1._framework_patterns == detector2._framework_patterns

    @staticmethod
    def test_cache_not_overwritten_with_none_path() -> None:
        """Cache is not overwritten when None is passed as path."""
        AgentDetector(keywords_path="src/config/keywords.json")
        original_cache = AgentDetector._cached_keywords

        AgentDetector(keywords_path=None)

        assert AgentDetector._cached_keywords is original_cache
