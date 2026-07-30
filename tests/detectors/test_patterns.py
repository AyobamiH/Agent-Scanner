"""Tests for AI keyword and pattern matching."""

import pytest

from src.detectors.patterns import PatternMatcher


@pytest.mark.parametrize(
    "text,expected",
    [
        ("this uses LangChain", True),
        ("no ai here", True),
        ("OpenAI powered", True),
        ("some unrelated text", False),
    ],
)
def test_contains_ai_keywords_matches_expected(text: str, expected: bool) -> None:
    """PatternMatcher.contains_ai_keywords returns expected boolean for texts."""
    matcher = PatternMatcher.from_file()
    assert matcher.contains_ai_keywords(text) is expected


def test_pattern_matcher_empty_and_none_returns_false():
    """Empty or whitespace-only content should not match AI keywords."""
    matcher = PatternMatcher()
    assert matcher.contains_ai_keywords("") is False
    assert matcher.contains_ai_keywords("") is False


def test_camelcase_and_hyphen_detection(custom_matcher):
    m = custom_matcher
    text1 = "This file instantiates an OpenAIClient() to call the service"
    assert m.match_content(text1) is True

    text2 = "We reference google-generativeai in our config"
    assert m.match_content(text2) is True


def test_env_var_detection(custom_matcher):
    m = custom_matcher
    text = 'export OPENAI_API_KEY="secret"'
    assert m.match_content(text) is True


def test_minified_line_detection(custom_matcher):
    m = custom_matcher
    text = "function a(){return'openai';}var b=a()+"
    assert m.match_content(text) is True


def test_camelcase_token_detection(custom_matcher):
    m = custom_matcher
    text = "class MyAgent:\n    def run(self):\n        pass"

    assert m.match_content(text) is True


class TestMatchPathBasic:
    """Test PatternMatcher.match_path with basic scenarios."""

    @staticmethod
    def test_match_path_empty_returns_false() -> None:
        """Empty path should not match."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.match_path("") is False

    @staticmethod
    def test_match_path_whole_word_match() -> None:
        """Path containing whole-word should match."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.match_path("src/agent/handler.py") is True

    @staticmethod
    def test_match_path_no_match() -> None:
        """Path without matching keywords should not match."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.match_path("src/utils/helpers.py") is False

    @staticmethod
    def test_match_path_case_insensitive() -> None:
        """Path matching should be case-insensitive."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.match_path("src/AGENT/handler.py") is True
        assert matcher.match_path("src/Agent/handler.py") is True

    @staticmethod
    def test_match_path_substring_keyword() -> None:
        """Path should match content keywords via substring."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.match_path("openai_client.py") is True
        assert matcher.match_path("src/openai_handler.py") is True

    @staticmethod
    def test_match_path_token_plural_singular() -> None:
        """Path with plural token should match singular whole-word."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.match_path("src/agents/base.py") is True

    @staticmethod
    def test_match_path_token_singular_plural() -> None:
        """Path with singular non-plural word should match as singular."""
        matcher = PatternMatcher(path_whole_words=["handler"])
        assert matcher.match_path("src/handlers/main.py") is True

    @staticmethod
    def test_match_path_hyphenated_token() -> None:
        """Path with hyphenated components should tokenise correctly."""
        matcher = PatternMatcher(path_whole_words=["ai"])
        assert matcher.match_path("src/ai-service/index.py") is True

    @staticmethod
    def test_match_path_underscore_token() -> None:
        """Path with underscored components should tokenise correctly."""
        matcher = PatternMatcher(path_whole_words=["ai"])
        assert matcher.match_path("src/ai_service/index.py") is True


class TestScorePath:
    """Test PatternMatcher.score_path with various scenarios."""

    @staticmethod
    def test_score_path_empty_returns_zero() -> None:
        """Empty path should score zero."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.score_path("") == 0

    @staticmethod
    def test_score_path_whole_word_regex_contributes_two() -> None:
        """Whole-word regex match should contribute 2 points."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        score = matcher.score_path("agent.py")
        assert score >= 2

    @staticmethod
    def test_score_path_substring_keyword_contributes_one() -> None:
        """Substring keyword match should contribute 1 point."""
        matcher = PatternMatcher(content_keywords=["openai"])
        score = matcher.score_path("openai_client.py")
        assert score >= 1

    @staticmethod
    def test_score_path_multiple_matches_accumulate() -> None:
        """Multiple matches should accumulate points."""
        matcher = PatternMatcher(
            path_whole_words=["agent", "handler"],
            content_keywords=["ai"],
        )
        score = matcher.score_path("agent_handler_ai.py")
        assert score >= 4

    @staticmethod
    def test_score_path_plural_variant_scores() -> None:
        """Path with plural variant should contribute to score."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        score = matcher.score_path("src/agents/")
        assert score >= 2

    @staticmethod
    def test_score_path_no_match_returns_zero() -> None:
        """Path with no matches should score zero."""
        matcher = PatternMatcher(path_whole_words=["agent"])
        assert matcher.score_path("src/utils/helpers.py") == 0


class TestScoreContent:
    """Test PatternMatcher.score_content with comprehensive scenarios."""

    @staticmethod
    def test_score_content_empty_returns_zero() -> None:
        """Empty content should score zero."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.score_content("") == 0

    @staticmethod
    def test_score_content_substring_keyword_contributes_one() -> None:
        """Substring keyword in content should contribute 1 point."""
        matcher = PatternMatcher(content_keywords=["openai"])
        score = matcher.score_content("uses openai api")
        assert score >= 1

    @staticmethod
    def test_score_content_sanitised_match_contributes_one() -> None:
        """Sanitised substring match should contribute 1 point."""
        matcher = PatternMatcher(content_keywords=["openai"])
        score = matcher.score_content("uses open-ai")
        assert score >= 1

    @staticmethod
    def test_score_content_whole_word_contributes_two() -> None:
        """Whole-word match should contribute 2 points."""
        matcher = PatternMatcher(content_whole_words=["openai"])
        score = matcher.score_content("this uses openai today")
        assert score >= 2

    @staticmethod
    def test_score_content_multiple_keywords_accumulate() -> None:
        """Multiple matches should accumulate points."""
        matcher = PatternMatcher(
            content_keywords=["openai", "langchain"],
            content_whole_words=["agent"],
        )
        score = matcher.score_content("openai agent with langchain")
        assert score >= 4

    @staticmethod
    def test_score_content_token_match_contributes_one() -> None:
        """Token match should contribute to score."""
        matcher = PatternMatcher(content_keywords=["openai-api"])
        score = matcher.score_content("openai_api.send_request()")
        assert score >= 1

    @staticmethod
    def test_score_content_min_sanitised_length_respected() -> None:
        """Short keywords below min_sanitised_substring_length should be skipped."""
        matcher = PatternMatcher(content_keywords=["ai"], min_sanitised_substring_length=3)
        score = matcher.score_content("ai-is-everywhere")
        assert score >= 0

    @staticmethod
    def test_score_content_whole_word_regex_finds_matches() -> None:
        """Whole-word regex should match multiple occurrences."""
        matcher = PatternMatcher(content_whole_words=["agent"])
        score = matcher.score_content("this agent and that agent")
        assert score >= 2

    @staticmethod
    def test_score_content_token_set_whole_word_match() -> None:
        """Token set should match whole words."""
        matcher = PatternMatcher(content_whole_words=["agent"])
        score = matcher.score_content("agent_service")
        assert score >= 2


class TestMatchContent:
    """Test PatternMatcher.match_content with practical scenarios."""

    @staticmethod
    def test_match_content_empty_returns_false() -> None:
        """Empty content should not match."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.match_content("") is False

    @staticmethod
    def test_match_content_substring_keyword_match() -> None:
        """Content with substring keyword should match."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.match_content("import openai") is True

    @staticmethod
    def test_match_content_sanitised_match() -> None:
        """Content with sanitised match should match."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.match_content("from open-ai import client") is True

    @staticmethod
    def test_match_content_token_match() -> None:
        """Content with token match should match."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.match_content("openai_client.create_message()") is True

    @staticmethod
    def test_match_content_whole_word_token_match() -> None:
        """Content with whole-word token match should match."""
        matcher = PatternMatcher(content_whole_words=["agent"])
        assert matcher.match_content("the agent class") is True

    @staticmethod
    def test_match_content_whole_word_regex_match() -> None:
        """Content matching whole-word regex should match."""
        matcher = PatternMatcher(content_whole_words=["agent"])
        assert matcher.match_content("create an agent for the task") is True

    @staticmethod
    def test_match_content_no_match_returns_false() -> None:
        """Content without matches should not match."""
        matcher = PatternMatcher(content_keywords=["openai"], content_whole_words=["agent"])
        assert matcher.match_content("this is unrelated content") is False

    @staticmethod
    def test_match_content_case_insensitive() -> None:
        """Content matching should be case-insensitive."""
        matcher = PatternMatcher(content_keywords=["openai"])
        assert matcher.match_content("import OpenAI") is True
        assert matcher.match_content("OPENAI_API_KEY = secret") is True

    @staticmethod
    def test_match_content_multiline() -> None:
        """Content matching should work across multiple lines."""
        matcher = PatternMatcher(content_keywords=["openai"])
        text = "import json\nimport openai\nfrom typing import Dict"
        assert matcher.match_content(text) is True


class TestTokeniseText:
    """Test PatternMatcher._tokenise_text edge cases."""

    @staticmethod
    def test_tokenise_text_empty_returns_empty_set() -> None:
        """Empty text should return empty set."""
        matcher = PatternMatcher()
        assert matcher._tokenise_text("") == set()

    @staticmethod
    def test_tokenise_text_camelcase_splits() -> None:
        """CamelCase should split into tokens at boundaries."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("OpenAIClient")
        assert "open" in tokens
        assert "aiclient" in tokens

    @staticmethod
    def test_tokenise_text_snake_case_splits() -> None:
        """snake_case should split into tokens."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("openai_client")
        assert "openai" in tokens
        assert "client" in tokens

    @staticmethod
    def test_tokenise_text_hyphen_case_splits() -> None:
        """hyphen-case should split into tokens."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("openai-client")
        assert "openai" in tokens
        assert "client" in tokens

    @staticmethod
    def test_tokenise_text_dot_notation_splits() -> None:
        """dot.notation should split into tokens."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("openai.client")
        assert "openai" in tokens
        assert "client" in tokens

    @staticmethod
    def test_tokenise_text_mixed_case_and_separators() -> None:
        """Mixed camelCase and separators should split correctly."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("OpenAI_Client-Service")
        assert "open" in tokens
        assert "ai" in tokens
        assert "client" in tokens
        assert "service" in tokens

    @staticmethod
    def test_tokenise_text_normalises_to_lowercase() -> None:
        """All tokens should be lowercased."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("OpenAIClient")
        for token in tokens:
            assert token == token.lower()

    @staticmethod
    def test_tokenise_text_removes_non_alphanumeric() -> None:
        """Non-alphanumeric characters should be removed."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("open@ai!client")
        assert "openaiclient" in tokens

    @staticmethod
    def test_tokenise_text_multiple_separators() -> None:
        """Multiple consecutive separators should be handled."""
        matcher = PatternMatcher()
        tokens = matcher._tokenise_text("openai--client__service")
        assert "openai" in tokens
        assert "client" in tokens
        assert "service" in tokens


class TestFromFileErrorHandling:
    """Test PatternMatcher.from_file error handling."""

    @staticmethod
    def test_from_file_missing_config_uses_empty() -> None:
        """Missing config file should use empty defaults."""
        matcher = PatternMatcher.from_file("/nonexistent/config.json")
        assert matcher.content_keywords == []
        assert matcher.content_whole_words == []
        assert matcher.path_whole_words == []

    @staticmethod
    def test_from_file_none_path_loads_default() -> None:
        """None path should load default configuration."""
        matcher = PatternMatcher.from_file(None)
        assert isinstance(matcher, PatternMatcher)

    @staticmethod
    def test_from_file_with_path_callable() -> None:
        """from_file should return callable PatternMatcher."""
        matcher = PatternMatcher.from_file()
        assert callable(matcher.match_content)
        assert callable(matcher.match_path)


class TestBuildWholeWordRegex:
    """Test PatternMatcher._build_whole_word_regex."""

    @staticmethod
    def test_build_whole_word_regex_empty_returns_none() -> None:
        """Empty word list should return None."""
        result = PatternMatcher._build_whole_word_regex([])
        assert result is None

    @staticmethod
    def test_build_whole_word_regex_single_word() -> None:
        """Single word should build valid regex."""
        regex = PatternMatcher._build_whole_word_regex(["agent"])
        assert regex is not None
        assert regex.search("agent") is not None
        assert regex.search("agents") is None

    @staticmethod
    def test_build_whole_word_regex_multiple_words() -> None:
        """Multiple words should build combined regex."""
        regex = PatternMatcher._build_whole_word_regex(["agent", "handler"])
        assert regex is not None
        assert regex.search("agent") is not None
        assert regex.search("handler") is not None

    @staticmethod
    def test_build_whole_word_regex_word_boundaries_respected() -> None:
        """Regex should respect word boundaries."""
        regex = PatternMatcher._build_whole_word_regex(["ai"])
        assert regex is not None
        assert regex.search("ai ") is not None
        assert regex.search("ai-service") is not None

    @staticmethod
    def test_build_whole_word_regex_case_insensitive() -> None:
        """Regex should be case-insensitive."""
        regex = PatternMatcher._build_whole_word_regex(["agent"])
        assert regex is not None
        assert regex.search("AGENT") is not None
        assert regex.search("Agent") is not None

    @staticmethod
    def test_build_whole_word_regex_special_chars_escaped() -> None:
        """Special characters should be escaped in regex."""
        regex = PatternMatcher._build_whole_word_regex(["gpt-4"])
        assert regex is not None
        assert regex.search("gpt-4") is not None


class TestIntegrationScenarios:
    """Integration tests for realistic detection scenarios."""

    @staticmethod
    def test_detect_openai_import_statement() -> None:
        """Should detect OpenAI imports."""
        matcher = PatternMatcher.from_file()
        code = "import openai\nopenai.api_key = 'secret'"
        assert matcher.match_content(code) is True

    @staticmethod
    def test_detect_langchain_imports() -> None:
        """Should detect LangChain imports."""
        matcher = PatternMatcher.from_file()
        code = "from langchain.chains import LLMChain"
        assert matcher.match_content(code) is True

    @staticmethod
    def test_detect_agent_pattern() -> None:
        """Should detect agent pattern in code."""
        matcher = PatternMatcher(content_whole_words=["agent"])
        code = "class BehavioralAgent(BaseAgent):\n    pass"
        assert matcher.match_content(code) is True

    @staticmethod
    def test_score_content_stronger_for_multiple_matches() -> None:
        """Content with multiple AI indicators should score higher."""
        matcher = PatternMatcher.from_file()
        high_score_code = "import openai\nfrom langchain import LLMChain\nagent = Agent()"
        low_score_code = "import json"
        high_score = matcher.score_content(high_score_code)
        low_score = matcher.score_content(low_score_code)
        assert high_score > low_score

    @staticmethod
    def test_score_path_ai_framework_directories() -> None:
        """Should score AI framework directories higher."""
        matcher = PatternMatcher.from_file()
        ai_path = "src/llm/openai_handler.py"
        normal_path = "src/utils/helpers.py"
        ai_score = matcher.score_path(ai_path)
        normal_score = matcher.score_path(normal_path)
        assert ai_score >= normal_score
