"""Tests for prompt phrase pattern matching in agent definitions."""

import json

from src.detectors.agents.structured_agents import StructuredAgentDetector
from src.detectors.patterns import PatternMatcher


class TestPromptPhrasePatterns:
    """Test suite for prompt phrase detection."""

    @staticmethod
    def test_pattern_matcher_loads_prompt_phrases() -> None:
        """Test that PatternMatcher loads prompt phrase patterns from config."""
        matcher = PatternMatcher.from_file()
        assert matcher._phrase_patterns is not None
        assert len(matcher._phrase_patterns) > 0
        assert any("you are" in p for p in matcher._phrase_patterns)

    @staticmethod
    def test_pattern_matcher_matches_prompt_phrases() -> None:
        """Test that PatternMatcher.match_prompt_phrases() detects agentic phrases."""
        matcher = PatternMatcher.from_file()

        assert matcher.match_prompt_phrases("You are an expert in Python")
        assert matcher.match_prompt_phrases("You are a helpful assistant")
        assert matcher.match_prompt_phrases("Your role is to analyse code")
        assert matcher.match_prompt_phrases("Act as a senior engineer")

        assert not matcher.match_prompt_phrases("This is a regular Python file")
        assert not matcher.match_prompt_phrases("import sys")

    @staticmethod
    def test_pattern_matcher_scores_prompt_phrases() -> None:
        """Test that PatternMatcher.score_prompt_phrases() assigns correct weights."""
        matcher = PatternMatcher.from_file()

        score_with_phrase = matcher.score_prompt_phrases("You are an expert in Jenkins")
        assert score_with_phrase > 0

        assert score_with_phrase >= matcher._phrase_scoring_weight

        assert matcher.score_prompt_phrases("import requests") == 0

    @staticmethod
    def test_bru_agent_example_detection() -> None:
        """Test detection of agent definition from Bru HTTP request body.

        This is the main use case - detecting the Jenkins migration agent
        from the user's Bru HTTP test file.
        """
        detector = StructuredAgentDetector()

        agent_json = {
            "name": "Jenkins_to_github_actions_migration_agent_v1_2_1",
            "description": "An agent that specialises in converting Jenkinsfiles into GitHub Actions workflow",
            "system_prompt": (
                "You are an expert Jenkins to GitHub Actions migration engineer. "
                "Your role is to analyse Jenkins pipelines and generate equivalent "
                "GitHub Actions workflows along with warning and error generated during "
                "migration process"
            ),
            "llm": "gpt-4",
            "input_schema": {
                "type": "object",
                "title": "input_schema",
                "properties": {"Jenkinsfile": {"type": "string"}},
            },
            "output_schema": {"type": "object", "title": "output_schema"},
        }

        assert detector._is_agent_definition(agent_json)

    @staticmethod
    def test_structured_agent_detector_prompt_phrases() -> None:
        """Test StructuredAgentDetector recognises prompt phrases."""
        detector = StructuredAgentDetector()

        assert detector._contains_prompt_phrases("You are an expert Python developer")
        assert detector._contains_prompt_phrases("Your task is to summarize text")
        assert detector._contains_prompt_phrases("Act as a code reviewer")

        assert not detector._contains_prompt_phrases("This is regular Python code")
        assert not detector._contains_prompt_phrases("import sys")

    @staticmethod
    def test_system_prompt_with_phrase_triggers_detection() -> None:
        """Test that system_prompt with agentic phrases triggers agent detection."""
        detector = StructuredAgentDetector()

        agent_config = {
            "name": "DataAnalyzer",
            "system_prompt": "You are a data analysis expert. Your role is to provide insights.",
            "model": "gpt-4",
        }

        assert detector._is_agent_definition(agent_config)

    @staticmethod
    def test_instruction_field_with_phrase_triggers_detection() -> None:
        """Test that instruction field with prompt phrases triggers detection."""
        detector = StructuredAgentDetector()

        agent_config = {
            "name": "CodeReviewer",
            "instructions": "You are a senior code reviewer. Act as a strict quality gate.",
            "llm": "anthropic",
        }

        assert detector._is_agent_definition(agent_config)

    @staticmethod
    def test_backstory_field_with_phrase_triggers_detection() -> None:
        """Test that backstory field with prompt phrases triggers detection (CrewAI style)."""
        detector = StructuredAgentDetector()

        agent_config = {
            "role": "Research Analyst",
            "goal": "Find market insights",
            "backstory": "You are an expert market researcher with 20 years of experience.",
        }

        assert detector._is_agent_definition(agent_config)

    @staticmethod
    def test_description_field_with_phrase_triggers_detection() -> None:
        """Test that description field with prompt phrases triggers detection."""
        detector = StructuredAgentDetector()

        agent_config = {
            "name": "TranslationAgent",
            "description": "You are a professional translator specializing in technical documents.",
            "llm": "azure-openai",
        }

        assert detector._is_agent_definition(agent_config)

    @staticmethod
    def test_json_agent_detection_with_prompt_phrases() -> None:
        """Test detecting agents from JSON content using prompt phrases."""
        detector = StructuredAgentDetector()

        json_content = json.dumps(
            {
                "agents": [
                    {
                        "name": "JenkinsExpert",
                        "system_prompt": "You are an expert in Jenkins administration and pipeline design.",
                        "llm": "openai",
                    }
                ]
            }
        )

        detections = detector.detect_in_json(json_content)
        assert len(detections) > 0

    @staticmethod
    def test_multiple_agents_in_json() -> None:
        """Test detecting multiple agents in a single JSON document."""
        detector = StructuredAgentDetector()

        json_content = json.dumps(
            {
                "agents": [
                    {
                        "name": "Agent1",
                        "system_prompt": "You are a helpful assistant.",
                        "role": "Assistant",
                    },
                    {
                        "name": "Agent2",
                        "system_prompt": "You are an expert engineer. Your role is to review code.",
                        "instructions": "Be thorough and critical.",
                    },
                ]
            }
        )

        detections = detector.detect_in_json(json_content)
        assert len(detections) >= 2

    @staticmethod
    def test_nested_agent_definitions_detected() -> None:
        """Test detection of nested agent definitions."""
        detector = StructuredAgentDetector()

        config = {
            "workflows": {
                "migration": {
                    "agents": {
                        "main_agent": {
                            "name": "MigrationAgent",
                            "system_prompt": "You are an expert in Jenkins to GitHub Actions migration.",
                            "model": "gpt-4",
                        }
                    }
                }
            }
        }

        detections = detector._analyse_structure(config, "json")
        assert any(d.get("name") == "MigrationAgent" for d in detections)

    @staticmethod
    def test_prompt_phrase_not_required_when_other_indicators_present() -> None:
        """Test that prompt phrases are optional - other indicators still work."""
        detector = StructuredAgentDetector()

        agent_without_prompt = {"name": "SimpleAgent", "llm": "openai"}
        assert detector._is_agent_definition(agent_without_prompt)

        agent_crew_style = {"role": "Analyst", "goal": "Analyze data"}
        assert detector._is_agent_definition(agent_crew_style)

    @staticmethod
    def test_prompt_phrase_boosts_confidence() -> None:
        """Test that prompt phrases are a strong indicator alongside other signals."""
        detector = StructuredAgentDetector()

        minimal_with_phrase = {
            "name": "Agent",
            "system_prompt": "You are a helpful assistant that specializes in data analysis.",
        }

        assert detector._is_agent_definition(minimal_with_phrase)

    @staticmethod
    def test_empty_prompt_phrase_does_not_trigger_prompt_detection() -> None:
        """Test that empty system_prompt doesn't trigger prompt phrase detection.

        However, empty system_prompt + name still matches the structural pattern.
        """
        detector = StructuredAgentDetector()

        agent_empty_prompt = {
            "name": "Agent",
            "system_prompt": "",
        }

        assert not detector._contains_prompt_phrases("")

        assert detector._is_agent_definition(agent_empty_prompt)

    @staticmethod
    def test_non_string_prompt_field_ignored() -> None:
        """Test that non-string system_prompt values are handled gracefully."""
        detector = StructuredAgentDetector()

        agent_dict_prompt = {
            "name": "Agent",
            "system_prompt": {"template": "some template"},
            "llm": "openai",
        }

        assert detector._is_agent_definition(agent_dict_prompt)

    @staticmethod
    def test_case_insensitive_phrase_matching() -> None:
        """Test that prompt phrase matching is case-insensitive."""
        detector = StructuredAgentDetector()

        assert detector._contains_prompt_phrases("YOU ARE A HELPFUL ASSISTANT")
        assert detector._contains_prompt_phrases("You Are A Helpful Assistant")
        assert detector._contains_prompt_phrases("you are a helpful assistant")
        assert detector._contains_prompt_phrases("YoU aRe A hElPfUl AssiStAnT")

    @staticmethod
    def test_phrase_with_special_characters() -> None:
        """Test prompt phrase detection with special characters and formatting."""
        detector = StructuredAgentDetector()

        assert detector._contains_prompt_phrases('"You are an expert in Python programming and software design."')
        assert detector._contains_prompt_phrases("System: You are a helpful assistant.\nInstructions: Be concise.")

    @staticmethod
    def test_pattern_matcher_content_score_includes_prompts() -> None:
        """Test that content scoring includes prompt phrase scoring."""
        matcher = PatternMatcher.from_file()

        text_with_prompts = """
        {
            "name": "Agent",
            "system_prompt": "You are an expert Jenkins to GitHub Actions migration engineer"
        }
        """

        score = matcher.score_content(text_with_prompts)
        assert score >= 1
