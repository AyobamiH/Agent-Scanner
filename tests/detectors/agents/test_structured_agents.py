"""Tests for structured agent detector (YAML/JSON configurations)."""

import logging
from unittest.mock import patch

import pytest

from src.detectors.agents.structured_agents import StructuredAgentDetector


@pytest.fixture
def detector() -> StructuredAgentDetector:
    """Create a StructuredAgentDetector instance for testing."""
    return StructuredAgentDetector()


def test_detector_initialisation() -> None:
    """Detector initialises successfully."""
    detector = StructuredAgentDetector()

    assert detector is not None


def test_detect_in_yaml_with_empty_content(detector: StructuredAgentDetector) -> None:
    """Detect in YAML returns empty list for empty content."""
    result = detector.detect_in_yaml("")

    assert result == []


def test_detect_in_yaml_with_whitespace_only(detector: StructuredAgentDetector) -> None:
    """Detect in YAML returns empty list for whitespace only content."""
    result = detector.detect_in_yaml("   \n\t  ")

    assert result == []


def test_detect_in_yaml_with_invalid_yaml(detector: StructuredAgentDetector, caplog) -> None:
    """Detect in YAML returns empty list and logs debug for invalid YAML."""
    invalid_yaml = "invalid: yaml: content: [unclosed"

    caplog.set_level(logging.DEBUG)
    result = detector.detect_in_yaml(invalid_yaml)

    assert result == []
    assert "Failed to parse yaml" in caplog.text


def test_detect_in_yaml_with_simple_agent(detector: StructuredAgentDetector) -> None:
    """Detect agent definition with required name and llm keys."""
    yaml_content = """
name: MyAgent
llm: gpt-4
tools:
  - calculator
  - search
"""

    result = detector.detect_in_yaml(yaml_content)

    assert len(result) == 1
    assert result[0]["detection_type"] == "agent_definition"
    assert result[0]["format"] == "yaml"
    assert result[0]["name"] == "MyAgent"
    assert result[0]["confidence"] > 0.0


def test_detect_in_yaml_with_name_and_model(detector: StructuredAgentDetector) -> None:
    """Detect agent definition with name and model keys."""
    yaml_content = """
name: DataProcessor
model: claude-3
instructions: Process data efficiently
"""

    result = detector.detect_in_yaml(yaml_content)

    assert len(result) == 1
    assert result[0]["name"] == "DataProcessor"


def test_detect_in_yaml_with_agent_name_and_model(detector: StructuredAgentDetector) -> None:
    """Detect agent definition with agent_name and model keys."""
    yaml_content = """
agent_name: CoordinatorAgent
model: gpt-4-turbo
backstory: Coordinates multiple agents
"""

    result = detector.detect_in_yaml(yaml_content)

    assert len(result) == 1
    assert result[0]["name"] == "CoordinatorAgent"


def test_detect_in_yaml_with_role_and_goal(detector: StructuredAgentDetector) -> None:
    """Detect agent definition with role and goal keys."""
    yaml_content = """
role: Researcher
goal: Find relevant information
tools:
  - web_search
"""

    result = detector.detect_in_yaml(yaml_content)

    assert len(result) == 1
    assert result[0]["name"] in ["Researcher", "UnnamedAgent"]


def test_detect_in_yaml_with_nested_agents(detector: StructuredAgentDetector) -> None:
    """Detect nested agent definitions within parent objects."""
    yaml_content = """
config:
  primary_agent:
    name: PrimaryAgent
    llm: gpt-4
  secondary_agent:
    name: SecondaryAgent
    model: claude-3
"""

    result = detector.detect_in_yaml(yaml_content)

    assert len(result) > 0


def test_detect_in_yaml_with_agent_list(detector: StructuredAgentDetector) -> None:
    """Detect agents within a list structure."""
    yaml_content = """
agents:
  - name: Agent1
    llm: gpt-3.5
  - name: Agent2
    model: claude-2
"""

    result = detector.detect_in_yaml(yaml_content)

    assert len(result) == 2
    assert all(r["detection_type"] == "agent_in_list" for r in result)


def test_detect_in_yaml_logs_success(detector: StructuredAgentDetector, caplog) -> None:
    """Detect in YAML logs info message when agents are found."""
    yaml_content = """
name: LoggedAgent
llm: gpt-4
"""

    caplog.set_level(logging.INFO)
    detector.detect_in_yaml(yaml_content)

    assert "Detected" in caplog.text
    assert "agent(s)" in caplog.text


def test_detect_in_yaml_without_pyyaml_imports(detector: StructuredAgentDetector, caplog) -> None:
    """Detect in YAML handles PyYAML import errors gracefully."""
    yaml_content = "name: Test\nllm: gpt-4"

    caplog.set_level(logging.DEBUG)

    with patch("builtins.__import__", side_effect=ImportError("No module named yaml")):
        result = detector.detect_in_yaml(yaml_content)

    assert result == []


def test_detect_in_json_with_empty_content(detector: StructuredAgentDetector) -> None:
    """Detect in JSON returns empty list for empty content."""
    result = detector.detect_in_json("")

    assert result == []


def test_detect_in_json_with_whitespace_only(detector: StructuredAgentDetector) -> None:
    """Detect in JSON returns empty list for whitespace only content."""
    result = detector.detect_in_json("   \n  ")

    assert result == []


def test_detect_in_json_with_invalid_json(detector: StructuredAgentDetector, caplog) -> None:
    """Detect in JSON returns empty list and logs debug for invalid JSON."""
    invalid_json = '{"invalid": json content}'

    caplog.set_level(logging.DEBUG)
    result = detector.detect_in_json(invalid_json)

    assert result == []
    assert "Failed to parse json" in caplog.text


def test_detect_in_json_with_simple_agent(detector: StructuredAgentDetector) -> None:
    """Detect agent definition in simple JSON structure."""
    json_content = '{"name": "JSONAgent", "llm": "gpt-4", "tools": ["search"]}'

    result = detector.detect_in_json(json_content)

    assert len(result) == 1
    assert result[0]["detection_type"] == "agent_definition"
    assert result[0]["format"] == "json"
    assert result[0]["name"] == "JSONAgent"


def test_detect_in_json_with_nested_agents(detector: StructuredAgentDetector) -> None:
    """Detect nested agent definitions in JSON."""
    json_content = (
        '{"agents": {"first": {"name": "First", "model": "gpt-4"}, ' '"second": {"name": "Second", "llm": "claude"}}}'
    )

    result = detector.detect_in_json(json_content)

    assert len(result) > 0


def test_detect_in_json_with_agent_array(detector: StructuredAgentDetector) -> None:
    """Detect agents in JSON array."""
    json_content = '{"team": [{"name": "A1", "llm": "gpt-4"}, {"name": "A2", "model": "claude"}]}'

    result = detector.detect_in_json(json_content)

    assert len(result) == 2
    assert all(r["detection_type"] == "agent_in_list" for r in result)
    assert all("index" in r for r in result)


def test_is_agent_definition_with_name_and_llm(detector: StructuredAgentDetector) -> None:
    """Check if dict with name and llm is detected as agent."""
    data = {"name": "Test", "llm": "gpt-4"}

    result = detector._is_agent_definition(data)

    assert result is True


def test_is_agent_definition_with_name_and_model(detector: StructuredAgentDetector) -> None:
    """Check if dict with name and model is detected as agent."""
    data = {"name": "Test", "model": "claude"}

    result = detector._is_agent_definition(data)

    assert result is True


def test_is_agent_definition_with_role_and_goal(detector: StructuredAgentDetector) -> None:
    """Check if dict with role and goal is detected as agent."""
    data = {"role": "Researcher", "goal": "Find info"}

    result = detector._is_agent_definition(data)

    assert result is True


def test_is_agent_definition_with_agent_key_and_llm(detector: StructuredAgentDetector) -> None:
    """Check if dict with agent-related key and LLM is detected."""
    data = {"agent_type": "conversational", "llm": "gpt-4"}

    result = detector._is_agent_definition(data)

    assert result is True


def test_is_agent_definition_with_agent_value_and_support_keys(detector: StructuredAgentDetector) -> None:
    """Check if dict with agent value indicator and support keys is detected."""
    data = {"style": "react", "tools": ["search"], "name": "ReactAgent"}

    result = detector._is_agent_definition(data)

    assert result is True


def test_is_agent_definition_with_non_dict(detector: StructuredAgentDetector) -> None:
    """Check if non-dict returns False."""
    result = detector._is_agent_definition("not a dict")

    assert result is False


def test_is_agent_definition_with_empty_dict(detector: StructuredAgentDetector) -> None:
    """Check if empty dict returns False."""
    result = detector._is_agent_definition({})

    assert result is False


def test_is_agent_definition_with_unrelated_keys(detector: StructuredAgentDetector) -> None:
    """Check if dict with unrelated keys returns False."""
    data = {"username": "john", "email": "john@example.com"}

    result = detector._is_agent_definition(data)

    assert result is False


def test_extract_agent_name_with_name_key(detector: StructuredAgentDetector) -> None:
    """Extract agent name from name key."""
    data = {"name": "ExtractedName", "llm": "gpt-4"}

    result = detector._extract_agent_name(data)

    assert result == "ExtractedName"


def test_extract_agent_name_with_agent_name_key(detector: StructuredAgentDetector) -> None:
    """Extract agent name from agent_name key."""
    data = {"agent_name": "AgentNameValue", "model": "claude"}

    result = detector._extract_agent_name(data)

    assert result == "AgentNameValue"


def test_extract_agent_name_with_role_key(detector: StructuredAgentDetector) -> None:
    """Extract agent name from role key when name not present."""
    data = {"role": "Coordinator", "goal": "Coordinate tasks"}

    result = detector._extract_agent_name(data)

    assert result == "Coordinator"


def test_extract_agent_name_with_empty_name(detector: StructuredAgentDetector) -> None:
    """Extract agent name returns UnnamedAgent for empty name."""
    data = {"name": "", "llm": "gpt-4"}

    result = detector._extract_agent_name(data)

    assert result == "UnnamedAgent"


def test_extract_agent_name_with_no_name_keys(detector: StructuredAgentDetector) -> None:
    """Extract agent name returns UnnamedAgent when no name keys present."""
    data = {"llm": "gpt-4", "tools": ["search"]}

    result = detector._extract_agent_name(data)

    assert result == "UnnamedAgent"


def test_extract_agent_name_with_non_dict(detector: StructuredAgentDetector) -> None:
    """Extract agent name returns UnnamedAgent for non-dict input."""
    result = detector._extract_agent_name("not a dict")

    assert result == "UnnamedAgent"


def test_calculate_confidence_with_required_keys(detector: StructuredAgentDetector) -> None:
    """Calculate confidence returns high score for required key patterns."""
    data = {"name": "Test", "llm": "gpt-4"}

    result = detector._calculate_confidence(data)

    assert result >= 0.4


def test_calculate_confidence_with_optional_keys(detector: StructuredAgentDetector) -> None:
    """Calculate confidence increases with optional keys."""
    data = {"name": "Test", "llm": "gpt-4", "tools": ["search"], "instructions": "Do work"}

    result = detector._calculate_confidence(data)

    assert result > 0.4


def test_calculate_confidence_with_value_indicators(detector: StructuredAgentDetector) -> None:
    """Calculate confidence increases with agent value indicators."""
    data = {"name": "Test", "llm": "gpt-4", "style": "react"}

    result = detector._calculate_confidence(data)

    assert result > 0.4


def test_calculate_confidence_capped_at_one(detector: StructuredAgentDetector) -> None:
    """Calculate confidence never exceeds 1.0."""
    data = {
        "name": "Test",
        "llm": "gpt-4",
        "tools": ["a", "b"],
        "instructions": "x",
        "system_prompt": "y",
        "backstory": "z",
        "style": "react",
    }

    result = detector._calculate_confidence(data)

    assert result <= 1.0


def test_calculate_confidence_with_non_dict(detector: StructuredAgentDetector) -> None:
    """Calculate confidence returns 0.0 for non-dict input."""
    result = detector._calculate_confidence("not a dict")

    assert result == pytest.approx(0.0)


def test_calculate_confidence_logs_debug(detector: StructuredAgentDetector, caplog) -> None:
    """Calculate confidence logs debug message with score."""
    data = {"name": "Test", "llm": "gpt-4"}

    caplog.set_level(logging.DEBUG)
    detector._calculate_confidence(data)

    assert "Agent detection confidence" in caplog.text


@pytest.mark.parametrize(
    "yaml_content,expected_count",
    [
        ("name: A1\nllm: gpt-4", 1),
        ("role: R1\ngoal: G1", 1),
        ("agents:\n  - name: A1\n    llm: gpt-4\n  - name: A2\n    model: claude", 2),
        ("random: data\nother: values", 0),
    ],
)
def test_detect_in_yaml_parametrised(detector: StructuredAgentDetector, yaml_content: str, expected_count: int) -> None:
    """Detect agents in various YAML structures with parametrised inputs."""
    result = detector.detect_in_yaml(yaml_content)

    assert len(result) == expected_count


@pytest.mark.parametrize(
    "json_content,expected_count",
    [
        ('{"name": "A1", "llm": "gpt-4"}', 1),
        ('{"role": "R1", "goal": "G1"}', 1),
        ('{"agents": [{"name": "A1", "llm": "gpt-4"}, {"name": "A2", "model": "claude"}]}', 2),
        ('{"random": "data", "other": "values"}', 0),
    ],
)
def test_detect_in_json_parametrised(detector: StructuredAgentDetector, json_content: str, expected_count: int) -> None:
    """Detect agents in various JSON structures with parametrised inputs."""
    result = detector.detect_in_json(json_content)

    assert len(result) == expected_count


class TestStructuredFileDetection:
    """Test detection in YAML and JSON files."""

    def test_get_structured_locations_with_yaml_file(self, tmp_path) -> None:  # NOSONAR S2325
        """YAML files are detected and locations returned correctly."""
        from src.detectors.agents.agents import AgentDetector

        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("""
name: MyAgent
llm: gpt-4
tools:
  - search
  - calculator
""")

        detector = AgentDetector()
        locations = detector.get_structured_agent_locations(yaml_file.read_text(), "agents.yaml")

        assert len(locations) >= 1
        assert locations[0]["name"] == "MyAgent"

    def test_get_structured_locations_with_json_file(self, tmp_path) -> None:  # NOSONAR S2325
        """JSON files are detected and locations returned correctly."""
        from src.detectors.agents.agents import AgentDetector

        json_file = tmp_path / "agents.json"
        json_file.write_text("""
{
  "name": "WorkerAgent",
  "model": "claude-3",
  "tools": ["search"]
}
""")

        detector = AgentDetector()
        locations = detector.get_structured_agent_locations(json_file.read_text(), "agents.json")

        assert len(locations) >= 1
        assert locations[0]["name"] == "WorkerAgent"

    @staticmethod
    def test_get_structured_locations_with_unsupported_file() -> None:
        """Unsupported file types return empty list."""
        from src.detectors.agents.agents import AgentDetector

        detector = AgentDetector()
        locations = detector.get_structured_agent_locations("content", "file.txt")

        assert locations == []

    def test_get_structured_locations_with_invalid_yaml(self, caplog) -> None:  # NOSONAR S2325
        """Invalid YAML returns empty locations and logs debug message."""
        from src.detectors.agents.agents import AgentDetector

        invalid_yaml = "invalid: [unclosed"

        detector = AgentDetector()

        with caplog.at_level(logging.DEBUG):
            locations = detector.get_structured_agent_locations(invalid_yaml, "config.yaml")

        assert locations == []

    def test_get_structured_locations_with_invalid_json(self, caplog) -> None:  # NOSONAR S2325
        """Invalid JSON returns empty locations and logs debug message."""
        from src.detectors.agents.agents import AgentDetector

        invalid_json = '{"invalid": [unclosed}'

        detector = AgentDetector()

        with caplog.at_level(logging.DEBUG):
            locations = detector.get_structured_agent_locations(invalid_json, "config.json")

        assert locations == []
