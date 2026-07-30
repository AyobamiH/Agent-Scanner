"""Tests for agent detection in Bru HTTP request files."""

import json

from src.detectors.agents.agents import AgentDetector
from src.detectors.agents.structured_agents import StructuredAgentDetector


class TestBruFileSupport:
    """Test suite for Bru HTTP request file detection."""

    @staticmethod
    def test_extract_json_from_bru_body_simple() -> None:
        """Test extracting simple JSON from Bru body:json section."""
        bru_content = """
meta {
  name: create agent
  type: http
}

post {
  url: {{agent_url}}/
  body: json
}

body:json {
  {
    "name": "TestAgent",
    "llm": "openai"
  }
}
"""
        extracted = StructuredAgentDetector._extract_json_from_bru_body(bru_content)
        assert extracted is not None
        assert "TestAgent" in extracted
        assert "openai" in extracted

    @staticmethod
    def test_extract_json_from_bru_with_trailing_comma() -> None:
        """Test JSON extraction handles trailing commas in Bru format."""
        bru_content = """
body:json {
  {
    "name": "Agent1",
    "model": "gpt-4",
  }
}
"""
        extracted = StructuredAgentDetector._extract_json_from_bru_body(bru_content)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["name"] == "Agent1"

    @staticmethod
    def test_extract_json_with_script_section() -> None:
        """Test extraction with script section following body:json."""
        bru_content = """
body:json {
  {
    "name": "ScriptAgent",
    "system_prompt": "You are helpful"
  }
}

script:post-response {
  const body = res.getBody();
}
"""
        extracted = StructuredAgentDetector._extract_json_from_bru_body(bru_content)
        assert extracted is not None
        assert "ScriptAgent" in extracted

    @staticmethod
    def test_detect_agent_in_bru_content() -> None:
        """Test full agent detection from Bru file content."""
        detector = StructuredAgentDetector()

        bru_content = """
meta {
  name: create agent
}

body:json {
  {
    "name": "Jenkins_to_github_actions_migration_agent",
    "system_prompt": "You are an expert Jenkins to GitHub Actions migration engineer",
    "llm": "gpt-4"
  }
}
"""
        detections = detector.detect_in_bru(bru_content)
        assert len(detections) > 0
        assert detections[0]["name"] == "Jenkins_to_github_actions_migration_agent"

    @staticmethod
    def test_agent_detector_recognises_bru_extension() -> None:
        """Test that AgentDetector.get_structured_agent_locations handles .bru files."""
        detector = AgentDetector()

        bru_content = """
body:json {
  {
    "name": "TestAgent",
    "role": "analyst",
    "goal": "analyse data"
  }
}
"""
        locations = detector.get_structured_agent_locations(bru_content, "test.bru")
        assert len(locations) > 0
        assert any(loc["name"] == "TestAgent" for loc in locations)

    @staticmethod
    def test_empty_bru_file_returns_empty() -> None:
        """Test that empty Bru files return empty detections."""
        detector = StructuredAgentDetector()

        bru_content = """
meta {
  name: test
  type: http
}

get {
  url: https://example.com
}
"""
        detections = detector.detect_in_bru(bru_content)
        assert len(detections) == 0

    @staticmethod
    def test_bru_with_complex_json_body() -> None:
        """Test extraction of complex JSON from Bru with nested structures."""
        bru_content = """
body:json {
  {
    "name": "ComplexAgent",
    "system_prompt": "You are an expert",
    "llm": "openai",
    "input_schema": {
      "type": "object",
      "properties": {
        "text": {"type": "string"}
      }
    },
    "output_schema": {
      "type": "object",
      "properties": {
        "result": {"type": "string"}
      }
    }
  }
}
"""
        extracted = StructuredAgentDetector._extract_json_from_bru_body(bru_content)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["name"] == "ComplexAgent"
        assert "input_schema" in parsed
        assert "output_schema" in parsed

    @staticmethod
    def test_bru_with_system_prompt_phrases() -> None:
        """Test that Bru files with prompt phrases are detected as agents."""
        detector = StructuredAgentDetector()

        bru_content = """
body:json {
  {
    "name": "PromptedAgent",
    "system_prompt": "You are an expert data analyst. Your role is to provide insights.",
    "description": "Analysis agent",
    "model": "gpt-4"
  }
}
"""
        detections = detector.detect_in_bru(bru_content)
        assert len(detections) > 0

    @staticmethod
    def test_bru_file_with_multiple_agents() -> None:
        """Test Bru files that define multiple agents in nested structures."""
        detector = StructuredAgentDetector()

        bru_content = """
body:json {
  {
    "agents": [
      {
        "name": "Agent1",
        "role": "researcher",
        "goal": "research"
      },
      {
        "name": "Agent2",
        "role": "analyst",
        "goal": "analyze"
      }
    ]
  }
}
"""
        detections = detector.detect_in_bru(bru_content)
        assert len(detections) > 0

    @staticmethod
    def test_extract_json_without_bru_format() -> None:
        """Test that extraction returns None for non-Bru content."""
        content = """
{
  "name": "DirectJSON",
  "llm": "openai"
}
"""
        extracted = StructuredAgentDetector._extract_json_from_bru_body(content)
        assert extracted is None

    @staticmethod
    def test_bru_case_insensitive_body_json() -> None:
        """Test that body:json detection is case-insensitive."""
        bru_content = """
BODY:JSON {
  {
    "name": "CaseAgent",
    "llm": "openai"
  }
}
"""
        extracted = StructuredAgentDetector._extract_json_from_bru_body(bru_content)
        assert extracted is not None

    @staticmethod
    def test_real_jenkins_migration_agent_bru() -> None:
        """Test with realistic Jenkins migration agent Bru file content."""
        detector = StructuredAgentDetector()

        bru_content = """
meta {
  name: create agent
  type: http
  seq: 1
}

post {
  url: {{agent_url}}/
  body: json
  auth: inherit
}

headers {
  Content-Type: application/json
  x-workspace-id: {{x-workspace-id}}
}

body:json {
  {
    "name": "Jenkins_to_github_actions_migration_agent_v1_2_1",
    "description": "An agent that specializes in converting Jenkinsfiles into GitHub Actions workflow",
    "system_prompt": "You are an expert engineer. Your role is to analyse pipelines and generate pipelines",
    "llm": "gpt-4",
    "input_schema": {
      "type": "object",
      "properties": {
        "Jenkinsfile": {"type": "string"}
      }
    }
  }
}

script:post-response {
  const body = res.getBody();
  bru.setEnvVar("agentId", body.data.agent_id);
}
"""
        detections = detector.detect_in_bru(bru_content)
        assert len(detections) > 0
        assert detections[0]["name"] == "Jenkins_to_github_actions_migration_agent_v1_2_1"
