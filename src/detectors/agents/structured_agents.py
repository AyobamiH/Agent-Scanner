"""Detection for YAML/JSON-based agent definitions."""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

AGENT_REQUIRED_KEY_PATTERNS = [
    {"name", "llm"},
    {"name", "model"},
    {"agent_name", "model"},
    {"role", "goal"},
    {"style", "llm"},
]

AGENT_OPTIONAL_KEYS = {
    "tools",
    "functions",
    "actions",
    "instructions",
    "system_prompt",
    "backstory",
    "description",
    "goal",
    "role",
    "spec_version",
    "style",
    "collaborators",
}

AGENT_VALUE_INDICATORS = {
    "react",
    "tool_calling",
    "conversational",
    "assistant",
    "coordinator",
    "worker",
}


class StructuredAgentDetector:
    """Detect agent definitions in YAML/JSON configuration files."""

    def detect_in_yaml(self, content: str) -> list[dict[str, Any]]:
        """Detect agent definitions in YAML content.

        Args:
            content: YAML file content as string

        Returns:
            List of detected agent definitions with structure info
        """
        try:
            import yaml

            parser = yaml.safe_load
        except ImportError:
            logger.debug("PyYAML not installed, skipping YAML detection")
            return []

        return self._parse_and_detect(content, parser, "yaml")

    def detect_in_json(self, content: str) -> list[dict[str, Any]]:
        """Detect agent definitions in JSON content.

        Args:
            content: JSON file content as string

        Returns:
            List of detected agent definitions with structure info
        """
        return self._parse_and_detect(content, json.loads, "json")

    def detect_in_bru(self, content: str) -> list[dict[str, Any]]:
        """Detect agent definitions in Bru HTTP request files.

        Bru files contain HTTP requests with JSON bodies. This extracts the JSON
        body from the `body:json { ... }` section and detects agents within it.

        Args:
            content: Bru file content as string

        Returns:
            List of detected agent definitions with structure info
        """
        json_body = self._extract_json_from_bru_body(content)
        if json_body:
            return self.detect_in_json(json_body)
        return []

    @staticmethod
    def _extract_json_from_bru_body(content: str) -> str | None:
        """Extract JSON body from Bru HTTP request format.

        Bru files have the format:
            body:json {
              { ... json content ... }
            }

        Args:
            content: Bru file content

        Returns:
            Extracted JSON string, or None if not found
        """
        if not content:
            return None

        pattern = r"body:json\s*\{(.*?)\}\s*(?:script:|$)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            json_content = match.group(1).strip()
            json_content = re.sub(r",\s*}", "}", json_content)
            json_content = re.sub(r",\s*]", "]", json_content)
            return json_content

        return None

    def _parse_and_detect(self, content: str, parser: Callable[[str], Any], format_type: str) -> list[dict[str, Any]]:
        """Parse content and detect agents.

        Args:
            content: File content as string
            parser: Parsing function (yaml.safe_load or json.loads)
            format_type: Format type label ("yaml" or "json")

        Returns:
            List of detected agent definitions
        """
        if not content or not content.strip():
            return []

        try:
            data = parser(content)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", format_type, exc)
            return []

        detections = self._analyse_structure(data, format_type)
        if detections:
            logger.info("Detected %d agent(s) in %s content", len(detections), format_type)
        return detections

    def _analyse_structure(self, data: Any, format_type: str) -> list[dict[str, Any]]:
        """Analyse parsed data structure for agent patterns.

        Args:
            data: Parsed YAML/JSON data
            format_type: "yaml" or "json"

        Returns:
            List of agent detections
        """
        if data is None:
            return []

        detections: list[dict[str, Any]] = []
        stack: list[tuple[Any, str | None, str, int | None]] = [(data, None, "root", None)]

        while stack:
            node, parent_key, container_type, index = stack.pop()

            if isinstance(node, dict):
                detection = self._build_detection(node, format_type, container_type, parent_key, index)
                if detection:
                    detections.append(detection)

                for key, value in node.items():
                    stack.append((value, key, "dict", None))

            if isinstance(node, list):
                for index, item in enumerate(node):
                    stack.append((item, parent_key, "list", index))

        return detections

    def _build_detection(
        self, data: dict[str, Any], format_type: str, container_type: str, parent_key: str | None, index: int | None
    ) -> dict[str, Any] | None:
        """Create a detection record when a node matches an agent pattern.

        Args:
            data: Candidate agent definition dictionary
            format_type: Source format label
            container_type: Origin container category (root, dict, list)
            parent_key: Parent mapping key when applicable
            index: Index within a list when applicable

        Returns:
            Detection dictionary or None if the node does not match
        """
        if not self._is_agent_definition(data):
            return None

        detection: dict[str, Any] = {
            "detection_type": self._resolve_detection_type(container_type),
            "format": format_type,
            "name": self._extract_agent_name(data),
            "confidence": self._calculate_confidence(data),
        }

        if parent_key and container_type != "root":
            detection["parent_key"] = parent_key
        if container_type == "list" and index is not None:
            detection["index"] = index

        return detection

    @staticmethod
    def _resolve_detection_type(container_type: str) -> str:
        """Resolve detection type label based on container context.

        Args:
            container_type: Origin container category (root, dict, list)

        Returns:
            Detection type label
        """
        return {
            "dict": "nested_agent",
            "list": "agent_in_list",
        }.get(container_type, "agent_definition")

    @staticmethod
    def _normalise_keys(data: dict[str, Any]) -> set[str]:
        """normalise dictionary keys to lowercase.

        Args:
            data: Dictionary to normalise

        Returns:
            Set of lowercase keys
        """
        return {str(k).lower() for k in data.keys()}

    @staticmethod
    def _contains_prompt_phrases(text: str) -> bool:
        """Check if text contains agentic prompt phrases.

        Common indicators include "You are", "You are a", "Your role is", etc.

        Args:
            text: Text to check for prompt phrases

        Returns:
            True if agentic prompt phrases are detected
        """
        if not text or not isinstance(text, str):
            return False

        prompt_indicators = [
            "you are an",
            "you are a",
            "you are the",
            "your role is",
            "your task is",
            "you are an expert",
            "act as",
            "act as a",
            "assume the role",
            "you will act",
            "you will behave",
            "behave as",
            "think of yourself",
            "pretend you are",
            "specializes in",
            "specialized in",
            "specialises in",
            "specialised in",
            "expertise in",
        ]

        lower_text = text.lower()
        for indicator in prompt_indicators:
            if indicator in lower_text:
                logger.debug("Prompt phrase detected: '%s'", indicator)
                return True

        return False

    def _is_agent_definition(self, data: dict[str, Any]) -> bool:
        """Check if a dict structure represents an agent definition.

        Args:
            data: Dictionary to check

        Returns:
            True if this looks like an agent definition
        """
        if not isinstance(data, dict):
            return False

        keys = self._normalise_keys(data)

        if any(required_set.issubset(keys) for required_set in AGENT_REQUIRED_KEY_PATTERNS):
            logger.debug("Agent definition detected with required keys")
            return True

        if any("agent" in k for k in keys):
            llm_keys = {"llm", "model", "language_model", "chat_model"}
            if llm_keys & keys:
                logger.debug("Agent definition detected with 'agent' key and LLM reference")
                return True

        for value in data.values():
            if isinstance(value, str) and value.lower() in AGENT_VALUE_INDICATORS:
                support_keys = {"tools", "instructions", "system_prompt", "name"}
                if support_keys & keys:
                    logger.debug("Agent definition detected with agent-style value: %s", value.lower())
                    return True

        type_val = data.get("type") or data.get("Type") or data.get("TYPE")
        if type_val and str(type_val).lower() in ["agent", "agentic", "conversational", "autonomous"]:
            logger.debug("Agent definition detected with type='agent'")
            return True

        for key in ["agent_class", "agentclass", "agent_type", "agent_framework"]:
            if key in keys:
                logger.debug("Agent definition detected with agent_class/type field")
                return True

        if ("tools" in keys or "functions" in keys) and "name" in keys:
            logger.debug("Agent definition detected with tools/functions + name")
            return True

        if ("instructions" in keys or "system_prompt" in keys) and ("name" in keys or "role" in keys):
            logger.debug("Agent definition detected with instructions/prompt + name/role")
            return True

        has_agentic_values = any(
            str(v).lower() in AGENT_VALUE_INDICATORS for v in data.values() if isinstance(v, (str, int))
        )
        if has_agentic_values and ("name" in keys or "role" in keys):
            logger.debug("Agent definition detected with agentic value indicator")
            return True

        context_fields = ["system_prompt", "instruction", "instructions", "prompt", "backstory", "description"]
        for field in context_fields:
            if field in keys:
                field_value = data.get(field)
                if isinstance(field_value, str) and self._contains_prompt_phrases(field_value):
                    if "name" in keys or "role" in keys:
                        logger.debug(
                            "Agent definition detected via prompt phrases in '%s' field",
                            field,
                        )
                        return True

        return False

    @staticmethod
    def _extract_agent_name(data: dict[str, Any]) -> str:
        """Extract agent name from definition.

        Args:
            data: Agent definition dict

        Returns:
            Agent name or "UnnamedAgent"
        """
        if not isinstance(data, dict):
            return "UnnamedAgent"

        for key in ["name", "agent_name", "id", "role", "title"]:
            if key in data:
                name = data[key]
                if isinstance(name, str) and name.strip():
                    return name

        return "UnnamedAgent"

    def _calculate_confidence(self, data: dict[str, Any]) -> float:
        """Calculate confidence score for agent detection.

        Args:
            data: Agent definition dict

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not isinstance(data, dict):
            return 0.0

        keys = self._normalise_keys(data)
        score = 0.0

        if any(required_set.issubset(keys) for required_set in AGENT_REQUIRED_KEY_PATTERNS):
            score += 0.4

        optional_present = keys & AGENT_OPTIONAL_KEYS
        score += min(0.4, len(optional_present) * 0.1)

        if any(isinstance(v, str) and v.lower() in AGENT_VALUE_INDICATORS for v in data.values()):
            score += 0.2

        if self._contains_prompt_phrases(str(data.get("system_prompt", ""))):
            score += 0.15

        final_score = min(1.0, score)
        logger.debug("Agent detection confidence: %.2f", final_score)
        return final_score
