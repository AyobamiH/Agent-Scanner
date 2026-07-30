"""Keywords loader for pattern matching configuration.

Loads and validates keyword configuration from JSON files for use in pattern
detection and dependency parsing.
"""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any, TypedDict

from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class AgentPatternDict(TypedDict, total=False):
    names: list[str]
    modules: list[str]


class KeywordsConfig(TypedDict, total=False):
    content_keywords: list[str]
    content_whole_words: list[str]
    path_whole_words: list[str]
    ignore_extensions: list[str]
    dependency_keywords: list[str]
    agent_instantiation_patterns: list[str] | dict[str, list[str] | AgentPatternDict | str]
    agent_base_classes: list[str]
    framework_modules: list[str]
    strong_agentic_methods: list[str]
    weak_agentic_methods: list[str]
    skip_methods: list[str]
    llm_parameter_names: list[str]
    agent_parameter_names: list[str]
    tools_parameter_names: list[str]
    prompt_phrase_patterns: dict[str, Any]
    llm_call_patterns: list[str]
    llm_provider_methods: list[str]
    llm_provider_modules: list[str]
    generic_role_names: list[str]
    setup_method_names: list[str]
    orchestration_patterns: dict[str, Any]
    framework_detection: dict[str, Any]
    settings: dict[str, Any]


def _normalise_str_list(config: dict[str, Any], key: str) -> list[str]:
    """Extract and validate a list of strings from a configuration mapping."""

    value = config.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        error_message = f"{key} must be a list of strings"
        raise ConfigurationError(error_message)

    normalised: list[str] = []
    for item in value:
        if not isinstance(item, str):
            error_message = f"{key} entries must be strings"
            raise ConfigurationError(error_message)
        stripped = item.strip()
        if stripped:
            normalised.append(stripped)
    return normalised


def _validate_agent_patterns(value: Any) -> list[str] | dict[str, list[str] | AgentPatternDict | str]:
    """Validate agent instantiation patterns allowing list or structured mapping."""

    if value is None:
        return []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str) or not item.strip():
                error_message = "agent_instantiation_patterns list entries must be non-empty strings"
                raise ConfigurationError(error_message)
        return [item.strip() for item in value if item.strip()]
    if not isinstance(value, dict):
        error_message = "agent_instantiation_patterns must be a list or object"
        raise ConfigurationError(error_message)

    validated: dict[str, list[str] | AgentPatternDict | str] = {}
    for key, patterns in value.items():
        if not isinstance(key, str) or not key.strip():
            error_message = "agent_instantiation_patterns keys must be non-empty strings"
            raise ConfigurationError(error_message)
        if isinstance(patterns, str):
            if not patterns.strip():
                error_message = "agent_instantiation_patterns string values must be non-empty"
                raise ConfigurationError(error_message)
            validated[key.strip()] = patterns.strip()
        elif isinstance(patterns, list):
            cleaned: list[str] = []
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern.strip():
                    error_message = "agent_instantiation_patterns list values must be non-empty strings"
                    raise ConfigurationError(error_message)
                cleaned.append(pattern.strip())
            validated[key.strip()] = cleaned
        elif isinstance(patterns, dict):
            nested: AgentPatternDict = {}
            if "names" in patterns:
                nested["names"] = _normalise_embedded_list(patterns["names"], "names")
            if "modules" in patterns:
                nested["modules"] = _normalise_embedded_list(patterns["modules"], "modules")
            validated[key.strip()] = nested
        else:
            error_message = (
                "agent_instantiation_patterns values must be strings, lists of strings, or pattern dictionaries"
            )
            raise ConfigurationError(error_message)
    return validated


def _normalise_embedded_list(value: Any, label: str) -> list[str]:
    """Normalise a nested list of strings used within pattern mappings."""

    if value is None:
        return []
    if not isinstance(value, list):
        error_message = f"{label} must be a list of strings"
        raise ConfigurationError(error_message)
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            error_message = f"{label} entries must be non-empty strings"
            raise ConfigurationError(error_message)
        cleaned.append(item.strip())
    return cleaned


def _normalise_mapping(value: Any, key: str) -> dict[str, Any]:
    """Ensure a configuration entry is a mapping."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        error_message = f"{key} must be an object"
        raise ConfigurationError(error_message)
    return value


def load_keywords(path: str | Path) -> KeywordsConfig:
    """Load and validate keywords configuration from a JSON file.

    Args:
        path: Path to the keywords configuration JSON file.

    Returns:
        Dictionary containing validated keyword lists and settings:
            - content_keywords: Substring keywords for content matching.
            - content_whole_words: Whole-word keywords for content matching.
            - path_whole_words: Whole-word keywords for path matching.
            - ignore_extensions: File extensions to skip.
            - dependency_keywords: Keywords for identifying AI dependencies.
            - agent_instantiation_patterns: Framework-specific agent patterns.
            - framework_detection: Framework detection configuration.
            - settings: Additional configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigurationError: If the JSON file is malformed or contains invalid types.
    """

    if path is None or (isinstance(path, str) and not path.strip()):
        error_message = "keywords path cannot be empty"
        raise ConfigurationError(error_message)

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file_handle:
            config: Any = json.load(file_handle)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file: {config_path}") from exc
    except JSONDecodeError as exc:
        raise ConfigurationError("Keywords JSON is malformed") from exc

    if not isinstance(config, dict):
        error_message = "Keywords JSON must be an object"
        raise ConfigurationError(error_message)

    result: KeywordsConfig = {
        "content_keywords": _normalise_str_list(config, "content_keywords"),
        "content_whole_words": _normalise_str_list(config, "content_whole_words"),
        "path_whole_words": _normalise_str_list(config, "path_whole_words"),
        "ignore_extensions": _normalise_str_list(config, "ignore_extensions"),
        "dependency_keywords": _normalise_str_list(config, "dependency_keywords"),
        "agent_instantiation_patterns": _validate_agent_patterns(config.get("agent_instantiation_patterns")),
        "agent_base_classes": _normalise_str_list(config, "agent_base_classes"),
        "framework_modules": _normalise_str_list(config, "framework_modules"),
        "strong_agentic_methods": _normalise_str_list(config, "strong_agentic_methods"),
        "weak_agentic_methods": _normalise_str_list(config, "weak_agentic_methods"),
        "skip_methods": _normalise_str_list(config, "skip_methods"),
        "llm_parameter_names": _normalise_str_list(config, "llm_parameter_names"),
        "agent_parameter_names": _normalise_str_list(config, "agent_parameter_names"),
        "tools_parameter_names": _normalise_str_list(config, "tools_parameter_names"),
        "prompt_phrase_patterns": _normalise_mapping(config.get("prompt_phrase_patterns"), "prompt_phrase_patterns"),
        "llm_call_patterns": _normalise_str_list(config, "llm_call_patterns"),
        "llm_provider_methods": _normalise_str_list(config, "llm_provider_methods"),
        "llm_provider_modules": _normalise_str_list(config, "llm_provider_modules"),
        "generic_role_names": _normalise_str_list(config, "generic_role_names"),
        "setup_method_names": _normalise_str_list(config, "setup_method_names"),
        "orchestration_patterns": _normalise_mapping(config.get("orchestration_patterns"), "orchestration_patterns"),
        "framework_detection": _normalise_mapping(config.get("framework_detection"), "framework_detection"),
        "settings": _normalise_mapping(config.get("settings"), "settings"),
    }

    logger.info(
        "Loaded keywords configuration from %s: %d content keywords, %d dependency keywords",
        config_path,
        len(result["content_keywords"]),
        len(result["dependency_keywords"]),
    )

    return result
