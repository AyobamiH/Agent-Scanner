"""Tests for keyword loading and pattern matching configuration."""

import json

import pytest

from src.detectors.keywords import load_keywords
from src.detectors.patterns import PatternMatcher
from src.exceptions import ConfigurationError


def test_load_keywords_defaults():
    """Default configuration should load successfully."""

    config = load_keywords("src/config/keywords.json")
    assert "content_keywords" in config
    assert isinstance(config["content_keywords"], list)


def test_pattern_matcher_path_and_content():
    """Pattern matcher should recognise common path and content signals."""

    matcher = PatternMatcher.from_file("src/config/keywords.json")
    assert matcher.match_path("agents/agent.py") is True
    assert matcher.match_content("This repo uses OpenAI and LangChain.") is True
    assert matcher.match_path("impainting/image.png") is False


def test_scoring_counts():
    """Scoring should award points for detected keywords."""

    matcher = PatternMatcher.from_file("src/config/keywords.json")
    assert matcher.score_path("agents/agent.py") >= 1
    s = "openai langchain llm"
    assert matcher.score_content(s) >= 3


def test_load_keywords_raises_on_malformed_json(tmp_path):
    """Malformed keywords JSON should raise ConfigurationError."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ not: valid json }")

    with pytest.raises(ConfigurationError):
        load_keywords(str(bad))


def test_load_keywords_raises_file_not_found():
    """Missing keywords file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_keywords("nonexistent/path.json")


def test_load_keywords_rejects_empty_path():
    """Empty keyword paths should be rejected early."""

    with pytest.raises(ConfigurationError):
        load_keywords("")


def test_load_keywords_rejects_non_object_json(tmp_path):
    """Non-object JSON payloads should raise ConfigurationError."""

    cfg = tmp_path / "config.json"
    cfg.write_text("[]")

    with pytest.raises(ConfigurationError):
        load_keywords(cfg)


def test_load_keywords_rejects_non_string_entries(tmp_path):
    """Lists with non-string entries are invalid."""

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dependency_keywords": [123, "openai"]}))

    with pytest.raises(ConfigurationError):
        load_keywords(cfg)


def test_load_keywords_accepts_agent_patterns_dict(tmp_path):
    """Agent patterns support structured dictionaries."""

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": ["openai"],
                "dependency_keywords": ["openai"],
                "agent_instantiation_patterns": {"framework": {"names": ["AgentOne"], "modules": ["pkg"]}},
            }
        )
    )

    config = load_keywords(cfg)

    assert config["agent_instantiation_patterns"] == {"framework": {"names": ["AgentOne"], "modules": ["pkg"]}}


def test_load_keywords_rejects_invalid_agent_patterns(tmp_path):
    """Agent pattern entries must be strings."""

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": ["openai"],
                "dependency_keywords": ["openai"],
                "agent_instantiation_patterns": {"framework": ["AgentOne", 1]},
            }
        )
    )

    with pytest.raises(ConfigurationError):
        load_keywords(cfg)


def test_load_keywords_rejects_agent_base_classes_non_list(tmp_path):
    """Non-list agent_base_classes should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"agent_base_classes": {"not": "list"}}))

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "agent_base_classes must be a list of strings" in str(exc.value)


def test_load_keywords_rejects_framework_modules_non_list(tmp_path):
    """Non-list framework_modules should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"framework_modules": 123}))

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "framework_modules must be a list of strings" in str(exc.value)


def test_load_keywords_rejects_strong_agentic_methods_non_list(tmp_path):
    """Non-list strong_agentic_methods should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"strong_agentic_methods": "single-string"}))

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "strong_agentic_methods must be a list of strings" in str(exc.value)


def test_load_keywords_rejects_none_path():
    """None path should be rejected."""
    with pytest.raises(ConfigurationError) as exc:
        load_keywords(None)
    assert "keywords path cannot be empty" in str(exc.value)


def test_load_keywords_rejects_whitespace_path():
    """Whitespace-only path should be rejected."""
    with pytest.raises(ConfigurationError) as exc:
        load_keywords("   ")
    assert "keywords path cannot be empty" in str(exc.value)


def test_load_keywords_rejects_non_list_content_keywords(tmp_path):
    """Non-list content_keywords should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"content_keywords": "not-a-list"}))

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "content_keywords must be a list of strings" in str(exc.value)


def test_load_keywords_rejects_non_string_list_entry(tmp_path):
    """List entries that are not strings should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": ["openai"],
                "dependency_keywords": [{"not": "string"}],
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "dependency_keywords entries must be strings" in str(exc.value)


def test_load_keywords_handles_oserror_on_read(tmp_path, monkeypatch):
    """OSError during file reading should raise ConfigurationError."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"content_keywords": ["test"]}))

    def mock_open(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "Could not read configuration file" in str(exc.value)


def test_load_keywords_strips_whitespace_from_list_entries(tmp_path):
    """Whitespace in list entries should be stripped."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": ["  openai  ", "\t langchain  \n"],
                "dependency_keywords": ["pytest"],
            }
        )
    )

    config = load_keywords(cfg)
    assert config["content_keywords"] == ["openai", "langchain"]


def test_load_keywords_skips_empty_string_entries(tmp_path):
    """Empty or whitespace-only entries should be skipped."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": ["openai", "   ", "", "langchain"],
                "dependency_keywords": ["pytest"],
            }
        )
    )

    config = load_keywords(cfg)
    assert config["content_keywords"] == ["openai", "langchain"]


def test_load_keywords_handles_none_in_list(tmp_path):
    """None values in lists should be treated as missing."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": None,
                "dependency_keywords": ["pytest"],
            }
        )
    )

    config = load_keywords(cfg)
    assert config["content_keywords"] == []


def test_load_keywords_agent_patterns_with_empty_string(tmp_path):
    """Empty strings in agent patterns should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "agent_instantiation_patterns": ["openai", ""],
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "agent_instantiation_patterns list entries must be non-empty strings" in str(exc.value)


def test_load_keywords_agent_patterns_as_list_of_strings(tmp_path):
    """Agent patterns can be a simple list of strings."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "agent_instantiation_patterns": ["Agent()", "Orchestrator()"],
            }
        )
    )

    config = load_keywords(cfg)
    assert config["agent_instantiation_patterns"] == ["Agent()", "Orchestrator()"]


def test_load_keywords_agent_patterns_with_non_string_value_in_dict(tmp_path):
    """Agent pattern dict with non-string value should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "agent_instantiation_patterns": {"framework": 123},
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "agent_instantiation_patterns values must be strings, lists of strings, or pattern dictionaries" in str(
        exc.value
    )


def test_load_keywords_agent_patterns_with_empty_key_in_dict(tmp_path):
    """Agent pattern dict with empty key should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "agent_instantiation_patterns": {"": ["Agent()"]},
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "agent_instantiation_patterns keys must be non-empty strings" in str(exc.value)


def test_load_keywords_agent_patterns_nested_names_and_modules(tmp_path):
    """Agent patterns support nested names and modules in dictionaries."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "agent_instantiation_patterns": {
                    "framework": {
                        "names": ["Agent", "Orchestrator"],
                        "modules": ["my_pkg.agents"],
                    }
                },
            }
        )
    )

    config = load_keywords(cfg)
    patterns = config["agent_instantiation_patterns"]
    assert patterns["framework"]["names"] == ["Agent", "Orchestrator"]
    assert patterns["framework"]["modules"] == ["my_pkg.agents"]


def test_load_keywords_agent_patterns_empty_nested_names(tmp_path):
    """Empty nested names should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "agent_instantiation_patterns": {"framework": {"names": [""]}},
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "names entries must be non-empty strings" in str(exc.value)


def test_load_keywords_orchestration_patterns_missing_treated_as_empty_dict(tmp_path):
    """Missing orchestration_patterns should default to empty dict."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dependency_keywords": ["pytest"]}))

    config = load_keywords(cfg)
    assert config["orchestration_patterns"] == {}


def test_load_keywords_orchestration_patterns_non_dict_rejected(tmp_path):
    """Non-dict orchestration_patterns should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "orchestration_patterns": ["not", "a", "dict"],
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "orchestration_patterns must be an object" in str(exc.value)


def test_load_keywords_framework_detection_missing_treated_as_empty_dict(tmp_path):
    """Missing framework_detection should default to empty dict."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dependency_keywords": ["pytest"]}))

    config = load_keywords(cfg)
    assert config["framework_detection"] == {}


def test_load_keywords_framework_detection_non_dict_rejected(tmp_path):
    """Non-dict framework_detection should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "framework_detection": "not-a-dict",
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "framework_detection must be an object" in str(exc.value)


def test_load_keywords_settings_missing_treated_as_empty_dict(tmp_path):
    """Missing settings should default to empty dict."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"dependency_keywords": ["pytest"]}))

    config = load_keywords(cfg)
    assert config["settings"] == {}


def test_load_keywords_settings_non_dict_rejected(tmp_path):
    """Non-dict settings should be rejected."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "settings": 42,
            }
        )
    )

    with pytest.raises(ConfigurationError) as exc:
        load_keywords(cfg)
    assert "settings must be an object" in str(exc.value)


def test_load_keywords_all_string_list_fields(tmp_path):
    """All string list fields should be properly normalised."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "content_keywords": ["ai"],
                "content_whole_words": ["agent"],
                "path_whole_words": ["agents"],
                "ignore_extensions": [".pyc"],
                "dependency_keywords": ["langchain"],
                "agent_base_classes": ["BaseAgent"],
                "framework_modules": ["langgraph"],
                "strong_agentic_methods": ["run"],
                "weak_agentic_methods": ["execute"],
                "skip_methods": ["__init__"],
                "llm_parameter_names": ["llm"],
                "agent_parameter_names": ["agent"],
                "tools_parameter_names": ["tools"],
                "llm_call_patterns": ["llm.invoke"],
                "llm_provider_methods": ["get_llm"],
                "llm_provider_modules": ["llm_providers"],
                "generic_role_names": ["user"],
                "setup_method_names": ["setup"],
            }
        )
    )

    config = load_keywords(cfg)
    assert config["content_keywords"] == ["ai"]
    assert config["content_whole_words"] == ["agent"]
    assert config["path_whole_words"] == ["agents"]
    assert config["ignore_extensions"] == [".pyc"]
    assert config["dependency_keywords"] == ["langchain"]
    assert config["agent_base_classes"] == ["BaseAgent"]
    assert config["framework_modules"] == ["langgraph"]
    assert config["strong_agentic_methods"] == ["run"]
    assert config["weak_agentic_methods"] == ["execute"]
    assert config["skip_methods"] == ["__init__"]
    assert config["llm_parameter_names"] == ["llm"]
    assert config["agent_parameter_names"] == ["agent"]
    assert config["tools_parameter_names"] == ["tools"]
    assert config["llm_call_patterns"] == ["llm.invoke"]
    assert config["llm_provider_methods"] == ["get_llm"]
    assert config["llm_provider_modules"] == ["llm_providers"]
    assert config["generic_role_names"] == ["user"]
    assert config["setup_method_names"] == ["setup"]
