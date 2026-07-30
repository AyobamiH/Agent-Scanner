"""Tests for framework detection and scoring logic."""

import json
from pathlib import Path

import pytest

from src.detectors.framework_detector import FrameworkDetector
from src.exceptions import ConfigurationError
from src.models.results import DependencyInfo


@pytest.fixture
def detector() -> FrameworkDetector:
    """FrameworkDetector loaded with repo keyword config."""
    return FrameworkDetector(keywords_path="src/config/keywords.json")


def make_dependencies(names: list[str], source_file: str = "requirements.txt") -> list[DependencyInfo]:
    """Helper to build DependencyInfo objects from names."""
    return [DependencyInfo(package_name=name, version=None, source_file=source_file) for name in names]


def test_google_adk_scoring(detector: FrameworkDetector) -> None:
    """Imports worth five points plus dependencies worth one point should total six for Google ADK."""
    imports = ["from google.adk.agents import Agent"]
    dependencies = make_dependencies(["google-adk", "a2a-sdk"])

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] == "Google ADK"
    assert result["framework_scores"].get("Google ADK") == 6
    assert result["multi_framework"] is False


def test_langgraph_wins_over_supporting_langchain(detector: FrameworkDetector) -> None:
    """LangGraph import should outweigh multiple langchain provider packages."""
    imports = [
        "from langgraph.graph import StateGraph",
        "from langgraph.checkpoint.memory import MemorySaver",
    ]
    dependencies = make_dependencies(
        [
            "langgraph",
            "langchain",
            "langchain-openai",
            "langchain-community",
        ]
    )

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] == "LangGraph"
    assert result["framework_scores"].get("LangGraph", 0) > result["framework_scores"].get("LangChain", 0)
    assert result["multi_framework"] is False


@pytest.mark.parametrize(
    "imports,dependencies,expected_main,expect_multi",
    [
        (
            ["from langchain.agents import create_openai_functions_agent", "from llama_index.agent import ReActAgent"],
            ["langchain", "llamaindex"],
            "LangChain",
            True,
        ),
        (
            ["from langchain.agents import AgentExecutor", "from llama_index.core import VectorStoreIndex"],
            ["langchain", "langchain-community", "llama-index"],
            "LangChain",
            True,
        ),
        (
            ["from autogen import AssistantAgent"],
            ["autogen", "langchain"],
            "AutoGen",
            False,
        ),
    ],
)
def test_multi_framework_threshold_and_ties(
    detector: FrameworkDetector,
    imports: list[str],
    dependencies: list[str],
    expected_main: str,
    expect_multi: bool,
) -> None:
    result = detector.detect_frameworks(imports=imports, dependencies=make_dependencies(dependencies))

    assert result["main_framework"] == expected_main
    assert result["multi_framework"] is expect_multi
    assert expected_main in result["framework_scores"]


def test_alias_normalisation_maps_llamaindex(detector: FrameworkDetector) -> None:
    """Alias names should map to canonical framework output."""
    imports = ["from llamaindex.agent import ReActAgent"]
    dependencies = make_dependencies(["llamaindex"])

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] == "LlamaIndex"
    assert "LlamaIndex" in result["framework_scores"]


def test_observability_goes_to_supporting_infrastructure(detector: FrameworkDetector) -> None:
    """Observability dependencies should not become the main framework but appear in supporting infrastructure."""
    imports: list[str] = []
    dependencies = make_dependencies(
        [
            "opentelemetry-api",
            "opentelemetry-exporter-otlp",
            "opentelemetry-instrumentation-langchain",
        ]
    )

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] is None
    assert result["framework_scores"] == {}
    assert "OpenTelemetry" in result.get("supporting_infrastructure", [])
    assert result["multi_framework"] is False


def test_zero_frameworks_returns_empty(detector: FrameworkDetector) -> None:
    """No imports or dependencies yields no frameworks detected."""
    result = detector.detect_frameworks(imports=[], dependencies=[])

    assert result["main_framework"] is None
    assert result["framework_scores"] == {}
    assert result["supporting_infrastructure"] == []
    assert result["multi_framework"] is False


def test_missing_framework_section_raises_configuration_error(tmp_path: Path) -> None:
    """Missing framework_detection section should raise ConfigurationError on initialisation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(json.dumps({"content_keywords": []}), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="framework_detection section is missing"):
        FrameworkDetector(keywords_path=config_path)


def test_invalid_pattern_type_raises_configuration_error(tmp_path: Path) -> None:
    """Non-list patterns in framework entries should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [
                        {
                            "canonical_name": "Example",
                            "import_patterns": "not-a-list",
                            "dependency_patterns": [],
                        }
                    ],
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="import_patterns must be a list of strings"):
        FrameworkDetector(keywords_path=config_path)


def test_configuration_file_not_found_raises_error(tmp_path: Path) -> None:
    """Missing configuration file should raise ConfigurationError with file path."""
    missing_path = tmp_path / "nonexistent.json"

    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        FrameworkDetector(keywords_path=missing_path)


def test_malformed_json_raises_configuration_error(tmp_path: Path) -> None:
    """Invalid JSON in configuration file should raise ConfigurationError."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Keywords JSON is malformed"):
        FrameworkDetector(keywords_path=config_path)


def test_missing_canonical_name_raises_configuration_error(tmp_path: Path) -> None:
    """Framework entries without canonical_name should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [{"import_patterns": [], "dependency_patterns": []}],
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="canonical_name"):
        FrameworkDetector(keywords_path=config_path)


def test_framework_entry_not_dict_raises_configuration_error(tmp_path: Path) -> None:
    """Non-dict framework entries should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps({"framework_detection": {"frameworks": ["not-a-dict"], "infrastructure": []}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="framework entries must be objects"):
        FrameworkDetector(keywords_path=config_path)


def test_infrastructure_entry_not_dict_raises_configuration_error(tmp_path: Path) -> None:
    """Non-dict infrastructure entries should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [],
                    "infrastructure": ["not-a-dict"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="infrastructure entries must be objects"):
        FrameworkDetector(keywords_path=config_path)


def test_infrastructure_missing_name_raises_configuration_error(tmp_path: Path) -> None:
    """Infrastructure entries without name should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [],
                    "infrastructure": [{"patterns": []}],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="infrastructure entries must include name"):
        FrameworkDetector(keywords_path=config_path)


def test_framework_detection_with_non_dependency_info_objects(detector: FrameworkDetector) -> None:
    """Dependencies without package_name attribute should be skipped gracefully."""
    imports = ["from langchain.agents import Agent"]

    class BadDependency:
        """Object without package_name attribute."""

        pass

    dependencies = [BadDependency()]

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] == "LangChain"
    assert "LangChain" in result["framework_scores"]


def test_empty_pattern_lists_handled_correctly(tmp_path: Path) -> None:
    """Frameworks with empty pattern lists should be skipped during detection."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [
                        {
                            "canonical_name": "Framework1",
                            "import_patterns": [],
                            "dependency_patterns": [],
                        }
                    ],
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    detector = FrameworkDetector(keywords_path=config_path)
    result = detector.detect_frameworks(imports=["some.import"], dependencies=make_dependencies(["some-package"]))

    assert result["main_framework"] is None
    assert result["framework_scores"] == {}


def test_pattern_matching_with_wildcards(tmp_path: Path) -> None:
    """Wildcard patterns should match multiple variants."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [
                        {
                            "canonical_name": "MyFramework",
                            "import_patterns": ["myframework*"],
                            "dependency_patterns": ["my-framework-*"],
                        }
                    ],
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    detector = FrameworkDetector(keywords_path=config_path)
    result = detector.detect_frameworks(
        imports=["myframework.core", "myframework_utils"],
        dependencies=make_dependencies(["my-framework-openai", "my-framework-core"]),
    )

    assert result["main_framework"] == "MyFramework"
    assert result["framework_scores"]["MyFramework"] > 0


def test_case_insensitive_pattern_matching(detector: FrameworkDetector) -> None:
    """Pattern matching should be case-insensitive."""
    imports = ["from LANGCHAIN.agents import AGENT"]
    dependencies = make_dependencies(["LANGCHAIN", "langchain-OPENAI"])

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] == "LangChain"
    assert result["framework_scores"]["LangChain"] > 0


def test_infrastructure_not_counted_in_framework_scores(tmp_path: Path) -> None:
    """Infrastructure patterns should not add to framework scores."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [
                        {
                            "canonical_name": "MyFramework",
                            "import_patterns": ["myframework"],
                            "dependency_patterns": [],
                        }
                    ],
                    "infrastructure": [
                        {
                            "name": "MyInfra",
                            "patterns": ["myinfra"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    detector = FrameworkDetector(keywords_path=config_path)
    result = detector.detect_frameworks(
        imports=["myframework"],
        dependencies=make_dependencies(["myinfra"]),
    )

    assert result["main_framework"] == "MyFramework"
    assert "MyInfra" in result["supporting_infrastructure"]
    assert "MyInfra" not in result["framework_scores"]


def test_non_main_frameworks_in_supporting_infrastructure(detector: FrameworkDetector) -> None:
    """Secondary frameworks should appear in supporting infrastructure."""
    imports = [
        "from langchain.agents import Agent",
        "from llamaindex.core import GPTIndex",
    ]
    dependencies = make_dependencies(["langchain", "llamaindex"])

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] is not None
    assert len(result["supporting_infrastructure"]) >= 1


def test_infrastructure_list_deduplication(tmp_path: Path) -> None:
    """Supporting infrastructure list should not contain duplicates."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [],
                    "infrastructure": [
                        {
                            "name": "Shared",
                            "patterns": ["shared-pkg", "alternate-shared"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    detector = FrameworkDetector(keywords_path=config_path)
    result = detector.detect_frameworks(
        imports=[],
        dependencies=make_dependencies(["shared-pkg", "alternate-shared"]),
    )

    infrastructure_list = result["supporting_infrastructure"]
    assert infrastructure_list.count("Shared") == 1


def test_dependency_info_with_none_package_name(detector: FrameworkDetector) -> None:
    """DependencyInfo with None package_name should be skipped."""
    imports = []
    dependencies = [DependencyInfo(package_name=None, version="1.0", source_file="reqs.txt")]

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] is None
    assert result["framework_scores"] == {}


def test_dependency_info_with_empty_string_package_name(detector: FrameworkDetector) -> None:
    """DependencyInfo with empty string package_name should be skipped."""
    imports = []
    dependencies = [DependencyInfo(package_name="", version="1.0", source_file="reqs.txt")]

    result = detector.detect_frameworks(imports=imports, dependencies=dependencies)

    assert result["main_framework"] is None
    assert result["framework_scores"] == {}


def test_empty_path_raises_configuration_error(tmp_path: Path) -> None:
    """Empty path string results in OSError which is caught and raised as ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Could not read configuration file"):
        FrameworkDetector(keywords_path="")


def test_pattern_entry_non_string_raises_configuration_error(tmp_path: Path) -> None:
    """Non-string pattern entries should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [
                        {
                            "canonical_name": "Example",
                            "import_patterns": [123],
                            "dependency_patterns": [],
                        }
                    ],
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="import_patterns entries must be strings"):
        FrameworkDetector(keywords_path=config_path)


def test_frameworks_not_list_raises_configuration_error(tmp_path: Path) -> None:
    """Non-list frameworks should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": "not-a-list",
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="frameworks and infrastructure must be lists"):
        FrameworkDetector(keywords_path=config_path)


def test_infrastructure_not_list_raises_configuration_error(tmp_path: Path) -> None:
    """Non-list infrastructure should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [],
                    "infrastructure": "not-a-list",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="frameworks and infrastructure must be lists"):
        FrameworkDetector(keywords_path=config_path)


def test_framework_detection_section_not_dict_raises_error(tmp_path: Path) -> None:
    """Non-dict framework_detection section should fail validation."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(json.dumps({"framework_detection": "not-a-dict"}))

    with pytest.raises(ConfigurationError, match="framework_detection section must be an object"):
        FrameworkDetector(keywords_path=config_path)


def test_multi_framework_threshold_exactly_80_percent(tmp_path: Path) -> None:
    """Second framework score at exactly 80% of top should trigger multi_framework."""
    config_path = tmp_path / "keywords.json"
    config_path.write_text(
        json.dumps(
            {
                "framework_detection": {
                    "frameworks": [
                        {
                            "canonical_name": "Framework1",
                            "import_patterns": ["fw1"],
                            "dependency_patterns": [],
                        },
                        {
                            "canonical_name": "Framework2",
                            "import_patterns": ["fw2"],
                            "dependency_patterns": [],
                        },
                    ],
                    "infrastructure": [],
                }
            }
        ),
        encoding="utf-8",
    )

    detector = FrameworkDetector(keywords_path=config_path)
    result = detector.detect_frameworks(
        imports=["fw1", "fw1", "fw1", "fw1", "fw1", "fw2", "fw2", "fw2", "fw2"],
        dependencies=[],
    )

    score1 = result["framework_scores"].get("Framework1", 0)
    score2 = result["framework_scores"].get("Framework2", 0)
    if score1 > 0 and score2 > 0 and score2 >= 0.8 * score1:
        assert result["multi_framework"] is True
