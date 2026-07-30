"""Edge cases for PatternMatcher and FrameworkDetector."""

import json
from pathlib import Path

import pytest

from src.detectors.framework_detector import FrameworkDetector
from src.detectors.patterns import PatternMatcher
from src.exceptions import ConfigurationError
from src.models.results import DependencyInfo


def test_pattern_matcher_sanitised_scoring():
    """Sanitised substrings should contribute to score_content."""

    matcher = PatternMatcher(content_keywords=["gpt-4"], content_whole_words=[], path_whole_words=[])
    score = matcher.score_content("uses gpt4 client")
    assert score >= 1


def test_pattern_matcher_plural_token_match():
    """Pluralised path tokens should match singular whole-word list."""

    matcher = PatternMatcher(content_keywords=[], content_whole_words=[], path_whole_words=["agent"])
    score = matcher.score_path("agents/utils.py")
    assert score >= 2


def test_framework_detector_multi_framework_tie(tmp_path: Path):
    """Close scores should mark multi_framework True and include supporting infra."""

    cfg = {
        "framework_detection": {
            "frameworks": [
                {"canonical_name": "fw1", "import_patterns": ["fw1"], "dependency_patterns": []},
                {"canonical_name": "fw2", "import_patterns": ["fw2"], "dependency_patterns": []},
            ],
            "infrastructure": [],
        }
    }
    cfg_path = tmp_path / "keywords.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    detector = FrameworkDetector(cfg_path)
    result = detector.detect_frameworks(imports=["fw1", "fw2"], dependencies=[])

    assert result["multi_framework"] is True
    assert set(result["supporting_infrastructure"]) >= {"fw2"}


def test_framework_detector_validation_error(tmp_path: Path):
    """Malformed config should raise ConfigurationError."""

    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        FrameworkDetector(cfg_path)


def test_framework_detector_dependency_patterns(tmp_path: Path):
    """Dependency patterns should contribute to framework score."""

    cfg = {
        "framework_detection": {
            "frameworks": [
                {
                    "canonical_name": "fw",
                    "import_patterns": [],
                    "dependency_patterns": ["fw-pkg"],
                }
            ],
            "infrastructure": [],
        }
    }
    cfg_path = tmp_path / "keywords.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    detector = FrameworkDetector(cfg_path)
    deps = [DependencyInfo(package_name="fw-pkg", version="1.0", source_file="reqs.txt")]
    result = detector.detect_frameworks(imports=[], dependencies=deps)

    assert result["main_framework"] == "fw"
