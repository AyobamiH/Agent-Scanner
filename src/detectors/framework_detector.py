"""Framework detection based on imports and dependencies."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from fnmatch import fnmatch
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.exceptions import ConfigurationError
from src.models.results import DependencyInfo

logger = logging.getLogger(__name__)

IMPORT_SCORE = 5
DEPENDENCY_SCORE = 1


def _normalise_patterns(value: Any, field_name: str) -> list[str]:
    """
    Ensure configuration pattern lists contain non-empty strings.

    Args:
        value: Raw value from configuration.
        field_name: Name of the field being normalised for error messages.

    Returns:
        A list of trimmed patterns.

    Raises:
        ConfigurationError: If the value is not a list of strings.
    """

    if not isinstance(value, list):
        raise ConfigurationError(f"{field_name} must be a list of strings")

    patterns: list[str] = []
    for pattern in value:
        if not isinstance(pattern, str):
            raise ConfigurationError(f"{field_name} entries must be strings")
        trimmed = pattern.strip()
        if trimmed:
            patterns.append(trimmed)

    return patterns


def _load_framework_config(path: Path) -> dict[str, Any]:
    """
    Load framework detection configuration from JSON.

    Args:
        path: Path to the keywords configuration file.

    Returns:
        Framework detection section of the configuration.

    Raises:
        ConfigurationError: If the configuration file is missing, invalid, or malformed.
    """

    if not path:
        error_message = "path cannot be empty"
        raise ConfigurationError(error_message)

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except FileNotFoundError as exc:
        error_message = f"Configuration file not found: {path}"
        raise ConfigurationError(error_message) from exc
    except OSError as exc:
        error_message = f"Could not read configuration file: {path}"
        raise ConfigurationError(error_message) from exc
    except JSONDecodeError as exc:
        error_message = f"Keywords JSON is malformed: {path}"
        raise ConfigurationError(error_message) from exc

    framework_section = data.get("framework_detection")
    if framework_section is None:
        raise ConfigurationError("framework_detection section is missing from configuration")
    if not isinstance(framework_section, dict):
        raise ConfigurationError("framework_detection section must be an object")

    frameworks = framework_section.get("frameworks", [])
    infrastructure = framework_section.get("infrastructure", [])
    if not isinstance(frameworks, list) or not isinstance(infrastructure, list):
        raise ConfigurationError("framework_detection frameworks and infrastructure must be lists")

    normalised_frameworks: list[dict[str, Any]] = []
    for entry in frameworks:
        if not isinstance(entry, dict):
            raise ConfigurationError("framework entries must be objects")
        canonical_name = entry.get("canonical_name")
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ConfigurationError("framework entries must include canonical_name")
        normalised_frameworks.append(
            {
                "canonical_name": canonical_name.strip(),
                "import_patterns": _normalise_patterns(entry.get("import_patterns", []), "import_patterns"),
                "dependency_patterns": _normalise_patterns(entry.get("dependency_patterns", []), "dependency_patterns"),
            }
        )

    normalised_infrastructure: list[dict[str, Any]] = []
    for entry in infrastructure:
        if not isinstance(entry, dict):
            raise ConfigurationError("infrastructure entries must be objects")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("infrastructure entries must include name")
        normalised_infrastructure.append(
            {
                "name": name.strip(),
                "patterns": _normalise_patterns(entry.get("patterns", []), "patterns"),
            }
        )

    logger.info(
        "Loaded framework configuration from %s: %d frameworks, %d infrastructure entries",
        path,
        len(normalised_frameworks),
        len(normalised_infrastructure),
    )

    return {"frameworks": normalised_frameworks, "infrastructure": normalised_infrastructure}


def _match_pattern(pattern: str, value: str) -> bool:
    """
    Perform a case-insensitive match supporting wildcards.

    Args:
        pattern: Pattern with optional "*" wildcards.
        value: Value to compare against the pattern.

    Returns:
        True when the value matches the pattern.
    """

    pattern_lower = pattern.lower()
    value_lower = value.lower()
    if "*" in pattern_lower:
        return fnmatch(value_lower, pattern_lower)
    return pattern_lower in value_lower


def _get_dependency_name(dependency: DependencyInfo | Any) -> str:
    """
    Safely extract the dependency package name.

    Args:
        dependency: Dependency object or any object with package_name attribute.

    Returns:
        Package name string, or empty string when unavailable.
    """

    if isinstance(dependency, DependencyInfo):
        return dependency.package_name or ""
    return getattr(dependency, "package_name", "") or ""


class FrameworkDetector:
    """
    Detect the primary agent framework and supporting infrastructure.

    Attributes:
        keywords_path: Path to the configuration file describing keywords.
        config: Parsed configuration content.
        frameworks: Framework detection entries.
        infrastructure: Supporting infrastructure detection entries.
    """

    def __init__(self, keywords_path: str | Path | None = None) -> None:
        """
        Build a detector using the supplied configuration file.

        Args:
            keywords_path: Path to the keywords configuration file.

        Raises:
            ConfigurationError: If the configuration cannot be loaded or validated.
        """
        if keywords_path is None:
            keywords_path = Path(__file__).parent.parent / "config" / "keywords.json"
        self.keywords_path = Path(keywords_path)
        self.config = _load_framework_config(self.keywords_path)
        self.frameworks = self.config.get("frameworks", [])
        self.infrastructure = self.config.get("infrastructure", [])

    def detect_frameworks(
        self,
        imports: Iterable[str] | None = None,
        dependencies: Iterable[DependencyInfo] | None = None,
    ) -> dict[str, Any]:
        """
        Determine frameworks and infrastructure from imports and dependencies.

        Args:
            imports: Collection of import statements or module paths observed.
            dependencies: Collection of dependencies to assess.

        Returns:
            A mapping containing the main framework, supporting infrastructure,
            individual scores, and multi-framework flag.
        """

        import_names = list(imports or [])
        dependency_items = list(dependencies or [])

        scores: dict[str, int] = {}

        matched_infrastructure_dependencies: set[str] = set()
        for infrastructure_entry in self.infrastructure:
            pattern_list = infrastructure_entry.get("patterns", [])
            for dependency in dependency_items:
                dependency_name = _get_dependency_name(dependency)
                if not dependency_name:
                    continue
                if any(_match_pattern(pattern, dependency_name) for pattern in pattern_list):
                    matched_infrastructure_dependencies.add(dependency_name.lower())

        for framework_entry in self.frameworks:
            canonical_name = framework_entry.get("canonical_name")
            if not canonical_name:
                continue
            score = 0
            import_patterns = framework_entry.get("import_patterns", [])
            dependency_patterns = framework_entry.get("dependency_patterns", [])

            for import_name in import_names:
                if any(_match_pattern(pattern, import_name) for pattern in import_patterns):
                    score += IMPORT_SCORE

            has_dependency_match = False
            for dependency in dependency_items:
                dependency_name = _get_dependency_name(dependency)
                if not dependency_name:
                    continue
                if dependency_name.lower() in matched_infrastructure_dependencies:
                    continue
                if any(_match_pattern(pattern, dependency_name) for pattern in dependency_patterns):
                    has_dependency_match = True
                    break

            if has_dependency_match:
                score += DEPENDENCY_SCORE

            if score > 0:
                scores[canonical_name] = score

        main_framework = None
        multi_framework = False

        if scores:
            sorted_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            top_name, top_score = sorted_scores[0]
            main_framework = top_name
            if len(sorted_scores) > 1:
                second_score = sorted_scores[1][1]
                if second_score >= 0.8 * top_score:
                    multi_framework = True

        supporting_infrastructure: list[str] = []
        for infrastructure_entry in self.infrastructure:
            infrastructure_name = infrastructure_entry.get("name")
            pattern_list = infrastructure_entry.get("patterns", [])
            if not infrastructure_name:
                continue
            for dependency in dependency_items:
                dependency_name = _get_dependency_name(dependency)
                if not dependency_name:
                    continue
                if any(_match_pattern(pattern, dependency_name) for pattern in pattern_list):
                    supporting_infrastructure.append(infrastructure_name)
                    break

        for framework_name in scores.keys():
            if framework_name != main_framework:
                supporting_infrastructure.append(framework_name)

        seen_infrastructure: set[str] = set()
        deduped_infrastructure: list[str] = []
        for infrastructure_name in supporting_infrastructure:
            if infrastructure_name in seen_infrastructure:
                continue
            seen_infrastructure.add(infrastructure_name)
            deduped_infrastructure.append(infrastructure_name)

        supporting_infrastructure = deduped_infrastructure

        logger.info(
            "Framework detection result: main=%s multi_framework=%s supporting=%s",
            main_framework,
            multi_framework,
            supporting_infrastructure,
        )

        return {
            "main_framework": main_framework,
            "supporting_infrastructure": supporting_infrastructure,
            "framework_scores": scores,
            "multi_framework": multi_framework,
        }
