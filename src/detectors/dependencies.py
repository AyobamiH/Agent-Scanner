"""Dependency file parsing for AI and agent framework detection.

Supports multiple dependency formats including requirements.txt, pyproject.toml,
package.json, and other common package manifests.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.detectors.keywords import load_keywords

_DEFAULT_KEYWORDS_PATH = Path(__file__).parent.parent / "config" / "keywords.json"

logger = logging.getLogger(__name__)

COMMON_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "Pipfile",
    "package.json",
]

PROJECT_DEF = "[project]"
TOML_PROJECT_SECTIONS = [PROJECT_DEF, "[tool.poetry.dependencies]"]
GIT_URL_PATTERNS = ["git+", "http", "ssh+", "git://", ".git"]
EGG_URL_SEPARATOR = "#egg="
DEPENDENCY_STRING_PATTERN = r"['\"]([^'\"]+)['\"]"


class DependencyParseError(Exception):
    """Raised when dependency manifest parsing fails."""


class DependencyParser:
    """Parser for extracting AI-related dependencies from package manifests.

    Supports parsing of:
        - Python: requirements.txt, pyproject.toml, Pipfile
        - JavaScript: package.json, yarn.lock
        - Other: go.mod, Cargo.toml, Gemfile, pom.xml

    Uses configurable keyword matching to identify AI and agent framework dependencies.
    """

    def __init__(self, keywords_path: str | None = None) -> None:
        config = load_keywords(keywords_path or str(_DEFAULT_KEYWORDS_PATH))
        self.dependency_keywords = set(config.get("dependency_keywords", []) or [])
        logger.info("Beginning dependency parser with %d AI keywords", len(self.dependency_keywords))

    def is_ai_dependency(self, pkg_name: str) -> bool:
        """Determine if a package name indicates an AI or agent framework.

        Supports wildcard patterns (* prefix or suffix) and substring matching
        against configured dependency keywords.

        Args:
            pkg_name: Package or dependency name to check.

        Returns:
            True if the package matches any AI dependency keyword pattern.
        """
        if not pkg_name:
            return False
        low = pkg_name.lower()
        for kw in self.dependency_keywords:
            if kw.endswith("*"):
                pref = kw[:-1].lower()
                if low.startswith(pref):
                    return True
            elif kw.startswith("*"):
                suf = kw[1:].lower()
                if low.endswith(suf):
                    return True
            else:
                if kw.lower() in low:
                    return True
        return False

    @staticmethod
    def _parse_editable_line(line: str) -> tuple[str, None]:
        """Parse editable (-e/--editable) dependency lines."""
        m = re.search(r"egg=([A-Za-z0-9_\-.]+)", line)
        if m:
            return (m.group(1), None)
        seg = line.split("#egg=")[-1]
        return (seg, None)

    @staticmethod
    def _is_url_line(line: str) -> bool:
        """Check if line is a URL-based dependency."""
        return ("git+" in line) or ("http" in line) or ("ssh+" in line) or ("git://" in line) or line.endswith(".git")

    @staticmethod
    def _parse_url_line(line: str) -> tuple[str, None]:
        """Parse URL-based dependency lines."""
        m = re.search(r"[#&]egg=([A-Za-z0-9_\-.]+)", line)
        if m:
            return (m.group(1), None)
        if EGG_URL_SEPARATOR in line:
            seg = line.split(EGG_URL_SEPARATOR)[-1]
            return (seg, None)
        seg = line.rstrip("/\\").split("/")[-1]
        if seg.endswith(".git"):
            seg = seg[:-4]
        return (seg, None)

    @staticmethod
    def _parse_version_line(line: str) -> tuple[str, str | None] | None:
        """Parse standard package[extra]==version style lines."""
        m = re.match(r"^([A-Za-z0-9_\-.]+)(?:\[[^\]]*\])?\s*([<>=!~].+)?$", line)
        if m:
            name = m.group(1)
            ver = m.group(2).strip() if m.group(2) else None
            return (name, ver)
        return None

    @staticmethod
    def parse_requirements(text: str) -> list[tuple[str, str | None]]:
        """
        Parse a requirements-style dependency specification into names and versions.

        Args:
            text: Raw contents of a requirements-format dependency file.

        Returns:
            List of (package_name, version_specifier or None) tuples.
        """
        out: list[tuple[str, str | None]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-e ", "--editable ")):
                out.append(DependencyParser._parse_editable_line(line))
                continue
            if DependencyParser._is_url_line(line):
                out.append(DependencyParser._parse_url_line(line))
                continue
            result = DependencyParser._parse_version_line(line)
            if result:
                out.append(result)
                continue
            parts = line.split()
            if parts:
                out.append((parts[0], None))
        return out

    @staticmethod
    def _load_toml_library() -> object | None:
        """Attempt to load a TOML parsing library.

        Returns:
            TOML library module (tomllib or toml) if available, None otherwise.
        """
        try:
            import tomllib as _std_toml

            return _std_toml
        except ImportError:
            pass
        try:
            import toml as _thirdparty_toml

            return _thirdparty_toml
        except ImportError:
            pass
        return None

    def parse_pyproject_toml(self, text: str) -> list[tuple[str, str | None]]:
        """
        Parse dependencies defined in a pyproject.toml file.

        Args:
            text: Raw contents of a pyproject.toml file.

        Returns:
            List of (package_name, version_specifier or None) tuples.
        """
        toml_loader = self._load_toml_library()

        if toml_loader is not None:
            parsed = self._parse_toml_with_library(text, toml_loader)
            if parsed is not None:
                return parsed
            logger.debug("TOML library parsing failed, falling back to regex")

        return self._parse_toml_fallback(text)

    @staticmethod
    def _extract_deps_from_data(data: dict[str, Any]) -> object:
        """
        Extract dependencies object from parsed TOML data.

        Args:
            data: Parsed TOML data.

        Returns:
            Dependencies dict/list or None.
        """
        deps = None
        if "tool" in data and "poetry" in data["tool"]:
            deps = data["tool"]["poetry"].get("dependencies", {}) or {}
        if "project" in data:
            proj_deps = data["project"].get("dependencies", None)
            if proj_deps is not None:
                deps = proj_deps
        return deps

    @staticmethod
    def _process_dict_deps(deps: dict) -> list[tuple[str, str | None]]:
        """Process dependencies in dictionary format."""
        out: list[tuple[str, str | None]] = []
        for k, v in deps.items():
            ver = None
            if isinstance(v, str):
                ver = v
            elif isinstance(v, dict):
                ver = v.get("version")
            out.append((k, ver))
        return out

    @staticmethod
    def _is_dependency_name_char(char: str) -> bool:
        """Check whether a character is valid in a dependency name.

        Only allows ASCII characters: letters, digits, underscore, hyphen, period.

        Args:
            char: Character to check.

        Returns:
            True if the character is valid in a dependency name.
        """
        return ("A" <= char <= "Z") or ("a" <= char <= "z") or ("0" <= char <= "9") or char in "_-."

    @staticmethod
    def _parse_dependency_item(item: str) -> tuple[str, str | None] | None:
        """Parse a dependency item into package name and version without regex backtracking.

        Manually extracts package name by scanning for valid ASCII characters,
        then captures the remainder as version specifier. Supports extras in brackets.

        Args:
            item: Dependency item text (e.g., 'requests>=2.0' or 'fastapi[standard]').

        Returns:
            Parsed (package_name, version_specifier or None) tuple, or None if no valid name.
        """
        s = item.strip()
        name_end = 0

        while name_end < len(s) and DependencyParser._is_dependency_name_char(s[name_end]):
            name_end += 1

        if name_end == 0:
            return None

        name = s[:name_end]
        rest = s[name_end:]

        # Skip past [extras] if present
        if rest.startswith("["):
            bracket_end = rest.find("]")
            if bracket_end != -1:
                rest = rest[bracket_end + 1 :]

        ver = rest.strip() or None
        return (name, ver)

    @staticmethod
    def _process_list_deps(deps: list) -> list[tuple[str, str | None]]:
        """Process dependencies in list format."""
        out: list[tuple[str, str | None]] = []
        for item in deps:
            if not isinstance(item, str):
                continue
            result = DependencyParser._parse_dependency_item(item)
            if result:
                out.append(result)
        return out

    @staticmethod
    def _parse_toml_with_library(text: str, toml_loader: object) -> list[tuple[str, str | None]] | None:
        """
        Parse dependencies from TOML using an available TOML library.

        Args:
            text: TOML file contents.
            toml_loader: TOML library module.

        Returns:
            List of (package_name, version_specifier or None) tuples, or None when parsing fails.
        """
        try:
            loads_fn = getattr(toml_loader, "loads", None)
            if not callable(loads_fn):
                return None
            data = loads_fn(text)
        except Exception as exc:
            logger.debug("TOML library parsing error: %s", exc)
            return None

        deps = DependencyParser._extract_deps_from_data(data)

        if isinstance(deps, dict):
            return DependencyParser._process_dict_deps(deps)
        if isinstance(deps, list):
            return DependencyParser._process_list_deps(deps)
        return []

    @staticmethod
    def _parse_toml_fallback(text: str) -> list[tuple[str, str | None]]:
        """
        Parse TOML dependencies using a regex fallback when no TOML library is available.

        Args:
            text: TOML file contents.

        Returns:
            List of (package_name, version_specifier or None) tuples.
        """
        out: list[tuple[str, str | None]] = []
        list_items: list[str] = []
        current_section = None
        collecting_list = False

        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            if s.startswith("[") and s.endswith("]"):
                current_section = s.strip()
                continue

            DependencyParser._handle_toml_line(s, current_section, collecting_list, list_items, out)
            if current_section == PROJECT_DEF and s.startswith("dependencies") and s.endswith("["):
                collecting_list = True
            elif collecting_list and s.endswith("]"):
                collecting_list = False

        DependencyParser._process_toml_list_items(list_items, out)
        return out

    @staticmethod
    def _handle_toml_line(
        s: str,
        current_section: str | None,
        collecting_list: bool,
        list_items: list[str],
        out: list[tuple[str, str | None]],
    ) -> None:
        """Handle a single line of TOML content."""
        if current_section == PROJECT_DEF:
            if s.startswith("dependencies") and "[" in s and "]" in s:
                items = re.findall(DEPENDENCY_STRING_PATTERN, s)
                list_items.extend(items)
                return

            if collecting_list:
                items = re.findall(DEPENDENCY_STRING_PATTERN, s)
                list_items.extend(items)
                return

        if current_section and current_section.startswith("[tool.poetry") and "dependencies" in current_section:
            DependencyParser._parse_poetry_dependency(s, out)

    @staticmethod
    def _parse_poetry_dependency(s: str, out: list[tuple[str, str | None]]) -> None:
        """Parse a single poetry dependency line."""
        if "=" not in s:
            return
        k, v = s.split("=", 1)
        k = k.strip().strip('"').strip("'")
        v = v.strip()
        if not k:
            return
        m = re.search(r"[\"']([^\"']+)[\"']", v)
        ver = m.group(1) if m else None
        out.append((k, ver))

    @staticmethod
    def _process_toml_list_items(list_items: list[str], out: list[tuple[str, str | None]]) -> None:
        """Process collected list items from TOML."""
        for item in list_items:
            result = DependencyParser._parse_dependency_item(item)
            if result:
                out.append(result)

    @staticmethod
    def parse_package_json(text: str) -> list[tuple[str, str | None]]:
        """
        Parse dependency sections from a package.json file.

        Args:
            text: Raw contents of a package.json file.

        Returns:
            List of (package_name, version) tuples from all dependency sections.
        """
        out: list[tuple[str, str | None]] = []
        try:
            data = json.loads(text)
        except Exception as exc:
            logger.debug("Failed to parse package.json: %s", exc)
            return out
        for key in (
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
        ):
            deps = data.get(key, {}) or {}
            for k, v in deps.items():
                out.append((k, str(v)))
        return out

    def parse_generic(self, filename: str, text: str) -> list[tuple[str, str | None]]:
        """
        Route dependency parsing based on filename heuristics.

        Args:
            filename: Name of the dependency file.
            text: File contents to parse.

        Returns:
            List of (package_name, version_specifier or None) tuples.
        """
        lower = filename.lower()
        try:
            if lower.endswith("requirements.txt") or lower.endswith("pipfile"):
                return self.parse_requirements(text)
            if lower.endswith("pyproject.toml"):
                return self.parse_pyproject_toml(text)
            if lower.endswith("package.json"):
                return self.parse_package_json(text)
            return self.parse_requirements(text)
        except Exception as exc:
            raise DependencyParseError(f"Failed to parse dependency file: {filename}") from exc

    def extract_ai_dependencies(self, filename: str, text: str) -> tuple[list[tuple[str, str | None]], str | None]:
        """Parse a dependency file and extract AI-related dependencies.

        Automatically selects the appropriate parser based on filename.
        Filters parsed dependencies to return only AI-related packages.

        Args:
            filename: Name of the dependency file (e.g., requirements.txt).
            text: File contents to parse.

        Returns:
            Tuple of (ai_dependencies, error_message).
            ai_dependencies is a list of (package_name, version) tuples.
            error_message is None on success or an error string on parse failure.
        """
        try:
            parsed = self.parse_generic(filename, text)
        except DependencyParseError as exc:
            logger.warning("Failed to parse %s", filename, exc_info=exc)
            return ([], str(exc))
        ais: list[tuple[str, str | None]] = []
        for name, ver in parsed:
            if self.is_ai_dependency(name):
                ais.append((name, ver))
        return (ais, None)
