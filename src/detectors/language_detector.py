"""Language detection for repositories.

Detects the primary programming language in a repository based on
file extensions and file counts.
"""

from __future__ import annotations

from typing import Any


class LanguageDetector:
    """Detect the main programming language in a repository."""

    LANGUAGE_EXTENSIONS = {
        "Python": [".py"],
        "JavaScript": [".js", ".jsx"],
        "TypeScript": [".ts", ".tsx"],
        "Java": [".java"],
        "Go": [".go"],
        "C#": [".cs"],
        "Ruby": [".rb"],
    }

    NON_CODE_FILES = {
        ".md",
        ".txt",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".lock",
        ".gitignore",
        ".env",
        ".dockerfile",
        "dockerfile",
        "license",
        "readme",
        "contributing",
        "changelog",
    }

    VENDOR_PATHS = {"node_modules", "vendor", ".git", "__pycache__", "dist", "build"}

    def detect_main_language(self, file_list: list[dict[str, Any]]) -> str | None:
        """Detect the main programming language in a file list.

        Args:
            file_list: List of file metadata dictionaries with 'path' and 'size' keys.

        Returns:
            The name of the main language, or None if no language detected.
        """
        if not file_list:
            return None

        language_counts: dict[str, int] = {}

        for file_info in file_list:
            path = file_info.get("path", "")

            if self._should_skip_file(path):
                continue

            language = self._get_language_from_path(path)
            if language:
                language_counts[language] = language_counts.get(language, 0) + 1

        if not language_counts:
            return None

        main_language = max(language_counts, key=lambda lang: language_counts[lang])
        return main_language

    def _should_skip_file(self, path: str) -> bool:
        """Check if file should be skipped in language detection.

        Args:
            path: File path to check.

        Returns:
            True if file should be skipped, False otherwise.
        """
        path_lower = path.lower()

        if any(vendor in path_lower for vendor in self.VENDOR_PATHS):
            return True

        filename = path.split("/")[-1]
        filename_lower = filename.lower()

        if any(filename_lower.endswith(ext) or filename_lower == ext for ext in self.NON_CODE_FILES):
            return True

        if filename.startswith(".") and filename != ".gitignore":
            return True

        return False

    def _get_language_from_path(self, path: str) -> str | None:
        """Extract language from file extension.

        Args:
            path: File path to analyse.

        Returns:
            Language name if recognised, None otherwise.
        """
        path_lower = path.lower()

        for language, extensions in self.LANGUAGE_EXTENSIONS.items():
            for ext in extensions:
                if path_lower.endswith(ext):
                    return language

        return None
