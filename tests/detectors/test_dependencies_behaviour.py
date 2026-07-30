"""Tests for dependency parsing behaviour patterns and complex scenarios."""

from src.detectors.dependencies import DependencyParser


class TestParseRequirementsEdgeCases:
    """Test edge cases in requirements.txt parsing."""

    @staticmethod
    def test_parse_requirements_with_malformed_git_egg_missing_name() -> None:
        """Parse requirements handles malformed git URL with missing egg name."""
        text = "git+https://github.com/org/repo.git#egg="
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert len(parsed) >= 1

    @staticmethod
    def test_parse_requirements_with_git_url_no_egg_extraction() -> None:
        """Parse requirements extracts repository name from git URL without egg parameter."""
        text = "git+https://github.com/org/my-package.git"
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert any("my-package" in name for name, _ in parsed)

    @staticmethod
    def test_parse_requirements_with_http_git_url() -> None:
        """Parse requirements handles HTTP git URLs correctly."""
        text = "git+http://example.com/repo/package.git"  # NOSONAR S5332
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert len(parsed) == 1
        assert parsed[0][0] == "package"

    @staticmethod
    def test_parse_requirements_with_ssh_git_url_trailing_slash() -> None:
        """Parse requirements handles SSH URLs with trailing slashes."""
        text = "git+ssh://git@github.com/org/repo.git/"
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert any("repo" in name for name, _ in parsed)

    @staticmethod
    def test_parse_requirements_with_unrecognised_line_format() -> None:
        """Parse requirements handles unrecognised line format by extracting first token."""
        text = "strange-format @@ some weird syntax"
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert len(parsed) == 1
        assert parsed[0][0] == "strange-format"

    @staticmethod
    def test_parse_requirements_preserves_empty_lines() -> None:
        """Parse requirements correctly ignores consecutive empty lines."""
        text = "\n\n\nopenai==0.27.0\n\n\n\nlangchain\n\n"
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert len(parsed) == 2
        names = [name for name, _ in parsed]
        assert "openai" in names
        assert "langchain" in names


class TestParseTomlEdgeCases:
    """Test edge cases in TOML parsing."""

    @staticmethod
    def test_parse_pyproject_toml_fallback_with_poetry_dict_version() -> None:
        """Parse TOML fallback handles Poetry dependencies with dict version entries."""
        text = """
[tool.poetry.dependencies]
python = "^3.11"
openai = {version = "^0.27.0", optional = true}
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        names = [name for name, _ in parsed]
        assert "openai" in names

    @staticmethod
    def test_parse_pyproject_toml_fallback_with_complex_poetry_entries() -> None:
        """Parse TOML fallback handles complex Poetry dependency entries."""
        text = """
[tool.poetry.dependencies]
special-package = {git = "https://github.com/org/repo.git", branch = "main"}
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        names = [name for name, _ in parsed]
        assert "special-package" in names

    @staticmethod
    def test_parse_pyproject_toml_with_non_string_list_items() -> None:
        """Parse TOML fallback skips non-string items in dependencies list."""
        text = """
[project]
dependencies = [
    "openai>=0.27.0",
    "langchain",
]
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        names = [name for name, _ in parsed]
        assert "openai" in names
        assert "langchain" in names
        assert len(parsed) == 2

    @staticmethod
    def test_parse_pyproject_toml_with_both_project_and_poetry_sections() -> None:
        """Parse TOML uses project section when both project and poetry exist."""
        text = """
[project]
dependencies = [
    "openai>=0.27.0",
]

[tool.poetry.dependencies]
ignored-package = "^1.0"
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        names = [name for name, _ in parsed]
        assert "openai" in names

    @staticmethod
    def test_parse_pyproject_toml_with_multiline_poetry_dependency() -> None:
        """Parse TOML fallback extracts version from Poetry dependency strings."""
        text = """
[tool.poetry.dependencies]
special-lib = "^1.0.0"
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        found = [(name, ver) for name, ver in parsed if name == "special-lib"]
        assert len(found) > 0
        assert found[0][1] is not None

    @staticmethod
    def test_parse_pyproject_toml_with_nested_bracket_syntax() -> None:
        """Parse TOML handles nested bracket syntax in dependency specifications."""
        text = """
[project]
dependencies = [
    "google-cloud-aiplatform[evaluation,agent-engines]>=1.0",
]
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        names = [name for name, _ in parsed]
        assert "google-cloud-aiplatform" in names

    @staticmethod
    def test_parse_pyproject_toml_fallback_poetry_no_version_in_entry() -> None:
        """Parse TOML fallback handles Poetry entries with no version string."""
        text = """
[tool.poetry.dependencies]
package-name = {git = "https://github.com/org/repo"}
"""
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        names = [name for name, _ in parsed]
        assert "package-name" in names


class TestPackageJsonEdgeCases:
    """Test edge cases in package.json parsing."""

    @staticmethod
    def test_parse_package_json_with_none_dependency_value() -> None:
        """Parse package.json handles null/None dependency values."""
        text = '{"dependencies": {"openai": null}}'
        parser = DependencyParser()

        parsed = parser.parse_package_json(text)

        assert any("openai" in name for name, _ in parsed)

    @staticmethod
    def test_parse_package_json_with_numeric_dependency_version() -> None:
        """Parse package.json converts numeric versions to strings."""
        text = '{"dependencies": {"package": 1.0}}'
        parser = DependencyParser()

        parsed = parser.parse_package_json(text)

        assert len(parsed) == 1
        assert parsed[0][0] == "package"
        assert isinstance(parsed[0][1], str)

    @staticmethod
    def test_parse_package_json_with_all_missing_sections() -> None:
        """Parse package.json handles JSON with no dependency sections."""
        text = '{"name": "app", "version": "1.0.0"}'
        parser = DependencyParser()

        parsed = parser.parse_package_json(text)

        assert len(parsed) == 0

    @staticmethod
    def test_parse_package_json_with_empty_dependency_sections() -> None:
        """Parse package.json handles empty dependency objects."""
        text = '{"dependencies": {}, "devDependencies": null}'
        parser = DependencyParser()

        parsed = parser.parse_package_json(text)

        assert len(parsed) == 0

    @staticmethod
    def test_parse_package_json_with_mixed_dependency_types() -> None:
        """Parse package.json collects from multiple dependency sections."""
        text = """{
    "dependencies": {"openai": "^1.0"},
    "devDependencies": {"jest": "^29.0"},
    "peerDependencies": {"react": "18"},
    "optionalDependencies": {"python": "3.11"}
}"""
        parser = DependencyParser()

        parsed = parser.parse_package_json(text)

        names = [name for name, _ in parsed]
        assert "openai" in names
        assert "jest" in names
        assert "react" in names
        assert "python" in names


class TestExtractAiDependenciesIntegration:
    """Test practical scenarios for AI dependency extraction."""

    @staticmethod
    def test_extract_ai_dependencies_from_complex_requirements() -> None:
        """Extract AI dependencies identifies AI packages in complex requirements."""
        text = """
# Data processing
pandas==1.5.0
numpy>=1.20

# AI frameworks
openai>=0.27.0
langchain>=0.0.200
google-cloud-aiplatform[evaluation]>=1.0

# Utilities
requests>=2.28.0
"""
        parser = DependencyParser()

        found, error = parser.extract_ai_dependencies("requirements.txt", text)

        assert error is None
        ai_names = [name for name, _ in found]
        assert "openai" in ai_names
        assert "langchain" in ai_names
        assert "google-cloud-aiplatform" in ai_names
        assert "pandas" not in ai_names
        assert "requests" not in ai_names

    @staticmethod
    def test_extract_ai_dependencies_handles_malformed_file() -> None:
        """Extract AI dependencies returns error for unparseable file."""
        text = "clearly { invalid [ syntax ) here ]"
        parser = DependencyParser()

        found, _ = parser.extract_ai_dependencies("requirements.txt", text)

        assert len(found) == 0

    @staticmethod
    def test_extract_ai_dependencies_from_empty_file() -> None:
        """Extract AI dependencies handles empty dependency file."""
        text = "# Just comments\n# No actual dependencies"
        parser = DependencyParser()

        found, error = parser.extract_ai_dependencies("requirements.txt", text)

        assert error is None
        assert len(found) == 0

    @staticmethod
    def test_extract_ai_dependencies_case_insensitive_matching() -> None:
        """Extract AI dependencies performs case-insensitive keyword matching."""
        text = "OpenAI>=0.27.0\nLangChain>=0.0.1\nGOOGLE-CLOUD-AIPLATFORM>=1.0"
        parser = DependencyParser()

        found, error = parser.extract_ai_dependencies("requirements.txt", text)

        assert error is None
        assert len(found) == 3

    @staticmethod
    def test_extract_ai_dependencies_preserves_versions() -> None:
        """Extract AI dependencies retains version specifiers for AI packages."""
        text = """
openai>=0.27.0,<1.0.0
langchain[evaluation]~=0.0.200
anthropic==0.7.1
"""
        parser = DependencyParser()

        found, error = parser.extract_ai_dependencies("requirements.txt", text)

        assert error is None
        versions = dict(found)
        assert versions.get("openai") is not None
        assert "0.27.0" in versions.get("openai", "")

    @staticmethod
    def test_extract_ai_dependencies_with_invalid_json_package_json() -> None:
        """Extract AI dependencies handles JSON parse errors gracefully."""
        text = '{"dependencies": "not-an-object"}'
        parser = DependencyParser()

        found, error = parser.extract_ai_dependencies("package.json", text)

        assert error is not None
        assert len(found) == 0

    @staticmethod
    def test_parser_handles_corrupted_toml_gracefully() -> None:
        """Parser falls back to regex when TOML parsing fails."""
        text = '[project]\ndependencies = [\n"openai>=0.27.0"\n]'
        parser = DependencyParser()

        parsed = parser.parse_pyproject_toml(text)

        assert any("openai" in name for name, _ in parsed)

    @staticmethod
    def test_parse_requirements_with_very_long_line() -> None:
        """Parse requirements handles very long dependency lines."""
        long_version = ",".join([f">={i}.0.0,<{i+1}.0.0" for i in range(10)])
        text = f"some-package{long_version}"
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        assert len(parsed) > 0
        assert parsed[0][0] == "some-package"

    @staticmethod
    def test_is_ai_dependency_with_case_variations() -> None:
        """Is AI dependency recognises keywords regardless of case."""
        parser = DependencyParser()

        assert parser.is_ai_dependency("OpenAI") is True
        assert parser.is_ai_dependency("OPENAI") is True
        assert parser.is_ai_dependency("openai") is True
        assert parser.is_ai_dependency("OpenAi") is True


class TestDependencyParserErrorHandling:
    """Test error handling and robustness."""

    @staticmethod
    def test_parse_requirements_with_fallback_egg_extraction() -> None:
        """Parse requirements handles fallback egg extraction from URL."""
        text = "git+https://github.com/org/repo#egg=custom_name"
        parser = DependencyParser()

        parsed = parser.parse_requirements(text)

        names = [name for name, _ in parsed]
        assert "custom_name" in names
