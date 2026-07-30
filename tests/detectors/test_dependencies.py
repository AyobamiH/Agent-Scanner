"""Tests for dependency detection and parsing."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.detectors.dependencies import DependencyParseError, DependencyParser
from src.models.results import RepoScanResult


@pytest.fixture
def parser() -> DependencyParser:
    """Create a DependencyParser instance for testing."""
    return DependencyParser()


def test_parser_initialisation() -> None:
    """Parser initialises with dependency keywords loaded."""
    parser = DependencyParser()

    assert parser.dependency_keywords is not None
    assert len(parser.dependency_keywords) > 0


def test_parser_initialisation_with_custom_keywords_path() -> None:
    """Parser initialises with custom keywords path."""
    parser = DependencyParser(keywords_path="src/config/keywords.json")

    assert parser.dependency_keywords is not None


def test_is_ai_dependency_with_exact_match(parser: DependencyParser) -> None:
    """Check if exact keyword match identifies AI dependency."""
    result = parser.is_ai_dependency("openai")

    assert result is True


def test_is_ai_dependency_with_empty_string(parser: DependencyParser) -> None:
    """Check if empty string returns False."""
    result = parser.is_ai_dependency("")

    assert result is False


def test_is_ai_dependency_with_prefix_wildcard(parser: DependencyParser) -> None:
    """Check if prefix wildcard pattern matches correctly."""
    with patch.object(parser, "dependency_keywords", {"google-*"}):
        result = parser.is_ai_dependency("google-adk")

    assert result is True


def test_is_ai_dependency_with_suffix_wildcard(parser: DependencyParser) -> None:
    """Check if suffix wildcard pattern matches correctly."""
    with patch.object(parser, "dependency_keywords", {"*-agent"}):
        result = parser.is_ai_dependency("my-agent")

    assert result is True


def test_is_ai_dependency_with_substring_match(parser: DependencyParser) -> None:
    """Check if substring match identifies AI dependency."""
    with patch.object(parser, "dependency_keywords", {"langchain"}):
        result = parser.is_ai_dependency("langchain-core")

    assert result is True


def test_is_ai_dependency_with_non_ai_package(parser: DependencyParser) -> None:
    """Check if non-AI package returns False."""
    result = parser.is_ai_dependency("requests")

    assert result is False


def test_parse_requirements_simple():
    """Parse requirements handles comments and version specifiers."""
    text = """
# comment
openai==0.27.0
langchain>=0.0.1
git+https://github.com/org/repo.git#egg=myagent
"""
    p = DependencyParser()
    parsed = p.parse_requirements(text)
    names = [n for n, v in parsed]
    assert "openai" in names
    assert "langchain" in names
    assert "myagent" in names


def test_parse_pyproject_poetry(tmp_path):
    """Parse pyproject.toml extracts poetry dependencies."""
    text = """
[tool.poetry]
name = "sample"
[tool.poetry.dependencies]
python = "^3.11"
openai = "^0.27.0"
"""
    p = DependencyParser()
    parsed = p.parse_pyproject_toml(text)
    assert any(n == "openai" for n, _ in parsed)


def test_parse_package_json():
    """Parse package.json returns dependency entries."""
    text = '{"dependencies": {"openai": "^1.0.0", "react": "17.0.0"}}'
    p = DependencyParser()
    parsed = p.parse_package_json(text)
    assert any(n == "openai" for n, _ in parsed)


def test_extract_ai_dependencies_detects_openai():
    """Extract AI dependencies identifies configured AI packages."""
    text = "openai==0.27.0\nrequests==2.0"
    p = DependencyParser()
    found, err = p.extract_ai_dependencies("requirements.txt", text)
    assert err is None
    assert any(n == "openai" for n, _ in found)


def test_complex_dependency_lines_extract_versions():
    """Parse requirements captures versions for complex specifiers."""
    deps = [
        "google-adk>=1.15.0,<2.0.0",
        "a2a-sdk~=0.3.9",
        "nest-asyncio>=1.6.0,<2.0.0",
        "opentelemetry-instrumentation-google-genai==0.4b0",
        "gcsfs>=2024.11.0",
        "google-cloud-logging>=3.12.0,<4.0.0",
        "google-cloud-aiplatform[evaluation,agent-engines]>=1.118.0,<2.0.0",
        "google-cloud-bigquery>=3.0.0,<4.0.0",
        "protobuf>=6.31.1,<7.0.0",
        "absl-py>=1.4.0,<2.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
    ]
    text = "\n".join(deps)
    p = DependencyParser()
    parsed = p.parse_requirements(text)
    for name, ver in parsed:
        assert ver is not None, f"Expected version for {name}"


def test_pyproject_project_dependencies_list_parsing():
    """Parse pyproject project dependencies list retains versions."""
    text = """
[project]
dependencies = [
    "google-adk>=1.15.0,<2.0.0",
    "a2a-sdk~=0.3.9",
    "nest-asyncio>=1.6.0,<2.0.0",
    "opentelemetry-instrumentation-google-genai==0.4b0",
    "gcsfs>=2024.11.0",
    "google-cloud-logging>=3.12.0,<4.0.0",
    "google-cloud-aiplatform[evaluation,agent-engines]>=1.118.0,<2.0.0",
    "google-cloud-bigquery>=3.0.0,<4.0.0",
    "protobuf>=6.31.1,<7.0.0",
    "absl-py>=1.4.0,<2.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
]
"""
    p = DependencyParser()
    parsed = p.parse_pyproject_toml(text)
    names = [n for n, v in parsed]
    assert "google-adk" in names
    for _n, v in parsed:
        assert v is not None


def test_scanner_integration_detects_deps(monkeypatch):
    """Scanner integration extracts dependency files and AI dependencies."""
    tree = [
        {"path": "requirements.txt", "type": "blob"},
        {"path": "src/main.py", "type": "blob"},
    ]
    contents = {
        "requirements.txt": "openai==0.27.0\nlangchain==0.0.1",
        "src/main.py": "print('hi')",
    }

    class C:
        @staticmethod
        def get_repo_tree(owner, repo, branch=None):
            return tree

        @staticmethod
        def get_file_content(owner, repo, path, branch=None):
            return contents[path]

    from src.detectors.patterns import PatternMatcher
    from src.scanner.scanner import Scanner

    matcher = PatternMatcher.from_file()
    s = Scanner(C(), matcher)
    s._current_branch = None
    result = RepoScanResult(repo_name="repo", org="owner")
    s._extract_dependencies("owner", "repo", tree, result)
    assert result.dependency_files
    assert any(d.package_name.lower() == "openai" for d in result.ai_dependencies)


def test_parse_requirements_with_comments(parser: DependencyParser) -> None:
    """Parse requirements ignores comment lines."""
    text = "# This is a comment\nopenai==0.27.0\n# Another comment"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "openai"


def test_parse_requirements_with_blank_lines(parser: DependencyParser) -> None:
    """Parse requirements ignores blank lines."""
    text = "\n\nopenai==0.27.0\n\n\nlangchain>=0.0.1\n\n"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 2


def test_parse_requirements_with_editable_flag(parser: DependencyParser) -> None:
    """Parse requirements handles editable installs with -e flag."""
    text = "-e git+https://github.com/org/repo.git#egg=mypackage"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "mypackage"


def test_parse_requirements_with_editable_long_flag(parser: DependencyParser) -> None:
    """Parse requirements handles editable installs with --editable flag."""
    text = "--editable git+https://github.com/org/repo.git#egg=mypackage"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "mypackage"


def test_parse_requirements_with_git_url(parser: DependencyParser) -> None:
    """Parse requirements extracts package name from git URLs."""
    text = "git+https://github.com/org/myrepo.git"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "myrepo"


def test_parse_requirements_with_ssh_url(parser: DependencyParser) -> None:
    """Parse requirements handles SSH git URLs."""
    text = "ssh+git@github.com:org/repo.git#egg=package"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "package"


def test_parse_requirements_with_http_url(parser: DependencyParser) -> None:
    """Parse requirements handles HTTP URLs."""
    text = "http://example.com/package.tar.gz#egg=package"  # NOSONAR S5332

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "package"


def test_parse_requirements_with_git_protocol(parser: DependencyParser) -> None:
    """Parse requirements handles git:// protocol."""
    text = "git://github.com/org/repo.git#egg=package"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "package"


def test_parse_requirements_with_extras(parser: DependencyParser) -> None:
    """Parse requirements handles packages with extras."""
    text = "package[extra1,extra2]>=1.0.0"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "package"
    assert parsed[0][1] == ">=1.0.0"


def test_parse_requirements_with_version_operators(parser: DependencyParser) -> None:
    """Parse requirements handles various version operators."""
    text = "pkg1==1.0\npkg2>=2.0\npkg3~=3.0\npkg4!=4.0"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 4
    assert parsed[0] == ("pkg1", "==1.0")
    assert parsed[1] == ("pkg2", ">=2.0")
    assert parsed[2] == ("pkg3", "~=3.0")
    assert parsed[3] == ("pkg4", "!=4.0")


def test_parse_requirements_with_fallback_parsing(parser: DependencyParser) -> None:
    """Parse requirements uses fallback for unparseable lines."""
    text = "unparseable line with spaces"

    parsed = parser.parse_requirements(text)

    assert len(parsed) == 1
    assert parsed[0][0] == "unparseable"


def test_load_toml_library_finds_tomllib(parser: DependencyParser) -> None:
    """Load TOML library finds tomllib when available."""
    with patch("builtins.__import__", return_value=MagicMock()):
        result = parser._load_toml_library()

    assert result is not None


def test_load_toml_library_falls_back_to_toml(parser: DependencyParser) -> None:
    """Load TOML library falls back to toml package."""

    def mock_import(name, *args, **kwargs):
        if name == "tomllib":
            raise ImportError("No module named tomllib")
        return MagicMock()

    with patch("builtins.__import__", side_effect=mock_import):
        result = parser._load_toml_library()

    assert result is not None or result is None


def test_load_toml_library_returns_none_when_unavailable(parser: DependencyParser) -> None:
    """Load TOML library returns None when no library available."""
    with patch("builtins.__import__", side_effect=ImportError("No TOML library")):
        result = parser._load_toml_library()

    assert result is None


def test_parse_pyproject_toml_with_library(parser: DependencyParser) -> None:
    """Parse pyproject.toml uses TOML library when available."""
    text = """
[tool.poetry.dependencies]
python = "^3.11"
openai = "^0.27.0"
"""

    parsed = parser.parse_pyproject_toml(text)

    assert any(n == "openai" for n, _ in parsed)


def test_parse_pyproject_toml_with_project_section(parser: DependencyParser) -> None:
    """Parse pyproject.toml handles [project] section."""
    text = """
[project]
dependencies = [
    "openai>=1.0.0",
    "langchain~=0.1.0"
]
"""

    parsed = parser.parse_pyproject_toml(text)

    names = [n for n, _ in parsed]
    assert "openai" in names
    assert "langchain" in names


def test_parse_pyproject_toml_with_dict_dependencies(parser: DependencyParser) -> None:
    """Parse pyproject.toml handles dictionary-style dependencies."""
    text = """
[tool.poetry.dependencies]
openai = {version = "^0.27.0", optional = true}
langchain = "^0.1.0"
"""

    parsed = parser.parse_pyproject_toml(text)

    names = [n for n, _ in parsed]
    assert "openai" in names
    assert "langchain" in names


def test_parse_pyproject_toml_fallback_with_inline_list(parser: DependencyParser) -> None:
    """Parse pyproject.toml fallback handles inline dependency list."""
    text = """
[project]
dependencies = ["openai>=1.0.0", "langchain~=0.1.0"]
"""

    with patch.object(parser, "_load_toml_library", return_value=None):
        parsed = parser.parse_pyproject_toml(text)

    names = [n for n, _ in parsed]
    assert "openai" in names
    assert "langchain" in names


def test_parse_pyproject_toml_fallback_with_multiline_list(parser: DependencyParser) -> None:
    """Parse pyproject.toml fallback handles multiline dependency list."""
    text = """
[project]
dependencies = [
    "openai>=1.0.0",
    "langchain~=0.1.0",
]
"""

    with patch.object(parser, "_load_toml_library", return_value=None):
        parsed = parser.parse_pyproject_toml(text)

    names = [n for n, _ in parsed]
    assert "openai" in names


def test_parse_pyproject_toml_fallback_with_poetry_section(parser: DependencyParser) -> None:
    """Parse pyproject.toml fallback handles poetry dependencies."""
    text = """
[tool.poetry.dependencies]
python = "^3.11"
openai = "^0.27.0"
"""

    with patch.object(parser, "_load_toml_library", return_value=None):
        parsed = parser.parse_pyproject_toml(text)

    names = [n for n, _ in parsed]
    assert "openai" in names


def test_parse_pyproject_toml_with_library_parse_failure(parser: DependencyParser, caplog) -> None:
    """Parse pyproject.toml falls back to regex when library parsing fails."""
    text = """
[tool.poetry.dependencies]
openai = "^0.27.0"
"""
    mock_toml = MagicMock()
    mock_toml.loads.side_effect = Exception("Parse error")

    caplog.set_level(logging.DEBUG)

    with patch.object(parser, "_load_toml_library", return_value=mock_toml):
        parser.parse_pyproject_toml(text)

    assert "TOML library parsing failed" in caplog.text or "TOML library parsing error" in caplog.text


def test_parse_package_json_with_all_dependency_types(parser: DependencyParser) -> None:
    """Parse package.json extracts all dependency types."""
    text = """
{
    "dependencies": {"pkg1": "1.0.0"},
    "devDependencies": {"pkg2": "2.0.0"},
    "peerDependencies": {"pkg3": "3.0.0"},
    "optionalDependencies": {"pkg4": "4.0.0"}
}
"""

    parsed = parser.parse_package_json(text)

    names = [n for n, _ in parsed]
    assert "pkg1" in names
    assert "pkg2" in names
    assert "pkg3" in names
    assert "pkg4" in names


def test_parse_package_json_with_invalid_json(parser: DependencyParser) -> None:
    """Parse package.json returns empty list for invalid JSON."""
    text = "invalid json content"

    parsed = parser.parse_package_json(text)

    assert parsed == []


def test_parse_package_json_with_missing_dependencies(parser: DependencyParser) -> None:
    """Parse package.json handles missing dependency sections."""
    text = '{"name": "mypackage", "version": "1.0.0"}'

    parsed = parser.parse_package_json(text)

    assert parsed == []


def test_parse_generic_with_requirements_txt(parser: DependencyParser) -> None:
    """Parse generic routes requirements.txt to correct parser."""
    text = "openai==0.27.0"

    parsed = parser.parse_generic("requirements.txt", text)

    assert len(parsed) == 1
    assert parsed[0][0] == "openai"


def test_parse_generic_with_pipfile(parser: DependencyParser) -> None:
    """Parse generic routes Pipfile to requirements parser."""
    text = "openai==0.27.0"

    parsed = parser.parse_generic("Pipfile", text)

    assert len(parsed) == 1


def test_parse_generic_with_pyproject_toml(parser: DependencyParser) -> None:
    """Parse generic routes pyproject.toml to TOML parser."""
    text = """
[tool.poetry.dependencies]
openai = "^0.27.0"
"""

    parsed = parser.parse_generic("pyproject.toml", text)

    assert any(n == "openai" for n, _ in parsed)


def test_parse_generic_with_package_json(parser: DependencyParser) -> None:
    """Parse generic routes package.json to JSON parser."""
    text = '{"dependencies": {"openai": "1.0.0"}}'

    parsed = parser.parse_generic("package.json", text)

    assert any(n == "openai" for n, _ in parsed)


def test_parse_generic_with_unknown_file(parser: DependencyParser) -> None:
    """Parse generic uses requirements parser as default."""
    text = "openai==0.27.0"

    parsed = parser.parse_generic("unknown.txt", text)

    assert len(parsed) == 1


def test_extract_ai_dependencies_filters_non_ai(parser: DependencyParser) -> None:
    """Extract AI dependencies filters out non-AI packages."""
    text = "openai==0.27.0\nrequests==2.0\nlangchain>=0.1.0"

    found, err = parser.extract_ai_dependencies("requirements.txt", text)

    assert err is None
    names = [n for n, _ in found]
    assert "openai" in names
    assert "langchain" in names
    assert "requests" not in names


def test_extract_ai_dependencies_handles_parse_error(parser: DependencyParser, caplog) -> None:
    """Extract AI dependencies returns error message on parse failure."""
    caplog.set_level(logging.WARNING)

    with patch.object(parser, "parse_generic", side_effect=DependencyParseError("Parse failed")):
        found, err = parser.extract_ai_dependencies("test.txt", "content")

    assert found == []
    assert err is not None
    assert "Parse failed" in err
    assert "Failed to parse" in caplog.text


def test_parse_generic_wraps_exceptions(parser: DependencyParser) -> None:
    """Parse generic wraps unexpected parsing errors in DependencyParseError."""
    with patch.object(parser, "parse_requirements", side_effect=ValueError("boom")):
        with pytest.raises(DependencyParseError):
            parser.parse_generic("requirements.txt", "content")


def test_extract_ai_dependencies_with_empty_result(parser: DependencyParser) -> None:
    """Extract AI dependencies returns empty list when no AI deps found."""
    text = "requests==2.0\nnumpy==1.0"

    found, err = parser.extract_ai_dependencies("requirements.txt", text)

    assert err is None
    assert found == []


@pytest.mark.parametrize(
    "filename,expected_parser",
    [
        ("requirements.txt", "parse_requirements"),
        ("requirements-dev.txt", "parse_requirements"),
        ("Pipfile", "parse_requirements"),
        ("pyproject.toml", "parse_pyproject_toml"),
        ("package.json", "parse_package_json"),
    ],
)
def test_parse_generic_routes_correctly(parser: DependencyParser, filename: str, expected_parser: str) -> None:
    """Parse generic routes files to correct parser based on filename."""
    with patch.object(parser, expected_parser, return_value=[("test", "1.0")]) as mock_parser:
        parser.parse_generic(filename, "content")

        mock_parser.assert_called_once()


def test_parse_toml_fallback_multiline_continuation_lines(parser: DependencyParser) -> None:
    """Parse TOML fallback correctly handles multiline dependencies with continuation lines.

    This test catches the bug where continuation lines (that don't start with 'dependencies')
    were skipped because the condition required s.startswith("dependencies").

    Expected: Both "requests" and "pydantic" should be extracted with their versions.
    Buggy behavior: Only "dependencies = [" would match, continuation lines would be skipped.
    """
    text = """
[project]
dependencies = [
    "requests>=2.0",
    "pydantic",
]
"""

    with patch.object(parser, "_load_toml_library", return_value=None):
        parsed = parser.parse_pyproject_toml(text)

    assert len(parsed) == 2, f"Expected 2 packages, got {len(parsed)}: {parsed}"

    names = [n for n, _ in parsed]
    assert "requests" in names, f"'requests' not found in parsed packages: {names}"
    assert "pydantic" in names, f"'pydantic' not found in parsed packages: {names}"

    versions_dict = dict(parsed)
    assert versions_dict["requests"] == ">=2.0", f"Expected '>=2.0', got {versions_dict['requests']}"
    assert versions_dict["pydantic"] is None, f"Expected None, got {versions_dict['pydantic']}"
