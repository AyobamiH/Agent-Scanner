"""Tests for agentic framework imports detection and exposure in scan output."""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

from src.detectors.agents.agents import AgentDetector
from src.models.results import RepoScanResult
from src.scanner.scanner import Scanner


class TestGetFrameworkImports:
    """_get_framework_imports should detect known agentic framework imports."""

    @staticmethod
    def _imports(source: str) -> set[str]:
        det = AgentDetector()
        tree = ast.parse(source)
        return det._get_framework_imports(tree)

    def test_google_adk_agents_from_import(self):
        src = "from google.adk.agents import Agent"
        result = self._imports(src)
        assert any("google.adk" in i for i in result), f"Expected a google.adk import in {result}"

    def test_google_adk_models_lite_llm(self):
        src = "from google.adk.models.lite_llm import LiteLlm"
        result = self._imports(src)
        assert any("google.adk" in i for i in result), f"Expected a google.adk import in {result}"

    def test_google_adk_tools_mcp_tool(self):
        src = "from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset"
        result = self._imports(src)
        assert any("google.adk" in i for i in result), f"Expected a google.adk import in {result}"

    def test_google_adk_cli_fast_api(self):
        src = "from google.adk.cli.fast_api import get_fast_api_app"
        result = self._imports(src)
        assert any("google.adk" in i for i in result), f"Expected a google.adk import in {result}"

    def test_multiple_google_adk_imports_all_detected(self):
        src = """
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.cli.fast_api import get_fast_api_app
"""
        result = self._imports(src)
        assert len(result) >= 1, "Expected at least one google.adk module detected, got none"
        for module in result:
            assert "google.adk" in module, f"Unexpected module detected: {module}"

    def test_langchain_import_detected(self):
        src = "from langchain.agents import AgentExecutor"
        result = self._imports(src)
        assert any("langchain" in i for i in result), f"Expected a langchain import in {result}"

    def test_non_agentic_import_not_detected(self):
        src = "import os\nimport json\nfrom pathlib import Path"
        result = self._imports(src)
        assert result == set(), f"Expected empty set, got {result}"


class TestRepoScanResultAgenticImportsField:
    """RepoScanResult must have an ``agentic_imports`` field."""

    @staticmethod
    def test_agentic_imports_field_exists():
        """RepoScanResult should have an agentic_imports attribute."""
        result = RepoScanResult(repo_name="repo", org="org")
        assert hasattr(result, "agentic_imports"), (
            "RepoScanResult is missing the 'agentic_imports' field. "
            "Add `agentic_imports: list[str] = field(default_factory=list)` to the dataclass."
        )

    @staticmethod
    def test_agentic_imports_defaults_to_empty_list():
        result = RepoScanResult(repo_name="repo", org="org")
        assert result.agentic_imports == [], "agentic_imports should default to an empty list"

    @staticmethod
    def test_agentic_imports_can_be_set():
        imports = ["google.adk.agents", "google.adk.models.lite_llm"]
        result = RepoScanResult(repo_name="repo", org="org", agentic_imports=imports)
        assert result.agentic_imports == imports


class TestRepoScanResultToDictAgenticImports:
    """to_dict() must include agentic_imports in the detected.agents section."""

    @staticmethod
    def test_to_dict_includes_agentic_imports_key():
        result = RepoScanResult(repo_name="repo", org="org")
        d = result.to_dict()
        agents_section = d["detected"]["agents"]
        assert "agentic_imports" in agents_section, (
            "to_dict() must include 'agentic_imports' in detected.agents. "
            "Add it to the agents dict in RepoScanResult.to_dict()."
        )

    @staticmethod
    def test_to_dict_agentic_imports_empty_by_default():
        result = RepoScanResult(repo_name="repo", org="org")
        d = result.to_dict()
        assert d["detected"]["agents"]["agentic_imports"] == []

    @staticmethod
    def test_to_dict_agentic_imports_populated():
        imports = ["google.adk.agents", "google.adk.models.lite_llm"]
        result = RepoScanResult(repo_name="repo", org="org", agentic_imports=imports)
        d = result.to_dict()
        assert d["detected"]["agents"]["agentic_imports"] == imports


class TestScannerStoresAgenticImports:
    """Scanner._extract_agents must store collected framework_imports on result."""

    @staticmethod
    def _make_scanner_with_google_adk_file() -> tuple[Scanner, RepoScanResult]:
        """Set up a Scanner whose file source returns a Google ADK Python file."""
        github_client = MagicMock()
        scanner = Scanner(github_client)

        adk_source = """
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)
"""
        scanner._file_source = MagicMock()
        scanner._file_source.get_file_content.return_value = adk_source
        scanner._is_ignored_path = MagicMock(return_value=False)

        result = RepoScanResult(repo_name="my-repo", org="my-org")
        tree = [{"type": "blob", "path": "agent.py"}]

        scanner._extract_agents("my-org", "my-repo", tree, result)
        return scanner, result

    @staticmethod
    def test_extract_agents_returns_non_empty_imports_for_adk_file():
        """_extract_agents should return a non-empty list of imports for ADK code."""
        github_client = MagicMock()
        scanner = Scanner(github_client)

        adk_source = """
from google.adk.agents import Agent

root_agent = Agent(name="root_agent", model="gemini-2.5-flash")
"""
        scanner._file_source = MagicMock()
        scanner._file_source.get_file_content.return_value = adk_source
        scanner._is_ignored_path = MagicMock(return_value=False)

        result = RepoScanResult(repo_name="my-repo", org="my-org")
        tree = [{"type": "blob", "path": "agent.py"}]

        framework_imports = scanner._extract_agents("my-org", "my-repo", tree, result)

        assert framework_imports, (
            "_extract_agents returned an empty imports list for Google ADK code. "
            "Check that 'google.adk' is in keywords.json framework_modules."
        )
        assert any(
            "google.adk" in imp for imp in framework_imports
        ), f"Expected google.adk import in {framework_imports}"

    def test_result_agentic_imports_populated_after_extract_agents(self):
        """After _extract_agents, result.agentic_imports should contain ADK imports.

        This test WILL FAIL until Scanner._extract_agents assigns the collected
        imports back to result.agentic_imports.
        """
        _scanner, result = self._make_scanner_with_google_adk_file()

        assert hasattr(result, "agentic_imports"), "RepoScanResult is missing 'agentic_imports' field"
        assert result.agentic_imports, (
            "result.agentic_imports is empty after scanning Google ADK code. "
            "Scanner._extract_agents must assign collected_imports to result.agentic_imports."
        )
        assert any(
            "google.adk" in imp for imp in result.agentic_imports
        ), f"Expected google.adk in result.agentic_imports, got: {result.agentic_imports}"

    def test_to_dict_agentic_imports_populated_after_full_scan_flow(self):
        """to_dict() agentic_imports must be populated after scanning ADK code.

        This is the end-to-end assertion: the imports must appear in the JSON output.
        """
        _scanner, result = self._make_scanner_with_google_adk_file()

        d = result.to_dict()
        agents_section = d.get("detected", {}).get("agents", {})

        assert "agentic_imports" in agents_section, "to_dict() detected.agents is missing 'agentic_imports' key"
        assert agents_section[
            "agentic_imports"
        ], "to_dict() detected.agents.agentic_imports is empty after scanning ADK code"


class TestKeywordsContainGoogleAdk:
    """google.adk must be present in keywords.json framework_modules."""

    @staticmethod
    def test_google_adk_in_framework_modules():
        """AgentDetector should recognise google.adk as a framework module."""
        det = AgentDetector()
        assert any("google.adk" in fw for fw in det._framework_modules), (
            "google.adk is not listed in framework_modules in keywords.json. "
            "Add 'google.adk' (or a prefix/sub-path) to the framework_modules list."
        )
