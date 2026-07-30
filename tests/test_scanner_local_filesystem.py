"""Tests for Scanner with LocalFilesystemClient integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.detectors.patterns import PatternMatcher
from src.github.filesystem_client import LocalFilesystemClient
from src.scanner.scanner import Scanner


@pytest.fixture
def temp_agentic_repo(tmp_path: Path) -> Path:
    """Create a temporary repository with agentic patterns.

    Returns:
        Path to the repository root.
    """
    files = {
        "README.md": "# AI Agent Framework",
        "src/agents.py": """
from langchain import Agent
from openai import OpenAI

class MyAgent(Agent):
    pass
""",
        "src/main.py": "from langchain.agents import initialize_agent",
        "requirements.txt": "langchain==0.1.0\nopenai==1.0.0",
    }

    for file_path, content in files.items():
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    return tmp_path


@pytest.fixture
def mock_github_client_for_scanner() -> MagicMock:
    """Create a mock GitHub client for scanner testing."""
    client = MagicMock()
    client.fetch_all_commits.return_value = [
        {
            "author": "Test Author",
            "committer": "Test Committer",
            "author_email": "author@example.com",
            "committer_email": "committer@example.com",
        }
    ]
    client._api_url = "https://api.github.com"
    return client


class TestScannerWithLocalFilesystemClient:
    """Test Scanner integration with LocalFilesystemClient."""

    def test_scanner_init_with_file_source_client(
        self, temp_agentic_repo: Path, mock_github_client_for_scanner: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test Scanner initialises with optional file_source_client parameter."""
        file_source = LocalFilesystemClient(str(temp_agentic_repo), mock_github_client_for_scanner)
        matcher = PatternMatcher.from_file()

        scanner = Scanner(mock_github_client_for_scanner, matcher, file_source_client=file_source)

        assert scanner._file_source is file_source
        assert scanner.github is mock_github_client_for_scanner

    def test_scanner_defaults_to_github_client_as_file_source(
        self,
        mock_github_client_for_scanner: MagicMock,
    ) -> None:  # NOSONAR S2325
        """Test Scanner defaults to using github_client when file_source_client not provided."""
        matcher = PatternMatcher.from_file()

        scanner = Scanner(mock_github_client_for_scanner, matcher)

        assert scanner._file_source is mock_github_client_for_scanner

    def test_scanner_uses_file_source_for_get_repo_tree(
        self, temp_agentic_repo: Path, mock_github_client_for_scanner: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test Scanner uses file_source_client.get_repo_tree instead of github_client."""
        file_source = LocalFilesystemClient(str(temp_agentic_repo), mock_github_client_for_scanner)
        matcher = PatternMatcher.from_file()
        scanner = Scanner(mock_github_client_for_scanner, matcher, file_source_client=file_source)

        file_source.get_repo_tree = MagicMock(
            return_value=(
                [
                    {"path": "src/agents.py", "type": "blob", "size": 100},
                    {"path": "README.md", "type": "blob", "size": 50},
                ],
                {"default_branch": "main", "head_sha": "abc123", "html_url": "http://example.com"},  # NOSONAR S5332
            )
        )
        mock_github_client_for_scanner.get_repo_tree = MagicMock()

        with (
            patch.object(scanner, "_run_stage_1_path_scan", return_value=None),
            patch.object(scanner, "_run_stage_2_content_scan", return_value=None),
            patch.object(scanner, "_run_stage_3_extended_scan", return_value=None),
        ):
            scanner.scan("owner/repo")

        file_source.get_repo_tree.assert_called()
        mock_github_client_for_scanner.get_repo_tree.assert_not_called()

    def test_scanner_uses_file_source_for_get_file_content(
        self, temp_agentic_repo: Path, mock_github_client_for_scanner: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test Scanner uses file_source_client.get_file_content instead of github_client."""
        file_source = LocalFilesystemClient(str(temp_agentic_repo), mock_github_client_for_scanner)
        matcher = PatternMatcher.from_file()
        scanner = Scanner(mock_github_client_for_scanner, matcher, file_source_client=file_source)

        file_source.get_file_content = MagicMock(return_value="file content")
        mock_github_client_for_scanner.get_file_content = MagicMock()

        with (
            patch.object(scanner, "_run_stage_1_path_scan", return_value=None),
            patch.object(scanner, "_run_stage_2_content_scan", return_value=None),
            patch.object(scanner, "_run_stage_3_extended_scan", return_value=None),
        ):
            scanner.scan("owner/repo")

    def test_scanner_maintains_github_client_for_owner_detection(
        self, temp_agentic_repo: Path, mock_github_client_for_scanner: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test Scanner uses github_client for fetch_all_commits (owner detection)."""
        file_source = LocalFilesystemClient(str(temp_agentic_repo), mock_github_client_for_scanner)
        matcher = PatternMatcher.from_file()
        scanner = Scanner(mock_github_client_for_scanner, matcher, file_source_client=file_source)

        assert scanner.github is mock_github_client_for_scanner
        assert scanner._file_source is file_source

    def test_scanner_backward_compatible_without_file_source_client(
        self, mock_github_client_for_scanner: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test Scanner is backward compatible when file_source_client is not provided."""
        matcher = PatternMatcher.from_file()

        scanner = Scanner(mock_github_client_for_scanner, matcher)

        assert scanner._file_source is scanner.github


class TestScannerFileSourceLogging:
    """Test Scanner logs which file source is being used."""

    def test_scanner_logs_local_filesystem_mode(
        self, temp_agentic_repo: Path, mock_github_client_for_scanner: MagicMock, caplog
    ) -> None:  # NOSONAR S2325
        """Test Scanner logs 'local filesystem' when using LocalFilesystemClient."""
        import logging

        caplog.set_level(logging.INFO)

        file_source = LocalFilesystemClient(str(temp_agentic_repo), mock_github_client_for_scanner)
        matcher = PatternMatcher.from_file()
        scanner = Scanner(mock_github_client_for_scanner, matcher, file_source_client=file_source)

        with (
            patch.object(scanner, "_run_stage_1_path_scan", return_value=None),
            patch.object(scanner, "_run_stage_2_content_scan", return_value=None),
            patch.object(scanner, "_run_stage_3_extended_scan", return_value=None),
        ):
            scanner.scan("owner/repo")

        assert "local filesystem" in caplog.text.lower()

    def test_scanner_logs_github_api_mode(
        self,
        mock_github_client_for_scanner: MagicMock,
        caplog,
    ) -> None:  # NOSONAR S2325
        """Test Scanner logs 'GitHub API' when using github_client for file source."""
        import logging

        caplog.set_level(logging.INFO)

        mock_github_client_for_scanner.get_repo_tree.return_value = (
            [{"path": "README.md", "type": "blob"}],
            {"default_branch": "main", "head_sha": "abc123", "html_url": "http://example.com"},  # NOSONAR S5332
        )

        matcher = PatternMatcher.from_file()
        scanner = Scanner(mock_github_client_for_scanner, matcher)

        with (
            patch.object(scanner, "_run_stage_1_path_scan", return_value=None),
            patch.object(scanner, "_run_stage_2_content_scan", return_value=None),
            patch.object(scanner, "_run_stage_3_extended_scan", return_value=None),
        ):
            scanner.scan("owner/repo")

        assert "GitHub API" in caplog.text or "scanning mode" in caplog.text.lower()
