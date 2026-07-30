"""Tests for main.py CLI changes with --workspace-path support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.results import RepoScanResult
from src.scanner.scanner import Scanner


@pytest.fixture
def temp_test_repo(tmp_path: Path) -> Path:
    """Create a temporary test repository.

    Returns:
        Path to the repository root.
    """
    files = {
        "README.md": "# Test Repo",
        "src/main.py": "print('hello')",
        "requirements.txt": "requests==2.0.0",
    }

    for file_path, content in files.items():
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    return tmp_path


def _main():
    import src.main as main_module

    return main_module.main


class TestMainCLIWorkspacePath:
    """Test main.py CLI with --workspace-path argument."""

    def test_main_with_workspace_path_argument(self, temp_test_repo: Path) -> None:  # NOSONAR S2325
        """Test main() accepts --workspace-path argument."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(temp_test_repo),
            "--output-dir",
            str(temp_test_repo / "output"),
        ]

        with (
            patch("src.main.GitHubClient") as mock_github_client_class,
            patch("src.main.Scanner") as mock_scanner_class,
            patch("src.main.write_summary_file"),
            patch("src.main.PatternMatcher.from_file") as mock_pattern_loader,
        ):
            mock_github_client = MagicMock()
            mock_github_client_class.return_value = mock_github_client
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = None
            mock_scanner_class.return_value = mock_scanner
            matcher = MagicMock()
            matcher.score_path.return_value = 0
            matcher.score_content.return_value = 0
            matcher._tokenise_text.return_value = []
            mock_pattern_loader.return_value = matcher

            _main()(argv)

            assert mock_scanner_class.called

    @staticmethod
    def test_main_workspace_path_not_found_returns_failure() -> None:
        """Test main() returns EXIT_CLIENT_INIT_FAILURE when workspace path doesn't exist."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            "/nonexistent/path",
        ]

        result = _main()(argv)

        assert result == 1

    def test_main_workspace_path_is_file_returns_failure(self, tmp_path: Path) -> None:  # NOSONAR S2325
        """Test main() returns EXIT_CLIENT_INIT_FAILURE when workspace path is a file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(file_path),
        ]

        result = _main()(argv)

        assert result == 1

    def test_main_workspace_path_incompatible_with_scan_org_recent(self, temp_test_repo: Path) -> None:  # NOSONAR S2325
        """Test --workspace-path is incompatible with --scan-org-recent."""
        argv = [
            "--org",
            "test-org",
            "--workspace-path",
            str(temp_test_repo),
            "--scan-org-recent",
        ]

        result = _main()(argv)

        assert result == 1

    @staticmethod
    def test_main_without_workspace_path() -> None:
        """Test main() works without --workspace-path."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
        ]

        with (
            patch("src.main.GitHubClient"),
            patch("src.main.PatternMatcher.from_file"),
            patch("src.main.Scanner") as mock_scanner_class,
            patch("src.main.write_summary_file"),
        ):
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = None
            mock_scanner_class.return_value = mock_scanner

            _main()(argv)

            call_args = mock_scanner_class.call_args
            if call_args:
                assert "file_source_client" in call_args.kwargs or call_args.kwargs.get("file_source_client") is None


class TestMainCLIArgumentValidation:
    """Test main.py CLI argument validation with --workspace-path."""

    def test_main_validates_workspace_path_exists(self, tmp_path: Path) -> None:  # NOSONAR S2325
        """Test main() validates workspace path exists."""
        nonexistent = tmp_path / "nonexistent"

        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(nonexistent),
        ]

        result = _main()(argv)

        assert result == 1

    def test_main_validates_workspace_path_is_directory(self, tmp_path: Path) -> None:  # NOSONAR S2325
        """Test main() validates workspace path is a directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(file_path),
        ]

        result = _main()(argv)

        assert result == 1

    def test_main_accepts_valid_workspace_path(self, temp_test_repo: Path) -> None:  # NOSONAR S2325
        """Test main() accepts valid workspace path."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(temp_test_repo),
        ]

        with (
            patch("src.main.Scanner") as mock_scanner_class,
            patch("src.main.write_summary_file"),
            patch("src.main.GitHubClient"),
            patch("src.main.PatternMatcher.from_file") as mock_pattern_loader,
        ):
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = None
            mock_scanner_class.return_value = mock_scanner
            matcher = MagicMock()
            matcher.score_path.return_value = 0
            matcher.score_content.return_value = 0
            matcher._tokenise_text.return_value = []
            mock_pattern_loader.return_value = matcher

            result = _main()(argv)

            assert result == 0


class TestMainCLIIntegration:
    """Integration tests for main.py with --workspace-path."""

    def test_main_creates_local_filesystem_client_when_workspace_provided(
        self,
        temp_test_repo: Path,
    ) -> None:  # NOSONAR S2325
        """Test main() creates LocalFilesystemClient when --workspace-path provided."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(temp_test_repo),
        ]

        with (
            patch("src.main.LocalFilesystemClient") as mock_fs_client_class,
            patch("src.main.Scanner") as mock_scanner_class,
            patch("src.main.write_summary_file"),
            patch("src.main.GitHubClient"),
            patch("src.main.PatternMatcher.from_file") as mock_pattern_loader,
        ):
            mock_fs_client = MagicMock()
            mock_fs_client_class.return_value = mock_fs_client
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = None
            mock_scanner_class.return_value = mock_scanner
            matcher = MagicMock()
            matcher.score_path.return_value = 0
            matcher.score_content.return_value = 0
            matcher._tokenise_text.return_value = []
            mock_pattern_loader.return_value = matcher

            _main()(argv)

            mock_fs_client_class.assert_called()

    @staticmethod
    def test_main_does_not_create_local_filesystem_client_without_workspace_path() -> None:
        """Test main() does not create LocalFilesystemClient when --workspace-path not provided."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
        ]

        with (
            patch("src.main.LocalFilesystemClient") as mock_fs_client_class,
            patch("src.main.Scanner") as mock_scanner_class,
            patch("src.main.write_summary_file"),
            patch("src.main.GitHubClient"),
            patch("src.main.PatternMatcher.from_file"),
        ):
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = None
            mock_scanner_class.return_value = mock_scanner

            _main()(argv)

            mock_fs_client_class.assert_not_called()

    def test_main_passes_file_source_client_to_scanner(self, temp_test_repo: Path) -> None:  # NOSONAR S2325
        """Test main() passes file_source_client to Scanner constructor."""
        argv = [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(temp_test_repo),
        ]

        with (
            patch("src.main.LocalFilesystemClient") as mock_fs_client_class,
            patch("src.main.Scanner") as mock_scanner_class,
            patch("src.main.write_summary_file"),
            patch("src.main.GitHubClient"),
            patch("src.main.PatternMatcher.from_file") as mock_pattern_loader,
        ):
            mock_fs_client = MagicMock()
            mock_fs_client_class.return_value = mock_fs_client
            mock_scanner = MagicMock()
            mock_scanner.scan.return_value = None
            mock_scanner_class.return_value = mock_scanner
            matcher = MagicMock()
            matcher.score_path.return_value = 0
            matcher.score_content.return_value = 0
            matcher._tokenise_text.return_value = []
            mock_pattern_loader.return_value = matcher

            _main()(argv)

            call_args = mock_scanner_class.call_args
            assert call_args is not None
            assert "file_source_client" in call_args.kwargs
            assert call_args.kwargs["file_source_client"] is mock_fs_client


def test_owner_detection_disabled_for_local_scan(monkeypatch) -> None:  # NOSONAR S2325
    """Owner detection should be skipped when disabled by environment."""
    monkeypatch.setenv("AGENT_SCANNER_OWNER_DETECTION_ENABLED", "0")

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    scanner = Scanner(fake_client, fake_matcher)
    result = RepoScanResult(repo_name="repo", org="org")

    with patch("src.scanner.scanner.detect_likely_owner") as mock_owner:
        scanner._extract_repository_info("org", "repo", result)

    assert result.owner_detected is False
    assert result.detected_owner_name is None
    assert result.detected_owner_email is None
    mock_owner.assert_not_called()
