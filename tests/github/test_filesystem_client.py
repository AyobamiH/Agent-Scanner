"""Tests for LocalFilesystemClient."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.exceptions import GitHubClientError
from src.github.filesystem_client import LocalFilesystemClient


@pytest.fixture
def temp_repo_structure(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a temporary repository structure for testing.

    Returns:
        Tuple of (repo_root, files_dict) where files_dict maps paths to contents.
    """
    files = {
        "README.md": "# Test Repository",
        "src/main.py": "print('hello')",
        "src/agents.py": "from langchain import Agent",
        "tests/test_main.py": "def test_something(): pass",
        "requirements.txt": "langchain==0.1.0\nopenai==1.0.0",
        ".gitignore": "*.pyc\n__pycache__",
    }

    for file_path, content in files.items():
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    return tmp_path, files


@pytest.fixture
def mock_github_client() -> MagicMock:
    """Create a mock GitHub client."""
    client = MagicMock()
    client.fetch_all_commits.return_value = [
        {
            "author": "John Doe",
            "committer": "Jane Smith",
            "author_email": "john@example.com",
            "committer_email": "jane@example.com",
        }
    ]
    return client


class TestLocalFilesystemClientInitialisation:
    """Test LocalFilesystemClient initialisation."""

    def test_init_with_valid_workspace_path(self, temp_repo_structure: tuple[Path, dict]) -> None:  # NOSONAR S2325
        """Test initialising with valid workspace path."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()

        client = LocalFilesystemClient(str(repo_root), mock_client)

        assert client.workspace_path == str(repo_root.resolve())
        assert client.github_client is mock_client

    def test_init_with_nonexistent_path_raises_error(self, mock_github_client: MagicMock) -> None:  # NOSONAR S2325
        """Test initialising with nonexistent path raises GitHubClientError."""
        with pytest.raises(GitHubClientError, match="Workspace path does not exist"):
            LocalFilesystemClient("/nonexistent/path", mock_github_client)

    def test_init_with_file_instead_of_directory_raises_error(
        self, tmp_path: Path, mock_github_client: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test initialising with file instead of directory raises GitHubClientError."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        with pytest.raises(GitHubClientError, match="Workspace path is not a directory"):
            LocalFilesystemClient(str(file_path), mock_github_client)

    def test_init_with_custom_max_file_size(self, temp_repo_structure: tuple[Path, dict]) -> None:  # NOSONAR S2325
        """Test initialising with custom max_file_size."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        custom_size = 5_000_000

        client = LocalFilesystemClient(str(repo_root), mock_client, max_file_size=custom_size)

        assert client.max_file_size == custom_size


class TestGetRepoTree:
    """Test get_repo_tree method."""

    def test_get_repo_tree_returns_all_files(self, temp_repo_structure: tuple[Path, dict]) -> None:  # NOSONAR S2325
        """Test get_repo_tree returns all files from workspace."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        tree, _ = client.get_repo_tree("owner", "repo", branch=None)

        assert isinstance(tree, list)
        assert len(tree) > 0
        assert any(entry["path"] == "README.md" for entry in tree)
        assert any(entry["path"] == "src/main.py" for entry in tree)

    def test_get_repo_tree_includes_metadata(self, temp_repo_structure: tuple[Path, dict]) -> None:  # NOSONAR S2325
        """Test get_repo_tree returns proper metadata."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        _, metadata = client.get_repo_tree("owner", "repo")

        assert "default_branch" in metadata
        assert "head_sha" in metadata
        assert "html_url" in metadata
        assert metadata["default_branch"] == "main"

    def test_get_repo_tree_ignores_git_and_cache_dirs(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_repo_tree ignores .git, __pycache__, and similar directories."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()

        (repo_root / ".git").mkdir()
        (repo_root / ".git" / "config").write_text("git config")
        (repo_root / "__pycache__").mkdir()
        (repo_root / "__pycache__" / "cache.pyc").write_text("cache")

        client = LocalFilesystemClient(str(repo_root), mock_client)
        tree, _ = client.get_repo_tree("owner", "repo")

        paths = [entry["path"] for entry in tree]

        assert not any(part == ".git" or part.startswith(".git/") for path in paths for part in path.split("/"))
        assert not any(
            part == "__pycache__" or part.startswith("__pycache__/") for path in paths for part in path.split("/")
        )

    def test_get_repo_tree_file_entries_have_correct_type(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_repo_tree file entries have type='blob'."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        tree, _ = client.get_repo_tree("owner", "repo")

        for entry in tree:
            if "." in entry["path"].split("/")[-1]:
                assert entry["type"] in ("blob", "tree")

    def test_get_repo_tree_validates_owner_and_repo(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_repo_tree validates owner and repo parameters."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        with pytest.raises(GitHubClientError, match="owner must be a non-empty string"):
            client.get_repo_tree("", "repo")

        with pytest.raises(GitHubClientError, match="repo must be a non-empty string"):
            client.get_repo_tree("owner", "")

    def test_get_repo_tree_enforces_repo_size_limit(
        self,
        temp_repo_structure: tuple[Path, dict],
        monkeypatch,
    ) -> None:  # NOSONAR S2325
        """Test get_repo_tree raises when repository exceeds size limit."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        monkeypatch.setenv("AGENT_SCANNER_MAX_REPO_BYTES", "10")

        client = LocalFilesystemClient(str(repo_root), mock_client)

        with pytest.raises(GitHubClientError, match="Repository size"):
            client.get_repo_tree("owner", "repo")


class TestGetFileContent:
    """Test get_file_content method."""

    def test_get_file_content_reads_file(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_file_content reads file content correctly."""
        repo_root, files = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        content = client.get_file_content("owner", "repo", "README.md")

        assert content == files["README.md"]

    def test_get_file_content_raises_on_nonexistent_file(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_file_content raises GitHubClientError for missing files."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        with pytest.raises(GitHubClientError, match="File not found"):
            client.get_file_content("owner", "repo", "nonexistent.txt")

    def test_get_file_content_respects_max_file_size(
        self,
        tmp_path: Path,
        mock_github_client: MagicMock,
    ) -> None:  # NOSONAR S2325
        """Test get_file_content raises error for files exceeding max size."""
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * (2 * 1_000_000))

        client = LocalFilesystemClient(str(tmp_path), mock_github_client)

        with pytest.raises(GitHubClientError, match="File exceeds maximum size"):
            client.get_file_content("owner", "repo", "large.txt")

    def test_get_file_content_handles_utf8_encoding(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_file_content handles UTF-8 encoded files."""
        repo_root, _ = temp_repo_structure
        utf8_file = repo_root / "utf8.txt"
        utf8_file.write_text("Hello 世界 🌍", encoding="utf-8")

        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        content = client.get_file_content("owner", "repo", "utf8.txt")

        assert "世界" in content


class TestFetchAllCommits:
    """Test fetch_all_commits behavior for local filesystem client."""

    def test_fetch_all_commits_pipeline_mode_uses_local_git(
        self, temp_repo_structure: tuple[Path, dict], monkeypatch
    ) -> None:  # NOSONAR S2325
        """Pipeline mode should read commit history from local git."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        monkeypatch.setenv("AGENT_SCANNER_PIPELINE_MODE", "1")

        class DummyResult:
            def __init__(self, stdout: str) -> None:
                self.stdout = stdout
                self.stderr = ""
                self.returncode = 0

        sample_output = "Alice\x1falice@example.com\x1fBob\x1fbob@example.com\x1e"

        def fake_run(*_args, **_kwargs):
            return DummyResult(sample_output)

        monkeypatch.setattr("src.github.filesystem_client.subprocess.run", fake_run)

        client = LocalFilesystemClient(str(repo_root), mock_client)
        commits = client.fetch_all_commits("owner/repo", max_commits=5)

        assert commits == [
            {
                "author": "Alice",
                "author_email": "alice@example.com",
                "committer": "Bob",
                "committer_email": "bob@example.com",
            }
        ]
        mock_client.fetch_all_commits.assert_not_called()

    def test_fetch_all_commits_non_pipeline_delegates_to_github(
        self, temp_repo_structure: tuple[Path, dict], monkeypatch
    ) -> None:  # NOSONAR S2325
        """Outside pipeline mode, commit history should delegate to GitHub client."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        monkeypatch.delenv("AGENT_SCANNER_PIPELINE_MODE", raising=False)

        client = LocalFilesystemClient(str(repo_root), mock_client)
        client.fetch_all_commits("owner/repo", max_commits=2)

        mock_client.fetch_all_commits.assert_called_once()

    def test_get_file_content_protects_against_path_traversal(
        self, temp_repo_structure: tuple[Path, dict], mock_github_client: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test get_file_content prevents path traversal attacks."""
        repo_root, _ = temp_repo_structure
        client = LocalFilesystemClient(str(repo_root), mock_github_client)

        with pytest.raises(GitHubClientError, match="Path traversal not allowed"):
            client.get_file_content("owner", "repo", "../../etc/passwd")

    def test_get_file_content_validates_parameters(
        self,
        temp_repo_structure: tuple[Path, dict],
    ) -> None:  # NOSONAR S2325
        """Test get_file_content validates owner, repo, and path parameters."""
        repo_root, _ = temp_repo_structure
        mock_client = MagicMock()
        client = LocalFilesystemClient(str(repo_root), mock_client)

        with pytest.raises(GitHubClientError, match="owner must be a non-empty string"):
            client.get_file_content("", "repo", "file.txt")

        with pytest.raises(GitHubClientError, match="repo must be a non-empty string"):
            client.get_file_content("owner", "", "file.txt")

        with pytest.raises(GitHubClientError, match="path must be a non-empty string"):
            client.get_file_content("owner", "repo", "")


class TestFetchAllCommitsDelegation:
    """Test fetch_all_commits delegation."""

    def test_fetch_all_commits_delegates_to_github_client(
        self, temp_repo_structure: tuple[Path, dict], mock_github_client: MagicMock, monkeypatch
    ) -> None:  # NOSONAR S2325
        """Test fetch_all_commits delegates to github_client."""
        repo_root, _ = temp_repo_structure
        monkeypatch.delenv("AGENT_SCANNER_PIPELINE_MODE", raising=False)
        client = LocalFilesystemClient(str(repo_root), mock_github_client)

        result = client.fetch_all_commits("owner/repo", max_commits=50, branch="main")

        mock_github_client.fetch_all_commits.assert_called_once_with("owner/repo", max_commits=50, branch="main")
        assert result == mock_github_client.fetch_all_commits.return_value

    def test_fetch_all_commits_returns_commit_structure(
        self, temp_repo_structure: tuple[Path, dict], mock_github_client: MagicMock, monkeypatch
    ) -> None:  # NOSONAR S2325
        """Test fetch_all_commits returns proper commit structure."""
        repo_root, _ = temp_repo_structure
        monkeypatch.delenv("AGENT_SCANNER_PIPELINE_MODE", raising=False)
        client = LocalFilesystemClient(str(repo_root), mock_github_client)

        commits = client.fetch_all_commits("owner/repo")

        assert isinstance(commits, list)
        if commits:
            commit = commits[0]
            assert "author" in commit
            assert "committer" in commit
            assert "author_email" in commit
            assert "committer_email" in commit


class TestStubMethods:
    """Test stub methods that are not implemented for local filesystem."""

    def test_list_branches_raises_error(
        self, temp_repo_structure: tuple[Path, dict], mock_github_client: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test list_branches raises GitHubClientError."""
        repo_root, _ = temp_repo_structure
        client = LocalFilesystemClient(str(repo_root), mock_github_client)

        with pytest.raises(GitHubClientError, match="list_branches is not supported"):
            client.list_branches("owner", "repo")

    def test_list_org_repos_raises_error(
        self, temp_repo_structure: tuple[Path, dict], mock_github_client: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test list_org_repos raises GitHubClientError."""
        repo_root, _ = temp_repo_structure
        client = LocalFilesystemClient(str(repo_root), mock_github_client)

        with pytest.raises(GitHubClientError, match="list_org_repos is not supported"):
            client.list_org_repos("org")

    def test_get_branches_raises_error(
        self, temp_repo_structure: tuple[Path, dict], mock_github_client: MagicMock
    ) -> None:  # NOSONAR S2325
        """Test get_branches raises GitHubClientError."""
        repo_root, _ = temp_repo_structure
        client = LocalFilesystemClient(str(repo_root), mock_github_client)

        with pytest.raises(GitHubClientError, match="get_branches is not supported"):
            client.get_branches("owner", "repo")
