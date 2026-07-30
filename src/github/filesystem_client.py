"""
Local filesystem-based repository client for reading files from disk.

This client mirrors the GitHubClient interface for file operations (get_repo_tree,
get_file_content) but reads from the local filesystem instead of the GitHub API.

For owner detection (fetch_all_commits), this client can use local git history in
local mode to avoid GitHub API calls. Outside this mode, it delegates to a
GitHubClient instance when provided.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from src.exceptions import GitHubClientError

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1_000_000
MAX_REPO_SIZE_ENV = "AGENT_SCANNER_MAX_REPO_BYTES"


def _get_git_commit_hash(repo_path: str) -> str:
    """Get the current HEAD commit hash from a git repository.

    Args:
        repo_path: Path to the git repository.

    Returns:
        The commit hash (40 hex characters) or all zeros if not a git repo or error.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            if len(commit_hash) == 40 and all(c in "0123456789abcdef" for c in commit_hash):
                return commit_hash
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Failed to get git commit hash from %s: %s", repo_path, exc)

    return "0" * 40


def _get_git_current_branch(repo_path: str) -> str | None:
    """Get the current branch name from a git repository.

    Args:
        repo_path: Path to the git repository.

    Returns:
        Branch name or None if not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch_name = result.stdout.strip()
            if branch_name and branch_name != "HEAD":
                return branch_name
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Failed to get git branch from %s: %s", repo_path, exc)

    return None


def _get_git_commit_history(repo_path: str, max_commits: int, branch: str | None = None) -> list[dict[str, Any]]:
    """Get commit history from local git log.

    Args:
        repo_path: Path to the git repository.
        max_commits: Maximum number of commits to fetch.
        branch: Optional branch name to fetch commits from.

    Returns:
        List of commit dictionaries with author/committer details.

    Raises:
        GitHubClientError: If git log fails or output cannot be parsed.
    """
    if max_commits <= 0:
        return []

    if branch and not re.match(r"^[a-zA-Z0-9/_.\-]+$", branch):
        raise GitHubClientError(f"Invalid branch name: {branch}")

    record_sep = "\x1e"
    field_sep = "\x1f"
    format_str = f"%an{field_sep}%ae{field_sep}%cn{field_sep}%ce{record_sep}"
    command = ["git", "log"]
    if branch:
        command.append(branch)
    command.extend(["-n", str(max_commits), f"--pretty=format:{format_str}"])

    try:
        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Failed to run git log in %s: %s", repo_path, exc)
        raise GitHubClientError("Failed to run git log for commit history") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger.debug("git log failed (%s): %s", result.returncode, stderr)
        raise GitHubClientError("Failed to retrieve commit history from git")

    output = result.stdout or ""
    if not output.strip():
        return []

    commits: list[dict[str, Any]] = []
    records = output.strip(record_sep).split(record_sep)
    for record in records:
        fields = record.split(field_sep)
        if len(fields) < 4:
            continue
        author_name, author_email, committer_name, committer_email = fields[:4]
        commits.append(
            {
                "author": author_name.strip(),
                "author_email": author_email.strip(),
                "committer": committer_name.strip(),
                "committer_email": committer_email.strip(),
            }
        )

    return commits


def _resolve_repo_size_limit() -> int | None:
    """Resolve max repository size from environment variable.

    Returns:
        Maximum repository size in bytes, or None if not configured.
    """
    configured = os.getenv(MAX_REPO_SIZE_ENV)
    if not configured:
        return None
    try:
        limit = int(configured)
    except ValueError:
        logger.warning("Invalid %s value: %s", MAX_REPO_SIZE_ENV, configured)
        return None
    if limit <= 0:
        return None
    return limit


class LocalFilesystemClient:
    """Local filesystem file source client.

    Reads files from a local directory (typically a cloned repository) instead of
    fetching from GitHub API. Implements the same public interface as GitHubClient
    for file operations to allow interchangeable use in Scanner.

    For commit fetching (owner detection), this client can use local git history
    in pipeline mode to avoid GitHub API calls. Outside pipeline mode, it will
    delegate to a GitHubClient instance when provided.

    Attributes:
        workspace_path: Absolute path to local repository directory.
        github_client: Optional GitHubClient instance for commit fetching (owner detection).
        max_file_size: Maximum file size in bytes to read (default 1MB).
        api_url: Optional API URL for building repo URLs in results.
    """

    def __init__(
        self,
        workspace_path: str,
        github_client: Any | None,
        max_file_size: int = MAX_FILE_SIZE,
        api_url: str | None = None,
    ) -> None:
        """Initialise the local filesystem client.

        Args:
            workspace_path: Absolute path to local repository directory.
            github_client: Optional GitHubClient instance for commit fetching (owner detection).
            max_file_size: Maximum file size in bytes to read (default 1MB).
            api_url: Optional API URL for building repo URLs in results.

        Raises:
            GitHubClientError: If workspace_path does not exist or is not a directory.
        """
        workspace_path_obj = Path(workspace_path)

        if not workspace_path_obj.exists():
            raise GitHubClientError(f"Workspace path does not exist: {workspace_path}")

        if not workspace_path_obj.is_dir():
            raise GitHubClientError(f"Workspace path is not a directory: {workspace_path}")

        self.workspace_path = str(workspace_path_obj.resolve())
        self.github_client = github_client
        self.max_file_size = max_file_size
        self._api_url = (api_url or os.getenv("GITHUB_API_URL") or "https://api.github.com").rstrip("/")
        self.api_stats: dict[str, int] = {}

        logger.info("Local filesystem client initialised with workspace: %s", self.workspace_path)

    def get_repo_tree(
        self, owner: str, repo: str, branch: str | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve the complete file tree from local filesystem.

        Recursively walks the workspace directory to build a file tree matching
        the GitHubClient interface. Ignores common non-code directories (e.g., .git,
        __pycache__, .venv, node_modules, .egg-info).

        Note: branch parameter is ignored for local filesystem (all files read
        from workspace directory as-is).

        Args:
            owner: Repository owner (used for logging/compatibility, not filesystem lookup).
            repo: Repository name (used for logging/compatibility, not filesystem lookup).
            branch: Optional branch name (ignored for local filesystem).

        Returns:
            Tuple of (tree_entries, metadata) where:
            - tree_entries: List of file tree entries with keys: path, mode, sha, type, size
            - metadata: Dict with keys: default_branch, head_sha, html_url

        Raises:
            GitHubClientError: If workspace directory cannot be read.
        """
        if not owner or not isinstance(owner, str):
            raise GitHubClientError("owner must be a non-empty string")
        if not repo or not isinstance(repo, str):
            raise GitHubClientError("repo must be a non-empty string")

        logger.debug("Building file tree from local workspace for %s/%s", owner, repo)

        tree_entries: list[dict[str, Any]] = []
        total_size = 0
        repo_size_limit = _resolve_repo_size_limit()
        ignored_dirs = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "env",
            "node_modules",
            ".egg-info",
            ".pytest_cache",
            ".tox",
            "dist",
            "build",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            ".vscode",
            ".idea",
        }

        try:
            workspace_path_obj = Path(self.workspace_path)

            for file_path in workspace_path_obj.rglob("*"):
                if any(part in ignored_dirs for part in file_path.parts):
                    continue

                try:
                    rel_path = file_path.relative_to(workspace_path_obj)
                except ValueError:
                    logger.debug("Could not compute relative path for %s", file_path)
                    continue

                if file_path.is_file():
                    try:
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        if repo_size_limit is not None and total_size > repo_size_limit:
                            error_message = f"Repository size {total_size} exceeds limit {repo_size_limit} bytes"
                            raise GitHubClientError(error_message)
                        tree_entries.append(
                            {
                                "path": str(rel_path).replace("\\", "/"),
                                "mode": "100644",
                                "type": "blob",
                                "size": file_size,
                                "sha": "",
                            }
                        )
                    except OSError as exc:
                        logger.debug("Failed to stat file %s: %s", file_path, exc)
                        continue
                elif file_path.is_dir():
                    tree_entries.append(
                        {
                            "path": str(rel_path).replace("\\", "/"),
                            "mode": "040000",
                            "type": "tree",
                            "sha": "",
                        }
                    )

            tree_entries.sort(key=lambda entry: (entry.get("path", ""), entry.get("type", "")))
            logger.info("Built file tree with %d entries from local filesystem", len(tree_entries))

            head_sha = _get_git_commit_hash(self.workspace_path)
            metadata = {
                "default_branch": "main",
                "head_sha": head_sha,
                "html_url": f"file://{self.workspace_path}",
            }

            return tree_entries, metadata

        except OSError as exc:
            logger.exception("Failed to read workspace directory: %s", self.workspace_path)
            raise GitHubClientError(f"Failed to read workspace directory: {exc}") from exc

    def get_file_content(self, owner: str, repo: str, path: str, branch: str | None = None) -> str:
        """Fetch file content from local filesystem.

        Reads file content directly from disk relative to workspace directory.
        Enforces max_file_size limit to prevent reading large binary files.

        Note: owner, repo, and branch parameters are ignored for local filesystem
        (path is resolved relative to workspace directory).

        Args:
            owner: Repository owner (used for logging/compatibility, not used).
            repo: Repository name (used for logging/compatibility, not used).
            path: File path relative to repository root.
            branch: Optional branch name (ignored for local filesystem).

        Returns:
            File contents as decoded UTF-8 string.

        Raises:
            GitHubClientError: If file cannot be read, does not exist, or exceeds size limits.
        """
        if not owner or not isinstance(owner, str):
            raise GitHubClientError("owner must be a non-empty string")
        if not repo or not isinstance(repo, str):
            raise GitHubClientError("repo must be a non-empty string")
        if not path or not isinstance(path, str):
            raise GitHubClientError("path must be a non-empty string")

        normalised_path = path.replace("\\", "/")

        file_path = (Path(self.workspace_path) / normalised_path).resolve()

        try:
            file_path.relative_to(Path(self.workspace_path).resolve())
        except ValueError as err:
            logger.warning("Path traversal attempt detected: %s", path)
            raise GitHubClientError(f"Path traversal not allowed: {path}") from err

        logger.debug("Reading file from local filesystem: %s", path)

        if not file_path.exists():
            logger.debug("File not found in local filesystem: %s", path)
            raise GitHubClientError(f"File not found: {path}")

        if not file_path.is_file():
            logger.debug("Path is not a file: %s", path)
            raise GitHubClientError(f"Path is not a file: {path}")

        try:
            file_size = file_path.stat().st_size

            if file_size > self.max_file_size:
                logger.debug("File exceeds max size (%d > %d): %s", file_size, self.max_file_size, path)
                raise GitHubClientError(f"File exceeds maximum size ({file_size} > {self.max_file_size}): {path}")

            content = file_path.read_text(encoding="utf-8")
            logger.debug("Successfully read file %s (%d bytes)", path, len(content))
            return content

        except UnicodeDecodeError as exc:
            logger.debug("File is not valid UTF-8: %s", path)
            raise GitHubClientError(f"File is not valid UTF-8: {path}") from exc
        except OSError as exc:
            logger.debug("Failed to read file %s: %s", path, exc)
            raise GitHubClientError(f"Failed to read file: {path}") from exc

    def fetch_all_commits(
        self, repo_full_name: str, max_commits: int = 100, branch: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch commit history for owner detection.

        In pipeline mode, uses local git history instead of the GitHub API.

        Args:
            repo_full_name: Repository in owner/repo format.
            max_commits: Maximum number of commits to fetch.
            branch: Optional branch name to fetch commits from.

        Returns:
            List of commit dictionaries with author, committer, and email information.

        Raises:
            GitHubClientError: If commit fetching fails.
        """
        pipeline_mode = os.getenv("AGENT_SCANNER_PIPELINE_MODE", "0") == "1"
        if pipeline_mode:
            logger.debug("Fetching commit history from local git for owner detection: %s", repo_full_name)
            return _get_git_commit_history(self.workspace_path, max_commits=max_commits, branch=branch)

        logger.debug("Delegating commit fetch to GitHub client for owner detection: %s", repo_full_name)
        if self.github_client is None:
            raise GitHubClientError("GitHub client not configured for commit fetching")
        return self.github_client.fetch_all_commits(repo_full_name, max_commits=max_commits, branch=branch)

    def get_repo_branch_and_head(self, _repo: str) -> tuple[str | None, str | None]:
        """Return branch name and HEAD commit hash from local git.

        Args:
            _repo: Repository name (unused for local filesystem).

        Returns:
            Tuple of (default_branch, head_commit_sha).
        """
        branch_name = _get_git_current_branch(self.workspace_path) or "main"
        head_sha = _get_git_commit_hash(self.workspace_path)
        return branch_name, head_sha

    @staticmethod
    def list_branches(owner: str, repo: str) -> list[dict[str, Any]]:
        """Placeholder for list_branches (not implemented for local filesystem).

        Args:
            owner: Repository owner (not used).
            repo: Repository name (not used).

        Raises:
            GitHubClientError: Always raises as operation is not supported for local filesystem.
        """
        raise GitHubClientError("list_branches is not supported for local filesystem client")

    @staticmethod
    def list_org_repos(org: str, page: int = 1, per_page: int = 30, **_kwargs: Any) -> list[dict[str, Any]]:
        """Placeholder for list_org_repos (not implemented for local filesystem).

        Args:
            org: Organisation name (not used).
            page: Page number (not used).
            per_page: Items per page (not used).

        Raises:
            GitHubClientError: Always raises as operation is not supported for local filesystem.
        """
        raise GitHubClientError("list_org_repos is not supported for local filesystem client")

    @staticmethod
    def get_branches(owner: str, repo: str) -> list[dict[str, Any]]:
        """Placeholder for get_branches (not implemented for local filesystem).

        Args:
            owner: Repository owner (not used).
            repo: Repository name (not used).

        Raises:
            GitHubClientError: Always raises as operation is not supported for local filesystem.
        """
        raise GitHubClientError("get_branches is not supported for local filesystem client")
