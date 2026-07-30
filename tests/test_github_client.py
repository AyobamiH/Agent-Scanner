"""Tests for GitHub API client core functionality."""

import os
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import GitHubClientError
from src.github.client import GitHubClient


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_repo_tree_and_file_content(mock_get):
    """GitHubClient.get_repo_tree and get_file_content return expected results."""

    def side_effect(url, params=None, timeout=None):
        mock_resp = MagicMock()
        if url.endswith("/repos/owner/repo"):
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"default_branch": "main"}
            return mock_resp
        if url.endswith("/git/refs/heads/main"):
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"object": {"sha": "sha1"}}
            return mock_resp
        if "/git/trees/" in url:
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"tree": [{"path": "a.py", "type": "blob"}]}
            return mock_resp

        mock_resp.status_code = 200
        mock_resp.text = 'print("hi")'
        mock_resp.encoding = "utf-8"
        return mock_resp

    mock_get.side_effect = side_effect

    client = GitHubClient()

    tree, metadata = client.get_repo_tree("owner", "repo")
    content = client.get_file_content("owner", "repo", "a.py")

    assert isinstance(tree, list)
    assert isinstance(metadata, dict)
    assert metadata["default_branch"] == "main"
    assert metadata["head_sha"] == "sha1"
    assert "print" in content


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake", "AGENT_SCANNER_MAX_REPO_BYTES": "5"})
def test_get_repo_tree_enforces_repo_size_limit(monkeypatch):
    """Repo size limits should raise GitHubClientError when exceeded."""
    client = GitHubClient()

    def fake_get(path, params=None):
        if "/git/refs/heads/" in path:
            return {"object": {"sha": "sha1"}}
        if "/git/trees/" in path:
            return {
                "tree": [
                    {"path": "a.py", "type": "blob", "size": 4},
                    {"path": "b.py", "type": "blob", "size": 4},
                ]
            }
        if path.startswith("/repos/owner/repo"):
            return {"default_branch": "main", "html_url": "https://github.com/owner/repo"}
        raise AssertionError("Unexpected path")

    monkeypatch.setattr(client, "_get_github_api", fake_get)
    monkeypatch.setattr(client, "_get_repo_tree_graphql", lambda *a, **k: None)

    with pytest.raises(GitHubClientError, match="Repository size"):
        client.get_repo_tree("owner", "repo")


@patch.dict(os.environ, {}, clear=True)
def test_github_client_requires_token():
    with pytest.raises(GitHubClientError):
        GitHubClient()


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test__get_non200_non_rate_limit(mock_get):
    """Raise GitHubClientError on non-200 and non-rate-limit response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "server error"
    mock_get.return_value = mock_resp

    client = GitHubClient()

    with pytest.raises(GitHubClientError):
        client._get_github_api("/some/path")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test__get_rate_limit_exhaustion(mock_get):
    """Exhausting retries on rate-limited responses raises GitHubClientError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "rate limited"
    mock_get.return_value = mock_resp

    client = GitHubClient()
    with patch("time.sleep", return_value=None):
        with pytest.raises(GitHubClientError):
            client._get_github_api("/some/path")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_repo_tree_missing_sha(mock_get):
    """Missing sha in commit object raises GitHubClientError."""
    repo_resp = MagicMock()
    repo_resp.status_code = 200
    repo_resp.json.return_value = {"default_branch": "main"}

    refs_resp = MagicMock()
    refs_resp.status_code = 200
    refs_resp.json.return_value = {"object": {}}

    mock_get.side_effect = [repo_resp, refs_resp]

    client = GitHubClient()
    with pytest.raises(GitHubClientError):
        client.get_repo_tree("owner", "repo")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_repo_tree_network_exception(mock_get):
    """Network exceptions from requests are wrapped into GitHubClientError."""
    mock_get.side_effect = Exception("network down")
    client = GitHubClient()
    with patch("time.sleep", return_value=None):
        with pytest.raises(GitHubClientError):
            client.get_repo_tree("owner", "repo")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_file_content_rate_limit_exhaustion(mock_get):
    """Repeated rate-limited raw content responses exhaust and raise GitHubClientError."""
    resp = MagicMock()
    resp.status_code = 429
    mock_get.return_value = resp

    client = GitHubClient()
    with patch("time.sleep", return_value=None):
        with pytest.raises(GitHubClientError):
            client.get_file_content("owner", "repo", "a.txt")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_github_client_backoff_retries_then_succeeds(mock_get, monkeypatch):
    """Client retries on 403 responses and eventually succeeds when rate-limit clears."""
    call_count = {"c": 0}

    def side_effect(url, params=None, timeout=None):
        mock_resp = MagicMock()
        if call_count["c"] < 2:
            mock_resp.status_code = 403
            mock_resp.text = "rate limit"
        else:
            if url.endswith("/repos/owner/repo"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"default_branch": "main"}
            elif url.endswith("/git/refs/heads/main"):
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"object": {"sha": "sha1"}}
            elif "/git/trees/" in url:
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"tree": []}
            else:
                mock_resp.status_code = 200
                mock_resp.text = ""
        call_count["c"] += 1
        return mock_resp

    mock_get.side_effect = side_effect
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = GitHubClient()

    tree, metadata = client.get_repo_tree("owner", "repo")
    assert isinstance(tree, list)
    assert isinstance(metadata, dict)


def test_get_file_content_skips_large(monkeypatch):
    class MockResp:
        status_code = 200
        headers = {"Content-Length": str(2_000_000)}
        text = ""
        encoding = "utf-8"

    mock_sess = MagicMock()
    mock_sess.get.return_value = MockResp()

    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    client = GitHubClient()
    client.session = mock_sess

    with pytest.raises(GitHubClientError):
        client.get_file_content("owner", "repo", "big.bin")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_list_org_repos_no_filters(mock_get):
    """list_org_repos returns all repos when no filters applied."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"name": "repo1", "full_name": "org/repo1", "pushed_at": "2025-01-01T00:00:00Z"},
        {"name": "repo2", "full_name": "org/repo2", "pushed_at": "2024-12-01T00:00:00Z"},
    ]
    mock_get.return_value = mock_resp

    client = GitHubClient()
    repos = client.list_org_repos("myorg")

    assert len(repos) == 2
    assert repos[0]["name"] == "repo1"
    assert repos[1]["name"] == "repo2"


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_list_org_repos_with_pushed_after_filter(mock_get):
    """list_org_repos stops when pushed_at falls below filter threshold."""
    from datetime import datetime

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"name": "repo1", "full_name": "org/repo1", "pushed_at": "2025-01-15T00:00:00Z"},
        {"name": "repo2", "full_name": "org/repo2", "pushed_at": "2024-12-01T00:00:00Z"},
    ]
    mock_get.return_value = mock_resp

    client = GitHubClient()
    cutoff = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    repos = client.list_org_repos("myorg", pushed_after=cutoff)

    assert len(repos) == 1
    assert repos[0]["name"] == "repo1"


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_list_org_repos_with_max_pages(mock_get):
    """list_org_repos respects max_pages limit."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"name": "repo", "full_name": "org/repo"}]
    mock_get.return_value = mock_resp

    client = GitHubClient()
    client.list_org_repos("myorg", max_pages=1)

    assert mock_get.call_count == 1


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_list_org_repos_with_max_repos(mock_get):
    """list_org_repos respects max_repos limit and stops early."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"name": f"repo{i}", "full_name": f"org/repo{i}"} for i in range(100)]
    mock_get.return_value = mock_resp

    client = GitHubClient()
    repos = client.list_org_repos("myorg", max_repos=5)

    assert len(repos) == 5


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_list_org_repos_requires_org_string(mock_get):
    """list_org_repos raises on empty or non-string org."""
    client = GitHubClient()

    with pytest.raises(GitHubClientError):
        client.list_org_repos("")

    # Intentionally passing None to verify input validation
    with pytest.raises(GitHubClientError):
        client.list_org_repos(None)  # NOSONAR S7125


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_fetch_all_commits_parses_author_data(mock_get):
    """fetch_all_commits normalises commit author and committer data."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "commit": {
                "author": {"name": "Alice", "email": "alice@example.com"},
                "committer": {"name": "Bob", "email": "bob@example.com"},
            }
        }
    ]
    mock_get.return_value = mock_resp

    client = GitHubClient()
    commits = client.fetch_all_commits("owner/repo", max_commits=10)

    assert len(commits) == 1
    assert commits[0]["author"] == "Alice"
    assert commits[0]["author_email"] == "alice@example.com"
    assert commits[0]["committer"] == "Bob"
    assert commits[0]["committer_email"] == "bob@example.com"


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_fetch_all_commits_requires_valid_repo_format(mock_get):
    """fetch_all_commits raises on invalid repo_full_name format."""
    client = GitHubClient()

    with pytest.raises(GitHubClientError):
        client.fetch_all_commits("invalid-format")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_file_content_validates_inputs(mock_get):
    """get_file_content validates owner, repo, and path parameters."""
    client = GitHubClient()

    with pytest.raises(GitHubClientError):
        client.get_file_content("", "repo", "path")

    with pytest.raises(GitHubClientError):
        client.get_file_content("owner", "", "path")

    with pytest.raises(GitHubClientError):
        client.get_file_content("owner", "repo", "")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_branches_validates_inputs(mock_get):
    """get_branches validates owner and repo parameters."""
    client = GitHubClient()

    with pytest.raises(GitHubClientError):
        client.get_branches("", "repo")

    with pytest.raises(GitHubClientError):
        client.get_branches("owner", "")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_get_repo_tree_validates_inputs(mock_get):
    """get_repo_tree validates owner and repo parameters."""
    client = GitHubClient()

    with pytest.raises(GitHubClientError):
        client.get_repo_tree("", "repo")

    with pytest.raises(GitHubClientError):
        client.get_repo_tree("owner", "")
