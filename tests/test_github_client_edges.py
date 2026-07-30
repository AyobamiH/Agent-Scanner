"""Additional edge-case tests for GitHubClient."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import GitHubClientError
from src.github.client import GitHubClient


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
def test_get_branches_uses_cache(monkeypatch):  # NOSONAR S2325
    """get_branches should return cached value without hitting API."""

    client = GitHubClient()
    cached = ["main", "dev"]

    monkeypatch.setattr(client, "_get_from_cache", lambda key: cached if key.startswith("branches:") else None)
    called = {"count": 0}
    monkeypatch.setattr(client, "_get_github_api", lambda *a, **k: called.__setitem__("count", called["count"] + 1))

    result = client.get_branches("owner", "repo")

    assert result == cached
    assert called["count"] == 0


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
def test_get_branches_paginates(monkeypatch):  # NOSONAR S2325
    """get_branches should concatenate paginated pages until short page returned."""

    client = GitHubClient()

    pages = [
        [{"name": "main"}] * 100,
        [{"name": "dev"}],
    ]

    def fake_get(path, params=None):
        if "branches" in path:
            index = params["page"] - 1 if params else 0
            return pages[index]
        raise AssertionError("Unexpected path")

    monkeypatch.setattr(client, "_get_github_api", fake_get)

    result = client.get_branches("owner", "repo")

    assert result == ["main"] * 100 + ["dev"]


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
def test_get_file_content_uses_404_cache(monkeypatch):  # NOSONAR S2325
    """Subsequent calls for cached 404 should raise without network."""

    client = GitHubClient()
    client._404_cache.add("file:owner/repo:HEAD:missing.txt")  # type: ignore[attr-defined]

    with pytest.raises(GitHubClientError):
        client.get_file_content("owner", "repo", "missing.txt")


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
def test_get_file_content_contents_then_raw(monkeypatch):  # NOSONAR S2325
    """Falls back to raw URL when contents API returns non-200."""

    client = GitHubClient()
    mock_session = MagicMock()

    def side_effect(url, params=None, timeout=None):
        resp = MagicMock()
        if "contents" in url:
            resp.status_code = 404
            return resp
        resp.status_code = 200
        resp.text = "raw-body"
        resp.encoding = "utf-8"
        return resp

    mock_session.get.side_effect = side_effect
    client.session = mock_session

    content = client.get_file_content("owner", "repo", "path/to/file.txt")

    assert content == "raw-body"
    assert mock_session.get.call_count == 2


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
def test_execute_request_with_retry_eventually_succeeds(monkeypatch):  # NOSONAR S2325
    """_execute_request_with_retry retries on rate limit then succeeds."""

    client = GitHubClient()
    responses = [429, 429, 200]

    class Resp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"Content-Length": "3"}

        def json(self):
            return {}

    def fake_get(url, params=None, timeout=None, headers=None):
        status = responses.pop(0)
        return Resp(status)

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr(client, "API_URL", client._api_url)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    resp = client._execute_request_with_retry(url="http://example.com", method="GET", max_attempts=3)  # NOSONAR S5332

    assert resp.status_code == 200


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_fetch_via_contents_api_handles_rate_limit_retries(mock_get):  # NOSONAR S2325
    """_fetch_via_contents_api retries on rate limit before giving up."""
    client = GitHubClient()
    responses = [429, 429, 404]

    class MockResp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}

    mock_get.side_effect = [MockResp(s) for s in responses]

    result = client._fetch_via_contents_api("owner", "repo", "file.txt")

    assert result is None
    assert mock_get.call_count == 2


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_fetch_via_contents_api_returns_on_200(mock_get):  # NOSONAR S2325
    """_fetch_via_contents_api returns content on 200 status."""
    client = GitHubClient()

    class MockResp:
        status_code = 200
        headers = {"Content-Length": "11"}
        text = "hello world"
        encoding = "utf-8"

    mock_get.return_value = MockResp()

    result = client._fetch_via_contents_api("owner", "repo", "file.txt")

    assert result == "hello world"


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_fetch_via_raw_url_handles_404_and_caches(mock_get):  # NOSONAR S2325
    """_fetch_via_raw_url adds 404 responses to cache."""
    client = GitHubClient()

    class MockResp:
        status_code = 404
        headers = {}

    mock_get.return_value = MockResp()

    cache_key = "file:owner/repo:HEAD:missing.txt"
    with pytest.raises(GitHubClientError):
        client._fetch_via_raw_url("owner", "repo", "missing.txt", cache_key)

    assert cache_key in client._404_cache


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.get")
def test_fetch_via_raw_url_retries_on_request_exception(mock_get):  # NOSONAR S2325
    """_fetch_via_raw_url retries on network errors."""
    import requests

    client = GitHubClient()
    responses = [
        requests.RequestException("timeout"),
        requests.RequestException("timeout"),
    ]

    class MockResp:
        status_code = 200
        headers = {"Content-Length": "5"}
        text = "hello"
        encoding = "utf-8"

    mock_get.side_effect = responses + [MockResp()]

    with patch("time.sleep"):
        result = client._fetch_via_raw_url("owner", "repo", "file.txt", "file:owner/repo:HEAD:file.txt")

    assert result == "hello"


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.post")
def test_get_graphql_calls_with_post_method(mock_post):  # NOSONAR S2325
    """_get_graphql uses POST method with query and variables."""
    client = GitHubClient()

    class MockResp:
        status_code = 200

        def json(self):
            return {"data": {"viewer": {"login": "test"}}}

    mock_post.return_value = MockResp()

    result = client._get_graphql("query { viewer { login } }", variables={"test": "value"})

    assert result["data"]["viewer"]["login"] == "test"
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "graphql" in call_args[0][0]


@patch.dict(os.environ, {"GITHUB_TOKEN": "fake"})
def test_get_repo_tree_graphql_with_custom_branch(monkeypatch):  # NOSONAR S2325
    """_get_repo_tree_graphql can fetch specific branch."""
    client = GitHubClient()

    class MockResp:
        status_code = 200

        def json(self):
            return {
                "data": {
                    "repository": {
                        "url": "https://github.com/owner/repo",
                        "defaultBranchRef": {"name": "main"},
                        "ref": {
                            "name": "develop",
                            "target": {
                                "oid": "abc123",
                                "tree": {
                                    "entries": [
                                        {
                                            "path": "file.py",
                                            "mode": "100644",
                                            "oid": "xyz",
                                            "type": "BLOB",
                                            "object": None,
                                        }
                                    ]
                                },
                            },
                        },
                    }
                }
            }

    def mock_post(url, **kwargs):
        return MockResp()

    monkeypatch.setattr(client.session, "post", mock_post)

    result = client._get_repo_tree_graphql("owner", "repo", branch="develop")

    assert result is not None
    entries, metadata = result
    assert len(entries) == 1
    assert entries[0]["path"] == "file.py"
    assert metadata["default_branch"] == "main"
