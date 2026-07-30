"""Tests for GitHub Enterprise API client compatibility."""

from unittest.mock import MagicMock

from src.github.client import GitHubClient


class MockResp:
    def __init__(self, status=200, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self.encoding = None

    def json(self):
        return {}


def test_get_file_content_contents_api_success(monkeypatch):  # NOSONAR S2325
    """When the contents API returns 200 with raw accept, client returns text."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    client = GitHubClient()
    mock_sess = MagicMock()

    def side_effect(url, headers=None, timeout=None, params=None):
        if "contents" in url:
            return MockResp(status=200, text="file content")
        return MockResp(status=404)

    mock_sess.get.side_effect = side_effect
    client.session = mock_sess
    content = client.get_file_content("owner", "repo", "path/to/file.txt")
    assert content == "file content"


def test_get_file_content_fallback_to_raw(monkeypatch):
    """When contents API 404, raw URL is attempted and returned."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    client = GitHubClient()
    mock_sess = MagicMock()

    def side_effect(url, headers=None, timeout=None, params=None):
        if "contents" in url:
            return MockResp(status=404)
        if "raw.githubusercontent.com" in url:
            return MockResp(status=200, text="raw content")
        return MockResp(status=404)

    mock_sess.get.side_effect = side_effect
    client.session = mock_sess
    content = client.get_file_content("owner", "repo", "path/to/file.txt")
    assert content == "raw content"
