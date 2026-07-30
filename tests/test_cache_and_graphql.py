"""Tests for file cache and GraphQL client functionality."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import CacheError, GitHubClientError
from src.github.client import GitHubClient
from src.utils.cache import FileCache


@pytest.fixture
def cache_file(tmp_path: Path) -> Path:
    """Create a temporary cache file path."""
    return tmp_path / "test_cache.json"


@pytest.fixture
def cache(cache_file: Path) -> FileCache:
    """Create a FileCache instance for testing."""
    return FileCache(cache_file, max_items=10, default_ttl=3600)


def test_filecache_rejects_empty_path() -> None:
    """FileCache initialisation fails for empty path."""
    with pytest.raises(CacheError):
        FileCache("", max_items=10, default_ttl=3600)


def test_filecache_rejects_invalid_configuration(cache_file: Path) -> None:
    """FileCache initialisation validates max_items and default_ttl."""
    with pytest.raises(CacheError):
        FileCache(cache_file, max_items=0, default_ttl=3600)

    with pytest.raises(CacheError):
        FileCache(cache_file, max_items=10, default_ttl=-1)

    with pytest.raises(CacheError):
        FileCache(cache_file, max_items="invalid", default_ttl=3600)

    with pytest.raises(CacheError):
        FileCache(cache_file, max_items=10, default_ttl="invalid")


def test_filecache_rejects_invalid_ttl(cache: FileCache) -> None:
    """FileCache set validates TTL input type."""
    with pytest.raises(CacheError):
        cache.set("key", "value", ttl="invalid")


def test_filecache_expiry(tmp_path, monkeypatch):
    """FileCache expires entries after TTL without real sleep."""
    p = tmp_path / "cache.json"
    c = FileCache(str(p), max_items=10, default_ttl=1)

    start = 1_000_000.0
    monkeypatch.setattr("time.time", lambda: start)
    c.set("k", "v", ttl=1)
    assert c.get("k") == "v"
    monkeypatch.setattr("time.time", lambda: start + 1.1)
    assert c.get("k") is None


def test_filecache_initialisation(cache_file: Path) -> None:
    """FileCache initialises with correct parameters."""
    cache = FileCache(cache_file, max_items=100, default_ttl=7200)

    assert cache.path == cache_file
    assert cache.max_items == 100
    assert cache.default_ttl == 7200
    assert cache._loaded is False


def test_filecache_set_and_get(cache: FileCache) -> None:
    """FileCache stores and retrieves values correctly."""
    cache.set("key1", "value1")

    result = cache.get("key1")

    assert result == "value1"


def test_filecache_get_empty_key(cache: FileCache) -> None:
    """FileCache raises CacheError for empty key retrieval."""
    with pytest.raises(CacheError):
        cache.get("")


def test_filecache_set_empty_key(cache: FileCache) -> None:
    """FileCache raises CacheError for empty key storage."""
    with pytest.raises(CacheError):
        cache.set("", "value")


def test_filecache_get_missing_key(cache: FileCache, caplog) -> None:
    """FileCache returns None and logs debug for missing key."""
    caplog.set_level(logging.DEBUG)

    result = cache.get("nonexistent")

    assert result is None
    assert "Cache miss" in caplog.text


def test_filecache_set_with_custom_ttl(cache: FileCache, monkeypatch) -> None:
    """FileCache respects custom TTL parameter."""
    start = 2_000_000.0
    monkeypatch.setattr("time.time", lambda: start)

    cache.set("key", "value", ttl=5)

    monkeypatch.setattr("time.time", lambda: start + 3)
    assert cache.get("key") == "value"

    monkeypatch.setattr("time.time", lambda: start + 6)
    assert cache.get("key") is None


def test_filecache_set_with_zero_ttl(cache: FileCache, monkeypatch) -> None:
    """FileCache does not expire entries with zero or negative TTL."""
    start = 3_000_000.0
    monkeypatch.setattr("time.time", lambda: start)

    cache.set("key", "value", ttl=0)

    monkeypatch.setattr("time.time", lambda: start + 10000)
    assert cache.get("key") == "value"


def test_filecache_persistence(cache_file: Path) -> None:
    """FileCache persists data to disk and loads on next instance."""
    cache1 = FileCache(cache_file, max_items=10, default_ttl=3600)
    cache1.set("persisted_key", "persisted_value")

    cache2 = FileCache(cache_file, max_items=10, default_ttl=3600)
    result = cache2.get("persisted_key")

    assert result == "persisted_value"


def test_filecache_load_nonexistent_file(cache: FileCache, caplog) -> None:
    """FileCache initialises empty data when cache file does not exist."""
    caplog.set_level(logging.DEBUG)

    cache._load()

    assert cache._data == {}
    assert "Cache file does not exist" in caplog.text


def test_filecache_load_malformed_json(cache_file: Path, caplog) -> None:
    """FileCache raises CacheError when cache file JSON is malformed."""
    cache_file.write_text("invalid json content")

    caplog.set_level(logging.ERROR)
    cache = FileCache(cache_file)

    with pytest.raises(CacheError):
        cache._load()
    assert "malformed JSON" in caplog.text


def test_filecache_load_os_error(cache_file: Path, caplog) -> None:
    """FileCache raises CacheError on OS errors during load."""
    cache_file.write_text("{}")

    caplog.set_level(logging.ERROR)

    with patch("pathlib.Path.open", side_effect=OSError("Permission denied")):
        cache = FileCache(cache_file)

        with pytest.raises(CacheError):
            cache._load()

    assert "Failed to read cache file" in caplog.text


def test_filecache_persist_creates_parent_directory(tmp_path: Path) -> None:
    """FileCache creates parent directories when persisting."""
    nested_cache = tmp_path / "nested" / "deep" / "cache.json"
    cache = FileCache(nested_cache)

    cache.set("key", "value")

    assert nested_cache.exists()
    assert nested_cache.parent.exists()


def test_filecache_persist_os_error(cache: FileCache, caplog) -> None:
    """FileCache raises CacheError on persist OS error."""
    cache.set("key", "value")

    caplog.set_level(logging.ERROR)

    with patch("pathlib.Path.open", side_effect=OSError("Disk full")):
        with pytest.raises(CacheError):
            cache._persist()

    assert "Failed to write cache file" in caplog.text


def test_filecache_persist_type_error(cache: FileCache, caplog) -> None:
    """FileCache raises CacheError on JSON serialisation error."""
    cache._data = {"key": {"value": object(), "expires_at": None}}

    caplog.set_level(logging.ERROR)

    with pytest.raises(CacheError):
        cache._persist()

    assert "Failed to serialise cache data" in caplog.text


def test_filecache_eviction_when_max_items_exceeded(cache: FileCache, monkeypatch) -> None:
    """FileCache evicts oldest entries when max_items is exceeded."""
    start = 4_000_000.0
    monkeypatch.setattr("time.time", lambda: start)

    for i in range(15):
        cache.set(f"key{i}", f"value{i}", ttl=100 + i)

    assert len(cache._data) == 10


def test_filecache_eviction_logs_debug(cache: FileCache, monkeypatch, caplog) -> None:
    """FileCache logs eviction events."""
    start = 5_000_000.0
    monkeypatch.setattr("time.time", lambda: start)

    caplog.set_level(logging.DEBUG)

    for i in range(12):
        cache.set(f"key{i}", f"value{i}")

    assert "Evicted" in caplog.text


def test_filecache_clear_removes_data_and_file(cache_file: Path) -> None:
    """FileCache clear removes all data and deletes cache file."""
    cache = FileCache(cache_file)
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    cache.clear()

    assert len(cache._data) == 0
    assert not cache_file.exists()


def test_filecache_clear_logs_info(cache: FileCache, caplog) -> None:
    """FileCache clear logs info message."""
    cache.set("key", "value")

    caplog.set_level(logging.INFO)
    cache.clear()

    assert "Cleared cache file" in caplog.text


def test_filecache_clear_handles_os_error(cache_file: Path, caplog) -> None:
    """FileCache clear raises CacheError when file deletion fails."""
    cache = FileCache(cache_file)
    cache.set("key", "value")

    caplog.set_level(logging.ERROR)

    with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
        with pytest.raises(CacheError):
            cache.clear()

    assert "Failed to delete cache file" in caplog.text


def test_filecache_clear_when_file_does_not_exist(cache: FileCache) -> None:
    """FileCache clear succeeds when file does not exist."""
    cache.set("key", "value")
    cache._data = {}

    cache.clear()

    assert len(cache._data) == 0


def test_filecache_get_logs_cache_hit(cache: FileCache, caplog) -> None:
    """FileCache get logs debug on cache hit."""
    cache.set("key", "value")

    caplog.set_level(logging.DEBUG)
    cache.get("key")

    assert "Cache hit" in caplog.text


def test_filecache_get_expired_entry_logs_debug(cache: FileCache, monkeypatch, caplog) -> None:
    """FileCache get logs debug when entry has expired."""
    start = 6_000_000.0
    monkeypatch.setattr("time.time", lambda: start)
    cache.set("key", "value", ttl=10)

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("time.time", lambda: start + 20)
    cache.get("key")

    assert "Cache entry expired" in caplog.text


def test_filecache_set_logs_debug(cache: FileCache, caplog) -> None:
    """FileCache set logs debug message."""
    caplog.set_level(logging.DEBUG)

    cache.set("key", "value")

    assert "Set cache key" in caplog.text


def test_filecache_multiple_loads_are_idempotent(cache: FileCache) -> None:
    """FileCache load is idempotent and only loads once."""
    cache.set("key", "value")

    cache._load()
    cache._load()
    cache._load()

    assert cache._loaded is True


def test_filecache_accepts_path_object(tmp_path: Path) -> None:
    """FileCache accepts Path object for cache file."""
    cache_path = tmp_path / "path_cache.json"
    cache = FileCache(cache_path)

    cache.set("key", "value")

    assert cache_path.exists()


def test_filecache_expires_and_deletes_key(cache: FileCache, monkeypatch) -> None:
    """FileCache deletes expired entry from internal data."""
    start = 7_000_000.0
    monkeypatch.setattr("time.time", lambda: start)
    cache.set("expired_key", "value", ttl=5)

    monkeypatch.setattr("time.time", lambda: start + 10)
    cache.get("expired_key")

    assert "expired_key" not in cache._data


@patch.dict("os.environ", {"GITHUB_TOKEN": "fake"})
@patch("requests.Session.post")
def test_get_graphql_raises_github_client_error_on_http_500(mock_post):
    """GraphQL request raises GitHubClientError on HTTP 500 responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "error"
    mock_post.return_value = mock_resp

    client = GitHubClient()
    with pytest.raises(GitHubClientError):
        client._get_graphql("query { repository { name } }")
