"""Simple file-backed cache with TTL and size limit.

Provides basic caching functionality for CLI applications. Not designed for
heavy concurrent use but suitable for simple single-process scenarios.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.exceptions import CacheError

logger = logging.getLogger(__name__)


class FileCache:
    """File-backed cache with automatic expiry and size limits.

    Implements a simple LRU-like eviction strategy based on expiry timestamps.
    Cache data is persisted to disk as JSON for durability across process restarts.

    Not thread-safe. Suitable for single-process CLI applications.
    """

    def __init__(self, path: str | Path, max_items: int = 1000, default_ttl: int = 3600) -> None:
        """Initialise the file cache.

        Args:
            path: Path to the cache file on disk.
            max_items: Maximum number of items to store before eviction (default 1000).
            default_ttl: Default time-to-live in seconds for cached items (default 3600).

        Raises:
            CacheError: If initial configuration is invalid.
        """
        if not path:
            raise CacheError("Cache path cannot be empty")

        try:
            max_items_value = int(max_items)
        except (TypeError, ValueError) as exc:
            raise CacheError("max_items must be a positive integer") from exc
        if max_items_value <= 0:
            raise CacheError("max_items must be a positive integer")

        try:
            default_ttl_value = int(default_ttl)
        except (TypeError, ValueError) as exc:
            raise CacheError("default_ttl must be a non-negative integer") from exc
        if default_ttl_value < 0:
            raise CacheError("default_ttl cannot be negative")

        self.path = Path(path)
        self.max_items = max_items_value
        self.default_ttl = default_ttl_value
        self._data: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        """Load cache data from disk into memory.

        Idempotent operation. Repeated calls after the first successful load do nothing.

        Raises:
            CacheError: If the cache file cannot be read or parsed.
        """
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            logger.debug("Cache file does not exist: %s", self.path)
            self._data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                self._data = json.load(fh)
            logger.debug("Loaded cache from %s (%d items)", self.path, len(self._data))
        except json.JSONDecodeError as exc:
            self._loaded = False
            logger.exception("Cache file %s is malformed JSON", self.path)
            raise CacheError(f"Cache file {self.path} is not valid JSON") from exc
        except OSError as exc:
            self._loaded = False
            logger.exception("Failed to read cache file %s", self.path)
            raise CacheError(f"Unable to read cache file {self.path}") from exc

    def _persist(self) -> None:
        """Persist the current cache state to disk.

        Raises:
            CacheError: If the cache cannot be written or serialised.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh)
            logger.debug("Persisted cache to %s (%d items)", self.path, len(self._data))
        except OSError as exc:
            logger.exception("Failed to write cache file %s", self.path)
            raise CacheError(f"Unable to write cache file {self.path}") from exc
        except TypeError as exc:
            logger.exception("Failed to serialise cache data for %s", self.path)
            raise CacheError(f"Cache data for {self.path} is not JSON serialisable") from exc

    def get(self, key: str) -> Any | None:
        """Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve. Must be non-empty.

        Returns:
            Cached value if present and not expired, None otherwise.

        Raises:
            CacheError: If the key is empty or cache data cannot be loaded.
        """
        if not key:
            raise CacheError("Cache key must be provided")
        self._load()
        entry = self._data.get(key)
        if not entry:
            logger.debug("Cache miss for key: %s", key)
            return None
        expires_at = entry.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            logger.debug("Cache entry expired for key: %s", key)
            try:
                del self._data[key]
            except KeyError:
                pass
            self._persist()
            return None
        logger.debug("Cache hit for key: %s", key)
        return entry.get("value")

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache with optional TTL.

        Performs simple eviction by removing oldest entries when max_items is exceeded.

        Args:
            key: Cache key. Must be non-empty.
            value: Value to cache (must be JSON-serializable).
            ttl: Optional time-to-live in seconds. If None, uses default_ttl.
                 Non-positive values disable expiry.

        Raises:
            CacheError: If the key is empty, TTL is invalid, or persistence fails.
        """
        if not key:
            raise CacheError("Cache key must be provided")
        self._load()
        try:
            ttl_value = int(ttl) if ttl is not None else self.default_ttl
        except (TypeError, ValueError) as exc:
            raise CacheError("ttl must be an integer or None") from exc

        expires_at = int(time.time() + ttl_value) if ttl_value > 0 else None
        self._data[key] = {"value": value, "expires_at": expires_at}
        logger.debug("Set cache key: %s (ttl=%s)", key, ttl_value if ttl_value > 0 else "never")
        self._evict_if_needed()
        self._persist()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max_items.

        Eviction is based on expiry timestamps. Entries with earliest expiry
        are removed first.
        """
        if len(self._data) <= self.max_items:
            return
        items = list(self._data.items())
        items.sort(key=lambda kv: kv[1].get("expires_at") or 0)
        evict_count = len(self._data) - self.max_items
        for k, _ in items[:evict_count]:
            try:
                del self._data[k]
            except KeyError:
                pass
        logger.debug("Evicted %d cache entries (max_items=%d)", evict_count, self.max_items)

    def clear(self) -> None:
        """Clear all cached data from memory and disk.

        Removes the cache file if it exists.

        Raises:
            CacheError: If the cache file cannot be deleted.
        """
        item_count = len(self._data)
        self._data = {}
        try:
            if self.path.exists():
                self.path.unlink()
                logger.info("Cleared cache file %s (%d items removed)", self.path, item_count)
        except OSError as exc:
            logger.exception("Failed to delete cache file %s", self.path)
            raise CacheError(f"Unable to delete cache file {self.path}") from exc
