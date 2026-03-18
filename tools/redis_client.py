"""tools/redis_client.py
Redis client wrapper with in-memory fallback.

When REDIS_URL is set, operations go to Redis.
When REDIS_URL is unset, ExactCache falls back to a local dict
so the app works without Redis in development.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from core.settings import get_settings
from observability.metrics import CACHE_TOOL_HITS, CACHE_TOOL_MISSES, REDIS_UP

logger = logging.getLogger(__name__)

_redis_client: Any = None
_redis_initialised = False
_local_json: dict[str, str] = {}


def get_redis():
    """Return a Redis client, or None if Redis is not configured/available."""
    global _redis_client, _redis_initialised
    if _redis_initialised:
        return _redis_client

    _redis_initialised = True
    settings = get_settings()
    if not settings.redis_url:
        logger.info("REDIS_URL not set — using in-memory cache fallback")
        return None

    try:
        import redis as redis_lib
        _redis_client = redis_lib.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        _redis_client.ping()
        REDIS_UP.set(1)
        logger.info("Redis connected: %s", settings.redis_url)
    except Exception:
        logger.warning("Redis connection failed — falling back to in-memory cache", exc_info=True)
        _redis_client = None
        REDIS_UP.set(0)

    return _redis_client


def _make_key(prefix: str, raw_key: str) -> str:
    """Build a namespaced Redis key: prefix:sha256(raw_key)[:16]."""
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def get_json(key: str, default: Any = None) -> Any:
    """Fetch JSON payload by explicit key, with in-memory fallback."""
    redis_client = get_redis()
    if redis_client is not None:
        try:
            value = redis_client.get(key)
            if value is None:
                return default
            return json.loads(value)
        except Exception:
            logger.warning("Redis JSON GET failed, falling back to local", exc_info=True)

    value = _local_json.get(key)
    if value is None:
        return default
    return json.loads(value)


def set_json(key: str, payload: Any, ttl: int | None = None) -> None:
    """Store JSON payload by explicit key, with in-memory fallback."""
    value = json.dumps(payload)
    redis_client = get_redis()
    if redis_client is not None:
        try:
            redis_client.set(key, value, ex=ttl)
            return
        except Exception:
            logger.warning("Redis JSON SET failed, falling back to local", exc_info=True)
    _local_json[key] = value


def append_json_list(key: str, item: Any, max_length: int, ttl: int | None = None) -> list[Any]:
    """Append an item to a JSON list value, trimming to max_length."""
    items = list(get_json(key, default=[]))
    items.append(item)
    if max_length > 0:
        items = items[-max_length:]
    else:
        items = []
    set_json(key, items, ttl=ttl)
    return items


class ExactCache:
    """Exact-match key-value cache backed by Redis, with dict fallback.

    Used for caching deterministic operations (API calls, embeddings)
    where the same input always produces the same output.
    """

    def __init__(self, prefix: str, ttl: int | None = None):
        self.prefix = prefix
        settings = get_settings()
        self.ttl = ttl if ttl is not None else settings.redis_cache_ttl
        self._redis = get_redis()
        # In-memory fallback when Redis is unavailable
        self._local: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        full_key = _make_key(self.prefix, key)
        if self._redis is not None:
            try:
                value = self._redis.get(full_key)
                if value is not None:
                    CACHE_TOOL_HITS.labels(tool=self.prefix).inc()
                    return value
                CACHE_TOOL_MISSES.labels(tool=self.prefix).inc()
                return None
            except Exception:
                logger.warning("Redis GET failed, falling back to local", exc_info=True)
        value = self._local.get(full_key)
        if value is not None:
            CACHE_TOOL_HITS.labels(tool=self.prefix).inc()
            return value
        CACHE_TOOL_MISSES.labels(tool=self.prefix).inc()
        return None

    def set(self, key: str, value: str) -> None:
        full_key = _make_key(self.prefix, key)
        if self._redis is not None:
            try:
                self._redis.set(full_key, value, ex=self.ttl)
                return
            except Exception:
                logger.warning("Redis SET failed, falling back to local", exc_info=True)
        self._local[full_key] = value

    def delete(self, key: str) -> None:
        full_key = _make_key(self.prefix, key)
        if self._redis is not None:
            try:
                self._redis.delete(full_key)
                return
            except Exception:
                pass
        self._local.pop(full_key, None)
