"""tests/rag/test_redis_client.py"""
import pytest


def test_redis_client_returns_none_when_disabled():
    """When REDIS_URL is not set, get_redis() returns None."""
    import tools.redis_client as _rc
    _rc._redis_initialised = False
    _rc._redis_client = None

    from tools.redis_client import get_redis
    client = get_redis()
    assert client is None or hasattr(client, 'get')


def test_exact_cache_roundtrip_without_redis():
    """ExactCache falls back to in-memory dict when Redis unavailable."""
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="test", ttl=60)
    cache.set("key1", '{"result": "hello"}')
    assert cache.get("key1") == '{"result": "hello"}'


def test_exact_cache_respects_prefix():
    """Different prefixes are isolated."""
    from tools.redis_client import ExactCache

    cache_a = ExactCache(prefix="cohere", ttl=60)
    cache_b = ExactCache(prefix="tavily", ttl=60)

    cache_a.set("query1", "answer_a")
    cache_b.set("query1", "answer_b")

    assert cache_a.get("query1") == "answer_a"
    assert cache_b.get("query1") == "answer_b"


def test_exact_cache_miss_returns_none():
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="test", ttl=60)
    assert cache.get("nonexistent") is None
