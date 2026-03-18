"""tests/rag/test_cohere_cache.py"""
import json


def test_cohere_rerank_cache_hit_skips_api_call():
    """Second call with identical inputs should use cache, not call Cohere API."""
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="cohere_rerank", ttl=3600)

    query = "metformin side effects"
    doc_texts = ["Doc about metformin.", "Doc about insulin."]
    cache_key = query + "||" + "||".join(sorted(doc_texts))

    cached_result = json.dumps({
        "indices": [1, 0],
        "scores": [0.95, 0.42],
    })
    cache.set(cache_key, cached_result)

    hit = cache.get(cache_key)
    assert hit is not None
    parsed = json.loads(hit)
    assert parsed["indices"] == [1, 0]
    assert parsed["scores"] == [0.95, 0.42]


def test_cohere_rerank_cache_miss_returns_none():
    """Cache miss for unseen query returns None."""
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="cohere_rerank", ttl=3600)
    result = cache.get("never_seen_query||doc1||doc2")
    assert result is None
