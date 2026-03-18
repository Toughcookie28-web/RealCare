"""tests/rag/test_tavily_cache.py"""
import json


def test_tavily_cache_stores_and_retrieves():
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="tavily", ttl=86400)

    query = "latest metformin research"
    results = [
        {"url": "https://nih.gov/1", "title": "Metformin Study", "content": "A study about metformin..."},
        {"url": "https://who.int/2", "title": "WHO Guidelines", "content": "Guidelines on metformin..."},
    ]

    cache.set(query, json.dumps(results))
    cached = cache.get(query)
    assert cached is not None
    parsed = json.loads(cached)
    assert len(parsed) == 2
    assert parsed[0]["url"] == "https://nih.gov/1"
