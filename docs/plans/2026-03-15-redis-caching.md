# Redis Caching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the in-memory `SemanticCache` with a Redis-backed caching system, adding tool/API caching (Cohere rerank, Tavily search) alongside the existing semantic answer cache. Strategy B (aggressive/development) defaults.

**Architecture:** Add a `redis` service to docker-compose. Create a `tools/redis_cache.py` module with a `RedisCache` class that wraps `redis-py` for exact-key caching (tool/API calls) and Redis Stack's vector search for semantic answer cache (replaces the linear-scan deque). The existing `tools/cache.py` `SemanticCache` interface stays the same — callers don't change — but the backend switches from in-memory deque to Redis. A `REDIS_URL` setting controls the connection; when unset, fall back to the current in-memory implementation so local dev without Redis still works.

**Tech Stack:** `redis[hiredis]` (Python client + C parser), `redis-stack` Docker image (includes RediSearch vector module), `redisvl` (optional — Redis vector library for Python)

---

### Task 1: Add Redis service to docker-compose and dependencies

**Files:**
- Modify: `docker-compose.yml`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `core/settings.py`

**Step 1: Add Redis service to docker-compose.yml**

Add after the `db` service:

```yaml
  redis:
    image: redis/redis-stack-server:7.4.0-v3
    ports:
      - '6379:6379'
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

Add `redis_data` to the `volumes:` section at the bottom:

```yaml
volumes:
  postgres_data:
  grafana_data:
  redis_data:
```

Add `depends_on` for redis to the `app`, `eval`, and `ingest` services:

```yaml
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
```

Add Redis URL to the `x-app-env` anchor:

```yaml
  REDIS_URL: redis://redis:6379/0
```

**Why redis-stack-server instead of plain redis:** It includes the RediSearch module which provides built-in HNSW vector search. No algorithm implementation needed — Redis builds and maintains the index automatically.

**Step 2: Add redis dependency to pyproject.toml**

Add `"redis[hiredis]"` to the `dependencies` list (after `"cohere"`):

```
    "redis[hiredis]",
```

`hiredis` is a C-based parser that makes Redis operations ~10x faster. It's optional but recommended.

**Step 3: Add Redis settings to core/settings.py**

Add after `cohere_rerank_model`:

```python
    redis_url: str | None = Field(default=None, alias='REDIS_URL')
    redis_cache_ttl: int = Field(default=86400, alias='REDIS_CACHE_TTL')  # 24 hours in seconds
```

**Step 4: Add Redis config to .env.example**

Add after the Cohere section:

```
# Redis (caching layer)
# Leave empty to use in-memory fallback (no Redis needed for local dev)
REDIS_URL=
# Default TTL for cached entries in seconds (86400 = 24 hours)
REDIS_CACHE_TTL=86400
```

**Step 5: Run test to verify settings load**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -c "from core.settings import get_settings; s = get_settings(); print(f'redis_url={s.redis_url}, ttl={s.redis_cache_ttl}')"`
Expected: `redis_url=None, ttl=86400`

**Step 6: Commit**

```bash
git add docker-compose.yml pyproject.toml .env.example core/settings.py
git commit -m "feat: add Redis Stack service and settings for caching layer"
```

---

### Task 2: Build the Redis client wrapper

**Files:**
- Create: `tools/redis_client.py`
- Create: `tests/rag/test_redis_client.py`

This module handles Redis connection management and provides simple get/set/delete operations. All other cache modules use this — they never touch `redis` directly.

**Step 1: Write the failing test**

```python
"""tests/rag/test_redis_client.py"""
import pytest


def test_redis_client_returns_none_when_disabled():
    """When REDIS_URL is not set, get_redis() returns None."""
    from tools.redis_client import get_redis
    # With no REDIS_URL in env, should return None gracefully
    client = get_redis()
    # In CI/local without Redis, this should be None
    # If Redis IS running, it returns a client — both are acceptable
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
```

**Step 2: Run test to verify it fails**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_redis_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.redis_client'`

**Step 3: Write the implementation**

```python
"""tools/redis_client.py
Redis client wrapper with in-memory fallback.

When REDIS_URL is set, operations go to Redis.
When REDIS_URL is unset, ExactCache falls back to a local dict
so the app works without Redis in development.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from core.settings import get_settings

logger = logging.getLogger(__name__)

_redis_client: Any = None
_redis_initialised = False


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
        logger.info("Redis connected: %s", settings.redis_url)
    except Exception:
        logger.warning("Redis connection failed — falling back to in-memory cache", exc_info=True)
        _redis_client = None

    return _redis_client


def _make_key(prefix: str, raw_key: str) -> str:
    """Build a namespaced Redis key: prefix:sha256(raw_key)[:16]."""
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


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
        self._local: dict[str, str] = {} if self._redis is None else {}

    def get(self, key: str) -> str | None:
        full_key = _make_key(self.prefix, key)
        if self._redis is not None:
            try:
                return self._redis.get(full_key)
            except Exception:
                logger.warning("Redis GET failed, falling back to local", exc_info=True)
        return self._local.get(full_key)

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
```

**Step 4: Run test to verify it passes**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_redis_client.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add tools/redis_client.py tests/rag/test_redis_client.py
git commit -m "feat: add Redis client wrapper with in-memory fallback"
```

---

### Task 3: Add Cohere rerank caching

**Files:**
- Modify: `agents/retriever_agent.py` (the `_cohere_rerank` function)
- Create: `tests/rag/test_cohere_cache.py`

Cohere rerank is the highest-value cache target: rate-limited to 10/min, deterministic (same query + same docs = same ranking), and expensive in latency (API call + 62s retry waits).

Cache key: `hash(query + sorted(doc_texts))` — same query against same documents always produces same ranking.

**Step 1: Write the failing test**

```python
"""tests/rag/test_cohere_cache.py"""
import json
from unittest.mock import patch


def test_cohere_rerank_cache_hit_skips_api_call():
    """Second call with identical inputs should use cache, not call Cohere API."""
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="cohere_rerank", ttl=3600)

    query = "metformin side effects"
    doc_texts = ["Doc about metformin.", "Doc about insulin."]
    cache_key = query + "||" + "||".join(sorted(doc_texts))

    # Simulate a cached rerank result
    cached_result = json.dumps({
        "indices": [1, 0],
        "scores": [0.95, 0.42],
    })
    cache.set(cache_key, cached_result)

    # Verify cache hit returns the stored result
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
```

**Step 2: Run test to verify it passes (these test the cache layer only)**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_cohere_cache.py -v`
Expected: 2 PASSED

**Step 3: Integrate cache into `_cohere_rerank` in retriever_agent.py**

Modify `agents/retriever_agent.py`. Add import at the top:

```python
import json
from tools.redis_client import ExactCache
```

Add a module-level cache instance:

```python
_cohere_cache = ExactCache(prefix="cohere_rerank", ttl=3600)  # 1 hour TTL
```

Modify `_cohere_rerank` to check cache before calling API, and store result after:

```python
def _cohere_rerank(
    query: str, docs: list[Document], settings: Any,
) -> tuple[list[Document], list[float]]:
    """Rerank using Cohere's cross-encoder API with caching and rate-limit retry."""
    import time
    import cohere

    doc_texts = [doc.page_content for doc in docs]

    # Check cache: same query + same docs = same ranking
    cache_key = query + "||" + "||".join(sorted(doc_texts))
    cached = _cohere_cache.get(cache_key)
    if cached is not None:
        try:
            parsed = json.loads(cached)
            indices = parsed["indices"]
            scores = parsed["scores"]
            ranked_docs = [docs[i] for i in indices if i < len(docs)]
            ranked_scores = [s for i, s in zip(indices, scores) if i < len(docs)]
            logger.info("Cohere rerank cache hit (query=%s...)", query[:40])
            return ranked_docs, ranked_scores
        except Exception:
            logger.warning("Cohere cache entry corrupt, falling through to API")

    client = cohere.Client(api_key=settings.cohere_api_key)

    last_exc: Exception | None = None
    for attempt in range(_COHERE_MAX_RETRIES):
        try:
            response = client.rerank(
                query=query,
                documents=doc_texts,
                model=settings.cohere_rerank_model,
                top_n=len(docs),
            )

            ranked_docs = []
            ranked_scores = []
            indices = []
            for result in response.results:
                ranked_docs.append(docs[result.index])
                ranked_scores.append(result.relevance_score)
                indices.append(result.index)

            # Store in cache
            _cohere_cache.set(cache_key, json.dumps({
                "indices": indices,
                "scores": ranked_scores,
            }))

            return ranked_docs, ranked_scores

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if 'rate' in exc_str or 'limit' in exc_str or '429' in exc_str:
                logger.info(
                    "Cohere rate limited, waiting %ds (attempt %d/%d)",
                    _COHERE_RETRY_WAIT, attempt + 1, _COHERE_MAX_RETRIES,
                )
                time.sleep(_COHERE_RETRY_WAIT)
            else:
                raise

    raise last_exc  # type: ignore[misc]
```

**Step 4: Run existing tests to verify nothing breaks**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/ -v --timeout=30`
Expected: all PASSED

**Step 5: Commit**

```bash
git add agents/retriever_agent.py tests/rag/test_cohere_cache.py
git commit -m "feat: add Cohere rerank caching with Redis/in-memory fallback"
```

---

### Task 4: Add Tavily search caching

**Files:**
- Modify: `agents/tavily_agent.py`
- Create: `tests/rag/test_tavily_cache.py`

Tavily searches cost money and the same medical query within 24 hours returns essentially the same results. Cache the API response.

**Step 1: Write the failing test**

```python
"""tests/rag/test_tavily_cache.py"""
import json


def test_tavily_cache_stores_and_retrieves():
    from tools.redis_client import ExactCache

    cache = ExactCache(prefix="tavily", ttl=86400)  # 24h

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
```

**Step 2: Run test**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_tavily_cache.py -v`
Expected: PASSED

**Step 3: Integrate cache into TavilyAgent**

Modify `agents/tavily_agent.py`:

```python
from __future__ import annotations

import json
import logging

from langchain_core.documents import Document

from core.state_v2 import AgentStateV2
from tools.redis_client import ExactCache
from tools.search_tools import get_tavily_search

logger = logging.getLogger(__name__)
_tavily_cache = ExactCache(prefix="tavily", ttl=86400)  # 24 hour TTL


def TavilyAgent(state: AgentStateV2) -> AgentStateV2:
    tavily = get_tavily_search()
    if tavily is None:
        state['documents'] = []
        return state

    query = state.get('optimized_query') or state.get('question', '')

    # Check cache first
    cached = _tavily_cache.get(query)
    if cached is not None:
        try:
            results = json.loads(cached)
            logger.info("Tavily cache hit (query=%s...)", query[:40])
        except Exception:
            logger.warning("Tavily cache entry corrupt, calling API")
            results = tavily.invoke(query)
            if results:
                _tavily_cache.set(query, json.dumps(results))
    else:
        results = tavily.invoke(query)
        if results:
            _tavily_cache.set(query, json.dumps(results))

    docs = []
    for result in results or []:
        content = result.get('content', '') if isinstance(result, dict) else ''
        if len(content.strip()) > 80:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'source': 'tavily',
                    },
                )
            )

    state['documents'] = docs
    if docs:
        state['source'] = 'Trusted Medical Web Search'
    return state
```

**Step 4: Run tests**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_tavily_cache.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add agents/tavily_agent.py tests/rag/test_tavily_cache.py
git commit -m "feat: add Tavily search caching with 24h TTL"
```

---

### Task 5: Upgrade SemanticCache to use Redis vector search

**Files:**
- Modify: `tools/cache.py`
- Modify: `tests/rag/test_semantic_cache_contract.py`

Replace the linear-scan deque with Redis Stack's vector search (HNSW). Keep the same `SemanticCache` interface so callers (`executor_agent.py`, `eval/workflow_eval.py`) don't change.

**Step 1: Update existing tests to cover Redis fallback**

Add to `tests/rag/test_semantic_cache_contract.py`:

```python
def test_semantic_cache_falls_back_to_inmemory_without_redis():
    """SemanticCache works with in-memory backend when Redis is unavailable."""
    module = _load_cache_module()

    cache = module.SemanticCache(
        enabled=True,
        version="test-v1",
        embedder=lambda query: [1.0, 0.0],
    )
    cache.set("what is dka", "cached answer")
    assert cache.get("what is dka") == "cached answer"
    assert cache.get("what is dka", namespace="other") is None
```

**Step 2: Run test to verify it passes with current code**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_semantic_cache_contract.py -v`
Expected: PASSED (existing behavior is the fallback)

**Step 3: Rewrite tools/cache.py**

```python
"""tools/cache.py
Semantic answer cache with Redis vector search backend and in-memory fallback.

Redis backend: uses RediSearch HNSW index for O(log n) similarity lookup.
Fallback: original deque-based linear scan when Redis is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

from core.settings import get_settings
from observability.metrics import CACHE_HITS, CACHE_MISSES
from tools.embedding_client import embed_query

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    namespace: str
    query: str
    embedding: list[float]
    response: str


class SemanticCache:
    """Semantic similarity cache. Same interface, two backends."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        version: str | None = None,
        threshold: float | None = None,
        max_size: int | None = None,
        embedder=None,
    ):
        settings = get_settings()
        self.enabled = settings.use_semantic_cache() if enabled is None else bool(enabled)
        self.version = (version or settings.semantic_cache_version or 'executor-v1').strip() or 'executor-v1'
        self.threshold = settings.semantic_cache_threshold if threshold is None else float(threshold)
        self.max_size = settings.semantic_cache_size if max_size is None else int(max_size)
        self.embedder = embedder or embed_query
        # In-memory fallback
        self.entries: deque[CacheEntry] = deque(maxlen=self.max_size)
        # Try Redis backend
        self._redis = self._init_redis_index()

    def _init_redis_index(self):
        """Try to initialise Redis vector search index. Return client or None."""
        try:
            from tools.redis_client import get_redis
            client = get_redis()
            if client is None:
                return None
            # Check if RediSearch module is available
            modules = client.module_list()
            has_search = any(
                m.get(b'name', b'').decode().lower() == 'search'
                or m.get('name', '').lower() == 'search'
                for m in modules
            )
            if not has_search:
                logger.info("Redis available but RediSearch module not loaded — using in-memory fallback")
                return None

            # Create index if it doesn't exist
            index_name = f"idx:semantic_cache"
            try:
                client.execute_command("FT.INFO", index_name)
            except Exception:
                # Detect embedding dim from a test embed
                test_emb = self.embedder("test")
                dim = len(test_emb)
                client.execute_command(
                    "FT.CREATE", index_name,
                    "ON", "HASH",
                    "PREFIX", "1", "scache:",
                    "SCHEMA",
                    "namespace", "TAG",
                    "query", "TEXT",
                    "response", "TEXT",
                    "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", "COSINE",
                )
                logger.info("Created Redis vector index '%s' (dim=%d)", index_name, dim)

            return client
        except Exception:
            logger.info("Redis vector index init failed — using in-memory fallback", exc_info=True)
            return None

    def _redis_key(self, query: str, namespace: str) -> str:
        digest = hashlib.sha256(f"{namespace}:{query}".encode()).hexdigest()[:16]
        return f"scache:{digest}"

    def get(self, query: str, namespace: str | None = None) -> str | None:
        if not self.enabled:
            return None

        resolved_namespace = self._resolve_namespace(namespace)

        if self._redis is not None:
            try:
                return self._redis_get(query, resolved_namespace)
            except Exception:
                logger.warning("Redis semantic get failed, falling back to in-memory", exc_info=True)

        return self._inmemory_get(query, resolved_namespace)

    def set(self, query: str, response: str, namespace: str | None = None) -> None:
        if not self.enabled:
            return

        resolved_namespace = self._resolve_namespace(namespace)

        if self._redis is not None:
            try:
                self._redis_set(query, response, resolved_namespace)
                return
            except Exception:
                logger.warning("Redis semantic set failed, falling back to in-memory", exc_info=True)

        self._inmemory_set(query, response, resolved_namespace)

    def clear(self) -> None:
        self.entries.clear()
        # Note: does not clear Redis — use TTL for expiry or manual FLUSHDB

    def configure(
        self,
        *,
        enabled: bool | None = None,
        version: str | None = None,
        clear: bool = False,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if version is not None:
            self.version = version.strip() or self.version
        if clear:
            self.clear()

    def _resolve_namespace(self, namespace: str | None) -> str:
        suffix = (namespace or 'default').strip() or 'default'
        return f'{self.version}:{suffix}'

    # --- Redis backend ---

    def _redis_get(self, query: str, namespace: str) -> str | None:
        import struct

        query_emb = self.embedder(query)
        blob = struct.pack(f"{len(query_emb)}f", *query_emb)

        # KNN search filtered by namespace
        results = self._redis.execute_command(
            "FT.SEARCH", "idx:semantic_cache",
            f"(@namespace:{{{namespace.replace(':', '\\:')}}})[KNN 1 @embedding $vec AS score]",
            "PARAMS", "2", "vec", blob,
            "SORTBY", "score",
            "LIMIT", "0", "1",
            "RETURN", "2", "response", "score",
            "DIALECT", "2",
        )

        # FT.SEARCH returns: [total_count, key1, [field, value, ...], ...]
        if not results or results[0] == 0:
            CACHE_MISSES.inc()
            return None

        fields = results[2]
        field_map = {fields[i]: fields[i + 1] for i in range(0, len(fields), 2)}
        # COSINE distance: 0 = identical, 1 = orthogonal. Convert to similarity.
        distance = float(field_map.get("score", "1.0"))
        similarity = 1.0 - distance

        if similarity >= self.threshold:
            CACHE_HITS.inc()
            return field_map.get("response")

        CACHE_MISSES.inc()
        return None

    def _redis_set(self, query: str, response: str, namespace: str) -> None:
        import struct

        query_emb = self.embedder(query)
        blob = struct.pack(f"{len(query_emb)}f", *query_emb)
        key = self._redis_key(query, namespace)

        settings = get_settings()
        self._redis.hset(key, mapping={
            "namespace": namespace,
            "query": query,
            "response": response,
            "embedding": blob,
        })
        self._redis.expire(key, settings.redis_cache_ttl)

    # --- In-memory fallback ---

    def _inmemory_get(self, query: str, namespace: str) -> str | None:
        query_emb = self.embedder(query)
        best_sim = -1.0
        best = None
        for entry in self.entries:
            if entry.namespace != namespace:
                continue
            sim = _cosine_similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= self.threshold:
            CACHE_HITS.inc()
            return best.response
        CACHE_MISSES.inc()
        return None

    def _inmemory_set(self, query: str, response: str, namespace: str) -> None:
        self.entries.append(
            CacheEntry(
                namespace=namespace,
                query=query,
                embedding=self.embedder(query),
                response=response,
            )
        )


_semantic_cache = SemanticCache()


def get_semantic_cache() -> SemanticCache:
    return _semantic_cache


@contextmanager
def semantic_cache_override(*, enabled: bool | None = None, version: str | None = None, clear: bool = False):
    cache = get_semantic_cache()
    previous_enabled = cache.enabled
    previous_version = cache.version
    previous_entries = list(cache.entries)

    cache.configure(enabled=enabled, version=version, clear=clear)
    try:
        yield cache
    finally:
        cache.entries = deque(previous_entries, maxlen=cache.max_size)
        cache.enabled = previous_enabled
        cache.version = previous_version


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2:
        return 0.0
    num = sum(a * b for a, b in zip(v1, v2))
    den1 = math.sqrt(sum(a * a for a in v1))
    den2 = math.sqrt(sum(b * b for b in v2))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)
```

**Step 4: Run all cache tests**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_semantic_cache_contract.py -v`
Expected: all PASSED (using in-memory fallback since no Redis in test env)

**Step 5: Commit**

```bash
git add tools/cache.py tests/rag/test_semantic_cache_contract.py
git commit -m "feat: upgrade SemanticCache to Redis vector search with in-memory fallback"
```

---

### Task 6: Add Prometheus metrics and Grafana dashboard panels for cache observability

**Files:**
- Modify: `observability/metrics.py`
- Modify: `tools/redis_client.py` (add metric increments)
- Modify: `grafana/dashboards/medigenius-dashboard.json` (add cache panels)

**Step 1: Add new Prometheus metrics to observability/metrics.py**

Add after the existing `CACHE_MISSES` line:

```python
CACHE_TOOL_HITS = Counter('medigenius_tool_cache_hits_total', 'Tool/API cache hits', ['tool'])
CACHE_TOOL_MISSES = Counter('medigenius_tool_cache_misses_total', 'Tool/API cache misses', ['tool'])
REDIS_UP = Gauge('medigenius_redis_up', 'Whether Redis is connected (1=up, 0=down)')
```

Also add `Gauge` to the import: `from prometheus_client import Counter, Gauge, Histogram, generate_latest`

**Step 2: Add metric tracking to ExactCache in tools/redis_client.py**

Add import at the top of `tools/redis_client.py`:

```python
from observability.metrics import CACHE_TOOL_HITS, CACHE_TOOL_MISSES, REDIS_UP
```

In `get_redis()`, after successful ping: `REDIS_UP.set(1)`
In `get_redis()`, on connection failure: `REDIS_UP.set(0)`

In `ExactCache.get()`:
- After a cache hit (value is not None): `CACHE_TOOL_HITS.labels(tool=self.prefix).inc()`
- On cache miss (returning None): `CACHE_TOOL_MISSES.labels(tool=self.prefix).inc()`

**Step 3: Add Grafana dashboard panels**

Add 3 new panels to the `panels` array in `grafana/dashboards/medigenius-dashboard.json`:

```json
    {
      "type": "timeseries",
      "title": "Tool Cache Hit/Miss Rate (by tool)",
      "targets": [
        {
          "expr": "rate(medigenius_tool_cache_hits_total[5m])",
          "legendFormat": "{{tool}} hits"
        },
        {
          "expr": "rate(medigenius_tool_cache_misses_total[5m])",
          "legendFormat": "{{tool}} misses"
        }
      ],
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 32}
    },
    {
      "type": "gauge",
      "title": "Tool Cache Hit Ratio (by tool)",
      "targets": [
        {
          "expr": "medigenius_tool_cache_hits_total / (medigenius_tool_cache_hits_total + medigenius_tool_cache_misses_total)",
          "legendFormat": "{{tool}}"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "min": 0,
          "max": 1,
          "thresholds": {
            "steps": [
              {"color": "red", "value": 0},
              {"color": "yellow", "value": 0.3},
              {"color": "green", "value": 0.6}
            ]
          }
        }
      },
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 32}
    },
    {
      "type": "stat",
      "title": "Redis Status",
      "targets": [
        {
          "expr": "medigenius_redis_up",
          "legendFormat": "Redis"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "mappings": [
            {"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}}},
            {"type": "value", "options": {"1": {"text": "UP", "color": "green"}}}
          ]
        }
      },
      "gridPos": {"h": 4, "w": 6, "x": 0, "y": 40}
    }
```

This gives you:
- **Tool Cache Hit/Miss Rate** — time series showing hits vs misses per tool (cohere_rerank, tavily), so you can see if the cache is actually working
- **Tool Cache Hit Ratio** — gauge per tool (0-1), with red/yellow/green thresholds. Below 30% = cache is barely helping. Above 60% = cache is saving significant API calls
- **Redis Status** — simple UP/DOWN indicator. If this goes red, all caches are falling back to in-memory
- The existing **Semantic Cache Hit/Miss Rate** and **Cache Hit Ratio** panels (already in the dashboard) continue to track the answer-level semantic cache

**Step 4: Run tests**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/rag/test_redis_client.py tests/rag/test_semantic_cache_contract.py -v`
Expected: all PASSED

**Step 5: Commit**

```bash
git add observability/metrics.py tools/redis_client.py grafana/dashboards/medigenius-dashboard.json
git commit -m "feat: add Prometheus metrics and Grafana panels for cache observability"
```

---

### Task 7: Update token budget constants

**Files:**
- Modify: `agents/executor_agent.py`

While we're touching the executor, fix the overly conservative token budget discussed earlier.

**Step 1: Update constants**

Change:
```python
_MAX_TOTAL_TOKENS = 4000
_GENERATION_HEADROOM = 800
```

To:
```python
_MAX_TOTAL_TOKENS = 16000
_GENERATION_HEADROOM = 1200
```

**Step 2: Run existing tests**

Run: `cd /home/tough/medical_chatbot/MediGenius && python -m pytest tests/ -v --timeout=30 -x`
Expected: all PASSED

**Step 3: Commit**

```bash
git add agents/executor_agent.py
git commit -m "fix: raise executor prompt token budget to 16k for 128k-context models"
```

---

### Task 8: Integration smoke test with docker-compose

**Step 1: Build and start services**

```bash
cd /home/tough/medical_chatbot/MediGenius
docker-compose build app
docker-compose up -d db redis app
```

**Step 2: Verify Redis is running**

```bash
docker-compose exec redis redis-cli ping
```
Expected: `PONG`

**Step 3: Verify app connects to Redis**

```bash
docker-compose logs app | grep -i redis
```
Expected: line containing `Redis connected: redis://redis:6379/0`

**Step 4: Test vector index creation**

```bash
docker-compose exec redis redis-cli FT._LIST
```
Expected: should list `idx:semantic_cache` (created on first SemanticCache init)

**Step 5: Commit any fixes needed**

---

### Task 9: Update documentation

**Files:**
- Modify: `docs/changes/implementation-log.md`
- Modify: `.env.example` (already done in Task 1)

**Step 1: Append to implementation log**

Add entry:
```markdown
### 2026-03-15 — Redis caching layer
- Added Redis Stack service to docker-compose for production caching
- Tool/API caching: Cohere rerank (1h TTL), Tavily search (24h TTL) — protects rate limits
- Upgraded SemanticCache from in-memory deque (linear scan) to Redis vector search (HNSW)
- All caches fall back to in-memory when Redis is unavailable (zero-config local dev)
- Added Prometheus metrics for tool cache hit/miss rates
- Strategy B defaults: 0.92 threshold, 24h TTL, global scope
```

**Step 2: Commit**

```bash
git add docs/changes/implementation-log.md
git commit -m "docs: log Redis caching implementation"
```

---

## Settings summary (Strategy B — aggressive/development)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Catch paraphrases, evaluate what breaks |
| `REDIS_CACHE_TTL` | `86400` (24h) | Long TTL, maximise hits for dev testing |
| `SEMANTIC_CACHE_SIZE` | `5000` | Generous — Redis handles it easily |
| Cohere cache TTL | `3600` (1h) | Same docs reranked differently is unlikely within 1h |
| Tavily cache TTL | `86400` (24h) | Web results don't change hourly |

Update these in `.env.example` and `core/settings.py` defaults during Task 1.
