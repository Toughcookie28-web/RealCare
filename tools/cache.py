"""tools/cache.py
Semantic answer cache with Redis vector search backend and in-memory fallback.

Redis backend: uses RediSearch HNSW index for O(log n) similarity lookup.
Fallback: original deque-based linear scan when Redis is unavailable.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any
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
        self._redis: Any = self._init_redis_index()

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
            index_name = "idx:semantic_cache"
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

    def _redis_namespace_filter(self, namespace: str) -> str:
        escaped_namespace = namespace.replace(":", r"\:")
        return f"(@namespace:{{{escaped_namespace}}})"

    def _redis_get(self, query: str, namespace: str) -> str | None:
        import struct

        query_emb = self.embedder(query)
        blob = struct.pack(f"{len(query_emb)}f", *query_emb)

        # KNN search filtered by namespace
        results = self._redis.execute_command(
            "FT.SEARCH", "idx:semantic_cache",
            f"{self._redis_namespace_filter(namespace)}[KNN 1 @embedding $vec AS score]",
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
