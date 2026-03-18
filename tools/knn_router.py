from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SEEDS_PATH = Path(__file__).resolve().parent.parent / 'data' / 'route_seeds.json'
_cached_seeds: list[tuple[list[float], str]] | None = None


def load_seeds(path: str | Path | None = None) -> list[tuple[list[float], str]]:
    """Load pre-embedded seed queries from JSON. Returns [] on missing/invalid file."""
    seeds_path = Path(path) if path else _DEFAULT_SEEDS_PATH
    if not seeds_path.exists():
        logger.warning("knn_seeds_missing", extra={"path": str(seeds_path)})
        return []
    try:
        with open(seeds_path) as f:
            data = json.load(f)
        seeds: list[tuple[list[float], str]] = []
        for route_label, entries in data.items():
            for entry in entries:
                embedding = entry.get('embedding', [])
                if embedding:
                    seeds.append((embedding, route_label))
        return seeds
    except Exception as e:
        logger.warning("knn_seeds_load_error", extra={"error": str(e)})
        return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def knn_route(
    query_embedding: list[float],
    seeds: list[tuple[list[float], str]],
    k: int = 5,
) -> str:
    """Route by majority vote of k nearest seed embeddings. Returns 'vector' as default."""
    if not seeds:
        return 'vector'

    scored = [
        (_cosine_similarity(query_embedding, seed_emb), label)
        for seed_emb, label in seeds
    ]
    scored.sort(reverse=True)
    top_k = scored[:k]

    votes: dict[str, int] = {}
    for _sim, label in top_k:
        votes[label] = votes.get(label, 0) + 1

    if not votes:
        return 'vector'

    winner = max(votes, key=votes.get)

    if votes[winner] / k < 0.4:
        logger.info("knn_low_confidence", extra={"votes": votes, "k": k})
        return 'vector'

    return winner


def knn_fallback_route(state: dict[str, Any]) -> str:
    """Fallback route using k-NN seed similarity. Returns 'vector' if seeds unavailable."""
    global _cached_seeds
    if _cached_seeds is None:
        _cached_seeds = load_seeds()

    if not _cached_seeds:
        return 'vector'

    query = state.get('optimized_query') or state.get('question', '')
    if not query:
        return 'vector'

    try:
        from tools.embedding_client import embed_query
        query_embedding = embed_query(query)
    except Exception as e:
        logger.warning("knn_embed_failed", extra={"error": str(e)})
        return 'vector'

    return knn_route(query_embedding, _cached_seeds, k=5)
