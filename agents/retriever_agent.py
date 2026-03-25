from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    from cohere import RequestOptions as _CohereRequestOptions
except Exception:  # cohere not installed or import error
    _CohereRequestOptions = None  # type: ignore[assignment,misc]

import numpy as np
from langchain_core.documents import Document

from core.settings import get_settings
from core.state_v2 import AgentStateV2
from db.repositories import InMemoryVectorRepository, VectorRepository
from tools.embedding_client import embed_query
from tools.redis_client import ExactCache

logger = logging.getLogger(__name__)

_cohere_cache = ExactCache(prefix="cohere_rerank", ttl=3600)


def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _section_quota_filter(docs: list, k: int, max_per_section: int = 2) -> list:
    """Fallback: cap how many chunks from the same section make it to top-k."""
    seen: dict = {}
    result = []
    for doc in docs:
        section = doc.metadata.get("section", "_unknown")
        count = seen.get(section, 0)
        if count < max_per_section:
            result.append(doc)
            seen[section] = count + 1
        if len(result) == k:
            break
    return result


def _mmr_filter(
    docs: list,
    emb_map: dict,
    query_emb: "np.ndarray",
    k: int = 5,
    lambda_: float = 0.7,
) -> list:
    """
    Maximal Marginal Relevance filter.
    Selects k docs that balance relevance to query and diversity from each other.
    Falls back to section-based quota when embeddings are missing.
    """
    if not docs:
        return docs

    have_embs = [d for d in docs if d.metadata.get("chunk_id") in emb_map]
    if len(have_embs) < len(docs) // 2:
        return _section_quota_filter(docs, k)

    selected: list = []
    remaining = list(docs)

    while remaining and len(selected) < k:
        if not selected:
            best = max(
                remaining,
                key=lambda d: _cosine_sim(
                    emb_map.get(d.metadata.get("chunk_id", ""), query_emb),
                    query_emb,
                ),
            )
        else:
            selected_embs = [
                emb_map[d.metadata["chunk_id"]]
                for d in selected
                if d.metadata.get("chunk_id") in emb_map
            ]
            best = max(
                remaining,
                key=lambda d: (
                    lambda_ * _cosine_sim(
                        emb_map.get(d.metadata.get("chunk_id", ""), query_emb),
                        query_emb,
                    )
                    - (1 - lambda_) * (
                        max((_cosine_sim(
                            emb_map.get(d.metadata.get("chunk_id", ""), query_emb),
                            s,
                        ) for s in selected_embs), default=0.0)
                    )
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


def _merge_and_deduplicate(
    primary: list[Document], stepback: list[Document]
) -> list[Document]:
    """Merge primary and step-back docs, deduplicating by chunk_id (primary wins)."""
    seen: set[str] = set()
    merged: list[Document] = []
    for doc in primary:
        cid = doc.metadata.get('chunk_id', id(doc))
        if cid not in seen:
            seen.add(cid)
            merged.append(doc)
    for doc in stepback:
        cid = doc.metadata.get('chunk_id', id(doc))
        if cid not in seen:
            seen.add(cid)
            merged.append(doc)
    return merged


def _resolve_parent_chunks(
    docs: list[Document],
    repo: VectorRepository | InMemoryVectorRepository,
) -> list[Document]:
    """Resolve parent chunk content for child chunks.

    For each child chunk that has a parent_chunk_id, fetch the parent's
    content and attach it as 'parent_content' in metadata.
    Non-child chunks pass through unchanged.
    """
    parent_ids: set[str] = set()
    for doc in docs:
        pid = doc.metadata.get("parent_chunk_id")
        if pid and doc.metadata.get("chunk_type") == "child":
            parent_ids.add(pid)

    if not parent_ids:
        return docs

    parent_docs = repo.get_chunks_by_ids(list(parent_ids))
    parent_map: dict[str, str] = {
        d.metadata["chunk_id"]: d.page_content for d in parent_docs
    }

    for doc in docs:
        pid = doc.metadata.get("parent_chunk_id")
        if pid and pid in parent_map:
            doc.metadata["parent_content"] = parent_map[pid]

    return docs


def RetrieverAgent(state: AgentStateV2) -> AgentStateV2:
    query = state.get('optimized_query') or state.get('question', '')

    # ReAct: on retry, use reflection's suggested_focus as the retrieval query
    is_retry = state.get('needs_retry', False)
    suggested_focus = state.get('reflection_suggested_focus', '')
    if is_retry and suggested_focus:
        query = suggested_focus
        logger.info("react_retry_refocused", extra={"suggested_focus": suggested_focus[:80]})

    repo = state.get('vector_repo')
    if repo is None:
        state['documents'] = []
        state['retrieval_confidence'] = 0.0
        state['retrieval_candidate_count'] = 0
        state['rerank_method'] = 'none'
        state['post_retrieval_route'] = 'executor'
        state['route_decision_reason'] = 'no_vector_repo'
        return state

    query_embedding = embed_query(query)
    docs = repo.hybrid_search(query=query, query_embedding=query_embedding, k=20)

    # On retry, skip stepback (it was for the original query, not the focused retry)
    if not is_retry:
        stepback = state.get('stepback_query', '')
        if stepback:
            sb_embedding = embed_query(stepback)
            sb_docs = repo.hybrid_search(query=stepback, query_embedding=sb_embedding, k=10)
            docs = _merge_and_deduplicate(docs, sb_docs)

    ranked, scores, rerank_method = rerank(query, docs)
    state['rerank_method'] = rerank_method

    # Apply MMR diversity filter to top-k selection
    settings = get_settings()
    top_k = 5
    if settings.diversity_mmr_enabled:
        chunk_ids = [d.metadata.get("chunk_id", "") for d in ranked[:20]]
        emb_map = repo.get_embeddings_for_chunk_ids(chunk_ids) if repo is not None else {}
        query_emb_arr = np.array(query_embedding) if query_embedding is not None else np.zeros(1)
        top_docs = _mmr_filter(ranked, emb_map, query_emb_arr, k=top_k, lambda_=settings.diversity_mmr_lambda)
    else:
        top_docs = ranked[:top_k]

    # Resolve parent chunks if parent-child chunking is active
    if any(d.metadata.get("chunk_type") == "child" for d in top_docs):
        top_docs = _resolve_parent_chunks(top_docs, repo)

    state['documents'] = top_docs
    state['retrieval_candidate_count'] = len(ranked)
    state['retrieval_confidence'] = scores[0] if scores else 0.0
    state['source'] = 'PostgreSQL pgvector + Hybrid Retrieval'

    _decide_post_retrieval_route(state)
    return state


def rerank(query: str, docs: list[Document]) -> tuple[list[Document], list[float], str]:
    """Rerank documents using Cohere cross-encoder, with BM25 fallback.

    Returns (ranked_docs, scores, method_name) where method_name is "cohere" or "bm25".
    """
    if not docs:
        return docs, [], "bm25"

    settings = get_settings()

    if settings.cohere_api_key:
        try:
            ranked, scores = _cohere_rerank(query, docs, settings)
            return ranked, scores, "cohere"
        except Exception:
            logger.warning("Cohere rerank failed, falling back to BM25", exc_info=True)

    ranked, scores = _bm25_rerank(query, docs)
    return ranked, scores, "bm25"


_COHERE_MAX_RETRIES = 2
_COHERE_RETRY_WAIT = 3  # seconds — short wait, fall back to BM25 quickly


def _cohere_rerank(
    query: str, docs: list[Document], settings: Any,
) -> tuple[list[Document], list[float]]:
    """Rerank using Cohere's cross-encoder API with rate-limit retry."""
    import time
    import cohere

    client = cohere.Client(api_key=settings.cohere_api_key)
    doc_texts = [doc.page_content for doc in docs]

    # --- Cache lookup ---
    cache_key = query + "||" + "||".join(sorted(doc_texts))
    cached = _cohere_cache.get(cache_key)
    if cached is not None:
        parsed = json.loads(cached)
        indices = parsed["indices"]
        ranked_scores = parsed["scores"]
        ranked_docs = [docs[i] for i in indices]
        logger.debug("Cohere rerank cache hit for query: %s", query[:80])
        logger.debug("rerank_method=cohere (cache)")
        return ranked_docs, ranked_scores

    # Build request_options with a 10s timeout if the SDK supports it
    request_options = None
    if _CohereRequestOptions is not None:
        try:
            request_options = _CohereRequestOptions(timeout_in_seconds=10)
        except Exception:
            pass

    last_exc: Exception | None = None
    for attempt in range(_COHERE_MAX_RETRIES):
        try:
            call_kwargs: dict[str, Any] = dict(
                query=query,
                documents=doc_texts,
                model=settings.cohere_rerank_model,
                top_n=len(docs),
            )
            if request_options is not None:
                call_kwargs["request_options"] = request_options

            response = client.rerank(**call_kwargs)

            ranked_docs = []
            ranked_scores = []
            indices = []
            for result in response.results:
                ranked_docs.append(docs[result.index])
                ranked_scores.append(result.relevance_score)
                indices.append(result.index)

            # --- Cache store ---
            _cohere_cache.set(cache_key, json.dumps({"indices": indices, "scores": ranked_scores}))

            logger.debug("rerank_method=cohere")
            return ranked_docs, ranked_scores

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if 'rate' in exc_str or 'limit' in exc_str or '429' in exc_str:
                if attempt < _COHERE_MAX_RETRIES - 1:
                    logger.info(
                        "Cohere rate limited, waiting %ds (attempt %d/%d)",
                        _COHERE_RETRY_WAIT, attempt + 1, _COHERE_MAX_RETRIES,
                    )
                    time.sleep(_COHERE_RETRY_WAIT)
            else:
                raise

    raise last_exc  # type: ignore[misc]


def _bm25_rerank(
    query: str, docs: list[Document],
) -> tuple[list[Document], list[float]]:
    """Fallback BM25 reranker."""
    query_tokens = _normalize_tokens(query)
    if not query_tokens:
        return docs, [0.0] * len(docs)

    settings = get_settings()
    k1, b = settings.bm25_k1, settings.bm25_b

    doc_token_lists = [_normalize_tokens(doc.page_content) for doc in docs]
    avg_dl = sum(len(dt) for dt in doc_token_lists) / max(len(doc_token_lists), 1)

    scored = []
    for doc, doc_tokens in zip(docs, doc_token_lists):
        dl = len(doc_tokens)
        doc_freq: dict[str, int] = {}
        for t in doc_tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1
        score = 0.0
        for qt in query_tokens:
            tf = doc_freq.get(qt, 0)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
            score += numerator / denominator
        scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    logger.debug("rerank_method=bm25")
    return [doc for doc, _ in scored], [score for _, score in scored]


def _normalize_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return [token for token in normalized.split() if len(token) > 2]


def _decide_post_retrieval_route(state: AgentStateV2) -> None:
    planned_route = state.get('planned_route') or state.get('route') or 'vector'
    has_docs = bool(state.get('documents'))
    confidence = float(state.get('retrieval_confidence', 0.0))
    threshold = get_settings().routing_rag_confidence_threshold

    if planned_route in {'vector', 'future_clinical_db'}:
        state['route'] = 'vector'
        state['post_retrieval_route'] = 'executor'
        state['route_decision_reason'] = 'retriever_first_vector'
        return

    if planned_route == 'web':
        if has_docs and confidence >= threshold:
            state['route'] = 'vector'
            state['post_retrieval_route'] = 'executor'
            state['route_decision_reason'] = 'retriever_hit_over_web'
            return
        state['route'] = 'web'
        state['post_retrieval_route'] = 'web'
        state['route_decision_reason'] = 'retriever_low_confidence_to_web'
        return

    if planned_route == 'literature':
        if has_docs and confidence >= threshold:
            state['route'] = 'vector'
            state['post_retrieval_route'] = 'executor'
            state['route_decision_reason'] = 'retriever_hit_over_literature'
            return
        state['route'] = 'literature'
        state['post_retrieval_route'] = 'literature'
        state['route_decision_reason'] = 'retriever_low_confidence_to_literature'
        return

    state['route'] = 'vector'
    state['post_retrieval_route'] = 'executor'
    state['route_decision_reason'] = 'retriever_default_vector'
