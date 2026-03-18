from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session

from core.settings import get_settings
from db.models import (
    ConversationSummaryModel,
    DocumentChunkModel,
    EvalRunMetricModel,
    EvalRunModel,
    EvalRunSliceModel,
    MessageModel,
    SessionModel,
    UserFactModel,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionPreview:
    session_id: str
    created_at: datetime
    last_active: datetime
    preview: str | None


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def ensure_session(self, session_id: str) -> None:
        existing = self.db.get(SessionModel, session_id)
        if existing is None:
            now = self._now_utc()
            existing = SessionModel(session_id=session_id, created_at=now, last_active=now)
            self.db.add(existing)
        existing.last_active = self._now_utc()
        self.db.commit()

    def add_message(self, session_id: str, role: str, content: str, source: str | None = None) -> None:
        self.ensure_session(session_id)
        self.db.add(
            MessageModel(
                session_id=session_id,
                role=role,
                content=content,
                source=source,
                created_at=self._now_utc(),
            )
        )
        self.db.commit()

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        rows = (
            self.db.execute(
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.created_at.asc(), MessageModel.id.asc())
            )
            .scalars()
            .all()
        )
        return [
            {
                "role": row.role,
                "content": row.content,
                "source": row.source,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def list_sessions(self) -> list[SessionPreview]:
        previews: list[SessionPreview] = []
        sessions = self.db.execute(select(SessionModel).order_by(desc(SessionModel.last_active))).scalars().all()
        for session in sessions:
            first_user = self.db.execute(
                select(MessageModel.content)
                .where(MessageModel.session_id == session.session_id, MessageModel.role == "user")
                .order_by(MessageModel.created_at.asc())
                .limit(1)
            ).scalar_one_or_none()
            preview = (first_user[:50] + "...") if first_user and len(first_user) > 50 else first_user
            previews.append(
                SessionPreview(
                    session_id=session.session_id,
                    created_at=session.created_at,
                    last_active=session.last_active,
                    preview=preview,
                )
            )
        return previews

    def delete_session(self, session_id: str) -> None:
        self.db.query(MessageModel).filter(MessageModel.session_id == session_id).delete()
        self.db.query(ConversationSummaryModel).filter(ConversationSummaryModel.session_id == session_id).delete()
        self.db.query(UserFactModel).filter(UserFactModel.session_id == session_id).delete()
        self.db.query(SessionModel).filter(SessionModel.session_id == session_id).delete()
        self.db.commit()

    def upsert_summary(self, session_id: str, summary: str) -> None:
        existing = self.db.get(ConversationSummaryModel, session_id)
        if existing is None:
            existing = ConversationSummaryModel(
                session_id=session_id,
                summary=summary,
                updated_at=self._now_utc(),
            )
            self.db.add(existing)
        else:
            existing.summary = summary
            existing.updated_at = self._now_utc()
        self.db.commit()

    def get_summary(self, session_id: str) -> str:
        existing = self.db.get(ConversationSummaryModel, session_id)
        return existing.summary if existing else ""

    def upsert_fact(self, session_id: str, fact_key: str, fact_value: str, confidence: float = 0.8) -> None:
        existing = self.db.execute(
            select(UserFactModel)
            .where(UserFactModel.session_id == session_id, UserFactModel.fact_key == fact_key)
            .order_by(UserFactModel.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if existing is None:
            existing = UserFactModel(
                session_id=session_id,
                fact_key=fact_key,
                fact_value=fact_value,
                confidence=confidence,
                created_at=self._now_utc(),
            )
            self.db.add(existing)
        else:
            existing.fact_value = fact_value
            existing.confidence = confidence
        self.db.commit()

    def list_facts(self, session_id: str) -> list[dict[str, Any]]:
        rows = (
            self.db.execute(
                select(UserFactModel)
                .where(UserFactModel.session_id == session_id)
                .order_by(desc(UserFactModel.created_at))
            )
            .scalars()
            .all()
        )
        return [{"key": row.fact_key, "value": row.fact_value, "confidence": row.confidence} for row in rows]


class EvalRunRepository:
    def __init__(self, db: Session):
        self.db = db

    def record_run(
        self,
        *,
        run_record: dict[str, Any],
        metric_rows: list[dict[str, Any]],
        slice_rows: list[dict[str, Any]],
    ) -> str:
        try:
            self.db.add(
                EvalRunModel(
                    run_id=run_record["run_id"],
                    eval_type=run_record["eval_type"],
                    mode=run_record["mode"],
                    split=run_record["split"],
                    sample_count=run_record["sample_count"],
                    git_sha=run_record["git_sha"],
                    dataset_path=run_record["dataset_path"],
                    snapshot_path=run_record["snapshot_path"],
                    judge_model=run_record.get("judge_model"),
                    metadata_json=run_record.get("metadata_json") or {},
                )
            )
            if metric_rows:
                self.db.add_all(
                    EvalRunMetricModel(
                        run_id=row["run_id"],
                        metric_name=row["metric_name"],
                        metric_value=row["metric_value"],
                    )
                    for row in metric_rows
                )
            if slice_rows:
                self.db.add_all(
                    EvalRunSliceModel(
                        run_id=row["run_id"],
                        slice_name=row["slice_name"],
                        metric_name=row["metric_name"],
                        metric_value=row["metric_value"],
                        sample_count=row["sample_count"],
                    )
                    for row in slice_rows
                )
            self.db.commit()
            return run_record["run_id"]
        except Exception:
            self.db.rollback()
            raise


class VectorRepository:
    def __init__(self, db: Session):
        self.db = db

    def purge_doc_chunks(self, doc_id: str) -> int:
        """Delete all chunks belonging to a doc_id. Returns the number of rows deleted."""
        count = (
            self.db.query(DocumentChunkModel)
            .filter(DocumentChunkModel.doc_id == doc_id)
            .delete(synchronize_session="fetch")
        )
        self.db.commit()
        return count

    def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: list[float],
        page: int | None = None,
        section: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = self.db.execute(
            select(DocumentChunkModel).where(DocumentChunkModel.chunk_id == chunk_id).limit(1)
        ).scalar_one_or_none()
        metadata = metadata or {}
        if row is None:
            row = DocumentChunkModel(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                page=page,
                section=section,
                metadata_json=metadata,
            )
            if hasattr(row, "embedding"):
                row.embedding = embedding
            else:
                row.embedding_json = embedding
                row.is_vector_ready = True
            self.db.add(row)
        else:
            row.content = content
            row.page = page
            row.section = section
            row.metadata_json = metadata
            if hasattr(row, "embedding"):
                row.embedding = embedding
            else:
                row.embedding_json = embedding
                row.is_vector_ready = True
        self.db.commit()

    def upsert_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
        replace_doc_id: str | None = None,
        batch_size: int = 250,
    ) -> int:
        """
        Bulk-insert chunks in a bounded number of ORM objects per flush.

        If replace_doc_id is provided, the replacement is purge-and-replace for
        that document inside one short-lived transaction.
        """
        if not chunks:
            return 0

        use_vector = hasattr(DocumentChunkModel, "embedding") and not hasattr(DocumentChunkModel, "embedding_json")
        try:
            if replace_doc_id:
                self.db.query(DocumentChunkModel).filter(DocumentChunkModel.doc_id == replace_doc_id).delete(
                    synchronize_session=False
                )
            else:
                chunk_ids = [c["chunk_id"] for c in chunks]
                self.db.query(DocumentChunkModel).filter(DocumentChunkModel.chunk_id.in_(chunk_ids)).delete(
                    synchronize_session=False
                )

            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                rows = []
                for c in batch:
                    row = DocumentChunkModel(
                        chunk_id=c["chunk_id"],
                        doc_id=c["doc_id"],
                        content=c["content"],
                        page=c.get("page"),
                        section=c.get("section"),
                        metadata_json=c.get("metadata") or {},
                    )
                    if use_vector:
                        row.embedding = c["embedding"]
                    else:
                        row.embedding_json = c["embedding"]
                        row.is_vector_ready = True
                    rows.append(row)

                self.db.add_all(rows)
                self.db.flush()
                self.db.expunge_all()

            self.db.commit()
            return len(chunks)
        except Exception:
            self.db.rollback()
            raise

    def count_chunks(self) -> int:
        return self.db.execute(select(func.count(DocumentChunkModel.id))).scalar_one()

    def _similarity_search_scored(
        self,
        query_embedding: list[float],
        k: int = 8,
    ) -> list[tuple[Document, float]]:
        """Return (doc, similarity_score) pairs sorted by descending similarity."""
        if hasattr(DocumentChunkModel, "embedding"):
            rows = (
                self.db.execute(
                    select(DocumentChunkModel)
                    .order_by(DocumentChunkModel.embedding.cosine_distance(query_embedding))
                    .limit(k)
                )
                .scalars()
                .all()
            )
            results = []
            for row in rows:
                dist = _cosine_distance(
                    list(row.embedding) if row.embedding is not None else [],
                    query_embedding,
                )
                doc = _orm_row_to_doc(row)
                results.append((doc, 1.0 - dist))
            return results
        else:
            rows = self.db.execute(select(DocumentChunkModel)).scalars().all()
            scored = [(r, 1.0 - _cosine_distance(r.embedding_json, query_embedding)) for r in rows]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [(_orm_row_to_doc(r), s) for r, s in scored[:k]]

    def similarity_search(self, query_embedding: list[float], k: int = 8) -> list[Document]:
        return [doc for doc, _ in self._similarity_search_scored(query_embedding, k)]

    def _keyword_search_scored(
        self,
        query: str,
        k: int = 8,
    ) -> list[tuple[Document, float]]:
        """Return (doc, bm25_score) pairs sorted by descending score."""
        terms = [re.sub(r"[^a-z0-9]", "", t) for t in query.lower().split()]
        terms = [t for t in terms if len(t) > 2]
        if not terms:
            return []

        settings = get_settings()
        rows = self.db.execute(select(DocumentChunkModel)).scalars().all()

        tokenized = []
        total_tokens = 0
        for row in rows:
            doc_tokens = [t for t in row.content.lower().split() if len(t) > 2]
            tokenized.append((row, doc_tokens))
            total_tokens += len(doc_tokens)
        avg_dl = total_tokens / max(len(tokenized), 1)

        scored = []
        for row, doc_tokens in tokenized:
            score = _bm25_score(terms, doc_tokens, avg_dl, k1=settings.bm25_k1, b=settings.bm25_b)
            if score > 0:
                scored.append((row, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(_orm_row_to_doc(row), s) for row, s in scored[:k]]

    def keyword_search(self, query: str, k: int = 8) -> list[Document]:
        return [doc for doc, _ in self._keyword_search_scored(query, k)]

    def hybrid_search(self, query: str, query_embedding: list[float], k: int = 8) -> list[Document]:
        settings = get_settings()
        dense_weight = settings.hybrid_dense_weight
        sparse_weight = 1.0 - dense_weight

        dense_scored = self._similarity_search_scored(query_embedding, k=k * 2)
        sparse_scored = self._keyword_search_scored(query, k=k * 2)

        scored: dict[str, tuple[Document, float]] = {}
        _merge_weighted(scored, dense_scored, dense_weight)
        _merge_weighted(scored, sparse_scored, sparse_weight)

        merged = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in merged[:k]]

    def get_adjacent_chunks(self, section: str, chunk_index: int) -> list[Document]:
        """Return chunks at chunk_index +/- 1 within the same section."""
        prev_idx = chunk_index - 1
        next_idx = chunk_index + 1
        stmt = text(
            "SELECT chunk_id, doc_id, page, section, content, metadata_json "
            "FROM document_chunks "
            "WHERE section = :sec "
            "AND (metadata_json->>'chunk_index')::int IN (:prev, :next) "
            "ORDER BY (metadata_json->>'chunk_index')::int ASC"
        )
        rows = self.db.execute(stmt, {"sec": section, "prev": prev_idx, "next": next_idx}).fetchall()
        return [
            Document(
                page_content=row.content,
                metadata={
                    "chunk_id": row.chunk_id,
                    "doc_id": row.doc_id,
                    "page": row.page,
                    "section": row.section,
                    **(row.metadata_json if row.metadata_json else {}),
                },
            )
            for row in rows
        ]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Document]:
        """Fetch chunks by their chunk_ids."""
        if not chunk_ids:
            return []
        rows = (
            self.db.execute(select(DocumentChunkModel).where(DocumentChunkModel.chunk_id.in_(chunk_ids)))
            .scalars()
            .all()
        )
        return [_orm_row_to_doc(row) for row in rows]


class InMemoryChatRepository:
    def __init__(self):
        self._messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._summaries: dict[str, str] = {}
        self._facts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    def ensure_session(self, session_id: str) -> None:
        self._messages.setdefault(session_id, [])

    def add_message(self, session_id: str, role: str, content: str, source: str | None = None) -> None:
        self.ensure_session(session_id)
        self._messages[session_id].append(
            {"role": role, "content": content, "source": source, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._messages.get(session_id, []))

    def list_sessions(self) -> list[SessionPreview]:
        now = datetime.now(timezone.utc)
        out = []
        for session_id, messages in self._messages.items():
            first_user = next((m["content"] for m in messages if m["role"] == "user"), None)
            preview = (first_user[:50] + "...") if first_user and len(first_user) > 50 else first_user
            out.append(SessionPreview(session_id=session_id, created_at=now, last_active=now, preview=preview))
        return out

    def delete_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._summaries.pop(session_id, None)
        self._facts.pop(session_id, None)

    def upsert_summary(self, session_id: str, summary: str) -> None:
        self._summaries[session_id] = summary

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def upsert_fact(self, session_id: str, fact_key: str, fact_value: str, confidence: float = 0.8) -> None:
        self._facts[session_id][fact_key] = {"key": fact_key, "value": fact_value, "confidence": confidence}

    def list_facts(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._facts.get(session_id, {}).values())


def _orm_row_to_doc(row: DocumentChunkModel) -> Document:
    """Convert an ORM row to a LangChain Document."""
    return Document(
        page_content=row.content,
        metadata={
            "chunk_id": row.chunk_id,
            "doc_id": row.doc_id,
            "page": row.page,
            "section": row.section,
            **(row.metadata_json or {}),
        },
    )


def _normalize_scores(scored: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
    """Min-max normalize scores to [0, 1]."""
    if not scored:
        return scored
    scores = [s for _, s in scored]
    s_min, s_max = min(scores), max(scores)
    s_range = s_max - s_min
    if s_range == 0:
        return [(item, 1.0) for item, _ in scored]
    return [(item, (s - s_min) / s_range) for item, s in scored]


def _merge_weighted(
    target: dict[str, tuple[Document, float]],
    scored: list[tuple[Document, float]],
    weight: float,
) -> None:
    """Merge normalized scored docs into target dict with given weight."""
    normed = _normalize_scores(scored)
    for doc, norm_score in normed:
        key = doc.metadata.get("chunk_id") or str(id(doc))
        existing = target.get(key, (doc, 0.0))[1]
        target[key] = (doc, existing + weight * norm_score)


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document (no IDF — single-query context)."""
    dl = len(doc_tokens)
    if dl == 0 or avg_dl == 0:
        return 0.0
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
    return score


class InMemoryVectorRepository:
    def __init__(self):
        self._chunks: dict[str, dict[str, Any]] = {}

    def purge_doc_chunks(self, doc_id: str) -> int:
        to_remove = [k for k, v in self._chunks.items() if v["doc_id"] == doc_id]
        for k in to_remove:
            del self._chunks[k]
        return len(to_remove)

    def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: list[float],
        page: int | None = None,
        section: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "embedding": embedding,
            "page": page,
            "section": section,
            "metadata": metadata or {},
        }

    def upsert_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
        replace_doc_id: str | None = None,
        batch_size: int = 250,
    ) -> int:
        if replace_doc_id:
            self.purge_doc_chunks(replace_doc_id)
        for c in chunks:
            self._chunks[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "content": c["content"],
                "embedding": c["embedding"],
                "page": c.get("page"),
                "section": c.get("section"),
                "metadata": c.get("metadata") or {},
            }
        return len(chunks)

    def count_chunks(self) -> int:
        return len(self._chunks)

    def _similarity_search_scored(
        self,
        query_embedding: list[float],
        k: int = 8,
    ) -> list[tuple[Document, float]]:
        scored = [(row, 1.0 - _cosine_distance(row["embedding"], query_embedding)) for row in self._chunks.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(_row_to_doc(row), s) for row, s in scored[:k]]

    def similarity_search(self, query_embedding: list[float], k: int = 8) -> list[Document]:
        return [doc for doc, _ in self._similarity_search_scored(query_embedding, k)]

    def _keyword_search_scored(
        self,
        query: str,
        k: int = 8,
    ) -> list[tuple[Document, float]]:
        terms = [t for t in query.lower().split() if len(t) > 2]
        if not terms:
            return []
        settings = get_settings()

        tokenized: list[tuple[dict[str, Any], list[str]]] = []
        total_tokens = 0
        for row in self._chunks.values():
            doc_tokens = [t for t in row["content"].lower().split() if len(t) > 2]
            tokenized.append((row, doc_tokens))
            total_tokens += len(doc_tokens)
        avg_dl = total_tokens / max(len(tokenized), 1)

        scored = []
        for row, doc_tokens in tokenized:
            score = _bm25_score(terms, doc_tokens, avg_dl, k1=settings.bm25_k1, b=settings.bm25_b)
            if score > 0:
                scored.append((row, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(_row_to_doc(row), s) for row, s in scored[:k]]

    def keyword_search(self, query: str, k: int = 8) -> list[Document]:
        return [doc for doc, _ in self._keyword_search_scored(query, k)]

    def hybrid_search(self, query: str, query_embedding: list[float], k: int = 8) -> list[Document]:
        settings = get_settings()
        dense_weight = settings.hybrid_dense_weight
        sparse_weight = 1.0 - dense_weight

        dense_scored = self._similarity_search_scored(query_embedding, k=k * 2)
        sparse_scored = self._keyword_search_scored(query, k=k * 2)

        scored: dict[str, tuple[Document, float]] = {}
        _merge_weighted(scored, dense_scored, dense_weight)
        _merge_weighted(scored, sparse_scored, sparse_weight)

        merged = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in merged[:k]]

    def get_adjacent_chunks(self, section: str, chunk_index: int) -> list[Document]:
        """Return chunks at chunk_index +/- 1 within the same section."""
        target_indices = {chunk_index - 1, chunk_index + 1}
        results = []
        for row in self._chunks.values():
            meta = row.get("metadata") or {}
            if meta.get("section") == section and meta.get("chunk_index") in target_indices:
                results.append(_row_to_doc(row))
        results.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        return results

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Document]:
        """Fetch chunks by their chunk_ids."""
        if not chunk_ids:
            return []
        results = []
        for cid in chunk_ids:
            row = self._chunks.get(cid)
            if row is not None:
                results.append(_row_to_doc(row))
        return results


def _cosine_distance(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2:
        return 1.0
    num = sum(a * b for a, b in zip(v1, v2))
    den_left = math.sqrt(sum(a * a for a in v1))
    den_right = math.sqrt(sum(b * b for b in v2))
    if den_left == 0 or den_right == 0:
        return 1.0
    return 1.0 - (num / (den_left * den_right))


def _row_to_doc(row: dict[str, Any]) -> Document:
    return Document(
        page_content=row["content"],
        metadata={
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "page": row["page"],
            "section": row["section"],
            **(row.get("metadata") or {}),
        },
    )
