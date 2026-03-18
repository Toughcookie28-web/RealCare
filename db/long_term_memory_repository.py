from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import ConversationMemoryModel, LongTermFactModel, UserModel

logger = logging.getLogger(__name__)


class LongTermMemoryRepository:
    """
    Handles all long-term, user-scoped memory operations.
    Separate from ChatRepository which is session-scoped.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── User identity ────────────────────────────────────────────────────────

    def get_or_create_user(self, user_token: str) -> str:
        """
        Resolve a persistent user_token to a user_id.
        Creates the user row on first visit. Returns user_id.
        """
        existing = self._session.get(UserModel, user_token)
        if existing:
            return existing.user_id

        user = UserModel(user_id=user_token)
        self._session.add(user)
        self._session.commit()
        logger.info("long_term_user_created", extra={"user_id": user_token})
        return user_token

    # ── Long-term facts ──────────────────────────────────────────────────────

    def upsert_long_term_fact(
        self,
        user_id: str,
        fact_key: str,
        fact_value: str,
        confidence: float = 0.7,
        source_session: str | None = None,
    ) -> None:
        """
        Upsert a user-scoped fact. Updates only when new confidence >= existing.
        """
        existing = (
            self._session.query(LongTermFactModel)
            .filter(
                LongTermFactModel.user_id == user_id,
                LongTermFactModel.fact_key == fact_key,
            )
            .first()
        )
        if existing:
            if confidence >= existing.confidence:
                existing.fact_value = fact_value
                existing.confidence = confidence
                if source_session:
                    existing.source_session = source_session
        else:
            row = LongTermFactModel(
                user_id=user_id,
                fact_key=fact_key,
                fact_value=fact_value,
                confidence=confidence,
                source_session=source_session,
            )
            self._session.add(row)

        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.warning(
                "long_term_fact_upsert_failed",
                extra={"user_id": user_id, "fact_key": fact_key},
                exc_info=True,
            )

    def get_user_facts(self, user_id: str) -> list[dict[str, Any]]:
        """Return all long-term facts for a user as [{key, value, confidence}]."""
        rows = (
            self._session.query(LongTermFactModel)
            .filter(LongTermFactModel.user_id == user_id)
            .all()
        )
        return [
            {"key": r.fact_key, "value": r.fact_value, "confidence": r.confidence}
            for r in rows
        ]

    # ── Episodic conversation memories ───────────────────────────────────────

    def add_conversation_memory(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        embedding: list[float],
        medical_entities: dict[str, Any] | None = None,
    ) -> None:
        """Store a vector-embedded episodic memory of a past turn."""
        row = ConversationMemoryModel(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            embedding=embedding,
            medical_entities=medical_entities or {},
        )
        self._session.add(row)
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.warning(
                "conversation_memory_add_failed",
                extra={"user_id": user_id, "session_id": session_id},
                exc_info=True,
            )

    def search_similar_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Return top-k episodic memories for this user ranked by cosine similarity.
        Uses pgvector <=> operator. Returns [] on error or empty.
        """
        if not query_embedding:
            return []

        try:
            results = self._session.execute(
                text(
                    """
                    SELECT summary, medical_entities, created_at,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                    FROM conversation_memories
                    WHERE user_id = :user_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :k
                    """
                ),
                {
                    "user_id": user_id,
                    "embedding": str(query_embedding),
                    "k": k,
                },
            ).fetchall()

            return [
                {
                    "summary": row.summary,
                    "medical_entities": row.medical_entities,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "similarity": float(row.similarity),
                }
                for row in results
            ]
        except Exception:
            logger.warning(
                "conversation_memory_search_failed",
                extra={"user_id": user_id},
                exc_info=True,
            )
            return []
