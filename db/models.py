from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from db.schema_contract import CONVERSATION_MEMORY_VECTOR_DIM, DOCUMENT_CHUNK_VECTOR_DIM


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    messages: Mapped[list["MessageModel"]] = relationship("MessageModel", back_populates="session")


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    role: Mapped[str] = mapped_column(String(16), index=True)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    session: Mapped[SessionModel] = relationship("SessionModel", back_populates="messages")


class ConversationSummaryModel(Base):
    __tablename__ = "conversation_summaries"

    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class UserFactModel(Base):
    __tablename__ = "user_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    fact_key: Mapped[str] = mapped_column(String(128), index=True)
    fact_value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class HitlReviewModel(Base):
    __tablename__ = "hitl_reviews"

    approval_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[Any | None] = mapped_column(Vector(DOCUMENT_CHUNK_VECTOR_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_document_chunks_doc_id_page", "doc_id", "page"),)


class UserModel(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class LongTermFactModel(Base):
    """User-scoped facts that persist across all sessions."""

    __tablename__ = "long_term_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    fact_key: Mapped[str] = mapped_column(String(128), index=True)
    fact_value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(default=0.6)
    source_session: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_long_term_facts_user_key", "user_id", "fact_key"),)


class ConversationMemoryModel(Base):
    """Vector-embedded episodic memory of past conversation turns."""

    __tablename__ = "conversation_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any | None] = mapped_column(Vector(CONVERSATION_MEMORY_VECTOR_DIM), nullable=True)
    medical_entities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (Index("ix_conversation_memories_user_id_created", "user_id", "created_at"),)


class EvalRunModel(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    eval_type: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(String(64), index=True)
    split: Mapped[str] = mapped_column(String(32), index=True)
    sample_count: Mapped[int] = mapped_column(Integer)
    git_sha: Mapped[str] = mapped_column(String(64), default="unknown")
    dataset_path: Mapped[str] = mapped_column(Text)
    snapshot_path: Mapped[str] = mapped_column(Text)
    judge_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    metrics: Mapped[list["EvalRunMetricModel"]] = relationship(
        "EvalRunMetricModel",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    slices: Mapped[list["EvalRunSliceModel"]] = relationship(
        "EvalRunSliceModel",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class EvalRunMetricModel(Base):
    __tablename__ = "eval_run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.run_id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_value: Mapped[float] = mapped_column(Float)

    run: Mapped[EvalRunModel] = relationship("EvalRunModel", back_populates="metrics")


class EvalRunSliceModel(Base):
    __tablename__ = "eval_run_slices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.run_id"), index=True)
    slice_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer)

    run: Mapped[EvalRunModel] = relationship("EvalRunModel", back_populates="slices")
