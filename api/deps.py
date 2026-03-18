from __future__ import annotations

import logging
from typing import Generator

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.repositories import (
    ChatRepository,
    VectorRepository,
)
from db.models import SessionModel, DocumentChunkModel
from db.session import SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _raise_persistence_unavailable(kind: str, exc: Exception) -> None:
    logging.getLogger(__name__).warning('Persistent %s repository unavailable', kind, exc_info=exc)
    raise RuntimeError('persistent database unavailable') from exc


def get_chat_repository(db: Session):
    try:
        db.execute(text('SELECT 1'))
        db.query(SessionModel).limit(1).all()
        return ChatRepository(db)
    except Exception as exc:
        _raise_persistence_unavailable('chat', exc)


def get_vector_repository(db: Session):
    try:
        db.execute(text('SELECT 1'))
        db.query(DocumentChunkModel).limit(1).all()
        return VectorRepository(db)
    except Exception as exc:
        _raise_persistence_unavailable('vector', exc)


def get_long_term_memory_repository(db: Session = Depends(get_db)):
    from db.long_term_memory_repository import LongTermMemoryRepository
    return LongTermMemoryRepository(session=db)
