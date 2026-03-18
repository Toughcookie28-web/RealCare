from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import sessionmaker

from core.settings import get_settings
from db.schema_contract import DOCUMENT_CHUNK_VECTOR_SQL_TYPE, SCHEMA_AUTHORITY


settings = get_settings()
ROOT = Path(__file__).resolve().parents[1]

try:
    engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
except Exception as exc:
    raise RuntimeError(
        f'Database engine initialization failed in {settings.env} mode. '
        'Persistent database access is required for the current runtime.'
    ) from exc
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_database() -> None:
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
        _assert_alembic_schema_is_current(conn)
        _assert_document_chunk_vector_type(conn)


def get_db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@lru_cache(maxsize=1)
def _expected_alembic_head() -> str:
    config = Config(str(ROOT / 'alembic.ini'))
    config.set_main_option('script_location', str(ROOT / 'alembic'))
    return ScriptDirectory.from_config(config).get_current_head()


def _assert_alembic_schema_is_current(conn: Connection) -> None:
    try:
        current_revision = conn.execute(text('SELECT version_num FROM alembic_version')).scalar_one_or_none()
    except Exception as exc:
        raise RuntimeError(
            f'Database schema is not managed by {SCHEMA_AUTHORITY}. '
            'Run `alembic upgrade head` before starting the app, ingest, or eval jobs.'
        ) from exc
    expected_revision = _expected_alembic_head()
    if current_revision != expected_revision:
        raise RuntimeError(
            f'Database schema is not managed by the current {SCHEMA_AUTHORITY} head. '
            f'Expected revision {expected_revision}, found {current_revision!r}. '
            'Run `alembic upgrade head` before starting the app, ingest, or eval jobs.'
        )


def _assert_document_chunk_vector_type(conn: Connection) -> None:
    vector_type = conn.execute(
        text(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relname = 'document_chunks'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """
        )
    ).scalar_one_or_none()
    if vector_type != DOCUMENT_CHUNK_VECTOR_SQL_TYPE:
        raise RuntimeError(
            'document_chunks.embedding does not match the schema contract. '
            f'Expected {DOCUMENT_CHUNK_VECTOR_SQL_TYPE}, found {vector_type!r}. '
            'Apply the latest Alembic migration and reindex the corpus.'
        )
