import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str, injected_modules: dict[str, types.ModuleType]):
    original_modules = {name: sys.modules.get(name) for name in injected_modules}
    sys.modules.update(injected_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_get_chat_repository_raises_when_db_is_unavailable():
    sqlalchemy_module = types.ModuleType("sqlalchemy")
    sqlalchemy_module.text = lambda value: value

    sqlalchemy_orm_module = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm_module.Session = object

    db_repositories_module = types.ModuleType("db.repositories")

    class ChatRepository:
        def __init__(self, db):
            self.db = db

    class VectorRepository:
        def __init__(self, db):
            self.db = db

    db_repositories_module.ChatRepository = ChatRepository
    db_repositories_module.VectorRepository = VectorRepository
    db_repositories_module.InMemoryChatRepository = object
    db_repositories_module.InMemoryVectorRepository = object

    db_models_module = types.ModuleType("db.models")
    db_models_module.SessionModel = type("SessionModel", (), {})
    db_models_module.DocumentChunkModel = type("DocumentChunkModel", (), {})

    db_session_module = types.ModuleType("db.session")
    db_session_module.SessionLocal = lambda: None

    module = _load_module(
        "batch_a_api_deps",
        "api/deps.py",
        {
            "sqlalchemy": sqlalchemy_module,
            "sqlalchemy.orm": sqlalchemy_orm_module,
            "db.repositories": db_repositories_module,
            "db.models": db_models_module,
            "db.session": db_session_module,
        },
    )

    class BrokenSession:
        def execute(self, _query):
            raise RuntimeError("database offline")

    with pytest.raises(RuntimeError, match="persistent database unavailable"):
        module.get_chat_repository(BrokenSession())


def test_session_module_no_longer_contains_sqlite_in_memory_fallback():
    text = (ROOT / "db/session.py").read_text(encoding="utf-8")
    assert "sqlite+pysqlite:///:memory:" not in text


def test_create_pending_review_requires_database():
    sqlalchemy_orm_module = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm_module.Session = object

    module = _load_module(
        "batch_a_core_hitl",
        "core/hitl.py",
        {
            "sqlalchemy.orm": sqlalchemy_orm_module,
        },
    )

    with pytest.raises(RuntimeError, match="persistent HITL database required"):
        module.create_pending_review(
            approval_id="approval-1",
            session_id="session-1",
            trace_id="trace-1",
            question="Need approval",
            db=None,
        )
