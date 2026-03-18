from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile

REQUIRED_RAG_IMPORTS = (
    "fastapi",
    "langchain_core",
    "langgraph",
    "docling",
)

REQUIRED_CACHE_PATHS = (
    "",
    "docling_artifacts",
    "rapidocr/models",
    "sentence_transformers",
)


def validate_rag_runtime_contract(import_checker=None, db_checker=None, cache_dir=None) -> list[str]:
    import_checker = import_checker or _default_import_checker
    db_checker = db_checker or _default_db_checker
    cache_dir = cache_dir or _default_cache_dir()

    errors: list[str] = []

    for name in REQUIRED_RAG_IMPORTS:
        if not import_checker(name):
            errors.append(f"missing dependency: {name}")

    if not _cache_dir_is_writable(cache_dir):
        errors.append(f"cache directory not writable: {cache_dir}")

    try:
        db_ok = bool(db_checker())
    except Exception as exc:
        errors.append(f"database check failed: {exc}")
    else:
        if not db_ok:
            errors.append("database check failed")

    return errors


def _default_import_checker(module_name: str) -> bool:
    try:
        import_module(module_name)
        return True
    except Exception:
        return False


def _default_db_checker() -> bool:
    from db.session import init_database

    init_database()
    return True


def _default_cache_dir() -> str:
    return os.environ.get("EMBEDDING_CACHE_DIR", "/tmp/medigenius-embeddings")


def _cache_dir_is_writable(cache_dir: str | Path) -> bool:
    root = Path(cache_dir)
    try:
        for relative_path in REQUIRED_CACHE_PATHS:
            target = root / relative_path
            target.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target, delete=True):
                pass
        return True
    except Exception:
        return False
