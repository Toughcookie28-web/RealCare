from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_CONTRACT = ROOT / "db" / "schema_contract.py"
SETTINGS_FILE = ROOT / "core" / "settings.py"
MODELS_FILE = ROOT / "db" / "models.py"
SESSION_FILE = ROOT / "db" / "session.py"
WORKFLOW_FILE = ROOT / ".github" / "workflows" / "eval.yml"
README_FILE = ROOT / "README.md"
COMPOSE_FILE = ROOT / "docker-compose.yml"
MIGRATION_FILE = ROOT / "alembic" / "versions" / "0003_align_document_chunks_vector_contract.py"

namespace: dict[str, object] = {}
exec(SCHEMA_CONTRACT.read_text(encoding="utf-8"), namespace)

DOCUMENT_CHUNK_VECTOR_DIM = namespace["DOCUMENT_CHUNK_VECTOR_DIM"]
default_embedding_dim = namespace["default_embedding_dim"]


def test_embedding_dim_matches_document_chunk_schema_contract():
    settings_text = SETTINGS_FILE.read_text(encoding="utf-8")

    assert default_embedding_dim() == DOCUMENT_CHUNK_VECTOR_DIM
    assert "default_factory=default_embedding_dim" in settings_text
    assert "require_document_chunk_vector_dim" in settings_text


def test_document_chunk_model_uses_the_schema_contract():
    models_text = MODELS_FILE.read_text(encoding="utf-8")

    assert "Vector(DOCUMENT_CHUNK_VECTOR_DIM)" in models_text
    assert "embedding_json" not in models_text


def test_runtime_schema_authority_is_alembic_only():
    session_text = SESSION_FILE.read_text(encoding="utf-8")
    migration_text = MIGRATION_FILE.read_text(encoding="utf-8")

    assert "create_all(" not in session_text
    assert "alembic_version" in session_text
    assert "alembic upgrade head" in session_text
    assert "TARGET_VECTOR_DIM = 768" in migration_text
    assert "down_revision = '0002_add_hitl_reviews'" in migration_text


def test_schema_contract_is_documented_and_checked_in_ci():
    workflow_text = WORKFLOW_FILE.read_text(encoding="utf-8")
    readme_text = README_FILE.read_text(encoding="utf-8")
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "python3 -m pytest tests/smoke/test_schema_contract.py -v" in workflow_text
    assert "Alembic is the only schema authority." in readme_text
    assert "document_chunks.embedding" in readme_text
    assert "python3 -m alembic -c alembic.ini upgrade head" in readme_text
    assert "python3 -m alembic -c /app/alembic.ini upgrade head" in readme_text
    assert "EMBEDDING_DIM: ${EMBEDDING_DIM:-768}" in compose_text
