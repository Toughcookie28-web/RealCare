import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_contracts import validate_rag_runtime_contract


def test_validate_rag_runtime_contract_reports_missing_dependencies():
    errors = validate_rag_runtime_contract(
        import_checker=lambda name: name not in {"fastapi", "langchain_core", "docling"},
        db_checker=lambda: True,
        cache_dir="/tmp/medigenius-contract-test",
    )

    assert "missing dependency: langchain_core" in errors
