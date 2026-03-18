from pathlib import Path


def test_services_use_explicit_cache_and_data_mounts_only():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "- ./:/app" not in compose
    assert "- ./eval:/app/eval" not in compose
    assert "- ./scripts:/app/scripts" not in compose
    assert "- ./data:/app/data:ro" in compose
    assert "- ./.cache/embeddings:/app/.cache/embeddings" in compose
    assert "- ./eval/golden:/app/eval/golden" in compose
    assert "- ./eval/ragas_snapshots:/app/eval/ragas_snapshots" in compose
    assert "- ./eval/retrieval_snapshots:/app/eval/retrieval_snapshots" in compose
    assert "- ./eval/workflow_snapshots:/app/eval/workflow_snapshots" in compose


def test_cache_contract_uses_explicit_docling_and_rapidocr_paths():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    bootstrap = Path("tools/embedding_bootstrap.py").read_text(encoding="utf-8")
    parser = Path("tools/pdf_parser.py").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "EMBEDDING_CACHE_DIR=/app/.cache/embeddings" in dockerfile
    assert 'DOCLING_ARTIFACTS_DIRNAME = "docling_artifacts"' in bootstrap
    assert 'RAPIDOCR_MODELS_DIRNAME = "rapidocr/models"' in bootstrap
    assert "ensure_docling_artifacts(" in parser
    assert "EMBEDDING_CACHE_DIR: /app/.cache/embeddings" in compose
    assert "DOCLING_ARTIFACTS_DIR: /app/.cache/embeddings/docling_artifacts" in compose
    assert "${EMBEDDING_CACHE_DIR:-/app/.cache/embeddings}" not in compose
