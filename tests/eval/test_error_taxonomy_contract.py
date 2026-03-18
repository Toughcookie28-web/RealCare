import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_error_taxonomy_lists_current_rag_failure_modes():
    text = (ROOT / "docs" / "evals" / "error-taxonomy.md").read_text(encoding="utf-8")

    assert "schema_mismatch" in text
    assert "runtime_fallback_masked_failure" in text
    assert "parser_artifact_permission_failure" in text
    assert "stale_semantic_cache_hit" in text
    assert "gate_contract_failure" in text


def test_review_sample_file_contains_reviewed_rows():
    path = ROOT / "eval" / "review_samples" / "rag_trace_review.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) >= 8
    assert all(row["review_status"] == "reviewed" for row in rows)
