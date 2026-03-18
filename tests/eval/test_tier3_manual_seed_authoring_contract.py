import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_manual_seed_authoring_artifacts_are_preserved() -> None:
    assert (ROOT / "eval" / "tier3_manual_seed_pdf_authoring.py").exists()
    assert (ROOT / "scripts" / "build_tier3_manual_seed_authoring_index.py").exists()
    assert (ROOT / "eval" / "golden" / "v1" / "tier3_manual_seed_authoring_index.jsonl").exists()
    assert not (ROOT / "eval" / "generate_tier3_rag.py").exists()
    assert not (ROOT / "eval" / "golden" / "v1" / "tier3_rag.raw.jsonl").exists()


def test_manual_seed_authoring_index_rows_have_expected_fields() -> None:
    path = ROOT / "eval" / "golden" / "v1" / "tier3_manual_seed_authoring_index.jsonl"
    first_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_row) == {
        "article_title",
        "page_start",
        "page_end",
        "local_headings",
        "body_preview",
        "chunk_ids",
        "chunk_pages",
        "chunk_sections",
    }
