import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "eval" / "golden" / "v1"
FINAL_PATH = GOLDEN_DIR / "tier3_rag.jsonl"
EXPECTED_IDS = {
    "MANUAL-T3-001",
    "MANUAL-T3-002",
    "MANUAL-T3-004",
    "MANUAL-T3-006",
    "MANUAL-T3-007",
    "MANUAL-T3-008",
    "MANUAL-T3-013",
    "MANUAL-T3-017",
    "MANUAL-T3-021",
    "MANUAL-T3-025",
    "MANUAL-T3-026",
    "MANUAL-PDF-CAND-T3-001",
    "MANUAL-PDF-CAND-T3-003",
    "MANUAL-PDF-CAND-T3-004",
    "MANUAL-PDF-CAND-T3-006",
    "MANUAL-PDF-CAND-T3-008",
    "MANUAL-PDF-CAND-T3-011",
    "MANUAL-PDF-CAND-T3-012",
    "MANUAL-PDF-CAND-T3-013",
    "MANUAL-PDF-CAND-T3-014",
    "MANUAL-PDF-CAND-T3-015",
    "MANUAL-PDF-CAND-T3-017",
    "MANUAL-PDF-CAND-T3-018",
    "MANUAL-PDF-CAND-T3-021",
    "MANUAL-PDF-CAND-T3-023",
    "MANUAL-PDF-CAND-T3-024",
    "MANUAL-PDF-CAND-T3-028",
    "MANUAL-PDF-RH-RW-T3-001",
    "MANUAL-PDF-RH-RW-T3-002",
    "MANUAL-PDF-RH-RW-T3-003",
    "MANUAL-PDF-RH-RW-T3-004",
    "MANUAL-PDF-RH-RW-T3-005",
    "MANUAL-PDF-RH-RW-T3-006",
    "MANUAL-PDF-RH-V2-T3-001",
    "MANUAL-PDF-RH-V2-T3-002",
    "MANUAL-PDF-RH-V2-T3-003",
    "MANUAL-PDF-RH-V2-T3-004",
    "MANUAL-PDF-RH-V2-T3-005",
    "MANUAL-PDF-RH-V2-T3-006",
    "MANUAL-PDF-RH-V2-T3-007",
    "MANUAL-PDF-RH-V2-T3-008",
    "MANUAL-PDF-RH-V2-T3-009",
    "MANUAL-PDF-RH-V2-T3-010",
    "MANUAL-PDF-RH-V2-T3-011",
    "MANUAL-PDF-RH-V2-T3-012",
    "MANUAL-PDF-RH-V2-T3-013",
    "MANUAL-PDF-RH-V2-T3-014",
    "MANUAL-PDF-RH-V2-T3-015",
    "MANUAL-PDF-RH-V2-T3-016",
    "MANUAL-PDF-RH-V2-T3-017",
    "MANUAL-PDF-RH-V2-T3-018",
    "MANUAL-PDF-RH-V2-T3-019",
    "MANUAL-PDF-RH-V2-T3-020",
}
EXPECTED_LANE_COUNTS = {
    "benchmark_lane:baseline": 11,
    "benchmark_lane:medium_hard": 16,
    "benchmark_lane:retrieval_hard": 26,
}
EXPECTED_SPLIT_COUNTS = {
    "dev": 14,
    "test": 39,
}
EXPECTED_SPLIT_LANE_COUNTS = {
    ("dev", "benchmark_lane:baseline"): 3,
    ("dev", "benchmark_lane:medium_hard"): 4,
    ("dev", "benchmark_lane:retrieval_hard"): 7,
    ("test", "benchmark_lane:baseline"): 8,
    ("test", "benchmark_lane:medium_hard"): 12,
    ("test", "benchmark_lane:retrieval_hard"): 19,
}


def _load_source_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


def _load_validate_module():
    return _load_source_module(
        "tier3_final_curated_validate_contract",
        ROOT / "eval" / "validate.py",
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_final_curated_tier3_fixture_is_schema_valid_and_exact():
    validate_module = _load_validate_module()
    assert FINAL_PATH.exists()

    rows = _load_jsonl(FINAL_PATH)
    errors = []
    for row in rows:
        errors.extend(validate_module.validate_sample(row, "rag"))

    assert len(rows) == 53
    assert errors == []
    assert {row["id"] for row in rows} == EXPECTED_IDS
    assert all("rag" in row.get("tags", []) for row in rows)
    assert all("manual_seed" in row.get("tags", []) for row in rows)
    assert all("benchmark_set:final_curated_v1" in row.get("tags", []) for row in rows)
    assert all((row.get("provenance") or {}).get("source_type") == "human_seed" for row in rows)
    assert all((row.get("provenance") or {}).get("review_status") == "approved" for row in rows)


def test_final_curated_tier3_fixture_has_expected_lane_balance_and_no_candidate_tags():
    rows = _load_jsonl(FINAL_PATH)
    lane_counts = Counter(
        next(tag for tag in row.get("tags", []) if tag.startswith("benchmark_lane:"))
        for row in rows
    )

    assert lane_counts == EXPECTED_LANE_COUNTS
    assert all(
        sum(tag.startswith("benchmark_lane:") for tag in row.get("tags", [])) == 1
        for row in rows
    )
    assert all("manual_seed_candidate" not in row.get("tags", []) for row in rows)
    assert all(
        not any(tag.startswith("candidate_batch:") for tag in row.get("tags", []))
        for row in rows
    )
    assert all(
        not any(tag.startswith("revision_of:") for tag in row.get("tags", []))
        for row in rows
    )


def test_final_curated_tier3_fixture_has_deterministic_dev_test_split():
    rows = _load_jsonl(FINAL_PATH)
    split_counts = Counter(row["split"] for row in rows)
    split_lane_counts = Counter(
        (
            row["split"],
            next(tag for tag in row.get("tags", []) if tag.startswith("benchmark_lane:")),
        )
        for row in rows
    )

    assert split_counts == EXPECTED_SPLIT_COUNTS
    assert split_lane_counts == EXPECTED_SPLIT_LANE_COUNTS


def test_final_curated_tier3_fixture_has_evidence_roles_and_stable_anchors():
    rows = _load_jsonl(FINAL_PATH)

    for row in rows:
        evidence = row.get("evidence")
        assert isinstance(evidence, dict)
        required = evidence.get("required")
        supporting = evidence.get("supporting")
        hard_negatives = evidence.get("hard_negative_candidates")

        assert isinstance(required, list) and required
        assert isinstance(supporting, list)
        assert isinstance(hard_negatives, list)

        gold_chunk_ids = set(row.get("ground_truth_chunk_ids") or [])
        required_ids = {anchor["chunk_id"] for anchor in required}
        supporting_ids = {anchor["chunk_id"] for anchor in supporting}
        negative_ids = {anchor["chunk_id"] for anchor in hard_negatives}

        assert required_ids <= gold_chunk_ids
        assert supporting_ids <= gold_chunk_ids
        assert required_ids
        assert not (negative_ids & gold_chunk_ids)

        for anchor in required + supporting + hard_negatives:
            assert set(anchor) == {
                "chunk_id",
                "doc_id",
                "page",
                "section",
                "content_type",
                "anchor_text",
            }
            assert anchor["anchor_text"]


def test_retrieval_hard_rows_have_hard_negative_candidates():
    rows = _load_jsonl(FINAL_PATH)
    retrieval_hard_rows = [
        row for row in rows if "benchmark_lane:retrieval_hard" in row.get("tags", [])
    ]

    assert retrieval_hard_rows
    assert all(row["evidence"]["hard_negative_candidates"] for row in retrieval_hard_rows)
