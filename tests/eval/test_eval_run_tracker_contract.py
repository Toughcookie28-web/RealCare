import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_run_tracker_module", ROOT / "eval" / "run_tracker.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_run_record_normalizes_snapshot_metadata_for_retrieval_report():
    module = _load_module()
    report = {
        "mode": "pipeline",
        "sample_count": 39,
        "fetch_k": 20,
        "latency_ms_mean": 2767.5,
        "hit_rate@5": 0.8205,
        "snapshot_meta": {
            "git_commit": "abc1234",
            "timestamp": "2026-03-16T12:00:00",
            "note": "phase c pipeline",
        },
    }

    record = module.build_run_record(
        run_id="run-1",
        eval_type="retrieval",
        report=report,
        split="test",
        dataset_path="/app/eval/golden/v1/tier3_rag.jsonl",
        snapshot_path="/app/eval/retrieval_snapshots/pipeline/phase-c-pipeline.json",
    )

    assert record["run_id"] == "run-1"
    assert record["eval_type"] == "retrieval"
    assert record["mode"] == "pipeline"
    assert record["split"] == "test"
    assert record["sample_count"] == 39
    assert record["git_sha"] == "abc1234"
    assert record["dataset_path"] == "/app/eval/golden/v1/tier3_rag.jsonl"
    assert record["snapshot_path"] == "/app/eval/retrieval_snapshots/pipeline/phase-c-pipeline.json"
    assert record["judge_model"] is None
    assert record["metadata_json"] == {
        "fetch_k": 20,
        "snapshot_note": "phase c pipeline",
        "snapshot_timestamp": "2026-03-16T12:00:00",
    }


def test_build_run_record_uses_case_count_for_workflow_reports_without_mode():
    module = _load_module()
    report = {
        "scope": "end_to_end_workflow",
        "case_count": 12,
        "avg_response_keyword_recall": 0.81,
        "table_hit_rate": 0.5,
        "semantic_cache_enabled": False,
    }

    record = module.build_run_record(
        run_id="run-workflow",
        eval_type="workflow",
        report=report,
        split="dev",
        dataset_path="/app/tests/fixtures/rag/workflow_cases.jsonl",
        snapshot_path="/app/eval/workflow_snapshots/workflow-dev.json",
    )

    assert record["mode"] == "workflow"
    assert record["sample_count"] == 12
    assert record["git_sha"] == "unknown"
    assert record["metadata_json"] == {
        "scope": "end_to_end_workflow",
        "semantic_cache_enabled": False,
    }


def test_extract_metric_rows_flattens_numeric_top_level_metrics_without_metadata_fields():
    module = _load_module()
    report = {
        "mode": "pipeline",
        "sample_count": 39,
        "fetch_k": 20,
        "latency_ms_mean": 2767.5,
        "hit_rate@5": 0.8205,
        "precision@5": 0.2667,
        "snapshot_meta": {"git_commit": "abc1234"},
        "slices": {"difficulty:reasoning": {"sample_count": 9, "recall@5": 0.6667}},
        "results": [{"id": "sample-1"}],
    }

    rows = module.extract_metric_rows(run_id="run-1", report=report)

    assert rows == [
        {"run_id": "run-1", "metric_name": "hit_rate@5", "metric_value": 0.8205},
        {"run_id": "run-1", "metric_name": "latency_ms_mean", "metric_value": 2767.5},
        {"run_id": "run-1", "metric_name": "precision@5", "metric_value": 0.2667},
    ]


def test_extract_slice_rows_flattens_slice_metrics_with_slice_sample_counts():
    module = _load_module()
    report = {
        "slices": {
            "content:text": {
                "sample_count": 39,
                "recall@5": 0.6538,
                "section_diversity@5": 0.8308,
            },
            "difficulty:reasoning": {
                "sample_count": 9,
                "mrr@5": 0.4574,
                "recall@5": 0.6667,
            },
        }
    }

    rows = module.extract_slice_rows(run_id="run-1", report=report)

    assert rows == [
        {
            "run_id": "run-1",
            "slice_name": "content:text",
            "metric_name": "recall@5",
            "metric_value": 0.6538,
            "sample_count": 39,
        },
        {
            "run_id": "run-1",
            "slice_name": "content:text",
            "metric_name": "section_diversity@5",
            "metric_value": 0.8308,
            "sample_count": 39,
        },
        {
            "run_id": "run-1",
            "slice_name": "difficulty:reasoning",
            "metric_name": "mrr@5",
            "metric_value": 0.4574,
            "sample_count": 9,
        },
        {
            "run_id": "run-1",
            "slice_name": "difficulty:reasoning",
            "metric_name": "recall@5",
            "metric_value": 0.6667,
            "sample_count": 9,
        },
    ]
