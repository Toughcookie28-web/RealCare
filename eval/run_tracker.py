from __future__ import annotations

from typing import Any
from uuid import uuid4

_METRIC_METADATA_KEYS = {
    "sample_count",
    "case_count",
    "fetch_k",
    "top_k",
}
_REPORT_METADATA_EXCLUDE_KEYS = {
    "mode",
    "judge_model",
    "results",
    "slices",
    "snapshot_meta",
}


def _resolve_mode(eval_type: str, report: dict[str, Any]) -> str:
    mode = str(report.get("mode") or "").strip()
    if mode:
        return mode
    if eval_type == "workflow":
        return "workflow"
    return "unknown"


def _resolve_sample_count(report: dict[str, Any]) -> int:
    for key in ("sample_count", "case_count"):
        value = report.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    results = report.get("results")
    if isinstance(results, list):
        return len(results)
    return 0


def _is_numeric_metric(name: str, value: Any) -> bool:
    return name not in _METRIC_METADATA_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)


def extract_metric_rows(*, run_id: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name in sorted(report):
        value = report[metric_name]
        if _is_numeric_metric(metric_name, value):
            rows.append(
                {
                    "run_id": run_id,
                    "metric_name": metric_name,
                    "metric_value": float(value),
                }
            )
    return rows


def extract_slice_rows(*, run_id: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slices = report.get("slices")
    if not isinstance(slices, dict):
        return rows

    for slice_name in sorted(slices):
        slice_report = slices[slice_name]
        if not isinstance(slice_report, dict):
            continue
        sample_count = slice_report.get("sample_count")
        if isinstance(sample_count, int) and not isinstance(sample_count, bool):
            resolved_sample_count = sample_count
        else:
            resolved_sample_count = 0
        for metric_name in sorted(slice_report):
            value = slice_report[metric_name]
            if not _is_numeric_metric(metric_name, value):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "slice_name": slice_name,
                    "metric_name": metric_name,
                    "metric_value": float(value),
                    "sample_count": resolved_sample_count,
                }
            )
    return rows


def _build_metadata_json(report: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metric_names = {row["metric_name"] for row in extract_metric_rows(run_id="metadata-probe", report=report)}

    for key, value in report.items():
        if key in _REPORT_METADATA_EXCLUDE_KEYS or key in metric_names or key in {"sample_count", "case_count"}:
            continue
        metadata[key] = value

    snapshot_meta = report.get("snapshot_meta")
    if isinstance(snapshot_meta, dict):
        timestamp = snapshot_meta.get("timestamp")
        note = snapshot_meta.get("note")
        if timestamp:
            metadata["snapshot_timestamp"] = timestamp
        if note:
            metadata["snapshot_note"] = note

    return metadata


def build_run_record(
    *,
    run_id: str,
    eval_type: str,
    report: dict[str, Any],
    split: str,
    dataset_path: str,
    snapshot_path: str,
    judge_model: str | None = None,
) -> dict[str, Any]:
    snapshot_meta = report.get("snapshot_meta")
    git_sha = "unknown"
    if isinstance(snapshot_meta, dict):
        git_commit = snapshot_meta.get("git_commit")
        if isinstance(git_commit, str) and git_commit.strip():
            git_sha = git_commit.strip()

    return {
        "run_id": run_id,
        "eval_type": eval_type,
        "mode": _resolve_mode(eval_type, report),
        "split": split,
        "sample_count": _resolve_sample_count(report),
        "git_sha": git_sha,
        "dataset_path": dataset_path,
        "snapshot_path": snapshot_path,
        "judge_model": judge_model or report.get("judge_model"),
        "metadata_json": _build_metadata_json(report),
    }


def persist_saved_run(
    *,
    eval_type: str,
    report: dict[str, Any],
    split: str,
    dataset_path: str,
    snapshot_path: str,
    session_factory,
    judge_model: str | None = None,
) -> str:
    from db.repositories import EvalRunRepository

    run_id = str(uuid4())
    record = build_run_record(
        run_id=run_id,
        eval_type=eval_type,
        report=report,
        split=split,
        dataset_path=dataset_path,
        snapshot_path=snapshot_path,
        judge_model=judge_model,
    )
    metric_rows = extract_metric_rows(run_id=run_id, report=report)
    slice_rows = extract_slice_rows(run_id=run_id, report=report)

    db = session_factory()
    try:
        EvalRunRepository(db).record_run(
            run_record=record,
            metric_rows=metric_rows,
            slice_rows=slice_rows,
        )
        return run_id
    finally:
        db.close()
