import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_rag_shadow import build_shadow_steps


def test_shadow_steps_include_reindex_and_live_eval_reports():
    steps = build_shadow_steps()
    labels = [step.label for step in steps]
    assert "reindex_pdf" in labels
    assert "retrieval_eval_live" in labels
    assert "chunking_eval_live" in labels


def test_shadow_steps_are_non_blocking_and_emit_artifact_hints():
    steps = {step.label: step for step in build_shadow_steps()}

    assert steps["reindex_pdf"].blocking is False
    assert steps["retrieval_eval_live"].artifact_hint == "eval/retrieval_snapshots/live-shadow.json"
    assert steps["chunking_eval_live"].artifact_hint == "eval/chunking_snapshots/live-shadow.json"


def test_workflow_registers_non_blocking_live_runtime_jobs():
    workflow = Path(".github/workflows/eval.yml").read_text(encoding="utf-8")

    assert "live-runtime-smoke:" in workflow
    assert "continue-on-error: true" in workflow
    assert "python3 scripts/run_live_runtime_smoke.py" in workflow
    assert "python3 scripts/run_live_rag_shadow.py" in workflow
