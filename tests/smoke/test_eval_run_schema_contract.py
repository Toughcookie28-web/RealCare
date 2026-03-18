from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_eval_run_models_exist_with_expected_tables_and_columns():
    models = import_module("db.models")

    assert hasattr(models, "EvalRunModel")
    assert hasattr(models, "EvalRunMetricModel")
    assert hasattr(models, "EvalRunSliceModel")

    tables = models.Base.metadata.tables
    assert "eval_runs" in tables
    assert "eval_run_metrics" in tables
    assert "eval_run_slices" in tables

    eval_runs = tables["eval_runs"].columns.keys()
    assert set(eval_runs) >= {
        "run_id",
        "created_at",
        "eval_type",
        "mode",
        "split",
        "sample_count",
        "git_sha",
        "dataset_path",
        "snapshot_path",
        "judge_model",
        "metadata_json",
    }

    metric_cols = tables["eval_run_metrics"].columns.keys()
    assert set(metric_cols) >= {"id", "run_id", "metric_name", "metric_value"}

    slice_cols = tables["eval_run_slices"].columns.keys()
    assert set(slice_cols) >= {"id", "run_id", "slice_name", "metric_name", "metric_value", "sample_count"}


def test_eval_run_tracking_migration_exists_and_creates_expected_tables():
    migration = ROOT / "alembic" / "versions" / "0005_add_eval_run_tracking.py"
    assert migration.exists()

    source = migration.read_text(encoding="utf-8")
    assert "create_table('eval_runs'" in source or 'create_table("eval_runs"' in source
    assert "create_table('eval_run_metrics'" in source or 'create_table("eval_run_metrics"' in source
    assert "create_table('eval_run_slices'" in source or 'create_table("eval_run_slices"' in source
