from pathlib import Path


def test_grafana_provisions_postgres_benchmark_datasource_and_db_env():
    datasource = Path("grafana/provisioning/datasources/datasource.yml").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "name: Postgres Benchmark" in datasource
    assert "type: postgres" in datasource
    assert "url: db:5432" in datasource
    assert "database: ${POSTGRES_DB:-medigenius}" in datasource
    assert "user: ${POSTGRES_USER:-postgres}" in datasource
    assert "password: ${POSTGRES_PASSWORD:-postgres}" in datasource
    assert "POSTGRES_USER: ${POSTGRES_USER:-postgres}" in compose
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}" in compose
    assert "POSTGRES_DB: ${POSTGRES_DB:-medigenius}" in compose


def test_grafana_dashboard_includes_benchmark_panels_backed_by_eval_run_tables():
    dashboard = Path("grafana/dashboards/medigenius-dashboard.json").read_text(encoding="utf-8")

    assert '"Tracked Retrieval Recall@5"' in dashboard
    assert '"Tracked Retrieval MRR@5"' in dashboard
    assert '"Tracked RAGAS Context Precision"' in dashboard
    assert '"Tracked RAGAS Faithfulness"' in dashboard
    assert '"Recent Benchmark Runs"' in dashboard
    assert "eval_run_metrics" in dashboard
    assert "eval_runs" in dashboard
    assert '"datasource": {"type": "postgres", "uid": "grafana-postgres-benchmark"}' in dashboard
