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
    import json

    dashboard_text = Path("grafana/dashboards/medigenius-dashboard.json").read_text(encoding="utf-8")
    dashboard = json.loads(dashboard_text)

    panel_titles = [p.get("title", "") for p in dashboard.get("panels", [])]

    assert "Tracked Retrieval Recall@5" in panel_titles
    assert "Tracked Retrieval MRR@5" in panel_titles
    assert "Tracked RAGAS Context Precision" in panel_titles
    assert "Tracked RAGAS Faithfulness" in panel_titles
    assert "Recent Benchmark Runs" in panel_titles
    assert "eval_run_metrics" in dashboard_text
    assert "eval_runs" in dashboard_text

    # Verify at least one panel uses the postgres benchmark datasource
    postgres_panels = [
        p for p in dashboard.get("panels", [])
        if isinstance(p.get("datasource"), dict)
        and p["datasource"].get("type") == "postgres"
        and p["datasource"].get("uid") == "grafana-postgres-benchmark"
    ]
    assert postgres_panels, "No panels use the grafana-postgres-benchmark datasource"
