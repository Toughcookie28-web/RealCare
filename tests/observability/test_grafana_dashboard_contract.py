import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_dashboard_includes_live_judge_panels():
    dashboard = json.loads((ROOT / "grafana" / "dashboards" / "medigenius-dashboard.json").read_text())
    titles = {panel["title"] for panel in dashboard["panels"]}

    assert "Live Judged Request Rate" in titles
    assert "Live Judge Failure Rate" in titles
    assert "Live Answer Relevance (avg 5m)" in titles
    assert "Live Groundedness (avg 5m)" in titles
    assert "Live Context Precision Proxy (avg 5m)" in titles
    assert "Live Context Coverage Proxy (avg 5m)" in titles
    assert "Live Judge Latency P95 (ms)" in titles
