import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import observability.metrics as metrics


def test_live_judge_metrics_exist():
    assert metrics.LIVE_JUDGED_REQUESTS._name == "medigenius_live_judged_requests"
    assert metrics.LIVE_JUDGE_FAILURES._name == "medigenius_live_judge_failures"
    assert metrics.LIVE_JUDGE_LATENCY._name == "medigenius_live_judge_latency_ms"
    assert metrics.LIVE_ANSWER_RELEVANCE._name == "medigenius_live_answer_relevance_score"
    assert metrics.LIVE_GROUNDEDNESS._name == "medigenius_live_groundedness_score"
    assert metrics.LIVE_CONTEXT_PRECISION_PROXY._name == "medigenius_live_context_precision_proxy"
    assert metrics.LIVE_CONTEXT_COVERAGE_PROXY._name == "medigenius_live_context_coverage_proxy"
