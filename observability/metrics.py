from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest


REQUEST_COUNT = Counter('medigenius_http_requests_total', 'Total HTTP requests', ['method', 'path', 'status'])
REQUEST_LATENCY = Histogram('medigenius_http_request_latency_seconds', 'HTTP request latency', ['method', 'path'])
NODE_LATENCY = Histogram('medigenius_node_latency_ms', 'Node latency in ms', ['node'])
NODE_ERRORS = Counter('medigenius_node_errors_total', 'Node errors', ['node', 'error_code'])
CACHE_HITS = Counter('medigenius_semantic_cache_hits_total', 'Semantic cache hits')
CACHE_MISSES = Counter('medigenius_semantic_cache_misses_total', 'Semantic cache misses')
CACHE_TOOL_HITS = Counter('medigenius_tool_cache_hits_total', 'Tool/API cache hits', ['tool'])
CACHE_TOOL_MISSES = Counter('medigenius_tool_cache_misses_total', 'Tool/API cache misses', ['tool'])
REDIS_UP = Gauge('medigenius_redis_up', 'Whether Redis is connected (1=up, 0=down)')
LIVE_JUDGED_REQUESTS = Counter('medigenius_live_judged_requests_total', 'Live judged requests', ['route'])
LIVE_JUDGE_FAILURES = Counter('medigenius_live_judge_failures_total', 'Live judge failures', ['route'])
LIVE_JUDGE_LATENCY = Histogram('medigenius_live_judge_latency_ms', 'Live judge latency in ms', ['route'])
LIVE_ANSWER_RELEVANCE = Histogram('medigenius_live_answer_relevance_score', 'Live answer relevance proxy', ['route'])
LIVE_GROUNDEDNESS = Histogram('medigenius_live_groundedness_score', 'Live groundedness proxy', ['route'])
LIVE_CONTEXT_PRECISION_PROXY = Histogram(
    'medigenius_live_context_precision_proxy',
    'Live context precision proxy',
    ['route'],
)
LIVE_CONTEXT_COVERAGE_PROXY = Histogram(
    'medigenius_live_context_coverage_proxy',
    'Live context coverage proxy',
    ['route'],
)


def metrics_payload() -> bytes:
    return generate_latest()
