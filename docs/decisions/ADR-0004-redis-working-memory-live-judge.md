# ADR-0004: Redis Working Memory Overlay and Async Live Judge

## Status

Accepted

## Context

The app already had two useful but separate memory surfaces:

- PostgreSQL-backed `summary` and `facts`
- short in-process conversation history used during workflow execution

That left two gaps:

1. important state such as allergies, conditions, and session goal could drift out of the active turn window
2. live frontend traffic produced runtime metrics, but not post-response quality signals in Grafana

The user also explicitly wanted to avoid changing the main database schema for this iteration.

## Decision

We added a Redis overlay rather than replacing the current database-backed memory path.

The overlay has two layers:

1. `recent_turns:{session_id}`
   - rolling short-term memory
   - prompt-oriented
   - bounded and expiring

2. `core_state:{session_id}`
   - locked high-value memory
   - versioned shape
   - stores stable medical facts and session context

We also added an async post-response live judge path:

- runs after the frontend response is returned
- emits Prometheus aggregate metrics for Grafana
- computes approximate live quality proxies, not benchmark-truth eval scores

## Consequences

### Positive

- Preserves important medical facts and session intent beyond the recent-turn window.
- Keeps the main request path fast by moving judgment and promotion work post-response.
- Improves Grafana with live quality signals from real frontend traffic.
- Avoids main DB schema churn for the first iteration.
- Reuses the current `facts`, `summary`, and `session_intent` contracts instead of inventing a second ontology.

### Negative

- Redis is now another runtime dependency, even though the app still degrades to the DB-backed path.
- Live proxy scores can be useful for monitoring, but they must not be confused with offline benchmark metrics.
- The working-memory overlay introduces one more place where memory behavior can diverge if not kept documented and tested.

## Guardrails

- Redis remains an overlay, not the source of truth.
- If Redis fails or is empty, the app must continue using PostgreSQL summary/facts/history.
- Prometheus labels for live judge metrics must stay low-cardinality.
- Live judge outputs must be described as proxies, not retrieval recall/precision or full RAGAS benchmark results.
- Core-state promotion must remain conservative and versioned.
