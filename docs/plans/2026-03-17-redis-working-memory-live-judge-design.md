# Redis Working Memory and Live Judge Design

## Goal

Add a Redis-backed working-memory layer and asynchronous live RAG-quality judging without changing the main database schema or blocking the frontend response path.

## Why This Design

The current app already has useful memory structure:

- extracted facts in [agents/memory_agent.py](/home/tough/medical_chatbot/MediGenius/agents/memory_agent.py)
- structured summary fields in [agents/memory_agent.py](/home/tough/medical_chatbot/MediGenius/agents/memory_agent.py)
- `session_intent`, `turn_intent`, `slots`, and `query_context` in [agents/query_rewriter_agent.py](/home/tough/medical_chatbot/MediGenius/agents/query_rewriter_agent.py)
- request `trace_id` and Prometheus metrics in `observability/`

So the right move is not a new independent memory system. The right move is a Redis working layer that reuses the current fact and summary contracts.

## Chosen Approach

We will build two Redis-backed layers plus one async judging path:

1. `recent_turns:{session_id}`
   - rolling short-term conversation memory
   - last `N` turns only
   - optimized for prompt assembly and continuity

2. `core_state:{session_id}`
   - locked core state that should not be lost when old turns slide out
   - optimized for stable medical facts and session goal

3. async live-judge path
   - triggered after the frontend response is returned
   - computes approximate live quality signals
   - updates Prometheus aggregate metrics for Grafana

This keeps the main request path fast while adding better memory stability and near-immediate quality visibility.

## Why Not Other Approaches

### Postgres-first persistent memory

Rejected for now because the user explicitly does not want to change the main DB path yet. The app already persists `summary` and `facts`, but using Postgres as the main working-memory store is slower and would make this iteration harder to isolate.

### Full queue-and-worker architecture from day one

Rejected for v1 because it adds more moving parts than needed. We can start with in-process background tasks and Prometheus aggregates, then move to a Redis queue and worker later if load or latency pressure demands it.

### Benchmark-style live scoring

Rejected because arbitrary frontend traffic does not have gold truth. Live traffic can support groundedness and relevance proxies, but not honest benchmark recall and precision without labeled evidence.

## Reused Existing Contracts

### Existing facts to preserve

From [agents/memory_agent.py](/home/tough/medical_chatbot/MediGenius/agents/memory_agent.py):

- `age`
- `allergy`
- `condition`

### Existing structured summary fields to preserve

From [agents/memory_agent.py](/home/tough/medical_chatbot/MediGenius/agents/memory_agent.py):

- `session_goal`
- `medical_context.conditions`
- `medical_context.drugs`
- `medical_context.procedures`
- `medical_context.populations`
- `key_findings`
- `open_questions`
- `conversation_trajectory`

### Existing intent/query understanding fields to preserve

From [agents/query_rewriter_agent.py](/home/tough/medical_chatbot/MediGenius/agents/query_rewriter_agent.py):

- `session_intent`
- `turn_intent`
- `slots`
- `query_context`

## Redis Data Model

### 1. Recent Turns

Key:

- `recent_turns:{session_id}`

Format:

```json
[
  {
    "role": "user",
    "content": "I am allergic to penicillin",
    "timestamp": "2026-03-17T09:20:00Z",
    "trace_id": "..."
  },
  {
    "role": "assistant",
    "content": "Understood. I will avoid penicillin-based examples.",
    "timestamp": "2026-03-17T09:20:02Z",
    "trace_id": "..."
  }
]
```

Rules:

- keep only the last `N` turns
- set TTL to allow inactive sessions to age out
- store only prompt-relevant fields

### 2. Core State

Key:

- `core_state:{session_id}`

Format:

```json
{
  "version": "v1",
  "updated_at": "2026-03-17T09:20:02Z",
  "medical_facts": {
    "age": null,
    "allergies": [],
    "conditions": []
  },
  "session_intent": "",
  "session_goal": "",
  "medical_context": {
    "conditions": [],
    "drugs": [],
    "procedures": [],
    "populations": []
  },
  "key_findings": [],
  "open_questions": []
}
```

Rules:

- only store stable, high-value information
- no entire transcript blobs
- version the payload so shape changes are safe later
- prefer merge/update semantics instead of destructive overwrite

## Main Request Path

### Synchronous

The frontend request path remains synchronous for:

1. load Redis working memory
2. fall back to existing DB summary/facts if Redis is empty
3. assemble state for the workflow
4. run route/rewrite/retrieve/answer
5. return response to the user
6. append the newest user/assistant turns to `recent_turns`

This means the user still gets the answer from the normal app path without waiting for judging.

## Async Path

### Background tasks after response

After the response is sent, the app should perform:

1. live approximate judging
2. core-state promotion
3. optional summary compaction
4. optional trace event record for later debugging

### Live approximate metrics

These are useful and honest for arbitrary frontend traffic:

- `live_answer_relevance`
- `live_groundedness`
- `live_context_precision_proxy`
- `live_context_coverage_proxy`
- `live_judge_latency`
- `live_judged_requests_total`
- `live_judge_failures_total`

These are not benchmark-truth metrics. They are live quality proxies for Grafana.

## Core-State Promotion Rules

The async updater promotes only stable information into `core_state`.

Examples:

- move `age` into `medical_facts.age`
- merge `allergy` into `medical_facts.allergies`
- merge `condition` into `medical_facts.conditions`
- copy `session_intent` when populated
- parse and merge `session_goal`, `medical_context`, `key_findings`, and `open_questions` from the structured summary

The goal is to preserve important state even when the rolling turn buffer no longer contains the original mention.

## Integration Strategy

### Phase 1

- Redis becomes a hot working-memory overlay
- existing Postgres `summary` and `facts` remain untouched
- if Redis is missing or unavailable, the app falls back cleanly

### Phase 2

- if the working-memory path proves stable, we can decide later whether Postgres should remain archival persistence only or continue as a parallel source of truth

This keeps risk low and avoids coupling the first iteration to DB migrations.

## Error Handling

- Redis unavailable:
  - fall back to current DB-backed summary/facts path
  - set Redis health metric accordingly
- live judge failure:
  - log it
  - increment failure metric
  - never fail the user response
- malformed core state:
  - reset that session's Redis payload to a safe empty default
  - preserve app availability

## Observability

Prometheus remains the aggregation layer for Grafana.

We will add:

- live judge score histograms or summaries
- judge failure counter
- working-memory hit/fallback counters
- optional core-state promotion counters

We will not put `trace_id` into Prometheus labels because that would create bad cardinality.

## Testing Strategy

We will test this in layers:

1. unit tests for Redis recent-turn and core-state stores
2. unit tests for merge/promotion logic
3. unit tests for live-judge result emission and metric recording
4. request-path tests showing:
   - Redis hit path
   - DB fallback path
   - async task does not block response

## Non-Goals For V1

- true benchmark recall and precision for live user traffic
- per-trace Grafana history
- Loki or a separate log store
- full Redis worker/queue infrastructure
- replacing the existing DB persistence path

## Final Recommendation

Build this in two connected slices:

1. Redis working memory:
   - `recent_turns`
   - `core_state`
   - safe request-time reads and response-time writes

2. async live judge:
   - post-response approximate scoring
   - Prometheus aggregate metrics for Grafana

This gives immediate product value with limited architectural risk and stays aligned with the current codebase.
