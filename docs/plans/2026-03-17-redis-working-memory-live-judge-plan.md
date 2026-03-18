# Redis Working Memory and Live Judge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Redis-backed recent-turn and locked-core working memory, then add asynchronous live approximate RAG-quality judging that feeds Grafana through Prometheus without changing the main DB schema.

**Architecture:** Reuse the current `facts`, `summary`, and `session_intent` contracts. Add a Redis working-memory layer as a hot overlay on top of the existing DB-backed path, then attach a post-response async judge that records approximate live quality metrics. Keep all new behavior failure-safe so frontend responses still work when Redis or the judge path fails.

**Tech Stack:** FastAPI, existing workflow service, Redis, Prometheus, Python background tasks, pytest

---

### Task 1: Add configuration for working memory and live judging

**Files:**
- Modify: `core/settings.py`
- Modify: `.env.example`
- Test: `tests/core/test_settings_contract.py`

**Step 1: Write the failing test**

Add tests that assert settings expose:

- Redis recent-turn TTL
- Redis core-state TTL
- recent-turn max length
- live judge enabled flag
- live judge sampling ratio

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v`

Expected: FAIL because the new settings fields do not exist yet.

**Step 3: Write minimal implementation**

Add typed settings fields with safe defaults in `core/settings.py` and document them in `.env.example`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_settings_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py .env.example tests/core/test_settings_contract.py
git commit -m "feat: add redis working memory settings"
```

### Task 2: Create Redis working-memory schemas and helpers

**Files:**
- Create: `tools/working_memory.py`
- Test: `tests/tools/test_working_memory_contract.py`

**Step 1: Write the failing test**

Add tests for:

- default empty `core_state`
- recent-turn trimming to max length
- merge behavior for `age`, `allergies`, `conditions`
- versioned payload shape

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_working_memory_contract.py -v`

Expected: FAIL because `tools/working_memory.py` does not exist.

**Step 3: Write minimal implementation**

Create:

- `build_empty_core_state()`
- `normalize_recent_turn()`
- `trim_recent_turns()`
- `merge_core_state(existing, facts, session_intent, summary_text)`

Keep the schema aligned with the validated design.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_working_memory_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/working_memory.py tests/tools/test_working_memory_contract.py
git commit -m "feat: add redis working memory schema helpers"
```

### Task 3: Extend Redis client support beyond exact cache

**Files:**
- Modify: `tools/redis_client.py`
- Test: `tests/tools/test_redis_client_contract.py`

**Step 1: Write the failing test**

Add tests for:

- JSON get/set helper
- list push + trim helper for recent turns
- graceful fallback when Redis is unavailable

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_redis_client_contract.py -v`

Expected: FAIL because those helpers do not exist yet.

**Step 3: Write minimal implementation**

Add:

- `get_json(key)`
- `set_json(key, payload, ttl)`
- `append_json_list(key, item, max_length, ttl)`

Do not remove or break `ExactCache`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_redis_client_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/redis_client.py tests/tools/test_redis_client_contract.py
git commit -m "feat: add redis json working memory helpers"
```

### Task 4: Add a working-memory service layer

**Files:**
- Create: `tools/working_memory_service.py`
- Test: `tests/tools/test_working_memory_service.py`

**Step 1: Write the failing test**

Add tests for:

- loading recent turns from Redis
- loading core state from Redis
- DB fallback behavior when Redis is empty
- appending new turns after a response

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_working_memory_service.py -v`

Expected: FAIL because the service does not exist.

**Step 3: Write minimal implementation**

Create a service that:

- loads `recent_turns:{session_id}`
- loads `core_state:{session_id}`
- falls back to existing `chat_repo.get_history/get_summary/list_facts`
- returns a normalized memory bundle for the workflow
- appends new turns back to Redis

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_working_memory_service.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/working_memory_service.py tests/tools/test_working_memory_service.py
git commit -m "feat: add redis working memory service"
```

### Task 5: Add summary parsing and core-state promotion logic

**Files:**
- Modify: `tools/working_memory.py`
- Test: `tests/tools/test_working_memory_promotion.py`

**Step 1: Write the failing test**

Add tests that promote:

- `age` from `facts`
- `allergy` into `medical_facts.allergies`
- `condition` into `medical_facts.conditions`
- `session_intent`
- `session_goal`
- `medical_context` sections parsed from the formatted summary text

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_working_memory_promotion.py -v`

Expected: FAIL because promotion is incomplete.

**Step 3: Write minimal implementation**

Implement deterministic parsing of the current formatted summary text produced by `memory_agent`, then merge those values into the versioned core-state object.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_working_memory_promotion.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/working_memory.py tests/tools/test_working_memory_promotion.py
git commit -m "feat: add core state promotion logic"
```

### Task 6: Add live-judge metric definitions

**Files:**
- Modify: `observability/metrics.py`
- Test: `tests/observability/test_live_judge_metrics_contract.py`

**Step 1: Write the failing test**

Add tests that assert the metrics module exports:

- `LIVE_JUDGED_REQUESTS`
- `LIVE_JUDGE_FAILURES`
- `LIVE_JUDGE_LATENCY`
- `LIVE_ANSWER_RELEVANCE`
- `LIVE_GROUNDEDNESS`
- `LIVE_CONTEXT_PRECISION_PROXY`
- `LIVE_CONTEXT_COVERAGE_PROXY`

**Step 2: Run test to verify it fails**

Run: `pytest tests/observability/test_live_judge_metrics_contract.py -v`

Expected: FAIL because these metrics are missing.

**Step 3: Write minimal implementation**

Add low-cardinality Prometheus counters and histograms. Keep labels limited to safe fields like route.

**Step 4: Run test to verify it passes**

Run: `pytest tests/observability/test_live_judge_metrics_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add observability/metrics.py tests/observability/test_live_judge_metrics_contract.py
git commit -m "feat: add live judge metrics"
```

### Task 7: Create the live-judge service

**Files:**
- Create: `observability/live_judge.py`
- Test: `tests/observability/test_live_judge.py`

**Step 1: Write the failing test**

Add tests for:

- payload normalization from workflow output
- no-op when judging is disabled
- metric emission on success
- failure metric emission on judge error

**Step 2: Run test to verify it fails**

Run: `pytest tests/observability/test_live_judge.py -v`

Expected: FAIL because the service does not exist.

**Step 3: Write minimal implementation**

Create a judge service that accepts:

- question
- answer
- retrieved context
- route
- trace_id

Then computes approximate scores and records Prometheus aggregates. Keep the first implementation adapter-friendly so the judge backend can evolve later.

**Step 4: Run test to verify it passes**

Run: `pytest tests/observability/test_live_judge.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add observability/live_judge.py tests/observability/test_live_judge.py
git commit -m "feat: add async live judge service"
```

### Task 8: Wire working memory into the request path

**Files:**
- Modify: `api/routes/chat.py`
- Modify: `core/workflow_service.py`
- Test: `tests/api/test_chat_working_memory_integration.py`

**Step 1: Write the failing test**

Add request-path tests for:

- Redis-backed recent turns are used when present
- Redis-backed core state enriches the workflow input
- DB fallback still works when Redis is unavailable

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_chat_working_memory_integration.py -v`

Expected: FAIL because the route still loads only DB-backed memory.

**Step 3: Write minimal implementation**

Update `chat.py` and `workflow_service.py` to accept the working-memory bundle:

- recent turns
- summary
- facts
- core-state overlays where relevant

Keep the current DB path as fallback.

**Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_chat_working_memory_integration.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add api/routes/chat.py core/workflow_service.py tests/api/test_chat_working_memory_integration.py
git commit -m "feat: wire redis working memory into chat path"
```

### Task 9: Add post-response background execution

**Files:**
- Modify: `api/routes/chat.py`
- Test: `tests/api/test_chat_async_post_response.py`

**Step 1: Write the failing test**

Add tests that verify:

- response returns without waiting for judge completion
- background task is scheduled
- background task handles judge failure safely

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_chat_async_post_response.py -v`

Expected: FAIL because no post-response async hook exists.

**Step 3: Write minimal implementation**

Use FastAPI/Starlette background-task support or an equivalent safe in-process mechanism to:

- append assistant turn if needed
- schedule live judging
- schedule core-state promotion

Do not let async failures affect the response.

**Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_chat_async_post_response.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add api/routes/chat.py tests/api/test_chat_async_post_response.py
git commit -m "feat: add post-response live judge scheduling"
```

### Task 10: Add Grafana live panels for aggregate quality metrics

**Files:**
- Modify: `grafana/dashboards/medigenius-dashboard.json`
- Test: `tests/observability/test_grafana_dashboard_contract.py`

**Step 1: Write the failing test**

Add assertions that the dashboard includes panels for:

- live answer relevance
- live groundedness
- live context precision proxy
- live context coverage proxy
- live judged request rate
- live judge error count

**Step 2: Run test to verify it fails**

Run: `pytest tests/observability/test_grafana_dashboard_contract.py -v`

Expected: FAIL because the panels are not present yet.

**Step 3: Write minimal implementation**

Add Prometheus-backed panels using rolling windows. Keep labels and queries low-cardinality.

**Step 4: Run test to verify it passes**

Run: `pytest tests/observability/test_grafana_dashboard_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add grafana/dashboards/medigenius-dashboard.json tests/observability/test_grafana_dashboard_contract.py
git commit -m "feat: add live judge grafana panels"
```

### Task 11: Add docs for the new memory and observability flow

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`
- Add: `docs/decisions/ADR-0004-redis-working-memory-live-judge.md`

**Step 1: Write the failing doc checklist**

Create a short checklist covering:

- working-memory architecture described
- Redis overlay vs DB fallback described
- async judge path described
- live metrics named consistently

**Step 2: Update docs**

Document:

- the new working-memory flow
- why Redis is overlay-first
- why live metrics are proxies, not benchmark truth

**Step 3: Review docs for consistency**

Confirm terminology matches code:

- `recent_turns`
- `core_state`
- `live_context_precision_proxy`
- `live_context_coverage_proxy`

**Step 4: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md docs/decisions/ADR-0004-redis-working-memory-live-judge.md
git commit -m "docs: record redis working memory architecture"
```

### Task 12: Run the full verification suite

**Files:**
- No new files

**Step 1: Run targeted tests**

Run:

```bash
pytest \
  tests/core/test_settings_contract.py \
  tests/tools/test_working_memory_contract.py \
  tests/tools/test_redis_client_contract.py \
  tests/tools/test_working_memory_service.py \
  tests/tools/test_working_memory_promotion.py \
  tests/observability/test_live_judge_metrics_contract.py \
  tests/observability/test_live_judge.py \
  tests/api/test_chat_working_memory_integration.py \
  tests/api/test_chat_async_post_response.py \
  tests/observability/test_grafana_dashboard_contract.py -v
```

Expected: PASS

**Step 2: Run syntax verification**

Run:

```bash
python3 -m py_compile core/settings.py tools/redis_client.py tools/working_memory.py tools/working_memory_service.py observability/metrics.py observability/live_judge.py api/routes/chat.py core/workflow_service.py
```

Expected: no output

**Step 3: Smoke-test the app path**

Run:

```bash
docker-compose up -d app redis grafana prometheus
```

Then:

```bash
curl -s http://localhost:8000/health/ready
curl -s -X POST http://localhost:8000/api/chat -H 'Content-Type: application/json' -d '{"message":"I am allergic to penicillin. What antibiotics are used for pneumonia?"}'
```

Expected:

- app responds successfully
- request path does not block on judge completion
- Grafana can scrape live judge metrics after a short delay

**Step 4: Commit final verification**

```bash
git add -A
git commit -m "test: verify redis working memory and live judge flow"
```
