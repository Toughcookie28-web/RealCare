# MediGenius Fix Plan (Tuned)

## Scope
Fixes only. No new product features. This plan is tuned to the current repository state and focuses on correctness, security, operability, and hygiene.

## Working Rules
- Each fix is atomic and independently deployable.
- Fixes must include a verification check.
- Use HITL approval gates at the checkpoints listed below.

## HITL Approval Gates
- Gate A: After Tier 1 security/correctness fixes.
- Gate B: After Tier 2 production-hardening fixes.
- Gate C: Before repo-cleanup/deletion actions in Tier 3.

---

## Tier 1 — Critical (correctness/security blockers)

### Fix 1.1: SSE frames are emitted with escaped newlines
**File:** `api/routes/chat.py`

**Problem:** Streaming emits `\n` text instead of real newline delimiters, which breaks SSE frame parsing.

**Fix:**
```python
yield f"event: {event['event']}\n"
yield f"data: {json.dumps(event['data'])}\n\n"
```

**Verify:** Browser/EventSource receives incremental `status` and `final` events without buffering until close.

---

### Fix 1.2: Streaming path drops memory persistence parity
**Files:** `api/routes/chat.py`, `core/workflow_service.py`

**Problem:** `POST /api/chat` persists summary/facts; `POST /api/chat/stream` currently persists only assistant message.

**Fix:**
1. Include `summary` and `facts` in final stream payload from `core/workflow_service.py`.
2. Persist them in `event_stream()` just like non-streaming flow.

**Verify:** After a streamed response, DB/in-memory repo contains updated summary and facts.

---

### Fix 1.3: Static hardcoded session secret
**Files:** `app.py`, `core/settings.py`, `.env.example`

**Problem:** `SessionMiddleware(secret_key='medigenius-session-secret')` is committed and forgeable.

**Fix:**
1. Add `session_secret` setting sourced from `SESSION_SECRET`.
2. Use it in `SessionMiddleware`.
3. Add `SESSION_SECRET=` to `.env.example`.

**Verify:** App fails safe or warns when weak/missing secret in non-dev.

---

### Fix 1.4: CORS wildcard + credentials is invalid and unsafe
**Files:** `app.py`, `core/settings.py`, `.env.example`

**Problem:** `allow_origins=['*']` with `allow_credentials=True` is rejected by browsers and unsafe.

**Fix:**
1. Add env-driven `ALLOWED_ORIGINS`.
2. Parse into `allowed_origins_list`.
3. Wire `CORSMiddleware` to that list.

**Verify:** Credentialed browser calls succeed from configured origins only.

---

### Fix 1.5: Template static links use Flask kwarg
**File:** `templates/index.html`

**Problem:** `url_for('static', filename=...)` is Flask style; Starlette static mount expects `path=`.

**Fix:**
```html
{{ url_for('static', path='css/style.css') }}
{{ url_for('static', path='js/main.js') }}
```

**Verify:** CSS/JS load with 200 status on FastAPI runtime.

---

### Fix 1.6: Readiness endpoint is unconditional
**File:** `api/routes/health.py`

**Problem:** `/health/ready` always returns ready even if DB is unavailable.

**Fix:** Add DB connectivity check (`SELECT 1`) and return `503` on failure.

**Verify:** Ready endpoint returns non-200 when DB is down.

---

### Fix 1.7: `.gitignore` is empty
**File:** `.gitignore`

**Problem:** Repo can accidentally track secrets, caches, local DB artifacts, and binaries.

**Fix:** Add standard ignore rules for `.env`, Python caches, test caches, IDE dirs, local DB/data artifacts.

**Verify:** `git status` excludes expected local/generated files.

---

### Fix 1.8: Missing `.dockerignore`
**File:** `.dockerignore` (new)

**Problem:** Docker build context currently sends unnecessary repo artifacts.

**Fix:** Ignore `.git`, local DB dirs, caches, planning docs, large media/model artifacts.

**Verify:** `docker build` context size drops materially.

---

## Tier 2 — High (production hardening)

### Fix 2.1: Prometheus path-cardinality risk
**File:** `observability/middleware.py`

**Problem:** Raw URL paths as labels can explode series count for ID-based routes.

**Fix:** Normalize dynamic segments to templates before metric labels.

**Verify:** `/metrics` path label count remains bounded under session-heavy load.

---

### Fix 2.2: Container hardening and runtime profile
**File:** `Dockerfile`

**Problem:** Runs as root, no HEALTHCHECK, single uvicorn process.

**Fix:**
1. Use non-root user.
2. Add `HEALTHCHECK` against `/health/live`.
3. Use Gunicorn+Uvicorn workers for production profile.

**Verify:** Container health transitions correctly and non-root UID is active.

---

### Fix 2.3: Streaming response headers missing anti-buffering/no-cache
**File:** `api/routes/chat.py`

**Problem:** Proxies can buffer SSE and break real-time UX.

**Fix:** Set headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`.

**Verify:** Stream emits progressively behind local reverse proxy.

---

### Fix 2.4: docker-compose reproducibility/security gaps
**File:** `docker-compose.yml`

**Problem:** Uses `latest` tags and default Grafana admin credentials.

**Fix:**
1. Pin image versions.
2. Move Grafana credentials to env.
3. Add DB healthcheck and `depends_on: condition: service_healthy`.

**Verify:** `docker compose up` waits for healthy DB before app starts.

---

### Fix 2.5: Duplicate entrypoints
**Files:** `app.py`, `main.py`

**Problem:** `main.py` duplicates startup path and causes maintenance drift.

**Fix:** Remove `main.py`; keep `app.py` as canonical entrypoint.

**Verify:** All run/deploy paths still boot correctly.

---

### Fix 2.6: CI coverage is too narrow
**File:** `.github/workflows/eval.yml`

**Problem:** Only tests/eval run. No lint, no security audit, no image build validation.

**Fix:** Add jobs for `ruff`, `pip-audit`, and Docker build.

**Verify:** CI fails on lint/security/container issues.

---

### Fix 2.7: HITL state is volatile in-memory
**Files:** `core/hitl.py`, `db/models.py`, `alembic/versions/*`, `api/routes/hitl.py`

**Problem:** `_PENDING` dict is lost on restart; no durable audit trail.

**Fix:** Persist HITL reviews in DB with status transitions and timestamps.

**Verify:** Pending/decision state survives restarts.

---

### Fix 2.8: Tracing exporter is console-only
**Files:** `observability/tracing.py`, `pyproject.toml`

**Problem:** Console exporter is not queryable and pollutes logs.

**Fix:** Prefer OTLP exporter when configured; fallback to console only in dev.

**Verify:** Traces appear in collector backend when OTLP env is set.

---

### Fix 2.9: Silent fallback from Postgres to in-memory SQLite behavior
**File:** `db/session.py`

**Problem:** Engine init falls back to in-memory SQLite on DB failure, which can mask production misconfiguration and data loss.

**Fix:**
1. Add strict mode setting (default strict outside dev).
2. In strict mode, fail fast on DB init failure.
3. Keep fallback only for explicit local-dev mode.

**Verify:** Production config fails startup when DB is unavailable.

---

## Tier 3 — Medium (hygiene and correctness quality)

### Fix 3.1: `alembic.ini` contains credential-like URL
**File:** `alembic.ini`

**Fix:** Replace with placeholder URL; keep actual value sourced via env in `alembic/env.py`.

---

### Fix 3.2: Naive UTC datetimes in models/runtime
**Files:** `db/models.py`, `db/repositories.py`, `core/hitl.py`

**Fix:**
1. Use DB-side `func.now()` in ORM defaults.
2. Use timezone-aware `datetime.now(timezone.utc)` in Python code.

---

### Fix 3.3: Render deployment spec under-defined
**File:** `render.yaml`

**Fix:** Add health check path, env var declarations, secure secret generation, and production start command profile.

---

### Fix 3.4: Pre-migration artifacts remain in repo tree
**Files:** `medical_db/`, `chat_db/`, `biogpt-merged/`

**Fix:** Ensure ignored; remove from tracking if tracked.

**HITL note:** Get approval before any tracked data/artifact deletion.

---

### Fix 3.5: Invalid/misleading default model identifiers
**Files:** `core/settings.py`, `.env.example`

**Fix:** Set provider-valid defaults and document accepted model names for each provider.

---

### Fix 3.6: Startup lifecycle uses deprecated hook
**File:** `app.py`

**Fix:** Move startup logic to FastAPI lifespan context manager.

---

### Fix 3.7: `__pycache__` hygiene
**Files:** repo-wide

**Fix:** Ignore and untrack bytecode artifacts.

---

## Tier 4 — Low (polish/future-proofing)

### Fix 4.1: Consolidate packaging metadata
**Files:** `requirements.txt` (remove), `pyproject.toml`, `Dockerfile`, `docker-compose.yml`

**Fix:** Use `pyproject.toml` as the single dependency source of truth, remove hand-maintained `requirements.txt`, and split eval-only dependencies into a separate install path.

---

### Fix 4.2: README drift
**File:** `README.md`

**Fix:** Align stack, architecture diagram, endpoints, and runbook with current implementation.

---

### Fix 4.3: Grafana dashboard depth
**File:** `grafana/dashboards/medigenius-dashboard.json`

**Fix:** Add 5xx rate, cache ratio, node error counts, P95/P99 latency, and DB-health panels.

---

### Fix 4.4: Eval dataset size and coverage
**File:** `tests/eval/testset_medical.jsonl`

**Fix:** Expand to 15-20+ cases spanning RAG, follow-ups/coref, chitchat, guardrail blocks, and ambiguity.

---

## Execution Order (Safe Sequence)
1. Fix 1.7
2. Fix 1.8
3. Fix 1.3
4. Fix 1.4
5. Fix 1.5
6. Fix 1.1
7. Fix 1.2
8. Fix 1.6
9. Fix 2.3
10. Fix 2.1
11. Fix 2.9
12. Fix 2.2
13. Fix 2.4
14. Fix 2.5
15. Fix 3.5
16. Fix 3.6
17. Fix 3.2
18. Fix 3.1
19. Fix 2.7
20. Fix 2.8
21. Fix 2.6
22. Fix 3.3
23. Fix 3.7
24. Fix 3.4
25. Fix 4.2
26. Fix 4.3
27. Fix 4.4
28. Fix 4.1

## Dependencies to add/update
- `gunicorn` (Tier 2.2)
- `opentelemetry-exporter-otlp-proto-grpc` (Tier 2.8)
- `ruff` (Tier 2.6)
- `pip-audit` (Tier 2.6)
