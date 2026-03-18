MediGenius Deployment and Operations Review
===========================================

Executive summary
-----------------

This review is based on the contents of the provided repository archive (MediGenius.zip) and focuses on **deployment and operations** across **every non-agentic file/script**. Per scope, detailed logic in `agents/`, `tools/`, and most of `db/` is treated as excluded (but **deployment hooks**—migrations, init scripts, environment configuration, startup behaviour—are included where they affect ops).

The application is a **single-service FastAPI monolith** (API + server-rendered frontend templates + static assets) that can optionally run with a local multi-service stack (Postgres/pgvector + Prometheus + Grafana) via Compose. FastAPI is a Python web framework for building APIs, designed for high performance and built around standard type hints. 

The most important repo-specific operational findings are:

**Critical risks to production correctness**

* **Template static linking appears Flask-style (likely broken under Starlette/FastAPI)**: `templates/index.html` uses `url_for('static', filename='...')`, but Starlette/FastAPI’s Jinja `url_for` examples use `path=` for mounted static routes. This can break HTML rendering or static asset routing in production, resulting in a “blank/unstyled UI” or 500 errors. 
* **CORS configuration is unsafe/invalid for credentialed traffic**: `allow_origins=['*']` combined with `allow_credentials=True` is not a safe production posture and is commonly blocked by browsers for credentialed requests. You should restrict origins explicitly and align credentials behaviour with the frontend deployment model. 
* **Hard-coded session middleware secret**: `SessionMiddleware(... secret_key='medigenius-session-secret')` is a production-grade secret and must be environment-driven (and rotated).
* **Readiness endpoint is a “static OK”**: `/health/ready` always returns ready. This can route traffic to an instance that has no working DB/vector extension or has failed startup initialisation.
* **Metrics cardinality risk**: request metrics label the **raw path**; endpoints with path params like `/api/session/{session_id}` can create unbounded label cardinality (Prometheus stability risk).
* **Docker image will be unnecessarily huge and brittle**: no `.dockerignore`, no non-root user, no build pinning and the repository includes large artefacts (model weights, PDFs, local DB files) that will get baked into images unless excluded.

**High-priority engineering gaps**

* CI currently runs tests/eval, but lacks: linting/formatting gates, dependency security audits, image build validation, release tagging automation.
* Deploy manifests (`render.yaml`) are minimal and require platform configuration (DB provisioning, secrets, health checks, scaling, timeouts). Render Blueprints are defined by a repo-root YAML spec file (by default `render.yaml`). 

**Prioritised action list**

* **Critical**: fix template static URL generation, remove hardcoded secrets, correct CORS, implement real readiness checks, add `.gitignore` and `.dockerignore`, stop packaging large artefacts into images.
* **High**: switch to a production process model (e.g., Gunicorn-managed multi-worker), add CI lint + security scanning, define migrations policy (Alembic vs `create_all`) and enforce it.
* **Medium**: improve observability export (OTLP exporter instead of console-only tracing), refine dashboards/alerts, add proxy-safe SSE headers and proxy configs.
* **Low**: packaging hygiene (move to `pyproject.toml`), documentation/runbooks, release automation.

Architecture and runtime interactions
-------------------------------------

### Runtime model: what runs, and how requests flow

At runtime, the system is essentially:

* A single **ASGI web server** (`uvicorn`) serving a `FastAPI` application instance.
* The app mounts:
  * API routes under `/api/*`
  * Health endpoints under `/health/*` and `/metrics`
  * A server-rendered UI at `/` (Jinja template)
  * Static assets under `/static/*`
* Optional local services: Postgres/pgvector, Prometheus, Grafana via Compose.

Uvicorn itself recommends a development vs production split (reload in dev; Gunicorn-managed workers for prod) and the deployment guidance explicitly calls out a production process manager. 

mermaid

Copy
    graph LR
      U[Browser] -->|GET /| A[FastAPI app]
      U -->|POST /api/chat| A
      U -->|POST /api/chat/stream SSE| A

      subgraph AppContainer["App runtime"]
        A --> M[Observability middleware]
        A --> R[API routers]
        R --> W[Workflow service<br/>(agentic - excluded logic)]
        R --> S[Session cookie logic]
      end

      A -->|DB calls| P[(Postgres + pgvector)]
      A -->|/metrics scrape| PR[Prometheus]
      PR --> G[Grafana]

      style W fill:#eee,stroke:#999,stroke-dasharray: 4 4

### Key runtime interaction points in this repo

**Entrypoint(s)**

* `app.py` defines `create_app()` and exports `app = create_app()`; also includes a `__main__` block running uvicorn.
* `main.py` also runs uvicorn pointing at `app:app`.

**Request lifecycle**

* `ObservabilityMiddleware` assigns/propagates a `trace_id` and records request latency and counters.
* Chat requests (`/api/chat` and `/api/chat/stream`) create/restore a “session_id” and perform the workflow run/stream, then set a cookie.

**SSE streaming**

* `/api/chat/stream` returns `text/event-stream` frames and the frontend reads the stream using `fetch()` + `ReadableStream`.
* Behind reverse proxies, streaming often requires disabling buffering. NGINX documents disabling proxy buffering and also supports toggling buffering via the `X-Accel-Buffering` response header. 
* Server-sent events require the MIME type `text/event-stream`, and frames are separated by double newlines. 

File-by-file inventory and purpose
----------------------------------

The repository contains ~111 files excluding `.git/`. The table below inventories every file and marks whether it is “ops-relevant”, “runtime-relevant”, or “excluded logic” (agents/tools/db).

### Application and runtime entrypoints

| Path      | Purpose                                                                                                | Ops notes                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `app.py`  | Main FastAPI app factory, middleware wiring, route inclusion, startup actions, mounts templates/static | Primary runtime entrypoint; contains hard-coded session secret, permissive CORS, startup DB init and optional ingest |
| `main.py` | Minimal uvicorn runner for `app:app`                                                                   | Alternative entrypoint; keep one canonical path for deploy                                                           |

### Backend API layer

| Path                         | Purpose                                                             | Ops notes                                                              |
| ---------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `api/deps.py`                | Dependency providers (DB session, repositories), fallback behaviour | Fallback to in-memory repos can mask DB failures in production         |
| `api/routes/chat.py`         | `/api/chat` and `/api/chat/stream` endpoints                        | SSE streaming; cookie handling; persistence depends on DB availability |
| `api/routes/health.py`       | Health endpoints and `/metrics` endpoint                            | `/health/ready` is “always ready”; `/metrics` emits Prometheus format  |
| `api/routes/history.py`      | Session and history endpoints for UI                                | Paths include session IDs → metrics/log path labelling risks           |
| `api/routes/hitl.py`         | Human-in-the-loop endpoints                                         | Needs auth/zoning if exposed publicly                                  |
| `api/routes/__init__` (none) | N/A                                                                 | N/A                                                                    |

### Frontend delivery

| Path                   | Purpose                                 | Ops notes                                                                                                               |
| ---------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `templates/index.html` | Server-rendered UI                      | Uses `url_for('static', filename=...)` which is Flask style; Starlette/FastAPI examples use `path=` for mounted static. |
| `static/js/main.js`    | Browser UI logic incl. streaming client | Implements streaming parsing; no proxy keepalive headers handled client-side                                            |
| `static/css/style.css` | UI styling                              | Large, but static                                                                                                       |

### Configuration and settings

| Path               | Purpose                                     | Ops notes                                                                               |
| ------------------ | ------------------------------------------- | --------------------------------------------------------------------------------------- |
| `core/settings.py` | Pydantic settings, env var schema, defaults | Central env var inventory; defaults include DB URL with password; no session secret env |
| `.env.example`     | Example environment configuration           | Basis for dev/staging/prod config                                                       |

### Deployment, containerisation, platform manifests

| Path                 | Purpose                                      | Ops notes                                                                                                    |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `Dockerfile`         | Builds Python image and runs uvicorn         | No non-root user; no `.dockerignore`; no healthcheck; installs unpinned deps; will copy large artefacts      |
| `docker-compose.yml` | Local stack: db + app + prometheus + grafana | Uses default passwords; `depends_on` does not ensure readiness. Docker documents dependency order behaviour. |
| `render.yaml`        | Render deploy manifest for web service       | Minimal; Render blueprint spec defines supported fields.                                                     |

### CI/CD

| Path                         | Purpose                                  | Ops notes                                                                |
| ---------------------------- | ---------------------------------------- | ------------------------------------------------------------------------ |
| `.github/workflows/eval.yml` | Runs pytest unit tests + eval regression | No lint/format/security scanning; no build artefacts; no container build |

### Observability

| Path                                           | Purpose                           | Ops notes                                                        |
| ---------------------------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| `observability/logging.py`                     | JSON structured logging to stdout | Good baseline; review for PII logging from excluded agentic logs |
| `observability/metrics.py`                     | Prometheus counters/histograms    | Ensure label cardinality stays bounded                           |
| `observability/middleware.py`                  | Trace IDs + request metrics       | Uses raw `request.url.path` as label → cardinality risk          |
| `observability/tracing.py`                     | Optional OpenTelemetry tracing    | Console exporter only; production should use OTLP exporters.     |
| `prometheus/prometheus.yml`                    | Prometheus scrape config          | Scrape config concept and `metrics_path` are documented.         |
| `grafana/dashboards/medigenius-dashboard.json` | Grafana dashboard definition      | Default admin credentials in compose should be changed           |

Prometheus’ exposition format `0.0.4` is a long-standing text format and is referenced in the OpenMetrics specification. 

### Migrations and DB hooks (logic excluded; ops hooks included)

| Path                                      | Purpose                        | Ops notes                                                               |
| ----------------------------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| `alembic.ini`                             | Alembic configuration          | Contains default URL; should not carry secrets in-repo                  |
| `alembic/env.py`                          | Alembic env wiring to settings | Uses settings’ `DATABASE_URL` dynamically                               |
| `alembic/versions/0001_initial_schema.py` | Initial schema migration       | Creates pgvector extension and required tables; date-stamped 2026-02-16 |
| `scripts/init_postgres.py`                | Script to run DB init          | Useful as a deploy hook/job                                             |
| `scripts/reindex_pdf.py`                  | Rebuild vector index from PDF  | Good as a one-off admin job; don’t run in web startup                   |

Backup/restore for Postgres should be planned explicitly; `pg_dump` + `pg_restore` are documented as the standard logical backup/restore mechanism. 

### Tests and evaluation harness

| Path                                          | Purpose                                              | Ops notes                                                                                        |
| --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `tests/test_app.py`                           | Basic endpoint tests                                 | Not hermetic if workflow requires external LLM keys (depends on excluded agentic implementation) |
| `tests/eval/run_eval.py`                      | Runs an eval pass and writes `last_eval_result.json` | Useful for regression; ensure CI can run without live API calls                                  |
| `tests/eval/test_eval_regression.py`          | Regression test thresholding eval results            | Must not become flaky                                                                            |
| `tests/eval/testset_medical.jsonl`            | Eval dataset                                         | Data governance required (licensing/PII)                                                         |
| `tests/__init__.py`, `tests/eval/__init__.py` | Package markers                                      | OK                                                                                               |
| `tests/**/__pycache__/*.pyc`                  | Build artefacts                                      | Should not be committed                                                                          |

### Packaging, metadata, docs, and artefacts

| Path                                          | Purpose                                       | Ops notes                                                                                            |
| --------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `requirements.txt`                            | Python dependencies                           | Unpinned; includes heavyweight ML deps (`torch`, `transformers`) impacting image size and cold start |
| `setup.py`                                    | Packaging metadata                            | Duplicates dependency list; may inadvertently include large files depending on packaging             |
| `LICENSE`                                     | Licence (MIT)                                 | OK                                                                                                   |
| `.gitattributes`                              | Git LFS hints for `.safetensors`              | Good intent, but archive contains large model artefacts anyway                                       |
| `.gitignore`                                  | Ignore rules                                  | Empty: should ignore `.env`, caches, artefacts, DB files                                             |
| `README.md`, `report.md`                      | Docs                                          | README contains API/structure claims; ensure it matches actual endpoints                             |
| `data/medical_book.pdf`                       | Source PDF                                    | Should not be copied into production images unless intended                                          |
| `biogpt-merged/*`                             | Model artefacts (`model.safetensors`, config) | Should not be shipped unless serving this model on-device                                            |
| `medical_db/*`, `chat_db/medigenius_chats.db` | Local DB artefacts                            | Never bake into prod images; treat as dev data only                                                  |
| `demo.mp4`, `app.png`                         | Media                                         | Exclude from builds                                                                                  |

### Excluded logic directories (still inventory-complete)

| Path       | Purpose                                       | Ops notes                                                                  |
| ---------- | --------------------------------------------- | -------------------------------------------------------------------------- |
| `agents/*` | Agent implementations                         | Excluded logic; review only for logging/PII risks                          |
| `tools/*`  | Tooling wrappers (LLM, embeddings, retrieval) | Excluded logic; impacts secrets and outbound traffic controls              |
| `db/*`     | Models/repos/session                          | Excluded logic, except for deploy hooks: connection pooling, init strategy |

Runtime configuration and environment variables
-----------------------------------------------

### How configuration is loaded

The settings layer is implemented using Pydantic Settings, which loads environment variables and can also read a `.env` file. This is a standard pattern for controlling behaviour across dev/staging/prod without code changes. The repository uses `core/settings.py` + `.env.example` for this.

### Environment variable inventory

The following variables are defined centrally and should be treated as the canonical configuration surface:

| Env var                    | Default                                                            | Controls                           | Security / ops notes                                                  |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------- | --------------------------------------------------------------------- |
| `ENV`                      | `development`                                                      | Environment name                   | Use to toggle “fail-fast vs fallback” policies                        |
| `DEBUG`                    | `false`                                                            | Debug mode                         | Must be `false` in production                                         |
| `API_HOST`                 | `0.0.0.0`                                                          | Bind host                          | Container OK; in local dev you might prefer `127.0.0.1`               |
| `API_PORT`                 | `8000`                                                             | Bind port                          | In managed platforms, prefer `$PORT` passed to uvicorn                |
| `DATABASE_URL`             | `postgresql+psycopg://postgres:postgres@localhost:5432/medigenius` | DB connection                      | Default contains credentials; never use in prod; use platform secrets |
| `PGVECTOR_ENABLED`         | `true`                                                             | Enable pgvector extension attempts | In managed Postgres, ensure extension is supported/enabled            |
| `GROQ_API_KEY`             | empty                                                              | LLM provider auth                  | Secret                                                                |
| `OPENAI_API_KEY`           | empty                                                              | LLM provider auth                  | Secret                                                                |
| `TAVILY_API_KEY`           | empty                                                              | Web search auth                    | Secret                                                                |
| `LLM_PRIMARY_MODEL`        | `openai/gpt-oss-120b`                                              | Primary model name                 | Keep consistent across environments for reproducibility               |
| `LLM_FALLBACK_MODEL`       | `gpt-4o-mini`                                                      | Fallback model                     | Ensure fallback is available if primary fails                         |
| `LLM_TEMPERATURE`          | `0.2`                                                              | Sampling                           | Stable tests prefer fixed low values                                  |
| `LLM_MAX_TOKENS`           | `1200`                                                             | Output cap                         | Impacts latency and cost                                              |
| `EMBEDDING_MODEL`          | `sentence-transformers/all-MiniLM-L6-v2`                           | Embedding model                    | Heavy runtime dependency; affects memory and cold start               |
| `EMBEDDING_DIM`            | `384`                                                              | Vector dimension                   | Must match schema (`vector(384)`)                                     |
| `TRUSTED_MEDICAL_DOMAINS`  | `nih.gov,mayoclinic.org,who.int,cdc.gov,pubmed.ncbi.nlm.nih.gov`   | Retrieval policy                   | Consider governance and audits                                        |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95`                                                             | Cache similarity threshold         | Impacts correctness vs speed                                          |
| `SEMANTIC_CACHE_SIZE`      | `500`                                                              | Cache capacity                     | Impacts memory usage                                                  |
| `AUTO_INGEST_ON_STARTUP`   | `false`                                                            | Startup ingestion                  | Should stay `false` in prod; ingestion should run as a job            |
| `ENABLE_HITL_INTERRUPTS`   | `true`                                                             | HITL behaviour                     | If enabled, secure the HITL endpoints                                 |
| `OTEL_ENABLED`             | `true`                                                             | Enable tracing setup               | Current tracing is console-only; production needs OTLP.               |

### Environment comparison: dev vs staging vs production

| Area            | Dev (local)                            | Staging (recommended)         | Prod (recommended)                                                |
| --------------- | -------------------------------------- | ----------------------------- | ----------------------------------------------------------------- |
| Debug           | `DEBUG=true` (Compose)                 | `DEBUG=false`                 | `DEBUG=false`                                                     |
| CORS            | Wide open often OK for localhost       | Restrict to staging origin(s) | Restrict to exact prod frontend origin(s)                         |
| Session cookies | `Secure` often off on localhost        | `Secure=true` if HTTPS        | `Secure=true`, `HttpOnly=true`, `SameSite=Lax/Strict` as required |
| DB              | Compose Postgres + pgvector            | Managed Postgres              | Managed Postgres + backups                                        |
| Migrations      | Run `alembic upgrade head` manually    | Same                          | Run migrations in deploy pipeline before traffic                  |
| Ingestion       | Manual script `scripts/reindex_pdf.py` | Job-run                       | Job-run; do not block app startup                                 |
| Observability   | Scrape `/metrics` locally              | Same + alerting thresholds    | Same + alerting + trace export                                    |
| Tests/eval      | Run without external keys if possible  | Run with restricted keys      | Consider smoke tests only, plus offline eval                      |

### Secrets management and cookie/session security

The code uses:

* A custom `session_id` cookie set by the API.
* Starlette’s `SessionMiddleware` is enabled, which uses signed cookie-based sessions; Starlette documents that session data is readable but not modifiable, and that it’s implemented as signed cookies. 

Repo-specific checks to make before production:

* Replace the hard-coded middleware secret with `SESSION_SECRET` (or equivalent) and rotate it.
* Set `Secure=True` on cookies in production (HTTPS-only).
* Confirm whether you actually need `SessionMiddleware`; if not used (`request.session`), remove it to reduce risk surface.

Containers, CI/CD, and deployment manifests
-------------------------------------------

### Dockerfile and image build quality

The current Dockerfile is a minimal Python build that installs dependencies and runs uvicorn. The Dockerfile reference documents the semantics of Dockerfile instructions like `COPY`, `CMD`, etc. 

Repo-specific concerns and remediations:

**Image size and leakage risk**

* Because the build does `COPY . /app` and the repo contains large artefacts (`biogpt-merged/model.safetensors`, PDFs, demo video, local DB files), the image will become very large and may include sensitive/dev-only data.
* Remediation: add `.dockerignore` (critical); split training artefacts into a separate branch or storage; mount PDFs at runtime rather than bake them.

**Run as non-root**

* The container currently runs as root. Use a non-root user in the Docker image for defence-in-depth. Docker documents running in rootless mode as a security mitigation (broader context: reducing root privileges). 
* Remediation: create a user/group in Dockerfile, `USER appuser`.

**Process model**

* For production deployments, run with Gunicorn-managed workers rather than a single uvicorn process. Uvicorn’s deployment guidance explicitly recommends Gunicorn with a Uvicorn worker for production process management. 

**Healthcheck**

* Add a container `HEALTHCHECK` that calls `/health/ready` once it becomes meaningful (after you implement real dependency checks).

### docker-compose stack behaviour

Compose is used to run:

* `db`: `pgvector/pgvector:pg16`
* `app`: build from repo
* `prometheus`: scrapes `/metrics`
* `grafana`: dashboards

Operational note: Compose’s `depends_on` controls start order but does not guarantee that a dependency is “ready to accept connections”. Docker documents dependency ordering and startup/shutdown control.   
Remediation:

* Add explicit healthchecks for DB and make app wait/retry (or implement app readiness as “DB reachable”).
* Pin image versions instead of `latest`; Docker notes “latest” is the default tag when omitted and tags influence reproducibility. 

Grafana security:

* Compose sets `GF_SECURITY_ADMIN_USER=admin` and `GF_SECURITY_ADMIN_PASSWORD=admin`. This is not safe beyond localhost; change these and enforce stronger policies. Grafana documents password policy configuration options. 

### Render deployment manifest

`render.yaml` defines a Python web service with:

* `buildCommand: pip install -r requirements.txt`
* `startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT`

Render’s Blueprint YAML reference describes how `render.yaml` defines services and their fields.   
Render’s “Your First Deploy” docs describe build and start command intent and show that production start commands often differ from local ones (e.g., Gunicorn vs dev servers).   
Render web services docs confirm Render supports Python web apps (including FastAPI) and also supports deploying Docker images. 

Repo-specific platform settings you will need to set in Render (because the manifest does not encode them yet):

* Environment variables: `DATABASE_URL`, LLM provider keys, `DEBUG=false`, plus any new `SESSION_SECRET`, `ALLOWED_ORIGINS`.
* A managed Postgres instance: Render provides managed Postgres and documents creation/connection patterns. 
* pgvector support: Render documents supported extensions, and Render also publishes material describing pgvector usage in managed Postgres contexts. 
* Health check path: ensure it hits a true readiness endpoint (after you implement it).

### CI/CD: current state and recommended hardening

Current CI (in `.github/workflows/eval.yml`) runs:

* install deps
* `pytest` unit tests
* eval regression

Key missing components (recommended):

* Lint/format: `ruff`, `black` (or similar)
* Type checking: optional but helpful (`mypy`, `pyright`)
* Security scanning:
  * dependency audit (e.g., pip-audit)
  * SAST (e.g., CodeQL). GitHub documents CodeQL as its code scanning engine for automation of security checks. 
* Build validation:
  * build the Docker image in CI and run smoke tests in-container
* Release tagging:
  * semantic version tags matching `setup.py` version
  * changelog automation (optional)

Show code
Observability, testing hooks, and operational runbook
-----------------------------------------------------

### Observability: logging, metrics, tracing, and PII risk

**Logging**

* JSON logging to stdout is an appropriate baseline for container platforms.
* Ensure that excluded agentic/tool layers do not log user medical content in clear-text, particularly in production.

**Metrics**

* `/metrics` is exposed in Prometheus text format (`text/plain; version=0.0.4`). This aligns with the Prometheus exposition format referenced in OpenMetrics docs. 
* Prometheus configuration uses a static scrape config with `metrics_path: /metrics`; Prometheus documents scrape config structure and job/target concepts. 

Repo-specific metric risks to fix:

* Normalise path labels: avoid storing raw path values containing IDs, to prevent label explosion.

**Tracing**

* Current OpenTelemetry tracing initialises a Console exporter only.
* For production, prefer OTLP exporters (send to a collector or vendor backend). OpenTelemetry documents OTLP exporters and their configuration surface. 

### Streaming/SSE behaviour behind proxies

The backend uses `StreamingResponse` with `text/event-stream`; the frontend reads the stream via `fetch()` and parses `event:` and `data:` frames.

Hardening recommendations:

* Add headers on streaming responses:
  * `Cache-Control: no-cache`
  * `Connection: keep-alive` (proxy-dependent)
  * (If using NGINX) `X-Accel-Buffering: no` to disable proxy buffering. NGINX documents this buffering toggle for proxied responses. 
* If deploying behind NGINX or another proxy, configure timeouts to allow long-lived streams.

### Runnable test hooks and validation commands

The following commands validate both local dev and production-like runs.

**Local dev (no containers)**

bash

Copy
    python -m venv .venv
    . .venv/bin/activate  # Windows: .venv\Scripts\activate

    pip install -r requirements.txt
    cp .env.example .env

    # Start API + UI
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

**Local dev (full stack via Compose)**

bash

Copy
    docker compose up --build
    # or detached
    docker compose up -d --build
    docker compose logs -f app

**Health checks**

bash

Copy
    curl -sS http://localhost:8000/health/live | jq
    curl -sS http://localhost:8000/health/ready | jq
    curl -sS http://localhost:8000/metrics | head

**Basic chat (non-streaming)**

bash

Copy
    curl -sS -X POST http://localhost:8000/api/chat \
      -H "Content-Type: application/json" \
      -d '{"message":"What are symptoms of flu?"}' | jq

**Streaming chat (SSE)**

bash

Copy
    curl -N -X POST http://localhost:8000/api/chat/stream \
      -H "Content-Type: application/json" \
      -H "Accept: text/event-stream" \
      -d '{"message":"Explain hypertension in simple terms.","stream":true}'

**Run migrations (recommended policy: use Alembic)**

bash

Copy
    # Ensure DATABASE_URL is set correctly in environment
    alembic upgrade head

**Index/reindex job**

bash

Copy
    python scripts/reindex_pdf.py

**Run tests**

bash

Copy
    pytest -q
    # CI mirrors:
    pytest -q tests/test_app.py
    pytest -q tests/eval/test_eval_regression.py

### Operational runbook

**Deploy**

1. Build artefact:
   * Prefer building a Docker image (pinned) or use platform build in Render.
2. Apply migrations:
   * Run `alembic upgrade head` as a pre-deploy step or as a release command.
3. Start app with production process model:
   * Use Gunicorn-managed workers (Uvicorn recommends this for production). 
4. Verify:
   * `/health/live`, `/health/ready`, `/metrics`, and a chat smoke test.

**Rollback**

* Keep the previous image tag or previous platform deploy available.
* If schema migrations are not backwards compatible, enforce “expand/contract” DB migration patterns.

**Backups**

* Use logical backups via `pg_dump` and restore via `pg_restore`; Postgres documents this mechanism and archive formats. 
* For managed platforms, also enable platform-native PITR/retention where available (platform-specific).

**Key rotation**

* Rotate `SESSION_SECRET`, DB credentials, and provider API keys.
* Plan rotation to avoid invalidating active sessions unexpectedly.

**Alerts**

* Alert on:
  * elevated 5xx rate (`medigenius_http_requests_total{status="500"}`)
  * latency SLO breaches (histograms)
  * DB connection failures (once readiness checks and metrics exist)
  * sustained error events in SSE paths

Concrete remediation plan with severity ratings
-----------------------------------------------

| Priority | Item                               | Repo-specific remediation                                                                                                                 |
| -------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Critical | Fix template static URL generation | Change `templates/index.html` to use Starlette/FastAPI style: `url_for('static', path='/css/style.css')` (and similarly for JS).          |
| Critical | Remove hardcoded session secret    | Add `SESSION_SECRET` env var; wire into `SessionMiddleware`. Rotate regularly.                                                            |
| Critical | Correct CORS                       | Replace wildcard origins with explicit allowlist; define `ALLOWED_ORIGINS` env; align credentials model to browser rules.                 |
| Critical | Implement meaningful readiness     | Make `/health/ready` validate DB reachable + migrations applied (or at least DB connectivity) and fail when dependencies are unavailable. |
| Critical | Add `.gitignore` + `.dockerignore` | Ignore `.env`, caches, `__pycache__/`, `.pytest_cache/`, local DB files, large artefacts; exclude them from Docker build context.         |
| High     | Production server mode             | Replace `CMD uvicorn ...` with Gunicorn-managed workers for production deployments.                                                       |
| High     | CI hardening                       | Add lint/format gates + dependency audit + CodeQL scanning.                                                                               |
| High     | Migrations policy                  | Decide: Alembic-only vs `create_all` and enforce one. Add a documented deploy migration step.                                             |
| Medium   | Streaming behind proxies           | Add headers (and NGINX buffering disable where relevant) to ensure SSE works reliably.                                                    |
| Medium   | OTLP tracing export                | Replace console-only tracing with OTLP exporter configuration.                                                                            |
| Medium   | Metrics label hygiene              | Normalise path labels to route templates to prevent cardinality blow-ups; add alerts.                                                     |
| Low      | Packaging modernisation            | Move towards `pyproject.toml` and dependency locking.                                                                                     |
| Low      | Align docs with code               | Update README endpoints to match `/api/*` routes; document env var contract and deploy steps.                 |
