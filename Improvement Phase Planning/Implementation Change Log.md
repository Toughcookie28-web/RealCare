# Implementation Change Log

Use this file as the authoritative audit log for all implementation changes.

## Entry Template
- `timestamp`:
- `milestone`:
- `intent`:
- `files_changed`:
- `api_schema_impact`:
- `dependency_changes`:
- `risk_level`:
- `tests_run`:
- `rollback_note`:
- `HITL_required`:
- `HITL_status`:
- `direction_change_note`:

---

## Entry 1
- `timestamp`: 2026-02-16
- `milestone`: Milestone 0
- `intent`: Create governance artifacts for unified upgrade and mandatory change tracking.
- `files_changed`:
  - `Improvement Phase Planning/Medi Genius Unified Upgrade Plan.md`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`: None.
- `rollback_note`: Remove newly created planning files.
- `HITL_required`: yes
- `HITL_status`: approved by user request to implement unified plan.
- `direction_change_note`: N/A.

## Entry 2
- `timestamp`: 2026-02-16
- `milestone`: Milestone 0
- `intent`: Add typed contracts, typed state v2, centralized settings, and environment contract.
- `files_changed`:
  - `core/contracts.py`
  - `core/state_v2.py`
  - `core/state.py`
  - `core/settings.py`
  - `.env.example`
- `api_schema_impact`:
  - Added typed request/response contracts (`ChatRequest`, `ChatResponse`, `NodeResult`, `StreamEvent`).
- `dependency_changes`:
  - Planned `pydantic-settings` for settings management.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
- `rollback_note`: Revert new contract/settings files and restore old state contract.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: None.

## Entry 3
- `timestamp`: 2026-02-16
- `milestone`: Milestone 1
- `intent`: Replace Flask runtime with FastAPI endpoints and SSE chat streaming.
- `files_changed`:
  - `app.py`
  - `main.py`
  - `api/deps.py`
  - `api/routes/chat.py`
  - `api/routes/history.py`
  - `api/routes/health.py`
  - `api/routes/hitl.py`
  - `core/workflow_service.py`
  - `static/js/main.js`
- `api_schema_impact`:
  - Active endpoints now served by FastAPI: `/api/chat`, `/api/chat/stream`, `/api/history`, `/api/sessions`, `/api/session/{session_id}`, `/api/clear`, `/api/new-chat`, `/health/live`, `/health/ready`, `/metrics`, `/api/hitl/*`.
- `dependency_changes`:
  - FastAPI runtime dependencies aligned.
- `risk_level`: High.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
  - `pytest -q` (skipped due missing FastAPI package in local environment)
- `rollback_note`: Restore previous Flask `app.py` and JS request path logic.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: Added HITL admin endpoints for manual approval workflow.

## Entry 4
- `timestamp`: 2026-02-16
- `milestone`: Milestone 2
- `intent`: Add PostgreSQL relational schema/repositories and baseline migration assets.
- `files_changed`:
  - `db/models.py`
  - `db/session.py`
  - `db/repositories.py`
  - `alembic.ini`
  - `alembic/env.py`
  - `alembic/versions/0001_initial_schema.py`
  - `scripts/init_postgres.py`
- `api_schema_impact`:
  - Session/history endpoints now repository-backed for Postgres schema.
- `dependency_changes`:
  - Added SQLAlchemy/Alembic/psycopg/pgvector stack in requirements/setup.
- `risk_level`: High.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
- `rollback_note`: Remove db/alembic modules and restore SQLite persistence path.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: In-memory fallback repositories added for local runtime resiliency if DB unavailable.

## Entry 5
- `timestamp`: 2026-02-16
- `milestone`: Milestone 3
- `intent`: Replace Chroma runtime path with pgvector-style repository retrieval and PDF reindex pipeline.
- `files_changed`:
  - `tools/vector_store.py`
  - `tools/pdf_loader.py`
  - `tools/embedding_client.py`
  - `scripts/reindex_pdf.py`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Added embedding/vector support packages.
- `risk_level`: High.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
- `rollback_note`: Revert vector ingestion/retrieval files to Chroma implementation.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: None.

## Entry 6
- `timestamp`: 2026-02-16
- `milestone`: Milestone 4
- `intent`: Add observability stack (structured logs, middleware metrics, tracing hooks, dashboards).
- `files_changed`:
  - `observability/logging.py`
  - `observability/metrics.py`
  - `observability/tracing.py`
  - `observability/middleware.py`
  - `docker-compose.yml`
  - `prometheus/prometheus.yml`
  - `grafana/dashboards/medigenius-dashboard.json`
- `api_schema_impact`:
  - Added `/metrics` endpoint.
- `dependency_changes`:
  - Added `prometheus-client`, `opentelemetry-api`, `opentelemetry-sdk`.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
- `rollback_note`: Remove observability middleware and infra files.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: None.

## Entry 7
- `timestamp`: 2026-02-16
- `milestone`: Milestones 5-7
- `intent`: Implement upgraded agentic workflow (rewriter, semantic planner, memory summary/facts, cache, reflection, guardrails, fallback, citations, HITL interrupt).
- `files_changed`:
  - `core/langgraph_workflow.py`
  - `agents/common.py`
  - `agents/guardrail_agent.py`
  - `agents/query_rewriter_agent.py`
  - `agents/planner_agent.py`
  - `agents/memory_agent.py`
  - `agents/retriever_agent.py`
  - `agents/literature_agent.py`
  - `agents/tavily_agent.py`
  - `agents/wikipedia_agent.py`
  - `agents/llm_agent.py`
  - `agents/executor_agent.py`
  - `agents/reflection_agent.py`
  - `agents/explanation_agent.py`
  - `tools/cache.py`
  - `tools/llm_client.py`
  - `tools/search_tools.py`
  - `core/hitl.py`
- `api_schema_impact`:
  - Chat responses now include route/citations and high-risk HITL guardrail path.
- `dependency_changes`:
  - Added `langchain-openai` fallback provider support.
- `risk_level`: High.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
- `rollback_note`: Revert workflow/agent/tool changes and restore previous planner/retriever/executor chain.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: Added runtime high-risk pause with manual approval endpoints.

## Entry 8
- `timestamp`: 2026-02-16
- `milestone`: Evaluation/CI enablement
- `intent`: Add early evaluation harness and CI workflow gates.
- `files_changed`:
  - `tests/eval/testset_medical.jsonl`
  - `tests/eval/run_eval.py`
  - `tests/eval/test_eval_regression.py`
  - `tests/eval/__init__.py`
  - `tests/test_app.py`
  - `tests/__init__.py`
  - `.github/workflows/eval.yml`
- `api_schema_impact`: None.
- `dependency_changes`: None beyond existing requirements changes.
- `risk_level`: Medium.
- `tests_run`:
  - `pytest -q` -> `2 skipped` (FastAPI package absent in local runtime)
- `rollback_note`: Remove eval harness and workflow test file additions.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: None.

## Entry 9
- `timestamp`: 2026-02-16
- `milestone`: Packaging/deployment alignment
- `intent`: Align runtime/build assets with FastAPI architecture.
- `files_changed`:
  - `requirements.txt`
  - `setup.py`
  - `Dockerfile`
  - `render.yaml`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Removed Flask/Chroma runtime requirement from active setup, added FastAPI/Postgres/observability stack.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
- `rollback_note`: Restore previous packaging/deploy files.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: None.

## Entry 10
- `timestamp`: 2026-02-16
- `milestone`: Documentation alignment
- `intent`: Update README architecture text to reflect FastAPI + PostgreSQL/pgvector stack.
- `files_changed`:
  - `README.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`: None.
- `rollback_note`: Revert README edits.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 11
- `timestamp`: 2026-02-16
- `milestone`: Runtime resiliency
- `intent`: Add provider/import fallbacks and streaming cookie fix for constrained local environments.
- `files_changed`:
  - `tools/llm_client.py`
  - `tools/embedding_client.py`
  - `api/routes/chat.py`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts tests`
- `rollback_note`: Revert to strict provider imports and previous stream response behavior.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 12
- `timestamp`: 2026-02-20
- `milestone`: Runtime setup hardening
- `intent`: Make Docker runtime boot path deterministic by wiring `.env` into app container and Grafana auto-provisioning (Prometheus datasource + dashboard provider), plus local default Grafana credentials for first run.
- `files_changed`:
  - `docker-compose.yml`
  - `grafana/provisioning/datasources/datasource.yml`
  - `grafana/provisioning/dashboards/dashboard.yml`
  - `.env`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 - <<'PY' ... yaml.safe_load('docker-compose.yml') ... PY` (YAML structure sanity check)
  - `docker compose config` not runnable in this environment (`docker compose` unavailable in current shell).
- `rollback_note`: Revert listed files to previous versions and remove Grafana provisioning mounts/files.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 13
- `timestamp`: 2026-02-21
- `milestone`: Runtime dependency fix
- `intent`: Fix container boot failure caused by missing `itsdangerous` dependency required by Starlette `SessionMiddleware`.
- `files_changed`:
  - `requirements.txt`
  - `pyproject.toml`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Added `itsdangerous` to runtime dependencies.
- `risk_level`: Low.
- `tests_run`:
  - Static dependency verification via file inspection (`requirements.txt`, `pyproject.toml`).
- `rollback_note`: Remove `itsdangerous` from dependency lists and rebuild image.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 14
- `timestamp`: 2026-02-21
- `milestone`: Runtime dependency alignment
- `intent`: Resolve container boot failure from `ModuleNotFoundError: No module named 'langchain'` by aligning declared dependencies with direct imports used in runtime modules.
- `files_changed`:
  - `requirements.txt`
  - `pyproject.toml`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Added `langchain` to runtime dependencies.
  - Normalized `langchain_huggingface` to `langchain-huggingface` in `requirements.txt`.
- `risk_level`: Low.
- `tests_run`:
  - Static import/dependency audit (`python3` AST-based check) confirming no unresolved direct third-party imports versus `requirements.txt`.
- `rollback_note`: Remove added dependency entries and rebuild image to restore prior package set.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 15
- `timestamp`: 2026-02-21
- `milestone`: LangChain 1.x compatibility hardening
- `intent`: Remove legacy `langchain.schema` and `langchain.text_splitter` imports that break on LangChain 1.x, and align runtime dependencies with split LangChain packages.
- `files_changed`:
  - `db/repositories.py`
  - `core/state_v2.py`
  - `tools/pdf_loader.py`
  - `agents/retriever_agent.py`
  - `agents/wikipedia_agent.py`
  - `agents/literature_agent.py`
  - `agents/tavily_agent.py`
  - `requirements.txt`
  - `pyproject.toml`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Removed direct runtime dependency on monolithic `langchain`.
  - Added `langchain-text-splitters`.
  - Kept `langchain-core`, `langchain-community`, and provider-specific packages as active runtime dependencies.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability`
  - `rg -n "from langchain\\.schema import Document|from langchain\\.text_splitter import RecursiveCharacterTextSplitter" --glob '*.py'` (no matches)
- `rollback_note`: Revert listed files and restore prior import/dependency definitions.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Shifted from temporary add-`langchain` workaround to long-term split-package compatibility path.

## Entry 16
- `timestamp`: 2026-02-21
- `milestone`: Local runtime preflight hardening
- `intent`: Reduce repeated rebuild/debug loops by hardening runtime against common local misconfiguration cases and aligning vector schema dimension with configured embedding dimension.
- `files_changed`:
  - `db/models.py`
  - `tools/embedding_client.py`
  - `tools/llm_client.py`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts tests`
  - Static import/dependency audit (`MISSING_IMPORT_PACKAGES=NONE`)
  - Env/settings contract audit (`ENV_MISSING=NONE`; optional provider keys may be empty)
  - Compose/mount sanity audit (`COMPOSE_MOUNT_FILES_MISSING=NONE`)
- `rollback_note`: Revert listed files to previous versions.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Added graceful fallbacks for embedding/LLM initialization and invocation failures to keep local runtime available.

## Entry 17
- `timestamp`: 2026-02-21
- `milestone`: Env-as-source-of-truth alignment
- `intent`: Remove provider hardcoding assumptions so runtime model/provider behavior is controlled by `.env`, and align fallback embedding dimension with configured `EMBEDDING_DIM`.
- `files_changed`:
  - `core/settings.py`
  - `tools/llm_client.py`
  - `tools/embedding_client.py`
  - `.env`
  - `.env.example`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability`
- `rollback_note`: Revert listed files to previous versions.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: LLM provider selection now supports explicit env-driven provider routing (`auto|groq|openai`) plus optional OpenAI-compatible base URL.

## Entry 18
- `timestamp`: 2026-02-21
- `milestone`: Env consistency update
- `intent`: Align updated model/provider config by switching fallback provider to Groq in `.env` and performing readiness audit against settings/runtime assumptions.
- `files_changed`:
  - `.env`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static `.env` readiness audit (provider/model/key coherence, settings alias coverage, compose override awareness).
- `rollback_note`: Revert `.env` fallback provider change.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 19
- `timestamp`: 2026-02-21
- `milestone`: Env-driven provider and embedding migration
- `intent`: Align primary LLM provider with user-updated `.env` (Groq) and add complete Gemini embedding path controlled by env configuration.
- `files_changed`:
  - `.env`
  - `.env.example`
  - `core/settings.py`
  - `tools/embedding_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None (Gemini embedding integrated via existing `requests` dependency).
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts tests`
  - Settings alias coverage audit (`MISSING=NONE`)
  - Env/provider coherence audit (`ISSUES=NONE`)
- `rollback_note`: Revert listed files and switch embedding flow back to previous provider logic.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Embedding provider control expanded from HuggingFace-only default to `auto|gemini|huggingface`, with Gemini-first selection when configured.

## Entry 20
- `timestamp`: 2026-02-21
- `milestone`: Env key robustness
- `intent`: Prevent quoted-secret formatting in `.env` from breaking provider authentication by normalizing API key/base URL values before LLM client initialization.
- `files_changed`:
  - `tools/llm_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability`
- `rollback_note`: Revert `tools/llm_client.py` secret normalization helper and related key usage changes.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 21
- `timestamp`: 2026-02-21
- `milestone`: Tavily integration modernization
- `intent`: Remove deprecation path by preferring `langchain_tavily` and maintain runtime compatibility with legacy fallback plus normalized Tavily response handling.
- `files_changed`:
  - `tools/search_tools.py`
  - `requirements.txt`
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Added `langchain-tavily`.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts tests`
- `rollback_note`: Revert listed files and restore legacy `langchain_community.tools.tavily_search.TavilySearchResults` as primary path.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Shifted Tavily tool wiring to the maintained package while preserving fallback behavior.

## Entry 22
- `timestamp`: 2026-02-21
- `milestone`: Compose/env alignment
- `intent`: Remove hardcoded database credentials from `docker-compose.yml` and align container runtime DB username/password/database with `.env` values.
- `files_changed`:
  - `docker-compose.yml`
  - `.env`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static config inspection of compose/env parity.
- `rollback_note`: Revert listed files to prior hardcoded compose DB settings.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 23
- `timestamp`: 2026-02-21
- `milestone`: Grafana env clarity
- `intent`: Clarify and preserve Grafana credential alignment by documenting required compose vars in `.env.example`.
- `files_changed`:
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static compose/env reference check for `GF_ADMIN_USER` and `GF_ADMIN_PASSWORD`.
- `rollback_note`: Revert `.env.example` Grafana variable additions.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 24
- `timestamp`: 2026-02-22
- `milestone`: Evaluation reset and re-baselining
- `intent`: Archive legacy evaluation scripts/datasets, disable CI calls to legacy test paths, and replace evaluation planning with a strict Phase 1 synthetic-data contract and gated phase exits.
- `files_changed`:
  - `bin/legacy_tests/tests/__init__.py`
  - `bin/legacy_tests/tests/test_app.py`
  - `bin/legacy_tests/tests/eval/__init__.py`
  - `bin/legacy_tests/tests/eval/run_eval.py`
  - `bin/legacy_tests/tests/eval/test_eval_regression.py`
  - `bin/legacy_tests/tests/eval/testset_medical.jsonl`
  - `bin/legacy_tests/README.md`
  - `tests/README.md`
  - `.github/workflows/eval.yml`
  - `Improvement Phase Planning/Evaluation and Experiment Planning.md`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - Static review of archive paths and CI workflow step changes.
- `rollback_note`: Move archived files back under `tests/`, restore old workflow test steps, and revert plan document.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: Evaluation implementation direction changed from incremental patching to full reset from archived baseline.

## Entry 25
- `timestamp`: 2026-02-22
- `milestone`: Evaluation plan reconciliation + retriever-first routing
- `intent`: Reconcile planning document by combining original detailed tier design with strict schema/gates, and implement retriever-first routing for all non-chitchat queries with controlled fallback to web/literature based on retrieval confidence.
- `files_changed`:
  - `Improvement Phase Planning/Evaluation and Experiment Planning.md`
  - `eval/golden/v1/README.md`
  - `core/settings.py`
  - `.env`
  - `.env.example`
  - `core/state_v2.py`
  - `agents/planner_agent.py`
  - `agents/retriever_agent.py`
  - `core/langgraph_workflow.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q app.py api core agents db tools observability scripts`
  - Static route/signal verification for `planned_route`, `post_retrieval_route`, and `ROUTING_RAG_CONFIDENCE_THRESHOLD`.
- `rollback_note`: Revert listed files to restore previous planner branching and prior plan document content.
- `HITL_required`: yes
- `HITL_status`: pending
- `direction_change_note`: Routing policy changed from planner-direct web/literature branching to retriever-first precheck for all non-chitchat traffic.

## Entry 26
- `timestamp`: 2026-02-22
- `milestone`: Evaluation plan refinement
- `intent`: Incorporate high-value external audit suggestions into the merged planning document by expanding Tier 1 and Tier 4 category coverage, adding category-level quality gates, and explicitly requiring Phase 2 CI to source datasets from `eval/golden/v1/`.
- `files_changed`:
  - `Improvement Phase Planning/Evaluation and Experiment Planning.md`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Planning/document review only.
- `rollback_note`: Revert planning document edits to previous merged revision.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 27
- `timestamp`: 2026-02-22
- `milestone`: Phase 1 eval validator hardening
- `intent`: Fix false-positive quality gates in `eval.validate` and align Tier 1 expansion provenance with env-driven model configuration.
- `files_changed`:
  - `eval/validate.py`
  - `eval/expand_tier1.py`
  - `eval/golden/v1/qa_report.json`
  - `eval/golden/v1/golden_v1_manifest.json`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`:
  - Strengthened eval schema enforcement by requiring `expected_planned_route` and `expected_final_route`.
  - Added strict guardrail/conversational category validation against required category sets.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q eval`
  - `python3 -m eval.validate --strict` (expected fail until Tier 1 expansion + Tier 3 population complete)
- `rollback_note`: Revert listed files to previous validator/expander logic and prior generated report/manifest.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 28
- `timestamp`: 2026-02-22
- `milestone`: Local runtime persistence alignment
- `intent`: Ensure files generated by in-container eval scripts persist to host workspace by bind-mounting the repo into the app container.
- `files_changed`:
  - `docker-compose.yml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static compose review confirming `app.volumes` includes `./:/app`.
- `rollback_note`: Remove `app.volumes` bind mount and recreate app container.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 29
- `timestamp`: 2026-02-22
- `milestone`: Tier 3 setup script reliability
- `intent`: Fix `ModuleNotFoundError: No module named 'db'` when running setup scripts as file paths inside Docker by adding repository-root path bootstrap.
- `files_changed`:
  - `scripts/reindex_pdf.py`
  - `scripts/init_postgres.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q scripts`
- `rollback_note`: Revert script path-bootstrap lines and run scripts using explicit `PYTHONPATH=/app`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 30
- `timestamp`: 2026-02-22
- `milestone`: Tier 3 dependency readiness
- `intent`: Resolve runtime permission/install issues for RAGAS by declaring Tier 3 dependencies in project manifests so they install during image build (root layer) rather than via in-container `pip install` as non-root user.
- `files_changed`:
  - `requirements.txt`
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Added `ragas`
  - Added `datasets`
- `risk_level`: Low.
- `tests_run`:
  - Static dependency manifest review only.
- `rollback_note`: Remove added dependencies and rebuild app image.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 31
- `timestamp`: 2026-02-22
- `milestone`: Tier 3 RAGAS generator hardening
- `intent`: Patch `eval/generate_tier3_rag.py` to align with production embedding/chunking behavior, enforce schema-valid context coverage, add near-duplicate filtering, improve provenance model accuracy, and add compatibility guards for RAGAS API/version drift.
- `files_changed`:
  - `eval/generate_tier3_rag.py`
  - `requirements.txt`
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`:
  - Tier 3 generator now skips rows lacking non-empty `ground_truth_contexts` to remain schema-valid for `eval.validate`.
- `dependency_changes`:
  - Constrained `ragas` to `>=0.2,<0.4`.
  - Constrained `datasets` to `>=2,<4`.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q eval`
- `rollback_note`: Revert listed files to restore previous Tier 3 generator behavior and dependency constraints.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 32
- `timestamp`: 2026-02-22
- `milestone`: Embedding configuration realignment
- `intent`: Make local embeddings the explicit primary path (`Alibaba-NLP/gte-base-en-v1.5`) with Gemini as backup-only, and align settings/env/runtime call sites to remove ambiguous provider-model coupling.
- `files_changed`:
  - `core/settings.py`
  - `tools/embedding_client.py`
  - `tools/vector_store.py`
  - `agents/retriever_agent.py`
  - `tools/cache.py`
  - `eval/generate_tier3_rag.py`
  - `.env`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q core tools agents eval scripts`
- `rollback_note`: Revert listed files to previous embedding provider/model settings and `embed_text` call paths.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Embedding strategy changed from mixed auto routing to explicit primary/backup provider order to reduce quota-related instability.

## Entry 33
- `timestamp`: 2026-02-22
- `milestone`: Local embedding strategy control
- `intent`: Make local embedding behavior explicit for `gte-base-en-v1.5` by defaulting to plain `model.encode(...)` and add a configurable strategy (`plain|asymmetric|auto`) for future model swaps.
- `files_changed`:
  - `core/settings.py`
  - `tools/embedding_client.py`
  - `.env`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q core tools`
- `rollback_note`: Revert listed files to prior local embedding strategy behavior.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Added strategy-level configurability to support model-dependent encode behavior without code edits.

## Entry 34
- `timestamp`: 2026-02-22
- `milestone`: Embedding integrity hardening
- `intent`: Eliminate silent vector trim/pad behavior by replacing normalization with strict dimension validation that raises on mismatch.
- `files_changed`:
  - `tools/embedding_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q tools/embedding_client.py`
- `rollback_note`: Revert strict validation function and restore previous normalization-based embedding handling.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 35
- `timestamp`: 2026-02-22
- `milestone`: Embedding consistency enforcement
- `intent`: Disable embedding backup during normal indexing flow to prevent cross-model vector-space mixing (local primary only).
- `files_changed`:
  - `.env`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static env contract review.
- `rollback_note`: Set `EMBEDDING_ENABLE_BACKUP=true` and recreate app container.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Backup policy tightened for retrieval integrity.

## Entry 36
- `timestamp`: 2026-02-22
- `milestone`: Local model cache reliability
- `intent`: Fix local embedding model load failures caused by non-writable default home/cache path in container by introducing explicit writable embedding cache dir configuration.
- `files_changed`:
  - `core/settings.py`
  - `tools/embedding_client.py`
  - `.env`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q core tools`
- `rollback_note`: Remove `EMBEDDING_CACHE_DIR` support and revert model loader cache folder settings.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 37
- `timestamp`: 2026-02-25
- `milestone`: Local embedding bootstrap hardening
- `intent`: Fix container local-embedding startup failures where Hugging Face cache paths resolve to non-writable home directories by forcing a writable cache/home layout before importing `sentence_transformers`.
- `files_changed`:
  - `tools/embedding_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q tools/embedding_client.py core/settings.py tools/vector_store.py`
- `rollback_note`: Revert `tools/embedding_client.py` to previous import/cache setup.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 38
- `timestamp`: 2026-02-25
- `milestone`: Local embedding cache-path enforcement
- `intent`: Fix residual `/home/appuser` permission failures during reindex by forcing additional Hugging Face/Transformers/Torch/XDG cache env vars and patching already-imported runtime constants to the writable cache tree.
- `files_changed`:
  - `tools/embedding_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q tools/embedding_client.py`
- `rollback_note`: Revert `tools/embedding_client.py` to Entry 37 state.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 39
- `timestamp`: 2026-02-25
- `milestone`: Startup-time embedding cache hardening
- `intent`: Prevent Transformers from freezing `/home/appuser` cache constants at import time by exporting writable Hugging Face/Transformers cache env vars in `docker-compose` app startup environment and adding a direct patch for `transformers.dynamic_module_utils.HF_MODULES_CACHE`.
- `files_changed`:
  - `docker-compose.yml`
  - `tools/embedding_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q tools/embedding_client.py`
  - YAML parse validation for `docker-compose.yml`
- `rollback_note`: Remove added app env vars in `docker-compose.yml` and revert dynamic module cache constant patch in `tools/embedding_client.py`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 40
- `timestamp`: 2026-02-25
- `milestone`: Consolidation note for Entries 36-39
- `intent`: Reduce ambiguity by consolidating the embedding-cache incident timeline. Entries 36-38 were incremental remediation attempts; Entry 39 is the final effective fix that resolved startup-time cache path freezing.
- `files_changed`:
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - n/a (documentation-only consolidation).
- `rollback_note`: Remove this consolidation note only; keep Entries 36-39 unchanged for historical traceability.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Resolution summary:
  - Entry 36: introduced explicit cache dir support.
  - Entry 37: improved import/cache bootstrap ordering.
  - Entry 38: expanded runtime cache overrides.
  - Entry 39: final fix by enforcing startup environment + dynamic module constant patch.

## Entry 41
- `timestamp`: 2026-02-28
- `milestone`: Dependency manifest cleanup
- `intent`: Remove unused and misplaced runtime dependencies to reduce image bloat while preserving current Tier 3 generation workflow inside the main app container.
- `files_changed`:
  - `requirements.txt`
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Removed `langchain-huggingface` from runtime/package dependencies because current runtime embedding path uses native `sentence-transformers`, not LangChain HF wrappers.
  - Removed `pytest` from `requirements.txt` because it is a test dependency already declared under `pyproject.toml` `dev` extras.
  - Kept `ragas` and `datasets` in base requirements because Tier 3 generation is currently executed inside the main Docker app container.
- `risk_level`: Low.
- `tests_run`:
  - Static import/dependency audit via `rg`.
- `rollback_note`: Re-add removed packages to `requirements.txt` / `pyproject.toml` and rebuild the app image.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Did not move `ragas`/`datasets` out of runtime yet because doing so would break the current in-container evaluation workflow until Docker/package install flow is refactored.

## Entry 42
- `timestamp`: 2026-02-28
- `milestone`: Packaging source-of-truth split
- `intent`: Make `pyproject.toml` the single dependency source of truth, remove hand-maintained `requirements.txt`, separate eval-only dependencies into an `eval` extra, and split runtime versus eval container install paths.
- `files_changed`:
  - `pyproject.toml`
  - `Dockerfile`
  - `docker-compose.yml`
  - `.github/workflows/eval.yml`
  - `render.yaml`
  - `eval/generate_tier3_rag.py`
  - `README.md`
  - `Improvement Phase Planning/Fix Plan.md`
  - `requirements.txt` (deleted)
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Moved `ragas>=0.2,<0.4` and `datasets>=2,<4` out of base dependencies into `pyproject.toml` `eval` extras.
  - Removed `requirements.txt` as an install source.
  - Runtime image now installs `pip install .`; eval image installs `pip install ".[eval]"` via Docker build arg.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 - <<'PY'\nimport tomllib\nwith open('pyproject.toml','rb') as f:\n    tomllib.load(f)\nprint('pyproject.toml: OK')\nPY`
  - `python3 - <<'PY'\nimport yaml\nwith open('docker-compose.yml','r',encoding='utf-8') as f:\n    yaml.safe_load(f)\nprint('docker-compose.yml: OK')\nPY`
  - `python3 - <<'PY'\nimport yaml\nwith open('.github/workflows/eval.yml','r',encoding='utf-8') as f:\n    yaml.safe_load(f)\nprint('.github/workflows/eval.yml: OK')\nPY`
  - `python3 - <<'PY'\nimport yaml\nwith open('render.yaml','r',encoding='utf-8') as f:\n    yaml.safe_load(f)\nprint('render.yaml: OK')\nPY`
- `rollback_note`: Restore `requirements.txt`, revert Docker/Compose/CI/deploy/docs to the previous `requirements.txt` install path, and rebuild both app and eval images.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Tier 3 generation is no longer expected to run inside the lean runtime app container; it now belongs to the profile-gated eval install path.

## Entry 43
- `timestamp`: 2026-02-28
- `milestone`: Repo hygiene ignore rules
- `intent`: Ignore local Claude workspace files and common local Python/tool caches so repo status reflects only meaningful source changes.
- `files_changed`:
  - `.gitignore`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static file audit of local-only directories.
- `rollback_note`: Remove the added ignore entries from `.gitignore`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 44
- `timestamp`: 2026-03-02
- `milestone`: Embedding cache persistence
- `intent`: Persist Hugging Face / sentence-transformers / torch cache artifacts across container recreates so runtime and eval services do not repeatedly re-download embedding model assets.
- `files_changed`:
  - `docker-compose.yml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 - <<'PY'\nimport yaml\nwith open('docker-compose.yml','r',encoding='utf-8') as f:\n    yaml.safe_load(f)\nprint('docker-compose.yml: OK')\nPY`
- `rollback_note`: Remove the `embedding_cache` named volume and its mounts from `docker-compose.yml`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 45
- `timestamp`: 2026-03-02
- `milestone`: Container runtime permissions and eval Git support
- `intent`: Fix bind-mount write permission issues for the eval dataset outputs and remove `ragas` import failures by aligning container user UID/GID with the host defaults and installing the `git` executable in the image.
- `files_changed`:
  - `Dockerfile`
  - `docker-compose.yml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`:
  - Added OS package `git` to the container image for `ragas` / `GitPython` compatibility.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 - <<'PY'\nimport yaml\nwith open('docker-compose.yml','r',encoding='utf-8') as f:\n    yaml.safe_load(f)\nprint('docker-compose.yml: OK')\nPY`
  - `python3 - <<'PY'\nfrom pathlib import Path\ntext = Path('Dockerfile').read_text(encoding='utf-8')\nassert 'apt-get install -y --no-install-recommends git' in text\nassert 'ARG APP_UID=1000' in text\nassert 'ARG APP_GID=1000' in text\nprint('Dockerfile: OK')\nPY`
- `rollback_note`: Remove the `git` install and UID/GID build args from `Dockerfile`, remove the new build args from `docker-compose.yml`, and rebuild the images.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 46
- `timestamp`: 2026-03-03
- `milestone`: LLM default configuration alignment
- `intent`: Remove misleading default-model/provider drift by aligning `core/settings.py` and `.env.example` with the current Groq primary/fallback configuration used in `.env`.
- `files_changed`:
  - `core/settings.py`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q core/settings.py tools/llm_client.py`
- `rollback_note`: Restore the previous default LLM models/providers in `core/settings.py` and `.env.example`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Defaults now match the active Groq runtime choice instead of older generic fallback values.

## Entry 47
- `timestamp`: 2026-03-03
- `milestone`: LLM provider flow simplification
- `intent`: Remove the unused `auto` provider-selection loop from the LLM client so primary/fallback behavior is explicit and easier to reason about under the current Groq-only provider configuration.
- `files_changed`:
  - `tools/llm_client.py`
  - `.env.example`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q tools/llm_client.py core/settings.py`
- `rollback_note`: Restore `_provider_candidates()` loop logic and the `auto` provider wording in `.env.example`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: OpenAI remains available as an explicit provider option, but provider auto-selection is no longer supported.

## Entry 48
- `timestamp`: 2026-03-03
- `milestone`: Embedding backup default alignment
- `intent`: Align `core/settings.py` default embedding backup behavior with the current `.env` contract by disabling backup by default to avoid cross-model vector-space mixing when env vars are absent or partial.
- `files_changed`:
  - `core/settings.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - `python3 -m compileall -q core/settings.py tools/embedding_client.py`
- `rollback_note`: Restore `embedding_enable_backup` default to `True` in `core/settings.py`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Code defaults now match the existing runtime/env embedding policy.

## Entry 49
- `timestamp`: 2026-03-03
- `milestone`: Embedding client cleanup and failure-surface simplification
- `intent`: Remove the dead hash-based embedding fallback, delete the unused `embed_text()` alias after confirming no live callers, keep explicit `plain|asymmetric` local strategy support, and move embedding cache/bootstrap setup into a dedicated helper so the client code reflects the actual runtime behavior more clearly.
- `files_changed`:
  - `tools/embedding_client.py`
  - `tools/embedding_bootstrap.py`
  - `core/settings.py`
  - `.env`
  - `.env.example`
  - `eval/generate_tier3_rag.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `rg -n "embed_text\\(|EMBEDDING_ALLOW_HASH_FALLBACK|hash_fallback" tools core eval agents .env .env.example`
  - `python3 -m compileall -q tools/embedding_client.py tools/embedding_bootstrap.py core/settings.py eval/generate_tier3_rag.py`
- `rollback_note`: Restore the removed hash-fallback setting/function/alias and inline cache bootstrap logic if the simplified embedding client causes regressions.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: `EMBEDDING_LOCAL_STRATEGY=auto` is now treated as a deprecated alias for `asymmetric` rather than a separately advertised mode because the prior implementation did not create materially different behavior.

## Entry 50
- `timestamp`: 2026-03-04
- `milestone`: Embedding code maintainability note
- `intent`: Add a concise code comment at the asymmetric local embedding dispatch point so future model swaps explicitly account for model-specific retrieval APIs instead of assuming `encode_query(...)` / `encode_document(...)` are universally available.
- `files_changed`:
  - `tools/embedding_client.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static code inspection.
- `rollback_note`: Remove the added comment and log entry.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 51
- `timestamp`: 2026-03-04
- `milestone`: Chunking and retrieval hardening follow-up
- `intent`: Close the remaining high-risk gaps in the new chunking/retrieval implementation by making document replacement transactional, adding bounded-flush batch upserts, separating embedding-oriented vs generation-oriented chunk payloads, tightening executor prompt budgeting around the assembled prompt, and adding runtime dependency support for tokenizer-aware budgeting.
- `files_changed`:
  - `tools/vector_store.py`
  - `db/repositories.py`
  - `agents/executor_agent.py`
  - `tools/pdf_chunker.py`
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: Internal chunk metadata expanded with `embedding_prefix`, `embedding_text`, `context_key`, and `table_serializer`. No external HTTP API changes.
- `dependency_changes`:
  - Added `tiktoken` to runtime dependencies for deterministic prompt-budget estimation in the executor.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q agents/executor_agent.py tools/vector_store.py db/repositories.py tools/pdf_chunker.py`
  - `rg -n "_MAX_TOTAL_TOKENS|_MAX_PROMPT_TOKENS|tiktoken|embedding_prefix|embedding_text|context_key|replace_doc_id|batch_size=250|table_serializer|table_quality" agents/executor_agent.py tools/vector_store.py db/repositories.py tools/pdf_chunker.py pyproject.toml`
- `rollback_note`: Restore the pre-hardening executor context builder, revert `tools/vector_store.py` to the simpler batch path, and remove the new chunk metadata fields and `tiktoken` dependency if the revised chunking/retrieval pipeline causes regressions.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: This entry hardens the user-implemented chunking foundation rather than changing the overall roadmap; the main change is that purge-and-replace and prompt budgeting now follow stricter safety rules.

## Entry 52
- `timestamp`: 2026-03-04
- `milestone`: Chunking, ingestion, and retrieval optimization consolidation
- `intent`: Record the full chunking/retrieval optimization pass in plain language so future review can distinguish the user-authored implementation work from the assistant-authored hardening work without relying on internal phase labels.
- `files_changed`:
  - `eval/chunking_eval.py`
  - `tools/pdf_parser.py`
  - `tools/pdf_chunker.py`
  - `tools/pdf_loader.py`
  - `core/settings.py`
  - `agents/executor_agent.py`
  - `tools/embedding_client.py`
  - `tools/vector_store.py`
  - `db/repositories.py`
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: Internal retrieval payloads and metadata were expanded to support structure-aware chunking and safer prompt assembly. New internal metadata now includes section-aware context and embedding-oriented fields such as `context_prefix`, `context_key`, `embedding_prefix`, `embedding_text`, `table_quality`, and `table_serializer`. No public HTTP route contract changed.
- `dependency_changes`:
  - Added `docling` under the optional `ingest` dependency group for structure-aware PDF parsing.
  - Added `tiktoken` to runtime dependencies so executor-side prompt budgeting can use deterministic token estimation instead of raw character counts.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q eval/chunking_eval.py tools/pdf_parser.py tools/pdf_chunker.py tools/pdf_loader.py core/settings.py agents/executor_agent.py tools/embedding_client.py tools/vector_store.py db/repositories.py`
  - `rg -n "CHUNKER_BACKEND|embed_documents_batch|upsert_chunks_batch|replace_doc_id|context_prefix|context_key|embedding_prefix|embedding_text|table_quality|table_serializer|tiktoken|_MAX_PROMPT_TOKENS" eval tools core agents db pyproject.toml`
- `rollback_note`: Revert the Docling parser/chunker path, restore legacy `process_pdf()` behavior, remove the batch embedding/upsert path, restore the older executor context assembly logic, and drop `docling` / `tiktoken` if the new chunking stack is rolled back.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: Consolidated summary of chunking/retrieval work completed across two authors.
  User-authored implementation work:
  - Added `eval/chunking_eval.py` to measure chunk counts, token distributions, context/section coverage, retrieval recall, and snapshot comparisons.
  - Added `tools/pdf_parser.py` to introduce Docling-based PDF parsing with typed elements, section tracking, table extraction, and fallback behavior.
  - Added `tools/pdf_chunker.py` to create structure-aware chunks, isolate tables, assign stable chunk IDs, and attach section/context metadata.
  - Rewired `tools/pdf_loader.py` so `process_pdf()` can select the new parser/chunker path while preserving a legacy fallback.
  - Added `CHUNKER_BACKEND` in `core/settings.py` to control which chunking backend is active.
  - Replaced the old per-document `[:900]` executor truncation with a first-pass global context cap in `agents/executor_agent.py`.
  - Added `embed_documents_batch()` in `tools/embedding_client.py` to batch local embedding calls where the model/runtime supports it.
  - Updated `tools/vector_store.py` to use batch embedding inputs and new chunk metadata fields.
  - Added the optional `ingest` dependency group in `pyproject.toml` so parser-heavy ingestion dependencies are separated from the lean runtime app image.
  Assistant-authored hardening work:
  - Updated `tools/vector_store.py` and `db/repositories.py` so document replacement is transactional, chunk insertion happens in bounded batches, and reindex runs replace existing rows for the same `doc_id` instead of leaving stale vectors behind.
  - Updated `agents/executor_agent.py` so prompt assembly is token-budgeted, grouped by context, and table-aware instead of relying on raw character slicing.
  - Updated `tools/pdf_chunker.py` so generation-facing content and embedding-facing content can differ, table serialization quality is tracked, and complex tables can use different retrieval vs generation payloads.
  - Added `tiktoken` to `pyproject.toml` to support deterministic executor budgeting.
  - Rewrote this change-log area so the work is described in direct operational language instead of internal phase terminology.

## Entry 53
- `timestamp`: 2026-03-05
- `milestone`: Chunking fallback and ingest isolation refinement
- `intent`: Address follow-up audit gaps by isolating heavy PDF ingestion from the runtime app container, aligning chunk-size token estimation with the executor tokenizer, replacing the chunker’s opaque regex-only fallback with a transparent plain-Python structural splitter, and preventing Markdown table syntax from being penalized by the lightweight reranker.
- `files_changed`:
  - `tools/pdf_chunker.py`
  - `agents/retriever_agent.py`
  - `docker-compose.yml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None beyond the previously added `tiktoken` runtime dependency and `ingest` optional dependency group.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q tools/pdf_chunker.py agents/retriever_agent.py`
  - Static verification that `docker-compose.yml` contains the new `ingest` service definition.
- `rollback_note`: Remove the `ingest` service from `docker-compose.yml`, restore the older `_split_long_text()` and `_count_tokens()` behavior in `tools/pdf_chunker.py`, and revert the reranker token normalization in `agents/retriever_agent.py`.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: This pass intentionally did not add automatic pgvector `REINDEX` calls. Index maintenance is being kept as an operational follow-up step after real latency/recall measurements rather than hardwiring an expensive database maintenance command into every ingestion run.

## Entry 54
- `timestamp`: 2026-03-05
- `milestone`: Chunking cleanup and workflow evaluation alignment
- `intent`: Remove the remaining legacy chunking path completely, make PDF parsing fail fast instead of silently falling back, tighten parser heuristics for excluded sections and fake headings, align chunk-analysis token counts with runtime budgeting, and add a real workflow-level evaluation harness so chunking changes can be compared through the actual retrieval/executor/LLM path.
- `files_changed`:
  - `tools/pdf_loader.py`
  - `tools/pdf_parser.py`
  - `core/settings.py`
  - `pyproject.toml`
  - `README.md`
  - `eval/chunking_eval.py`
  - `eval/workflow_eval.py`
  - `eval/generate_tier3_rag.py`
  - `docker-compose.yml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None on public HTTP routes. Internal ingest behavior changed from fallback-capable to fail-fast; `process_pdf()` now represents only the Docling-based chunking pipeline.
- `dependency_changes`:
  - Removed `langchain-text-splitters` from runtime dependencies.
  - Updated the `eval` compose service to install both `eval` and `ingest` extras because `generate_tier3_rag.py` now depends on the Docling-based loader path.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q tools/pdf_loader.py tools/pdf_parser.py eval/chunking_eval.py eval/workflow_eval.py core/settings.py agents/retriever_agent.py`
  - `rg -n "CHUNKER_BACKEND|chunker_backend|RecursiveCharacterTextSplitter|PyPDFLoader|langchain-text-splitters|backend: 'docling' or 'legacy'|falling back to legacy|Falling back to basic PyPDFLoader" .`
- `rollback_note`: Restore the legacy PDF loading/splitting path, restore `CHUNKER_BACKEND`, add `langchain-text-splitters` back to dependencies, and remove `eval/workflow_eval.py` if the fail-fast Docling-only chunking pipeline is rolled back.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: This change intentionally raises the bar for chunking comparisons. The repository no longer keeps a silent fallback chunker because that would mix parser behaviors and make old-vs-new comparisons scientifically ambiguous. The new evaluation split is: `eval/chunking_eval.py` for component-level chunk structure and retrieval substrate checks, and `eval/workflow_eval.py` for end-to-end workflow behavior.

## Entry 55
- `timestamp`: 2026-03-05
- `milestone`: Retrieval evaluation and gold-label strengthening
- `intent`: Replace the earlier lightweight retrieval benchmark with a stronger experiment-grade retrieval evaluation harness, and strengthen future Tier 3 sample generation so retrieval experiments can use stable chunk-level labels instead of relying only on free-text context matching.
- `files_changed`:
  - `eval/retrieval_eval.py`
  - `eval/generate_tier3_rag.py`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None on public HTTP routes. Tier 3 generated JSONL samples now include retrieval-grounding fields such as `ground_truth_chunk_ids`, `ground_truth_doc_ids`, `ground_truth_pages`, and `ground_truth_content_types` when alignment can be inferred from the production chunk set.
- `dependency_changes`: None.
- `risk_level`: Medium.
- `tests_run`:
  - `python3 -m compileall -q eval/retrieval_eval.py eval/generate_tier3_rag.py eval/workflow_eval.py eval/chunking_eval.py tools/pdf_loader.py tools/pdf_parser.py tools/pdf_chunker.py`
  - `rg -n "MarkdownTextSplitter|RecursiveCharacterTextSplitter|PyPDFLoader|langchain-text-splitters|CHUNKER_BACKEND" tools eval pyproject.toml README.md`
- `rollback_note`: Restore the earlier retrieval-eval harness, remove the extra Tier 3 alignment fields from generated samples, and return to context-only retrieval scoring if the stronger benchmark contract is rolled back.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: The evaluation stack is now split more explicitly into three layers: component chunk analysis (`eval/chunking_eval.py`), retrieval benchmarking over the real indexed retriever (`eval/retrieval_eval.py`), and end-to-end workflow behavior (`eval/workflow_eval.py`). This raises the quality bar for future chunking and retrieval experiments, but it also means old Tier 3 datasets without chunk-level labels are now a weaker benchmark than newly generated ones.

## Entry 56
- `timestamp`: 2026-03-08
- `milestone`: Packaging fix for eval/ingest image builds
- `intent`: Fix Docker image build failure caused by an invalid empty URL in `pyproject.toml`, which prevented the `eval` image from rebuilding with the new `ingest` dependencies and left `docling` unavailable at runtime.
- `files_changed`:
  - `pyproject.toml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static validation by matching the Docker build error to `project.urls.Repository = ""` in `pyproject.toml`.
- `rollback_note`: Restore the removed `project.urls` section only if it is replaced with a valid URL value.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.

## Entry 57
- `timestamp`: 2026-03-08
- `milestone`: Cache-path permission fix for eval/ingest runtime
- `intent`: Fix Docling/Hugging Face cache permission failures in the `eval` and `ingest` containers by moving the shared model cache root from a Docker-managed named volume to a repo-local bind-mounted path under `/app/.cache/embeddings`, which matches the non-root container UID/GID and avoids root-owned volume directories.
- `files_changed`:
  - `docker-compose.yml`
  - `Improvement Phase Planning/Implementation Change Log.md`
- `api_schema_impact`: None.
- `dependency_changes`: None.
- `risk_level`: Low.
- `tests_run`:
  - Static verification of `docker-compose.yml` cache-path changes against the observed permission error path `/tmp/medigenius-embeddings/huggingface_hub`.
- `rollback_note`: Restore the named `embedding_cache` volume only if a root-owned volume init/chown strategy is introduced at container startup.
- `HITL_required`: no
- `HITL_status`: n/a
- `direction_change_note`: None.
