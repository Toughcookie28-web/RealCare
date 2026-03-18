# MediGenius Unified Upgrade Plan (Greenfield FastAPI + PostgreSQL/pgvector)

## Summary
This plan replaces the current Flask + SQLite/Chroma stack with a direct FastAPI cutover and a fresh PostgreSQL/pgvector foundation (no data backfill), then layers observability and evaluation early before advanced agentic features. It also formalizes human-in-the-loop (HITL) approval gates and a mandatory detailed change log.

## Governing Artifacts
1. `Improvement Phase Planning/Medi Genius Unified Upgrade Plan.md`
2. `Improvement Phase Planning/Implementation Change Log.md`

## Locked Directional Decisions
1. Architecture style: greenfield rebuild (no preservation migration complexity).
2. API transition: direct FastAPI cutover.
3. Data strategy: fresh start (no backfill of SQLite/Chroma files).
4. Database target: PostgreSQL + pgvector via Docker for development; env-driven for managed production.
5. Observability baseline: OpenTelemetry + Prometheus + Grafana.
6. HITL mode: milestone gates plus mandatory pause on direction changes.

## Milestone 0: Governance, Contracts, Skeleton
- Create governance files and change-log policy.
- Define typed contracts for API payloads, node outputs, and workflow state.
- Define environment contract in `.env.example`.
- Define error taxonomy and node result conventions.

## Milestone 1: FastAPI Platform Foundation
- Replace Flask runtime with FastAPI.
- Keep current frontend UX and wire existing endpoints to FastAPI.
- Add `/health/live`, `/health/ready`, `/metrics`.
- Keep endpoint parity for chat/history/session flows.

## Milestone 2: PostgreSQL Relational Layer
- Add SQLAlchemy 2.0 data model and repositories.
- Add Alembic migration baseline.
- Persist sessions, messages, summaries, and user facts in Postgres.
- Remove runtime SQLite dependency.

## Milestone 3: pgvector Knowledge Layer
- Add pgvector extension/table support.
- Re-index medical PDFs into Postgres vector table.
- Replace Chroma retrieval path with pgvector repository.
- Preserve chunk metadata for explainability.

## Milestone 4: Observability + Early Evaluation
- Structured JSON logs with trace correlation.
- OpenTelemetry tracing hooks.
- Prometheus metrics and baseline Grafana dashboard assets.
- Evaluation harness with regression thresholds.

## Milestone 5: Core Agentic Intelligence
- Query rewriting node.
- Semantic routing (`chitchat`, `vector`, `web`, `future_clinical_db`).
- Summary-buffer memory and user fact extraction.
- Semantic cache for repeated/near-duplicate queries.

## Milestone 6: Retrieval and Safety Hardening
- Hybrid retrieval (dense + sparse).
- Reranker stage before generation.
- Trusted medical domain policy.
- PubMed/Europe PMC tool for literature requests.
- Input guardrail node.

## Milestone 7: Reliability, XAI, and UX Transparency
- Reflection/critique node with bounded retries.
- Primary/fallback executor path.
- Citation mapping to retrieved chunks.
- SSE status+token streaming.
- Optional runtime HITL interrupt for high-risk intents.

## Acceptance Targets
1. FastAPI is sole backend runtime.
2. SQLite/Chroma removed from active code path.
3. PostgreSQL/pgvector are active storage layers.
4. Observability and eval are available before advanced node rollout.
5. Change log is complete and auditable.
6. HITL gates recorded before milestone transitions.

## Assumptions
1. Existing local SQLite/Chroma data is disposable.
2. Source medical PDFs are available for re-ingestion.
3. Python 3.11+, Docker, docker-compose available.
4. Model API keys provided through env vars.
