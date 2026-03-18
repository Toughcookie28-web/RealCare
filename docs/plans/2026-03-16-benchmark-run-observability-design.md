# Benchmark Run Observability Design

## Goal
Add durable benchmark run tracking for retrieval, RAGAS, and workflow evaluations in Grafana, while keeping live request / runtime proxy metrics in Prometheus. The benchmark surface should be queryable over time, comparable across runs, and linked back to the saved snapshot artifacts on disk.

## Why this design
Prometheus is the right place for live operational signals such as request rate, route mix, cache hit rate, node latency, and error counts. It is not a good long-term store for benchmark-history metadata because eval runs need stable identifiers, artifact paths, dataset provenance, git SHA, and slice-level summary rows without turning labels into high-cardinality experiment tracking.

The clean split is:
- Prometheus: live runtime proxies
- PostgreSQL: benchmark run history
- JSON snapshots: full per-run artifacts
- Grafana: one dashboard that combines both data sources

## Recommended architecture

### 1. Persist benchmark run summaries in PostgreSQL
Add a small eval-run schema in Postgres for summary-level observability.

Recommended tables:
- `eval_runs`
  - `run_id`
  - `created_at`
  - `eval_type` (`retrieval`, `ragas`, `workflow`)
  - `mode`
  - `split`
  - `sample_count`
  - `git_sha`
  - `dataset_path`
  - `snapshot_path`
  - `judge_model` (nullable)
  - `notes` / `metadata_json` (nullable JSONB)
- `eval_run_metrics`
  - `run_id`
  - `metric_name`
  - `metric_value`
- `eval_run_slices`
  - `run_id`
  - `slice_name`
  - `metric_name`
  - `metric_value`
  - `sample_count`

This stays intentionally summary-level. We keep the full per-sample detail in the snapshot JSON files instead of duplicating that in SQL.

### 2. Add a shared eval-run tracker
Create one internal helper used by:
- `eval/retrieval_eval.py`
- `eval/ragas_eval.py`
- `eval/workflow_eval.py`

Responsibilities:
- generate / accept a stable `run_id`
- collect common metadata
- persist run summary rows after snapshot save
- remain optional and fail-loud enough for debugging, but not corrupt the eval report object

The tracker should not own benchmark logic. It only records already-computed results.

### 3. Keep snapshots as the artifact of record
The JSON snapshots remain the detailed artifact. The DB row should point to them via `snapshot_path`.

This gives you:
- Grafana trend and comparison views
- disk artifact for deeper inspection
- no duplication of full per-sample payload into the DB

### 4. Add a PostgreSQL Grafana datasource
Grafana already reads Prometheus. Add a second provisioned datasource for the app Postgres DB and build a benchmark dashboard section from SQL queries.

Dashboard sections:

**Benchmark Runs**
- latest retrieval metrics
- retrieval trends over time
- latest RAGAS metrics
- RAGAS trends over time
- workflow trends over time
- latest-vs-baseline table
- run inventory table:
  - `run_id`
  - `created_at`
  - `eval_type`
  - `mode`
  - `split`
  - `git_sha`
  - `snapshot_path`

**Live Runtime**
- existing Prometheus panels stay:
  - HTTP request rate
  - latency
  - node errors
  - cache hit/miss
  - Redis status
- optional additions later:
  - route distribution
  - `/api/chat` error rate
  - retry / reflection count

## Data flow

### Eval run path
1. User runs `retrieval_eval`, `ragas_eval`, or `workflow_eval`
2. Script computes report as it does today
3. Snapshot is saved to disk
4. Shared tracker writes run metadata + summary metrics + slices to Postgres
5. Grafana PostgreSQL panels pick up the new run automatically

### Live runtime path
1. User traffic hits the app
2. FastAPI middleware and node-level metrics emit Prometheus data
3. Prometheus scrapes `/metrics`
4. Grafana Prometheus panels show runtime proxies

## Scope decisions

### In scope
- benchmark run history for:
  - retrieval eval
  - RAGAS eval
  - workflow eval
- Grafana benchmark panels backed by Postgres
- live runtime proxy panels remain in Prometheus
- run metadata that makes comparisons trustworthy

### Out of scope
- storing full per-sample eval payloads in SQL
- replacing snapshots with Grafana
- making live user traffic appear as benchmark truth
- external experiment trackers such as MLflow / W&B

## Required run metadata
Each run record should capture at least:
- `run_id`
- `eval_type`
- `mode`
- `split`
- `sample_count`
- `git_sha`
- `dataset_path`
- `snapshot_path`
- `judge_model` when applicable
- `created_at`

This is the minimum needed to make future analysis trustworthy instead of guesswork.

## Error handling and correctness
- Eval result computation must stay independent from tracking persistence.
- If run tracking fails:
  - the eval should still print/save the main report
  - but log a clear error
- Snapshot save should happen before DB tracking so the DB never points to a missing path from normal success flow.
- Grafana dashboard queries should tolerate empty tables before the first tracked run.

## Migration strategy
- Add a new Alembic migration for eval-run tables.
- Do not attempt historical backfill in the first pass.
- Start tracking only new runs from the point this lands.

That keeps the first version small and reliable.

## Success criteria
- Running any supported eval writes:
  - JSON snapshot artifact
  - one DB run record
  - metric rows
  - slice rows when available
- Grafana shows benchmark trends over time from Postgres
- Grafana still shows live runtime proxies from Prometheus
- Future run analysis no longer depends on memory or chat history to know what code/config produced a result

## Recommended next implementation slice
Build this in four steps:
1. schema + tracker helper
2. wire tracker into eval scripts
3. Grafana Postgres datasource provisioning
4. benchmark dashboard panels

That sequence minimizes risk and gives useful value early.
