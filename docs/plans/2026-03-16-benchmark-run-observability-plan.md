# Benchmark Run Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add benchmark run tracking to PostgreSQL for retrieval, RAGAS, and workflow evals, and surface those run summaries in Grafana while preserving Prometheus for live runtime proxies.

**Architecture:** Eval scripts keep saving JSON snapshots to disk, then write compact run summaries to a new Postgres schema using a shared tracker helper. Grafana gets a PostgreSQL datasource for benchmark panels and keeps Prometheus for live runtime panels, so benchmark truth and live traffic proxies stay clearly separated.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, PostgreSQL, Prometheus, Grafana, pytest

---

### Task 1: Add failing tests for benchmark run tracking schema and helper contracts

**Files:**
- Create: `tests/eval/test_eval_run_tracker_contract.py`
- Test: `tests/eval/test_eval_run_tracker_contract.py`

**Step 1: Write the failing test**

Add tests that assert:
- a tracker helper can build a normalized run payload from a retrieval report
- a tracker helper can flatten top-level metrics and slice metrics into DB-ready rows
- a tracker helper preserves required metadata:
  - `run_id`
  - `eval_type`
  - `mode`
  - `split`
  - `sample_count`
  - `git_sha`
  - `dataset_path`
  - `snapshot_path`

Use small fake reports instead of importing the live eval scripts.

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest -q tests/eval/test_eval_run_tracker_contract.py
```

Expected:
- fail because tracker module/functions do not exist yet

**Step 3: Write minimal implementation**

Create a shared tracker module with pure helpers only:
- `eval/run_tracker.py`

Add functions such as:
- `build_run_record(...)`
- `extract_metric_rows(...)`
- `extract_slice_rows(...)`

Do not write DB persistence yet in this task if the test only covers normalization.

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest -q tests/eval/test_eval_run_tracker_contract.py
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tests/eval/test_eval_run_tracker_contract.py eval/run_tracker.py
git commit -m "test: add eval run tracker normalization contract"
```

### Task 2: Add database schema for eval run tracking

**Files:**
- Create: `alembic/versions/0005_add_eval_run_tracking.py`
- Modify: `db/models.py`
- Test: `tests/smoke/test_eval_run_schema_contract.py`

**Step 1: Write the failing test**

Create `tests/smoke/test_eval_run_schema_contract.py` asserting:
- new ORM models exist for benchmark runs
- expected columns are present in model source:
  - `eval_runs`
  - `eval_run_metrics`
  - `eval_run_slices`
- migration file exists and creates the new tables

Keep this source-level and schema-level, not live DB dependent.

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest -q tests/smoke/test_eval_run_schema_contract.py
```

Expected:
- FAIL because models/migration do not exist yet

**Step 3: Write minimal implementation**

Add ORM models in `db/models.py`:
- `EvalRunModel`
- `EvalRunMetricModel`
- `EvalRunSliceModel`

Add Alembic migration:
- create three tables
- add foreign keys from metric/slice rows to `eval_runs.run_id`
- use simple scalar columns; do not overdesign

Recommended columns:
- `eval_runs`
  - `run_id` primary key
  - `created_at`
  - `eval_type`
  - `mode`
  - `split`
  - `sample_count`
  - `git_sha`
  - `dataset_path`
  - `snapshot_path`
  - `judge_model` nullable
  - `metadata_json` nullable JSONB
- `eval_run_metrics`
  - `id`
  - `run_id`
  - `metric_name`
  - `metric_value`
- `eval_run_slices`
  - `id`
  - `run_id`
  - `slice_name`
  - `metric_name`
  - `metric_value`
  - `sample_count`

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest -q tests/smoke/test_eval_run_schema_contract.py
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add alembic/versions/0005_add_eval_run_tracking.py db/models.py tests/smoke/test_eval_run_schema_contract.py
git commit -m "feat: add eval run tracking schema"
```

### Task 3: Add repository/persistence layer for benchmark runs

**Files:**
- Modify: `db/repositories.py`
- Modify: `eval/run_tracker.py`
- Test: `tests/eval/test_eval_run_tracker_contract.py`

**Step 1: Write the failing test**

Extend the tracker contract test to assert:
- a persistence function writes one run row and related metric/slice rows into a fake session
- the tracker does not try to persist missing metrics
- the tracker accepts reports from retrieval, RAGAS, and workflow shapes

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest -q tests/eval/test_eval_run_tracker_contract.py
```

Expected:
- FAIL because DB persistence function is missing

**Step 3: Write minimal implementation**

Add a small persistence interface in `db/repositories.py`, for example:
- `EvalRunRepository.record_run(...)`

Responsibilities:
- insert one `EvalRunModel`
- insert metric rows
- insert slice rows
- commit once

Update `eval/run_tracker.py` to call that repository with normalized rows.

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest -q tests/eval/test_eval_run_tracker_contract.py
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add db/repositories.py eval/run_tracker.py tests/eval/test_eval_run_tracker_contract.py
git commit -m "feat: persist eval run summaries"
```

### Task 4: Wire run tracking into retrieval, RAGAS, and workflow evals

**Files:**
- Modify: `eval/retrieval_eval.py`
- Modify: `eval/ragas_eval.py`
- Modify: `eval/workflow_eval.py`
- Test: `tests/eval/test_retrieval_regression.py`
- Test: `tests/eval/test_workflow_regression.py`
- Create: `tests/eval/test_ragas_tracking_contract.py`

**Step 1: Write the failing tests**

Add tests asserting:
- retrieval eval passes saved snapshot metadata into the tracker after `--save`
- workflow eval passes saved snapshot metadata into the tracker after `--save`
- RAGAS eval passes saved snapshot metadata into the tracker after `--save`
- no tracking is attempted when save is not requested, unless you intentionally decide to track unsaved runs

Use monkeypatches/fakes; do not call real DB or LLMs.

**Step 2: Run tests to verify they fail**

Run:
```bash
.venv/bin/python -m pytest -q tests/eval/test_retrieval_regression.py tests/eval/test_workflow_regression.py tests/eval/test_ragas_tracking_contract.py
```

Expected:
- FAIL in the new tracking expectations

**Step 3: Write minimal implementation**

In each eval script:
- after snapshot save
- collect:
  - `run_id`
  - `eval_type`
  - `mode`
  - `split`
  - `sample_count`
  - `git_sha`
  - `dataset_path`
  - `snapshot_path`
- call shared tracker persistence

Keep snapshot save first, tracking second.

**Step 4: Run tests to verify they pass**

Run:
```bash
.venv/bin/python -m pytest -q tests/eval/test_retrieval_regression.py tests/eval/test_workflow_regression.py tests/eval/test_ragas_tracking_contract.py
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add eval/retrieval_eval.py eval/ragas_eval.py eval/workflow_eval.py tests/eval/test_retrieval_regression.py tests/eval/test_workflow_regression.py tests/eval/test_ragas_tracking_contract.py
git commit -m "feat: track eval runs in benchmark history"
```

### Task 5: Provision Grafana PostgreSQL datasource for benchmark queries

**Files:**
- Modify: `docker-compose.yml`
- Modify: `grafana/provisioning/datasources/datasource.yml`
- Test: `tests/smoke/test_container_cache_contract.py`
- Create: `tests/smoke/test_grafana_benchmark_datasource_contract.py`

**Step 1: Write the failing test**

Create a test asserting:
- Grafana provisioning includes a PostgreSQL datasource
- the datasource targets the project Postgres service
- existing Prometheus datasource remains present

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest -q tests/smoke/test_grafana_benchmark_datasource_contract.py
```

Expected:
- FAIL because PostgreSQL datasource is not provisioned yet

**Step 3: Write minimal implementation**

Add a provisioned Grafana Postgres datasource using the existing DB service connection.

Likely files:
- `grafana/provisioning/datasources/datasource.yml`
- maybe `docker-compose.yml` env if Grafana needs DB credentials injected explicitly

Do not change existing Prometheus datasource behavior.

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest -q tests/smoke/test_grafana_benchmark_datasource_contract.py
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add docker-compose.yml grafana/provisioning/datasources/datasource.yml tests/smoke/test_grafana_benchmark_datasource_contract.py
git commit -m "feat: provision grafana postgres datasource for benchmark runs"
```

### Task 6: Add Grafana benchmark panels

**Files:**
- Modify: `grafana/dashboards/medigenius-dashboard.json`
- Test: `tests/smoke/test_grafana_dashboard_contract.py`

**Step 1: Write the failing test**

Add/update dashboard contract tests to assert new benchmark panels exist, for example:
- latest retrieval metrics
- latest RAGAS metrics
- workflow trend panel
- run inventory table

Do not overfit to pixel positions; assert titles and SQL datasource usage.

**Step 2: Run test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest -q tests/smoke/test_grafana_dashboard_contract.py
```

Expected:
- FAIL because the benchmark panels are missing

**Step 3: Write minimal implementation**

Extend the dashboard JSON with a new benchmark section backed by the PostgreSQL datasource.

Recommended panels:
- stat: latest `recall@5`
- stat: latest `mrr@5`
- stat: latest `faithfulness`
- stat: latest `context_precision`
- timeseries: retrieval metrics by run over time
- timeseries: RAGAS metrics by run over time
- table: recent benchmark runs

Keep existing live runtime panels intact.

**Step 4: Run test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest -q tests/smoke/test_grafana_dashboard_contract.py
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add grafana/dashboards/medigenius-dashboard.json tests/smoke/test_grafana_dashboard_contract.py
git commit -m "feat: add benchmark run panels to grafana"
```

### Task 7: Document the new benchmark observability contract

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`
- Optionally create: `docs/decisions/ADR-<date>-benchmark-run-observability.md`

**Step 1: Write the failing doc check**

If there is a suitable smoke/doc contract test, extend it. If not, keep this as a documentation-only task.

**Step 2: Update docs**

Document:
- benchmark runs now persist to Postgres
- Grafana uses:
  - Prometheus for live runtime proxies
  - PostgreSQL for benchmark history
- snapshots remain the detailed artifact of record

If the architecture feels like a lasting pattern, add a short ADR.

**Step 3: Verify docs references**

Run:
```bash
rg -n "benchmark run|PostgreSQL datasource|snapshot_path|Prometheus for live" docs/architecture docs/changes docs/decisions
```

Expected:
- updated docs references appear

**Step 4: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md docs/decisions
git commit -m "docs: record benchmark observability architecture"
```

### Task 8: End-to-end verification

**Files:**
- No code changes unless verification reveals a bug

**Step 1: Run targeted tests**

Run:
```bash
.venv/bin/python -m pytest -q \
  tests/eval/test_eval_run_tracker_contract.py \
  tests/eval/test_retrieval_regression.py \
  tests/eval/test_workflow_regression.py \
  tests/eval/test_ragas_tracking_contract.py \
  tests/smoke/test_eval_run_schema_contract.py \
  tests/smoke/test_grafana_benchmark_datasource_contract.py \
  tests/smoke/test_grafana_dashboard_contract.py \
  tests/smoke/test_container_cache_contract.py
```

Expected:
- PASS

**Step 2: Run compile checks**

Run:
```bash
python3 -m py_compile \
  eval/run_tracker.py \
  eval/retrieval_eval.py \
  eval/ragas_eval.py \
  eval/workflow_eval.py \
  db/repositories.py
```

Expected:
- no output

**Step 3: Run a live smoke path**

Run:
```bash
docker-compose up -d db redis app eval prometheus grafana
docker-compose exec -T eval python3 -m eval.retrieval_eval --mode pipeline --split test --save benchmark_observability_smoke
docker-compose exec -T eval python3 -m eval.ragas_eval --split test --judge-model gpt-4o-mini --save benchmark_observability_smoke_ragas
```

Then verify:
- snapshot files exist
- benchmark rows exist in Postgres
- Grafana dashboard shows the benchmark run

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: add benchmark run observability in grafana"
```
