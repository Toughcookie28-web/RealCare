# ADR-0003: Track Benchmark Runs in PostgreSQL, Not Prometheus Labels

## Status
Accepted

## Date
2026-03-16

## Context
MediGenius already saves eval snapshots as JSON artifacts, and Grafana already exists for live runtime metrics from Prometheus. That was enough to see live latency and errors, but not enough to answer benchmark questions reliably over time:
- which code path produced a given retrieval/RAGAS/workflow result
- how metrics changed across saved runs
- which snapshot file and git SHA belong to a run

Using Prometheus alone for benchmark history would push experiment metadata into labels, which is a poor fit for durable run tracking and would blur benchmark truth with live runtime proxies.

## Decision
We will keep JSON snapshots as the artifact of record and add PostgreSQL-backed benchmark run tracking:
- `eval_runs`
- `eval_run_metrics`
- `eval_run_slices`

Saved retrieval, RAGAS, and workflow evals now:
1. save the snapshot to disk
2. normalize the report through `eval/run_tracker.py`
3. persist a compact run summary into PostgreSQL

Grafana uses:
- Prometheus for live runtime proxy panels
- PostgreSQL for benchmark history panels

## Consequences
### Positive
- Benchmark runs become queryable and comparable over time.
- Grafana can show benchmark trends without abusing Prometheus labels.
- Snapshots remain the detailed artifact while SQL stores only compact summaries.
- Future analysis no longer depends on remembering which code/config produced a run.

### Negative
- Eval saves now depend on an additional schema and repository path.
- Grafana provisioning is slightly more complex because it uses two datasources.
- Historical runs are not backfilled in the first version.

## Guardrails
- Eval result computation must remain independent from run tracking persistence.
- Snapshot save happens before DB tracking.
- Tracking failure must not invalidate the main eval run.
- Live user traffic proxies in Prometheus must not be presented as benchmark truth.
