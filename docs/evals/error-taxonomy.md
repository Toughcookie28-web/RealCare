# Eval Error Taxonomy

This taxonomy is the reviewed vocabulary for RAG and eval failures in the current stabilization phase.
It exists to keep parser/runtime failures, ingest/index failures, retrieval failures, workflow failures,
cache failures, dataset-review failures, judge/generation failures, and CI gate failures separated.

## Category families

### Parser and runtime
- `dependency_or_import_failure`
  - Required Python or native runtime dependency is missing.
- `parser_artifact_permission_failure`
  - Parser or OCR model artifacts tried to write into a non-writable path.
- `parser_model_download_failure`
  - Runtime artifact download failed because of network, path, or cache setup.

### Schema and persistence
- `schema_mismatch`
  - Live database schema does not match the reviewed Alembic/vector contract.
- `persistence_contract_failure`
  - Required persistent storage is unavailable and the request cannot proceed safely.
- `runtime_fallback_masked_failure`
  - A live failure was previously hidden by in-memory or degraded fallback behavior.

### Ingest and index
- `empty_or_partial_index`
  - Ingest completed incompletely or the index no longer contains the expected corpus.
- `ingest_stage_count_mismatch`
  - Parse, chunk, embed, and index counts do not line up with the reviewed ingest contract.

### Retrieval
- `retrieval_miss`
  - A gold chunk or acceptable gold context was not retrieved at the required rank.
- `wrong_chunk_priority`
  - The correct chunk was retrieved, but ranked behind a clearly worse chunk.
- `table_retrieval_miss`
  - A table-grounded question did not retrieve table content when required.

### Workflow
- `workflow_route_mismatch`
  - The workflow routed the request through the wrong high-level path.
- `workflow_answer_contract_failure`
  - Final answer violated the reviewed keyword/source/shape contract for the case.

### Cache
- `stale_semantic_cache_hit`
  - Semantic cache returned an answer from an older prompt/retrieval/corpus contract.
- `cache_masked_retry_failure`
  - Retry or reflection path reused cached output instead of exercising a fresh retrieval/generation path.

### Eval dataset and review
- `eval_dataset_schema_failure`
  - Eval case file does not match the expected reviewed schema.
- `unreviewed_failure_mode`
  - A new failure mode was observed but has not been reviewed and added to the taxonomy.

### Offline generation and judge
- `generator_runtime_failure`
  - Tier 3 offline generation failed before producing a dataset artifact.
- `judge_validation_gap`
  - Dataset or judge approval process did not provide enough reviewed evidence to trust the outcome.

### Gate contract
- `gate_contract_failure`
  - CI or required local regression path is invoking unstable or live integration workloads instead of the frozen suite.

## Review rules
- Retrieval regression must be explainable from frozen fixture cases under `tests/fixtures/rag/`.
- Workflow regression must be explainable from frozen workflow cases under `tests/fixtures/rag/`.
- Tier 3 generation is an offline utility and should be reviewed separately from the required regression gate.
- New failure modes must be added here and represented in `eval/review_samples/rag_trace_review.jsonl` before they are used in decision-making.
