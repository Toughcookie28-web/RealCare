# Implementation Log

Use this file for meaningful project changes only.

For each entry, capture:
- what changed
- why it changed
- tradeoff introduced
- what must remain true afterward

---

## 2026-03-16 — Fix semantic cache Redis namespace filter boot regression

What changed:
- Added `SemanticCache._redis_namespace_filter()` in [tools/cache.py](/home/tough/medical_chatbot/MediGenius/tools/cache.py) and routed the RediSearch query builder through it instead of embedding the colon escape directly inside an f-string expression.
- Added a cache contract test covering namespace escaping in [test_semantic_cache_contract.py](/home/tough/medical_chatbot/MediGenius/tests/rag/test_semantic_cache_contract.py).
- Updated [run_final_pipeline.py](/home/tough/medical_chatbot/MediGenius/scripts/run_final_pipeline.py) so the readiness poller retries through transient `ConnectionResetError` / socket-level startup failures instead of aborting the whole run immediately.
- Updated [db/repositories.py](/home/tough/medical_chatbot/MediGenius/db/repositories.py) so chat/session/summary/fact create paths stamp required timestamps explicitly before commit.
- Added [test_chat_repository_timestamp_contract.py](/home/tough/medical_chatbot/MediGenius/tests/db/test_chat_repository_timestamp_contract.py) to lock those timestamp requirements.

Why:
- The app container was failing to boot because the old Redis namespace filter used a backslash escape inside an f-string expression, which Gunicorn hit at import time while loading the app.
- The final pipeline runner timed out on `/health/ready` only because the app process had already crashed.

Tradeoff:
- None materially; the runtime behavior stays the same and the Redis query string is now built in a Python-version-safe way.

Must remain true:
- Semantic cache namespace filters must continue to escape `:` for Redis TAG queries.
- Cache-related import-time code must remain syntactically valid under the Python version used in the app image.
- Final pipeline readiness checks must tolerate transient startup connection resets while the app is still settling.
- ChatRepository must not rely on missing DB-side timestamp defaults for `sessions`, `messages`, `conversation_summaries`, or `user_facts`.

## 2026-03-16 — Final local pipeline runner + persistent RAGAS snapshot mount

What changed:
- Updated [docker-compose.yml](/home/tough/medical_chatbot/MediGenius/docker-compose.yml) so the long-lived `eval` container now bind-mounts [eval/ragas_snapshots](/home/tough/medical_chatbot/MediGenius/eval/ragas_snapshots) onto `/app/eval/ragas_snapshots`.
- Added [run_final_pipeline.py](/home/tough/medical_chatbot/MediGenius/scripts/run_final_pipeline.py), an operator script that:
  - brings up `db`, `redis`, `app`, `eval`, Prometheus, and Grafana
  - optionally rebuilds app/eval images
  - waits for the app and Grafana to become reachable
  - opens the frontend and Grafana dashboard
  - replays a small HTTP workload through `/api/chat` so Grafana sees real app traffic
  - saves a pipeline retrieval snapshot and a full RAGAS snapshot
- Extended the compose contract test so snapshot mounts now include `eval/ragas_snapshots`.

Why:
- RAGAS snapshots were being "saved" inside the eval container and then disappearing unless they were manually copied out.
- Running `eval` sidecar code alone does not populate Grafana meaningfully because Prometheus scrapes the `app` container, not the sidecar.
- A single operator command is safer than hand-running build, app warmup, workload replay, retrieval eval, and RAGAS eval out of order.

Tradeoff:
- The helper script is intentionally local-ops oriented and depends on `docker-compose`, an available browser opener, and the local FastAPI/Grafana ports.
- Grafana visibility comes from replayed HTTP traffic, not from the eval container process itself, so dashboard numbers and snapshot numbers are related but not identical measurements.

Must remain true:
- `eval/ragas_snapshots` must stay host-mounted for any containerized RAGAS save path to be trustworthy.
- Final-run Grafana traffic must hit the HTTP app path (`/api/chat`), not only in-process eval code.
- The helper script must keep saving retrieval pipeline snapshots under `eval/retrieval_snapshots/pipeline/` and full RAGAS snapshots under `eval/ragas_snapshots/`.

## 2026-03-17 — Redis working memory overlay + async live judge proxies

What changed:
- Added Redis working-memory primitives in [tools/working_memory.py](/home/tough/medical_chatbot/MediGenius/tools/working_memory.py) and [tools/working_memory_service.py](/home/tough/medical_chatbot/MediGenius/tools/working_memory_service.py).
- Added explicit settings for recent-turn TTL, core-state TTL, recent-turn max length, live judge enablement, and live judge sampling ratio in [core/settings.py](/home/tough/medical_chatbot/MediGenius/core/settings.py) and [.env.example](/home/tough/medical_chatbot/MediGenius/.env.example).
- Extended [tools/redis_client.py](/home/tough/medical_chatbot/MediGenius/tools/redis_client.py) with JSON helpers and graceful in-memory fallback for working-memory payloads.
- Added [observability/live_judge.py](/home/tough/medical_chatbot/MediGenius/observability/live_judge.py) plus new Prometheus metrics in [observability/metrics.py](/home/tough/medical_chatbot/MediGenius/observability/metrics.py) for live answer relevance, groundedness, context precision proxy, context coverage proxy, judge latency, and judge failures.
- Added [api/chat_runtime.py](/home/tough/medical_chatbot/MediGenius/api/chat_runtime.py) and wired [api/routes/chat.py](/home/tough/medical_chatbot/MediGenius/api/routes/chat.py) to:
  - load Redis working memory before workflow execution
  - append recent turns after the response
  - promote stable facts and summary-derived context into Redis `core_state`
  - run live judge work post-response
- Updated [core/workflow_service.py](/home/tough/medical_chatbot/MediGenius/core/workflow_service.py) so stream-final events carry internal post-response data without exposing that internal payload to the frontend event body.

Why:
- Important medical state such as allergies, conditions, and session goal should not disappear when older turns leave the active window.
- The frontend needed immediate Grafana visibility from real user traffic, but benchmark snapshots remain offline eval artifacts.
- The user wanted this iteration to avoid changing the main database schema.

Tradeoff:
- Live judge scores are observability proxies, not benchmark-truth recall/precision or offline RAGAS results.
- Redis is now part of the hot-path memory overlay, though the app still degrades to PostgreSQL-backed memory when Redis is unavailable.

Must remain true:
- Redis working memory must remain a safe overlay, not a required source of truth.
- Working-memory promotion must stay conservative and versioned.
- Post-response live judge failures must never fail the user response.
- Grafana panels built on live judge metrics must continue to describe those scores as proxies rather than benchmark metrics.

## 2026-03-15 — Retrieval eval: graded relevance scoring + pipeline-aware mode

What changed:
- Replaced binary fuzzy-match scoring with graded chunk-ID-only relevance (3=required, 2=supporting, 1=hard_negative, 0=irrelevant).
- Metrics now use grade thresholds: recall/MRR/hit_rate require grade 3, precision/MAP require grade 2+, nDCG uses all grades.
- Removed all fuzzy text matching (`_match_score`, `_normalize_text`, `_tokenize`, `_binary_relevance_for_doc`, `match_threshold` parameter).
- Added `section_diversity@k` metric (unique sections in top-k / k).
- Added `--mode pipeline` that runs the full RAG pipeline path: LLM query rewriter → hybrid search → step-back merge → metadata boost → BM25 rerank.
- "all" mode now excludes pipeline (since it makes LLM calls).
- Evidence schema supports `evidence.required`, `evidence.supporting`, `evidence.hard_negative_candidates` with fallback to flat `ground_truth_chunk_ids`.

Why:
- Binary scoring could not distinguish essential vs. supportive context, making precision/recall metrics misleading.
- Fuzzy text matching was fragile and did not align with chunk-ID-based ground truth labeling.
- Pipeline mode needed to measure whether the query rewriter actually improves retrieval quality.

Tradeoff:
- Pipeline mode requires LLM API access and is slower/costlier — excluded from "all" runs.
- Graded scoring requires labeled evidence tiers in eval data; rows with only `ground_truth_chunk_ids` are treated as all-required (grade 3).

Must remain true:
- `_extract_evidence_ids()` must return `(required, supporting, hard_negative)` sets from either `evidence` dict or flat `ground_truth_chunk_ids`.
- Pipeline mode must mirror the production retrieval path (rewriter → hybrid → step-back → metadata boost → rerank).
- Tests in `test_retrieval_regression.py` must pass with mocked dependencies (no real LLM/DB calls).

---

## 2026-03-15 — Curated Phase A k-NN routing seeds

What changed:
- Added `data/route_seeds_raw.json` with 10 medically realistic seed queries for each fallback route: `vector`, `chitchat`, `web`, `memory`, `literature`, and `future_clinical_db`.
- Added `tests/rag/test_route_seed_contract.py` to lock route coverage, per-route seed counts, and within-route uniqueness.
- Added design and implementation notes under `docs/plans/2026-03-15-route-seed-curation-*.md`.

Why:
- The k-NN fallback router depends on semantically clean seed coverage, and sloppy or mixed-intent seed examples would blur route boundaries.
- `future_clinical_db` needed seed coverage even before the backing route exists so the fallback space does not silently omit it later.

Tradeoff:
- The seed set is intentionally conservative and precision-oriented, which helps routing accuracy but may need later expansion if real user phrasing is broader.
- `data/route_seeds.json` still must be regenerated whenever the raw seed file changes.

Must remain true:
- Seed examples should stay single-intent and route-specific.
- `web` seeds must encode recency/currentness, while `literature` seeds must explicitly ask for studies, papers, or reviews.
- `future_clinical_db` seeds should reflect chart/EHR requests, not conversation memory.

---

## 2026-03-15 — Pipeline retrieval snapshots separated from standard retrieval snapshots

What changed:
- Added dedicated pipeline snapshot storage under `eval/retrieval_snapshots/pipeline/`.
- Updated `eval/retrieval_eval.py` so `--mode pipeline --save NAME` writes to the pipeline subfolder instead of the flat snapshot directory.
- Kept backward-compatible loading so `--compare` in pipeline mode still finds older flat pipeline snapshots if they already exist.
- Added a regression test covering pipeline snapshot save/load behavior.

Why:
- Pipeline-mode retrieval measures the rewriter + step-back + metadata boost + rerank path, which is materially different from the non-LLM retrieval-only modes.
- Keeping those snapshots separate reduces comparison mistakes and makes later A/B review easier.

Tradeoff:
- Snapshot lookup is now slightly more mode-aware, so the save/load path has a little more logic.
- Existing flat snapshots remain readable, but new pipeline snapshots should live only in the dedicated subfolder.

Must remain true:
- Non-pipeline retrieval snapshots keep using `eval/retrieval_snapshots/`.
- Pipeline snapshots save to `eval/retrieval_snapshots/pipeline/`.
- Pipeline compare mode must continue to load older flat snapshots as fallback until they are migrated or discarded.

---

## 2026-03-14 — Pipeline upgrade Phase B: metadata-powered retrieval + executor optimization

What changed:
- Retriever applies soft metadata boost (+0.15) to chunks whose `keywords`/`medical_entities` match slot values, combined with BM25 scores before final ranking.
- Post-retrieval slot validation computes `slot_coverage` and adjusts `retrieval_confidence` by 0.7x when coverage < 0.3 and intent is specific (dosage_lookup, drug_comparison, side_effects, mechanism_of_action, differential_diagnosis).
- Executor prompt optimized with intent-aware formatting instructions (6 intent types), `query_context` injection, and `session_intent` awareness.
- Context assembly prepends `context_summary` metadata headers and sorts intro-position chunks first.
- Answer generation appends deduplicated section/page citations as a Sources block.

Why:
- Uses structured query understanding from Phase A (slots, intents) and optional chunk enrichment metadata to improve retrieval precision, answer formatting, and source transparency.

Tradeoff:
- Full benefit requires `ENRICH_CHUNKS_WITH_LLM=true` during ingest so chunks carry `keywords`, `medical_entities`, `context_summary`. Without enrichment, all features degrade gracefully to current behavior.
- Citation block appended to generation increases response length slightly.
- Cache namespace unchanged — prompt template V2 is used at generation time but V1 template is still used for cache key calculation, so cache invalidation is not triggered by this change alone.

Must remain true:
- All Phase B features are no-ops when enrichment metadata is absent.
- `slot_coverage` is observable in state for debugging and routing decisions.
- Citations are appended to generation text AND stored structurally in `state['citations']`.
- Intro prioritization must not break adjacent expansion or chunk deduplication.

---

## 2026-03-14 — Pipeline upgrade Phase A: unified rewriter, dispatcher, k-NN, memory route, adjacent expansion

What changed:
- Expanded query rewriter to output route, session_intent, turn_intent, slots, query_context in a single LLM call alongside optimized_query and stepback_query.
- Converted planner agent from LLM-based routing to zero-LLM dispatcher that reads route from state.
- Added k-NN fallback router using pre-embedded seed queries with cosine similarity majority vote. Seeds loaded from `data/route_seeds.json`.
- Added memory route: when user asks about previous conversation, executor generates from history/summary/facts without vector retrieval.
- Added adjacent chunk expansion in executor: fetches ±1 neighbor chunks by chunk_index within the same section during context assembly, controlled by token budget.

Why:
- Eliminates a redundant LLM call (planner) by consolidating query understanding into the rewriter.
- k-NN fallback is more robust than keyword matching for route classification on LLM failure.
- Memory route prevents irrelevant vector search results when users ask about conversation history.
- Adjacent expansion provides surrounding context that may complete partial answers at chunk boundaries.

Tradeoff:
- Rewriter prompt is larger and outputs more fields, slightly increasing JSON parse failure risk (mitigated by existing graceful degradation).
- k-NN fallback requires maintaining seed queries and re-embedding when seeds change.
- Adjacent expansion adds one DB query per retrieved chunk in the executor, adding minor latency.

Must stay true:
- Rewriter failure must never block the pipeline: all tiers degrade to using original question with route="" which triggers k-NN fallback.
- k-NN returns "vector" as safe default when seeds are missing or confidence is low.
- Adjacent expansion is opt-in: skipped when chunk_index is missing or vector_repo is unavailable.
- Memory route must not hallucinate: if no history exists, say so honestly.

---

## 2026-03-13 — Query transformation: medical normalization + conditional step-back

What changed:
- Replaced simple text rewriter with structured JSON output using Pydantic `RewriteResult` model.
- Added medical term normalization in the rewriter prompt: abbreviation expansion, synonym addition, conversational reference resolution.
- Added conditional step-back query generation: LLM produces a broader principle-level query for mechanism/comparison/dosage questions, empty string for simple lookups.
- Added tiered graceful degradation: Pydantic validation → partial JSON extraction → raw text as query → original question fallback. No LLM retry on failure.
- Added `_merge_and_deduplicate()` in retriever: when step-back query is present, retrieves supplementary docs (k=10), merges with primary (k=20), deduplicates by `chunk_id`, then BM25 reranks the combined pool against the specific query.
- Added `stepback_query` field to `AgentStateV2`.

Why:
- Retrieval recall was limited by vocabulary gap between user queries and textbook language. Users say "heart attack" but the textbook says "myocardial infarction".
- Complex questions (mechanisms, comparisons) benefit from broader context that a step-back query retrieves.

Tradeoff:
- Step-back search adds one extra embedding call + hybrid search (k=10) per query when step-back is generated. The LLM call cost is zero (rewriter call already exists).
- Merge + dedup adds minor compute but BM25 reranking against the specific query ensures relevance is preserved.

Must stay true:
- Query transformation must never block the pipeline: all failures degrade to using the original question.
- BM25 reranking always uses `optimized_query` (not step-back) so specific relevance wins.
- `stepback_query` is empty string by default and the retriever skips the extra search when empty.

---

## 2026-03-13 — Chunk enrichment, table linearization, and retrieval improvements

What changed:
- Added BM25-based reranking in the retriever agent and `tsvector` + GIN sparse search in the vector repository (replacing ILIKE).
- Added schema-aware table linearization: table embedding text now uses `Columns = [...]; Row = [...]` format instead of raw markdown pipes.
- Added rich chunk metadata: chunk position (intro/body/conclusion), cross-references (structured `[{type, normalized}]`), table structure metadata (columns, row count, size tier).
- Added optional LLM-based chunk enrichment behind `ENRICH_CHUNKS_WITH_LLM` (context summary, keywords, medical entities via Groq).
- Externalized chunk sizing via `ChunkConfig` wired to Settings for A/B testing.
- Fixed `embed_documents_batch` to respect provider order instead of always trying OpenAI first.

Why:
- Retrieval quality was limited by binary keyword matching, flat embedding text for tables, and minimal chunk metadata for filtering or reranking.
- The chunking A/B comparison phase requires externalized chunk configuration and richer metadata to evaluate different strategies.

Tradeoff:
- LLM enrichment adds latency and cost to reindex; it is disabled by default and opt-in only.
- Schema-aware linearization changes embedding text format, which means a reindex is required after upgrading.
- BM25 reranking adds a small amount of compute per retrieval call but should improve ranking quality.

Must stay true:
- `page_content` (what the LLM reads at generation time) stays unchanged; enrichment only affects metadata and `embedding_text`.
- LLM enrichment must remain behind `ENRICH_CHUNKS_WITH_LLM=true` and not run by default.
- Chunk sizing must stay configurable through Settings for A/B testing.
- The `tsvector` migration (0004) must be applied before sparse search works.

## 2026-03-09 — Stabilized RAG, eval, and runtime contracts

What changed:
- Added a RAG runtime preflight contract and CI entrypoint.
- Removed silent persistence fallbacks from RAG/eval paths and removed ingest side effects from API startup.
- Unified install/runtime behavior across Docker, CI, Render, and compose-mounted development paths.
- Made Alembic the only schema authority and locked the document chunk vector contract to `vector(768)`.
- Split ingest into explicit parse, chunk, embed, and index stages, with an [IngestReport](/home/tough/medical_chatbot/MediGenius/tools/ingest_report.py) returned by the reindex path.
- Changed semantic cache behavior so it is explicitly versioned, explicitly disableable, disabled by default in development/local/test/eval-style environments, and disabled by default in workflow eval.

Why:
- The repository was repeatedly failing in containerized eval/reindex work because schema state, dependency installation, cache behavior, startup behavior, and ingest behavior were not deterministic enough to debug confidently.
- The current stabilization phase is intended to make RAG and eval behavior inspectable before any broader architecture redesign.

Tradeoff:
- Operators must now run migrations and reindexing more explicitly; the app no longer hides those steps at startup.
- Semantic cache is less aggressive by default in non-production environments, which reduces accidental speedups during debugging but improves trustworthiness.
- Existing document embeddings must be regenerated after the vector-dimension alignment migration.

Must stay true:
- [db/schema_contract.py](/home/tough/medical_chatbot/MediGenius/db/schema_contract.py) remains the single source of truth for the document chunk vector contract.
- Alembic remains the only supported schema mutator; runtime must not reintroduce `create_all()` or extension-creation side effects.
- Ingest must remain explicitly separated into parse, chunk, embed, and index stages.
- Workflow eval must keep semantic cache disabled by default unless explicitly opted in.
- Containerized RAG/eval work must continue using the same install/runtime contract documented in [README.md](/home/tough/medical_chatbot/MediGenius/README.md).

## 2026-03-09 — Added frozen eval gate and reviewed failure taxonomy

What changed:
- Added a reviewed eval failure taxonomy in [docs/evals/error-taxonomy.md](/home/tough/medical_chatbot/MediGenius/docs/evals/error-taxonomy.md) plus reviewed trace samples in [rag_trace_review.jsonl](/home/tough/medical_chatbot/MediGenius/eval/review_samples/rag_trace_review.jsonl).
- Added frozen retrieval/workflow regression fixtures and tests under [tests/eval/](/home/tough/medical_chatbot/MediGenius/tests/eval) and [tests/fixtures/rag/](/home/tough/medical_chatbot/MediGenius/tests/fixtures/rag).
- Updated [eval/retrieval_eval.py](/home/tough/medical_chatbot/MediGenius/eval/retrieval_eval.py) and [eval/workflow_eval.py](/home/tough/medical_chatbot/MediGenius/eval/workflow_eval.py) so they can run against injected frozen repos/runners instead of only the live runtime path.
- Demoted [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) to an offline utility and removed live preflight from the required CI eval gate.

Why:
- Eval trustworthiness was still too dependent on live runtime state, which mixed parser failures, runtime setup failures, retrieval regressions, workflow regressions, and offline dataset generation into one unstable path.
- The current phase needs a small, explicit regression contract that can fail for reviewed reasons only.

Tradeoff:
- The frozen regression gate is narrower than the full live integration path and does not prove that live PDF parsing, model downloads, or container runtime permissions are healthy.
- Tier 3 dataset generation now sits outside the required CI gate and must be reviewed as an offline workflow instead of a blocking regression signal.

Must stay true:
- `tests/eval` and `tests/fixtures/rag` remain frozen and independent from live parser, DB, network, and LLM state.
- `scripts/preflight_rag.py` remains a runtime diagnostic, not a required eval gate.
- Tier 3 generation remains an offline utility and must not be reintroduced as a required CI regression step.
- New failure modes must be added to the reviewed taxonomy and represented in reviewed trace samples before they are relied on for evaluation decisions.

## 2026-03-09 — Authored the next validation and chunking comparison plan

What changed:
- Added [docs/plans/2026-03-09-runtime-eval-chunking-phase2.md](/home/tough/medical_chatbot/MediGenius/docs/plans/2026-03-09-runtime-eval-chunking-phase2.md).
- Updated [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) to mark stabilization complete and set the next phase priorities to live Docker/runtime validation, frozen eval expansion, and chunking A/B preparation.

Why:
- The stabilization phase is complete, but the repo still needs a concrete execution order for validating the real container/runtime path and comparing chunking strategies without mixing variables.

Tradeoff:
- This is a planning artifact only. It does not yet improve runtime correctness or eval coverage until the tasks are executed.

Must stay true:
- The next phase keeps blocking CI deterministic and code-based.
- Live Docker/runtime validation and chunking A/B remain explicit tracks, not hidden side effects of the existing gate.
- The primary chunking A/B comparison must change only chunking, while parser, embeddings, retrieval configuration, and cache policy stay fixed.

## 2026-03-09 — Added Batch 1 live runtime validation contracts

What changed:
- Added [core/live_runtime_contract.py](/home/tough/medical_chatbot/MediGenius/core/live_runtime_contract.py), [scripts/run_live_runtime_smoke.py](/home/tough/medical_chatbot/MediGenius/scripts/run_live_runtime_smoke.py), and [scripts/run_live_rag_shadow.py](/home/tough/medical_chatbot/MediGenius/scripts/run_live_rag_shadow.py).
- Added smoke/contract coverage for the new runtime and shadow runners in [tests/smoke/test_live_runtime_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_live_runtime_contract.py), [tests/smoke/test_live_runtime_smoke_runner.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_live_runtime_smoke_runner.py), and [tests/smoke/test_live_rag_shadow_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_live_rag_shadow_contract.py).
- Added [docs/evals/live-runtime-validation.md](/home/tough/medical_chatbot/MediGenius/docs/evals/live-runtime-validation.md), updated [README.md](/home/tough/medical_chatbot/MediGenius/README.md), and wired non-blocking live runtime jobs into [.github/workflows/eval.yml](/home/tough/medical_chatbot/MediGenius/.github/workflows/eval.yml).
- Tightened the Alembic part of the live smoke path so the image includes `alembic/` and `alembic.ini`, and the smoke runner uses `python3 -m alembic -c /app/alembic.ini upgrade head` explicitly inside the container.

Why:
- The next phase needs an explicit boundary between what can be trusted in the frozen regression suite and what still has to be validated in the real Docker/WSL/runtime path.

Tradeoff:
- These additions define and automate the command paths, but they do not prove live Docker success in this environment because container execution still must be run by the operator.

Must stay true:
- The minimal live runtime smoke path remains the local/operator blocking path for Batch 1.
- The extended live shadow path remains non-blocking until it proves stable.
- The required CI gate remains the frozen code-based suite, while live runtime jobs publish evidence without being trusted as merge blockers yet.

## 2026-03-09 — Fixed Docling artifact cache root for compose-based live shadow runs

What changed:
- Updated [docker-compose.yml](/home/tough/medical_chatbot/MediGenius/docker-compose.yml) so containerized `app`, `eval`, and `ingest` services always use `/app/.cache/embeddings` as the in-container cache/artifact root instead of interpolating `EMBEDDING_CACHE_DIR` from `.env`.
- Updated [tests/smoke/test_container_cache_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_container_cache_contract.py), [README.md](/home/tough/medical_chatbot/MediGenius/README.md), [docs/evals/live-runtime-validation.md](/home/tough/medical_chatbot/MediGenius/docs/evals/live-runtime-validation.md), and [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) to document and lock the compose cache-root contract.
- Clarified in [.env.example](/home/tough/medical_chatbot/MediGenius/.env.example) that `/tmp/medigenius-embeddings` remains the host-local default for non-container runs only.

Why:
- The live shadow run was still reading `EMBEDDING_CACHE_DIR=/tmp/medigenius-embeddings` from `.env`, which caused Docling to look for `model.safetensors` under `/tmp/.../docling_artifacts` instead of the bind-mounted `/app/.cache/embeddings/docling_artifacts` path used by the container runtime contract.

Tradeoff:
- Container jobs no longer allow `.env` to remap the embedding cache root. That reduces flexibility but removes a configuration footgun that was breaking Docling artifact provisioning.

Must stay true:
- Compose-based runtime and eval jobs always use `/app/.cache/embeddings` inside the container.
- Host-local Python runs may still use `/tmp/medigenius-embeddings`, but that host path must not leak into compose-managed container runs.

## 2026-03-09 — Added Docling runtime artifact bootstrap for live ingest

What changed:
- Added [ensure_docling_artifacts()](/home/tough/medical_chatbot/MediGenius/tools/embedding_bootstrap.py) to provision Docling layout/table models into the configured writable cache when `model.safetensors` is missing.
- Updated [tools/pdf_parser.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_parser.py) so Docling artifact bootstrap runs before `DocumentConverter()` initializes.
- Added [tests/smoke/test_docling_bootstrap_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_docling_bootstrap_contract.py) and updated [tests/smoke/test_container_cache_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_container_cache_contract.py) to lock the bootstrap behavior and parser call site.

Why:
- After the cache-root mismatch was fixed, live shadow runs still failed because the correct artifact directory existed but the Docling layout model file `model.safetensors` had never been provisioned there.

Tradeoff:
- The first Docling-based ingest on a fresh cache now performs an explicit model download step before parsing begins, which increases first-run latency but avoids repeated image rebuilds or oversized prewarmed images.

Must stay true:
- Compose-based ingest/eval runs keep using the writable mounted cache at `/app/.cache/embeddings`.
- Docling artifact bootstrap happens only when the required layout model is missing.

## 2026-03-09 — Normalized nested Docling layout snapshots into the runtime artifact root

What changed:
- Updated [tools/embedding_bootstrap.py](/home/tough/medical_chatbot/MediGenius/tools/embedding_bootstrap.py) so `ensure_docling_artifacts()` now surfaces nested layout snapshot files such as `docling-project--docling-layout-heron/model.safetensors` into the root `docling_artifacts/` directory after download.
- Extended [tests/smoke/test_docling_bootstrap_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_docling_bootstrap_contract.py) with a nested-snapshot regression case.

Why:
- On the live eval image, Docling downloads were landing under a nested Hugging Face snapshot directory, but the runtime layout loader was checking only `/app/.cache/embeddings/docling_artifacts/model.safetensors`.

Tradeoff:
- The bootstrap now contains a small normalization step that links or copies nested layout snapshot files into the runtime root. That adds a little complexity, but it is still much smaller than prewarming Docling artifacts into every image rebuild.

Must stay true:
- `docling_artifacts/model.safetensors` remains the runtime contract expected by the current Docling layout loader.
- Nested snapshot downloads are normalized automatically before parse starts when the root artifact is missing.

## 2026-03-09 — Fixed PictureItem markdown-export compatibility in the Docling parser loop

What changed:
- Updated [tools/pdf_parser.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_parser.py) so generic text extraction no longer calls `export_to_markdown()` on picture/figure items before the figure branch runs.
- Added [tests/smoke/test_pdf_parser_contract.py](/home/tough/medical_chatbot/MediGenius/tests/smoke/test_pdf_parser_contract.py) to lock the safe picture-item behavior and preserve non-picture markdown export behavior.

Why:
- The live Docling runtime was raising `TypeError: PictureItem.export_to_markdown() missing 1 required positional argument: 'doc'` because the parser’s generic fallback called `export_to_markdown()` on picture items before type-specific handling.

Tradeoff:
- Figure items now prefer the existing metadata-only path rather than trying to force a markdown export. That is intentionally conservative, but it preserves table/text extraction and avoids a parser crash.

Must stay true:
- Unsupported picture exports must not abort the parse loop.
- Table and non-picture markdown extraction behavior must remain intact.

## 2026-03-09 — Fixed Tier 3 eval dependency contract and added Step 1 corpus checkpointing

What changed:
- Added `rapidfuzz` to the `eval` optional dependency set in [pyproject.toml](/home/tough/medical_chatbot/MediGenius/pyproject.toml) so the eval image installs the string-distance dependency required by the current Tier 3 generation path.
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so Step 1 now checkpoints production chunks to [tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl) and prefers reusing that frozen corpus on reruns.
- Added an `auto/cache/db/pdf` source contract so Tier 3 generation can derive its corpus from the already indexed `document_chunks` table before reparsing the PDF.
- Added [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock the dependency and Step 1 reuse behavior.

Why:
- Tier 3 generation was reparsing the PDF every rerun and then failing in Step 2 because the eval dependency contract did not actually include `rapidfuzz`.
- Once the PDF had already been indexed successfully, reparsing the same corpus again was wasted runtime and made the offline utility much more painful to debug.

Tradeoff:
- Tier 3 now persists an intermediate chunk-corpus artifact under `eval/golden/v1`, which slightly mixes generator input artifacts with golden data files, but it keeps the artifact on a mounted path that survives compose-based reruns.
- `auto` mode prefers the cached corpus for repeatability, so operators must use `--source db` or `--source pdf` explicitly when they want to refresh the corpus after a new reindex.

Must stay true:
- The eval image keeps installing `rapidfuzz` anywhere Tier 3 generation is expected to run.
- Tier 3 reruns must be able to start from the saved Step 1 chunk corpus without reparsing the PDF.
- Reindex once, derive the corpus once, then reuse that frozen corpus for repeated Tier 3 generation runs unless the operator explicitly refreshes it.

## 2026-03-09 — Authored the Tier 3 backend migration plan

What changed:
- Added [docs/plans/2026-03-09-tier3-backend-migration.md](/home/tough/medical_chatbot/MediGenius/docs/plans/2026-03-09-tier3-backend-migration.md).
- Updated [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) to record that Tier 3 backend selection is still coupled to the shared app LLM settings and fallback path.

Why:
- The next Tier 3 phase needs a narrow, explicit plan for moving from a Groq-backed online path to a backend-neutral Groq or local/vLLM path without turning the generator into an HPC-only branch.

Tradeoff:
- This is a planning artifact only. It does not yet change Tier 3 behavior, checkpoint depth, or backend abstraction until the plan is executed.

Must stay true:
- Tier 3 backend work remains scoped to generator robustness, backend abstraction, resume safety, and artifact management.
- The migration must keep Groq, laptop-local, and future HPC/local-model runs under one contract instead of creating a one-off HPC path.

## 2026-03-09 — Fixed Tier 3 async embedding adapter compatibility for RAGAS 0.3.x

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so `_ProjectEmbeddingsAdapter` now implements `aembed_query()` and `aembed_documents()` in addition to the existing sync methods.
- Extended [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock the async adapter contract.

Why:
- The Tier 3 run against GPT-4o progressed into the RAGAS extractor stage and then degraded on `'_ProjectEmbeddingsAdapter' object has no attribute 'aembed_documents'`, which is a compatibility gap with the installed RAGAS 0.3.x pipeline.

Tradeoff:
- The async adapter methods currently wrap the existing synchronous local embedding calls with `asyncio.to_thread`, which keeps the implementation small and compatible but does not make local embeddings truly non-blocking.

Must stay true:
- Tier 3 must be able to use the project’s local embedding backend through both sync and async RAGAS code paths.
- This compatibility fix must not change the generator backend contract or reintroduce PDF reparsing on rerun.

## 2026-03-10 — Added a deterministic Tier 3 curated-subset builder

What changed:
- Added [scripts/build_tier3_chunk_subset.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_chunk_subset.py) to derive an eval-only curated subset from the frozen [tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl).
- Added [tests/eval/test_tier3_chunk_subset_builder.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_chunk_subset_builder.py) to lock the deterministic selection behavior.
- Generated [tier3_chunk_corpus.curated.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.curated.jsonl) as the first reusable curated subset for Tier 3 toy and medium runs.
- Updated [README.md](/home/tough/medical_chatbot/MediGenius/README.md) and [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) to document the new workflow.

Why:
- The raw frozen chunk corpus is useful, but feeding the first N rows directly into Tier 3 generation produced admin/front-matter questions, near-duplicates, and low-value output.
- We needed a repeatable way to improve eval-input quality without changing the production chunking or vector-store contract.

Tradeoff:
- The first pass is heuristic and intentionally conservative: it keeps `text_section` chunks only, drops known low-signal sections, and caps repeated local page/section chunks. That improves toy-run quality, but it also means table-driven Tier 3 cases are deferred to a later explicit pass.
- The refined pass is stricter still: it now prefers higher-signal medical sections (`Diagnosis`, `Treatment`, `Causes and symptoms`, `Prevention`, `Prognosis`) and caps chunks per local page window to reduce topic duplication. That produces a smaller curated subset, but it is more useful for toy and medium Tier 3 quality checks.

Must stay true:
- The curated-subset builder must remain deterministic for the same input corpus and flags.
- This builder is eval-only; it must not change production chunking, indexing, or retrieval behavior.
- Tier 3 scale-up runs should be able to reuse the curated subset directly instead of hand-editing chunk files between runs.

## 2026-03-10 — Hardened Tier 3 conversion with raw-row checkpoints and cross-topic filtering

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so raw RAGAS rows are checkpointed to [tier3_rag.raw.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.raw.jsonl) before schema conversion.
- Added a conservative conversion-time filter that skips obviously blended cross-topic rows when aligned pages are far apart and the underlying contexts have very low topic overlap.
- Extended [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock raw-row checkpointing, multi-context preservation for related rows, and rejection of far-apart low-overlap blends.
- Updated [README.md](/home/tough/medical_chatbot/MediGenius/README.md) and [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) to document the new Tier 3 review/debug workflow.

Why:
- The curated subset improved chunk quality, but Tier 3 was still producing bad blended questions such as unrelated disease/topic joins while also collapsing everything into `rag_simple`.
- We needed the raw generator output preserved for inspection and a narrow deterministic guard against the most obviously bad synthetic rows before the final JSONL is frozen for review.

Tradeoff:
- The new filter is intentionally conservative and only catches a subset of bad blends. It does not solve all Tier 3 quality issues, and it may still leave some weak `simple` rows for human review.
- Tier 3 now hard-fails when all raw rows are missing RAGAS evolution metadata. That is stricter than before, but it is better than silently accepting a low-trust all-`simple` dataset.

Must stay true:
- Tier 3 raw RAGAS rows remain inspectable after every run.
- Conversion-time filtering stays narrow and deterministic; it must not silently rewrite or hallucinate questions.
- This hardening remains in the eval-only Tier 3 path and must not affect the production RAG pipeline.
- Missing `evolution_type` / `synthesizer_name` across the full raw RAGAS output is now treated as a hard failure, not a warning.

## 2026-03-10 — Fixed Tier 3 difficulty mapping for real RAGAS synthesizer names

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so Tier 3 conversion normalizes real RAGAS 0.3.x synthesizer family names such as `single_hop_specifc_query_synthesizer` and `multi_hop_abstract_query_synthesizer` before assigning internal `difficulty` and `category` labels.
- Added [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) coverage that reproduces the observed raw output shape and verifies that `single_hop_*` maps to `rag_simple` while `multi_hop_*` maps to `rag_multi_context`.

Why:
- The current raw Tier 3 output proved RAGAS was returning real synthesizer names, but the conversion layer only understood the repo's internal labels (`simple`, `reasoning`, `multi_context`, `conditional`).
- That mismatch caused unknown synthesizer names to fall through to `simple`, making the whole generated dataset look like `rag_simple` even when RAGAS had emitted multi-hop rows.

Tradeoff:
- The normalization is intentionally narrow and family-based: `single_hop_*` becomes `simple`, `multi_hop_*` becomes `multi_context`. It fixes the current false labeling without overfitting the generator to every possible future synthesizer name.

Must stay true:
- Tier 3 difficulty labels must be derived from the real RAGAS metadata shape currently emitted by the installed generator, not from silent defaults.
- Unknown synthesizer families should remain inspectable in [tier3_rag.raw.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.raw.jsonl) so future mapping changes can be made deliberately.

## 2026-03-10 — Added deterministic Tier 3 quality gates for caption leakage, typo-heavy rows, and weak joins

What changed:
- Updated [scripts/build_tier3_chunk_subset.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_chunk_subset.py) so the eval-only curated subset excludes photo-credit and caption markers such as `Photograph by`, `Phototake NYC`, `Photo Researchers`, and `Reproduced by permission`.
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so Tier 3 conversion rejects obvious low-quality questions before they are frozen:
  - typo-heavy question surfaces that fuzzy-match too many context terms
  - credit/source questions such as `Phototake NYC`
  - weak multi-hop bridge questions whose contexts share too little medical topic overlap even when the pages are not far apart
- Extended [tests/eval/test_tier3_chunk_subset_builder.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_chunk_subset_builder.py) and [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) with regression cases for those exact failure modes.

Why:
- The chunk corpus and label mapping had become good enough to inspect real Tier 3 output, and the remaining failures were now specific synthetic-eval quality defects: caption leakage, typo-heavy wording, and weakly related multi-hop joins.
- The right direction at this stage is not to treat every generated row as valid; it is to generate candidates broadly and then accept only deterministic, reviewable rows into the frozen Tier 3 set.

Tradeoff:
- These gates are intentionally conservative and may reduce final sample count. That is acceptable because a smaller trusted Tier 3 set is more useful than a larger noisy one.
- The typo filter is heuristic and context-based. It is designed to remove obviously poor rows, not to perform general spell-correction.

Must stay true:
- These filters remain eval-only and must not affect production chunking, indexing, retrieval, or answer generation.
- Tier 3 quality promotion must stay deterministic and test-backed, not manual cherry-picking hidden in code.
- Raw RAGAS rows remain preserved in [tier3_rag.raw.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.raw.jsonl) so filtered-out candidates can still be audited.

## 2026-03-10 — Added manual-seed constraints and run-health contracts to Tier 3

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so Tier 3 can validate schema-compatible manual RAG seed rows from [tier3_rag.manual.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.jsonl), derive approved question archetypes from approved `human_seed` rows, and reject synthetic rows that fall outside those approved archetypes.
- Added an approved benchmark output at [tier3_rag.benchmark.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.benchmark.jsonl) so approved manual seeds and approved synthetic rows can be merged separately from the draft synthetic candidate file.
- Added explicit Tier 3 run-health thresholds for accepted-row count, accepted/raw survival rate, accepted multi-hop count, and typo/source leakage rates. Structurally unhealthy runs now exit non-zero instead of being treated as healthy dataset-generation successes.
- Tightened the eval-only multi-hop policy: `multi_hop_abstract` rows are disabled by default, and `multi_hop_specific` rows now require stronger topical cohesion derived from section/page/topic overlap before they survive conversion.
- Added an explicit upstream-stage support gate: requesting persona/scenario checkpointing or resume now raises with code-level evidence when the current installed RAGAS generator only exposes monolithic `generate*()` entrypoints.
- Extended [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock manual-seed validation, archetype derivation, disallowed `multi_hop_abstract` filtering, run-health fail-fast behavior, and the truthful upstream-stage failure path.

Why:
- The recent Tier 3 work had improved chunk selection and row-level filtering, but the dataset still lacked a clean way to promote reviewed human seeds into a first-class benchmark input or to stop low-yield paid runs from being treated as acceptable.
- We also needed to stop implying that persona/scenario checkpointing existed when the repo code only calls RAGAS through final `generate*()` methods.

Tradeoff:
- Manual-seed constraints and run-health checks are intentionally strict and may reduce final synthetic yield further. That is acceptable because a small trusted benchmark or candidate queue is more useful than a larger but structurally unhealthy output.
- Upstream persona/scenario checkpointing remains unavailable until a future RAGAS/runtime contract exposes clean public stage boundaries. This patch chooses explicit failure over a fake abstraction.

Must stay true:
- Manual Tier 3 seeds must remain valid schema_v1 RAG rows with `provenance.source_type='human_seed'`.
- Approved manual seeds and draft synthetic rows must stay separately reviewable; the benchmark file is the trusted merged view, not the draft candidate queue.
- Tier 3 run-health thresholds must remain explicit and configurable. If a run is structurally unhealthy, the command must exit non-zero.
- Unsupported upstream-stage checkpointing must continue to fail fast with evidence instead of being emulated implicitly in repo code.

## 2026-03-10 — Added the first approved Tier 3 manual seed set and explicit seed-control audit

What changed:
- Added [tier3_rag.manual.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.jsonl) with 26 approved `human_seed` Tier 3 rows covering single-hop retrieval, diagnosis/mechanism/management questions, same-topic multi-hop rows, and two control rows.
- Added [docs/evals/tier3-seed-control-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-seed-control-audit.md) to document the current pre-generation vs post-generation control boundary.
- Extended [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) so the repo now validates the manual seed fixture itself, its intended difficulty mix, and the audit-note wording.

Why:
- The Tier 3 generator already had a manual-seed code path, but the repo did not yet include an actual approved seed set.
- We also needed to make the control boundary explicit: approved seeds currently constrain survival and benchmark composition after generation, but they do not yet steer RAGAS upstream.

Tradeoff:
- The manual seed file now carries richer design-spec tags such as topic family, evidence expectation, bridge policy, and control markers. Most of that metadata is design memory today, not active upstream generation control.
- The approved benchmark view will initially be dominated by approved manual seeds until synthetic survivors are reviewed and promoted explicitly.

Must stay true:
- Manual seed rows remain valid `schema_v1` Tier 3 rows with `provenance.source_type='human_seed'` and `review_status='approved'`.
- The seed-control audit stays honest about what is pre-generation, what is post-generation, and what is unsupported by the current installed RAGAS integration.
- If future work adds real upstream seed steering, the audit note and Tier 3 docs must be updated at the same time.

## 2026-03-10 — Activated topic-family and bridge-policy controls for Tier 3 seed-gated acceptance

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so approved manual seeds now derive not only allowed archetypes but also allowed topic families and bridge policies.
- Synthetic Tier 3 rows are now rejected when their inferred topic family or inferred bridge policy falls outside the approved manual-seed set.
- Added a structured run summary that prints raw-row count, accepted-row count, accepted rows by difficulty, and non-zero rejection counts before the run-health gate is evaluated.
- Extended [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to cover topic-family gating, bridge-policy gating, and the new conversion summary output.
- Updated [docs/evals/tier3-seed-control-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-seed-control-audit.md), [README.md](/home/tough/medical_chatbot/MediGenius/README.md), and [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) so the control-activation matrix matches the code.

Why:
- Manual seeds had become a first-class benchmark artifact, but only archetype gating was code-active. Topic family and bridge policy were still descriptive tags.
- We needed a stricter, explicit acceptance contract for medical multi-hop rows and a clearer summary of whether a paid run produced useful yield.

Tradeoff:
- These controls are still post-generation. They can improve acceptance quality and reporting, but they do not raise the upstream RAGAS generation ceiling on their own.
- Topic-family inference is deterministic but heuristic when no aligned seed chunk is reused; a row can be rejected because it cannot be confidently placed in an approved topic family.

Must stay true:
- Topic-family and bridge-policy gating remain eval-only acceptance controls and must not leak into production retrieval or answer generation.
- The run summary must be printed before the structural health gate raises so low-yield paid runs are diagnosable without another rerun.
- The seed-control audit must stay aligned with which seed tags are actually active in code versus descriptive benchmark metadata.

## 2026-03-10 — Guarded the chunker long-text fallback against non-progress recursion

What changed:
- Updated [tools/pdf_chunker.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_chunker.py) so `_split_long_text()` now skips structural splitters that do not create any boundary before falling through to the next fallback.
- Added [tests/rag/test_pdf_chunker_contract.py](/home/tough/medical_chatbot/MediGenius/tests/rag/test_pdf_chunker_contract.py) to lock the delimiter-free long-text behavior and ensure the chunker hard-wraps instead of recursively retrying the same unchanged input.

Why:
- Delimiter-poor oversized text could keep re-entering `_split_long_text()` through `_pack_segments()` without making progress, which is an unsafe ingest failure mode.

Tradeoff:
- When a structural splitter cannot help, the chunker now moves to lower-priority splitters or hard-wraps sooner instead of repeatedly retrying the same boundary strategy.

Must stay true:
- Structural splitters remain preferred when they actually create meaningful boundaries.
- Oversized text with no usable heading/paragraph/line/sentence boundaries must still be chunked deterministically under the token limit.

## 2026-03-10 — Reduced Tier 3 RAGAS generation to a minimal draft-generation contract

What changed:
- Simplified [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so the active path now only loads the chunk corpus, runs RAGAS, checkpoints raw rows, converts them to schema_v1 draft samples, and supports resume from raw rows.
- Removed downstream seed-gating, benchmark-writing, heuristic acceptance filters, run-health thresholds, and persona/scenario checkpoint scaffolding from the active generator path.
- Removed `rapidfuzz` from the `eval` optional dependency set in [pyproject.toml](/home/tough/medical_chatbot/MediGenius/pyproject.toml) because the simplified generator no longer uses that fuzzy-filter dependency.
- Replaced the oversized Tier 3 generator contract suite in [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) with a smaller contract aligned to the simplified generator and the current 11-row manual seed file.
- Updated [README.md](/home/tough/medical_chatbot/MediGenius/README.md), [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md), and [docs/evals/tier3-seed-control-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-seed-control-audit.md) so the repo no longer documents the removed downstream control flow as active behavior.

Why:
- The downstream RAGAS acceptance logic had become large, hard to review, and was not outperforming manual seed curation for trusted Tier 3 benchmark inputs.
- The repo needs a smaller, more readable generation path that can be frozen while manual seed work continues separately.

Tradeoff:
- Synthetic Tier 3 output is now less aggressively filtered. Draft rows are easier to inspect, but more review burden shifts to manual curation outside the generator.
- The generator no longer treats the manual seed file as a control surface; that separation is intentional, but it removes one layer of automated downstream restriction.

Must stay true:
- Tier 3 generation remains an offline utility, not a regression gate.
- Raw RAGAS rows must continue to be checkpointed before schema conversion.
- Manual seeds remain a separate curated artifact with schema-valid approved rows, not a hidden generator configuration file.

## 2026-03-12 — Restored `rapidfuzz` to the Tier 3 eval dependency contract

What changed:
- Added `rapidfuzz` back to the `eval` optional dependency set in [pyproject.toml](/home/tough/medical_chatbot/MediGenius/pyproject.toml).
- Updated [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) so the contract matches the current article-document RAGAS pipeline.

Why:
- The corrected document-first Tier 3 generator now relies on RAGAS default transforms during live generation.
- With the installed RAGAS version, those transforms import `rapidfuzz` for overlap and string-distance relationship building, so removing it from the eval extras made the live path fail at runtime.

Tradeoff:
- The eval dependency set is slightly heavier again, but the repo contract now matches the real live generation path instead of failing on first RAGAS transform construction.

Must stay true:
- `rapidfuzz` remains an eval-only dependency and must not leak into production serving paths.
- Resume-from-raw smoke runs should stay lighter than full live generation, but the documented eval extras must satisfy the full article-doc RAGAS path.

## 2026-03-12 — Added OpenAI embeddings support to the shared embedding client

What changed:
- Updated [core/settings.py](/home/tough/medical_chatbot/MediGenius/core/settings.py) so `EMBEDDING_PROVIDER=openai` is a valid configuration and `EMBEDDING_OPENAI_MODEL` is an explicit setting.
- Updated [tools/embedding_client.py](/home/tough/medical_chatbot/MediGenius/tools/embedding_client.py) so the shared embedding client can call the OpenAI embeddings API for both single-text and batched document embeddings.
- Added [tests/rag/test_embedding_client_contract.py](/home/tough/medical_chatbot/MediGenius/tests/rag/test_embedding_client_contract.py) to lock the OpenAI provider contract, including the `text-embedding-3-large` dimensions request and the batched embedding path.

Why:
- The corrected document-first Tier 3 generator depends on a working embedding backend during live RAGAS graph construction.
- The current local Alibaba embedding model was failing inside remote model code, so the repo needed a stable remote embedding provider that could be selected cleanly through settings instead of silently falling back to the broken local path.

Tradeoff:
- The shared embedding client now has another remote provider branch and depends on correct OpenAI API credentials when `EMBEDDING_PROVIDER=openai` is selected.
- For `text-embedding-3-*` models, the client explicitly requests the repo `EMBEDDING_DIM` as the output dimension so the existing `vector(768)` schema contract stays intact, which trades away the model's full default width in favor of compatibility.

Must stay true:
- The document chunk schema contract remains `vector(768)` unless a separate schema decision changes it.
- OpenAI embeddings support must stay explicit and configuration-driven; unsupported provider names must not silently masquerade as successful OpenAI usage.
- Tier 3 live generation can use OpenAI embeddings without changing the frozen retrieval corpus or curated benchmark artifacts.

## 2026-03-12 — Added generation-only Tier 3 embedding overrides

What changed:
- Updated [core/settings.py](/home/tough/medical_chatbot/MediGenius/core/settings.py) with `TIER3_EMBEDDING_PROVIDER`, `TIER3_EMBEDDING_OPENAI_MODEL`, and `TIER3_EMBEDDING_DIM`.
- Updated [tools/embedding_client.py](/home/tough/medical_chatbot/MediGenius/tools/embedding_client.py) with explicit `EmbeddingOverrides` support so callers can request a different provider/model/dimension per embedding call without mutating the shared retrieval contract.
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so the Tier 3 RAGAS path uses those generation-only embedding overrides when configured.
- Added regression coverage in [tests/rag/test_embedding_client_contract.py](/home/tough/medical_chatbot/MediGenius/tests/rag/test_embedding_client_contract.py) and [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py).

Why:
- `text-embedding-3-large` is only compelling if Tier 3 generation can use its wider output dimension.
- The repo-wide embedding contract is still tied to `document_chunks.embedding = vector(768)`, so widening the global `EMBEDDING_DIM` would have broken retrieval/indexing instead of just improving the RAGAS generation lane.

Tradeoff:
- The shared embedding client now carries a small override contract for callers that genuinely need a different embedding runtime than the production retrieval path.
- Tier 3 generation configuration is more flexible, but the split between production retrieval embeddings and generation-only embeddings must stay explicit in docs and operator commands.

Must stay true:
- Retrieval/indexing remains bound to the schema-controlled `EMBEDDING_DIM=768` path unless a separate schema decision changes that contract.
- Tier 3 generation overrides are optional and scoped to the RAGAS generator path only.
- Using `TIER3_EMBEDDING_DIM=3072` must not change the dimensions written to the production vector store.

## 2026-03-10 — Simplified manual Tier 3 seed metadata and refreshed golden artifact docs

What changed:
- Simplified [tier3_rag.manual.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.jsonl) so the current 11 approved manual rows now carry only the minimal tags `rag` and `manual_seed`.
- Updated [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock the simpler manual-seed tag contract.
- Updated [docs/evals/tier3-seed-control-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-seed-control-audit.md) and [eval/golden/v1/README.md](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/README.md) so the repo no longer presents the retired `seed_*` taxonomy as active metadata.

Why:
- After removing downstream seed-gating code, the old `seed_*` tags had become stale design residue rather than active behavior.
- The Tier 3 manual seed file should be easy to review as a curated dataset, not cluttered with inactive control tags.

Tradeoff:
- The manual seed rows now carry less descriptive taxonomy inline. That makes the file easier to audit, but any richer design commentary now belongs in docs or future explicit benchmark notes instead of per-row inactive tags.

Must stay true:
- The manual seed file remains schema-valid and fully approved.
- Manual seed metadata should stay minimal unless new tags have an active, documented purpose.

## 2026-03-10 — Added topic-bounded Tier 3 subset generation for narrow RAGAS experiments

What changed:
- Extended [scripts/build_tier3_chunk_subset.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_chunk_subset.py) with repeatable `--topic-keyword` flags so the existing eval-only subset builder can produce a topic-bounded chunk corpus from the frozen [tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl).
- Added deterministic topic-specific default output names like `tier3_chunk_corpus.topic-cervical-cancer-pap-test.jsonl` when callers omit `--output`.
- Extended [tests/eval/test_tier3_chunk_subset_builder.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_chunk_subset_builder.py) to lock topic filtering, multi-keyword matching, and the topic-specific output path behavior.
- Updated [README.md](/home/tough/medical_chatbot/MediGenius/README.md) and [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md) to document the narrow-topic Tier 3 workflow.

Why:
- Broad medical subsets were still giving RAGAS too many opportunities to create weak cross-topic joins.
- We needed a cheaper, eval-only way to test whether RAGAS behaves better on a tighter disease/concept family before spending more tokens on broad-corpus synthetic generation.

Tradeoff:
- Topic matching is deliberately deterministic and string-based. It avoids hidden semantic heuristics, but it also means a topic slice may miss relevant chunks that use different wording than the supplied keywords.
- This is still an input-selection policy only. It does not steer RAGAS upstream or change production chunking, indexing, or retrieval behavior.

Must stay true:
- No-topic runs must behave exactly like the existing curated-subset builder.
- Topic-bounded runs remain eval-only and must not affect production chunking or the indexed corpus.
- Topic slices must stay deterministic for the same input corpus and keyword list.

## 2026-03-10 — Staged a reasoning-heavy Tier 3 manual-seed candidate batch

What changed:
- Added [tier3_rag.manual.candidates.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.candidates.jsonl) with a separate 38-row draft `human_seed` batch focused on evidence-grounded multi-step reasoning, same-topic synthesis, and a small number of explicit control rows.
- Added [tests/eval/test_tier3_manual_seed_candidates_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_candidates_contract.py) to lock the staged candidate file path, schema validity, reasoning-heavy difficulty mix, and required audit metadata.
- Updated [docs/evals/tier3-seed-control-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-seed-control-audit.md), [docs/architecture/current-state.md](/home/tough/medical_chatbot/MediGenius/docs/architecture/current-state.md), and [eval/golden/v1/README.md](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/README.md) so the repo distinguishes clearly between the approved manual set and the staged candidate queue.

Why:
- The approved 11-row manual seed file is still useful, but it underrepresents same-topic multi-step reasoning and tightly bounded multi-context synthesis for the current medical-book RAG eval goal.
- Candidate authoring needed a separate staging surface so stronger draft rows could be reviewed without mutating the approved seed contract.

Tradeoff:
- The staged candidate file carries richer audit tags than the approved file. That improves later review and filtering, but it deliberately keeps more metadata in the draft queue than in the frozen approved set.
- This adds another Tier 3 support artifact to maintain, but it is intentionally separated from the approved benchmark path so review decisions remain explicit.

Must stay true:
- The approved manual seed file is not overwritten by candidate authoring.
- Candidate rows remain schema-compatible `human_seed` draft rows and stay outside the generator control path.
- Promotion from candidate to approved remains a human review step, not an implicit code path.

## 2026-03-11 — Added a PDF-authored Tier 3 candidate path and stronger structural contract

What changed:
- Added [tier3_rag.manual.candidates.pdf_v1.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.candidates.pdf_v1.jsonl) with a separate 38-row draft batch authored from the book-level PDF and mapped back to frozen chunk IDs.
- Added [tier3_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_pdf_authoring.py), [build_tier3_pdf_authoring_index.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_pdf_authoring_index.py), and [tier3_pdf_authoring_index.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_pdf_authoring_index.jsonl) so manual seed authoring can start from article/page structure instead of isolated chunks.
- Added [test_tier3_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_pdf_authoring.py), [test_tier3_manual_seed_pdf_candidates_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py), and [tier3-manual-seed-pdf-batch-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-manual-seed-pdf-batch-audit.md) to lock the new authoring helper, staged fixture path, structural-difficulty deltas, and topic-diversity deltas.

Why:
- The earlier staged candidate batch was still too single-chunk-heavy and too concentrated in one gynecology topic pocket for the harder retrieval-plus-synthesis benchmark now being targeted.
- Manual authoring needed a PDF-first path so question design could follow article-level evidence paths, missing-subject chunks, and neighbor/heading dependence before chunk IDs were attached.

Tradeoff:
- Tier 3 now carries one more staged candidate artifact and an authoring index to maintain.
- The authoring index is heuristic and eval-only; it improves manual review and row design, but it is not a production retrieval asset or a benchmark file.

Must stay true:
- The approved manual seed file and the earlier staged candidate file remain unchanged.
- The new PDF-authored batch stays a draft `human_seed` artifact for later human review, not an approved benchmark.
- Structural hardness is enforced by chunk-count and challenge-type contract checks, not just by difficulty labels.

## 2026-03-11 — Staged a retrieval-hard Tier 3 manual-seed slice

What changed:
- Added [tier3_rag.manual.candidates.retrieval_hard_v1.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.candidates.retrieval_hard_v1.jsonl) with a separate 20-row draft batch biased toward chunk-fragment retrieval stress.
- Added [test_tier3_manual_seed_retrieval_hard_candidates_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_retrieval_hard_candidates_contract.py) to lock the staged path, schema validity, retrieval-hard challenge mix, topic-diversity criteria, and low-overlap wording checks.
- Added [tier3-manual-seed-retrieval-hard-batch-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-manual-seed-retrieval-hard-batch-audit.md) and updated the Tier 3 artifact docs so the retrieval-hard slice stays an explicit review artifact rather than an implicit benchmark promotion.

Why:
- The earlier PDF-authored batch had stronger local synthesis, but it still underweighted the chunk-quality failure modes the corpus actually exposes: subject recovery, generic heading dependence, adjacent chunk completion, and distractor resistance.
- A separate retrieval-hard slice keeps those stress cases reviewable without diluting the medium-hard PDF-authored set or mutating the approved benchmark file.

Tradeoff:
- Tier 3 now carries one more staged manual-seed file and one more contract test.
- The retrieval-hard slice is smaller than the PDF-authored batch, so its topic-diversity contract is enforced by explicit family-share limits rather than raw row count alone.

Must stay true:
- The retrieval-hard slice remains a draft `human_seed` artifact for later human review.
- The approved manual seed file and the earlier staged candidate files remain untouched.
- Hardness claims for this slice must continue to be backed by evidence-path checks, not just difficulty labels.

## 2026-03-11 — Finalized the retrieval-hard manual-seed lane with rewrites and a stricter v2 batch

What changed:
- Added [tier3_rag.manual.candidates.retrieval_hard_v1_rewrites.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.candidates.retrieval_hard_v1_rewrites.jsonl) as a six-row sibling rewrite set for the weaker `retrieval_hard_v1` rows `002`, `003`, `009`, `012`, `016`, and `018`.
- Added [tier3_rag.manual.candidates.retrieval_hard_v2.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.candidates.retrieval_hard_v2.jsonl) as a new 20-row draft batch that pushes harder on cross-page title/body dependence, subject recovery, adjacent-chunk completion, and low-overlap phrasing.
- Added [test_tier3_manual_seed_retrieval_hard_revision_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_retrieval_hard_revision_contract.py) and [test_tier3_manual_seed_retrieval_hard_v2_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_retrieval_hard_v2_contract.py) to lock the sibling rewrite path, exact source-row coverage, stronger structural criteria, topic-diversity limits, and phrase-overlap ceiling.
- Added [tier3-manual-seed-retrieval-hard-rewrites-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-manual-seed-retrieval-hard-rewrites-audit.md) and [tier3-manual-seed-retrieval-hard-v2-audit.md](/home/tough/medical_chatbot/MediGenius/docs/evals/tier3-manual-seed-retrieval-hard-v2-audit.md), and updated the Tier 3 artifact docs to include both new staged files.

Why:
- `retrieval_hard_v1` improved challenge coverage, but too many rows still relied on same-page local bridging and a few of the questions remained too textbook-like or too close to the source wording.
- The user wanted a final retrieval-hard lane that is visibly stricter on chunk-fragment recovery and better isolated for later human review, without mutating either the approved file or the earlier staged candidate files.

Tradeoff:
- Tier 3 now carries two more staged manual-seed artifacts and two more contract tests.
- The sibling rewrite set preserves comparison value, but it also means review decisions now span the original retrieval-hard slice, targeted rewrites, and the final stricter batch.

Must stay true:
- Neither `retrieval_hard_v1` nor any approved Tier 3 manual-seed artifact is overwritten by this pass.
- The rewrite file and the v2 file remain draft `human_seed` artifacts for later human review.
- Retrieval-hard claims for the final lane continue to be backed by explicit evidence-structure and lexical-overlap tests, not only by labels.

## 2026-03-11 — Collapsed Tier 3 subsets into one final curated benchmark

What changed:
- Rebuilt [tier3_rag.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.jsonl) as a `53`-row approved `human_seed` benchmark with explicit lane tags:
  - `11` baseline rows
  - `16` medium-hard rows
  - `26` retrieval-hard rows
- Promoted the selected rows into one file with `benchmark_set:final_curated_v1` and `benchmark_lane:*` tags, and set all merged rows to approved provenance.
- Moved the offline generator draft output path in [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) from `tier3_rag.jsonl` to `tier3_rag.generated_draft.jsonl` so synthetic runs cannot overwrite the curated benchmark.
- Removed the retired committed subset fixtures, their artifact-specific audit notes, and their contract tests:
  - manual-only Tier 3 files
  - staged candidate subset files
  - retrieval-hard intermediate subset files
  - committed chunk-corpus subset outputs such as `small`, `curated`, and topic-bounded JSONLs
- Added [test_tier3_final_curated_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_final_curated_contract.py) to lock the exact 53-row composition, lane balance, approved provenance, and retirement of the older subset files.

Why:
- The repo had reached the point where the subset files were more useful as source material than as ongoing committed artifacts.
- Retrieval, reranking, and chunking comparisons now need one stable trusted Tier 3 benchmark instead of multiple overlapping staged subsets.

Tradeoff:
- The intermediate subset files are no longer available as committed JSONL artifacts in this worktree.
- Historical implementation-log entries still record how those subsets were created, but the active benchmark surface is intentionally smaller and cleaner now.

Must stay true:
- `tier3_rag.jsonl` is the single committed Tier 3 benchmark.
- The offline generator must not write synthetic drafts into the curated benchmark path.
- The full chunk corpus and PDF authoring index remain available for future audit or authoring work, but committed subset JSONLs stay retired.

## 2026-03-12 — Preserved the PDF-first manual-seed authoring path in the main repo

What changed:
- Added [tier3_manual_seed_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_manual_seed_pdf_authoring.py), [build_tier3_manual_seed_authoring_index.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_manual_seed_authoring_index.py), and [tier3_manual_seed_authoring_index.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_manual_seed_authoring_index.jsonl) under explicit manual-seed names.
- Preserved [tier3_rag.raw.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.raw.jsonl) in the main repo so the synthetic path still has an inspectable raw checkpoint.
- Added [test_tier3_manual_seed_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_pdf_authoring.py) and [test_tier3_manual_seed_authoring_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_manual_seed_authoring_contract.py) to lock the preserved helper path and artifact presence.

Why:
- The final curated merge intentionally collapsed the committed Tier 3 benchmark into one file, but the harder manual-seed workflow still depends on a PDF-first authoring aid rather than isolated chunk writing.
- Renaming the helper path makes its role explicit: manual-seed authoring support, not production retrieval logic and not benchmark truth.

Tradeoff:
- The repo keeps a small eval-only authoring surface alongside the curated benchmark.
- The raw RAGAS checkpoint remains available for synthetic-path inspection even though the curated benchmark no longer depends on synthetic rows.

Must stay true:
- The PDF-first helper remains eval-only and must not become part of the production retrieval path.
- `tier3_rag.jsonl` remains the single committed Tier 3 benchmark.
- `tier3_rag.raw.jsonl` remains an offline checkpoint, not benchmark truth.

## 2026-03-12 — Switched Tier 3 RAGAS generation to article-entry source documents

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so RAGAS now consumes article-entry source documents instead of retrieval chunks.
- Added [eval/tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_article_documents.py) as the neutral shared layer for article-window recovery and article-to-chunk alignment, then rewired [tier3_manual_seed_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_manual_seed_pdf_authoring.py) to reuse it instead of owning parallel logic.
- Added [scripts/run_tier3_article_generation_smoke.py](/home/tough/medical_chatbot/MediGenius/scripts/run_tier3_article_generation_smoke.py) as the bounded smoke entrypoint that resumes from the raw checkpoint by default, limits article count, and writes only to `/tmp`.
- Extended the Tier 3 generator and authoring contract tests to lock:
  - cached article-corpus loading
  - context-to-chunk alignment from article docs
  - neutral shared alignment logic
  - smoke-run command planning and output isolation

Why:
- The previous synthetic path reused retrieval chunks as if they were source documents, which was the wrong boundary for RAGAS and pushed generation toward shallow, fragmented, low-trust questions.
- The medical book is naturally article-structured, so article-entry documents are the coherent source unit while the chunk corpus should remain a downstream alignment and retrieval-eval artifact.

Tradeoff:
- Tier 3 generation now carries one more intermediate cache artifact, `tier3_article_corpus.jsonl`, when the article corpus is materialized.
- Article-window recovery is heuristic and still depends on stable PDF title/footer structure, so document quality now depends more on that shared extraction layer than before.

Must stay true:
- RAGAS must receive article-entry documents, not retrieval chunks, on the active Tier 3 generation path.
- The frozen chunk corpus remains available for schema alignment and retrieval experiments, but it is not the generation substrate.
- The smoke runner must keep writing to `/tmp` and default to raw-checkpoint resume so the first validation pass stays cheap and safe.

## 2026-03-12 — Cleaned article bodies and promoted the temporary cancer experiment to same-family generation docs

What changed:
- Updated [eval/tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_article_documents.py) so article body extraction now drops any leading carryover lines that appear before a detected start-title on the first page of an article.
- Updated [eval/tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_article_documents.py) so article body extraction stops at intra-article back-matter markers such as `BOOKS`, `PERIODICALS`, `ORGANIZATIONS`, and `OTHER`.
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so the temporary cancer-family experiment no longer feeds the four selected articles directly into RAGAS. It now builds two larger same-family generation docs:
  - `cancer-core` from `Cancer`
  - `cancer-therapy` from `Cancer therapy, definitive`, `Cancer therapy, palliative`, and `Cancer therapy, supportive`
- Added regression coverage in [test_tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_article_documents.py) and [test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py).

Why:
- The previous narrow cancer-family run was operationally healthy but still produced poor rows because the generation docs were polluted by adjacent-entry carryover and bibliography/organization material.
- The measured structure of this book showed that article docs are already dense and section docs are usually too small. A temporary same-family bundle layer is a better fit for medically coherent multi-hop than raw sections or much larger chapter blobs.

Tradeoff:
- The bundle logic is still temporary and hardcoded for the current experiment.
- The article-body cleanup remains heuristic and tuned to this encyclopedia-style PDF layout.

Must stay true:
- Tier 3 generation remains document-first.
- Retrieval chunking stays a separate concern from the generation substrate.
- Any later generalization of bundle docs should replace the hardcoded cancer-family logic with an explicit, reviewable contract.

## 2026-03-12 — Added a temporary cancer-family RAGAS profile for narrow Tier 3 generation experiments

What changed:
- Added a temporary hardcoded cancer-family source-doc selector in [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) for live Tier 3 generation runs. The current experiment keeps only `Cancer`, `Cancer therapy, definitive`, `Cancer therapy, palliative`, and `Cancer therapy, supportive` before calling RAGAS.
- Added an explicit Tier 3 query distribution in [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) that excludes `MultiHopAbstractQuerySynthesizer` and biases toward `MultiHopSpecificQuerySynthesizer` over `SingleHopSpecificQuerySynthesizer`.
- Added an explicit Tier 3 transform profile in [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) built from `SummaryExtractor`, `EmbeddingExtractor`, `ThemesExtractor`, `NERExtractor`, `CustomNodeFilter`, and document-level similarity builders instead of relying on `default_transforms()`.
- Added regression coverage in [test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock the temporary selector and the explicit RAGAS profile wiring.

Why:
- The document-first substrate fixed the earlier chunk-fed generation problem, but the default RAGAS query mix still produced low-value abstract multi-hop rows across unrelated medical topics.
- A narrow, coherent topic family is the fastest way to check whether the remaining quality problem is really RAGAS profile configuration rather than the article-doc boundary itself.

Tradeoff:
- This is intentionally hardcoded and temporary. It improves reviewability for the current experiment, but it is not the final generalized Tier 3 generation interface and should be removed or generalized once the quality direction is proven.

Must stay true:
- The active generation substrate remains document-first.
- The temporary experiment must stay isolated from the production retrieval schema and the curated Tier 3 benchmark.
- If this experiment is later generalized, the selector and explicit RAGAS profiles should become configurable instead of silently staying hardcoded.

## 2026-03-12 — Simplified Tier 3 chunk alignment and removed the retired chunk-subset experiment path

What changed:
- Refactored [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so chunk loading is row-based only; the generator no longer carries the old chunk-`Document` bridge just to build alignment inputs.
- Softened article-doc to chunk alignment so pre-aligned chunk ids are only a preferred hint, not a hard gate. When aligned ids miss, conversion now falls back to page-span candidates before leaving `ground_truth_chunk_ids` empty.
- Removed the retired local chunk-subset experiment code:
  - `scripts/build_tier3_chunk_subset.py`
  - `tests/eval/test_tier3_chunk_subset_builder.py`

Why:
- The active document-first generator did not need chunk `Document` objects anymore; retaining them made the code look like it still had two competing generation substrates.
- `ground_truth_chunk_ids` are useful support metadata, but a good generated row should not degrade just because the first alignment hint was incomplete.
- The chunk-subset experiment path was left over from the older chunk-fed RAGAS era and was no longer part of the intended Tier 3 flow.

Tradeoff:
- The generator is now narrower and easier to review, but there is less local tooling for topic-slice experiments unless that path is rebuilt later on top of the document-first pipeline.
- Chunk-id alignment remains heuristic; the code now chooses to preserve valid rows rather than over-constrain them.

Must stay true:
- Valid RAGAS rows with non-empty `ground_truth_contexts` remain convertible even when chunk-id alignment is empty.
- The active Tier 3 generation path stays document-first.
- Any future topic-slice or curation experiments must be reintroduced explicitly instead of drifting back in as leftover helper code.

## 2026-03-12 — Filtered non-entry article docs out of Tier 3 RAGAS generation

What changed:
- Added a shared Tier 3 article-doc eligibility filter in [eval/tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_article_documents.py) so non-medical-entry windows such as contents/front matter, contributor pages, bibliography/reference sections, index/glossary-style back matter, and author-bio pages are excluded from source-document generation.
- Applied that filter both when building article docs from the PDF and when loading a cached [tier3_article_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_article_corpus.jsonl), so stale caches cannot keep poisoning later runs.
- Added regression coverage in [test_tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_article_documents.py) and [test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py).

Why:
- The first live article-doc smoke run proved the substrate change worked, but quality was still badly contaminated because the earliest article-doc windows included contributors and other non-benchmark material.
- That is an upstream corpus-selection problem, not something worth solving again with downstream rejection heuristics.

Tradeoff:
- The article-doc filter is heuristic and intentionally conservative; if the book later adds unusual medical entry types that look like back matter, they may need a targeted contract update.
- Cached article corpora generated before this change will now load with fewer docs, which is the intended behavior.

Must stay true:
- Tier 3 RAGAS generation should only consume medically meaningful entry documents.
- Contributor rosters, bibliography/reference pages, index/glossary-style pages, and similar non-entry material must not reach RAGAS as generation source docs.
- Filtering must happen both on fresh PDF extraction and when reusing cached article corpora.

## 2026-03-12 — Added Tier 3-only LLM overrides so RAGAS generation no longer depends on the app LLM

What changed:
- Added Tier 3 generation-only LLM settings in [core/settings.py](/home/tough/medical_chatbot/MediGenius/core/settings.py):
  - `TIER3_LLM_PRIMARY_PROVIDER`
  - `TIER3_LLM_PRIMARY_MODEL`
  - `TIER3_LLM_FALLBACK_PROVIDER`
  - `TIER3_LLM_FALLBACK_MODEL`
- Added [resolve_llm()](/home/tough/medical_chatbot/MediGenius/tools/llm_client.py) to [tools/llm_client.py](/home/tough/medical_chatbot/MediGenius/tools/llm_client.py) so the generator can request a provider/model pair without changing the shared app LLM contract.
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so Tier 3 generation now prefers the Tier 3-specific LLM settings and only falls back to the shared app settings when those overrides are absent.
- Added a regression test in [test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock the Tier 3-only LLM boundary.

Why:
- The earlier generation run still used Groq for the RAGAS LLM because only embeddings had been isolated for Tier 3. That made the generator depend on the main app LLM settings and on Groq quota even when Tier 3 was supposed to run on OpenAI.
- Tier 3 generation should be an eval-specific offline utility with its own model controls, not a side effect of the production app defaults.

Tradeoff:
- There is now one more eval-specific settings boundary to document and maintain, but it removes surprising coupling between the app and the offline generator.

Must stay true:
- Tier 3 generation model selection must not silently depend on the main app LLM path when Tier 3 overrides are set.
- Main app LLM behavior remains unchanged unless the shared `LLM_*` settings are edited directly.

## 2026-03-12 — Filtered the temporary Tier 3 query profile against the built knowledge graph

What changed:
- Updated [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) so the temporary cancer-family generator no longer hands its custom query profile directly to `generate_with_langchain_docs()`.
- The generator now builds the RAGAS knowledge graph explicitly with the chosen transforms, filters the custom query distribution against that graph using each synthesizer's `get_node_clusters(...)`, renormalizes the remaining weights, and then calls `generate(...)` with the filtered profile.
- Added regression coverage in [tests/eval/test_tier3_generator_contract.py](/home/tough/medical_chatbot/MediGenius/tests/eval/test_tier3_generator_contract.py) to lock query-profile filtering and the explicit `knowledge_graph`-backed generation path.

Why:
- The narrow same-family cancer run was failing in RAGAS scenario generation with `ValueError: No clusters found in the knowledge graph` because the custom profile forced `MultiHopSpecificQuerySynthesizer` even when the temporary graph had no `entities_overlap` edges.
- Official RAGAS default query selection filters unavailable synthesizers against the built graph. The temporary profile needed the same behavior instead of a hard failure.

Tradeoff:
- A narrow run may degrade to single-hop-only generation when the graph does not support the requested multi-hop synthesizer. That is less ambitious, but it is a truthful signal instead of a crash.

Must stay true:
- Temporary Tier 3 query profiles must be filtered against the actual built knowledge graph before scenario generation.
- A narrow Tier 3 experiment should fail only when no configured synthesizer is usable at all, not because one unavailable multi-hop synthesizer was forced unconditionally.

## 2026-03-12 — Retired the active Tier 3 RAGAS generation lane and kept the manual benchmark path only

What changed:
- Moved the active Tier 3 synthetic generator, smoke runner, raw checkpoint, and generator-specific tests into [bin/legacy_tier3_ragas](/home/tough/medical_chatbot/MediGenius/bin/legacy_tier3_ragas) so they are frozen reference artifacts instead of active eval code.
- Removed the active `eval/generate_tier3_rag.py` path from the repo surface and updated the Tier 3 contracts to treat manual seeds as the only supported benchmark source.
- Simplified [eval/tier3_article_documents.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_article_documents.py) so it now serves only the manual PDF/article extraction and chunk-alignment workflow instead of carrying RAGAS-specific document-building helpers.
- Updated the active Tier 3 docs and tests to reflect the new boundary and added [ADR-0002-tier3-manual-seed-only.md](/home/tough/medical_chatbot/MediGenius/docs/decisions/ADR-0002-tier3-manual-seed-only.md).

Why:
- The curated manual benchmark is the only Tier 3 source currently trusted for later RAG pipeline evaluation.
- Keeping the synthetic generator active was increasing architectural confusion and maintenance cost without serving the trusted benchmark path.

Tradeoff:
- There is no supported synthetic Tier 3 expansion path in the active eval package anymore.
- Future synthetic experiments must be reintroduced explicitly instead of drifting back in through archived code.

Must stay true:
- The active Tier 3 surface is the curated benchmark, the frozen retrieval corpus, and the manual authoring helpers.
- Archived RAGAS generation code under `bin/legacy_tier3_ragas/` is reference-only and must not be wired back into CI or the main eval package by accident.

## 2026-03-13 — Hardened the manual Tier 3 benchmark with explicit evidence roles, stable anchors, and a dev split

What changed:
- Updated [tier3_rag.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.jsonl) so every row now carries a deterministic `split` (`14` `dev`, `39` `test`) and an `evidence` object with `required`, `supporting`, and `hard_negative_candidates`.
- Stable evidence anchors now include `chunk_id`, `doc_id`, `page`, `section`, `content_type`, and `anchor_text` instead of relying on `chunk_id` alone to explain the gold evidence.
- Updated [eval/validate.py](/home/tough/medical_chatbot/MediGenius/eval/validate.py), [eval/retrieval_eval.py](/home/tough/medical_chatbot/MediGenius/eval/retrieval_eval.py), and [eval/ragas_eval.py](/home/tough/medical_chatbot/MediGenius/eval/ragas_eval.py) so the active eval surface validates and can target the new split-aware Tier 3 benchmark shape.

Why:
- The manual benchmark is now the long-term source of truth for RAG evaluation, so it needs stronger evidence semantics than a flat bag of chunk IDs.
- Distinguishing required evidence from supporting evidence and adding explicit hard negatives makes later retrieval comparisons more interpretable and less forgiving.

Tradeoff:
- The benchmark rows are denser and the schema is stricter, which slightly raises maintenance cost during manual review.

Must stay true:
- Tier 3 remains manual-only and trusted; these additions harden that benchmark instead of reintroducing synthetic generation complexity.
- `chunk_id` remains useful for exact retrieval scoring, but stable anchors must remain present so the benchmark survives chunking changes more gracefully.

## 2026-03-13 — Raised the ingest container memory cap after full-book reindex OOM

What changed:
- Updated [docker-compose.yml](/home/tough/medical_chatbot/MediGenius/docker-compose.yml) so the dedicated `ingest` service now uses `mem_limit: 8g` instead of `4g`.

Why:
- The active reindex path is materially heavier than the earlier lightweight PDF path:
  - [tools/pdf_loader.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_loader.py) now uses the structure-aware Docling parser instead of the old PyPDF + recursive splitter path.
  - [tools/pdf_parser.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_parser.py) retains richer typed document structure in memory during parsing.
  - [tools/vector_store.py](/home/tough/medical_chatbot/MediGenius/tools/vector_store.py) still materializes the full parsed document, full chunk list, and full embedding batch before the final document-level database replace.
- A live full-book reindex in the `ingest` container was OOM-killed with exit code `137` while the old committed `4379` rows remained intact, proving the process exceeded the old container cap before the final commit.

Tradeoff:
- The larger cap unblocks the current full-book reindex, but it does not solve the deeper memory shape of the ingest pipeline. The process is still all-at-once rather than streamed.

Must stay true:
- The `ingest` service remains isolated from the app runtime budget.
- If reindexing still OOMs at `8g`, the next fix should be pipeline memory reduction or streaming, not repeated arbitrary cap increases.

## 2026-03-08 — Initialized project guidance and architecture docs

What changed:
- Added AGENTS.md, CLAUDE.md, architecture doc stub, implementation log, and ADR template.

Why:
- The project needs stronger global guidance and design memory to prevent local patching and architectural drift.

Tradeoff:
- Slightly more documentation overhead, but much better long-term consistency.

Must stay true:
- Shared project rules remain concise and current.
- Architectural changes are documented when material.

---

## 2026-03-12 — BM25 upgrade for sparse retrieval and reranking

What changed:
- Added Alembic migration `0004_content_tsv`: adds `content_tsv tsvector` column, GIN index, backfill, and auto-populate trigger on `document_chunks`.
- Replaced `ILIKE` conjunction keyword search in `VectorRepository` with `ts_query` + `ts_rank` against the GIN-indexed tsvector column.
- Replaced token-overlap reranker in `retriever_agent.py` with BM25 scoring (term frequency with length normalization).
- Replaced binary term counting in `InMemoryVectorRepository.keyword_search` with the same BM25 scoring for test parity.
- Added `BM25_K1` (default 1.2) and `BM25_B` (default 0.75) to `core/settings.py` — no hardcoded constants.

Why:
- RAGAS baseline showed context_recall=0.00 and context_precision=0.30, confirming retrieval is the primary bottleneck.
- `ILIKE` conjunction requires all terms to match (boolean AND) and returns no ranking signal.
- Token-overlap reranker has no term-frequency awareness or document-length normalization.

Tradeoff:
- PostgreSQL `ts_rank` is not full BM25 (no IDF across corpus), but it is a large improvement over ILIKE and avoids external dependencies.
- Migration must be run before the new keyword search works (`alembic upgrade head`).

Must stay true:
- All schema changes go through Alembic migrations only.
- BM25 parameters are configurable via settings, not hardcoded.
- `InMemoryVectorRepository` matches `VectorRepository` scoring semantics for test parity.

---

## 2026-03-14 — Tier 3 supporting-evidence curation completed

What changed:
- Filled the previously empty `evidence.supporting` buckets across the manual Tier 3 benchmark in `eval/golden/v1/tier3_rag.jsonl`.
- Kept the existing three-bucket grading model intact: `required` -> grade 3, `supporting` -> grade 2, `hard_negative_candidates` -> grade 1.
- Tightened four rows where the same chunk had accidentally been labeled as both `supporting` and `hard_negative_candidates` by swapping in nearby same-topic distractors or dropping the conflicting distractor.

Why:
- `retrieval_eval.py` now scores graded evidence directly from these buckets, so missing or ambiguous supporting labels weaken the benchmark and can distort `precision@k`, `MAP`, and `nDCG`.
- Overlapping support and hard-negative labels make the benchmark internally contradictory.

Tradeoff:
- Supporting evidence was added conservatively: each new anchor is same-topic and genuinely helpful, but not required for a correct answer.
- This improves ranking sensitivity without widening gold evidence so far that retrieval gets free credit.

Must stay true:
- `supporting` anchors remain subsets of `ground_truth_chunk_ids`.
- `hard_negative_candidates` never overlap gold evidence.
- Evidence labels should be curated slowly and semantically, not bulk-generated from proximity alone.

### 2026-03-15 — Redis caching layer

What changed:
- Added Redis Stack service (`redis/redis-stack-server:7.4.0-v3`) to docker-compose with health checks and persistent volume.
- Created `tools/redis_client.py`: `ExactCache` class for deterministic key-value caching (Cohere rerank, Tavily search) with in-memory dict fallback when Redis is unavailable.
- Upgraded `tools/cache.py` `SemanticCache` from in-memory deque (O(n) linear scan) to Redis Stack vector search (HNSW, O(log n)) with automatic in-memory fallback.
- Added caching to `agents/retriever_agent.py` (`_cohere_rerank`: 1h TTL) and `agents/tavily_agent.py` (24h TTL).
- Added Prometheus metrics: `medigenius_tool_cache_hits_total`, `medigenius_tool_cache_misses_total` (by tool label), `medigenius_redis_up` gauge.
- Added 3 Grafana dashboard panels: tool cache hit/miss rate, tool hit ratio gauge, Redis status indicator.
- Strategy B defaults: threshold 0.92, TTL 24h, cache size 5000.
- Raised executor prompt token budget from 4000 to 16000 (matching 128k-context models).
- Removed hardcoded `_INTENT_INSTRUCTIONS` dict and `turn_intent` parameter from executor prompt rendering.

Why:
- Cohere rerank is rate-limited to 10 calls/min; caching prevents rate limit exhaustion and 62s retry waits.
- Tavily costs money per call; caching identical medical queries within 24h avoids redundant spend.
- In-memory semantic cache with linear scan doesn't persist across restarts, isn't shared across workers, and scales poorly.
- Token budget of 4000 was using 2.5% of 128k context window; neighbor expansion could theoretically hit the cap.
- Intent instructions fired unpredictably because `turn_intent` is free-form LLM output matched against 6 hardcoded keys.

Tradeoff:
- Redis is a new infrastructure dependency, but all code gracefully falls back to in-memory when Redis is unavailable.
- Strategy B (0.92 threshold, global scope) may serve slightly wrong answers for semantically similar but different questions — acceptable for development, tighten for production.

Must stay true:
- `SemanticCache` public interface (get, set, clear, configure, `get_semantic_cache()`, `semantic_cache_override()`) must not change — callers depend on it.
- All caches must fall back to in-memory when `REDIS_URL` is unset or Redis is down.
- Tool cache TTLs must be shorter than the data's freshness window (rerank: 1h, web search: 24h).
- Grafana dashboard JSON must remain valid and loadable by Grafana provisioning.

## 2026-03-16 — Thinking patterns: Structured Reflection + ReAct retry + Plan-and-Execute stub

**What changed:**
- Upgraded `ReflectionAgent` from binary judge (is_relevant/has_hallucinations) to structured output with `failure_category`, `suggested_focus`, `confidence`, and `feedback`.
- Added ReAct-style targeted re-retrieval: on retry, the retriever uses `reflection_suggested_focus` as the search query instead of repeating the original query. Step-back query is skipped on retry.
- Stubbed `DecomposerAgent` for Plan-and-Execute pattern. Detects multi-part questions with rule-based heuristics. Disabled by default via `ENABLE_QUERY_DECOMPOSITION=false`.

**Why:**
- Binary reflection couldn't guide the retry — same query produced same results. Structured feedback with suggested_focus enables targeted re-retrieval (ReAct pattern).
- Plan-and-Execute is scaffolded now for future multi-PDF ingestion where complex questions may need to be decomposed across document sources.

**What must remain true:**
- Reflection max attempts = 2 (hardcoded in agent).
- Retry path still goes reflection → retriever → executor → reflection (no workflow graph changes).
- DecomposerAgent is disabled by default and NOT wired into the workflow graph.
- All new state fields have defaults in `initialize_state` and `reset_query_state`.

## 2026-03-16 — Parent-child chunk hierarchy

**What changed:**
- Added two-tier chunking: small child chunks (~200 tokens) for retrieval, large parent chunks (~800–1000 tokens) for executor context.
- `group_children_into_parents()` in pdf_chunker.py groups consecutive same-section children. Tables become their own parent.
- Parents stored in DB with NULL embedding (never searched). Children reference parents via `parent_chunk_id` in metadata.
- `get_chunks_by_ids()` added to both VectorRepository and InMemoryVectorRepository.
- `_resolve_parent_chunks()` in retriever_agent fetches parent content after retrieval.
- Executor uses `parent_content` metadata when available, skipping adjacent expansion.
- Feature gated by `ENABLE_PARENT_CHILD_CHUNKING=false` (default).

**Why:**
- Smaller chunks improve retrieval precision (more specific matches). Larger context improves generation quality (LLM sees full paragraph/section context). Parent-child gives both without compromise.
- Replaces the ±1 adjacent expansion approach with a cleaner, index-time solution.

**What must remain true:**
- When disabled (default), behavior is identical to pre-existing single-tier chunking.
- Parents MUST have NULL embedding (never searched).
- Toggling the feature requires reindex.
- `get_chunks_by_ids()` must exist on both repo implementations.

---

## 2026-03-17 — Long-term user memory layer

**What changed:** Added cross-session user memory: persistent anonymous `user_token` (localStorage UUID → `X-User-Token` header → `users` table), `long_term_facts` table (user-scoped structured facts, always loaded by `MemoryAgent` when `user_id` present), `conversation_memories` table (pgvector episodic memories, semantically recalled top-K by cosine similarity). New DB models `UserModel`, `LongTermFactModel`, `ConversationMemoryModel` in `db/models.py`. Alembic migration `0005_add_long_term_memory`. `LongTermMemoryRepository` in `db/long_term_memory_repository.py`. Write path runs in existing `BackgroundTask` via `write_long_term_memory()` in `api/chat_runtime.py`. `AgentStateV2` extended with `user_id`, `episodic_memories`, `long_term_memory_repo`. `ExecutorAgent` injects episodic block into the prompt. Gated by `LONG_TERM_MEMORY_ENABLED=false`.

**Why:** Session-scoped memory meant users had to repeat conditions/medications every session — poor UX and clinical quality for a medical chatbot.

**Tradeoff:** ~1 extra DB read + 1 pgvector query per turn for known users. Embedding latency is async (BackgroundTask). Anonymous users (no `X-User-Token`) and disabled-feature users degrade gracefully with zero overhead.

**Must remain true:** `LONG_TERM_MEMORY_ENABLED=false` by default. `user_id=None` → all paths skip silently. Confidence-gated upsert (higher confidence wins). Long-term memory write failures must never fail the user response.

## 2026-03-19 — Restore benchmark run tracking and Grafana benchmark datasource

**What changed:** Restored the missing benchmark tracking slice. Added `eval/run_tracker.py` to normalize saved retrieval, RAGAS, and workflow reports into compact run rows plus metric/slice rows. Added `EvalRunModel`, `EvalRunMetricModel`, and `EvalRunSliceModel` in `db/models.py`, plus `EvalRunRepository.record_run()` in `db/repositories.py`. Added Alembic migration `alembic/versions/0005_add_eval_run_tracking.py` for `eval_runs`, `eval_run_metrics`, and `eval_run_slices`. Wired `eval/ragas_eval.py`, `eval/retrieval_eval.py`, and `eval/workflow_eval.py` to save snapshots first and then best-effort persist run summaries. Added the Grafana `Postgres Benchmark` datasource and passed Postgres env vars into the Grafana service. Updated the stale executor prompt regression to match the current `query_context` / `session_intent` prompt contract.

**Why:** CI contracts and the accepted benchmark observability design expected saved eval snapshots to also produce queryable benchmark history in PostgreSQL and Grafana. The repo still had the tests, dashboard queries, and ADR, but the implementation slice was missing.

**Tradeoff:** Saved evals now have an extra Postgres persistence path, but it is explicitly best-effort so tracking failures do not block the main eval artifact. The new migration file name follows the existing test contract even though its Alembic revision is chained after the current `0005` head.

**Must remain true:** Snapshot save remains the artifact of record and happens before DB tracking. Tracking failures must never fail the main eval run. Grafana benchmark panels must use PostgreSQL, not Prometheus labels, for benchmark truth.
