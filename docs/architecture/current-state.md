# Current System State

## Project purpose
MediGenius is a FastAPI-based medical education assistant with:
- a LangGraph workflow for routing, retrieval, answer generation, reflection, and explanation
- a PostgreSQL + pgvector backing store for conversation memory and document retrieval
- explicit local/container RAG ingestion and evaluation entrypoints
- observability hooks for logs, metrics, and tracing

The current active engineering phase is post-stabilization backend validation: prove the live Docker/runtime path, expand the reviewed frozen eval contract, and prepare a chunking A/B comparison. Frontend and broader multi-agent redesign remain intentionally out of scope.

## High-level architecture
- **UI and HTTP entrypoints**
  - [app.py](/home/tough/medical_chatbot/MediGenius/app.py) creates the FastAPI app and mounts chat, history, HITL, and health routes.
  - The frontend is server-rendered templates plus static assets, but UI work is not part of the current stabilization phase.
- **Workflow orchestration**
  - [core/workflow_service.py](/home/tough/medical_chatbot/MediGenius/core/workflow_service.py) owns the process-level workflow service singleton.
  - [core/langgraph_workflow.py](/home/tough/medical_chatbot/MediGenius/core/langgraph_workflow.py) composes the LangGraph node flow.
  - Agent nodes live in [agents/](/home/tough/medical_chatbot/MediGenius/agents) and still operate as a single in-process workflow, not isolated services.
- **Persistence and schema**
  - [db/models.py](/home/tough/medical_chatbot/MediGenius/db/models.py) defines ORM models.
  - [db/session.py](/home/tough/medical_chatbot/MediGenius/db/session.py) now verifies schema state instead of creating tables.
  - [db/schema_contract.py](/home/tough/medical_chatbot/MediGenius/db/schema_contract.py) is the single source of truth for the document chunk vector contract.
  - Alembic is the only supported schema mutator.
- **RAG ingest pipeline**
  - [tools/pdf_parser.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_parser.py) parses PDFs into a structured `ParsedDocument`.
  - The parser now bootstraps Docling layout/table artifacts into the writable embedding cache before creating the converter when `model.safetensors` is missing.
  - The bootstrap also normalizes nested Docling snapshot downloads so the runtime loader can still find `docling_artifacts/model.safetensors` at the root path it expects.
  - Picture and figure items are handled safely: unsupported picture-markdown exports are skipped so they cannot break text/table extraction.
  - [tools/pdf_chunker.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_chunker.py) turns parsed elements into retrieval chunks with rich metadata: chunk position (intro/body/conclusion), cross-references, schema-aware table linearization for embedding text, and table structure metadata (columns, row count, size tier).
  - [tools/chunk_enrichment.py](/home/tough/medical_chatbot/MediGenius/tools/chunk_enrichment.py) provides optional LLM-based chunk enrichment (`context_summary` only) behind `ENRICH_CHUNKS_WITH_LLM`.
  - [tools/vector_store.py](/home/tough/medical_chatbot/MediGenius/tools/vector_store.py) orchestrates explicit parse, chunk, (optional enrichment), embed, and index stages.
  - [tools/ingest_report.py](/home/tough/medical_chatbot/MediGenius/tools/ingest_report.py) captures stage counts for reindex/debug visibility.
- **Retrieval and answer generation**
  - [db/repositories.py](/home/tough/medical_chatbot/MediGenius/db/repositories.py) provides vector and chat repository access. Sparse keyword search uses PostgreSQL `tsvector` + GIN index with `ts_rank` scoring.
  - [agents/query_rewriter_agent.py](/home/tough/medical_chatbot/MediGenius/agents/query_rewriter_agent.py) is the unified query understanding node. A single LLM call produces: optimized_query (medical normalization), stepback_query (conditional), route classification, two-level intent (session + turn), semi-structured slot extraction, and free-form query_context. Uses Pydantic validation with tiered graceful degradation.
  - [agents/planner_agent.py](/home/tough/medical_chatbot/MediGenius/agents/planner_agent.py) is a zero-LLM dispatcher that reads route from state (set by rewriter). Falls back to k-NN seed-based routing via [tools/knn_router.py](/home/tough/medical_chatbot/MediGenius/tools/knn_router.py) when route is missing or invalid.
  - [agents/retriever_agent.py](/home/tough/medical_chatbot/MediGenius/agents/retriever_agent.py) runs hybrid retrieval and BM25-based reranking (configurable `BM25_K1` and `BM25_B` in settings). When a step-back query is present, retrieves supplementary docs (k=10), merges and deduplicates by `chunk_id`, then BM25 reranks the combined pool against the specific query. On reflection retry, the retriever uses `reflection_suggested_focus` from state as the search query instead of the original `optimized_query`, and skips step-back retrieval since the step-back query was generated for the original question.
  - [agents/executor_agent.py](/home/tough/medical_chatbot/MediGenius/agents/executor_agent.py) assembles context and generates answers. Supports memory route (generates from conversation history without retrieval) and adjacent chunk expansion (fetches ±1 neighbor chunks within the same section during context assembly). Context assembly prepends `context_summary` metadata as headers when available and prioritizes intro-position chunks first. Prompt includes `query_context` and `session_intent`. Generated answers include deduplicated section/page citations appended as a Sources block.
- **Cache and memory**
  - [tools/cache.py](/home/tough/medical_chatbot/MediGenius/tools/cache.py) provides the semantic cache with explicit enable/disable and namespace controls.
  - [tools/working_memory.py](/home/tough/medical_chatbot/MediGenius/tools/working_memory.py) defines the Redis working-memory contract: rolling `recent_turns` plus versioned `core_state`.
  - [tools/working_memory_service.py](/home/tough/medical_chatbot/MediGenius/tools/working_memory_service.py) is the hot-memory overlay that reads Redis first, falls back to PostgreSQL `summary`/`facts`, and projects locked core state back into the legacy `summary` + `facts` shape so existing agents can benefit without a state-schema rewrite.
  - Conversation summary, facts, history, and HITL reviews are persisted in PostgreSQL-backed repositories.
  - Redis working memory is now an overlay, not the source of truth. If Redis is empty or unavailable, the app still runs on the existing PostgreSQL-backed memory path.
  - **Long-term user memory** (gated by `LONG_TERM_MEMORY_ENABLED=false`): a three-layer cross-session memory system anchored by a persistent anonymous `user_token` (UUID in browser `localStorage` → `X-User-Token` header → `users` table row). `long_term_facts` (user-scoped structured facts, always loaded by `MemoryAgent` when `user_id` is present) and `conversation_memories` (pgvector episodic summaries, semantically recalled top-K by cosine similarity) persist across sessions. Write path runs in the existing `BackgroundTask`. `AgentStateV2` now carries `user_id`, `episodic_memories`, and `long_term_memory_repo`. See ADR-0006.

  Memory stack (revised 2026-03-17):
  - Request scope   → `AgentStateV2`
  - Session (hot)   → Redis working memory (`recent_turns` TTL=12h, `core_state` TTL=7d)
  - Session (cold)  → PostgreSQL `messages`, `conversation_summaries`, `user_facts` (session-scoped)
  - User (always)   → PostgreSQL `long_term_facts` (user-scoped, always loaded when `user_id` present)
  - User (recalled) → PostgreSQL `conversation_memories` + pgvector (semantic recall, top-K cosine similarity)
- **Live observability and quality proxies**
  - [observability/live_judge.py](/home/tough/medical_chatbot/MediGenius/observability/live_judge.py) computes post-response live quality proxies for arbitrary frontend traffic.
  - [api/chat_runtime.py](/home/tough/medical_chatbot/MediGenius/api/chat_runtime.py) owns working-memory loading and post-response side effects so chat routes stay thin.
  - [observability/metrics.py](/home/tough/medical_chatbot/MediGenius/observability/metrics.py) now exposes live proxy metrics for Grafana: answer relevance, groundedness, context precision proxy, context coverage proxy, judge latency, and judge failures.
- **Eval entrypoints**
  - [eval/workflow_eval.py](/home/tough/medical_chatbot/MediGenius/eval/workflow_eval.py) runs end-to-end workflow evaluation and now disables semantic cache by default.
  - [eval/retrieval_eval.py](/home/tough/medical_chatbot/MediGenius/eval/retrieval_eval.py) can now run either against the live repository or frozen fixture repositories for regression testing.
  - [eval/ragas_eval.py](/home/tough/medical_chatbot/MediGenius/eval/ragas_eval.py) writes snapshots to the host-mounted [eval/ragas_snapshots](/home/tough/medical_chatbot/MediGenius/eval/ragas_snapshots) path when run inside the long-lived `eval` container.
  - [scripts/run_final_pipeline.py](/home/tough/medical_chatbot/MediGenius/scripts/run_final_pipeline.py) is the operator helper for the local final-run loop: it brings up `app` + `eval` + Prometheus + Grafana, replays HTTP workload through `/api/chat` so dashboard metrics appear in Grafana, and saves both pipeline retrieval and full RAGAS snapshots.
  - Tier 3 synthetic RAGAS generation is no longer part of the active eval package. The old code path is frozen under [bin/legacy_tier3_ragas](/home/tough/medical_chatbot/MediGenius/bin/legacy_tier3_ragas) for reference only.
  - The active Tier 3 eval surface is now the curated benchmark [tier3_rag.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.jsonl), the frozen retrieval corpus [tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl), and the PDF-first manual authoring helpers.
  - [docs/evals/error-taxonomy.md](/home/tough/medical_chatbot/MediGenius/docs/evals/error-taxonomy.md) and [rag_trace_review.jsonl](/home/tough/medical_chatbot/MediGenius/eval/review_samples/rag_trace_review.jsonl) define the reviewed failure vocabulary for current eval work.
  - [docs/evals/live-runtime-validation.md](/home/tough/medical_chatbot/MediGenius/docs/evals/live-runtime-validation.md) defines the minimal live Docker/runtime contract and the extended non-blocking shadow path.
- **Container and runtime**
  - [Dockerfile](/home/tough/medical_chatbot/MediGenius/Dockerfile) builds the runtime image from `pyproject.toml`.
  - [docker-compose.yml](/home/tough/medical_chatbot/MediGenius/docker-compose.yml) runs built images and mounts only mutable data/cache/eval artifact paths.
  - The `eval` service now bind-mounts [eval/ragas_snapshots](/home/tough/medical_chatbot/MediGenius/eval/ragas_snapshots) in addition to retrieval/workflow snapshot folders so RAGAS saves survive container exit.
  - The dedicated `ingest` service is capped separately from the app runtime and now uses `mem_limit: 8g`. The structure-aware Docling parser plus local embedding path no longer fits reliably inside the older `4g` cap for full-book reindex runs.
  - Containerized RAG/eval jobs always use `/app/.cache/embeddings` as the in-container cache root; host-local `.env` values are not allowed to remap that path.
  - `.env` and [core/settings.py](/home/tough/medical_chatbot/MediGenius/core/settings.py) remain the main runtime configuration surface.
  - The shared embedding client now supports `local`, `gemini`, and `openai` providers. Retrieval/indexing still follows the repo-wide `EMBEDDING_DIM=768` contract, while Tier 3 generation can optionally override its embedding provider/model/dimension without changing the retrieval schema contract.

## Current important flows

### Chat request flow
1. A request enters FastAPI through the chat route layer in [api/routes/chat.py](/home/tough/medical_chatbot/MediGenius/api/routes/chat.py).
2. Before the workflow runs, [api/chat_runtime.py](/home/tough/medical_chatbot/MediGenius/api/chat_runtime.py) loads a memory bundle through [tools/working_memory_service.py](/home/tough/medical_chatbot/MediGenius/tools/working_memory_service.py): Redis `recent_turns` and `core_state` first, PostgreSQL summary/facts/history as fallback.
3. The workflow runs guardrail, memory, unified query understanding (rewrite + route + intent + slots), dispatcher planning, retrieval (or memory/web/literature fallback), execution with adjacent chunk expansion, reflection, and explanation.
4. The executor may serve from semantic cache only when the current cache contract is enabled and the request is not in a retry/reflection path.
5. Results are returned through the API along with source/citation metadata and persisted PostgreSQL session memory.
6. After the response is sent, the app asynchronously:
   - appends the latest turns to Redis `recent_turns`
   - promotes stable facts + summary-derived context into Redis `core_state`
   - runs live approximate quality judging and emits Prometheus metrics for Grafana

Important distinction:
- These live scores are observability proxies for arbitrary traffic.
- They are not benchmark-truth metrics like offline retrieval recall or saved RAGAS eval snapshots.

### Retrieval flow
1. Reindex is an explicit operation through [scripts/reindex_pdf.py](/home/tough/medical_chatbot/MediGenius/scripts/reindex_pdf.py). API startup does not auto-ingest.
2. `parse` stage: [tools/pdf_loader.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_loader.py) calls [tools/pdf_parser.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_parser.py) to produce a `ParsedDocument`.
3. `chunk` stage: [tools/pdf_chunker.py](/home/tough/medical_chatbot/MediGenius/tools/pdf_chunker.py) produces retrieval-ready LangChain `Document` objects with section/table metadata.
4. `embed` stage: [tools/vector_store.py](/home/tough/medical_chatbot/MediGenius/tools/vector_store.py) builds embedding texts and requests vectors from [tools/embedding_client.py](/home/tough/medical_chatbot/MediGenius/tools/embedding_client.py).
5. `index` stage: the same module builds the batch payload and upserts it through [db/repositories.py](/home/tough/medical_chatbot/MediGenius/db/repositories.py).
6. Retrieval uses the indexed chunks through the vector repository and the retriever/executor path inside the workflow.

### Eval flow
- **Preflight**
  - [scripts/preflight_rag.py](/home/tough/medical_chatbot/MediGenius/scripts/preflight_rag.py) validates imports, writable cache paths, DB availability, Alembic head, and vector type contract before RAG/eval work.
  - Preflight is a runtime diagnostic only; it is not part of the frozen regression gate.
- **Live runtime validation**
  - [scripts/run_live_runtime_smoke.py](/home/tough/medical_chatbot/MediGenius/scripts/run_live_runtime_smoke.py) runs the minimal blocking local smoke path for the containerized runtime contract.
  - The live smoke path uses `python3 -m alembic -c /app/alembic.ini upgrade head` so the container contract does not depend on implicit config discovery.
  - [scripts/run_live_rag_shadow.py](/home/tough/medical_chatbot/MediGenius/scripts/run_live_rag_shadow.py) runs the non-blocking extended live shadow path for reindex, retrieval eval, and chunking eval.
- **Frozen regression suite**
  - [tests/eval/](/home/tough/medical_chatbot/MediGenius/tests/eval) contains the trusted regression gate for retrieval/workflow behavior and eval-contract checks.
  - [tests/fixtures/rag/](/home/tough/medical_chatbot/MediGenius/tests/fixtures/rag) contains frozen fixture corpora and cases so regression runs do not depend on live parser, DB, or LLM state.
- **Workflow eval**
  - [eval/workflow_eval.py](/home/tough/medical_chatbot/MediGenius/eval/workflow_eval.py) runs either the real workflow against the live vector store or injected frozen fixture cases, and disables semantic cache by default.
  - Cache can be re-enabled only with the explicit `--semantic-cache` flag.
- **Tier 3 benchmark and authoring**
  - [tier3_rag.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.jsonl) is the single curated Tier 3 benchmark, with explicit `benchmark_lane:baseline`, `benchmark_lane:medium_hard`, and `benchmark_lane:retrieval_hard` tags.
  - The benchmark now carries a deterministic `dev` / `test` split plus explicit `evidence.required`, `evidence.supporting`, and `evidence.hard_negative_candidates` anchors so retrieval experiments can distinguish must-hit evidence from supporting evidence and harder distractors.
  - Stable evidence anchors now carry `chunk_id`, `doc_id`, `page`, `section`, `content_type`, and `anchor_text`; retrieval eval still uses chunk IDs first, but the benchmark no longer depends on `chunk_id` alone to explain why a row is correct.
  - [tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl) remains the frozen retrieval corpus for retrieval evals and manual benchmark review.
  - [tier3_manual_seed_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_manual_seed_pdf_authoring.py) and [tier3_manual_seed_authoring_index.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_manual_seed_authoring_index.jsonl) are the active PDF-first manual-seed helpers: recover article/page structure from the book PDF, then align it back to the frozen chunk corpus so harder rows can be authored from article-level evidence paths instead of isolated chunks.
  - The Tier 3 synthetic RAGAS generation lane has been retired from the active eval package. Its old generator, smoke runner, raw checkpoint, and generator-specific tests are frozen under [bin/legacy_tier3_ragas](/home/tough/medical_chatbot/MediGenius/bin/legacy_tier3_ragas) for reference only.

### Cache and memory flow
- **Semantic cache**
  - Process-local only.
  - Keyed by semantic similarity plus an explicit cache namespace/version contract.
  - Disabled by default in development/local/test/eval-style environments unless `SEMANTIC_CACHE_ENABLED=true` is set.
  - Expected invalidation path is version bump through `SEMANTIC_CACHE_VERSION` when prompt/retrieval contract changes are intentional.
- **Conversation memory**
  - Stored in PostgreSQL through the chat repository, summaries table, and user facts table.
- **HITL reviews**
  - Persisted in PostgreSQL. The old in-memory fallback path has been removed from the RAG/eval runtime.

### Reflection and retry flow
1. After executor generates an answer, the reflection agent judges quality using structured criteria.
2. The judge returns: `failure_category` (none/irrelevant/hallucination/incomplete/unsafe), `suggested_focus` (targeted retrieval hint), `confidence` (0.0–1.0), and `feedback`.
3. If `failure_category != 'none'` and attempts < 2, the workflow retries.
4. On retry, the retriever uses `suggested_focus` as the search query instead of the original `optimized_query` (ReAct pattern: observe feedback, act on it).
5. The step-back query is skipped on retry since it was generated for the original question, not the focused retry.
6. Maximum 2 reflection attempts, then finalize regardless.

### Plan-and-Execute (stubbed)
- `DecomposerAgent` can detect multi-part questions and split them into sub-questions.
- Disabled by default (`ENABLE_QUERY_DECOMPOSITION=false`).
- Not yet wired into the LangGraph workflow graph.
- Intended for future multi-PDF support where sub-questions may route to different document sources.

### Parent-child chunk hierarchy (optional)
- When `ENABLE_PARENT_CHILD_CHUNKING=true`, the chunker creates two tiers:
  - **Child chunks** (~200 tokens): embedded and used for retrieval precision.
  - **Parent chunks** (~800–1000 tokens): stored with NULL embedding, never searched directly. Groups of 3–5 consecutive children within the same section.
- Tables are their own parent (self-contained retrieval units).
- After retrieval finds relevant children, `_resolve_parent_chunks()` fetches parent content from the DB via `get_chunks_by_ids()`.
- The executor uses parent content for context assembly, providing richer context than the small child chunk alone.
- When parent content is available, adjacent chunk expansion is skipped (parent already provides the broader context).
- Disabled by default. Requires reindex when toggling.
- Settings: `CHUNK_CHILD_TARGET=200`, `CHUNK_CHILD_MAX=250`, `CHUNK_PARENT_MAX_CHILDREN=5`.

## Current explicit contracts
- **Schema authority**
  - Alembic only. Runtime refuses to start against missing/stale schema state.
- **Vector dimension**
  - `document_chunks.embedding` must be `vector(768)`.
  - `EMBEDDING_DIM` must match the schema contract.
- **Ingest boundaries**
  - Parse, chunk, embed, and index are distinct stages and must stay separately inspectable.
- **Eval cache policy**
  - Workflow eval does not use semantic cache unless explicitly opted in.
  - Frozen retrieval/workflow regression does not depend on semantic cache at all.

## Current known pain points
- [x] Unclear workflow boundaries in ingest have been reduced, but retrieval/executor/reflection coupling is still dense.
- [x] Cache invalidation is now explicit, but cache correctness still depends on disciplined versioning for intentional contract changes.
- [x] API startup no longer mutates schema or ingests the corpus, but operators must now run migrations/reindex explicitly.
- [x] Retriever quality upgraded: sparse search uses `tsvector` + `ts_rank`, reranking uses BM25 scoring.
- [x] Tier 3 generation has been demoted to an offline utility; the required regression path is now frozen and code-based.
- [ ] CI now has non-blocking live runtime and live shadow jobs, but they are not yet trusted enough to be required gates.

## Current priorities
1. Validate that the stabilized system runs correctly in live Docker/runtime.
2. Expand the reviewed frozen eval contract without making it noisy.
3. Prepare a chunking A/B comparison where chunking is the intended variable and runtime stays fixed.
4. Avoid frontend or architectural expansion until the runtime/eval/chunking validation phase is complete.
