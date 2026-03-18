# Pipeline Upgrade Phase B: Metadata-Powered Retrieval + Executor Optimization

**Goal:** Use the structured query understanding from Phase A (slots, intents, query_context) plus chunk metadata (enrichment fields, section, page) to improve retrieval precision, answer quality, and source transparency.

**Architecture:** Five features that plug into existing hooks — no new LLM calls except the existing executor call, no new services, no schema changes. All features degrade gracefully when metadata is absent.

---

## 1. Pre-Retrieval Metadata Boost

After hybrid search, before BM25 rerank, add a soft score boost (+0.15) to chunks whose `keywords` or `medical_entities` metadata matches slot values (drug, condition, population). No hard filtering — all chunks remain candidates. No-op when slots are empty or enrichment metadata is absent.

## 2. Post-Retrieval Slot Validation + Confidence Adjustment

After reranking, validate top-5 chunks against slots. Compute `slot_coverage` (fraction of non-empty slots found in results' content or metadata). When `slot_coverage < 0.3` and `turn_intent` is specific (dosage_lookup, drug_comparison, etc.), lower `retrieval_confidence` by 30%. Store `slot_coverage` in state for observability.

## 3. Executor Prompt Optimization

Map `turn_intent` to formatting instructions injected into the executor system prompt:
- `definition` → structured definition with one-sentence summary
- `mechanism_of_action` → step-by-step causal chain
- `drug_comparison` → structured comparison format
- `dosage_lookup` → specific dosage with population/route
- `side_effects` → organized by frequency/severity
- `differential_diagnosis` → structured differential

Append `query_context` to user prompt. Add `session_intent` to system prompt when present.

## 4. Context Assembly with Metadata Enrichment

In `_build_context_blocks()`:
- Prepend `context_summary` as a header when present: `[Context: {summary}]`
- Prioritize `chunk_position == 'intro'` chunks earlier in context
- Add `section_path` to context block headers
- Summaries trimmed first when token budget is tight

## 5. Answer Grounding with Citations

After answer generation, extract `section`, `page`, `doc_id` from chunks used in context. Append a deduplicated "Sources" block. Store structured citations in `state['citations']`.

## Graceful Degradation

| Feature | Missing data | Behavior |
|---|---|---|
| Metadata boost | No enrichment metadata on chunks | Boost skipped silently |
| Slot validation | No slots from rewriter | Validation skipped, confidence unchanged |
| Prompt optimization | No turn_intent | No extra instruction added |
| Context enrichment | No context_summary | Context assembly works as today |
| Citations | No section/page metadata | Citations block omitted |

## Phase B-2 (Future — after eval evidence)

| Feature | Why deferred |
|---|---|
| Hallucination detection (slot cross-check) | Needs prompt engineering + eval for false positive rate |
| Progressive fact disclosure | Needs fact-query relevance scoring |
| Diversity enforcement | Needs eval to prove benefit vs. precision loss |

## Files Changed

| File | Change |
|---|---|
| `agents/retriever_agent.py` | Add `metadata_boost()`, `validate_retrieval()` |
| `agents/executor_agent.py` | Add `_intent_instruction()`, enrich context blocks, add `_build_citations()` |
| `core/state_v2.py` | Add `slot_coverage`, `citations` |
| `tests/rag/test_metadata_boost.py` | New: boost scoring, empty slots, missing metadata |
| `tests/rag/test_slot_validation.py` | New: coverage calc, confidence adjustment, empty slots |
| `tests/rag/test_executor_prompt_optimization.py` | New: intent mapping, query_context injection |
| `tests/rag/test_context_enrichment.py` | New: context_summary prepend, intro prioritization |
| `tests/rag/test_answer_citations.py` | New: citation extraction, dedup, missing metadata |
