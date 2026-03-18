# Query Transformation Design: Medical Normalization + Conditional Step-Back

**Goal:** Improve retrieval recall by bridging the vocabulary gap between user queries and textbook language, using medical term normalization and conditional step-back query generation.

**Architecture:** Single LLM call in the existing rewriter agent produces structured JSON with two queries: a normalized `optimized_query` (always) and a broader `stepback_query` (when the LLM judges it useful). The retriever searches both, merges and deduplicates, then BM25 reranks the combined pool against the specific query. Pydantic validates the output with tiered graceful degradation on parse failure.

---

## Layer 1: Medical Normalization (in rewrite prompt)

The rewriter prompt is upgraded to instruct the LLM to:

1. **Resolve conversational references** — replace pronouns and implicit references using conversation history
2. **Expand medical abbreviations** — "MI" → "myocardial infarction", "BP" → "blood pressure"
3. **Add medical synonyms** — "heart attack (myocardial infarction)", "blood thinner (anticoagulant)"
4. **Preserve specificity** — keep dosages, populations, timeframes exactly as stated

This runs on every query at zero extra latency (rewriter LLM call already exists).

## Layer 2: Conditional Step-Back Query

The same LLM call optionally produces a `stepback_query` — a broader, principle-level query that retrieves governing context (pharmacology sections, mechanism overviews, guideline summaries).

**When to generate step-back:** questions involving mechanisms of action, drug comparisons, timeline-specific lookups, dosage-specific queries, or questions about treatment rationale.

**When NOT to generate:** simple definitions, single-fact lookups, chitchat-adjacent queries.

The LLM makes this decision based on prompt-embedded criteria.

## Output Contract

```python
class RewriteResult(BaseModel):
    optimized_query: str
    stepback_query: str = ""
    reasoning: str = ""
```

### Graceful degradation chain

1. `invoke_json()` → `RewriteResult.model_validate()` → success
2. Pydantic validation fails → extract `optimized_query` from partial JSON, ignore rest
3. JSON parsing fails entirely → use raw LLM text as `optimized_query`, empty `stepback_query`
4. LLM call fails entirely → use original `question` as `optimized_query`, empty `stepback_query`

No LLM retry on failure. The pipeline must never block on query transformation.

## Example Transformations

| User query | optimized_query | stepback_query |
|---|---|---|
| "what does aspirin do for heart attacks?" | "aspirin mechanism of action for myocardial infarction (heart attack)" | "pharmacology of antiplatelet agents in cardiovascular disease" |
| "what's the dosage?" (history: ibuprofen) | "ibuprofen dosage recommendations" | "" |
| "how does it compare to tylenol?" (history: ibuprofen) | "ibuprofen compared to acetaminophen (tylenol) efficacy and safety" | "NSAID versus non-NSAID analgesic comparison" |
| "what is hypertension?" | "hypertension (high blood pressure) definition and overview" | "" |

## State Changes

Add to `AgentStateV2`:
```python
stepback_query: str  # initialized as "" in initialize_state and reset_query_state
```

## Retriever Changes

Current:
```
query = optimized_query or question
docs = hybrid_search(query, k=20)
ranked = rerank(query, docs)
return top 5
```

New:
```
query = optimized_query or question
docs = hybrid_search(query, k=20)

stepback = state.get('stepback_query', '')
if stepback:
    stepback_docs = hybrid_search(stepback, k=10)
    docs = merge_and_deduplicate(docs, stepback_docs)  # dedupe by chunk_id

ranked = rerank(query, docs)  # always against optimized_query
return top 5
```

- Primary search: k=20 (unchanged)
- Step-back search: k=10 (supplementary)
- Deduplication by `chunk_id` metadata
- BM25 reranking always against `optimized_query` so specific relevance wins

No changes to planner, executor, tavily, wikipedia, literature, reflection, or explanation agents.

## Files Changed

| File | Change |
|---|---|
| `agents/query_rewriter_agent.py` | New prompt, structured JSON output, Pydantic validation, graceful degradation |
| `agents/retriever_agent.py` | Conditional step-back search, merge + dedup before rerank |
| `core/state_v2.py` | Add `stepback_query` field |
| `tests/rag/test_query_rewriter_contract.py` | New: Pydantic contract, degradation tiers, mock LLM output tests |
| `tests/rag/test_retriever_stepback.py` | New: merge/dedup logic, step-back skip when empty |

## Testing

- All tests mock `invoke_json()` — no live LLM calls in regression suite
- Contract tests verify Pydantic model validates well-formed JSON and provides defaults
- Degradation tests verify each fallback tier
- Retriever tests verify dedup by chunk_id, combined reranking, and empty-stepback skip

## Observability

- `reasoning` field logged as status event via existing `run_node` wrapper
- Log `stepback_generated: true/false`
- Log retriever merge stats: `primary_docs`, `stepback_docs`, `after_dedup`

## Eval Plan

After implementation:
1. `eval/retrieval_eval.py --mode all` — compare retrieval metrics before vs after
2. `eval/ragas_eval.py` — compare generation quality
3. Save snapshots to `eval/ragas_snapshots/` for A/B comparison
