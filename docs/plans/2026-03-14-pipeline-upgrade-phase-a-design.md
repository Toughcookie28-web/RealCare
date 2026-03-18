# Pipeline Upgrade Phase A: Core Pipeline

**Goal:** Consolidate query understanding into a single LLM call (rewrite + route + intent + slots), eliminate the planner's LLM call, add k-NN fallback routing, memory route, and adjacent chunk expansion.

**Architecture:** The rewriter becomes the single query-understanding node, outputting structured JSON with 7 fields. The planner becomes a zero-LLM dispatcher reading route from state. k-NN provides fallback routing when the rewriter LLM fails. Memory route skips retrieval and generates from conversation history. Adjacent chunk expansion happens in the executor during context assembly, controlled by token budget.

---

## 1. Unified Rewriter

The rewriter LLM call expands from 2 fields to 7. Same single call, same conversation history input:

```python
class RewriteResult(BaseModel):
    optimized_query: str
    stepback_query: str = ""
    route: str = "vector"              # vector, web, literature, memory, chitchat
    session_intent: str = ""           # ongoing conversation goal
    turn_intent: str = ""              # what this specific message wants
    slots: dict[str, str] = {}         # semi-structured: drug, condition, population, aspect, timeframe
    query_context: str = ""            # free-form description for executor
    reasoning: str = ""
```

Prompt additions:
- `route`: classify where to search (vector, web, literature, memory, chitchat)
- `session_intent` / `turn_intent`: two-level intent from conversation history
- `slots`: extract medical entities with suggested keys (drug, condition, population, aspect, timeframe) — LLM can add others
- `query_context`: free-form description of user need, passed to executor

Graceful degradation unchanged (4 tiers). If JSON parsing fails, `route` defaults to `"vector"`, slots to `{}`, intents to `""`.

### State changes

Add to `AgentStateV2`:
```python
session_intent: str
turn_intent: str
slots: dict[str, str]
query_context: str
```

Initialize as `""`, `""`, `{}`, `""` in `initialize_state()` and `reset_query_state()`.

## 2. Planner as Dispatcher + Memory Route

The planner becomes a zero-LLM dispatcher:

```python
VALID_ROUTES = {'chitchat', 'vector', 'web', 'future_clinical_db', 'literature', 'memory'}

def PlannerAgent(state):
    route = state.get('route', '')
    if route not in VALID_ROUTES:
        route = knn_fallback_route(state)
    state['route'] = route
    state['planned_route'] = route
    state['current_tool'] = route
    return state
```

### Workflow changes

`route_after_planner` adds memory path:

```python
def route_after_planner(state) -> str:
    route = state.get('route', 'vector')
    if route == 'chitchat':
        return 'chitchat'
    if route == 'memory':
        return 'memory'
    return 'retriever'
```

Memory route goes directly to executor (no retrieval). The executor detects `route == 'memory'` and switches to a memory-aware prompt that generates from `conversation_history` + `summary` + `facts`.

## 3. k-NN Fallback Router

When rewriter LLM fails and `route` is invalid, planner calls k-NN instead of keyword matching.

### Components

1. **Seed file** — `data/route_seeds.json`: pre-embedded seed queries per route (~30 per route, 5 routes)
2. **Seed generation script** — `scripts/generate_route_seeds.py`: takes hand-written seeds, embeds via `embed_query()`, saves JSON
3. **k-NN router module** — `tools/knn_router.py`: loads seeds on first call (module-level cache), cosine similarity, top-k majority vote
4. **Integration** — planner calls `knn_fallback_route(state)` which embeds the query and calls `knn_route()`

If seeds file is missing or confidence < 40%, returns `"vector"` as safe default.

User deliverable: write ~30 seed queries per route before implementation.

## 4. Adjacent Chunk Expansion in Executor

During context assembly, the executor expands each retrieved chunk by ±1 neighbors within the same section.

### Repository addition

Both `VectorRepository` and `InMemoryVectorRepository` get:
```python
def get_adjacent_chunks(self, section: str, chunk_index: int) -> list[Document]
```

### Executor changes

`_build_context_blocks()` receives vector_repo and for each chunk:
1. Reads `chunk_index` and `section` from metadata
2. Fetches neighbors via `get_adjacent_chunks()`
3. Inserts before/after original chunk in the same context block
4. Deduplicates against other top-5 chunks
5. Token budget loop naturally limits inclusion

### Boundaries
- Only expand within same section
- Skip if `chunk_index` missing from metadata
- Skip if `vector_repo` not available
- No cross-section expansion

## 5. Graceful Degradation

| Component | Failure | Fallback |
|---|---|---|
| Rewriter LLM call fails | `optimized_query` = original question, `route` = "" | Planner → k-NN fallback |
| Rewriter JSON parse fails | Tiered: partial JSON → raw text, `route` = "" | Planner → k-NN fallback |
| k-NN no seeds file | Log warning, return "vector" | Safe default |
| k-NN low confidence (<40%) | Return "vector" | Safe default |
| Memory route, no history | Executor generates "no previous context" message | Graceful user message |
| Adjacent expansion: no chunk_index | Skip expansion for that chunk | No degradation |
| Adjacent expansion: no vector_repo | Skip all expansion | Current behavior |

## Files Changed

| File | Change |
|---|---|
| `core/state_v2.py` | Add `session_intent`, `turn_intent`, `slots`, `query_context` |
| `agents/query_rewriter_agent.py` | Expand `RewriteResult`, update prompt, write new fields to state |
| `agents/planner_agent.py` | Remove LLM call, read route from state, call k-NN fallback on invalid route |
| `core/langgraph_workflow.py` | Add `memory` route to `route_after_planner` conditional edges |
| `tools/knn_router.py` | New: load seeds, cosine similarity, majority vote |
| `scripts/generate_route_seeds.py` | New: embed seed queries, save to JSON |
| `db/repositories.py` | Add `get_adjacent_chunks()` to both repository classes |
| `agents/executor_agent.py` | Memory-aware prompt branch, adjacent chunk expansion in `_build_context_blocks()` |
| `tests/rag/test_query_rewriter_contract.py` | Expand for new fields |
| `tests/rag/test_planner_dispatcher.py` | New: dispatcher reads state, k-NN fallback |
| `tests/rag/test_knn_router.py` | New: seed loading, majority vote, missing file fallback |
| `tests/rag/test_adjacent_expansion.py` | New: executor expansion, boundary cases |

## Phase B (deferred)

The following features depend on Phase A being complete and will be designed separately:

- Pre-retrieval metadata boost using slots
- Post-retrieval slot validation + diversity enforcement + confidence adjustment
- Context assembly with `context_summary` from enriched metadata
- Answer grounding with section/page citations
- Hallucination detection via slot cross-check
- Executor prompt optimization using intent + query_context
- Progressive fact disclosure
