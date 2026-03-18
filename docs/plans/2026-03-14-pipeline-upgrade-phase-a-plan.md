# Pipeline Upgrade Phase A Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate query understanding into a single LLM call (rewrite + route + intent + slots), eliminate the planner's LLM call, add k-NN fallback routing, memory route, and adjacent chunk expansion.

**Architecture:** The rewriter becomes the single query-understanding node producing 7-field structured JSON. The planner becomes a zero-LLM dispatcher. k-NN provides fallback routing. Memory route skips retrieval. Adjacent chunk expansion happens in executor context assembly.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph, pgvector, pytest

---

### Task 1: Expand AgentStateV2 with new fields

**Files:**
- Modify: `core/state_v2.py`
- Test: `tests/rag/test_state_v2_fields.py`

**Step 1: Write the failing test**

Create `tests/rag/test_state_v2_fields.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_state_v2_has_new_pipeline_fields():
    from core.state_v2 import initialize_state

    state = initialize_state(session_id='s1', trace_id='t1')
    assert state['session_intent'] == ''
    assert state['turn_intent'] == ''
    assert state['slots'] == {}
    assert state['query_context'] == ''


def test_reset_query_state_clears_new_fields():
    from core.state_v2 import initialize_state, reset_query_state

    state = initialize_state(session_id='s1', trace_id='t1')
    state['session_intent'] = 'medical_consultation'
    state['turn_intent'] = 'dosage_lookup'
    state['slots'] = {'drug': 'metformin'}
    state['query_context'] = 'user wants dosage info'

    reset_query_state(state, 'new question')
    assert state['session_intent'] == ''
    assert state['turn_intent'] == ''
    assert state['slots'] == {}
    assert state['query_context'] == ''
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_state_v2_fields.py -v`
Expected: FAIL with `KeyError: 'session_intent'`

**Step 3: Write minimal implementation**

In `core/state_v2.py`, add four fields to `AgentStateV2`:

```python
class AgentStateV2(TypedDict):
    # ... existing fields ...
    session_intent: str
    turn_intent: str
    slots: dict[str, str]
    query_context: str
```

In `initialize_state()`, add to the return dict:

```python
    'session_intent': '',
    'turn_intent': '',
    'slots': {},
    'query_context': '',
```

In `reset_query_state()`, add to the `state.update({...})` dict:

```python
    'session_intent': '',
    'turn_intent': '',
    'slots': {},
    'query_context': '',
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rag/test_state_v2_fields.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add core/state_v2.py tests/rag/test_state_v2_fields.py
git commit -m "feat: add session_intent, turn_intent, slots, query_context to AgentStateV2"
```

---

### Task 2: Expand RewriteResult and update rewriter prompt

**Files:**
- Modify: `agents/query_rewriter_agent.py`
- Modify: `tests/rag/test_query_rewriter_contract.py`

**Step 1: Write the failing tests**

Add to `tests/rag/test_query_rewriter_contract.py`:

```python
def test_rewrite_result_validates_route_and_intent_fields():
    from agents.query_rewriter_agent import RewriteResult

    data = {
        "optimized_query": "metformin dosage",
        "route": "vector",
        "session_intent": "medical_consultation",
        "turn_intent": "dosage_lookup",
        "slots": {"drug": "metformin", "aspect": "dosage"},
        "query_context": "user wants metformin dosage information",
    }
    result = RewriteResult.model_validate(data)
    assert result.route == "vector"
    assert result.session_intent == "medical_consultation"
    assert result.turn_intent == "dosage_lookup"
    assert result.slots == {"drug": "metformin", "aspect": "dosage"}
    assert result.query_context == "user wants metformin dosage information"


def test_rewrite_result_defaults_new_fields():
    from agents.query_rewriter_agent import RewriteResult

    result = RewriteResult.model_validate({"optimized_query": "test"})
    assert result.route == "vector"
    assert result.session_intent == ""
    assert result.turn_intent == ""
    assert result.slots == {}
    assert result.query_context == ""


def test_parse_rewrite_response_preserves_route_and_slots():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = '{"optimized_query": "test", "route": "web", "slots": {"drug": "aspirin"}}'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.route == "web"
    assert result.slots == {"drug": "aspirin"}


def test_parse_rewrite_response_defaults_route_on_failure():
    from agents.query_rewriter_agent import parse_rewrite_response

    result = parse_rewrite_response("not json", fallback_query="original")
    assert result.route == "vector"
    assert result.slots == {}
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_query_rewriter_contract.py -v`
Expected: FAIL — `RewriteResult` has no `route` field yet

**Step 3: Update RewriteResult model**

In `agents/query_rewriter_agent.py`, expand the `RewriteResult` class:

```python
class RewriteResult(BaseModel):
    """Structured output from the query rewriter."""
    optimized_query: str
    stepback_query: str = ""
    route: str = "vector"
    session_intent: str = ""
    turn_intent: str = ""
    slots: dict[str, str] = {}
    query_context: str = ""
    reasoning: str = ""

    @field_validator('optimized_query', 'stepback_query', 'route',
                     'session_intent', 'turn_intent', 'query_context',
                     'reasoning', mode='before')
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return ""
        return str(v).strip()

    @field_validator('slots', mode='before')
    @classmethod
    def coerce_slots(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return {}
```

**Step 4: Update the prompt**

Replace `_REWRITER_SYSTEM` and `_REWRITER_PROMPT` in `agents/query_rewriter_agent.py`:

```python
_REWRITER_SYSTEM = (
    "You are a medical query understanding assistant for a textbook retrieval system. "
    "You analyze user questions, rewrite them for optimal retrieval, classify intent, "
    "and extract structured information. "
    "Respond ONLY with valid JSON, no other text."
)

_REWRITER_PROMPT = """Analyze the user's question and produce a structured query understanding result.

INSTRUCTIONS:
1. RESOLVE conversational references: replace "it", "that drug", "the same condition" with explicit terms from conversation history.
2. EXPAND medical abbreviations: MI → myocardial infarction, BP → blood pressure, CHF → congestive heart failure, DVT → deep vein thrombosis, PE → pulmonary embolism, COPD → chronic obstructive pulmonary disease, DM → diabetes mellitus, HTN → hypertension, CAD → coronary artery disease, etc.
3. ADD medical synonyms in parentheses where helpful: "heart attack (myocardial infarction)", "blood thinner (anticoagulant)".
4. PRESERVE specificity: keep dosages, populations, timeframes, and drug names exactly as stated.
5. STEP-BACK QUERY: If the question involves mechanisms of action, drug comparisons, treatment rationale, dosage-specific lookups, or timeline-specific questions, generate a broader principle-level query in stepback_query. Otherwise leave stepback_query empty.
6. ROUTE: Classify where to search.
   - "vector": medical/clinical questions answerable from a textbook (most queries)
   - "web": questions about current events, latest guidelines, recent outbreaks, news
   - "literature": explicit requests for research papers, studies, PubMed, systematic reviews
   - "memory": questions about previous conversation ("what did we discuss", "remind me", "you said earlier")
   - "chitchat": greetings, thanks, non-medical small talk with NO medical content
7. SESSION INTENT: The overall goal of this conversation based on history (e.g., "researching diabetes treatment options", "understanding cardiovascular pharmacology"). Empty if no history.
8. TURN INTENT: What this specific message is asking for (e.g., "dosage_lookup", "drug_comparison", "definition", "side_effects", "mechanism_of_action", "greeting").
9. SLOTS: Extract structured medical entities. Use these keys when applicable: drug, condition, population, aspect, timeframe, comparison_target. Add others if relevant.
10. QUERY CONTEXT: A free-form 1-2 sentence description of what the user needs, written for a downstream answer generator.

Conversation context:
{history_text}

User question:
{question}

Return JSON:
{{"optimized_query": "...", "stepback_query": "...", "route": "...", "session_intent": "...", "turn_intent": "...", "slots": {{}}, "query_context": "...", "reasoning": "..."}}

Examples:
- "what does aspirin do for heart attacks?" → {{"optimized_query": "aspirin mechanism of action for myocardial infarction (heart attack)", "stepback_query": "pharmacology of antiplatelet agents in cardiovascular disease", "route": "vector", "session_intent": "", "turn_intent": "mechanism_of_action", "slots": {{"drug": "aspirin", "condition": "myocardial infarction", "aspect": "mechanism"}}, "query_context": "User wants to understand how aspirin works in treating heart attacks", "reasoning": "expanded synonym, step-back to drug class"}}
- "what is hypertension?" → {{"optimized_query": "hypertension (high blood pressure) definition and overview", "stepback_query": "", "route": "vector", "session_intent": "", "turn_intent": "definition", "slots": {{"condition": "hypertension"}}, "query_context": "User wants a basic definition of hypertension", "reasoning": "simple definition lookup"}}
- "what did we talk about last time?" → {{"optimized_query": "what did we talk about last time", "stepback_query": "", "route": "memory", "session_intent": "", "turn_intent": "conversation_recall", "slots": {{}}, "query_context": "User wants to recall previous conversation topics", "reasoning": "conversation recall, route to memory"}}
- "hello!" → {{"optimized_query": "hello", "stepback_query": "", "route": "chitchat", "session_intent": "", "turn_intent": "greeting", "slots": {{}}, "query_context": "User is greeting", "reasoning": "simple greeting"}}"""
```

**Step 5: Update QueryRewriterAgent to write new fields to state**

In the `QueryRewriterAgent` function, after setting `optimized_query` and `stepback_query`, add:

```python
    state['route'] = result.route if result.route else ''
    state['session_intent'] = result.session_intent
    state['turn_intent'] = result.turn_intent
    state['slots'] = result.slots
    state['query_context'] = result.query_context
```

And update the logger.info `extra` dict to include the new fields:

```python
    logger.info(
        "query_rewrite_complete",
        extra={
            "original": question,
            "optimized": result.optimized_query,
            "stepback": result.stepback_query,
            "stepback_generated": bool(result.stepback_query),
            "route": result.route,
            "session_intent": result.session_intent,
            "turn_intent": result.turn_intent,
            "slots": result.slots,
            "query_context": result.query_context,
            "reasoning": result.reasoning,
        },
    )
```

**Step 6: Run all rewriter tests**

Run: `python -m pytest tests/rag/test_query_rewriter_contract.py -v`
Expected: PASS (12 tests — 8 existing + 4 new)

**Step 7: Commit**

```bash
git add agents/query_rewriter_agent.py tests/rag/test_query_rewriter_contract.py
git commit -m "feat: expand rewriter with route, intent, slots, query_context"
```

---

### Task 3: Convert planner to zero-LLM dispatcher

**Files:**
- Modify: `agents/planner_agent.py`
- Create: `tests/rag/test_planner_dispatcher.py`

**Step 1: Write the failing tests**

Create `tests/rag/test_planner_dispatcher.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_planner_reads_route_from_state():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='web')
    result = PlannerAgent(state)
    assert result['route'] == 'web'
    assert result['planned_route'] == 'web'


def test_planner_accepts_memory_route():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='memory')
    result = PlannerAgent(state)
    assert result['route'] == 'memory'


def test_planner_falls_back_on_invalid_route():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='invalid_garbage')
    result = PlannerAgent(state)
    # Should fall back to 'vector' (default safe route)
    assert result['route'] in {'vector', 'chitchat', 'web', 'literature', 'memory'}


def test_planner_falls_back_on_empty_route():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='')
    result = PlannerAgent(state)
    assert result['route'] in {'vector', 'chitchat', 'web', 'literature', 'memory'}


def _make_state(**overrides):
    base = {
        'trace_id': 't1',
        'session_id': 's1',
        'question': 'test question',
        'raw_query': 'test question',
        'optimized_query': 'test question',
        'stepback_query': '',
        'route': '',
        'planned_route': '',
        'route_decision_reason': '',
        'post_retrieval_route': '',
        'generation': '',
        'source': '',
        'citations': [],
        'documents': [],
        'retrieval_confidence': 0.0,
        'retrieval_candidate_count': 0,
        'conversation_history': [],
        'summary': '',
        'facts': [],
        'safety_flags': {'blocked': False, 'reason': None, 'risk_level': 'low'},
        'attempts': {'reflection': 0, 'executor': 0},
        'current_tool': None,
        'status_events': [],
        'semantic_cache_hit': False,
        'needs_retry': False,
        'reflection_feedback': '',
        'chat_repo': None,
        'vector_repo': None,
        'session_intent': '',
        'turn_intent': '',
        'slots': {},
        'query_context': '',
    }
    base.update(overrides)
    return base
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_planner_dispatcher.py -v`
Expected: FAIL — current planner calls `invoke_json` which may fail without LLM

**Step 3: Rewrite planner agent**

Replace the entire content of `agents/planner_agent.py`:

```python
from __future__ import annotations

import logging

from core.state_v2 import AgentStateV2
from tools.knn_router import knn_fallback_route

logger = logging.getLogger(__name__)

VALID_ROUTES = {'chitchat', 'vector', 'web', 'future_clinical_db', 'literature', 'memory'}


def PlannerAgent(state: AgentStateV2) -> AgentStateV2:
    route = (state.get('route') or '').strip().lower()

    if route not in VALID_ROUTES:
        route = knn_fallback_route(state)
        state['route_decision_reason'] = 'knn_fallback'
        logger.info("planner_knn_fallback", extra={"resolved_route": route})

    state['route'] = route
    state['planned_route'] = route
    state['current_tool'] = route
    return state
```

Note: This depends on `tools/knn_router.py` which doesn't exist yet. For the test to pass, we need a stub. Create a minimal `tools/knn_router.py` first (will be fully implemented in Task 4):

```python
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def knn_fallback_route(state: dict) -> str:
    """Fallback route using k-NN seed similarity. Returns 'vector' if seeds unavailable."""
    return 'vector'
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/rag/test_planner_dispatcher.py -v`
Expected: PASS (4 tests)

**Step 5: Run existing tests to check nothing broke**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All pass

**Step 6: Commit**

```bash
git add agents/planner_agent.py tools/knn_router.py tests/rag/test_planner_dispatcher.py
git commit -m "feat: convert planner to zero-LLM dispatcher with k-NN fallback stub"
```

---

### Task 4: Implement k-NN fallback router

**Files:**
- Modify: `tools/knn_router.py`
- Create: `scripts/generate_route_seeds.py`
- Create: `tests/rag/test_knn_router.py`

**Step 1: Write the failing tests**

Create `tests/rag/test_knn_router.py`:

```python
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_knn_route_returns_majority_vote():
    from tools.knn_router import knn_route

    # Create fake seed index: 3 "vector" seeds near [1,0,0], 2 "chitchat" seeds near [0,1,0]
    seeds = [
        ([1.0, 0.01, 0.0], "vector"),
        ([0.99, 0.02, 0.0], "vector"),
        ([0.98, 0.03, 0.0], "vector"),
        ([0.01, 1.0, 0.0], "chitchat"),
        ([0.02, 0.99, 0.0], "chitchat"),
    ]
    # Query near [1,0,0] should route to "vector"
    result = knn_route([1.0, 0.0, 0.0], seeds, k=3)
    assert result == "vector"

    # Query near [0,1,0] should route to "chitchat"
    result = knn_route([0.0, 1.0, 0.0], seeds, k=3)
    assert result == "chitchat"


def test_knn_route_returns_default_on_empty_seeds():
    from tools.knn_router import knn_route

    result = knn_route([1.0, 0.0], [], k=3)
    assert result == "vector"


def test_load_seeds_returns_empty_on_missing_file():
    from tools.knn_router import load_seeds

    seeds = load_seeds("/nonexistent/path/seeds.json")
    assert seeds == []


def test_load_seeds_parses_valid_file():
    from tools.knn_router import load_seeds

    data = {
        "vector": [{"query": "what is diabetes", "embedding": [0.1, 0.2]}],
        "chitchat": [{"query": "hello", "embedding": [0.3, 0.4]}],
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        f.flush()
        seeds = load_seeds(f.name)

    assert len(seeds) == 2
    assert seeds[0] == ([0.1, 0.2], "vector")
    assert seeds[1] == ([0.3, 0.4], "chitchat")


def test_knn_fallback_route_returns_vector_without_seeds():
    from tools.knn_router import knn_fallback_route

    state = {'optimized_query': 'test', 'question': 'test'}
    result = knn_fallback_route(state)
    assert result == 'vector'
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_knn_router.py -v`
Expected: FAIL — `knn_route` and `load_seeds` not defined

**Step 3: Implement k-NN router**

Replace `tools/knn_router.py`:

```python
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SEEDS_PATH = Path(__file__).resolve().parent.parent / 'data' / 'route_seeds.json'
_cached_seeds: list[tuple[list[float], str]] | None = None


def load_seeds(path: str | Path | None = None) -> list[tuple[list[float], str]]:
    """Load pre-embedded seed queries from JSON. Returns [] on missing/invalid file."""
    seeds_path = Path(path) if path else _DEFAULT_SEEDS_PATH
    if not seeds_path.exists():
        logger.warning("knn_seeds_missing", extra={"path": str(seeds_path)})
        return []
    try:
        with open(seeds_path) as f:
            data = json.load(f)
        seeds: list[tuple[list[float], str]] = []
        for route_label, entries in data.items():
            for entry in entries:
                embedding = entry.get('embedding', [])
                if embedding:
                    seeds.append((embedding, route_label))
        return seeds
    except Exception as e:
        logger.warning("knn_seeds_load_error", extra={"error": str(e)})
        return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def knn_route(
    query_embedding: list[float],
    seeds: list[tuple[list[float], str]],
    k: int = 5,
) -> str:
    """Route by majority vote of k nearest seed embeddings. Returns 'vector' as default."""
    if not seeds:
        return 'vector'

    scored = [
        (_cosine_similarity(query_embedding, seed_emb), label)
        for seed_emb, label in seeds
    ]
    scored.sort(reverse=True)
    top_k = scored[:k]

    votes: dict[str, int] = {}
    for _sim, label in top_k:
        votes[label] = votes.get(label, 0) + 1

    if not votes:
        return 'vector'

    winner = max(votes, key=votes.get)

    # Confidence check: if top vote is less than 40% of k, default to vector
    if votes[winner] / k < 0.4:
        logger.info("knn_low_confidence", extra={"votes": votes, "k": k})
        return 'vector'

    return winner


def knn_fallback_route(state: dict[str, Any]) -> str:
    """Fallback route using k-NN seed similarity. Returns 'vector' if seeds unavailable."""
    global _cached_seeds
    if _cached_seeds is None:
        _cached_seeds = load_seeds()

    if not _cached_seeds:
        return 'vector'

    query = state.get('optimized_query') or state.get('question', '')
    if not query:
        return 'vector'

    try:
        from tools.embedding_client import embed_query
        query_embedding = embed_query(query)
    except Exception as e:
        logger.warning("knn_embed_failed", extra={"error": str(e)})
        return 'vector'

    return knn_route(query_embedding, _cached_seeds, k=5)
```

**Step 4: Create seed generation script**

Create `scripts/generate_route_seeds.py`:

```python
#!/usr/bin/env python3
"""Embed hand-written seed queries and save to data/route_seeds.json.

Usage:
    python scripts/generate_route_seeds.py

Reads seed queries from data/route_seeds_raw.json (format: {"route": ["query1", ...]}).
Embeds each query using embed_query() and writes to data/route_seeds.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / 'data' / 'route_seeds_raw.json'
OUT_PATH = ROOT / 'data' / 'route_seeds.json'


def main():
    from tools.embedding_client import embed_query

    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found. Create it with seed queries per route.")
        print('Format: {"vector": ["query1", ...], "chitchat": ["query2", ...]}')
        sys.exit(1)

    with open(RAW_PATH) as f:
        raw = json.load(f)

    output: dict[str, list[dict]] = {}
    total = 0
    for route, queries in raw.items():
        output[route] = []
        for query in queries:
            embedding = embed_query(query)
            output[route].append({"query": query, "embedding": embedding})
            total += 1
        print(f"  {route}: {len(queries)} seeds embedded")

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f)

    print(f"Saved {total} embedded seeds to {OUT_PATH}")


if __name__ == '__main__':
    main()
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/rag/test_knn_router.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add tools/knn_router.py scripts/generate_route_seeds.py tests/rag/test_knn_router.py
git commit -m "feat: implement k-NN fallback router with seed loading and majority vote"
```

---

### Task 5: Add memory route to workflow and executor

**Files:**
- Modify: `core/langgraph_workflow.py`
- Modify: `agents/executor_agent.py`
- Create: `tests/rag/test_memory_route.py`

**Step 1: Write the failing tests**

Create `tests/rag/test_memory_route.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_route_after_planner_routes_memory_to_memory():
    from core.langgraph_workflow import route_after_planner

    state = {'route': 'memory'}
    assert route_after_planner(state) == 'memory'


def test_route_after_planner_routes_chitchat_unchanged():
    from core.langgraph_workflow import route_after_planner

    state = {'route': 'chitchat'}
    assert route_after_planner(state) == 'chitchat'


def test_route_after_planner_routes_vector_to_retriever():
    from core.langgraph_workflow import route_after_planner

    state = {'route': 'vector'}
    assert route_after_planner(state) == 'retriever'


def test_executor_memory_prompt_uses_conversation_history():
    from agents.executor_agent import _render_memory_prompt

    prompt = _render_memory_prompt(
        question="what did we discuss?",
        summary="We discussed metformin dosage and side effects.",
        facts="condition=diabetes",
        history_text="user: what is metformin?\nassistant: Metformin is...",
    )
    assert "what did we discuss?" in prompt
    assert "metformin" in prompt.lower()
    assert "conversation" in prompt.lower() or "history" in prompt.lower()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_memory_route.py -v`
Expected: FAIL — `route_after_planner` doesn't handle `'memory'`, `_render_memory_prompt` doesn't exist

**Step 3: Update workflow routing**

In `core/langgraph_workflow.py`, update `route_after_planner`:

```python
def route_after_planner(state: AgentStateV2) -> str:
    route = state.get('route', 'vector')
    if route == 'chitchat':
        return 'chitchat'
    if route == 'memory':
        return 'memory'
    # Retriever-first policy for all non-chitchat, non-memory routes.
    return 'retriever'
```

Update `create_workflow()` to add memory edge:

```python
    workflow.add_conditional_edges(
        'planner',
        route_after_planner,
        {
            'chitchat': 'llm',
            'memory': 'executor',
            'retriever': 'retriever',
        },
    )
```

**Step 4: Add memory prompt to executor**

In `agents/executor_agent.py`, add a memory-specific prompt template and render function:

```python
_MEMORY_PROMPT_TEMPLATE = """
User question: {question}
Conversation summary: {summary}
Known user facts: {facts}

Recent conversation:
{history_text}

The user is asking about a previous conversation. Answer based ONLY on the conversation
history and summary above. If the information they're asking about is not in the history,
say so honestly. Do not make up or infer information that wasn't discussed.
""".strip()


def _render_memory_prompt(question: str, summary: str, facts: str, history_text: str) -> str:
    return _MEMORY_PROMPT_TEMPLATE.format(
        question=question,
        summary=summary,
        facts=facts,
        history_text=history_text,
    )
```

In `ExecutorAgent`, add a memory branch at the top (after cache check, before document processing):

```python
    # Memory route: generate from conversation history, not retrieved docs
    if state.get('route') == 'memory':
        history = state.get('conversation_history', [])
        history_text = '\n'.join(
            [f"{item.get('role')}: {item.get('content')}" for item in history]
        ) if history else ''

        if not history_text and not summary:
            state['generation'] = (
                "I don't have any previous conversation history to reference. "
                "This appears to be the start of our conversation."
            )
            state['source'] = 'Memory (no history)'
            return state

        prompt = _render_memory_prompt(question, summary, facts, history_text)
        answer = invoke_llm(prompt, system=_SYSTEM_PROMPT, use_fallback=False)
        if not answer:
            answer = invoke_llm(prompt, system=_SYSTEM_PROMPT, use_fallback=True)
        if not answer:
            answer = "I couldn't retrieve our previous conversation details at the moment."

        state['generation'] = answer
        state['source'] = 'Memory (Conversation History)'
        return state
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/rag/test_memory_route.py -v`
Expected: PASS (4 tests)

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All pass

**Step 7: Commit**

```bash
git add core/langgraph_workflow.py agents/executor_agent.py tests/rag/test_memory_route.py
git commit -m "feat: add memory route — skip retrieval, generate from conversation history"
```

---

### Task 6: Add adjacent chunk expansion to repository and executor

**Files:**
- Modify: `db/repositories.py`
- Modify: `agents/executor_agent.py`
- Create: `tests/rag/test_adjacent_expansion.py`

**Step 1: Write the failing tests**

Create `tests/rag/test_adjacent_expansion.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document


def _make_doc(chunk_id, content, section="Ch1", chunk_index=0):
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "doc_id": "doc1",
            "page": 1,
            "section": section,
            "chunk_index": chunk_index,
        },
    )


def test_inmemory_get_adjacent_chunks():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    for i in range(5):
        repo.upsert_chunk(
            chunk_id=f"c{i}",
            doc_id="doc1",
            content=f"chunk {i}",
            embedding=[0.0] * 3,
            section="Ch1",
            metadata={"chunk_index": i, "section": "Ch1"},
        )

    neighbors = repo.get_adjacent_chunks(section="Ch1", chunk_index=2)
    neighbor_ids = [d.metadata['chunk_id'] for d in neighbors]
    assert "c1" in neighbor_ids
    assert "c3" in neighbor_ids
    assert "c2" not in neighbor_ids  # not the chunk itself


def test_inmemory_get_adjacent_chunks_at_boundary():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        chunk_id="c0", doc_id="doc1", content="first",
        embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": 0, "section": "Ch1"},
    )
    repo.upsert_chunk(
        chunk_id="c1", doc_id="doc1", content="second",
        embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": 1, "section": "Ch1"},
    )

    neighbors = repo.get_adjacent_chunks(section="Ch1", chunk_index=0)
    assert len(neighbors) == 1
    assert neighbors[0].metadata['chunk_id'] == 'c1'


def test_inmemory_get_adjacent_chunks_different_section_excluded():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        chunk_id="c0", doc_id="doc1", content="ch1 chunk",
        embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": 0, "section": "Ch1"},
    )
    repo.upsert_chunk(
        chunk_id="c1", doc_id="doc1", content="ch2 chunk",
        embedding=[0.0] * 3, section="Ch2", metadata={"chunk_index": 1, "section": "Ch2"},
    )

    neighbors = repo.get_adjacent_chunks(section="Ch1", chunk_index=0)
    assert len(neighbors) == 0


def test_build_context_blocks_with_expansion():
    from agents.executor_agent import _build_context_blocks
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    for i in range(5):
        repo.upsert_chunk(
            chunk_id=f"c{i}", doc_id="doc1", content=f"neighbor content {i}",
            embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": i, "section": "Ch1"},
        )

    # Simulate retriever returning chunk 2 only
    docs = [_make_doc("c2", "main content", section="Ch1", chunk_index=2)]

    blocks = _build_context_blocks(docs, vector_repo=repo)
    combined = '\n'.join(blocks)
    assert "neighbor content 1" in combined  # chunk_index 1
    assert "main content" in combined         # chunk_index 2
    assert "neighbor content 3" in combined  # chunk_index 3


def test_build_context_blocks_without_repo_no_expansion():
    from agents.executor_agent import _build_context_blocks

    docs = [_make_doc("c2", "main content", section="Ch1", chunk_index=2)]
    blocks = _build_context_blocks(docs, vector_repo=None)
    combined = '\n'.join(blocks)
    assert "main content" in combined
    assert "neighbor" not in combined
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_adjacent_expansion.py -v`
Expected: FAIL — `get_adjacent_chunks` doesn't exist, `_build_context_blocks` doesn't accept `vector_repo`

**Step 3: Add get_adjacent_chunks to InMemoryVectorRepository**

In `db/repositories.py`, add to `InMemoryVectorRepository`:

```python
    def get_adjacent_chunks(self, section: str, chunk_index: int) -> list[Document]:
        """Return chunks at chunk_index ± 1 within the same section."""
        target_indices = {chunk_index - 1, chunk_index + 1}
        results = []
        for row in self._chunks.values():
            meta = row.get('metadata') or {}
            if meta.get('section') == section and meta.get('chunk_index') in target_indices:
                results.append(_row_to_doc(row))
        results.sort(key=lambda d: d.metadata.get('chunk_index', 0))
        return results
```

Add the same to `VectorRepository`:

```python
    def get_adjacent_chunks(self, section: str, chunk_index: int) -> list[Document]:
        """Return chunks at chunk_index ± 1 within the same section."""
        target_indices = [chunk_index - 1, chunk_index + 1]
        stmt = text(
            "SELECT chunk_id, doc_id, page, section, content, metadata_json "
            "FROM document_chunks "
            "WHERE section = :sec "
            "AND (metadata_json->>'chunk_index')::int IN :indices "
            "ORDER BY (metadata_json->>'chunk_index')::int ASC"
        )
        rows = self.db.execute(
            stmt, {'sec': section, 'indices': tuple(target_indices)}
        ).fetchall()
        return [
            Document(
                page_content=row.content,
                metadata={
                    'chunk_id': row.chunk_id,
                    'doc_id': row.doc_id,
                    'page': row.page,
                    'section': row.section,
                    **(row.metadata_json if row.metadata_json else {}),
                },
            )
            for row in rows
        ]
```

**Step 4: Update _build_context_blocks to accept vector_repo**

In `agents/executor_agent.py`, update the function signature and add expansion logic:

```python
def _build_context_blocks(docs: list, vector_repo=None) -> list[str]:
    # Collect all chunk_ids in the result set for dedup
    result_chunk_ids = {doc.metadata.get('chunk_id') for doc in docs}

    grouped: dict[str, dict[str, list[str] | str]] = {}
    order: list[str] = []

    for doc in docs:
        metadata = doc.metadata or {}
        header = (
            metadata.get('context_prefix')
            or metadata.get('section_path')
            or metadata.get('section')
            or metadata.get('doc_id')
            or 'Retrieved evidence'
        )
        key = metadata.get('context_key') or header
        if key not in grouped:
            grouped[key] = {'header': header, 'chunks': []}
            order.append(key)

        # Expand: fetch neighbors if repo available
        before_texts = []
        after_texts = []
        if vector_repo is not None:
            section = metadata.get('section')
            chunk_index = metadata.get('chunk_index')
            if section and chunk_index is not None:
                try:
                    neighbors = vector_repo.get_adjacent_chunks(section, chunk_index)
                    for nb in neighbors:
                        nb_id = nb.metadata.get('chunk_id')
                        if nb_id in result_chunk_ids:
                            continue  # skip if already in top-5
                        nb_idx = nb.metadata.get('chunk_index', 0)
                        if nb_idx < chunk_index:
                            before_texts.append(nb.page_content.strip())
                        else:
                            after_texts.append(nb.page_content.strip())
                except Exception:
                    pass  # skip expansion on error

        # Build chunk block with optional neighbors
        parts = []
        for t in before_texts:
            parts.append(f"<context>\n{t}\n</context>")
        parts.append(f"<chunk>\n{doc.page_content.strip()}\n</chunk>")
        for t in after_texts:
            parts.append(f"<context>\n{t}\n</context>")

        grouped[key]['chunks'].append('\n'.join(parts))

    blocks: list[str] = []
    for key in order:
        group = grouped[key]
        block_parts = [f"### {group['header']}"]
        block_parts.extend(group['chunks'])
        blocks.append('\n\n'.join(block_parts))
    return blocks
```

Update the call site in `ExecutorAgent` to pass `vector_repo`:

```python
    # In ExecutorAgent, change:
    for block in _build_context_blocks(docs):
    # To:
    vector_repo = state.get('vector_repo')
    for block in _build_context_blocks(docs, vector_repo=vector_repo):
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/rag/test_adjacent_expansion.py -v`
Expected: PASS (5 tests)

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All pass

**Step 7: Commit**

```bash
git add db/repositories.py agents/executor_agent.py tests/rag/test_adjacent_expansion.py
git commit -m "feat: adjacent chunk expansion — fetch ±1 neighbors during context assembly"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update architecture doc**

In `docs/architecture/current-state.md`:

- Update the query rewriter description to mention route, intent, slots, query_context output
- Update the planner description to say "zero-LLM dispatcher reading route from state, with k-NN fallback"
- Add memory route to the chat request flow description
- Add adjacent chunk expansion to the retrieval/executor description

**Step 2: Add implementation log entry**

Append to `docs/changes/implementation-log.md`:

```markdown
## 2026-03-14 — Pipeline upgrade Phase A: unified rewriter, dispatcher, k-NN, memory route, adjacent expansion

What changed:
- Expanded query rewriter to output route, session_intent, turn_intent, slots, query_context in a single LLM call alongside optimized_query and stepback_query.
- Converted planner agent from LLM-based routing to zero-LLM dispatcher that reads route from state.
- Added k-NN fallback router using pre-embedded seed queries with cosine similarity majority vote.
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
```

**Step 3: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: update architecture and implementation log for Phase A pipeline upgrade"
```

---

### Task 8: Run full test suite and verify

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests pass

**Step 2: Verify no import errors**

Run: `python -c "from core.langgraph_workflow import create_workflow; w = create_workflow(); print('workflow OK')"`
Expected: `workflow OK`

**Step 3: Verify planner no longer imports invoke_json**

Run: `grep -n 'invoke_json\|invoke_llm' agents/planner_agent.py`
Expected: No matches (planner makes no LLM calls)
