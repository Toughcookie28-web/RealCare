# HITL Proactive Clarification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When the query rewriter detects low intent confidence or competing route hypotheses, ask the user a clarifying question instead of guessing and generating a wrong answer.

**Architecture:** Add `needs_clarification: bool` and `clarification_question: str` to `RewriteResult`. When `needs_clarification=True`, add a `clarify` route that short-circuits directly to `ExecutorAgent` (which generates the clarification question as the response, no retrieval). This is purely conversational — no approval queue, no DB storage. The next user message re-enters the pipeline normally with the clarified query. Controlled by `HITL_CLARIFICATION_ENABLED` setting (default True) and `HITL_CLARIFICATION_CONFIDENCE_THRESHOLD` (default 0.4).

**Tech Stack:** Python, Pydantic, LangGraph, existing `invoke_json`

**Note:** The old HITL approval queue (core/hitl.py, api/routes/hitl.py) should be deleted before or alongside this plan (see `2026-03-17-delete-old-hitl.md`).

---

### Task 1: Add clarification settings

**Files:**
- Modify: `core/settings.py`
- Modify: `tests/core/test_settings_contract.py`

**Step 1: Write the failing test**

```python
def test_clarification_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.hitl_clarification_enabled is True
    assert s.hitl_clarification_confidence_threshold == 0.4
```

**Step 2: Run to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v -k "clarification" 2>&1`
Expected: FAIL

**Step 3: Add settings**

In `core/settings.py`, add after `enable_query_decomposition`:
```python
hitl_clarification_enabled: bool = Field(default=True, alias='HITL_CLARIFICATION_ENABLED')
hitl_clarification_confidence_threshold: float = Field(default=0.4, alias='HITL_CLARIFICATION_CONFIDENCE_THRESHOLD')
```

**Step 4: Verify**

Run: `pytest tests/core/test_settings_contract.py -v -k "clarification" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings_contract.py
git commit -m "feat: add HITL clarification settings"
```

---

### Task 2: Add clarification fields to RewriteResult and QueryRewriterAgent

**Files:**
- Modify: `agents/query_rewriter_agent.py`

**Step 1: Write failing tests**

Create `tests/agents/test_clarification.py`:
```python
def test_rewrite_result_has_clarification_fields():
    from agents.query_rewriter_agent import RewriteResult
    r = RewriteResult(optimized_query="test")
    assert r.needs_clarification is False
    assert r.clarification_question == ""
    assert r.intent_confidence == 1.0


def test_rewrite_result_with_clarification():
    from agents.query_rewriter_agent import RewriteResult
    r = RewriteResult(
        optimized_query="aspirin dosage",
        needs_clarification=True,
        clarification_question="Are you asking about adult or pediatric dosage?",
        intent_confidence=0.3,
    )
    assert r.needs_clarification is True
    assert "dosage" in r.clarification_question


def test_parse_rewrite_response_handles_clarification():
    from agents.query_rewriter_agent import parse_rewrite_response
    raw = """{
        "optimized_query": "aspirin dosage",
        "route": "vector",
        "needs_clarification": true,
        "clarification_question": "Are you asking about aspirin for pain or heart disease?",
        "intent_confidence": 0.3,
        "stepback_query": "", "session_intent": "", "turn_intent": "", "slots": {}, "query_context": ""
    }"""
    result = parse_rewrite_response(raw, fallback_query="aspirin dosage")
    assert result.needs_clarification is True
    assert result.intent_confidence == 0.3
```

**Step 2: Run to verify they fail**

Run: `pytest tests/agents/test_clarification.py -v 2>&1`
Expected: FAIL — fields not in RewriteResult

**Step 3: Add fields to RewriteResult**

In `agents/query_rewriter_agent.py`, update `RewriteResult`:
```python
class RewriteResult(BaseModel):
    optimized_query: str
    stepback_query: str = ""
    route: str = "vector"
    session_intent: str = ""
    turn_intent: str = ""
    slots: dict[str, str] = {}
    query_context: str = ""
    reasoning: str = ""
    intent_confidence: float = 1.0          # ADD
    needs_clarification: bool = False       # ADD
    clarification_question: str = ""        # ADD
```

Add validators for the new fields:
```python
@field_validator('intent_confidence', mode='before')
@classmethod
def coerce_confidence(cls, v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 1.0

@field_validator('needs_clarification', mode='before')
@classmethod
def coerce_needs_clarification(cls, v):
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    return bool(v)

@field_validator('clarification_question', mode='before')
@classmethod
def coerce_clarification_question(cls, v):
    return str(v or '').strip()
```

**Step 4: Update the rewriter prompt**

In `_REWRITER_PROMPT`, add at the end of STEP 2 rules section:
```
8. INTENT CONFIDENCE: 0.0-1.0. How confident are you in your route/intent classification?
   - <0.4: ambiguous — user message could plausibly mean multiple very different things
   - 0.4-0.7: reasonable confidence but some ambiguity
   - >0.7: clear intent
9. NEEDS CLARIFICATION: true only if intent_confidence < 0.4 AND you cannot reasonably guess
   the correct route from context. Do NOT set true for follow-up questions where prior
   conversation provides enough context.
10. CLARIFICATION QUESTION: If needs_clarification is true, write a short, friendly question
    that would resolve the ambiguity. E.g., "Are you asking about X or Y?" Leave empty otherwise.
```

Update the STEP 3 JSON output line to include new fields:
```
{{"reasoning": "...", "optimized_query": "...", "stepback_query": "...", "route": "...",
  "session_intent": "...", "turn_intent": "...", "slots": {{}}, "query_context": "...",
  "intent_confidence": 0.9, "needs_clarification": false, "clarification_question": ""}}
```

**Step 5: Update QueryRewriterAgent to write new fields to state**

In `QueryRewriterAgent`, after existing state assignments, add:
```python
state['intent_confidence'] = result.intent_confidence
state['needs_clarification'] = result.needs_clarification
state['clarification_question'] = result.clarification_question
```

**Step 6: Run tests**

Run: `pytest tests/agents/test_clarification.py -v 2>&1`
Expected: PASS

**Step 7: Commit**

```bash
git add agents/query_rewriter_agent.py tests/agents/test_clarification.py
git commit -m "feat: add intent_confidence and clarification fields to QueryRewriter"
```

---

### Task 3: Add clarification fields to state

**Files:**
- Modify: `core/state_v2.py`

**Step 1: Add fields to AgentStateV2 TypedDict**

```python
intent_confidence: float
needs_clarification: bool
clarification_question: str
```

**Step 2: Add to initialize_state()**

```python
'intent_confidence': 1.0,
'needs_clarification': False,
'clarification_question': '',
```

**Step 3: Add to reset_query_state()**

```python
'intent_confidence': 1.0,
'needs_clarification': False,
'clarification_question': '',
```

**Step 4: Verify**

Run: `python -c "from core.state_v2 import initialize_state; s=initialize_state('s','t'); print(s['needs_clarification'])"`
Expected: `False`

**Step 5: Commit**

```bash
git add core/state_v2.py
git commit -m "feat: add clarification state fields"
```

---

### Task 4: Wire clarify route into LangGraph workflow

**Files:**
- Modify: `core/langgraph_workflow.py`
- Modify: `agents/executor_agent.py`

**Step 1: Write failing test**

Append to `tests/agents/test_clarification.py`:
```python
def test_clarify_route_generates_question_without_retrieval(monkeypatch):
    from core.state_v2 import initialize_state

    monkeypatch.setattr(
        "agents.query_rewriter_agent.invoke_json",
        lambda *a, **kw: {
            "optimized_query": "aspirin",
            "route": "vector",
            "needs_clarification": True,
            "clarification_question": "Are you asking about aspirin for pain or heart disease?",
            "intent_confidence": 0.3,
            "stepback_query": "", "session_intent": "dosage", "turn_intent": "dosage_lookup",
            "slots": {}, "query_context": "",
        },
    )

    from agents.query_rewriter_agent import QueryRewriterAgent
    state = initialize_state("s1", "t1")
    state["question"] = "what should I take"
    result = QueryRewriterAgent(state)

    assert result["needs_clarification"] is True
    assert "pain or heart" in result["clarification_question"]
```

**Step 2: Run to verify**

Run: `pytest tests/agents/test_clarification.py::test_clarify_route_generates_question_without_retrieval -v 2>&1`
Expected: PASS (this tests the rewriter, not the graph yet)

**Step 3: Add clarify route to PlannerAgent**

In `agents/planner_agent.py`, check settings and set route to `clarify` when appropriate:
```python
from core.settings import get_settings

def PlannerAgent(state: AgentStateV2) -> AgentStateV2:
    settings = get_settings()
    # Check if rewriter flagged a clarification need
    if (
        settings.hitl_clarification_enabled
        and state.get('needs_clarification')
        and state.get('intent_confidence', 1.0) < settings.hitl_clarification_confidence_threshold
    ):
        state['route'] = 'clarify'
        state['planned_route'] = 'clarify'
        state['route_decision_reason'] = 'planner_low_confidence_clarify'
        return state
    # ... rest of existing PlannerAgent logic ...
```

**Step 4: Update ExecutorAgent to handle clarify route**

In `agents/executor_agent.py`, add a clarify route handler before the memory route handler:
```python
# Clarify route: return the clarification question as the response
if state.get('route') == 'clarify':
    clarification_q = state.get('clarification_question', '')
    if clarification_q:
        state['generation'] = clarification_q
    else:
        state['generation'] = (
            "Could you clarify what you're looking for? "
            "Your question could be interpreted in a few different ways."
        )
    state['source'] = 'Clarification Request'
    return state
```

**Step 5: Update route_after_planner in langgraph_workflow.py**

```python
def route_after_planner(state: AgentStateV2) -> str:
    route = state.get('route', 'vector')
    if route == 'chitchat':
        return 'chitchat'
    if route == 'memory':
        return 'memory'
    if route == 'clarify':          # ADD
        return 'executor'           # ADD
    return 'retriever'
```

**Step 6: Run full test suite**

Run: `pytest tests/agents/test_clarification.py tests/core/test_settings_contract.py -v 2>&1`
Expected: all PASS

**Step 7: Commit**

```bash
git add core/langgraph_workflow.py agents/executor_agent.py agents/planner_agent.py
git commit -m "feat: wire clarify route into workflow — rewriter triggers clarification on low-confidence intent"
```

---

### Task 5: Update docs

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`
- Add: `docs/decisions/ADR-0005-hitl-proactive-clarification.md`

ADR key points:
- **Decision:** Lightweight conversational clarification replaces approval-queue HITL
- **Why:** Old HITL was reviewer-facing (safety queue), not user-facing. Proactive clarification is more useful for ambiguous medical queries where wrong assumptions produce useless answers
- **Trigger:** `intent_confidence < 0.4` AND `needs_clarification=True` from rewriter
- **Mechanism:** `route='clarify'` → executor returns clarification question → no retrieval, no DB storage

Log entry: `2026-03-17 — HITL proactive clarification added. Old approval queue deleted.`
