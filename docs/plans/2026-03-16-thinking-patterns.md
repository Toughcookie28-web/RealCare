# Thinking Patterns Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the reflection→retry loop with structured feedback, add ReAct-style targeted re-retrieval on failure, and stub Plan-and-Execute for future multi-PDF decomposition.

**Architecture:** Three incremental upgrades to the existing LangGraph workflow. Pattern 1 (Reflection) replaces the binary judge with structured actionable feedback. Pattern 2 (ReAct) wires reflection feedback into the retry path so the retriever uses a refined query instead of repeating the same search. Pattern 3 (Plan-and-Execute) adds a disabled decomposer node that splits complex queries into sub-questions — stubbed until multi-PDF ingestion is ready.

**Tech Stack:** LangGraph, Pydantic, existing `invoke_json` LLM client, pytest

---

## Current State

### Reflection (binary judge)
```
executor → reflection → [retry→retriever | finalize→explanation]
```
- `ReflectionAgent` asks LLM: "is_relevant? has_hallucinations? feedback?"
- On failure: sets `needs_retry=True`, sends back to retriever with **same query**
- Max 2 reflection attempts
- Feedback is a free-form string, not structured or actionable

### Problems
1. **Feedback is unstructured** — the retry path doesn't know *what* to fix
2. **Same query on retry** — retriever re-runs the identical search, likely returning identical results
3. **No targeted re-retrieval** — even when the LLM says "the answer didn't address dosage", the system can't focus on dosage

---

## Pattern 1: Structured Reflection (Tasks 1–3)

Upgrade `ReflectionAgent` to return structured, actionable feedback:
- `failure_category`: one of `irrelevant`, `hallucination`, `incomplete`, `unsafe`, `none`
- `suggested_focus`: a specific retrieval hint (e.g., "dosage information for pediatric patients")
- `confidence`: 0.0–1.0 score for the answer quality
- `feedback`: human-readable explanation (kept for logging/debugging)

## Pattern 2: ReAct Follow-Up Retrieval (Tasks 4–6)

Wire reflection feedback into the retry path:
- When `needs_retry=True`, set `optimized_query` to the `suggested_focus` from reflection
- Clear `stepback_query` on retry (the original step-back is stale)
- The retriever then searches for what was actually missing, not the same query

## Pattern 3: Plan-and-Execute Stub (Tasks 7–9)

Add a disabled `DecomposerAgent` that:
- Detects multi-part questions (e.g., "Compare drug A side effects with drug B dosing")
- Splits into sub-questions with individual routing
- Synthesizes sub-answers into a final response
- **Disabled by default** via `ENABLE_QUERY_DECOMPOSITION=false` setting
- Stubbed interface only — the decompose→execute→synthesize loop is not wired into LangGraph yet

---

### Task 1: Structured Reflection — Write the Pydantic model and test

**Files:**
- Create: `tests/eval/test_reflection_structured.py`
- Modify: `agents/reflection_agent.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_reflection_structured.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_parse_reflection_response_extracts_structured_fields():
    from agents.reflection_agent import _parse_reflection_response

    raw = {
        "is_relevant": True,
        "has_hallucinations": False,
        "failure_category": "none",
        "suggested_focus": "",
        "confidence": 0.85,
        "feedback": "Good answer.",
    }
    result = _parse_reflection_response(raw)
    assert result.failure_category == "none"
    assert result.confidence == 0.85
    assert result.suggested_focus == ""
    assert result.is_relevant is True


def test_parse_reflection_response_defaults_on_missing_fields():
    from agents.reflection_agent import _parse_reflection_response

    raw = {"is_relevant": False, "has_hallucinations": True}
    result = _parse_reflection_response(raw)
    assert result.failure_category == "irrelevant"
    assert result.confidence == 0.0
    assert result.suggested_focus == ""


def test_parse_reflection_response_coerces_types():
    from agents.reflection_agent import _parse_reflection_response

    raw = {
        "is_relevant": "true",
        "has_hallucinations": "false",
        "failure_category": "incomplete",
        "suggested_focus": "dosage for children",
        "confidence": "0.7",
        "feedback": "Missing pediatric info.",
    }
    result = _parse_reflection_response(raw)
    assert result.is_relevant is True
    assert result.confidence == 0.7
    assert result.suggested_focus == "dosage for children"
    assert result.failure_category == "incomplete"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_reflection_structured.py -v`
Expected: FAIL with "cannot import name '_parse_reflection_response'"

**Step 3: Write the Pydantic model and parser in `agents/reflection_agent.py`**

Add the following to `agents/reflection_agent.py` (above `ReflectionAgent`):

```python
from pydantic import BaseModel, field_validator


class ReflectionResult(BaseModel):
    """Structured output from the reflection judge."""
    is_relevant: bool = True
    has_hallucinations: bool = False
    failure_category: str = "none"  # none | irrelevant | hallucination | incomplete | unsafe
    suggested_focus: str = ""
    confidence: float = 0.0
    feedback: str = ""

    @field_validator('failure_category', mode='before')
    @classmethod
    def coerce_category(cls, v):
        valid = {'none', 'irrelevant', 'hallucination', 'incomplete', 'unsafe'}
        val = str(v or 'none').strip().lower()
        return val if val in valid else 'none'

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator('is_relevant', 'has_hallucinations', mode='before')
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes')
        return bool(v)

    @field_validator('suggested_focus', 'feedback', mode='before')
    @classmethod
    def coerce_str(cls, v):
        return str(v or '').strip()


def _parse_reflection_response(raw: dict) -> ReflectionResult:
    """Parse LLM judge output into structured ReflectionResult."""
    if not raw:
        return ReflectionResult()

    # Infer failure_category from legacy fields if not provided
    if 'failure_category' not in raw:
        is_rel = raw.get('is_relevant', True)
        has_hal = raw.get('has_hallucinations', False)
        if isinstance(is_rel, str):
            is_rel = is_rel.strip().lower() in ('true', '1', 'yes')
        if isinstance(has_hal, str):
            has_hal = has_hal.strip().lower() in ('true', '1', 'yes')
        if has_hal:
            raw['failure_category'] = 'hallucination'
        elif not is_rel:
            raw['failure_category'] = 'irrelevant'
        else:
            raw['failure_category'] = 'none'

    try:
        return ReflectionResult.model_validate(raw)
    except Exception:
        return ReflectionResult()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_reflection_structured.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/eval/test_reflection_structured.py agents/reflection_agent.py
git commit -m "feat: add structured ReflectionResult model with Pydantic validation"
```

---

### Task 2: Upgrade ReflectionAgent to use structured prompt and output

**Files:**
- Modify: `agents/reflection_agent.py`
- Test: `tests/eval/test_reflection_structured.py`

**Step 1: Write the failing test**

Add to `tests/eval/test_reflection_structured.py`:

```python
from unittest.mock import patch


def test_reflection_agent_sets_structured_state():
    from agents.reflection_agent import ReflectionAgent

    mock_result = {
        "is_relevant": True,
        "has_hallucinations": False,
        "failure_category": "none",
        "suggested_focus": "",
        "confidence": 0.9,
        "feedback": "Accurate and complete.",
    }

    state = {
        "question": "What is the dosage of aspirin?",
        "generation": "Aspirin is typically dosed at 81-325mg daily.",
        "attempts": {"reflection": 0, "executor": 0},
        "needs_retry": False,
        "reflection_feedback": "",
    }

    with patch("agents.reflection_agent.invoke_json", return_value=mock_result):
        result = ReflectionAgent(state)

    assert result["needs_retry"] is False
    assert result["reflection_feedback"] == "Accurate and complete."
    assert result.get("reflection_suggested_focus") == ""
    assert result.get("reflection_confidence") == 0.9
    assert result.get("reflection_failure_category") == "none"


def test_reflection_agent_triggers_retry_with_focus():
    from agents.reflection_agent import ReflectionAgent

    mock_result = {
        "is_relevant": False,
        "has_hallucinations": False,
        "failure_category": "incomplete",
        "suggested_focus": "pediatric aspirin dosing guidelines",
        "confidence": 0.3,
        "feedback": "Answer lacks pediatric dosing information.",
    }

    state = {
        "question": "What is the pediatric dosage of aspirin?",
        "generation": "Aspirin is used for pain relief.",
        "attempts": {"reflection": 0, "executor": 0},
        "needs_retry": False,
        "reflection_feedback": "",
    }

    with patch("agents.reflection_agent.invoke_json", return_value=mock_result):
        result = ReflectionAgent(state)

    assert result["needs_retry"] is True
    assert result["reflection_suggested_focus"] == "pediatric aspirin dosing guidelines"
    assert result["reflection_failure_category"] == "incomplete"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_reflection_structured.py::test_reflection_agent_sets_structured_state -v`
Expected: FAIL (state keys don't exist yet)

**Step 3: Rewrite `ReflectionAgent` to use structured prompt and output**

Replace the entire `ReflectionAgent` function in `agents/reflection_agent.py`:

```python
import logging

logger = logging.getLogger(__name__)

_REFLECTION_SYSTEM = "You are a strict medical QA reviewer. Evaluate answer quality and safety."

_REFLECTION_PROMPT = """Evaluate this medical Q&A pair.

Question: {question}
Answer: {answer}

Judge the response on these criteria:
1. Is the answer relevant to the question?
2. Does it contain hallucinations (claims not supported by typical medical knowledge)?
3. Is it complete enough to be useful?
4. Is it safe (no dangerous unsupported medical advice)?

Return JSON ONLY with these keys:
- "is_relevant": boolean
- "has_hallucinations": boolean
- "failure_category": one of "none", "irrelevant", "hallucination", "incomplete", "unsafe"
- "suggested_focus": if the answer is incomplete or irrelevant, write a specific search query that would find the missing information. Otherwise empty string.
- "confidence": 0.0-1.0 score for overall answer quality
- "feedback": brief explanation of your judgment
"""


def ReflectionAgent(state: AgentStateV2) -> AgentStateV2:
    question = state.get('question', '')
    answer = state.get('generation', '')

    judge_prompt = _REFLECTION_PROMPT.format(question=question, answer=answer)
    raw = invoke_json(judge_prompt, system=_REFLECTION_SYSTEM)
    result = _parse_reflection_response(raw)

    attempts = state.get('attempts', {'reflection': 0, 'executor': 0})
    attempts['reflection'] = attempts.get('reflection', 0) + 1
    state['attempts'] = attempts

    needs_retry = result.failure_category != 'none' and attempts['reflection'] < 2
    state['needs_retry'] = needs_retry
    state['reflection_feedback'] = result.feedback
    state['reflection_suggested_focus'] = result.suggested_focus
    state['reflection_confidence'] = result.confidence
    state['reflection_failure_category'] = result.failure_category

    logger.info(
        "reflection_complete",
        extra={
            "failure_category": result.failure_category,
            "confidence": result.confidence,
            "needs_retry": needs_retry,
            "suggested_focus": result.suggested_focus[:80] if result.suggested_focus else "",
            "attempt": attempts['reflection'],
        },
    )

    return state
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_reflection_structured.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add agents/reflection_agent.py tests/eval/test_reflection_structured.py
git commit -m "feat: upgrade ReflectionAgent to structured prompt with failure_category and suggested_focus"
```

---

### Task 3: Add new state fields to AgentStateV2

**Files:**
- Modify: `core/state_v2.py`
- Test: existing tests should still pass

**Step 1: Add the new fields to `AgentStateV2`**

Add these fields to the `AgentStateV2` TypedDict:

```python
reflection_suggested_focus: str
reflection_confidence: float
reflection_failure_category: str
```

**Step 2: Add defaults to `initialize_state` and `reset_query_state`**

In `initialize_state`, add:
```python
'reflection_suggested_focus': '',
'reflection_confidence': 0.0,
'reflection_failure_category': '',
```

In `reset_query_state`, add the same three fields to the reset dict.

**Step 3: Run tests to verify nothing breaks**

Run: `pytest tests/ -v --timeout=30`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add core/state_v2.py
git commit -m "feat: add reflection structured fields to AgentStateV2"
```

---

### Task 4: ReAct Follow-Up — Write the retry query rewrite test

**Files:**
- Create: `tests/eval/test_react_retry.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_react_retry.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import patch


def test_retry_uses_suggested_focus_as_query():
    """On retry, the retriever should use reflection's suggested_focus, not the original query."""
    from agents.retriever_agent import RetrieverAgent
    from langchain_core.documents import Document
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()

    state = {
        "question": "What is the pediatric dosage of aspirin?",
        "optimized_query": "pediatric aspirin dosing guidelines",
        "needs_retry": True,
        "reflection_suggested_focus": "aspirin dose milligrams children weight-based",
        "attempts": {"reflection": 1, "executor": 0},
        "vector_repo": repo,
        "stepback_query": "general pharmacology of aspirin",
        "documents": [],
        "retrieval_confidence": 0.0,
        "retrieval_candidate_count": 0,
        "route": "vector",
        "planned_route": "vector",
        "post_retrieval_route": "executor",
        "route_decision_reason": "",
    }

    # The retriever should pick up suggested_focus as the query on retry
    with patch("agents.retriever_agent.embed_query") as mock_embed, \
         patch.object(repo, "hybrid_search", return_value=[]) as mock_search:
        mock_embed.return_value = [0.0] * 768
        RetrieverAgent(state)

        # The first call should use the suggested_focus, not the original query
        first_call_query = mock_search.call_args_list[0][1].get("query", mock_search.call_args_list[0][0][0] if mock_search.call_args_list[0][0] else "")
        assert "aspirin dose milligrams" in first_call_query or first_call_query == "aspirin dose milligrams children weight-based"


def test_retry_clears_stepback_query():
    """On retry, stepback_query should not be used (it was for the original query)."""
    from agents.retriever_agent import RetrieverAgent
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()

    state = {
        "question": "What is the pediatric dosage of aspirin?",
        "optimized_query": "pediatric aspirin dosing guidelines",
        "needs_retry": True,
        "reflection_suggested_focus": "aspirin dose children",
        "stepback_query": "general pharmacology of aspirin",
        "attempts": {"reflection": 1, "executor": 0},
        "vector_repo": repo,
        "documents": [],
        "retrieval_confidence": 0.0,
        "retrieval_candidate_count": 0,
        "route": "vector",
        "planned_route": "vector",
        "post_retrieval_route": "executor",
        "route_decision_reason": "",
    }

    with patch("agents.retriever_agent.embed_query") as mock_embed, \
         patch.object(repo, "hybrid_search", return_value=[]) as mock_search:
        mock_embed.return_value = [0.0] * 768
        RetrieverAgent(state)

        # Should only be called once (no stepback search)
        assert mock_search.call_count == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_react_retry.py -v`
Expected: FAIL (retriever doesn't use suggested_focus yet)

---

### Task 5: ReAct Follow-Up — Wire suggested_focus into RetrieverAgent

**Files:**
- Modify: `agents/retriever_agent.py`

**Step 1: Modify `RetrieverAgent` to use reflection feedback on retry**

At the top of the `RetrieverAgent` function, after `query = ...`, add retry-aware query selection:

```python
def RetrieverAgent(state: AgentStateV2) -> AgentStateV2:
    query = state.get('optimized_query') or state.get('question', '')

    # ReAct: on retry, use reflection's suggested_focus as the retrieval query
    is_retry = state.get('needs_retry', False)
    suggested_focus = state.get('reflection_suggested_focus', '')
    if is_retry and suggested_focus:
        query = suggested_focus
        logger.info("react_retry_refocused", extra={"suggested_focus": suggested_focus[:80]})

    repo = state.get('vector_repo')
    if repo is None:
        # ... existing no-repo handling unchanged
        ...

    query_embedding = embed_query(query)
    docs = repo.hybrid_search(query=query, query_embedding=query_embedding, k=20)

    # On retry, skip stepback (it was for the original query, not the focused retry)
    if not is_retry:
        stepback = state.get('stepback_query', '')
        if stepback:
            sb_embedding = embed_query(stepback)
            sb_docs = repo.hybrid_search(query=stepback, query_embedding=sb_embedding, k=10)
            docs = _merge_and_deduplicate(docs, sb_docs)

    ranked, scores = rerank(query, docs)
    # ... rest unchanged
```

**Step 2: Run the retry tests**

Run: `pytest tests/eval/test_react_retry.py -v`
Expected: PASS (2 tests)

**Step 3: Run all tests to verify no regressions**

Run: `pytest tests/ -v --timeout=30`
Expected: All PASS

**Step 4: Commit**

```bash
git add agents/retriever_agent.py tests/eval/test_react_retry.py
git commit -m "feat: ReAct follow-up — use reflection suggested_focus as retry query"
```

---

### Task 6: ReAct — Update workflow routing for retry clarity

**Files:**
- Modify: `core/langgraph_workflow.py` (no changes needed — existing `retry→retriever` edge already works)
- Test: verify the flow works end-to-end conceptually

**Step 1: Verify the existing workflow wiring supports the new pattern**

The current workflow already has:
```python
workflow.add_conditional_edges(
    'reflection',
    route_after_reflection,
    {'retry': 'retriever', 'finalize': 'explanation'},
)
```

This means on retry, the flow goes: reflection → retriever → executor → reflection.
The retriever now picks up `reflection_suggested_focus` from state, so the loop naturally works.

**No code changes needed for the workflow graph.**

**Step 2: Write an integration-level test to document the flow**

Add to `tests/eval/test_react_retry.py`:

```python
def test_retry_flow_documented():
    """Document the expected ReAct retry flow:

    1. executor generates answer
    2. reflection judges → failure_category != 'none', suggested_focus set
    3. needs_retry=True → route_after_reflection returns 'retry'
    4. retriever uses suggested_focus as query (not original optimized_query)
    5. executor generates new answer from new context
    6. reflection judges again (attempt 2)
    7. If still failing, needs_retry=False (max attempts reached) → finalize
    """
    from core.langgraph_workflow import route_after_reflection

    # Retry case
    state_retry = {"needs_retry": True}
    assert route_after_reflection(state_retry) == "retry"

    # Finalize case
    state_done = {"needs_retry": False}
    assert route_after_reflection(state_done) == "finalize"
```

**Step 3: Run test**

Run: `pytest tests/eval/test_react_retry.py -v`
Expected: PASS (3 tests)

**Step 4: Commit**

```bash
git add tests/eval/test_react_retry.py
git commit -m "test: document ReAct retry flow with route verification"
```

---

### Task 7: Plan-and-Execute Stub — Write the decomposer interface and test

**Files:**
- Create: `agents/decomposer_agent.py`
- Create: `tests/eval/test_decomposer_stub.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_decomposer_stub.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_decomposer_disabled_by_default():
    """When ENABLE_QUERY_DECOMPOSITION is False, decomposer passes through."""
    from agents.decomposer_agent import DecomposerAgent

    state = {
        "question": "Compare aspirin and ibuprofen for pain",
        "optimized_query": "compare aspirin ibuprofen pain relief",
        "sub_questions": [],
        "decomposition_enabled": False,
    }
    result = DecomposerAgent(state)
    assert result.get("sub_questions") == []
    assert result.get("optimized_query") == "compare aspirin ibuprofen pain relief"


def test_decompose_multi_part_question():
    """Decomposer should split a multi-part question into sub-questions."""
    from agents.decomposer_agent import decompose_question

    sub_qs = decompose_question(
        "What are aspirin's side effects and what is the recommended dosage for children?"
    )
    # The function should return a list of sub-question dicts
    assert isinstance(sub_qs, list)
    # Each sub-question has 'question' and 'route' keys
    for sq in sub_qs:
        assert "question" in sq
        assert "route" in sq


def test_is_multi_part_detects_compound_questions():
    from agents.decomposer_agent import is_multi_part_question

    assert is_multi_part_question("What are side effects and dosage of aspirin?") is True
    assert is_multi_part_question("What is aspirin?") is False
    assert is_multi_part_question("Compare drug A with drug B and explain mechanisms") is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_decomposer_stub.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write the stubbed decomposer**

```python
# agents/decomposer_agent.py
"""Plan-and-Execute: query decomposition agent (STUBBED).

This agent detects multi-part questions and decomposes them into
sub-questions for independent retrieval and answer generation.

Currently DISABLED by default (ENABLE_QUERY_DECOMPOSITION=false).
The decompose→execute→synthesize loop is not yet wired into the
LangGraph workflow. This module provides the interface and detection
logic for future activation when multi-PDF ingestion is ready.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from core.state_v2 import AgentStateV2

logger = logging.getLogger(__name__)

# Compound question indicators
_COMPOUND_PATTERNS = [
    r'\band\b.*\b(?:what|how|why|when|which|compare|explain|describe)\b',
    r'\b(?:compare|contrast|difference between)\b',
    r'\b(?:also|additionally|furthermore)\b',
    r'(?:1\.|2\.|a\)|b\))',  # numbered sub-questions
]
_COMPOUND_RE = [re.compile(p, re.IGNORECASE) for p in _COMPOUND_PATTERNS]

# Conjunction splitters
_SPLIT_CONJUNCTIONS = re.compile(
    r'\b(?:and also|and what|and how|and explain|and describe|and compare)\b',
    re.IGNORECASE,
)


def is_multi_part_question(question: str) -> bool:
    """Heuristic check: does this question contain multiple distinct sub-questions?"""
    if not question:
        return False
    # Check for compound patterns
    matches = sum(1 for p in _COMPOUND_RE if p.search(question))
    return matches >= 1 and len(question.split()) > 8


def decompose_question(question: str) -> list[dict[str, str]]:
    """Split a compound question into sub-questions.

    Returns a list of dicts with 'question' and 'route' keys.
    This is a rule-based stub — future versions will use LLM decomposition.
    """
    if not is_multi_part_question(question):
        return [{"question": question, "route": "vector"}]

    # Simple conjunction-based splitting
    parts = _SPLIT_CONJUNCTIONS.split(question)
    parts = [p.strip().rstrip('?').strip() + '?' for p in parts if p.strip()]

    if len(parts) <= 1:
        return [{"question": question, "route": "vector"}]

    return [{"question": part, "route": "vector"} for part in parts]


def DecomposerAgent(state: AgentStateV2) -> AgentStateV2:
    """Plan-and-Execute decomposer node.

    When disabled (default), passes through without modification.
    When enabled, decomposes multi-part questions into sub_questions.
    """
    enabled = state.get('decomposition_enabled', False)

    if not enabled:
        state['sub_questions'] = state.get('sub_questions', [])
        return state

    question = state.get('question', '')
    if is_multi_part_question(question):
        sub_qs = decompose_question(question)
        state['sub_questions'] = sub_qs
        logger.info(
            "query_decomposed",
            extra={"original": question[:80], "sub_count": len(sub_qs)},
        )
    else:
        state['sub_questions'] = []

    return state
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_decomposer_stub.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add agents/decomposer_agent.py tests/eval/test_decomposer_stub.py
git commit -m "feat: stub Plan-and-Execute decomposer agent (disabled by default)"
```

---

### Task 8: Plan-and-Execute — Add state fields and settings

**Files:**
- Modify: `core/state_v2.py`
- Modify: `core/settings.py`

**Step 1: Add `sub_questions` and `decomposition_enabled` to state**

In `AgentStateV2`, add:
```python
sub_questions: list[dict[str, str]]
decomposition_enabled: bool
```

In `initialize_state`, add:
```python
'sub_questions': [],
'decomposition_enabled': False,
```

In `reset_query_state`, add:
```python
'sub_questions': [],
```

**Step 2: Add setting to `core/settings.py`**

```python
enable_query_decomposition: bool = Field(default=False, alias='ENABLE_QUERY_DECOMPOSITION')
```

**Step 3: Run all tests**

Run: `pytest tests/ -v --timeout=30`
Expected: All PASS

**Step 4: Commit**

```bash
git add core/state_v2.py core/settings.py
git commit -m "feat: add Plan-and-Execute state fields and setting (disabled by default)"
```

---

### Task 9: Documentation — Update architecture and implementation log

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update architecture doc**

Add to the "Current important flows" section, under the existing reflection description:

```markdown
### Reflection and retry flow
1. After executor generates an answer, the reflection agent judges quality using structured criteria.
2. The judge returns: `failure_category` (none/irrelevant/hallucination/incomplete/unsafe), `suggested_focus` (targeted retrieval hint), `confidence` (0.0–1.0), and `feedback`.
3. If `failure_category != 'none'` and attempts < 2, the workflow retries.
4. On retry, the retriever uses `suggested_focus` as the search query instead of the original `optimized_query` (ReAct pattern: observe feedback → act on it).
5. The step-back query is skipped on retry since it was generated for the original question, not the focused retry.
6. Maximum 2 reflection attempts, then finalize regardless.

### Plan-and-Execute (stubbed)
- `DecomposerAgent` can detect multi-part questions and split them into sub-questions.
- Disabled by default (`ENABLE_QUERY_DECOMPOSITION=false`).
- Not yet wired into the LangGraph workflow graph.
- Intended for future multi-PDF support where sub-questions may route to different document sources.
```

**Step 2: Add implementation log entry**

Append to `docs/changes/implementation-log.md`:

```markdown
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
```

**Step 3: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: update architecture and log for thinking patterns implementation"
```

---

## Summary of changes

| Component | Change | Pattern |
|-----------|--------|---------|
| `agents/reflection_agent.py` | Structured ReflectionResult, new prompt, new state fields | Reflection |
| `agents/retriever_agent.py` | Use `reflection_suggested_focus` on retry, skip stepback | ReAct |
| `agents/decomposer_agent.py` | NEW — stubbed decomposer with heuristic detection | Plan-and-Execute |
| `core/state_v2.py` | Add `reflection_suggested_focus`, `reflection_confidence`, `reflection_failure_category`, `sub_questions`, `decomposition_enabled` | All |
| `core/settings.py` | Add `ENABLE_QUERY_DECOMPOSITION` setting | Plan-and-Execute |
| `core/langgraph_workflow.py` | No changes — existing retry edge already supports new pattern | — |
| `tests/eval/test_reflection_structured.py` | NEW — 5 tests for structured reflection | Reflection |
| `tests/eval/test_react_retry.py` | NEW — 3 tests for ReAct retry | ReAct |
| `tests/eval/test_decomposer_stub.py` | NEW — 3 tests for decomposer stub | Plan-and-Execute |
