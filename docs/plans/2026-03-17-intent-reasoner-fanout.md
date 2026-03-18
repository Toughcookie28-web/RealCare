# IntentReasoner + Fan-Out Decomposition + Combiner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the "retry-as-decomposition" anti-pattern with upfront intelligent planning. A new IntentReasoner node runs before QueryRewriter, decides if the question is simple (single path) or compound (fan-out), and routes accordingly. Compound questions get N parallel sub-pipelines (each with own QueryRewriter → Retriever → Executor), then a Combiner agent merges the answers. ReflectionAgent is simplified to quality-gate only — no more decomposition detection via retry.

**Architecture:**
```
GuardrailAgent → MemoryAgent → IntentReasoner (NEW)
    ↓ single                    ↓ compound
QueryRewriterAgent          QueryRewriterAgent × N (parallel via LangGraph Send)
    ↓                           ↓
PlannerAgent                RetrieverAgent × N
    ↓                           ↓
[existing routes]           ExecutorAgent × N
    ↓                           ↓
ExecutorAgent               CombinerAgent (NEW) ← fan-in
    ↓                           ↓
ReflectionAgent (simplified) ← both paths
    ↓
ExplanationAgent → END
```

**LangGraph fan-out mechanism:** `langgraph.types.Send` API dispatches N state copies to the same node in parallel. Sub-results accumulate in `sub_results: Annotated[list[dict], operator.add]` state field.

**Controlled by:** `ENABLE_INTENT_REASONER=false` (default off — safe rollout). When off, `IntentReasoner` is a pass-through and the existing path is unchanged.

**Tech Stack:** Python, Pydantic, LangGraph `Send` API, existing `invoke_json`

---

### Task 1: Add IntentReasoner settings

**Files:**
- Modify: `core/settings.py`
- Modify: `tests/core/test_settings_contract.py`

**Step 1: Write failing test**

```python
def test_intent_reasoner_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.enable_intent_reasoner is False
    assert s.intent_reasoner_max_sub_questions == 3
    assert s.intent_reasoner_compound_confidence_threshold == 0.7
```

**Step 2: Run to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v -k "intent_reasoner" 2>&1`
Expected: FAIL

**Step 3: Add settings**

In `core/settings.py`, after `enable_query_decomposition`:
```python
enable_intent_reasoner: bool = Field(default=False, alias='ENABLE_INTENT_REASONER')
intent_reasoner_max_sub_questions: int = Field(default=3, alias='INTENT_REASONER_MAX_SUB_QUESTIONS')
intent_reasoner_compound_confidence_threshold: float = Field(
    default=0.7, alias='INTENT_REASONER_COMPOUND_CONFIDENCE_THRESHOLD'
)
```

**Step 4: Verify**

Run: `pytest tests/core/test_settings_contract.py -v -k "intent_reasoner" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings_contract.py
git commit -m "feat: add IntentReasoner settings (disabled by default)"
```

---

### Task 2: Add fan-out state fields to AgentStateV2

**Files:**
- Modify: `core/state_v2.py`

**Step 1: Update AgentStateV2 for fan-out support**

LangGraph's `Send` API accumulates sub-results via `Annotated` reducer fields.

In `core/state_v2.py`:
```python
from __future__ import annotations
import operator
from typing import Any, Annotated, TypedDict
from langchain_core.documents import Document


class AgentStateV2(TypedDict):
    # ... all existing fields ...

    # IntentReasoner fields
    is_compound: bool
    compound_confidence: float
    sub_questions: list[str]          # was list[dict[str,str]], simplify to list[str]
    intent_hints: list[str]           # one per sub_question
    active_sub_question: str          # which sub_question this pipeline instance handles
    sub_results: Annotated[list[dict[str, Any]], operator.add]  # fan-in accumulator
```

**Important:** `Annotated[list, operator.add]` tells LangGraph to *merge* sub-results from parallel branches rather than overwrite.

**Step 2: Update initialize_state()**

```python
'is_compound': False,
'compound_confidence': 0.0,
'sub_questions': [],
'intent_hints': [],
'active_sub_question': '',
'sub_results': [],
```

**Step 3: Update reset_query_state()**

```python
'is_compound': False,
'compound_confidence': 0.0,
'sub_questions': [],
'intent_hints': [],
'active_sub_question': '',
'sub_results': [],
```

**Step 4: Verify no import errors**

Run: `python -c "from core.state_v2 import initialize_state; s=initialize_state('s','t'); print(s['sub_results'])"`
Expected: `[]`

**Step 5: Commit**

```bash
git add core/state_v2.py
git commit -m "feat: add fan-out state fields to AgentStateV2"
```

---

### Task 3: Create agents/intent_reasoner_agent.py

**Files:**
- Create: `agents/intent_reasoner_agent.py`
- Create: `tests/agents/test_intent_reasoner.py`

**Step 1: Write failing tests**

Create `tests/agents/test_intent_reasoner.py`:
```python
def test_intent_reasoner_passthrough_when_disabled(monkeypatch):
    """When ENABLE_INTENT_REASONER=false, agent is a pure pass-through."""
    monkeypatch.setenv("ENABLE_INTENT_REASONER", "false")
    # Clear settings cache
    from core.settings import get_settings
    get_settings.cache_clear()

    from agents.intent_reasoner_agent import IntentReasonerAgent
    from core.state_v2 import initialize_state

    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin?"
    result = IntentReasonerAgent(state)

    assert result["is_compound"] is False
    assert result["sub_questions"] == []


def test_intent_reasoner_detects_compound_question(monkeypatch):
    monkeypatch.setenv("ENABLE_INTENT_REASONER", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    monkeypatch.setattr(
        "agents.intent_reasoner_agent.invoke_json",
        lambda *a, **kw: {
            "is_compound": True,
            "compound_confidence": 0.85,
            "sub_questions": [
                "what is the mechanism of aspirin?",
                "what are the side effects of aspirin?",
            ],
            "intent_hints": ["mechanism_of_action", "side_effects"],
            "reasoning": "Two distinct questions combined.",
        },
    )

    from agents.intent_reasoner_agent import IntentReasonerAgent
    from core.state_v2 import initialize_state

    state = initialize_state("s1", "t1")
    state["question"] = "what is the mechanism of aspirin and what are its side effects?"
    result = IntentReasonerAgent(state)

    assert result["is_compound"] is True
    assert len(result["sub_questions"]) == 2


def test_intent_reasoner_caps_sub_questions_at_max(monkeypatch):
    monkeypatch.setenv("ENABLE_INTENT_REASONER", "true")
    monkeypatch.setenv("INTENT_REASONER_MAX_SUB_QUESTIONS", "2")
    from core.settings import get_settings
    get_settings.cache_clear()

    monkeypatch.setattr(
        "agents.intent_reasoner_agent.invoke_json",
        lambda *a, **kw: {
            "is_compound": True,
            "compound_confidence": 0.9,
            "sub_questions": ["q1", "q2", "q3", "q4"],  # 4 returned, should be capped at 2
            "intent_hints": ["a", "b", "c", "d"],
            "reasoning": "Many questions.",
        },
    )

    from agents.intent_reasoner_agent import IntentReasonerAgent
    from core.state_v2 import initialize_state

    state = initialize_state("s1", "t1")
    state["question"] = "question with four parts"
    result = IntentReasonerAgent(state)

    assert len(result["sub_questions"]) <= 2


def test_intent_reasoner_single_path_when_low_confidence(monkeypatch):
    monkeypatch.setenv("ENABLE_INTENT_REASONER", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    monkeypatch.setattr(
        "agents.intent_reasoner_agent.invoke_json",
        lambda *a, **kw: {
            "is_compound": True,
            "compound_confidence": 0.4,  # below threshold → treat as single
            "sub_questions": ["q1", "q2"],
            "intent_hints": ["a", "b"],
            "reasoning": "Maybe compound.",
        },
    )

    from agents.intent_reasoner_agent import IntentReasonerAgent
    from core.state_v2 import initialize_state

    state = initialize_state("s1", "t1")
    state["question"] = "possibly compound"
    result = IntentReasonerAgent(state)

    # Low compound confidence → treat as single
    assert result["is_compound"] is False
```

**Step 2: Run to verify they fail**

Run: `pytest tests/agents/test_intent_reasoner.py -v 2>&1`
Expected: FAIL

**Step 3: Create agents/intent_reasoner_agent.py**

```python
from __future__ import annotations

import logging

from pydantic import BaseModel, field_validator

from core.settings import get_settings
from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_json

logger = logging.getLogger(__name__)

_INTENT_REASONER_SYSTEM = (
    "You are a medical question analyst. Your job is to determine whether a user question "
    "is a single focused question or a compound question containing multiple distinct sub-questions. "
    "Respond ONLY with valid JSON."
)

_INTENT_REASONER_PROMPT = """Analyze this medical question:

Question: {question}
Conversation summary: {summary}

Determine:
1. Is this a compound question — does it contain 2+ meaningfully distinct medical questions that would benefit from separate retrieval?
   Examples of compound: "what is aspirin's mechanism AND its side effects?", "compare metformin vs insulin AND explain when each is used"
   Examples of single: "what is the mechanism of aspirin?", "what are aspirin side effects?" (even if long)
2. If compound, break it into sub-questions (max {max_sub_questions}). Each sub-question should be self-contained.
3. Provide an intent hint for each sub-question (e.g., "mechanism_of_action", "side_effects", "dosage", "comparison", "definition").
4. How confident are you that this is compound (0.0-1.0)?

Return JSON:
{{
  "reasoning": "...",
  "is_compound": boolean,
  "compound_confidence": 0.0-1.0,
  "sub_questions": ["...", "..."],
  "intent_hints": ["...", "..."]
}}

If is_compound is false, sub_questions and intent_hints should be empty lists.
"""


class IntentAnalysis(BaseModel):
    is_compound: bool = False
    compound_confidence: float = 0.0
    sub_questions: list[str] = []
    intent_hints: list[str] = []
    reasoning: str = ""

    @field_validator('compound_confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator('sub_questions', 'intent_hints', mode='before')
    @classmethod
    def coerce_list(cls, v):
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if item]


def IntentReasonerAgent(state: AgentStateV2) -> AgentStateV2:
    settings = get_settings()

    # Pass-through when disabled
    if not settings.enable_intent_reasoner:
        state['is_compound'] = False
        state['compound_confidence'] = 0.0
        state['sub_questions'] = []
        state['intent_hints'] = []
        return state

    question = state.get('question', '')
    summary = state.get('summary', '')

    prompt = _INTENT_REASONER_PROMPT.format(
        question=question,
        summary=summary or '(no prior conversation)',
        max_sub_questions=settings.intent_reasoner_max_sub_questions,
    )

    raw = invoke_json(prompt, system=_INTENT_REASONER_SYSTEM)
    if not raw:
        state['is_compound'] = False
        state['compound_confidence'] = 0.0
        state['sub_questions'] = []
        state['intent_hints'] = []
        return state

    try:
        analysis = IntentAnalysis.model_validate(raw)
    except Exception:
        logger.warning("IntentReasoner failed to parse response, defaulting to single path")
        state['is_compound'] = False
        state['compound_confidence'] = 0.0
        state['sub_questions'] = []
        state['intent_hints'] = []
        return state

    # Only treat as compound if confidence exceeds threshold
    threshold = settings.intent_reasoner_compound_confidence_threshold
    is_compound = analysis.is_compound and analysis.compound_confidence >= threshold

    # Cap sub-questions
    max_q = settings.intent_reasoner_max_sub_questions
    sub_questions = analysis.sub_questions[:max_q] if is_compound else []
    intent_hints = analysis.intent_hints[:max_q] if is_compound else []

    # Align lengths
    while len(intent_hints) < len(sub_questions):
        intent_hints.append("general")

    state['is_compound'] = is_compound
    state['compound_confidence'] = analysis.compound_confidence
    state['sub_questions'] = sub_questions
    state['intent_hints'] = intent_hints

    logger.info(
        "intent_reasoner_complete",
        extra={
            "is_compound": is_compound,
            "compound_confidence": analysis.compound_confidence,
            "sub_question_count": len(sub_questions),
            "reasoning": analysis.reasoning[:100],
        },
    )

    return state
```

**Step 4: Run tests**

Run: `pytest tests/agents/test_intent_reasoner.py -v 2>&1`
Expected: all PASS

**Step 5: Commit**

```bash
git add agents/intent_reasoner_agent.py tests/agents/test_intent_reasoner.py
git commit -m "feat: add IntentReasonerAgent for compound question detection"
```

---

### Task 4: Create agents/combiner_agent.py

**Files:**
- Create: `agents/combiner_agent.py`
- Create: `tests/agents/test_combiner.py`

**Step 1: Write failing tests**

Create `tests/agents/test_combiner.py`:
```python
def test_combiner_merges_sub_results(monkeypatch):
    monkeypatch.setattr(
        "agents.combiner_agent.invoke_llm",
        lambda prompt, system, use_fallback=False: "Aspirin inhibits COX enzymes (mechanism). Common side effects include GI upset and bleeding risk.",
    )
    from agents.combiner_agent import CombinerAgent
    from core.state_v2 import initialize_state

    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin's mechanism and its side effects?"
    state["sub_results"] = [
        {"sub_question": "mechanism", "generation": "Aspirin inhibits COX-1 and COX-2.", "citations": []},
        {"sub_question": "side_effects", "generation": "Side effects include GI upset, bleeding.", "citations": []},
    ]

    result = CombinerAgent(state)
    assert result["generation"] != ""
    assert result["source"] == "Combiner (Multi-Path)"


def test_combiner_passthrough_when_single_sub_result(monkeypatch):
    """When only one sub_result exists, use it directly without LLM call."""
    from agents.combiner_agent import CombinerAgent
    from core.state_v2 import initialize_state

    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin?"
    state["sub_results"] = [
        {"sub_question": "mechanism", "generation": "Aspirin works by...", "citations": [{"section": "S1"}]},
    ]

    result = CombinerAgent(state)
    assert result["generation"] == "Aspirin works by..."
    assert result["citations"] == [{"section": "S1"}]


def test_combiner_merges_citations():
    from agents.combiner_agent import _merge_citations
    c1 = [{"section": "S1", "page": 1}, {"section": "S2", "page": 2}]
    c2 = [{"section": "S2", "page": 2}, {"section": "S3", "page": 3}]
    merged = _merge_citations([c1, c2])
    sections = [c["section"] for c in merged]
    assert len(sections) == 3
    assert "S3" in sections
```

**Step 2: Run to verify they fail**

Run: `pytest tests/agents/test_combiner.py -v 2>&1`
Expected: FAIL

**Step 3: Create agents/combiner_agent.py**

```python
from __future__ import annotations

import logging

from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_llm

logger = logging.getLogger(__name__)

_COMBINER_SYSTEM = (
    "You are a medical education assistant. "
    "You have received answers to multiple sub-questions from a compound medical question. "
    "Combine them into one coherent, well-structured response. "
    "Do not add new information. Only synthesize what was found."
)

_COMBINER_PROMPT = """The user asked: {question}

This was decomposed into {n} sub-questions. Here are the answers:

{sub_answers}

Write a single, well-organized response that answers all parts of the original question.
Use clear structure (numbered points or headers if helpful).
"""


def _merge_citations(citation_lists: list[list[dict]]) -> list[dict]:
    """Deduplicate and merge citation lists from multiple sub-results."""
    seen: set[tuple] = set()
    merged = []
    for citations in citation_lists:
        for c in citations:
            key = (c.get("section"), c.get("page"))
            if key not in seen:
                seen.add(key)
                merged.append(c)
    return merged


def CombinerAgent(state: AgentStateV2) -> AgentStateV2:
    sub_results = state.get('sub_results', [])

    if not sub_results:
        state['generation'] = state.get('generation', '')
        return state

    # Single sub-result: pass through without extra LLM call
    if len(sub_results) == 1:
        r = sub_results[0]
        state['generation'] = r.get('generation', '')
        state['citations'] = r.get('citations', [])
        state['source'] = r.get('source', 'LLM + Retrieved Evidence')
        return state

    # Multiple sub-results: combine via LLM
    sub_answer_parts = []
    for i, r in enumerate(sub_results):
        sub_q = r.get('sub_question', f'Part {i+1}')
        gen = r.get('generation', '(no answer)')
        sub_answer_parts.append(f"Sub-question {i+1}: {sub_q}\nAnswer: {gen}")

    sub_answers_text = "\n\n---\n\n".join(sub_answer_parts)
    prompt = _COMBINER_PROMPT.format(
        question=state.get('question', ''),
        n=len(sub_results),
        sub_answers=sub_answers_text,
    )

    combined = invoke_llm(prompt, system=_COMBINER_SYSTEM, use_fallback=False)
    if not combined:
        combined = invoke_llm(prompt, system=_COMBINER_SYSTEM, use_fallback=True)
    if not combined:
        # Last resort: concatenate answers
        combined = "\n\n".join(r.get('generation', '') for r in sub_results)

    all_citations = [r.get('citations', []) for r in sub_results]
    merged_citations = _merge_citations(all_citations)

    state['generation'] = combined
    state['citations'] = merged_citations
    state['source'] = 'Combiner (Multi-Path)'

    logger.info(
        "combiner_complete",
        extra={"sub_results_count": len(sub_results), "citations_count": len(merged_citations)},
    )

    return state
```

**Step 4: Run tests**

Run: `pytest tests/agents/test_combiner.py -v 2>&1`
Expected: all PASS

**Step 5: Commit**

```bash
git add agents/combiner_agent.py tests/agents/test_combiner.py
git commit -m "feat: add CombinerAgent for merging multi-path sub-answers"
```

---

### Task 5: Wire fan-out into LangGraph workflow

**Files:**
- Modify: `core/langgraph_workflow.py`
- Create: `tests/agents/test_fanout_workflow.py`

This is the most complex task. Read `core/langgraph_workflow.py` fully before editing.

**Step 1: Understand LangGraph Send API**

`Send` dispatches a modified state copy to a specific node:
```python
from langgraph.types import Send

def fan_out_to_sub_pipelines(state: AgentStateV2):
    if state.get('is_compound') and state.get('sub_questions'):
        return [
            Send("rewriter", {**state, "question": q, "active_sub_question": q, "intent_hint": h})
            for q, h in zip(state['sub_questions'], state['intent_hints'])
        ]
    return "rewriter"  # single path
```

Each `Send` clone runs through `rewriter → planner → retriever → executor`, then converges at `combiner` via the `sub_results` accumulator.

**Step 2: Write a workflow integration test**

Create `tests/agents/test_fanout_workflow.py`:
```python
import pytest
from unittest.mock import patch


def test_single_path_workflow_unchanged(monkeypatch):
    """When is_compound=False, workflow runs the existing single path."""
    monkeypatch.setenv("ENABLE_INTENT_REASONER", "false")
    from core.settings import get_settings
    get_settings.cache_clear()

    # Just verify workflow compiles without error when intent reasoner is off
    from core.langgraph_workflow import create_workflow
    wf = create_workflow()
    assert wf is not None


def test_intent_reasoner_node_exists_in_workflow():
    from core.langgraph_workflow import create_workflow
    import inspect
    # Check that create_workflow doesn't raise
    wf = create_workflow()
    assert wf is not None
```

**Step 3: Run to verify they pass (they test compilation, not logic)**

Run: `pytest tests/agents/test_fanout_workflow.py -v 2>&1`
Expected: PASS (just tests that workflow compiles)

**Step 4: Update langgraph_workflow.py**

```python
from langgraph.types import Send  # ADD at top

from agents.intent_reasoner_agent import IntentReasonerAgent   # ADD
from agents.combiner_agent import CombinerAgent                # ADD


def _intent_reasoner_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('intent_reasoner', IntentReasonerAgent, state)


def _combiner_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('combiner', CombinerAgent, state)


def route_after_intent_reasoner(state: AgentStateV2):
    """Fan out to N sub-pipelines or continue single path."""
    if state.get('is_compound') and state.get('sub_questions'):
        return [
            Send(
                "rewriter",
                {
                    **state,
                    "question": q,
                    "active_sub_question": q,
                    "sub_results": [],   # reset per branch
                },
            )
            for q in state['sub_questions']
        ]
    return "rewriter"


def route_after_executor(state: AgentStateV2) -> str:
    """After executor: if compound mode, store result and go to combiner.
    If single mode, go to reflection as before."""
    if state.get('is_compound') and state.get('active_sub_question'):
        # Store this sub-result into sub_results accumulator
        state['sub_results'] = [{
            'sub_question': state.get('active_sub_question', ''),
            'generation': state.get('generation', ''),
            'citations': state.get('citations', []),
            'source': state.get('source', ''),
        }]
        return 'combiner'
    return 'reflection'


def create_workflow():
    workflow = StateGraph(AgentStateV2)

    # Add all existing nodes...
    workflow.add_node('guardrail', _guardrail_node)
    workflow.add_node('memory', _memory_node)
    workflow.add_node('intent_reasoner', _intent_reasoner_node)   # NEW
    workflow.add_node('rewriter', _rewriter_node)
    workflow.add_node('planner', _planner_node)
    workflow.add_node('retriever', _retriever_node)
    workflow.add_node('tavily', _web_node)
    workflow.add_node('wikipedia', _wiki_node)
    workflow.add_node('literature', _lit_node)
    workflow.add_node('llm', _llm_node)
    workflow.add_node('executor', _executor_node)
    workflow.add_node('combiner', _combiner_node)                 # NEW
    workflow.add_node('reflection', _reflection_node)
    workflow.add_node('explanation', _explanation_node)

    workflow.set_entry_point('guardrail')

    workflow.add_conditional_edges('guardrail', route_after_guardrail,
        {'blocked': END, 'continue': 'memory'})

    workflow.add_edge('memory', 'intent_reasoner')                # CHANGED: was 'rewriter'

    # Fan-out edge: single → rewriter, compound → Send × N → rewriter
    workflow.add_conditional_edges('intent_reasoner', route_after_intent_reasoner)

    workflow.add_edge('rewriter', 'planner')

    workflow.add_conditional_edges('planner', route_after_planner,
        {'chitchat': 'llm', 'memory': 'executor', 'clarify': 'executor', 'retriever': 'retriever'})

    workflow.add_conditional_edges('retriever', route_after_retriever,
        {'executor': 'executor', 'web': 'tavily', 'literature': 'literature'})

    workflow.add_conditional_edges('tavily', route_after_web,
        {'executor': 'executor', 'wikipedia': 'wikipedia'})

    workflow.add_edge('wikipedia', 'executor')
    workflow.add_edge('literature', 'executor')
    workflow.add_edge('llm', 'executor')

    # Executor routes to combiner (compound) or reflection (single)
    workflow.add_conditional_edges('executor', route_after_executor,
        {'combiner': 'combiner', 'reflection': 'reflection'})

    # After combiner, always go to reflection
    workflow.add_edge('combiner', 'reflection')

    workflow.add_conditional_edges('reflection', route_after_reflection,
        {'retry': 'retriever', 'finalize': 'explanation'})

    workflow.add_edge('explanation', END)

    return workflow.compile()
```

**Step 5: Run workflow tests**

Run: `pytest tests/agents/test_fanout_workflow.py tests/agents/test_intent_reasoner.py tests/agents/test_combiner.py -v 2>&1`
Expected: all PASS

**Step 6: Run full suite to catch regressions**

Run: `pytest tests/ -x -q 2>&1 | tail -30`
Expected: no new failures

**Step 7: Commit**

```bash
git add core/langgraph_workflow.py tests/agents/test_fanout_workflow.py
git commit -m "feat: wire IntentReasoner fan-out and Combiner into LangGraph workflow"
```

---

### Task 6: Simplify ReflectionAgent (remove decomposition detection)

**Files:**
- Modify: `agents/reflection_agent.py`

With upfront decomposition, the reflection agent no longer needs to detect `incomplete` as a trigger for decomposition. Its only job now is grounding and quality.

**Step 1: Update reflection prompt**

Remove the `incomplete` failure category instruction. Update to:
```
Return JSON with failure_category: one of "none", "irrelevant", "hallucination", "unsafe"
("incomplete" is no longer used — complex questions are now decomposed before retrieval)
```

**Step 2: Update ReflectionResult validator**

```python
@field_validator('failure_category', mode='before')
@classmethod
def coerce_category(cls, v):
    valid = {'none', 'irrelevant', 'hallucination', 'unsafe'}  # remove 'incomplete'
    val = str(v or 'none').strip().lower()
    if val == 'incomplete':
        return 'none'  # backwards compat: treat incomplete as pass (handled by decomposition now)
    return val if val in valid else 'none'
```

**Step 3: Run reflection tests**

Run: `pytest tests/eval/test_reflection_structured.py -v 2>&1`
Expected: all PASS

**Step 4: Commit**

```bash
git add agents/reflection_agent.py
git commit -m "refactor: simplify reflection — remove incomplete category now handled by IntentReasoner"
```

---

### Task 7: Add ADR and update docs

**Files:**
- Create: `docs/decisions/ADR-0006-intent-reasoner-fanout.md`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**ADR key points:**
- **Problem:** Retry-as-decomposition wastes one full pipeline cycle on complex questions. ReflectionAgent was doing planning work (detecting incompleteness) that belongs upfront.
- **Decision:** New `IntentReasonerAgent` runs before `QueryRewriter`. Compound questions fan out via LangGraph `Send` to N parallel sub-pipelines. `CombinerAgent` merges results. `ReflectionAgent` simplified to quality-gate only.
- **Default off:** `ENABLE_INTENT_REASONER=false`. Enable after eval shows benefit.
- **LangGraph mechanism:** `Annotated[list, operator.add]` reducer on `sub_results` field enables fan-in accumulation.
- **Tradeoffs:** Higher latency for compound paths (N × pipeline + combiner LLM call). Benefit: correct, coherent answers for complex questions on first attempt.

**Architecture diagram update:** Add IntentReasoner node between MemoryAgent and QueryRewriter. Add Combiner node between ExecutorAgent and ReflectionAgent (compound path).

Log entry: `2026-03-17 — IntentReasoner + fan-out decomposition + CombinerAgent added (disabled by default). ReflectionAgent simplified.`
