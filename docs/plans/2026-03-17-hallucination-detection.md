# Hallucination Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the ReflectionAgent to cross-check the generated answer against the actual retrieved chunks, not just general medical knowledge. This catches cases where the executor fabricates details that aren't in the source material.

**Architecture:** The reflection prompt currently judges answer quality without seeing the retrieved context. We pass the top chunks (condensed) into the reflection prompt alongside the answer. The judge then identifies specific claims in the answer that cannot be traced back to any chunk. If grounding score < threshold, `failure_category = 'hallucination'` is set and the retry is triggered. A new `grounding_score` state field tracks this. Controlled by `REFLECTION_GROUNDING_ENABLED` setting (default True).

**Tech Stack:** Python, Pydantic, existing `invoke_json`

---

### Task 1: Add grounding settings

**Files:**
- Modify: `core/settings.py`
- Modify: `tests/core/test_settings_contract.py`

**Step 1: Write the failing test**

```python
def test_grounding_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.reflection_grounding_enabled is True
    assert s.reflection_grounding_threshold == 0.5
```

**Step 2: Run to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v -k "grounding" 2>&1`
Expected: FAIL

**Step 3: Add to settings**

In `core/settings.py`, after `live_judge_sampling_ratio`, add:
```python
reflection_grounding_enabled: bool = Field(default=True, alias='REFLECTION_GROUNDING_ENABLED')
reflection_grounding_threshold: float = Field(default=0.5, alias='REFLECTION_GROUNDING_THRESHOLD')
```

**Step 4: Verify test passes**

Run: `pytest tests/core/test_settings_contract.py -v -k "grounding" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings_contract.py
git commit -m "feat: add reflection grounding settings"
```

---

### Task 2: Add grounding_score to state

**Files:**
- Modify: `core/state_v2.py`

**Step 1: Add field to AgentStateV2**

In the `AgentStateV2` TypedDict, after `reflection_failure_category`, add:
```python
grounding_score: float
```

In `initialize_state()`, add:
```python
'grounding_score': 0.0,
```

In `reset_query_state()`, add:
```python
'grounding_score': 0.0,
```

**Step 2: Verify no import errors**

Run: `python -c "from core.state_v2 import initialize_state; print(initialize_state('s','t')['grounding_score'])"`
Expected: `0.0`

**Step 3: Commit**

```bash
git add core/state_v2.py
git commit -m "feat: add grounding_score to AgentStateV2"
```

---

### Task 3: Upgrade ReflectionAgent with chunk grounding

**Files:**
- Modify: `agents/reflection_agent.py`
- Modify: `tests/eval/test_reflection_structured.py`

**Step 1: Write failing tests**

Append to `tests/eval/test_reflection_structured.py`:
```python
def test_reflection_agent_sets_grounding_score(monkeypatch):
    from agents.reflection_agent import ReflectionAgent
    from core.state_v2 import initialize_state

    monkeypatch.setattr(
        "agents.reflection_agent.invoke_json",
        lambda *a, **kw: {
            "is_relevant": True,
            "has_hallucinations": False,
            "failure_category": "none",
            "suggested_focus": "",
            "confidence": 0.85,
            "grounding_score": 0.9,
            "feedback": "well grounded",
        },
    )
    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin?"
    state["generation"] = "Aspirin inhibits COX-1 and COX-2."
    state["documents"] = []

    result = ReflectionAgent(state)
    assert result["grounding_score"] == 0.9


def test_reflection_agent_detects_hallucination_via_grounding(monkeypatch):
    from agents.reflection_agent import ReflectionAgent
    from core.state_v2 import initialize_state

    monkeypatch.setattr(
        "agents.reflection_agent.invoke_json",
        lambda *a, **kw: {
            "is_relevant": True,
            "has_hallucinations": True,
            "failure_category": "hallucination",
            "suggested_focus": "aspirin mechanism COX inhibition",
            "confidence": 0.3,
            "grounding_score": 0.2,
            "feedback": "answer claims not found in chunks",
        },
    )
    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin?"
    state["generation"] = "Aspirin cures cancer."
    state["documents"] = []

    result = ReflectionAgent(state)
    assert result["grounding_score"] == 0.2
    assert result["reflection_failure_category"] == "hallucination"
    assert result["needs_retry"] is True
```

**Step 2: Run to verify they fail**

Run: `pytest tests/eval/test_reflection_structured.py -v -k "grounding" 2>&1`
Expected: FAIL

**Step 3: Update ReflectionResult model**

In `agents/reflection_agent.py`, add `grounding_score` field to `ReflectionResult`:
```python
class ReflectionResult(BaseModel):
    is_relevant: bool = True
    has_hallucinations: bool = False
    failure_category: str = "none"
    suggested_focus: str = ""
    confidence: float = 0.0
    grounding_score: float = 1.0   # ADD THIS
    feedback: str = ""
```

Add validator for `grounding_score` (same as `confidence`):
```python
@field_validator('grounding_score', mode='before')
@classmethod
def coerce_grounding_score(cls, v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 1.0
```

**Step 4: Update reflection prompt to include chunk context**

Replace `_REFLECTION_PROMPT` with a version that accepts chunk context:
```python
_REFLECTION_PROMPT = """Evaluate this medical Q&A pair with access to the source chunks used to generate the answer.

Question: {question}
Answer: {answer}

Source chunks used for this answer:
{chunk_context}

Judge the response on these criteria:
1. Is the answer relevant to the question?
2. Does it contain hallucinations? Specifically: are there claims in the answer that CANNOT be traced to any of the source chunks above?
3. Is it complete enough to be useful?
4. Is it safe (no dangerous unsupported medical advice)?

Return JSON ONLY with these keys:
- "is_relevant": boolean
- "has_hallucinations": boolean — true if any answer claim is absent from the source chunks
- "failure_category": one of "none", "irrelevant", "hallucination", "incomplete", "unsafe"
- "suggested_focus": if the answer is incomplete or irrelevant, write a specific search query that would find the missing information. Otherwise empty string.
- "confidence": 0.0-1.0 score for overall answer quality
- "grounding_score": 0.0-1.0 — what fraction of the answer's key claims are supported by the source chunks (1.0 = fully grounded, 0.0 = no grounding)
- "feedback": brief explanation of your judgment
"""
```

**Step 5: Update ReflectionAgent to pass chunk context**

In `ReflectionAgent`, build chunk context from `state['documents']` before the judge call:
```python
def _build_chunk_context(docs: list, max_chars: int = 2000) -> str:
    """Build a condensed view of retrieved chunks for the reflection judge."""
    if not docs:
        return "(no source chunks — answer was generated from memory or cache)"
    parts = []
    total = 0
    for i, doc in enumerate(docs[:5]):
        snippet = doc.page_content[:400].strip()
        section = doc.metadata.get("section", "unknown")
        part = f"[Chunk {i+1} — {section}]\n{snippet}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)
```

Then in `ReflectionAgent`:
```python
def ReflectionAgent(state: AgentStateV2) -> AgentStateV2:
    question = state.get('question', '')
    answer = state.get('generation', '')
    docs = state.get('documents', [])

    chunk_context = _build_chunk_context(docs)
    judge_prompt = _REFLECTION_PROMPT.format(
        question=question,
        answer=answer,
        chunk_context=chunk_context,
    )
    raw = invoke_json(judge_prompt, system=_REFLECTION_SYSTEM)
    result = _parse_reflection_response(raw)

    # ... existing attempts/retry logic ...

    state['grounding_score'] = result.grounding_score   # ADD
    # ... rest of state assignments unchanged ...
```

Also update logger to include `grounding_score`.

**Step 6: Run all reflection tests**

Run: `pytest tests/eval/test_reflection_structured.py -v 2>&1`
Expected: all PASS

**Step 7: Commit**

```bash
git add agents/reflection_agent.py core/state_v2.py tests/eval/test_reflection_structured.py
git commit -m "feat: reflection agent cross-checks answer against source chunks for grounding"
```

---

### Task 4: Update docs

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

Architecture note: "ReflectionAgent now receives condensed source chunks and judges grounding_score (0–1). Answers with ungrounded claims trigger hallucination failure_category and retry."

Log entry: `2026-03-17 — Hallucination detection via chunk grounding added to ReflectionAgent.`
