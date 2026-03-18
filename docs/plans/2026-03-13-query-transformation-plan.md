# Query Transformation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve retrieval recall by adding medical term normalization and conditional step-back query generation to the rewriter agent, with Pydantic-validated structured output and retriever merge logic.

**Architecture:** Single LLM call produces structured JSON (`optimized_query` + optional `stepback_query`) validated by Pydantic. The retriever conditionally searches both queries, deduplicates by `chunk_id`, and BM25 reranks the combined pool against `optimized_query`. Tiered graceful degradation ensures the pipeline never blocks on parse failures.

**Tech Stack:** Python, Pydantic, LangGraph, pytest, existing `invoke_json()` from `tools/llm_client.py`

---

### Task 1: Add `stepback_query` to state

**Files:**
- Modify: `core/state_v2.py:7-34` (AgentStateV2 TypedDict)
- Modify: `core/state_v2.py:37-66` (initialize_state)
- Modify: `core/state_v2.py:69-94` (reset_query_state)

**Step 1: Add the field to AgentStateV2**

In `core/state_v2.py`, add after line 12 (`optimized_query: str`):

```python
    stepback_query: str
```

**Step 2: Add initialization**

In `initialize_state()`, add after line 43 (`'optimized_query': '',`):

```python
        'stepback_query': '',
```

In `reset_query_state()`, add after line 74 (`'optimized_query': '',`):

```python
            'stepback_query': '',
```

**Step 3: Run tests**

Run: `python -m pytest tests/ -x -q`
Expected: ALL PASS (adding a TypedDict field is backwards compatible)

**Step 4: Commit**

```bash
git add core/state_v2.py
git commit -m "feat: add stepback_query field to AgentStateV2"
```

---

### Task 2: Pydantic RewriteResult model and parsing logic

**Files:**
- Modify: `agents/query_rewriter_agent.py`
- Create: `tests/rag/test_query_rewriter_contract.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_query_rewriter_contract.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_rewrite_result_validates_well_formed_json():
    from agents.query_rewriter_agent import RewriteResult

    data = {
        "optimized_query": "aspirin mechanism of action for myocardial infarction",
        "stepback_query": "pharmacology of antiplatelet agents",
        "reasoning": "expanded medical terms",
    }
    result = RewriteResult.model_validate(data)
    assert result.optimized_query == "aspirin mechanism of action for myocardial infarction"
    assert result.stepback_query == "pharmacology of antiplatelet agents"
    assert result.reasoning == "expanded medical terms"


def test_rewrite_result_provides_defaults_for_missing_fields():
    from agents.query_rewriter_agent import RewriteResult

    data = {"optimized_query": "test query"}
    result = RewriteResult.model_validate(data)
    assert result.optimized_query == "test query"
    assert result.stepback_query == ""
    assert result.reasoning == ""


def test_rewrite_result_requires_optimized_query():
    from agents.query_rewriter_agent import RewriteResult
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError):
        RewriteResult.model_validate({})


def test_parse_rewrite_response_handles_valid_json():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = '{"optimized_query": "test", "stepback_query": "broad test"}'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.optimized_query == "test"
    assert result.stepback_query == "broad test"


def test_parse_rewrite_response_handles_malformed_json():
    from agents.query_rewriter_agent import parse_rewrite_response

    result = parse_rewrite_response("not json at all", fallback_query="original question")
    assert result.optimized_query == "not json at all"
    assert result.stepback_query == ""


def test_parse_rewrite_response_handles_empty_string():
    from agents.query_rewriter_agent import parse_rewrite_response

    result = parse_rewrite_response("", fallback_query="original question")
    assert result.optimized_query == "original question"
    assert result.stepback_query == ""


def test_parse_rewrite_response_handles_partial_json():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = '{"optimized_query": "good query", "stepback_query": 123}'
    result = parse_rewrite_response(raw, fallback_query="original")
    # Pydantic should coerce 123 to string or fail gracefully
    assert result.optimized_query == "good query"


def test_parse_rewrite_response_extracts_json_from_markdown():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = 'Here is the result:\n```json\n{"optimized_query": "extracted query"}\n```'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.optimized_query == "extracted query"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_query_rewriter_contract.py -v`
Expected: FAIL with ImportError

**Step 3: Implement RewriteResult and parse_rewrite_response**

Replace the entire content of `agents/query_rewriter_agent.py`:

```python
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, field_validator

from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_json

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    """Structured output from the query rewriter."""
    optimized_query: str
    stepback_query: str = ""
    reasoning: str = ""

    @field_validator('optimized_query', 'stepback_query', 'reasoning', mode='before')
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return ""
        return str(v).strip()


_REWRITER_SYSTEM = (
    "You are a medical query rewriting assistant for a textbook retrieval system. "
    "You transform user questions into optimized search queries. "
    "Respond ONLY with valid JSON, no other text."
)

_REWRITER_PROMPT = """Rewrite the user's question into an optimized medical search query.

INSTRUCTIONS:
1. RESOLVE conversational references: replace "it", "that drug", "the same condition" with explicit terms from conversation history.
2. EXPAND medical abbreviations: MI → myocardial infarction, BP → blood pressure, CHF → congestive heart failure, DVT → deep vein thrombosis, PE → pulmonary embolism, COPD → chronic obstructive pulmonary disease, DM → diabetes mellitus, HTN → hypertension, CAD → coronary artery disease, etc.
3. ADD medical synonyms in parentheses where helpful: "heart attack (myocardial infarction)", "blood thinner (anticoagulant)", "high blood pressure (hypertension)".
4. PRESERVE specificity: keep dosages, populations, timeframes, and drug names exactly as stated.
5. STEP-BACK QUERY: If the question involves mechanisms of action, drug comparisons, treatment rationale, dosage-specific lookups, or timeline-specific questions, generate a broader principle-level query in stepback_query. Otherwise leave stepback_query empty.

Conversation context:
{history_text}

User question:
{question}

Return JSON:
{{"optimized_query": "...", "stepback_query": "...", "reasoning": "..."}}

Examples:
- "what does aspirin do for heart attacks?" → {{"optimized_query": "aspirin mechanism of action for myocardial infarction (heart attack)", "stepback_query": "pharmacology of antiplatelet agents in cardiovascular disease", "reasoning": "expanded synonym, step-back to drug class"}}
- "what is hypertension?" → {{"optimized_query": "hypertension (high blood pressure) definition and overview", "stepback_query": "", "reasoning": "simple definition lookup, no step-back needed"}}
- "how does it compare to tylenol?" (history about ibuprofen) → {{"optimized_query": "ibuprofen compared to acetaminophen (tylenol) efficacy and safety", "stepback_query": "NSAID versus non-NSAID analgesic comparison", "reasoning": "resolved 'it' to ibuprofen, expanded tylenol, step-back to drug class comparison"}}
"""


def parse_rewrite_response(raw: str, fallback_query: str) -> RewriteResult:
    """Parse LLM rewrite response with tiered graceful degradation."""
    if not raw or not raw.strip():
        return RewriteResult(optimized_query=fallback_query)

    text = raw.strip()

    # Tier 1: Try invoke_json-style parsing + Pydantic validation
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown fences or embedded braces
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    if parsed is not None:
        try:
            return RewriteResult.model_validate(parsed)
        except Exception:
            # Tier 2: Pydantic failed, extract what we can
            if isinstance(parsed, dict) and parsed.get('optimized_query'):
                return RewriteResult(optimized_query=str(parsed['optimized_query']).strip())

    # Tier 3: JSON parsing failed entirely, use raw text as query
    return RewriteResult(optimized_query=text)


def QueryRewriterAgent(state: AgentStateV2) -> AgentStateV2:
    question = state.get('question', '')
    history = state.get('conversation_history', [])[-6:]
    history_text = '\n'.join(
        [f"{item.get('role')}: {item.get('content')}" for item in history]
    ) if history else '(no prior conversation)'

    prompt = _REWRITER_PROMPT.format(
        history_text=history_text,
        question=question,
    )

    raw_json = invoke_json(prompt, system=_REWRITER_SYSTEM)

    # invoke_json returns a dict (already parsed) or {} on failure
    if raw_json:
        try:
            result = RewriteResult.model_validate(raw_json)
        except Exception:
            result = RewriteResult(
                optimized_query=str(raw_json.get('optimized_query', question)).strip() or question
            )
    else:
        # Tier 4: LLM call returned nothing
        result = RewriteResult(optimized_query=question)

    state['optimized_query'] = result.optimized_query or question
    state['stepback_query'] = result.stepback_query
    logger.info(
        "query_rewrite_complete",
        extra={
            "original": question,
            "optimized": result.optimized_query,
            "stepback": result.stepback_query,
            "stepback_generated": bool(result.stepback_query),
            "reasoning": result.reasoning,
        },
    )
    return state
```

**Step 4: Run tests**

Run: `python -m pytest tests/rag/test_query_rewriter_contract.py -v`
Expected: ALL PASS

Run: `python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/query_rewriter_agent.py tests/rag/test_query_rewriter_contract.py
git commit -m "feat: structured query rewriter with medical normalization and Pydantic validation"
```

---

### Task 3: Retriever step-back merge logic

**Files:**
- Modify: `agents/retriever_agent.py:13-31` (RetrieverAgent function)
- Create: `tests/rag/test_retriever_stepback.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_retriever_stepback.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_doc(chunk_id: str, content: str):
    from langchain_core.documents import Document
    return Document(page_content=content, metadata={"chunk_id": chunk_id})


def test_merge_and_deduplicate_removes_duplicates():
    from agents.retriever_agent import _merge_and_deduplicate

    primary = [_make_doc("c1", "doc one"), _make_doc("c2", "doc two")]
    stepback = [_make_doc("c2", "doc two duplicate"), _make_doc("c3", "doc three")]

    merged = _merge_and_deduplicate(primary, stepback)
    chunk_ids = [d.metadata["chunk_id"] for d in merged]
    assert chunk_ids == ["c1", "c2", "c3"]
    # Primary version of c2 is kept (not stepback version)
    assert merged[1].page_content == "doc two"


def test_merge_and_deduplicate_preserves_order_primary_first():
    from agents.retriever_agent import _merge_and_deduplicate

    primary = [_make_doc("c1", "first"), _make_doc("c2", "second")]
    stepback = [_make_doc("c3", "third"), _make_doc("c4", "fourth")]

    merged = _merge_and_deduplicate(primary, stepback)
    chunk_ids = [d.metadata["chunk_id"] for d in merged]
    assert chunk_ids == ["c1", "c2", "c3", "c4"]


def test_merge_and_deduplicate_handles_empty_stepback():
    from agents.retriever_agent import _merge_and_deduplicate

    primary = [_make_doc("c1", "only")]
    merged = _merge_and_deduplicate(primary, [])
    assert len(merged) == 1
    assert merged[0].metadata["chunk_id"] == "c1"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag/test_retriever_stepback.py -v`
Expected: FAIL with ImportError

**Step 3: Add merge logic and update RetrieverAgent**

Add `_merge_and_deduplicate` function to `agents/retriever_agent.py` after the imports (after line 10):

```python
def _merge_and_deduplicate(
    primary: list[Document],
    stepback: list[Document],
) -> list[Document]:
    """Merge two document lists, deduplicating by chunk_id. Primary docs take precedence."""
    seen: set[str] = set()
    merged: list[Document] = []
    for doc in primary:
        cid = doc.metadata.get('chunk_id', '')
        if cid not in seen:
            seen.add(cid)
            merged.append(doc)
    for doc in stepback:
        cid = doc.metadata.get('chunk_id', '')
        if cid not in seen:
            seen.add(cid)
            merged.append(doc)
    return merged
```

Update `RetrieverAgent` function — replace lines 13-31 with:

```python
def RetrieverAgent(state: AgentStateV2) -> AgentStateV2:
    query = state.get('optimized_query') or state.get('question', '')
    repo = state.get('vector_repo')
    if repo is None:
        state['documents'] = []
        state['retrieval_confidence'] = 0.0
        state['retrieval_candidate_count'] = 0
        state['post_retrieval_route'] = 'executor'
        state['route_decision_reason'] = 'no_vector_repo'
        return state

    docs = repo.hybrid_search(query=query, query_embedding=embed_query(query), k=20)

    # Conditional step-back: search broader query and merge
    stepback = state.get('stepback_query', '')
    if stepback:
        stepback_docs = repo.hybrid_search(
            query=stepback, query_embedding=embed_query(stepback), k=10,
        )
        docs = _merge_and_deduplicate(docs, stepback_docs)
        logger.info(
            "retriever_stepback_merge",
            extra={
                "primary_docs": 20,
                "stepback_docs": len(stepback_docs),
                "after_dedup": len(docs),
            },
        )

    ranked, scores = rerank(query, docs)
    state['documents'] = ranked[:5]
    state['retrieval_candidate_count'] = len(ranked)
    state['retrieval_confidence'] = scores[0] if scores else 0.0
    state['source'] = 'PostgreSQL pgvector + Hybrid Retrieval'
    _decide_post_retrieval_route(state)
    return state
```

Also add `import logging` and `logger = logging.getLogger(__name__)` at the top of the file after the existing imports.

**Step 4: Run tests**

Run: `python -m pytest tests/rag/test_retriever_stepback.py -v`
Expected: ALL PASS

Run: `python -m pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/retriever_agent.py tests/rag/test_retriever_stepback.py
git commit -m "feat: retriever step-back merge with deduplication by chunk_id"
```

---

### Task 4: Update documentation

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update current-state.md**

In the "Retrieval and answer generation" section, update the retriever description to mention step-back query support and medical normalization.

In the "Current important flows > Chat request flow" section, update step 3 to mention the rewriter now produces structured output with medical normalization and conditional step-back.

**Step 2: Add implementation log entry**

Append a new entry to `docs/changes/implementation-log.md`:

```markdown
## 2026-03-13 — Query transformation: medical normalization + conditional step-back

What changed:
- Upgraded the query rewriter to produce structured JSON output (Pydantic-validated) with medical term normalization, abbreviation expansion, synonym addition, and conditional step-back query generation.
- The retriever now conditionally searches both the optimized query and the step-back query, merges and deduplicates by chunk_id, then BM25 reranks the combined pool against the specific query.
- Added `stepback_query` field to AgentStateV2.

Why:
- Retrieval recall was limited by vocabulary mismatch between user queries (colloquial medical terms) and textbook language (formal medical terminology).
- Conceptual questions benefit from broader principle-level retrieval that the specific query alone cannot surface.

Tradeoff:
- Step-back search adds one extra embedding + hybrid search call when generated (~50% of queries based on LLM judgment). This increases retrieval latency for those queries.
- The rewriter now uses structured JSON output which adds a small parse failure surface, mitigated by tiered graceful degradation.

Must stay true:
- The rewriter must never block the pipeline on parse failure — graceful degradation to original query is required.
- BM25 reranking must always score against `optimized_query` (specific), not `stepback_query` (broad).
- Step-back search is supplementary (k=10) to the primary search (k=20), not a replacement.
```

**Step 3: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: update architecture and implementation log for query transformation"
```

---

### Task 5: Run full test suite and verify

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Verify the full commit history**

Run: `git log --oneline -5`
Expected: 4 new commits for Tasks 1-4
