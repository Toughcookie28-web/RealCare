# Pipeline Upgrade Phase B: Metadata-Powered Retrieval + Executor Optimization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Use structured query understanding (slots, intents, query_context) and chunk metadata (keywords, medical_entities, context_summary) to improve retrieval precision, answer quality, and source transparency.

**Architecture:** Five features plugging into existing retriever and executor hooks. No new LLM calls, no new services, no schema migrations. All features degrade gracefully when metadata is absent.

**Tech Stack:** Python, Pydantic, LangChain Document, pytest

---

### Task 1: State Field — slot_coverage

**Files:**
- Modify: `core/state_v2.py`
- Test: `tests/rag/test_state_v2_fields.py`

**Step 1: Write the failing test**

```python
# tests/rag/test_state_v2_fields.py — append to existing file

def test_state_has_slot_coverage():
    state = initialize_state(session_id='s1', trace_id='t1')
    assert 'slot_coverage' in state
    assert state['slot_coverage'] == 0.0


def test_reset_clears_slot_coverage():
    state = initialize_state(session_id='s1', trace_id='t1')
    state['slot_coverage'] = 0.8
    reset_query_state(state, 'new question')
    assert state['slot_coverage'] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_state_v2_fields.py -v -k slot_coverage`
Expected: FAIL — `slot_coverage` not in state

**Step 3: Write minimal implementation**

In `core/state_v2.py`:
- Add `slot_coverage: float` to `AgentStateV2` TypedDict
- Add `'slot_coverage': 0.0` to `initialize_state()`
- Add `'slot_coverage': 0.0` to `reset_query_state()`

**Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_state_v2_fields.py -v -k slot_coverage`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/rag/ -v`
Expected: all pass

**Step 6: Commit**

```bash
git add core/state_v2.py tests/rag/test_state_v2_fields.py
git commit -m "feat: add slot_coverage to AgentStateV2 for Phase B retrieval validation"
```

---

### Task 2: Pre-Retrieval Metadata Boost

**Files:**
- Modify: `agents/retriever_agent.py`
- Create: `tests/rag/test_metadata_boost.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_metadata_boost.py

from langchain_core.documents import Document
from agents.retriever_agent import metadata_boost


def _doc(content, chunk_id, keywords=None, medical_entities=None):
    meta = {'chunk_id': chunk_id}
    if keywords is not None:
        meta['keywords'] = keywords
    if medical_entities is not None:
        meta['medical_entities'] = medical_entities
    return Document(page_content=content, metadata=meta)


def test_boost_adds_score_for_matching_keyword():
    docs = [
        _doc('about aspirin', 'c1', keywords=['aspirin', 'antiplatelet']),
        _doc('about statins', 'c2', keywords=['statin', 'cholesterol']),
    ]
    boosted = metadata_boost(docs, {'drug': 'aspirin'})
    assert boosted[0][1] > 0.0  # c1 gets boost
    assert boosted[1][1] == 0.0  # c2 gets no boost


def test_boost_matches_medical_entities():
    docs = [
        _doc('heart disease', 'c1', medical_entities={
            'conditions': ['myocardial infarction'],
            'drugs': [],
        }),
    ]
    boosted = metadata_boost(docs, {'condition': 'myocardial infarction'})
    assert boosted[0][1] > 0.0


def test_boost_empty_slots_returns_zero_scores():
    docs = [_doc('anything', 'c1', keywords=['aspirin'])]
    boosted = metadata_boost(docs, {})
    assert boosted[0][1] == 0.0


def test_boost_no_metadata_returns_zero_scores():
    docs = [_doc('no metadata', 'c1')]
    boosted = metadata_boost(docs, {'drug': 'aspirin'})
    assert boosted[0][1] == 0.0


def test_boost_case_insensitive():
    docs = [_doc('about aspirin', 'c1', keywords=['Aspirin', 'Antiplatelet'])]
    boosted = metadata_boost(docs, {'drug': 'aspirin'})
    assert boosted[0][1] > 0.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_metadata_boost.py -v`
Expected: FAIL — `metadata_boost` not found

**Step 3: Write minimal implementation**

In `agents/retriever_agent.py`, add:

```python
_BOOST_SCORE = 0.15

def metadata_boost(
    docs: list[Document], slots: dict[str, str]
) -> list[tuple[Document, float]]:
    """Score docs by metadata overlap with slots. Returns (doc, boost_score) pairs."""
    if not slots:
        return [(doc, 0.0) for doc in docs]

    slot_values = {v.lower() for v in slots.values() if v}
    if not slot_values:
        return [(doc, 0.0) for doc in docs]

    result = []
    for doc in docs:
        meta = doc.metadata or {}
        searchable: list[str] = []

        # Collect keywords
        for kw in meta.get('keywords', []):
            searchable.append(str(kw).lower())

        # Collect medical entities (dict of lists)
        entities = meta.get('medical_entities', {})
        if isinstance(entities, dict):
            for entity_list in entities.values():
                if isinstance(entity_list, list):
                    for e in entity_list:
                        searchable.append(str(e).lower())

        # Check for any slot value match
        boost = 0.0
        for sv in slot_values:
            if any(sv in s for s in searchable):
                boost = _BOOST_SCORE
                break

        result.append((doc, boost))
    return result
```

**Step 4: Integrate into RetrieverAgent**

In `RetrieverAgent`, after `docs = _merge_and_deduplicate(...)` (or after initial hybrid_search if no stepback), before `rerank()`:

```python
    # Apply metadata boost from slots
    slots = state.get('slots', {})
    boost_pairs = metadata_boost(docs, slots)

    # Rerank with BM25
    ranked, scores = rerank(query, docs)

    # Add boost scores to BM25 scores
    boost_map = {id(doc): boost for doc, boost in boost_pairs}
    final_scores = [s + boost_map.get(id(doc), 0.0) for doc, s in zip(ranked, scores)]

    # Re-sort by combined score
    combined = sorted(zip(ranked, final_scores), key=lambda x: x[1], reverse=True)
    ranked = [doc for doc, _ in combined]
    scores = [s for _, s in combined]
```

**Step 5: Run tests**

Run: `pytest tests/rag/test_metadata_boost.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agents/retriever_agent.py tests/rag/test_metadata_boost.py
git commit -m "feat: add pre-retrieval metadata boost using slots in retriever"
```

---

### Task 3: Post-Retrieval Slot Validation + Confidence Adjustment

**Files:**
- Modify: `agents/retriever_agent.py`
- Create: `tests/rag/test_slot_validation.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_slot_validation.py

from langchain_core.documents import Document
from agents.retriever_agent import compute_slot_coverage, adjust_confidence


def _doc(content, keywords=None, medical_entities=None):
    meta = {}
    if keywords is not None:
        meta['keywords'] = keywords
    if medical_entities is not None:
        meta['medical_entities'] = medical_entities
    return Document(page_content=content, metadata=meta)


def test_slot_coverage_full_match():
    docs = [_doc('aspirin for hypertension',
                 keywords=['aspirin', 'hypertension'])]
    coverage = compute_slot_coverage(docs, {'drug': 'aspirin', 'condition': 'hypertension'})
    assert coverage == 1.0


def test_slot_coverage_partial_match():
    docs = [_doc('aspirin info', keywords=['aspirin'])]
    coverage = compute_slot_coverage(docs, {'drug': 'aspirin', 'condition': 'hypertension'})
    assert coverage == 0.5


def test_slot_coverage_no_match():
    docs = [_doc('statins info', keywords=['statin'])]
    coverage = compute_slot_coverage(docs, {'drug': 'aspirin', 'condition': 'hypertension'})
    assert coverage == 0.0


def test_slot_coverage_empty_slots():
    docs = [_doc('anything')]
    coverage = compute_slot_coverage(docs, {})
    assert coverage == 1.0  # vacuously true


def test_slot_coverage_checks_content_too():
    docs = [_doc('aspirin is used for pain')]
    coverage = compute_slot_coverage(docs, {'drug': 'aspirin'})
    assert coverage == 1.0  # found in content


def test_adjust_confidence_low_coverage_specific_intent():
    conf = adjust_confidence(0.8, slot_coverage=0.2, turn_intent='dosage_lookup')
    assert conf < 0.8  # reduced


def test_adjust_confidence_high_coverage_unchanged():
    conf = adjust_confidence(0.8, slot_coverage=0.9, turn_intent='dosage_lookup')
    assert conf == 0.8


def test_adjust_confidence_generic_intent_unchanged():
    conf = adjust_confidence(0.8, slot_coverage=0.1, turn_intent='definition')
    assert conf == 0.8  # definition is not "specific" enough to trigger


def test_adjust_confidence_empty_intent_unchanged():
    conf = adjust_confidence(0.8, slot_coverage=0.1, turn_intent='')
    assert conf == 0.8
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_slot_validation.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `agents/retriever_agent.py`, add:

```python
_SPECIFIC_INTENTS = {'dosage_lookup', 'drug_comparison', 'side_effects', 'mechanism_of_action', 'differential_diagnosis'}
_COVERAGE_THRESHOLD = 0.3
_CONFIDENCE_PENALTY = 0.7  # multiply by this when coverage is low


def compute_slot_coverage(docs: list[Document], slots: dict[str, str]) -> float:
    """Fraction of non-empty slot values found in docs' content or metadata."""
    slot_values = [v.lower() for v in slots.values() if v]
    if not slot_values:
        return 1.0  # vacuously true

    # Build searchable text pool from all docs
    pool: list[str] = []
    for doc in docs:
        pool.append(doc.page_content.lower())
        meta = doc.metadata or {}
        for kw in meta.get('keywords', []):
            pool.append(str(kw).lower())
        entities = meta.get('medical_entities', {})
        if isinstance(entities, dict):
            for entity_list in entities.values():
                if isinstance(entity_list, list):
                    for e in entity_list:
                        pool.append(str(e).lower())

    pool_text = ' '.join(pool)
    matched = sum(1 for sv in slot_values if sv in pool_text)
    return matched / len(slot_values)


def adjust_confidence(
    confidence: float, slot_coverage: float, turn_intent: str
) -> float:
    """Lower confidence when slot coverage is poor and intent is specific."""
    if turn_intent not in _SPECIFIC_INTENTS:
        return confidence
    if slot_coverage < _COVERAGE_THRESHOLD:
        return confidence * _CONFIDENCE_PENALTY
    return confidence
```

**Step 4: Integrate into RetrieverAgent**

After setting `state['documents']` and `state['retrieval_confidence']`:

```python
    # Post-retrieval slot validation
    slots = state.get('slots', {})
    turn_intent = state.get('turn_intent', '')
    slot_cov = compute_slot_coverage(state['documents'], slots)
    state['slot_coverage'] = slot_cov
    state['retrieval_confidence'] = adjust_confidence(
        state['retrieval_confidence'], slot_cov, turn_intent
    )
```

**Step 5: Run tests**

Run: `pytest tests/rag/test_slot_validation.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agents/retriever_agent.py tests/rag/test_slot_validation.py
git commit -m "feat: add post-retrieval slot validation and confidence adjustment"
```

---

### Task 4: Executor Prompt Optimization

**Files:**
- Modify: `agents/executor_agent.py`
- Create: `tests/rag/test_executor_prompt_optimization.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_executor_prompt_optimization.py

from agents.executor_agent import _intent_instruction, _render_prompt_optimized


def test_intent_instruction_definition():
    instr = _intent_instruction('definition', {})
    assert 'definition' in instr.lower() or 'summary' in instr.lower()


def test_intent_instruction_mechanism():
    instr = _intent_instruction('mechanism_of_action', {})
    assert 'step' in instr.lower() or 'mechanism' in instr.lower()


def test_intent_instruction_drug_comparison():
    instr = _intent_instruction('drug_comparison', {})
    assert 'compare' in instr.lower() or 'difference' in instr.lower()


def test_intent_instruction_dosage():
    instr = _intent_instruction('dosage_lookup', {})
    assert 'dosage' in instr.lower()


def test_intent_instruction_side_effects():
    instr = _intent_instruction('side_effects', {})
    assert 'side effect' in instr.lower() or 'frequency' in instr.lower()


def test_intent_instruction_unknown_returns_empty():
    instr = _intent_instruction('unknown_intent', {})
    assert instr == ''


def test_intent_instruction_empty_returns_empty():
    instr = _intent_instruction('', {})
    assert instr == ''


def test_render_prompt_includes_query_context():
    prompt = _render_prompt_optimized(
        question='what is aspirin?',
        query='aspirin mechanism',
        summary='',
        facts='',
        context='some context',
        query_context='User wants to understand aspirin mechanism for heart attacks',
        turn_intent='mechanism_of_action',
        session_intent='',
    )
    assert 'User wants to understand aspirin mechanism' in prompt


def test_render_prompt_includes_session_intent():
    prompt = _render_prompt_optimized(
        question='dosage?',
        query='aspirin dosage',
        summary='',
        facts='',
        context='some context',
        query_context='',
        turn_intent='dosage_lookup',
        session_intent='researching cardiovascular pharmacology',
    )
    assert 'cardiovascular pharmacology' in prompt


def test_render_prompt_includes_intent_instruction():
    prompt = _render_prompt_optimized(
        question='what is hypertension?',
        query='hypertension definition',
        summary='',
        facts='',
        context='some context',
        query_context='',
        turn_intent='definition',
        session_intent='',
    )
    assert 'definition' in prompt.lower() or 'summary' in prompt.lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_executor_prompt_optimization.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `agents/executor_agent.py`, add:

```python
_INTENT_INSTRUCTIONS = {
    'definition': 'Provide a clear, structured definition. Start with a one-sentence summary.',
    'mechanism_of_action': 'Explain the mechanism step by step. Use a causal chain.',
    'drug_comparison': 'Compare in a structured format. Highlight key differences.',
    'dosage_lookup': 'Provide specific dosage information. Include population and route if available.',
    'side_effects': 'List side effects organized by frequency or severity.',
    'differential_diagnosis': 'Present as a structured differential with distinguishing features.',
}


def _intent_instruction(turn_intent: str, slots: dict[str, str]) -> str:
    """Return formatting instruction for the given intent."""
    return _INTENT_INSTRUCTIONS.get(turn_intent, '')
```

Update `_PROMPT_TEMPLATE` to a function `_render_prompt_optimized()`:

```python
_PROMPT_TEMPLATE_V2 = """
User question: {question}
Rewritten query: {query}
Conversation summary: {summary}
Known user facts: {facts}
{session_line}
{query_context_line}

Retrieved context:
{context}

The retrieved context may contain prose, Markdown tables, or HTML-like table
structure. Pay careful attention to section headers, table titles/captions,
column headers, and row alignment when extracting facts.

If user asks diagnosis or dangerous dosage, decline diagnosis and perform educational pivot.
{intent_line}
Provide concise, practical, medically safe educational guidance in 3-6 bullet points.
""".strip()


def _render_prompt_optimized(
    question: str, query: str, summary: str, facts: str, context: str,
    query_context: str = '', turn_intent: str = '', session_intent: str = '',
) -> str:
    intent_instr = _intent_instruction(turn_intent, {})
    return _PROMPT_TEMPLATE_V2.format(
        question=question,
        query=query,
        summary=summary,
        facts=facts,
        context=context,
        session_line=f'Session goal: {session_intent}' if session_intent else '',
        query_context_line=f'Query context: {query_context}' if query_context else '',
        intent_line=f'Format instruction: {intent_instr}' if intent_instr else '',
    )
```

Update `ExecutorAgent` to call `_render_prompt_optimized()` instead of `_render_prompt()`, passing the new state fields.

Keep `_render_prompt()` as-is for backward compatibility in cache namespace calculation.

**Step 4: Run tests**

Run: `pytest tests/rag/test_executor_prompt_optimization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agents/executor_agent.py tests/rag/test_executor_prompt_optimization.py
git commit -m "feat: add intent-aware prompt optimization to executor"
```

---

### Task 5: Context Assembly with Metadata Enrichment

**Files:**
- Modify: `agents/executor_agent.py` (in `_build_context_blocks()`)
- Create: `tests/rag/test_context_enrichment.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_context_enrichment.py

from langchain_core.documents import Document
from agents.executor_agent import _build_context_blocks


def _doc(content, section='sec1', context_summary=None, chunk_position=None,
         section_path=None, chunk_index=0):
    meta = {'section': section, 'chunk_index': chunk_index}
    if context_summary is not None:
        meta['context_summary'] = context_summary
    if chunk_position is not None:
        meta['chunk_position'] = chunk_position
    if section_path is not None:
        meta['section_path'] = section_path
    return Document(page_content=content, metadata=meta)


def test_context_summary_prepended():
    docs = [_doc('Aspirin inhibits COX-1.',
                 context_summary='Overview of aspirin pharmacology')]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    assert '[Context: Overview of aspirin pharmacology]' in block_text
    assert 'Aspirin inhibits COX-1.' in block_text


def test_no_context_summary_unchanged():
    docs = [_doc('Aspirin inhibits COX-1.')]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    assert '[Context:' not in block_text
    assert 'Aspirin inhibits COX-1.' in block_text


def test_intro_chunks_ordered_first():
    docs = [
        _doc('Body content', section='pharmacology', chunk_position='body', chunk_index=1),
        _doc('Intro content', section='pharmacology', chunk_position='intro', chunk_index=0),
    ]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    intro_pos = block_text.index('Intro content')
    body_pos = block_text.index('Body content')
    assert intro_pos < body_pos


def test_section_path_in_header():
    docs = [_doc('content', section_path='Ch5 > Cardiovascular > Pharmacology')]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    assert 'Ch5 > Cardiovascular > Pharmacology' in block_text
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_context_enrichment.py -v`
Expected: FAIL (some may pass if section_path is already used as header — check the intro ordering test)

**Step 3: Modify `_build_context_blocks()`**

Changes to make inside `_build_context_blocks()`:

1. **Sort docs so intro chunks come first** (before the grouping loop):
   ```python
   def _position_sort_key(doc):
       pos = (doc.metadata or {}).get('chunk_position', 'body')
       return 0 if pos == 'intro' else 1 if pos == 'body' else 2
   docs = sorted(docs, key=_position_sort_key)
   ```

2. **Prepend context_summary** (inside the chunk content assembly):
   ```python
   context_summary = metadata.get('context_summary')
   if context_summary:
       parts.insert(0, f"[Context: {context_summary}]")
   ```

3. **Section path in header** — already used via `metadata.get('section_path')` fallback in header (line 93-94). Verify it's working.

**Step 4: Run tests**

Run: `pytest tests/rag/test_context_enrichment.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/rag/ -v`
Expected: all pass (verify adjacent expansion tests still work)

**Step 6: Commit**

```bash
git add agents/executor_agent.py tests/rag/test_context_enrichment.py
git commit -m "feat: enrich context assembly with metadata summaries and intro prioritization"
```

---

### Task 6: Answer Grounding with Citations

**Files:**
- Modify: `agents/executor_agent.py`
- Create: `tests/rag/test_answer_citations.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_answer_citations.py

from langchain_core.documents import Document
from agents.executor_agent import _build_citations


def _doc(section=None, page=None, doc_id='book1'):
    meta = {'doc_id': doc_id}
    if section is not None:
        meta['section'] = section
    if page is not None:
        meta['page'] = page
    return Document(page_content='content', metadata=meta)


def test_citations_basic():
    docs = [
        _doc(section='Cardiovascular', page=142),
        _doc(section='Antiplatelet', page=145),
    ]
    citations = _build_citations(docs)
    assert len(citations) == 2
    assert citations[0]['section'] == 'Cardiovascular'
    assert citations[0]['page'] == 142


def test_citations_dedup_same_section_page():
    docs = [
        _doc(section='Cardiovascular', page=142),
        _doc(section='Cardiovascular', page=142),
    ]
    citations = _build_citations(docs)
    assert len(citations) == 1


def test_citations_missing_section():
    docs = [_doc(page=100)]
    citations = _build_citations(docs)
    assert len(citations) == 0  # skip when section missing


def test_citations_missing_page():
    docs = [_doc(section='Cardiovascular')]
    citations = _build_citations(docs)
    assert len(citations) == 1
    assert citations[0].get('page') is None


def test_citations_format_string():
    docs = [
        _doc(section='Pharmacology', page=42),
        _doc(section='Toxicology', page=88),
    ]
    citations = _build_citations(docs)
    text = _format_citations_block(citations)
    assert 'Pharmacology' in text
    assert '42' in text
    assert 'Toxicology' in text


def test_citations_empty_docs():
    citations = _build_citations([])
    assert citations == []
```

Note: also import `_format_citations_block` from executor.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_answer_citations.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `agents/executor_agent.py`, add:

```python
def _build_citations(docs: list) -> list[dict[str, Any]]:
    """Extract deduplicated section/page citations from retrieved docs."""
    seen: set[tuple] = set()
    citations: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        section = meta.get('section')
        if not section:
            continue
        page = meta.get('page')
        key = (section, page)
        if key in seen:
            continue
        seen.add(key)
        citation = {'section': section, 'doc_id': meta.get('doc_id', '')}
        if page is not None:
            citation['page'] = page
        citations.append(citation)
    return citations


def _format_citations_block(citations: list[dict[str, Any]]) -> str:
    """Format citations as a readable Sources block."""
    if not citations:
        return ''
    lines = ['**Sources:**']
    for c in citations:
        parts = [f"Section: \"{c['section']}\""]
        if c.get('page') is not None:
            parts.append(f"Page {c['page']}")
        lines.append(f"- {', '.join(parts)}")
    return '\n'.join(lines)
```

**Step 4: Integrate into ExecutorAgent**

After answer generation (after `state['generation'] = answer`), add:

```python
    # Build citations from retrieved docs
    citations = _build_citations(docs)
    state['citations'] = citations
    if citations:
        citation_block = _format_citations_block(citations)
        state['generation'] = f"{answer}\n\n{citation_block}"
```

**Step 5: Run tests**

Run: `pytest tests/rag/test_answer_citations.py -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `pytest tests/rag/ -v`
Expected: all pass

**Step 7: Commit**

```bash
git add agents/executor_agent.py tests/rag/test_answer_citations.py
git commit -m "feat: add answer grounding with section/page citations"
```

---

### Task 7: Integration Verification + Documentation

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Run full test suite**

Run: `pytest tests/rag/ -v`
Expected: all pass

**Step 2: Update architecture doc**

In `docs/architecture/current-state.md`, update the retrieval and executor sections to mention:
- Pre-retrieval metadata boost using slots
- Post-retrieval slot validation and confidence adjustment
- Intent-aware prompt optimization
- Context assembly with metadata enrichment
- Answer grounding with citations

**Step 3: Update implementation log**

Append to `docs/changes/implementation-log.md`:

```markdown
## 2026-03-14 — Pipeline Upgrade Phase B: Metadata-Powered Retrieval + Executor Optimization

**What changed:**
- Retriever now applies soft metadata boost to chunks matching slots before reranking.
- Post-retrieval slot validation computes slot_coverage and adjusts confidence for specific intents.
- Executor prompt optimized with intent-aware formatting instructions, query_context, session_intent.
- Context assembly prepends context_summary headers and prioritizes intro chunks.
- Answer generation appends deduplicated section/page citations.

**Why:** Use structured query understanding (Phase A) and chunk enrichment metadata to improve retrieval precision, answer formatting, and source transparency.

**Tradeoff:** Features degrade gracefully when enrichment metadata is absent (ENRICH_CHUNKS_WITH_LLM=false). Full benefit requires running enrichment during ingest.

**Must remain true:**
- All features are no-ops when metadata is absent — no regressions without enrichment.
- slot_coverage is observable in state for debugging.
- Citation block is appended to generation, not injected into the prompt.
- Cache namespace must be updated if prompt template changes affect answer determinism.
```

**Step 4: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: update architecture and log for Phase B implementation"
```
