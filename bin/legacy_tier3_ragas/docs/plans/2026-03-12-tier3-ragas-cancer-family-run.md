# Tier 3 Cancer-Family RAGAS Run Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run one temporary Tier 3 RAGAS experiment on a tightly related cancer-family document slice using explicit RAGAS query and transform profiles instead of the default broad document mix.

**Architecture:** Keep the current document-first Tier 3 pipeline. Add a temporary hardcoded source-doc selector for four cancer-family article docs, add an explicit RAGAS query distribution that excludes abstract multi-hop, and add an explicit transform profile tuned for short-to-medium same-family article docs. Keep this isolated to `eval/generate_tier3_rag.py` so the app runtime and retrieval schema stay untouched.

**Tech Stack:** Python, RAGAS 0.3.2, LangChain document wrappers, pytest

---

### Task 1: Lock The Experiment Contract In Tests

**Files:**
- Modify: `tests/eval/test_tier3_generator_contract.py`
- Modify: `eval/generate_tier3_rag.py`

**Step 1: Write the failing tests**

Add tests for:
- `_select_experiment_source_documents()` keeping only the four cancer-family titles
- `_build_tier3_query_distribution()` excluding `MultiHopAbstractQuerySynthesizer`
- `generate_testset()` passing explicit `transforms=` and `query_distribution=` into `generate_with_langchain_docs(...)`

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/eval/test_tier3_generator_contract.py -k "cancer_family or query_distribution or explicit_ragas_profiles"
```

Expected: FAIL because the selector/profile helpers do not exist yet and `generate_testset()` still uses default RAGAS behavior.

**Step 3: Write minimal implementation**

Add helper functions in `eval/generate_tier3_rag.py`:
- `_select_experiment_source_documents()`
- `_build_tier3_query_distribution()`
- `_build_tier3_transforms()`

Wire them into the live generation path.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest -q tests/eval/test_tier3_generator_contract.py -k "cancer_family or query_distribution or explicit_ragas_profiles"
```

Expected: PASS

### Task 2: Narrow The Source Documents To The Temporary Cancer Family

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_generator_contract.py`

**Step 1: Add a temporary hardcoded title allowlist**

Use exactly:
- `Cancer`
- `Cancer therapy, definitive`
- `Cancer therapy, palliative`
- `Cancer therapy, supportive`

Apply the selector after article-doc construction and after cached article-doc loading.

**Step 2: Fail fast on bad selection**

If the experiment filter leaves fewer than 2 source docs, raise a clear error instead of silently falling back to broad generation.

**Step 3: Re-run focused tests**

Run:

```bash
pytest -q tests/eval/test_tier3_generator_contract.py -k "cancer_family"
```

Expected: PASS

### Task 3: Replace Default RAGAS Profiles For This Experiment

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_generator_contract.py`

**Step 1: Build a narrow query distribution**

Use installed RAGAS synthesizers:
- `SingleHopSpecificQuerySynthesizer`
- `MultiHopSpecificQuerySynthesizer`

Exclude:
- `MultiHopAbstractQuerySynthesizer`

Bias toward same-family synthesis with a heavier weight on `MultiHopSpecific`.

**Step 2: Build a narrow transform profile**

Use explicit RAGAS transforms from the installed `0.3.2` module layout instead of `default_transforms()`:
- `SummaryExtractor`
- `EmbeddingExtractor`
- `ThemesExtractor`
- `NERExtractor`
- `CosineSimilarityBuilder`
- `OverlapScoreBuilder`
- `CustomNodeFilter`

Do not use `HeadlinesExtractor` or `HeadlineSplitter` for this temporary slice.

**Step 3: Pass both profiles explicitly**

When `generate_with_langchain_docs(...)` is used, pass:
- `transforms=...`
- `query_distribution=...`

**Step 4: Re-run focused tests**

Run:

```bash
pytest -q tests/eval/test_tier3_generator_contract.py -k "query_distribution or explicit_ragas_profiles"
```

Expected: PASS

### Task 4: Update Docs For The Temporary Experiment Boundary

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update current-state**

Document that:
- the active Tier 3 generation substrate is article docs
- a temporary hardcoded cancer-family experiment path exists
- the experiment uses Tier 3-specific RAGAS query/transform profiles

**Step 2: Append implementation log**

Add a concise entry covering:
- temporary narrow cancer-family source selection
- explicit query distribution excluding abstract multi-hop
- explicit transform profile for the experiment

**Step 3: Verify docs changed cleanly**

Run:

```bash
git diff -- docs/architecture/current-state.md docs/changes/implementation-log.md
```

Expected: only the experiment-related updates

### Task 5: Verify The Experiment End To End

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_generator_contract.py`

**Step 1: Run focused contract coverage**

```bash
pytest -q tests/eval/test_tier3_generator_contract.py
```

Expected: PASS

**Step 2: Run syntax verification**

```bash
python3 -m py_compile eval/generate_tier3_rag.py
```

Expected: PASS

**Step 3: Print the exact live experiment command**

Use a fresh `/tmp` output set and the Tier 3-only OpenAI settings already configured in `.env`:

```bash
.venv/bin/python -m eval.generate_tier3_rag \
  --size 4 \
  --article-limit 12 \
  --article-corpus /tmp/tier3_article_corpus.cancer.jsonl \
  --raw-output /tmp/tier3_rag.cancer.raw.jsonl \
  --output /tmp/tier3_rag.cancer.jsonl
```

Expected: the run should log only the four cancer-family source docs and should no longer rely on default abstract multi-hop generation.
