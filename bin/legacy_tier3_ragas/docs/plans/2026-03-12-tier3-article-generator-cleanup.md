# Tier 3 Article-Generator Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the Tier 3 synthetic generator around article-entry RAGAS inputs, make chunk alignment explicitly soft support metadata, and remove obsolete chunk-first experiment code that no longer helps the active pipeline.

**Architecture:** Keep the active path as `article documents -> RAGAS -> raw rows -> schema conversion`. Preserve chunk rows only for optional post-generation alignment and retrieval-oriented metadata. Remove local chunk-first experiment surfaces that are no longer part of the active generation contract.

**Tech Stack:** Python 3.11+, RAGAS 0.2-0.3 APIs, LangChain `Document`, JSONL artifacts, pytest.

---

### Task 1: Make article-doc to chunk alignment explicitly soft

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_generator_contract.py`

**Step 1: Write the failing tests**
- Add a contract that if `aligned_chunk_ids` are present but produce no good row match, alignment falls back to page-span candidates instead of returning nothing.
- Add a contract that rows with valid `ground_truth_contexts` still survive conversion even when `ground_truth_chunk_ids` stays empty.

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'alignment_fallback or missing_chunk_ids'`
Expected: FAIL

**Step 3: Write minimal implementation**
- In `_candidate_chunk_rows_for_source_doc()`, do not treat `aligned_chunk_ids` as an absolute gate.
- In `_infer_ground_truth_alignment()`, if aligned candidates do not produce a passing chunk-row match, retry against page-span candidates before giving up.
- Keep missing chunk IDs non-fatal.

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'alignment_fallback or missing_chunk_ids'`
Expected: PASS

**Step 5: Commit**
```bash
git add eval/generate_tier3_rag.py tests/eval/test_tier3_generator_contract.py
git commit -m "fix: soften tier3 article alignment fallback"
```

### Task 2: Remove the generator's chunk-as-document bridge if no longer needed

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_generator_contract.py`
- Check: `docs/architecture/current-state.md`

**Step 1: Write the failing test**
- Add a contract that the active RAGAS generation path never calls `load_documents()` for source docs.
- Add a contract that chunk loading exists only for alignment rows.

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'source_docs_not_chunk_docs'`
Expected: FAIL

**Step 3: Write minimal implementation**
- Either:
  - keep `load_documents()` as a private backward-compat shim only for tests, or
  - replace it with row-first helpers if no callers remain.
- Remove any remaining comments/help strings that imply chunks are the generation substrate.

**Step 4: Run test to verify it passes**
Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'source_docs_not_chunk_docs'`
Expected: PASS

**Step 5: Commit**
```bash
git add eval/generate_tier3_rag.py tests/eval/test_tier3_generator_contract.py docs/architecture/current-state.md
git commit -m "refactor: isolate chunk loading to tier3 alignment path"
```

### Task 3: Delete obsolete chunk-first Tier 3 experiment helpers if unused

**Files:**
- Delete: `scripts/build_tier3_chunk_subset.py`
- Delete: `tests/eval/test_tier3_chunk_subset_builder.py`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Verify usage first**
Run: `rg -n "build_tier3_chunk_subset|tier3_chunk_subset" .`
Expected: only docs/tests or no active callers.

**Step 2: Delete files if unused**
- Remove the script and its contract test.
- Remove stale doc references.

**Step 3: Run tests to verify no breakage**
Run: `pytest -q tests/eval`
Expected: PASS

**Step 4: Commit**
```bash
git add -A
git commit -m "chore: remove obsolete tier3 chunk subset experiment path"
```

### Task 4: Decide whether the manual-seed wrapper should remain

**Files:**
- Modify or delete: `eval/tier3_manual_seed_pdf_authoring.py`
- Modify or delete: `scripts/build_tier3_manual_seed_authoring_index.py`
- Modify: `tests/eval/test_tier3_manual_seed_pdf_authoring.py`
- Modify: `tests/eval/test_tier3_manual_seed_authoring_contract.py`

**Step 1: Verify whether manual-seed authoring is still a supported workflow**
- If yes, keep the wrapper thin.
- If no, replace references with the neutral `eval/tier3_article_documents.py` entrypoints and delete the wrapper.

**Step 2: Update tests/docs accordingly**
Run: `pytest -q tests/eval/test_tier3_manual_seed_pdf_authoring.py tests/eval/test_tier3_manual_seed_authoring_contract.py`
Expected: PASS

**Step 3: Commit**
```bash
git add eval/tier3_manual_seed_pdf_authoring.py scripts/build_tier3_manual_seed_authoring_index.py tests/eval/test_tier3_manual_seed_pdf_authoring.py tests/eval/test_tier3_manual_seed_authoring_contract.py
git commit -m "refactor: clarify tier3 manual authoring support surface"
```

### Task 5: Re-run the corrected article-doc generator before adding custom RAGAS internals

**Files:**
- Use: `scripts/run_tier3_article_generation_smoke.py`
- Use: `eval/generate_tier3_rag.py`
- Inspect: `eval/golden/v1/tier3_rag.raw.jsonl`

**Step 1: Run smoke from the real project environment**
Run:
```bash
python3 scripts/run_tier3_article_generation_smoke.py
```
Expected: PASS in the project env, or fail fast on missing runtime deps.

**Step 2: Run a small true article-doc generation**
Run:
```bash
python3 -m eval.generate_tier3_rag \
  --size 8 \
  --article-limit 24 \
  --article-corpus /tmp/tier3_article_corpus.small.jsonl \
  --raw-output /tmp/tier3_rag.small.raw.jsonl \
  --output /tmp/tier3_rag.small.jsonl
```
Expected: produces a small draft set for inspection.

**Step 3: Review output quality before adding custom transforms/synthesizers**
- If quality improves materially, keep the generator minimal.
- Only if article-doc defaults are still weak should we add custom `transforms`, `query_distribution`, or `base_query` synthesizers.

**Step 4: Commit any minimal follow-up changes**
```bash
git add eval/generate_tier3_rag.py scripts/run_tier3_article_generation_smoke.py docs/changes/implementation-log.md
git commit -m "test: validate tier3 article-doc ragas smoke and small run"
```
