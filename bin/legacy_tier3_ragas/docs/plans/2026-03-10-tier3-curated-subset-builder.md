# Tier 3 Curated Subset Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deterministic utility that derives a better Tier 3 chunk subset from the frozen production chunk corpus without changing production chunking or retrieval behavior.

**Architecture:** Keep the production chunk corpus as the source of truth, then build a narrow eval-only curation layer on top of it. The builder should apply deterministic heuristics to remove obvious junk, reduce repetitive local chunks, preserve later medically relevant content, and write a reusable curated JSONL file that Tier 3 generation can consume with `--chunk-corpus`.

**Tech Stack:** Python 3.11, JSONL files, pytest, existing Tier 3 eval workflow

---

### Task 1: Add a failing regression test for subset selection

**Files:**
- Create: `tests/eval/test_tier3_chunk_subset_builder.py`
- Read: `eval/generate_tier3_rag.py`

**Step 1: Write the failing test**

Add tests that assert the builder:
- drops front-matter/admin chunks
- keeps medically relevant `text_section` chunks
- ignores `table` chunks in the first pass
- spreads results across page ranges instead of taking the first N rows

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_tier3_chunk_subset_builder.py -v`
Expected: FAIL because the builder module does not exist yet.

**Step 3: Write minimal implementation**

Create the builder module/CLI with deterministic heuristics only.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_tier3_chunk_subset_builder.py -v`
Expected: PASS

### Task 2: Implement the eval-only curated subset builder

**Files:**
- Create: `scripts/build_tier3_chunk_subset.py`
- Optionally create: `eval/tier3_chunk_subset.py`

**Step 1: Implement selection helpers**

Implement functions that:
- load chunk-corpus rows
- classify rows as keep/drop based on:
  - `metadata.content_type == "text_section"`
  - minimum page threshold
  - excluded sections such as `STAFF`, `ORGANIZATIONS`, `PERIODICALS`, `BOOKS`, `INTRODUCTION`, `SCOPE`
  - excluded text fragments like editorial/copyright markers
- bucket rows by coarse page band so selection is distributed
- cap repeated chunks from the same page/section neighborhood

**Step 2: Implement CLI**

CLI should accept:
- `--input`
- `--output`
- `--limit`
- optional `--min-page`

It should print:
- input row count
- output row count
- drop counts by reason
- kept page span

**Step 3: Verify CLI**

Run:
`python3 scripts/build_tier3_chunk_subset.py --input eval/golden/v1/tier3_chunk_corpus.jsonl --output /tmp/tier3_chunk_corpus.curated.jsonl --limit 200`

Expected:
- command exits 0
- output file exists
- summary counts print to stdout

### Task 3: Document the new Tier 3 subset workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update README**

Add the recommended workflow:
1. reindex once
2. checkpoint the full chunk corpus once
3. build a curated chunk subset
4. run Tier 3 from the curated subset

**Step 2: Update architecture doc**

Note that Tier 3 now has an eval-only deterministic subset builder layered on top of the frozen production chunk corpus.

**Step 3: Update implementation log**

Add a concise entry describing the curated subset builder and why it exists.

### Task 4: Verify the end-to-end subset workflow

**Files:**
- Use generated output only

**Step 1: Run focused tests**

Run: `pytest tests/eval/test_tier3_chunk_subset_builder.py -v`
Expected: PASS

**Step 2: Run broader eval/smoke suite**

Run: `pytest tests/eval tests/rag tests/smoke -v`
Expected: PASS

**Step 3: Run CLI against the real chunk corpus**

Run:
`python3 scripts/build_tier3_chunk_subset.py --input eval/golden/v1/tier3_chunk_corpus.jsonl --output eval/golden/v1/tier3_chunk_corpus.curated.jsonl --limit 200`

Expected:
- output file written deterministically
- printed summary shows non-zero kept rows
- the file is ready for the next Tier 3 toy run
