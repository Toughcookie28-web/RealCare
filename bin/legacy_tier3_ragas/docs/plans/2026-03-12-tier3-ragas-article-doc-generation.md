# Tier 3 RAGAS Article-Document Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace chunk-fed Tier 3 RAGAS generation with article-entry source documents, preserve chunk alignment for schema output, and add a smoke-first path before any scale-up run.

**Architecture:** Extract shared article-boundary recovery into a neutral eval module, cache article-level source documents under `eval/golden/v1`, feed those documents into the official RAGAS document-generation path, and post-align generated contexts back to the frozen retrieval chunk corpus when writing schema_v1 draft rows. Keep `tier3_rag.raw.jsonl` as the raw checkpoint and keep the manual-seed authoring path compatible by reusing the same article extraction layer.

**Tech Stack:** Python 3.12, LangChain `Document`, existing PDF/`mutool` tooling, RAGAS testset generator, JSONL artifacts, pytest.

---

### Task 1: Add failing tests for a neutral article-document extraction module

**Files:**
- Create: `tests/eval/test_tier3_article_documents.py`
- Modify: `tests/eval/test_tier3_manual_seed_pdf_authoring.py`
- Reference: `eval/tier3_manual_seed_pdf_authoring.py`

**Step 1: Write the failing tests**

```python
def test_build_article_source_documents_groups_pages_into_article_entries():
    docs = module.build_article_source_documents(parsed_pages)
    assert docs[0].metadata["article_title"] == "Congenital heart disease"
    assert docs[0].metadata["source_doc_type"] == "tier3_article"
    assert docs[0].metadata["page_start"] == 290
    assert docs[0].metadata["page_end"] == 291


def test_build_article_source_documents_preserves_headings_and_body_text():
    docs = module.build_article_source_documents(parsed_pages)
    assert docs[0].metadata["local_headings"] == ["Definition", "Diagnosis", "Treatment"]
    assert "Echocardiography is used to confirm it." in docs[0].page_content
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/eval/test_tier3_article_documents.py tests/eval/test_tier3_manual_seed_pdf_authoring.py`
Expected: FAIL because `eval/tier3_article_documents.py` does not exist and the manual-seed helper still owns the article logic directly.

**Step 3: Write minimal implementation**

Create `eval/tier3_article_documents.py` with:
- a neutral article-window dataclass
- a `build_article_windows(parsed_pages)` function
- a `build_article_source_documents(parsed_pages)` function that emits LangChain `Document` objects with article metadata
- utility functions for title/headings/body extraction

Keep the first implementation minimal. Do not add chunk alignment in this task.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/eval/test_tier3_article_documents.py tests/eval/test_tier3_manual_seed_pdf_authoring.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/eval/test_tier3_article_documents.py tests/eval/test_tier3_manual_seed_pdf_authoring.py eval/tier3_article_documents.py
git commit -m "refactor: extract tier3 article document builder"
```

### Task 2: Rewire the manual-seed authoring helper to use the shared article module

**Files:**
- Modify: `eval/tier3_manual_seed_pdf_authoring.py`
- Modify: `scripts/build_tier3_manual_seed_authoring_index.py`
- Test: `tests/eval/test_tier3_manual_seed_pdf_authoring.py`
- Test: `tests/eval/test_tier3_manual_seed_authoring_contract.py`

**Step 1: Write the failing test**

```python
def test_manual_seed_authoring_helper_uses_shared_article_windows():
    module = _load_module()
    windows = module.build_article_windows(parsed_pages)
    assert windows[0].article_title == "Congenital heart disease"
```

Make the test assert behavior through the public helper surface, not the internal implementation.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/eval/test_tier3_manual_seed_pdf_authoring.py`
Expected: FAIL after Task 1 if imports or compatibility wrappers are missing.

**Step 3: Write minimal implementation**

Update `eval/tier3_manual_seed_pdf_authoring.py` to:
- import article-window logic from `eval/tier3_article_documents.py`
- keep manual-seed-specific alignment/index helpers only
- preserve the existing public API used by the manual-seed tests and index builder

Update `scripts/build_tier3_manual_seed_authoring_index.py` only as needed to match the shared API.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/eval/test_tier3_manual_seed_pdf_authoring.py tests/eval/test_tier3_manual_seed_authoring_contract.py`
Expected: PASS

**Step 5: Commit**

```bash
git add eval/tier3_manual_seed_pdf_authoring.py scripts/build_tier3_manual_seed_authoring_index.py tests/eval/test_tier3_manual_seed_pdf_authoring.py tests/eval/test_tier3_manual_seed_authoring_contract.py
git commit -m "refactor: share article extraction with manual seed helper"
```

### Task 3: Add failing generator contract tests for article-doc inputs and chunk-aligned outputs

**Files:**
- Modify: `tests/eval/test_tier3_generator_contract.py`
- Reference: `eval/generate_tier3_rag.py`
- Reference: `eval/tier3_article_documents.py`

**Step 1: Write the failing tests**

Add focused tests for the new contract:

```python
def test_load_source_documents_prefers_cached_article_corpus(tmp_path):
    docs = module.load_source_documents("missing.pdf", source="auto", article_corpus_path=corpus_path)
    assert docs[0].metadata["source_doc_type"] == "tier3_article"


def test_load_source_documents_builds_article_docs_from_pdf_when_cache_missing(tmp_path, monkeypatch):
    docs = module.load_source_documents(str(pdf_path), source="auto", article_corpus_path=corpus_path)
    assert docs[0].metadata["article_title"] == "Campylobacteriosis"


def test_context_alignment_maps_article_context_back_to_underlying_chunk_ids():
    alignment = module._infer_ground_truth_alignment(contexts, article_docs, chunk_rows)
    assert alignment["ground_truth_chunk_ids"] == ["medical_book-sec1724-p290-t000"]
```

Also add a small contract for smoke-safe CLI overrides:

```python
def test_generator_supports_output_overrides_for_smoke_runs():
    assert module.TIER3_PATH.name == "tier3_rag.generated_draft.jsonl"
    assert module.RAW_RAGAS_ROWS_PATH.name == "tier3_rag.raw.jsonl"
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py`
Expected: FAIL because the generator still loads chunk docs and alignment assumes `chunk_id` on the source docs.

**Step 3: Write minimal implementation**

Do not implement all behavior yet. Only rename/add the new function surfaces in the tests if needed so the failures point at the missing article-doc behavior.

**Step 4: Run test to verify the failure is about behavior, not import errors**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'article or alignment or smoke'`
Expected: FAIL with behavior assertions, not module-load errors.

**Step 5: Commit**

```bash
git add tests/eval/test_tier3_generator_contract.py
git commit -m "test: define article document tier3 generator contract"
```

### Task 4: Replace the generator’s Step 1 source substrate with article documents

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Create: `eval/golden/v1/tier3_article_corpus.jsonl` (artifact generated by the code path, not hand-authored)
- Reference: `eval/tier3_article_documents.py`
- Reference: `eval/golden/v1/tier3_chunk_corpus.jsonl`

**Step 1: Write the failing test**

Use the failing tests from Task 3. Do not add more unless needed.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'load_source_documents or article'`
Expected: FAIL

**Step 3: Write minimal implementation**

Change `eval/generate_tier3_rag.py` so that:
- it introduces `ARTICLE_CORPUS_PATH = GOLDEN_DIR / "tier3_article_corpus.jsonl"`
- it replaces `load_documents(...)` with `load_source_documents(...)`
- `load_source_documents` reads cached article docs first
- if missing, it parses the PDF and builds article-entry docs via `eval/tier3_article_documents.py`
- it checkpoints those article docs to `tier3_article_corpus.jsonl`
- it does **not** use retrieval chunks as the direct RAGAS source docs anymore

Keep the chunk corpus available separately because later alignment still needs it.

Suggested metadata on each article source document:

```python
{
    "source_doc_type": "tier3_article",
    "article_doc_id": "medical_book-article-018-campylobacteriosis",
    "article_title": "Campylobacteriosis",
    "page_start": 18,
    "page_end": 19,
    "local_headings": ["Definition", "Description", "Causes and symptoms"],
}
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'load_source_documents or article'`
Expected: PASS

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py eval/tier3_article_documents.py tests/eval/test_tier3_generator_contract.py
git commit -m "feat: feed tier3 ragas with article documents"
```

### Task 5: Restore precise schema alignment by mapping article contexts back to retrieval chunks

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_generator_contract.py`
- Reference: `eval/golden/v1/tier3_chunk_corpus.jsonl`

**Step 1: Write the failing test**

Use the alignment failure from Task 3 and sharpen it if needed:

```python
def test_context_alignment_maps_article_context_back_to_best_matching_chunk():
    alignment = module._infer_ground_truth_alignment(contexts, article_docs, chunk_rows)
    assert alignment["ground_truth_chunk_ids"] == ["chunk-colposcopy"]
    assert alignment["ground_truth_pages"] == [101]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'alignment'`
Expected: FAIL because the current alignment logic expects the source docs themselves to carry `chunk_id`.

**Step 3: Write minimal implementation**

Update `_infer_ground_truth_alignment(...)` so it works in two stages:
1. match each generated context to the best article source document
2. within that matched article document, match the context to the best aligned retrieval chunk row

Inputs should include both:
- article source docs
- retrieval chunk rows loaded from `tier3_chunk_corpus.jsonl`

Do not fan out every article context to all 8 aligned chunks. Pick the best matching chunk rows.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'alignment or dataset_to_schema_v1'`
Expected: PASS

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py tests/eval/test_tier3_generator_contract.py
git commit -m "fix: align tier3 article contexts back to retrieval chunks"
```

### Task 6: Add a smoke-first path for small article-doc runs before scale-up

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Modify: `tests/eval/test_tier3_generator_contract.py`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/evals/tier3-seed-control-audit.md`
- Modify: `eval/golden/v1/README.md`

**Step 1: Write the failing test**

Add a focused contract test for smoke-oriented overrides:

```python
def test_generator_supports_smoke_safe_overrides(tmp_path):
    parser = module._build_arg_parser()
    args = parser.parse_args([
        "--size", "4",
        "--article-limit", "12",
        "--output", str(tmp_path / "tier3_smoke.jsonl"),
        "--raw-output", str(tmp_path / "tier3_smoke.raw.jsonl"),
        "--article-corpus", str(tmp_path / "tier3_article_corpus.jsonl"),
    ])
    assert args.article_limit == 12
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'smoke_safe_overrides'`
Expected: FAIL because the CLI does not yet expose smoke-safe override paths.

**Step 3: Write minimal implementation**

Add CLI support for:
- `--output`
- `--raw-output`
- `--article-corpus`
- `--article-limit`

Use these to support a true smoke run without touching committed golden artifacts.

Document a canonical smoke command such as:

```bash
python3 -m eval.generate_tier3_rag \
  --source auto \
  --article-limit 12 \
  --size 4 \
  --article-corpus /tmp/tier3_article_corpus.smoke.jsonl \
  --raw-output /tmp/tier3_rag.smoke.raw.jsonl \
  --output /tmp/tier3_rag.smoke.jsonl
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/eval/test_tier3_generator_contract.py -k 'smoke_safe_overrides'`
Expected: PASS

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py tests/eval/test_tier3_generator_contract.py docs/architecture/current-state.md docs/evals/tier3-seed-control-audit.md eval/golden/v1/README.md
git commit -m "feat: add smoke-safe tier3 article generation path"
```

### Task 7: Update docs and run full verification

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/evals/tier3-seed-control-audit.md`
- Modify: `docs/changes/implementation-log.md`
- Modify: `eval/golden/v1/README.md`
- Optional if clean: `README.md`

**Step 1: Write the failing test**

No new behavioral tests. Use the existing suite as the regression gate.

**Step 2: Run focused tests before docs edits are finalized**

Run:
```bash
pytest -q tests/eval/test_tier3_article_documents.py \
  tests/eval/test_tier3_manual_seed_pdf_authoring.py \
  tests/eval/test_tier3_manual_seed_authoring_contract.py \
  tests/eval/test_tier3_generator_contract.py \
  tests/eval/test_tier3_final_curated_contract.py
```
Expected: PASS

**Step 3: Update docs**

Record the new Tier 3 generation boundary clearly:
- RAGAS now consumes article-entry documents
- retrieval chunks remain alignment/eval artifacts, not generation substrate
- smoke-first run is the supported first step before scale-up
- raw checkpoint is still preserved
- old downstream filtering logic remains intentionally absent until the new substrate is evaluated

**Step 4: Run full verification**

Run:
```bash
pytest -q
python3 -m eval.validate
```
Expected:
- `pytest -q` passes
- `eval.validate` keeps the same known `review_completion` failure unless unrelated golden review state changes

**Step 5: Commit**

```bash
git add docs/architecture/current-state.md docs/evals/tier3-seed-control-audit.md docs/changes/implementation-log.md eval/golden/v1/README.md
git commit -m "docs: record article-level tier3 ragas pipeline"
```
