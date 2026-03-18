# Tier 3 PDF-Authored Candidate Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a new staged Tier 3 manual-seed candidate batch authored from the book-level PDF and only mapped back to the frozen chunk corpus at the end, so the batch is structurally harder, broader in topic coverage, and more diagnostic of retrieval and faithfulness failures than the current candidate file.

**Architecture:** Keep [eval/golden/v1/tier3_rag.manual.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.jsonl) and [eval/golden/v1/tier3_rag.manual.candidates.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.manual.candidates.jsonl) unchanged. Add a PDF-first authoring helper that reads [data/medical_book.pdf](/home/tough/medical_chatbot/MediGenius/data/medical_book.pdf) through the existing parser, groups parsed elements into article/local-cluster authoring windows, aligns those windows back to [eval/golden/v1/tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl), and uses that alignment to create a new staged candidate file, an audit note, and stronger contract tests. The eval target remains the chunk corpus, but authoring no longer starts from isolated chunk text.

**Tech Stack:** Python 3.12, existing `tools.pdf_loader` / `tools.pdf_parser`, JSONL fixtures, `pytest`, schema validator in `eval.validate`

---

### Task 1: Lock The New PDF-Authored Batch Contract First

**Files:**
- Create: `tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py`
- Test: `tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py`
- Reference: `tests/eval/test_tier3_manual_seed_candidates_contract.py`
- Reference: `eval/validate.py`

**Step 1: Write the failing test for the new staged file path and schema contract**

Create a new test file with constants for:
- approved path: `eval/golden/v1/tier3_rag.manual.jsonl`
- previous candidate path: `eval/golden/v1/tier3_rag.manual.candidates.jsonl`
- new PDF-authored candidate path: `eval/golden/v1/tier3_rag.manual.candidates.pdf_v1.jsonl`
- audit note path: `docs/evals/tier3-manual-seed-pdf-batch-audit.md`

Start with tests shaped like:

```python
def test_pdf_authored_candidate_fixture_is_schema_valid_and_staged_separately():
    assert PDF_CANDIDATE_PATH.exists()

    rows = _load_jsonl(PDF_CANDIDATE_PATH)
    approved = _load_jsonl(APPROVED_PATH)
    previous = _load_jsonl(PREVIOUS_CANDIDATE_PATH)

    assert {row["id"] for row in rows}.isdisjoint({row["id"] for row in approved})
    assert {row["id"] for row in rows}.isdisjoint({row["id"] for row in previous})

    errors = []
    for row in rows:
        errors.extend(validate_module.validate_sample(row, "rag"))

    assert errors == []
```

**Step 2: Add explicit structural-difficulty assertions**

The new test file must compare the new batch against the previous candidate batch by explicit criteria, not just labels:

```python
def test_pdf_authored_candidate_batch_is_structurally_harder_than_previous_batch():
    old_rows = _load_jsonl(PREVIOUS_CANDIDATE_PATH)
    new_rows = _load_jsonl(PDF_CANDIDATE_PATH)

    old_non_simple = [r for r in old_rows if r["difficulty"] != "simple"]
    new_non_simple = [r for r in new_rows if r["difficulty"] != "simple"]

    old_single_chunk_non_simple = sum(len(r["ground_truth_chunk_ids"]) == 1 for r in old_non_simple)
    new_single_chunk_non_simple = sum(len(r["ground_truth_chunk_ids"]) == 1 for r in new_non_simple)

    assert new_single_chunk_non_simple < old_single_chunk_non_simple
    assert all(len(r["ground_truth_chunk_ids"]) >= 2 for r in new_rows if r["difficulty"] == "multi_context")
    assert sum(r["difficulty"] == "simple" for r in new_rows) <= 4
```

For single-chunk `reasoning` rows, require explicit structural tags:

```python
ALLOWED_SINGLE_CHUNK_REASONING = {
    "challenge_type:subject_recovery",
    "challenge_type:heading_dependence",
    "challenge_type:adjacent_chunk_completion",
}
```

and assert any single-chunk `reasoning` row carries at least one of those tags.

**Step 3: Add explicit topic-diversity assertions**

Use explicit criteria that compare the new batch to the previous batch:

```python
def test_pdf_authored_candidate_batch_is_more_topic_diverse_than_previous_batch():
    old_rows = _load_jsonl(PREVIOUS_CANDIDATE_PATH)
    new_rows = _load_jsonl(PDF_CANDIDATE_PATH)

    old_families = _topic_families(old_rows)
    new_families = _topic_families(new_rows)

    assert len(new_families) > len(old_families)
    assert _largest_topic_family_share(new_rows) <= 0.20
```

Do not rely on free-form `topic:*` tags alone. The new batch should carry exactly one `topic_family:*` tag per row so the diversity test has a stable unit.

**Step 4: Add challenge-type coverage assertions**

Require the new batch to include all of the following tags at least once:

```python
REQUIRED_CHALLENGE_TYPES = {
    "challenge_type:subject_recovery",
    "challenge_type:heading_dependence",
    "challenge_type:adjacent_chunk_completion",
    "challenge_type:screening_confirmation",
    "challenge_type:symptom_test_implication",
    "challenge_type:mechanism_implication",
    "challenge_type:escalation",
    "challenge_type:compare_contrast",
    "challenge_type:distractor_control",
}
```

**Step 5: Run the new test file and verify it fails**

Run:

```bash
pytest -q tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py
```

Expected: `FAIL` because `tier3_rag.manual.candidates.pdf_v1.jsonl` and the new audit note do not exist yet.

**Step 6: Commit**

```bash
git add tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py
git commit -m "test: add pdf-authored tier3 candidate contract"
```

### Task 2: Build A PDF-First Authoring Index That Maps Back To Chunks

**Files:**
- Create: `eval/tier3_pdf_authoring.py`
- Create: `scripts/build_tier3_pdf_authoring_index.py`
- Create: `tests/eval/test_tier3_pdf_authoring.py`
- Reference: `tools/pdf_loader.py`
- Reference: `eval/golden/v1/tier3_chunk_corpus.jsonl`

**Step 1: Write a failing unit test for pure helper behavior**

Do not test Docling end-to-end. Test pure, deterministic helpers using fake parsed elements and fake chunk rows.

Start with tests like:

```python
def test_build_article_windows_groups_heading_and_local_body_elements():
    parsed = FakeParsedDocument(...)
    windows = build_article_windows(parsed)
    assert windows[0].article_title == "Congenital heart disease"
    assert "Diagnosis" in windows[0].local_headings


def test_align_article_windows_to_chunks_prefers_page_and_section_overlap():
    windows = [...]
    chunk_rows = [...]
    aligned = align_article_windows_to_chunks(windows, chunk_rows)
    assert aligned[0].chunk_ids == ["medical_book-sec1724-p290-t000", "medical_book-sec1725-p290-t000"]
```

**Step 2: Run the unit test and verify it fails**

Run:

```bash
pytest -q tests/eval/test_tier3_pdf_authoring.py
```

Expected: `FAIL` because the helper module does not exist yet.

**Step 3: Implement the pure helper module**

Create `eval/tier3_pdf_authoring.py` with plain data helpers only:

```python
from dataclasses import dataclass


@dataclass
class PdfArticleWindow:
    article_title: str
    page_start: int
    page_end: int
    local_headings: list[str]
    body_text: str


@dataclass
class ChunkAlignment:
    article_title: str
    page_span: tuple[int, int]
    chunk_ids: list[str]
    chunk_sections: list[str]
```

Implement:
- `load_chunk_corpus_rows(path: Path) -> list[dict]`
- `build_article_windows(parsed_doc) -> list[PdfArticleWindow]`
- `align_article_windows_to_chunks(windows, chunk_rows) -> list[ChunkAlignment]`
- `summarize_alignment(aligned_rows) -> dict[str, int]`

Alignment rules:
- PDF is the source of truth for authoring windows
- chunk rows are used only for mapping evidence back to retrievable units
- alignment should prefer:
  - same page overlap
  - same normalized heading text
  - nearby page band when a body chunk omits the topic name
- do not invent chunks when alignment is weak; drop weak alignments explicitly

**Step 4: Implement the CLI builder**

Create `scripts/build_tier3_pdf_authoring_index.py`:

```python
from pathlib import Path

from eval.tier3_pdf_authoring import (
    align_article_windows_to_chunks,
    build_article_windows,
    load_chunk_corpus_rows,
)
from tools.pdf_loader import load_parsed_pdf
```

CLI inputs:
- `--pdf` default `data/medical_book.pdf`
- `--chunk-corpus` default `eval/golden/v1/tier3_chunk_corpus.jsonl`
- `--output` default `eval/golden/v1/tier3_pdf_authoring_index.jsonl`

Each output row should include:
- `article_title`
- `page_start`
- `page_end`
- `local_headings`
- `body_preview`
- `chunk_ids`
- `chunk_pages`
- `chunk_sections`

This file is an authoring aid only. It is not a benchmark artifact.

**Step 5: Run tests and build the authoring index**

Run:

```bash
pytest -q tests/eval/test_tier3_pdf_authoring.py
python3 scripts/build_tier3_pdf_authoring_index.py
```

Expected:
- unit tests pass
- CLI writes `eval/golden/v1/tier3_pdf_authoring_index.jsonl`
- CLI prints a non-zero count of aligned article windows

**Step 6: Commit**

```bash
git add eval/tier3_pdf_authoring.py scripts/build_tier3_pdf_authoring_index.py tests/eval/test_tier3_pdf_authoring.py eval/golden/v1/tier3_pdf_authoring_index.jsonl
git commit -m "feat: add pdf-first tier3 authoring index"
```

### Task 3: Audit The Approved And Previous Candidate Sets Before Writing New Rows

**Files:**
- Create: `docs/evals/tier3-manual-seed-pdf-batch-audit.md`
- Reference: `eval/golden/v1/tier3_rag.manual.jsonl`
- Reference: `eval/golden/v1/tier3_rag.manual.candidates.jsonl`
- Reference: `eval/golden/v1/tier3_pdf_authoring_index.jsonl`

**Step 1: Compute the baseline audit stats**

Use a one-off local command or a short checked-in helper inside the test file to compute:
- approved row count and difficulty mix
- previous candidate row count and difficulty mix
- previous candidate single-chunk counts split by difficulty
- previous candidate dominant topic families
- challenge types missing from the previous candidate batch

Run:

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

def load(name):
    return [json.loads(line) for line in Path(name).read_text(encoding='utf-8').splitlines() if line.strip()]

approved = load('eval/golden/v1/tier3_rag.manual.jsonl')
previous = load('eval/golden/v1/tier3_rag.manual.candidates.jsonl')

print('approved difficulty', Counter(r['difficulty'] for r in approved))
print('previous difficulty', Counter(r['difficulty'] for r in previous))
print('previous chunk counts', Counter(len(r['ground_truth_chunk_ids']) for r in previous))
PY
```

Expected: baseline stats print successfully and match the known weakness of the previous candidate file.

**Step 2: Write the concise audit note**

Create `docs/evals/tier3-manual-seed-pdf-batch-audit.md` with four short sections:
- current overconcentration
- current single-chunk weakness
- challenge types missing or underrepresented
- what the new PDF-authored batch is meant to add

Keep it concise and concrete. Use the measured counts from Step 1.

**Step 3: Commit**

```bash
git add docs/evals/tier3-manual-seed-pdf-batch-audit.md
git commit -m "docs: add tier3 pdf-authored batch audit note"
```

### Task 4: Author The New Candidate Batch From PDF Article Clusters, Then Map Back To Chunks

**Files:**
- Create: `eval/golden/v1/tier3_rag.manual.candidates.pdf_v1.jsonl`
- Reference: `data/medical_book.pdf`
- Reference: `eval/golden/v1/tier3_pdf_authoring_index.jsonl`
- Reference: `eval/golden/v1/tier3_chunk_corpus.jsonl`
- Reference: `eval/golden/v1/tier3_rag.manual.candidates.jsonl`

**Step 1: Define the target composition before writing rows**

Use this target mix:
- total rows: `30-40`
- simple rows: `2-4`
- reasoning rows: `14-18`
- multi_context rows: `14-18`
- zero `multi_context` rows that are fully supported by one chunk
- at least `10` unique `topic_family:*` tags
- no single `topic_family:*` should cover more than `20%` of rows

Each row must include:
- `rag`
- `manual_seed_candidate`
- `candidate_batch:2026-03-10-pdf-v1`
- exactly one `topic_family:*`
- one or more `topic:*`
- one or more `challenge_type:*`

**Step 2: Author rows from the PDF/article view, not the chunk text**

Use the authoring index and the PDF as the starting point:
- choose an article or tightly bounded local cluster from the PDF
- inspect neighboring headings and local content
- draft the question in paraphrased natural language
- define the evidence path first
- only then map the answer back to chunk IDs

Reject any non-simple draft if:
- one chunk answers it fully without title/neighbor recovery
- it depends on outside knowledge
- it is broad or underspecified
- it is “hard-sounding” only because of wording

**Step 3: Broaden topic families across the book**

Do not let this batch collapse back into cervical/Pap/gynecology.

Intentionally broaden into clearly different families available in the PDF, for example:
- prenatal testing
- congenital heart disease
- fetal monitoring
- gastrointestinal or infectious disease clusters
- hereditary disorder diagnosis clusters
- other oncology families beyond gynecology

The exact families can be adjusted during authoring, but the batch must be visibly broader than the current candidate file.

**Step 4: Keep the row contract strict**

For `reasoning` and `multi_context` rows:
- prefer `2+` evidence chunks
- if a `reasoning` row uses one chunk, it must be justified by:
  - `challenge_type:subject_recovery`
  - or `challenge_type:heading_dependence`
  - or `challenge_type:adjacent_chunk_completion`

For `multi_context` rows:
- require `2+` chunk IDs
- require a real evidence path such as:
  - screening -> confirmation
  - symptom -> test -> implication
  - mechanism -> implication
  - escalation / follow-up
  - same-topic compare/contrast

**Step 5: Run the new contract tests and fix the batch until green**

Run:

```bash
pytest -q tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py
```

Expected:
- fail initially while the batch is incomplete
- pass only when the new file satisfies schema, difficulty, challenge coverage, and diversity constraints

**Step 6: Commit**

```bash
git add eval/golden/v1/tier3_rag.manual.candidates.pdf_v1.jsonl
git commit -m "feat: add pdf-authored tier3 candidate batch"
```

### Task 5: Update Minimal Tier 3 Docs And Artifact Inventory

**Files:**
- Modify: `eval/golden/v1/README.md`
- Modify: `docs/evals/tier3-seed-control-audit.md`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`
- Reference: `docs/evals/tier3-manual-seed-pdf-batch-audit.md`

**Step 1: Update the golden artifact inventory**

Add the new staged file and the PDF authoring index to `eval/golden/v1/README.md`.

**Step 2: Update the Tier 3 seed note**

Document that:
- the new PDF-authored candidate file is staged separately
- authoring is PDF-first but evidence remains chunk-mapped
- the current generator still does not read any manual candidate file

**Step 3: Update current-state architecture**

Add one concise bullet describing the PDF-first authoring index as an eval-only manual-seed aid.

**Step 4: Add an implementation log entry**

Record:
- why the old candidate batch was structurally too weak
- why PDF-first authoring was introduced
- what contract now distinguishes the new batch from the old candidate file

**Step 5: Commit**

```bash
git add eval/golden/v1/README.md docs/evals/tier3-seed-control-audit.md docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: record pdf-authored tier3 candidate workflow"
```

### Task 6: Run Final Verification And Prepare Handoff

**Files:**
- Test: `tests/eval/test_tier3_pdf_authoring.py`
- Test: `tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py`
- Test: `tests/eval/test_tier3_manual_seed_candidates_contract.py`

**Step 1: Run focused Tier 3 tests**

Run:

```bash
pytest -q \
  tests/eval/test_tier3_pdf_authoring.py \
  tests/eval/test_tier3_manual_seed_candidates_contract.py \
  tests/eval/test_tier3_manual_seed_pdf_candidates_contract.py
```

Expected: all three files pass.

**Step 2: Run full repo tests**

Run:

```bash
pytest -q
```

Expected: full suite passes without changing the approved manual seed file or the existing candidate file.

**Step 3: Summarize the delivered artifacts**

The implementation handoff should list:
- the new staged candidate file path
- the audit note path
- the PDF authoring index path
- the structural-difficulty deltas versus the previous candidate batch
- the topic-family diversity deltas versus the previous candidate batch

**Step 4: Commit**

```bash
git add -A
git commit -m "test: verify pdf-authored tier3 candidate path"
```
