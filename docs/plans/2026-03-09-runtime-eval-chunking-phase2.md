# Runtime Validation, Eval Expansion, and Chunking A/B Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate that the stabilized system works in live Docker/runtime, expand the reviewed frozen eval contract without adding noise, and prepare a chunking A/B comparison where chunking is the only intended variable.

**Architecture:** This phase keeps the current stabilized runtime and frozen eval structure. The work is split into three tracks executed in order: first prove the live Docker/runtime contract, then grow the frozen reviewed eval set carefully, then add an eval-only chunking A/B harness that isolates chunking from parser, embedding, retrieval, and workflow drift. Blocking CI stays deterministic and code-based; live/runtime and A/B jobs run as shadow checks until they are stable enough to trust.

**Tech Stack:** FastAPI, PostgreSQL/pgvector, Alembic, LangGraph, Docling, PyPDFLoader, RecursiveCharacterTextSplitter, pytest, Docker, docker-compose, GitHub Actions

---

## Scope Guardrails

- Do not redesign the workflow or agent architecture in this phase.
- Do not make Tier 3 generation a required gate again.
- Do not use LLM judges for checks that can be done with code.
- Keep the blocking CI path deterministic, offline-friendly, and small.
- Keep live Docker/runtime validation and chunking A/B as separate explicit tracks.

## Eval Audit Summary Applied To This Plan

What is now good:
- The repo has a reviewed failure taxonomy in `docs/evals/error-taxonomy.md`.
- The repo has a frozen code-based regression suite in `tests/eval`, `tests/rag`, and `tests/smoke`.
- Tier 3 generation has been demoted to an offline utility.

What is still weak:
- The frozen eval set is still tiny and mostly fixture-driven.
- The live Docker/runtime path is still not exercised by the required CI gate.
- The current chunking comparison harness only evaluates the current pipeline and does not yet support a controlled old-vs-current comparison.

This plan therefore starts with live Docker/runtime validation, then expands the reviewed frozen set, then adds an eval-only chunking A/B harness.

## Required Order Of Work

1. Define the live Docker/runtime validation contract.
2. Add a blocking local/CI-friendly Docker smoke path for the minimal live runtime contract.
3. Add a non-blocking extended live runtime shadow path for reindex/retrieval/chunking reports.
4. Define the frozen eval expansion policy before adding more cases.
5. Expand frozen retrieval cases in reviewed, taxonomy-tagged increments.
6. Expand frozen workflow cases in reviewed, code-checkable increments.
7. Refresh reviewed trace samples and dataset policy checks so new cases do not become noisy.
8. Capture the historical chunking baseline contract from git and define the A/B rules.
9. Implement an eval-only chunking strategy harness where the primary A/B comparison changes only chunking.
10. Add non-blocking live A/B reporting over the real PDF/runtime.

## What Must Be Validated In Live Docker/Runtime

Blocking live runtime validation target:
- Docker image build succeeds for `app` and `eval`.
- `db` reaches healthy state.
- Alembic migration runs to head against the live DB.
- `app` serves `/health/live`.
- `app` serves `/health/ready`.
- `eval` container can run `python3 scripts/preflight_rag.py` successfully.
- Writable cache/artifact directories exist inside the containerized runtime.

Non-blocking extended live runtime validation target:
- `python3 scripts/reindex_pdf.py` succeeds in the containerized runtime.
- Reindex stage counts are non-zero and internally consistent.
- `python3 -m eval.retrieval_eval` can run against the live indexed corpus.
- `python3 -m eval.chunking_eval` can run and save a snapshot.
- Logs/artifacts from the run are saved for review.

## How To Expand The Frozen Eval Set Without Making It Noisy

- Add reviewed cases in small batches of 3-5 cases per PR.
- Every new case must map to exactly one primary failure mode from `docs/evals/error-taxonomy.md`.
- Prefer exact chunk IDs and code-checkable expectations over holistic judge text.
- Keep case fields stable and minimal: query, expected route, expected source family, required keywords, forbidden keywords, table-hit expectation, expected chunk IDs.
- Separate retrieval-only cases from workflow cases; do not overload one case with both routing and generation ambiguity unless that ambiguity is the point of the case.
- Require reviewed trace evidence for every new failure mode before adding many synthetic cases for it.
- Maintain coverage targets by category rather than chasing large case counts.

## How To Design The Chunking A/B Test So Only Chunking Changes

Primary A/B design:
- Parse the PDF once with the current Docling parser into a `ParsedDocument`.
- Keep the same embedding provider/model, same vector dimension, same repository implementation, same retrieval modes, same frozen retrieval questions, same cache-off policy, and same evaluation metrics.
- Compare two chunking strategies over the same parsed input:
  - `legacy_flat_recursive`: an eval-only baseline using the historical recursive splitter settings from commit `6e84ba6` (`chunk_size=512`, `chunk_overlap=128`, separators `["\\n\\n", ". ", "\\n", " "]`) applied to a deterministic linearization of the same parsed document.
  - `docling_structured`: the current `chunk_parsed_document(...)` strategy.

Secondary shadow comparison:
- Optionally run the exact historical `PyPDFLoader + RecursiveCharacterTextSplitter` path as an informational shadow report only.
- Do not use the full historical path as the blocking comparison because it changes both parser and chunking.

Isolation rules:
- Use separate doc IDs or isolated in-memory repositories per strategy.
- Disable semantic cache.
- Use the same retrieval queries/cases for both strategies.
- Compare retrieval metrics first; treat workflow/LLM output comparisons as non-blocking shadow signals because they introduce extra variance.

## Metrics To Compare

Chunk-level metrics:
- total chunk count
- text chunk count
- table chunk count
- token mean / median / p95
- percent with section metadata
- percent with page metadata
- percent with context prefix
- ingestion runtime seconds

Retrieval metrics:
- hit_rate@1 / @3 / @5
- recall@1 / @3 / @5
- MRR@5
- nDCG@5
- table_hit_rate
- wrong_chunk_priority count
- average best-match score

Workflow shadow metrics:
- route match rate
- source family match rate
- required keyword recall
- forbidden keyword violation count
- table_hit_rate

## CI Policy

Keep blocking in CI:
- `python3 -m pytest tests/eval tests/rag tests/smoke -v`
- unit/contract tests for the live runtime validation manifest
- unit/contract tests for the frozen dataset policy
- unit/contract tests for the chunking strategy manifest and eval-only A/B harness

Keep non-blocking in CI:
- Docker live runtime smoke job until it proves stable over multiple runs
- extended live runtime shadow job with reindex + retrieval/chunking reports
- live chunking A/B job over the real PDF
- Tier 3 generation and dataset refresh jobs

Promotion rule:
- A current non-blocking job can only become blocking after it has passed reliably and produced reviewed artifacts across multiple consecutive runs.

---

### Task 1: Define the live runtime validation manifest

**Files:**
- Create: `core/live_runtime_contract.py`
- Create: `tests/smoke/test_live_runtime_contract.py`
- Create: `docs/evals/live-runtime-validation.md`

**Step 1: Write the failing test**

```python
from core.live_runtime_contract import build_live_runtime_checks


def test_live_runtime_checks_cover_minimal_docker_contract():
    checks = build_live_runtime_checks()
    assert [check.name for check in checks] == [
        "build_app_image",
        "build_eval_image",
        "db_healthy",
        "alembic_head",
        "app_live",
        "app_ready",
        "eval_preflight",
        "cache_paths_writable",
    ]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_live_runtime_contract.py::test_live_runtime_checks_cover_minimal_docker_contract -v`

Expected: FAIL because `core.live_runtime_contract` does not exist yet.

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class LiveRuntimeCheck:
    name: str
    command: list[str]
    blocking: bool = True


def build_live_runtime_checks() -> list[LiveRuntimeCheck]:
    return [
        LiveRuntimeCheck("build_app_image", ["docker", "compose", "build", "app"]),
        LiveRuntimeCheck("build_eval_image", ["docker", "compose", "build", "eval"]),
        LiveRuntimeCheck("db_healthy", ["docker", "compose", "up", "-d", "db"]),
        LiveRuntimeCheck("alembic_head", ["docker", "compose", "exec", "-T", "app", "python3", "-m", "alembic", "upgrade", "head"]),
        LiveRuntimeCheck("app_live", ["curl", "-fsS", "http://localhost:8000/health/live"]),
        LiveRuntimeCheck("app_ready", ["curl", "-fsS", "http://localhost:8000/health/ready"]),
        LiveRuntimeCheck("eval_preflight", ["docker", "compose", "run", "--rm", "eval", "python3", "scripts/preflight_rag.py"]),
        LiveRuntimeCheck("cache_paths_writable", ["docker", "compose", "run", "--rm", "eval", "python3", "-c", "from core.runtime_contracts import validate_rag_runtime_contract; raise SystemExit(0 if not validate_rag_runtime_contract() else 1)"]),
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_live_runtime_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/live_runtime_contract.py tests/smoke/test_live_runtime_contract.py docs/evals/live-runtime-validation.md
git commit -m "test: define live docker runtime contract"
```

### Task 2: Add the executable Docker smoke runner and operator runbook

**Files:**
- Create: `scripts/run_live_runtime_smoke.py`
- Create: `tests/smoke/test_live_runtime_smoke_runner.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
from scripts.run_live_runtime_smoke import build_command_plan


def test_command_plan_runs_blocking_checks_in_contract_order():
    plan = build_command_plan(compose_bin="docker compose")
    assert plan[0].label == "build_app_image"
    assert plan[-1].label == "cache_paths_writable"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_live_runtime_smoke_runner.py::test_command_plan_runs_blocking_checks_in_contract_order -v`

Expected: FAIL because the runner does not exist yet.

**Step 3: Write minimal implementation**

```python
from core.live_runtime_contract import build_live_runtime_checks


def build_command_plan(compose_bin: str = "docker compose"):
    checks = build_live_runtime_checks()
    return [
        type("PlannedCommand", (), {"label": check.name, "command": check.command})
        for check in checks
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_live_runtime_smoke_runner.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_live_runtime_smoke.py tests/smoke/test_live_runtime_smoke_runner.py README.md
git commit -m "feat: add docker live runtime smoke runner"
```

### Task 3: Add the extended live runtime shadow run

**Files:**
- Create: `scripts/run_live_rag_shadow.py`
- Create: `tests/smoke/test_live_rag_shadow_contract.py`
- Modify: `.github/workflows/eval.yml`

**Step 1: Write the failing test**

```python
from scripts.run_live_rag_shadow import build_shadow_steps


def test_shadow_steps_include_reindex_and_live_eval_reports():
    steps = build_shadow_steps()
    labels = [step.label for step in steps]
    assert "reindex_pdf" in labels
    assert "retrieval_eval_live" in labels
    assert "chunking_eval_live" in labels
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_live_rag_shadow_contract.py::test_shadow_steps_include_reindex_and_live_eval_reports -v`

Expected: FAIL because the shadow runner does not exist yet.

**Step 3: Write minimal implementation**

```python
def build_shadow_steps():
    return [
        Step("reindex_pdf", ["docker", "compose", "run", "--rm", "eval", "python3", "scripts/reindex_pdf.py"]),
        Step("retrieval_eval_live", ["docker", "compose", "run", "--rm", "eval", "python3", "-m", "eval.retrieval_eval"]),
        Step("chunking_eval_live", ["docker", "compose", "run", "--rm", "eval", "python3", "-m", "eval.chunking_eval", "--save", "shadow"]),
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_live_rag_shadow_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_live_rag_shadow.py tests/smoke/test_live_rag_shadow_contract.py .github/workflows/eval.yml
git commit -m "ci: add non-blocking live rag shadow run"
```

### Task 4: Define the frozen eval expansion policy

**Files:**
- Create: `docs/evals/frozen-regression-policy.md`
- Create: `tests/eval/test_frozen_regression_policy.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_policy_requires_small_reviewed_taxonomy_tagged_batches():
    text = Path("docs/evals/frozen-regression-policy.md").read_text(encoding="utf-8")
    assert "3-5 cases per PR" in text
    assert "one primary failure mode" in text
    assert "code-checkable expectations" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_frozen_regression_policy.py::test_policy_requires_small_reviewed_taxonomy_tagged_batches -v`

Expected: FAIL because the policy doc does not exist yet.

**Step 3: Write minimal implementation**

```markdown
# Frozen Regression Policy

- Add 3-5 reviewed cases per PR.
- Each case must map to one primary failure mode.
- Prefer code-checkable expectations over holistic judgments.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_frozen_regression_policy.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/evals/frozen-regression-policy.md tests/eval/test_frozen_regression_policy.py
git commit -m "docs: define frozen regression expansion policy"
```

### Task 5: Expand frozen retrieval cases in reviewed increments

**Files:**
- Modify: `tests/fixtures/rag/retrieval_cases.jsonl`
- Create: `tests/eval/test_retrieval_dataset_balance.py`
- Modify: `tests/eval/test_retrieval_regression.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path


def test_retrieval_dataset_has_reviewed_coverage_for_key_rag_failures():
    rows = [
        json.loads(line)
        for line in Path("tests/fixtures/rag/retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 12
    categories = {row["category"] for row in rows}
    assert {"dosage", "symptoms", "contraindications", "interactions", "table_lookup"} <= categories
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_retrieval_dataset_balance.py::test_retrieval_dataset_has_reviewed_coverage_for_key_rag_failures -v`

Expected: FAIL because the current dataset only has two cases.

**Step 3: Write minimal implementation**

```json
{"id":"case-metformin-contra","question":"metformin contraindications", ...}
{"id":"case-warfarin-table","question":"warfarin interaction table", ...}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_retrieval_dataset_balance.py tests/eval/test_retrieval_regression.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/fixtures/rag/retrieval_cases.jsonl tests/eval/test_retrieval_dataset_balance.py tests/eval/test_retrieval_regression.py
git commit -m "test: expand frozen retrieval regression coverage"
```

### Task 6: Expand frozen workflow cases without adding noise

**Files:**
- Modify: `tests/fixtures/rag/workflow_cases.jsonl`
- Create: `tests/eval/test_workflow_dataset_balance.py`
- Modify: `tests/eval/test_workflow_regression.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path


def test_workflow_dataset_covers_vector_and_guardrail_behaviors():
    rows = [
        json.loads(line)
        for line in Path("tests/fixtures/rag/workflow_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 8
    routes = {row["expected_route"] for row in rows}
    assert "vector" in routes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_workflow_dataset_balance.py::test_workflow_dataset_covers_vector_and_guardrail_behaviors -v`

Expected: FAIL because the workflow dataset is still minimal.

**Step 3: Write minimal implementation**

```json
{"id":"workflow-metformin","query":"What are the contraindications of metformin?", ...}
{"id":"workflow-warfarin","query":"Show the warfarin interaction table.", ...}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_workflow_dataset_balance.py tests/eval/test_workflow_regression.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/fixtures/rag/workflow_cases.jsonl tests/eval/test_workflow_dataset_balance.py tests/eval/test_workflow_regression.py
git commit -m "test: expand frozen workflow regression coverage"
```

### Task 7: Refresh reviewed trace coverage and policy checks

**Files:**
- Modify: `eval/review_samples/rag_trace_review.jsonl`
- Modify: `docs/evals/error-taxonomy.md`
- Create: `tests/eval/test_review_sample_coverage.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path


def test_review_samples_cover_all_active_frozen_failure_modes():
    rows = [
        json.loads(line)
        for line in Path("eval/review_samples/rag_trace_review.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    modes = {row["failure_mode"] for row in rows}
    assert "wrong_chunk_priority" in modes
    assert "workflow_route_mismatch" in modes
    assert "stale_semantic_cache_hit" in modes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_review_sample_coverage.py::test_review_samples_cover_all_active_frozen_failure_modes -v`

Expected: FAIL once the frozen datasets reference failure modes not represented in reviewed trace samples.

**Step 3: Write minimal implementation**

```json
{"trace_id":"trace-review-013","failure_mode":"wrong_chunk_priority","review_status":"reviewed", ...}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_review_sample_coverage.py tests/eval/test_error_taxonomy_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add eval/review_samples/rag_trace_review.jsonl docs/evals/error-taxonomy.md tests/eval/test_review_sample_coverage.py
git commit -m "test: refresh reviewed trace coverage for frozen evals"
```

### Task 8: Capture the historical chunking baseline contract

**Files:**
- Create: `eval/chunking_strategy_manifest.py`
- Create: `tests/eval/test_chunking_strategy_manifest.py`
- Create: `docs/evals/chunking-ab-design.md`

**Step 1: Write the failing test**

```python
from eval.chunking_strategy_manifest import get_chunking_strategies


def test_chunking_manifest_contains_isolated_and_shadow_baselines():
    strategies = {item.name: item for item in get_chunking_strategies()}
    assert "legacy_flat_recursive" in strategies
    assert "docling_structured" in strategies
    assert strategies["legacy_flat_recursive"].blocking is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_chunking_strategy_manifest.py::test_chunking_manifest_contains_isolated_and_shadow_baselines -v`

Expected: FAIL because the manifest does not exist yet.

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    blocking: bool
    description: str


def get_chunking_strategies():
    return [
        ChunkingStrategy("legacy_flat_recursive", False, "Historical recursive splitter applied to the same parsed input"),
        ChunkingStrategy("docling_structured", True, "Current structure-aware Docling chunker"),
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_chunking_strategy_manifest.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add eval/chunking_strategy_manifest.py tests/eval/test_chunking_strategy_manifest.py docs/evals/chunking-ab-design.md
git commit -m "docs: define chunking ab strategy contract"
```

### Task 9: Add the eval-only chunking A/B harness

**Files:**
- Create: `eval/chunking_strategies.py`
- Modify: `eval/chunking_eval.py`
- Create: `tests/eval/test_chunking_ab_contract.py`

**Step 1: Write the failing test**

```python
from eval.chunking_strategies import build_chunks_for_strategy


def test_legacy_and_docling_strategies_accept_the_same_parsed_document(fixture_parsed_doc):
    legacy = build_chunks_for_strategy("legacy_flat_recursive", fixture_parsed_doc)
    current = build_chunks_for_strategy("docling_structured", fixture_parsed_doc)
    assert legacy
    assert current
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_chunking_ab_contract.py::test_legacy_and_docling_strategies_accept_the_same_parsed_document -v`

Expected: FAIL because the strategy adapter does not exist yet.

**Step 3: Write minimal implementation**

```python
def build_chunks_for_strategy(name, parsed_doc):
    if name == "docling_structured":
        return chunk_parsed_document(parsed_doc)
    if name == "legacy_flat_recursive":
        text = linearize_parsed_document(parsed_doc)
        return split_with_recursive_baseline(text, chunk_size=512, chunk_overlap=128)
    raise ValueError(name)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_chunking_ab_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add eval/chunking_strategies.py eval/chunking_eval.py tests/eval/test_chunking_ab_contract.py
git commit -m "feat: add eval-only chunking ab harness"
```

### Task 10: Add the non-blocking live chunking A/B shadow run

**Files:**
- Create: `scripts/run_chunking_ab_shadow.py`
- Create: `tests/smoke/test_chunking_ab_shadow_contract.py`
- Modify: `.github/workflows/eval.yml`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
from scripts.run_chunking_ab_shadow import build_chunking_ab_steps


def test_chunking_ab_shadow_runs_both_strategies_and_saves_reports():
    steps = build_chunking_ab_steps()
    labels = [step.label for step in steps]
    assert "legacy_flat_recursive_eval" in labels
    assert "docling_structured_eval" in labels
    assert "compare_reports" in labels
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_chunking_ab_shadow_contract.py::test_chunking_ab_shadow_runs_both_strategies_and_saves_reports -v`

Expected: FAIL because the shadow script does not exist yet.

**Step 3: Write minimal implementation**

```python
def build_chunking_ab_steps():
    return [
        Step("legacy_flat_recursive_eval", ["docker", "compose", "run", "--rm", "eval", "python3", "-m", "eval.chunking_eval", "--strategy", "legacy_flat_recursive", "--save", "legacy"]),
        Step("docling_structured_eval", ["docker", "compose", "run", "--rm", "eval", "python3", "-m", "eval.chunking_eval", "--strategy", "docling_structured", "--save", "docling"]),
        Step("compare_reports", ["docker", "compose", "run", "--rm", "eval", "python3", "-m", "eval.chunking_eval", "--strategy", "docling_structured", "--compare", "legacy"]),
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_chunking_ab_shadow_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_chunking_ab_shadow.py tests/smoke/test_chunking_ab_shadow_contract.py .github/workflows/eval.yml README.md
git commit -m "ci: add non-blocking live chunking ab shadow run"
```
