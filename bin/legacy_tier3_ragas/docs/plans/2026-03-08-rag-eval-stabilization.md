# RAG and Eval Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the current RAG pipeline, make evals trustworthy enough to guide changes, and fix only the container/runtime issues that directly block RAG and eval work.

**Architecture:** This plan does not redesign the agent system. It narrows the work to four tracks: fail fast on broken runtime contracts, make ingestion/indexing deterministic, isolate evals from live runtime volatility, and make the Docker/dev path behave like the real image. The implementation should prefer small code-based regression checks over new LLM-judge infrastructure.

**Tech Stack:** FastAPI, SQLAlchemy, PostgreSQL/pgvector, Alembic, LangGraph, Docling, sentence-transformers, pytest, Docker, docker-compose

---

## Scope Guardrails

- Exclude frontend and UI work.
- Exclude sub-agent redesign unless a current behavior is a direct blocker for RAG or eval stability.
- Do not add new judge models or broad eval dashboards in this phase.
- Prefer code-based smoke/regression checks over new subjective evaluators.

## Eval Audit Summary Applied To This Plan

The current repo has local eval scripts but not a trustworthy eval system:

- No active CI test/eval execution in [.github/workflows/eval.yml](/home/tough/medical_chatbot/MediGenius/.github/workflows/eval.yml)
- No documented error taxonomy or reviewed trace set in [docs/](/home/tough/medical_chatbot/MediGenius/docs)
- No judge validation artifacts, no train/dev/test split, and no stable labeled dataset
- `eval/generate_tier3_rag.py` currently behaves like a live integration workload, not a reliable regression gate

This plan therefore starts with runtime contracts and trace/error analysis before trying to trust any end-to-end metric.

## Recommended Phase Order

1. Immediate blockers
2. Container/dev environment stabilization
3. RAG architecture cleanup
4. Eval isolation and trustworthiness

---

## Immediate Blockers

### Task 1: Add an explicit RAG runtime preflight contract

**Why it matters now:** RAG and eval failures are currently discovered only after the app boots or the eval job starts. A fast preflight check should fail before PDF parsing, embedding, DB access, or workflow execution begin.

**Required now or can wait:** Required now

**Recommended order:** 1

**Files:**
- Create: `core/runtime_contracts.py`
- Create: `scripts/preflight_rag.py`
- Create: `tests/smoke/test_runtime_contracts.py`
- Modify: `README.md:124-152`
- Modify: `.github/workflows/eval.yml:33-47`

**Step 1: Write the failing test**

```python
from core.runtime_contracts import validate_rag_runtime_contract


def test_validate_rag_runtime_contract_reports_missing_dependencies():
    errors = validate_rag_runtime_contract(
        import_checker=lambda name: name not in {"fastapi", "langchain_core", "docling"},
        db_checker=lambda: True,
        cache_dir="/tmp/medigenius-contract-test",
    )
    assert "missing dependency: langchain_core" in errors
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_runtime_contracts.py::test_validate_rag_runtime_contract_reports_missing_dependencies -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `core.runtime_contracts` does not exist yet.

**Step 3: Write minimal implementation**

```python
def validate_rag_runtime_contract(import_checker=None, db_checker=None, cache_dir=None):
    import_checker = import_checker or _default_import_checker
    errors = []
    for name in ("fastapi", "langchain_core", "langgraph"):
        if not import_checker(name):
            errors.append(f"missing dependency: {name}")
    return errors
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_runtime_contracts.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/runtime_contracts.py scripts/preflight_rag.py tests/smoke/test_runtime_contracts.py README.md .github/workflows/eval.yml
git commit -m "feat: add rag runtime preflight contract"
```

### Task 2: Remove silent persistence fallback from RAG and eval paths

**Why it matters now:** The system silently shifts from PostgreSQL to in-memory behavior in the exact paths used for retrieval, session history, and HITL. That makes debugging misleading and invalidates eval results.

**Required now or can wait:** Required now

**Recommended order:** 2

**Files:**
- Modify: `db/session.py:12-39`
- Modify: `api/deps.py:19-48`
- Modify: `core/hitl.py:9-101`
- Modify: `api/routes/health.py:20-31`
- Create: `tests/smoke/test_persistence_fail_fast.py`

**Step 1: Write the failing test**

```python
import pytest

from api.deps import get_chat_repository


def test_get_chat_repository_raises_when_db_is_unavailable(fake_broken_session):
    with pytest.raises(RuntimeError, match="persistent database unavailable"):
        get_chat_repository(fake_broken_session)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_persistence_fail_fast.py::test_get_chat_repository_raises_when_db_is_unavailable -v`

Expected: FAIL because `get_chat_repository()` currently falls back to an in-memory repository.

**Step 3: Write minimal implementation**

```python
def get_chat_repository(db):
    try:
        db.execute(text("SELECT 1"))
        db.query(SessionModel).limit(1).all()
        return ChatRepository(db)
    except Exception as exc:
        raise RuntimeError("persistent database unavailable") from exc
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_persistence_fail_fast.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add db/session.py api/deps.py core/hitl.py api/routes/health.py tests/smoke/test_persistence_fail_fast.py
git commit -m "fix: fail fast on broken persistence in rag paths"
```

### Task 3: Remove startup side effects from API boot

**Why it matters now:** App startup currently performs database setup and optional indexing behavior that belongs in explicit jobs. That turns ordinary API boot and eval setup into a mixed serving-plus-migration-plus-ingest flow.

**Required now or can wait:** Required now

**Recommended order:** 3

**Files:**
- Modify: `app.py:26-47`
- Modify: `scripts/reindex_pdf.py:13-23`
- Modify: `README.md:124-152`
- Create: `tests/smoke/test_app_startup_contract.py`

**Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient

from app import create_app


def test_app_startup_does_not_trigger_pdf_ingest(monkeypatch):
    called = {"ingest": 0}
    monkeypatch.setattr("app.ingest_pdf_to_vector_store", lambda *args, **kwargs: called.__setitem__("ingest", 1))
    with TestClient(create_app()):
        pass
    assert called["ingest"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_app_startup_contract.py::test_app_startup_does_not_trigger_pdf_ingest -v`

Expected: FAIL because startup currently retains ingest side effects.

**Step 3: Write minimal implementation**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    yield
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_app_startup_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app.py scripts/reindex_pdf.py README.md tests/smoke/test_app_startup_contract.py
git commit -m "refactor: remove ingest side effects from api startup"
```

---

## Container/Dev Environment Stabilization

### Task 4: Unify install behavior across Docker, CI, and Render

**Why it matters now:** The same repository is installed three different ways today, which is why the image can build, CI can pass, and runtime can still miss core packages. RAG and eval work need one dependency contract.

**Required now or can wait:** Required now

**Recommended order:** 4

**Files:**
- Modify: `Dockerfile:1-42`
- Modify: `.github/workflows/eval.yml:8-47`
- Modify: `render.yaml:1-25`
- Modify: `README.md:96-152`
- Create: `tests/smoke/test_install_contract.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_all_runtime_install_paths_reference_the_same_profile():
    dockerfile = Path("Dockerfile").read_text()
    workflow = Path(".github/workflows/eval.yml").read_text()
    render = Path("render.yaml").read_text()
    assert "python -m pip install ." in dockerfile
    assert "python -m pip install ." in workflow
    assert "pip install ." in render
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_install_contract.py::test_all_runtime_install_paths_reference_the_same_profile -v`

Expected: FAIL because the current install commands and dependency resolution logic are inconsistent.

**Step 3: Write minimal implementation**

```python
INSTALL_COMMAND = "python -m pip install ."
EVAL_INSTALL_COMMAND = 'python -m pip install ".[eval,ingest]"'
```

Use those exact commands everywhere instead of hand-rolled requirement extraction.

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_install_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add Dockerfile .github/workflows/eval.yml render.yaml README.md tests/smoke/test_install_contract.py
git commit -m "build: unify install paths for runtime and eval"
```

### Task 5: Normalize compose mounts and writable artifact/cache locations

**Why it matters now:** The built image and the compose runtime do not behave the same way, and Docling/RapidOCR/Hugging Face artifacts still depend on writable runtime paths. That directly blocks PDF ingestion and Tier 3 generation.

**Required now or can wait:** Required now

**Recommended order:** 5

**Files:**
- Modify: `docker-compose.yml:1-88`
- Modify: `Dockerfile:34-42`
- Modify: `tools/embedding_bootstrap.py:8-57`
- Modify: `tools/pdf_parser.py:116-145`
- Create: `tests/smoke/test_container_cache_contract.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_eval_service_uses_explicit_writable_cache_mount():
    compose = Path("docker-compose.yml").read_text()
    assert "./.cache/embeddings:/app/.cache/embeddings" in compose
    assert "./:/app" not in compose
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_container_cache_contract.py::test_eval_service_uses_explicit_writable_cache_mount -v`

Expected: FAIL if the compose file still allows repo-wide mounts or lacks a stable writable cache contract.

**Step 3: Write minimal implementation**

```python
APP_CACHE_ROOT = "/app/.cache/embeddings"
RAPIDOCR_MODELS_DIR = f"{APP_CACHE_ROOT}/rapidocr/models"
DOCLING_ARTIFACTS_DIR = f"{APP_CACHE_ROOT}/docling_artifacts"
```

Mount only those cache/artifact paths and keep them writable in both `Dockerfile` and `docker-compose.yml`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_container_cache_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docker-compose.yml Dockerfile tools/embedding_bootstrap.py tools/pdf_parser.py tests/smoke/test_container_cache_contract.py
git commit -m "fix: normalize cache and artifact mounts for rag containers"
```

---

## RAG Architecture Cleanup

### Task 6: Make schema and vector dimension deterministic

**Why it matters now:** Indexing stability depends on one schema definition, one vector dimension, and one migration path. Right now model shape changes by environment and startup still mutates schema.

**Required now or can wait:** Required now

**Recommended order:** 6

**Files:**
- Create: `db/schema_contract.py`
- Modify: `core/settings.py:17-23`
- Modify: `core/settings.py:48-50`
- Modify: `db/models.py:16-18`
- Modify: `db/models.py:79-110`
- Modify: `db/session.py:31-39`
- Create: `alembic/versions/0003_align_document_chunks_vector_contract.py`
- Create: `tests/smoke/test_schema_contract.py`

**Step 1: Write the failing test**

```python
from db.schema_contract import DOCUMENT_CHUNK_VECTOR_DIM
from core.settings import get_settings


def test_embedding_dim_matches_document_chunk_schema_contract():
    assert get_settings().embedding_dim == DOCUMENT_CHUNK_VECTOR_DIM
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_schema_contract.py::test_embedding_dim_matches_document_chunk_schema_contract -v`

Expected: FAIL because no single schema contract exists yet.

**Step 3: Write minimal implementation**

```python
DOCUMENT_CHUNK_VECTOR_DIM = 768


def default_embedding_dim() -> int:
    return DOCUMENT_CHUNK_VECTOR_DIM
```

Make the runtime settings and ORM model consume the same contract, then write a migration that aligns the database.

**Step 4: Run test to verify it passes**

Run: `pytest tests/smoke/test_schema_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add db/schema_contract.py core/settings.py db/models.py db/session.py alembic/versions/0003_align_document_chunks_vector_contract.py tests/smoke/test_schema_contract.py
git commit -m "fix: make vector schema contract deterministic"
```

### Task 7: Split ingestion into explicit parse, chunk, embed, and index stages

**Why it matters now:** The current ingest path is one monolithic function call. When it fails, you cannot tell whether the break was parser, chunker, embedding, or database/index behavior.

**Required now or can wait:** Required now

**Recommended order:** 7

**Files:**
- Create: `tools/ingest_report.py`
- Modify: `tools/pdf_loader.py:11-37`
- Modify: `tools/vector_store.py:15-61`
- Modify: `scripts/reindex_pdf.py:13-23`
- Create: `tests/rag/test_ingest_report.py`

**Step 1: Write the failing test**

```python
from tools.ingest_report import IngestReport


def test_ingest_report_preserves_stage_counts():
    report = IngestReport(doc_id="medical_book", parsed_elements=10, chunks=5, embeddings=5, inserted=5)
    assert report.chunks == 5
    assert report.inserted == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_ingest_report.py::test_ingest_report_preserves_stage_counts -v`

Expected: FAIL because `tools.ingest_report` does not exist yet.

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass
class IngestReport:
    doc_id: str
    parsed_elements: int
    chunks: int
    embeddings: int
    inserted: int
```

Then return that report from the ingest path instead of only a final row count.

**Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_ingest_report.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/ingest_report.py tools/pdf_loader.py tools/vector_store.py scripts/reindex_pdf.py tests/rag/test_ingest_report.py
git commit -m "refactor: expose ingest stage report for rag pipeline"
```

### Task 8: Make semantic cache safe for debugging and eval

**Why it matters now:** Retrieval and prompt changes cannot be trusted while semantically similar old answers are still returned. That blocks both debugging and eval interpretation.

**Required now or can wait:** Required now

**Recommended order:** 8

**Files:**
- Modify: `core/settings.py:57-60`
- Modify: `tools/cache.py:12-49`
- Modify: `agents/executor_agent.py:75-126`
- Modify: `eval/workflow_eval.py:94-165`
- Create: `tests/rag/test_semantic_cache_contract.py`

**Step 1: Write the failing test**

```python
from tools.cache import SemanticCache


def test_semantic_cache_can_be_disabled():
    cache = SemanticCache(enabled=False)
    cache.set("what is dka", "cached answer")
    assert cache.get("what is dka") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_semantic_cache_contract.py::test_semantic_cache_can_be_disabled -v`

Expected: FAIL because the cache currently has no explicit disable/version control.

**Step 3: Write minimal implementation**

```python
class SemanticCache:
    def __init__(self, enabled=True, version="v1"):
        self.enabled = enabled
        self.version = version

    def get(self, query):
        if not self.enabled:
            return None
```

Disable the cache in eval by default and include a version key tied to prompt/retrieval configuration.

**Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_semantic_cache_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tools/cache.py agents/executor_agent.py eval/workflow_eval.py tests/rag/test_semantic_cache_contract.py
git commit -m "fix: make semantic cache safe for eval and debugging"
```

---

## Eval Isolation and Trustworthiness

### Task 9: Create a reviewed error taxonomy before expanding eval logic

**Why it matters now:** Per the eval audit, there is no documented failure taxonomy. Without one, new eval work will optimize generic or irrelevant metrics instead of the actual RAG failure modes seen in this repo.

**Required now or can wait:** Required now

**Recommended order:** 9

**Files:**
- Create: `docs/evals/error-taxonomy.md`
- Create: `eval/review_samples/rag_trace_review.jsonl`
- Create: `tests/eval/test_error_taxonomy_contract.py`
- Modify: `docs/architecture/current-state.md:18-44`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_error_taxonomy_lists_current_rag_failure_modes():
    text = Path("docs/evals/error-taxonomy.md").read_text()
    assert "schema_mismatch" in text
    assert "runtime_fallback_masked_failure" in text
    assert "parser_artifact_permission_failure" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_error_taxonomy_contract.py::test_error_taxonomy_lists_current_rag_failure_modes -v`

Expected: FAIL because the taxonomy document does not exist yet.

**Step 3: Write minimal implementation**

```markdown
## Failure Modes

- `schema_mismatch`
- `runtime_fallback_masked_failure`
- `parser_artifact_permission_failure`
- `empty_or_partial_index`
- `stale_semantic_cache_hit`
```

Also add 20-50 representative trace rows in `eval/review_samples/rag_trace_review.jsonl`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_error_taxonomy_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/evals/error-taxonomy.md eval/review_samples/rag_trace_review.jsonl tests/eval/test_error_taxonomy_contract.py docs/architecture/current-state.md
git commit -m "docs: add rag eval error taxonomy"
```

### Task 10: Build a frozen, code-based regression suite for retrieval and workflow

**Why it matters now:** The repo needs a small trusted eval set that does not depend on live PDF parsing, live LLM generation, or full-doc runtime downloads. This is the minimum trustworthy gate for RAG work.

**Required now or can wait:** Required now

**Recommended order:** 10

**Files:**
- Create: `tests/fixtures/rag/mini_chunks.jsonl`
- Create: `tests/fixtures/rag/retrieval_cases.jsonl`
- Create: `tests/eval/test_retrieval_regression.py`
- Create: `tests/eval/test_workflow_regression.py`
- Modify: `eval/retrieval_eval.py:1-278`
- Modify: `eval/workflow_eval.py:1-239`

**Step 1: Write the failing test**

```python
from eval.retrieval_eval import run_retrieval_eval


def test_retrieval_eval_can_run_against_frozen_fixture_corpus():
    report = run_retrieval_eval(
        samples=[{"question": "adult acetaminophen dose", "ground_truth_chunk_ids": ["mini-001"], "ground_truth_contexts": []}],
        mode="hybrid",
        ks=(1, 3),
        fetch_k=3,
        match_threshold=0.55,
    )
    assert report["sample_count"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_retrieval_regression.py::test_retrieval_eval_can_run_against_frozen_fixture_corpus -v`

Expected: FAIL because the eval path currently assumes the live database/index.

**Step 3: Write minimal implementation**

```python
def load_fixture_repo(path):
    repo = InMemoryVectorRepository()
    for row in read_jsonl(path):
        repo.upsert_chunk(**row)
    return repo
```

Use frozen fixture repos in the regression tests while keeping live DB eval as a separate manual mode.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_retrieval_regression.py tests/eval/test_workflow_regression.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/fixtures/rag/mini_chunks.jsonl tests/fixtures/rag/retrieval_cases.jsonl tests/eval/test_retrieval_regression.py tests/eval/test_workflow_regression.py eval/retrieval_eval.py eval/workflow_eval.py
git commit -m "test: add frozen rag regression suite"
```

### Task 11: Demote Tier 3 generation from gate to offline data-generation utility

**Why it matters now:** `eval/generate_tier3_rag.py` is useful, but it is not a trustworthy regression gate because it depends on live parsing, embeddings, and LLM generation. CI and routine validation should not rely on it.

**Required now or can wait:** Required now

**Recommended order:** 11

**Files:**
- Modify: `eval/generate_tier3_rag.py:118-198`
- Modify: `.github/workflows/eval.yml:40-47`
- Modify: `README.md:138-152`
- Create: `tests/eval/test_eval_gate_contract.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_ci_does_not_use_tier3_generator_as_regression_gate():
    workflow = Path(".github/workflows/eval.yml").read_text()
    assert "python -m eval.generate_tier3_rag" not in workflow
    assert "pytest tests/eval" in workflow
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_eval_gate_contract.py::test_ci_does_not_use_tier3_generator_as_regression_gate -v`

Expected: FAIL because the workflow currently has no real eval gate at all.

**Step 3: Write minimal implementation**

```yaml
- name: Run RAG regression suite
  run: pytest tests/eval tests/smoke -v
```

Update `README.md` to describe `generate_tier3_rag.py` as an offline dataset-generation command, not a gate.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_eval_gate_contract.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py .github/workflows/eval.yml README.md tests/eval/test_eval_gate_contract.py
git commit -m "ci: gate rag changes with stable regression tests"
```

---

## What Can Wait Until After This Plan

- LangGraph node redesign
- Planner/agent route redesign
- Frontend polish or session UX changes
- New LLM judges, scorecards, or dashboards
- Broader repository cleanup outside RAG/eval/runtime boundaries

## Definition of Done For This Phase

- A broken RAG runtime fails at preflight, not halfway through PDF ingestion
- API boot does not mutate schema or trigger indexing side effects
- RAG/eval paths do not silently fall back to in-memory persistence
- Docker, CI, and deploy install the app the same way
- Eval regression uses frozen fixtures and code-based checks
- Tier 3 generation is preserved as an offline utility, not mistaken for a stable gate
- The docs reflect the real current-state RAG/eval architecture
