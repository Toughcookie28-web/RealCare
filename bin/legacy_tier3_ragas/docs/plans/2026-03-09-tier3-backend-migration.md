# Tier 3 Backend Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate Tier 3 synthetic test generation from a Groq-only online path to a backend-neutral architecture that supports both Groq and local/vLLM backends, while making expensive stages checkpointed and resume-safe.

**Architecture:** Keep Tier 3 generation logic stable and move backend differences behind a small generator-specific runtime contract. Tier 3 should consume a resolved LangChain chat model, embeddings adapter, and artifact store, regardless of whether the backend is Groq or an OpenAI-compatible local/vLLM server. Expensive stages should write explicit artifacts and state manifests so reruns can resume without reparsing the PDF or regenerating already completed stages.

**Tech Stack:** Python 3.11, LangChain chat models, RAGAS 0.3.x, SQLAlchemy/PostgreSQL, JSON/JSONL artifacts, Docker Compose, optional vLLM OpenAI-compatible server

---

## Scope and non-goals

This plan is intentionally narrow.

In scope:
- Tier 3 generator backend abstraction
- Groq and local/vLLM backend support
- fail-fast behavior for generator backend selection
- stage checkpointing and resume safety
- persisted artifacts for reruns and backend comparison
- minimal container/runtime changes required to persist generator artifacts

Out of scope:
- broader app-runtime LLM refactor
- general RAG redesign
- HPC scheduler integration
- judge redesign or broader eval taxonomy changes
- prompt optimization or quality tuning of generated samples

## Current audit summary

### Current problems in the Tier 3 path

1. **Generator backend is hardcoded to app-wide chat settings**
   - [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) resolves its LLM through [tools/llm_client.py](/home/tough/medical_chatbot/MediGenius/tools/llm_client.py), which uses shared `llm_primary_*` and `llm_fallback_*` settings from [core/settings.py](/home/tough/medical_chatbot/MediGenius/core/settings.py).
   - Impact: Tier 3 cannot cleanly choose a generator-specific backend without affecting the rest of the app contract.

2. **Fallback behavior increases workload and hides failures**
   - [tools/llm_client.py](/home/tough/medical_chatbot/MediGenius/tools/llm_client.py) automatically falls back when the primary model errors.
   - Impact: this conflicts with the desired Tier 3 behavior of failing fast instead of silently increasing workload with another model/backend.

3. **Step 1 is checkpointed, but Step 2/3 are not**
   - [eval/generate_tier3_rag.py](/home/tough/medical_chatbot/MediGenius/eval/generate_tier3_rag.py) now persists `tier3_chunk_corpus.jsonl`, but raw RAGAS output and schema-converted draft samples are not persisted until late.
   - Impact: failures after or during RAGAS generation still force expensive rework.

4. **Artifact layout is not yet a clean run-oriented contract**
   - The current chunk corpus is stored in `eval/golden/v1/`, which persists across compose runs but mixes generator inputs with frozen golden datasets.
   - Impact: it works operationally, but it is not ideal for future Groq vs local/vLLM comparisons.

5. **Backend comparison would not be trustworthy today**
   - There is no run manifest capturing backend, model, corpus hash, RAGAS version, generation parameters, or completion state.
   - Impact: later comparisons can drift silently because more than one variable may change across runs.

## Proposed target design

### 1. One Tier 3 backend interface, two implementations

Introduce a small generator-specific backend contract, for example:

```python
class Tier3GeneratorBackend(Protocol):
    def build_chat_model(self) -> BaseChatModel: ...
    def label(self) -> str: ...
```

Implement:
- `GroqTier3Backend`
- `OpenAICompatibleTier3Backend`

Important: local/vLLM should use the OpenAI-compatible backend contract, not an HPC-only code path. That keeps laptop-local, Docker-local, and HPC served models under one architecture.

### 2. Tier 3-specific settings, separate from app chat routing

Add a generator-specific setting surface instead of reusing `llm_primary_*` and `llm_fallback_*`:

- `TIER3_LLM_BACKEND=groq|openai_compatible`
- `TIER3_LLM_MODEL=<model name>`
- `TIER3_LLM_BASE_URL=<optional base url>`
- `TIER3_LLM_API_KEY=<optional api key or token>`
- `TIER3_LLM_MAX_TOKENS=<generator cap>`
- `TIER3_LLM_TEMPERATURE=<generator temperature>`
- `TIER3_FAIL_FAST=true`

Do not add a fallback model path for Tier 3. If the configured backend is unavailable or rate-limited beyond the retry policy, fail.

### 3. Generator logic stays stable

The core generator flow should remain:

1. Load frozen chunk corpus
2. Build LLM backend + embeddings adapter
3. Run RAGAS generation
4. Convert to schema_v1
5. Write draft dataset

Only the backend resolution changes. The generation logic should not branch on “Groq vs HPC.”

### 4. Explicit artifact store and resume model

Define a dedicated artifact layout, preferably:

```text
eval/artifacts/tier3/<run_name>/
  manifest.json
  state.json
  chunk_corpus.jsonl
  raw_ragas_output.json
  tier3_rag.draft.jsonl
  metrics.json
```

Artifact meanings:
- `manifest.json`: backend, model, corpus hash, ragas version, generator params
- `state.json`: stage completion markers and timestamps
- `chunk_corpus.jsonl`: frozen Step 1 generator input
- `raw_ragas_output.json`: raw generator return before schema conversion
- `tier3_rag.draft.jsonl`: converted draft output
- `metrics.json`: counts, timings, duplicates, skipped rows, failure totals

### 5. Resume semantics

First implementation should support:
- resume from `chunk_corpus.jsonl`
- resume from `raw_ragas_output.json`
- resume from `tier3_rag.draft.jsonl` if only final write/validation failed

Important limitation:
- If RAGAS fails in the middle of its internal transformation loop, first-pass resume will still restart Step 2.
- That limitation is acceptable for the first implementation as long as Steps 1, 3, and 4 are checkpointed and the limitation is explicit.

### 6. Comparison contract for Groq vs local/vLLM

To compare backends later, hold these constant:
- identical `chunk_corpus.jsonl`
- identical `testset_size`
- identical RAGAS version
- identical difficulty mapping and duplicate filtering
- identical schema conversion code

Only change:
- backend kind
- backend model
- backend serving endpoint

The comparison report should include:
- run id
- backend label
- corpus hash
- requested vs generated sample count
- duplicate count
- empty-context count
- difficulty distribution
- average generation duration

---

## Smallest changes needed first

These are the first changes to implement before anything broader:

1. Add Tier 3-specific backend settings and a generator backend factory.
2. Route the current Groq path through that factory with fail-fast behavior and no fallback.
3. Add an OpenAI-compatible backend implementation for local/vLLM.
4. Add a dedicated Tier 3 artifact store with `manifest.json`, `state.json`, and `raw_ragas_output.json`.
5. Move Step 1 corpus checkpointing into that artifact store.

Do not start with HPC-specific deployment code. The first success condition is: the same generator code can point at Groq or an OpenAI-compatible local/vLLM endpoint by changing Tier 3 settings only.

---

### Task 1: Lock the current Tier 3 contract with failing tests

**Files:**
- Create: `tests/eval/test_tier3_backend_runtime_contract.py`
- Modify: `tests/eval/test_tier3_generator_contract.py`
- Reference: `eval/generate_tier3_rag.py`
- Reference: `tools/llm_client.py`
- Reference: `core/settings.py`

**Step 1: Write the failing tests**

Add tests that prove the current gaps:

```python
def test_tier3_does_not_depend_on_app_llm_primary_settings():
    ...

def test_tier3_backend_contract_rejects_fallback_mode():
    ...

def test_tier3_artifact_store_paths_are_explicit():
    ...
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py -v
```

Expected:
- failure because there is no Tier 3-specific backend contract yet

**Step 3: Write minimal implementation later**

This task exists only to lock the desired contract before touching backend code.

**Step 4: Run tests again once implementation exists**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py tests/eval/test_tier3_generator_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add tests/eval/test_tier3_backend_runtime_contract.py tests/eval/test_tier3_generator_contract.py
git commit -m "test: lock tier3 backend runtime contract"
```

### Task 2: Introduce Tier 3-specific settings

**Files:**
- Modify: `core/settings.py`
- Create: `eval/tier3_config.py`
- Test: `tests/eval/test_tier3_backend_runtime_contract.py`

**Step 1: Write the failing test**

Add test cases for:
- `TIER3_LLM_BACKEND`
- `TIER3_LLM_MODEL`
- `TIER3_LLM_BASE_URL`
- `TIER3_LLM_API_KEY`
- `TIER3_FAIL_FAST`
- no Tier 3 fallback settings

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py::test_tier3_settings_are_separate_from_app_llm_settings -v
```

Expected:
- fail because settings do not exist yet

**Step 3: Write minimal implementation**

Create a small typed config surface, for example:

```python
@dataclass(frozen=True)
class Tier3Config:
    backend: str
    model: str
    base_url: str | None
    api_key: str | None
    max_tokens: int
    temperature: float
    fail_fast: bool
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py -v
```

Expected:
- pass for settings contract

**Step 5: Commit**

```bash
git add core/settings.py eval/tier3_config.py tests/eval/test_tier3_backend_runtime_contract.py
git commit -m "feat: add tier3-specific backend settings"
```

### Task 3: Add a generator backend factory with Groq and OpenAI-compatible backends

**Files:**
- Create: `eval/tier3_backend.py`
- Modify: `eval/generate_tier3_rag.py`
- Test: `tests/eval/test_tier3_backend_runtime_contract.py`

**Step 1: Write the failing test**

Cover:
- Groq backend resolves `ChatGroq`
- OpenAI-compatible backend resolves `ChatOpenAI`
- local/vLLM path uses the OpenAI-compatible backend
- invalid backend raises immediately
- fail-fast prevents fallback use

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py::test_openai_compatible_backend_is_supported_for_local_vllm -v
```

Expected:
- fail because backend factory does not exist yet

**Step 3: Write minimal implementation**

Implement:

```python
def build_tier3_chat_model(config: Tier3Config) -> BaseChatModel:
    ...
```

Rules:
- `groq` -> `ChatGroq`
- `openai_compatible` -> `ChatOpenAI` with `base_url`
- no fallback resolution
- missing/invalid config raises `RuntimeError`

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add eval/tier3_backend.py eval/generate_tier3_rag.py tests/eval/test_tier3_backend_runtime_contract.py
git commit -m "feat: add tier3 backend factory"
```

### Task 4: Route Tier 3 generation through the new backend factory

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Reference: `tools/llm_client.py`
- Test: `tests/eval/test_tier3_backend_runtime_contract.py`

**Step 1: Write the failing test**

Add a test that proves:
- Tier 3 no longer calls `get_fallback_llm()`
- Tier 3 backend label comes from the explicit Tier 3 backend factory

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py::test_tier3_generation_uses_backend_factory_not_global_fallback_llm -v
```

Expected:
- fail

**Step 3: Write minimal implementation**

Replace `_resolve_llm()` in the generator with a generator-specific backend resolver.

Keep:
- `_resolve_embeddings()`
- RAGAS wrapping and schema conversion

Change:
- backend resolution path only

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_backend_runtime_contract.py tests/eval/test_tier3_generator_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py tests/eval/test_tier3_backend_runtime_contract.py tests/eval/test_tier3_generator_contract.py
git commit -m "refactor: decouple tier3 generation from app llm fallback"
```

### Task 5: Create a dedicated Tier 3 artifact store

**Files:**
- Create: `eval/tier3_artifacts.py`
- Modify: `eval/generate_tier3_rag.py`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Test: `tests/eval/test_tier3_artifact_store_contract.py`

**Step 1: Write the failing test**

Test for:
- explicit run directory under `eval/artifacts/tier3/`
- explicit `manifest.json`, `state.json`, `chunk_corpus.jsonl`
- compose mount persists `eval/artifacts`

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_artifact_store_contract.py -v
```

Expected:
- fail because artifact store and compose mount do not exist

**Step 3: Write minimal implementation**

Add a helper like:

```python
@dataclass(frozen=True)
class Tier3ArtifactPaths:
    run_dir: Path
    manifest_path: Path
    state_path: Path
    chunk_corpus_path: Path
    raw_dataset_path: Path
    draft_samples_path: Path
    metrics_path: Path
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_artifact_store_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add eval/tier3_artifacts.py eval/generate_tier3_rag.py docker-compose.yml README.md tests/eval/test_tier3_artifact_store_contract.py
git commit -m "feat: add tier3 artifact store"
```

### Task 6: Move Step 1 corpus checkpointing into the artifact store and add manifest/state files

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Modify: `eval/tier3_artifacts.py`
- Test: `tests/eval/test_tier3_generator_contract.py`
- Test: `tests/eval/test_tier3_artifact_store_contract.py`

**Step 1: Write the failing test**

Verify:
- Step 1 writes `chunk_corpus.jsonl`
- manifest records backend/model/ragas version/source hash
- state records `chunk_corpus_ready=true`

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_artifact_store_contract.py::test_step1_updates_manifest_and_state -v
```

Expected:
- fail

**Step 3: Write minimal implementation**

Add stage-state updates:
- `chunk_corpus_ready`
- `raw_dataset_ready`
- `schema_draft_ready`
- `final_dataset_written`

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_generator_contract.py tests/eval/test_tier3_artifact_store_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py eval/tier3_artifacts.py tests/eval/test_tier3_generator_contract.py tests/eval/test_tier3_artifact_store_contract.py
git commit -m "feat: add tier3 manifest and stage state"
```

### Task 7: Checkpoint raw RAGAS output and schema-converted draft samples

**Files:**
- Modify: `eval/generate_tier3_rag.py`
- Modify: `eval/tier3_artifacts.py`
- Test: `tests/eval/test_tier3_artifact_store_contract.py`

**Step 1: Write the failing test**

Test for:
- raw generator output persisted after Step 2
- schema-converted draft persisted after Step 3
- rerun can skip Step 2 when `raw_ragas_output.json` exists
- rerun can skip Step 3 when `tier3_rag.draft.jsonl` exists

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_artifact_store_contract.py::test_tier3_resume_skips_completed_stages -v
```

Expected:
- fail

**Step 3: Write minimal implementation**

Rules:
- do not change sample conversion logic
- only add checkpoint/load behavior around stage boundaries
- if `--resume` is true, prefer existing artifacts for completed stages

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_artifact_store_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add eval/generate_tier3_rag.py eval/tier3_artifacts.py tests/eval/test_tier3_artifact_store_contract.py
git commit -m "feat: add tier3 stage checkpoints and resume"
```

### Task 8: Add a minimal run comparison contract for Groq vs local/vLLM

**Files:**
- Create: `eval/tier3_compare_runs.py`
- Create: `tests/eval/test_tier3_compare_contract.py`
- Modify: `README.md`

**Step 1: Write the failing test**

Test for:
- comparison rejects runs with different corpus hashes
- comparison rejects runs with different RAGAS versions unless explicitly allowed
- comparison reports backend label, model, sample counts, duplicate counts, empty-context counts

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_compare_contract.py -v
```

Expected:
- fail because comparison tool does not exist

**Step 3: Write minimal implementation**

The script can initially be simple:

```python
def compare_runs(left_manifest: Path, right_manifest: Path) -> dict[str, Any]:
    ...
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/eval/test_tier3_compare_contract.py -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add eval/tier3_compare_runs.py tests/eval/test_tier3_compare_contract.py README.md
git commit -m "feat: add tier3 backend comparison contract"
```

### Task 9: Add operator docs for Groq, local laptop, and HPC/vLLM usage

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Write the failing doc contract test**

If desired, add or extend a smoke/doc test to check:
- Tier 3 backend config is documented
- artifact store path is documented
- resume behavior is documented
- local/vLLM path is documented as OpenAI-compatible, not HPC-only

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/eval/test_tier3_artifact_store_contract.py tests/eval/test_tier3_backend_runtime_contract.py -v
```

Expected:
- fail on missing docs assertions

**Step 3: Write minimal documentation**

Document:
- how to run Tier 3 with Groq
- how to run Tier 3 with local/vLLM
- which artifacts are reused on rerun
- what must match for valid backend comparison

**Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/eval tests/rag tests/smoke -v
```

Expected:
- pass

**Step 5: Commit**

```bash
git add README.md docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: document tier3 backend migration and artifact contract"
```

---

## Recommended execution order

Phase 1:
1. Task 1
2. Task 2
3. Task 3
4. Task 4

Phase 2:
5. Task 5
6. Task 6
7. Task 7

Phase 3:
8. Task 8
9. Task 9

## Acceptance criteria

The migration is successful when all of the following are true:

- Tier 3 backend selection is explicit and generator-specific.
- Tier 3 can run against Groq or an OpenAI-compatible local/vLLM endpoint without changing generation logic.
- Tier 3 does not silently fall back to another backend/model.
- Step 1 never reparses the PDF if a valid chunk corpus already exists.
- Step 2 and Step 3 artifacts are checkpointed and reusable.
- A run manifest exists that makes Groq vs local/vLLM comparisons trustworthy.
- The required regression gate remains frozen and code-based; Tier 3 stays an offline utility.
