# Parent-Child Chunk Hierarchy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two-tier chunking where small child chunks are used for precise retrieval and large parent chunks are passed to the executor for richer context.

**Architecture:** Child chunks (~200 tokens) are embedded and searched via hybrid retrieval. Parent chunks (~800–1000 tokens) group 3–5 consecutive children within the same section. After retrieval finds relevant children, the system resolves their parent chunk IDs and fetches parent content from the DB. The executor uses parent content for context assembly instead of the smaller child content. Both tiers are stored in `document_chunks` — parents with NULL embedding (never searched), children with real embeddings. The feature is behind `ENABLE_PARENT_CHILD_CHUNKING` (default False) so existing single-tier behavior is preserved until the user reindexes.

**Tech Stack:** LangGraph, PostgreSQL + pgvector, LangChain Documents, Pydantic Settings, pytest

---

## How it works

```
Current (flat):
  pdf → chunk (400 tok) → embed → index → search → executor

Parent-child:
  pdf → chunk (200 tok children) → group into parents (800-1000 tok)
      → embed children only → index both → search children
      → resolve parent IDs → fetch parent content → executor uses parent
```

### Key design decisions

1. **Parents stored in DB with NULL embedding** — PostgreSQL cosine_distance sorts NULLs last (never in top-K). InMemoryVectorRepository returns 0.0 similarity for empty embeddings. Parents never appear in search results.
2. **Children reference parents via `parent_chunk_id` in metadata** — no schema migration needed.
3. **`get_chunks_by_ids()` added to repos** — needed to fetch parent content after retrieval.
4. **Feature flag** — `ENABLE_PARENT_CHILD_CHUNKING=false` by default. When disabled, behavior is identical to today. When enabled, chunker uses smaller children and creates parents. Requires reindex.
5. **Replaces adjacent chunk expansion** — parent chunks subsume the ±1 neighbor expansion. When parent-child is enabled, the executor skips adjacent expansion (parent already provides the broader context).

---

### Task 1: Add parent-child settings

**Files:**
- Modify: `core/settings.py`
- Test: `tests/rag/test_pdf_chunker_contract.py` (existing)

**Step 1: Add settings**

Add these fields to `core/settings.py` Settings class, after the existing chunk settings:

```python
enable_parent_child_chunking: bool = Field(default=False, alias='ENABLE_PARENT_CHILD_CHUNKING')
chunk_child_target: int = Field(default=200, alias='CHUNK_CHILD_TARGET')
chunk_child_max: int = Field(default=250, alias='CHUNK_CHILD_MAX')
chunk_parent_max_children: int = Field(default=5, alias='CHUNK_PARENT_MAX_CHILDREN')
```

**Step 2: Run existing tests**

Run: `pytest tests/rag/test_pdf_chunker_contract.py -v`
Expected: All PASS (settings are backward compatible)

**Step 3: Commit**

```bash
git add core/settings.py
git commit -m "feat: add parent-child chunking settings (disabled by default)"
```

---

### Task 2: Add parent-child grouping function to pdf_chunker + tests

**Files:**
- Modify: `tools/pdf_chunker.py`
- Create: `tests/rag/test_parent_child_chunking.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_parent_child_chunking.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document


def test_group_children_into_parents_creates_parents():
    from tools.pdf_chunker import group_children_into_parents

    children = [
        Document(page_content=f"Child {i} content about aspirin.", metadata={
            "chunk_id": f"doc-sec01-p1-t{i:03d}",
            "doc_id": "test-doc",
            "section_path": "Chapter 1 > Pain",
            "page_start": 1,
            "page_end": 1,
            "content_type": "text_section",
            "context_key": "doc-sec01",
        })
        for i in range(6)
    ]
    parents, updated_children = group_children_into_parents(children, max_children_per_parent=3)

    # Should create 2 parents (6 children / 3 per parent)
    assert len(parents) == 2
    assert all(p.metadata["chunk_type"] == "parent" for p in parents)

    # Every child should have parent_chunk_id set
    assert all(c.metadata.get("parent_chunk_id") for c in updated_children)
    assert all(c.metadata["chunk_type"] == "child" for c in updated_children)

    # Parent content should be concatenation of children
    first_parent = parents[0]
    assert "Child 0" in first_parent.page_content
    assert "Child 2" in first_parent.page_content


def test_group_children_respects_section_boundaries():
    from tools.pdf_chunker import group_children_into_parents

    children = [
        Document(page_content="Section A content.", metadata={
            "chunk_id": "doc-sec01-p1-t000",
            "doc_id": "test-doc",
            "section_path": "Section A",
            "context_key": "doc-sec01",
            "content_type": "text_section",
        }),
        Document(page_content="Section B content.", metadata={
            "chunk_id": "doc-sec02-p2-t000",
            "doc_id": "test-doc",
            "section_path": "Section B",
            "context_key": "doc-sec02",
            "content_type": "text_section",
        }),
    ]
    parents, updated_children = group_children_into_parents(children, max_children_per_parent=5)

    # Different sections -> different parents
    assert len(parents) == 2
    parent_ids = {c.metadata["parent_chunk_id"] for c in updated_children}
    assert len(parent_ids) == 2


def test_table_chunks_get_own_parent():
    from tools.pdf_chunker import group_children_into_parents

    children = [
        Document(page_content="Text content.", metadata={
            "chunk_id": "doc-sec01-p1-t000",
            "doc_id": "test-doc",
            "section_path": "Chapter 1",
            "context_key": "doc-sec01",
            "content_type": "text_section",
        }),
        Document(page_content="| Drug | Dose |", metadata={
            "chunk_id": "doc-sec01-tbl01-r00",
            "doc_id": "test-doc",
            "section_path": "Chapter 1",
            "context_key": "doc-sec01-tbl01",
            "content_type": "table",
        }),
    ]
    parents, updated_children = group_children_into_parents(children, max_children_per_parent=5)

    # Table should be its own parent (tables are self-contained)
    table_child = updated_children[1]
    text_child = updated_children[0]
    assert table_child.metadata["parent_chunk_id"] != text_child.metadata["parent_chunk_id"]


def test_group_children_empty_input():
    from tools.pdf_chunker import group_children_into_parents

    parents, children = group_children_into_parents([], max_children_per_parent=5)
    assert parents == []
    assert children == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_parent_child_chunking.py -v`
Expected: FAIL with "cannot import name 'group_children_into_parents'"

**Step 3: Implement `group_children_into_parents` in `tools/pdf_chunker.py`**

Add this function at the end of `tools/pdf_chunker.py`:

```python
def group_children_into_parents(
    chunks: list[Document],
    max_children_per_parent: int = 5,
) -> tuple[list[Document], list[Document]]:
    """Group consecutive same-section text chunks into parent chunks.

    Tables become their own parent (they are self-contained retrieval units).
    Text chunks within the same section are grouped into parents of up to
    max_children_per_parent children.

    Returns (parent_chunks, updated_child_chunks) where each child has
    'parent_chunk_id' and 'chunk_type' set in metadata.
    """
    if not chunks:
        return [], []

    parents: list[Document] = []
    updated_children: list[Document] = []

    # Group chunks by context_key (section identifier)
    groups: list[tuple[str, list[Document]]] = []
    current_key: str | None = None
    current_group: list[Document] = []

    for chunk in chunks:
        meta = chunk.metadata
        key = meta.get("context_key", "")
        content_type = meta.get("content_type", "text_section")

        # Tables always start a new group (self-contained)
        if content_type == "table":
            if current_group:
                groups.append((current_key or "", current_group))
                current_group = []
            groups.append((key, [chunk]))
            current_key = None
            continue

        # Same section -> accumulate
        if key == current_key:
            current_group.append(chunk)
        else:
            if current_group:
                groups.append((current_key or "", current_group))
            current_key = key
            current_group = [chunk]

    if current_group:
        groups.append((current_key or "", current_group))

    # Create parents from groups
    parent_idx = 0
    for _section_key, group_chunks in groups:
        # Split large groups into sub-groups of max_children_per_parent
        for start in range(0, len(group_chunks), max_children_per_parent):
            sub_group = group_chunks[start:start + max_children_per_parent]
            first_meta = sub_group[0].metadata

            parent_chunk_id = f"{first_meta.get('doc_id', 'doc')}-parent-{parent_idx:04d}"
            parent_idx += 1

            # Build parent content by joining children
            parent_content = "\n\n".join(c.page_content for c in sub_group)

            # Collect page range
            pages = [c.metadata.get("page_start") or c.metadata.get("page") for c in sub_group]
            pages = [p for p in pages if p is not None]
            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None

            parent = Document(
                page_content=parent_content,
                metadata={
                    "chunk_id": parent_chunk_id,
                    "doc_id": first_meta.get("doc_id", ""),
                    "section_path": first_meta.get("section_path", ""),
                    "content_type": first_meta.get("content_type", "text_section"),
                    "context_key": first_meta.get("context_key", ""),
                    "context_prefix": first_meta.get("context_prefix", ""),
                    "chunk_type": "parent",
                    "child_chunk_ids": [c.metadata["chunk_id"] for c in sub_group],
                    "page_start": page_start,
                    "page_end": page_end,
                    "page": page_start,
                    "source": first_meta.get("source", ""),
                },
            )
            parents.append(parent)

            # Tag children
            for child in sub_group:
                child.metadata["parent_chunk_id"] = parent_chunk_id
                child.metadata["chunk_type"] = "child"
                updated_children.append(child)

    return parents, updated_children
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/rag/test_parent_child_chunking.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add tools/pdf_chunker.py tests/rag/test_parent_child_chunking.py
git commit -m "feat: add parent-child chunk grouping function"
```

---

### Task 3: Add `get_chunks_by_ids()` to repositories + tests

**Files:**
- Modify: `db/repositories.py`
- Create: `tests/rag/test_parent_lookup.py`

**Step 1: Write the failing tests**

```python
# tests/rag/test_parent_lookup.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_inmemory_get_chunks_by_ids():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk("chunk-1", "doc-1", "Content one", [0.1] * 768, metadata={"chunk_type": "parent"})
    repo.upsert_chunk("chunk-2", "doc-1", "Content two", [0.2] * 768, metadata={"chunk_type": "child"})
    repo.upsert_chunk("chunk-3", "doc-1", "Content three", [0.3] * 768, metadata={"chunk_type": "parent"})

    results = repo.get_chunks_by_ids(["chunk-1", "chunk-3"])
    assert len(results) == 2
    ids = {doc.metadata["chunk_id"] for doc in results}
    assert ids == {"chunk-1", "chunk-3"}


def test_inmemory_get_chunks_by_ids_missing():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk("chunk-1", "doc-1", "Content", [0.1] * 768)

    results = repo.get_chunks_by_ids(["chunk-1", "nonexistent"])
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "chunk-1"


def test_inmemory_get_chunks_by_ids_empty():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    results = repo.get_chunks_by_ids([])
    assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_parent_lookup.py -v`
Expected: FAIL with "InMemoryVectorRepository has no attribute 'get_chunks_by_ids'"

**Step 3: Add `get_chunks_by_ids` to both repositories**

In `db/repositories.py`, add to `VectorRepository` class:

```python
def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Document]:
    """Fetch chunks by their chunk_ids."""
    if not chunk_ids:
        return []
    rows = self.db.execute(
        select(DocumentChunkModel)
        .where(DocumentChunkModel.chunk_id.in_(chunk_ids))
    ).scalars().all()
    return [_orm_row_to_doc(row) for row in rows]
```

Add to `InMemoryVectorRepository` class:

```python
def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Document]:
    """Fetch chunks by their chunk_ids."""
    if not chunk_ids:
        return []
    results = []
    for cid in chunk_ids:
        row = self._chunks.get(cid)
        if row is not None:
            results.append(_row_to_doc(row))
    return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/rag/test_parent_lookup.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add db/repositories.py tests/rag/test_parent_lookup.py
git commit -m "feat: add get_chunks_by_ids to vector repositories"
```

---

### Task 4: Update vector_store.py to handle parent-child during ingest

**Files:**
- Modify: `tools/vector_store.py`
- Test: `tests/rag/test_parent_child_chunking.py` (add ingest test)

**Step 1: Write the failing test**

Append to `tests/rag/test_parent_child_chunking.py`:

```python
def test_build_index_batch_handles_parents_with_none_embedding():
    from tools.vector_store import build_index_batch

    parent = Document(page_content="Parent content.", metadata={
        "chunk_id": "doc-parent-0000",
        "doc_id": "test-doc",
        "chunk_type": "parent",
        "section_path": "Chapter 1",
    })

    batch = build_index_batch([parent], [None], default_doc_id="test-doc")
    assert len(batch) == 1
    assert batch[0]["chunk_id"] == "doc-parent-0000"
    assert batch[0]["embedding"] is None


def test_embed_chunk_documents_skips_parents():
    from unittest.mock import patch
    from tools.vector_store import embed_chunk_documents

    parent = Document(page_content="Parent content.", metadata={
        "chunk_type": "parent",
        "embedding_prefix": "Ch1",
    })
    child = Document(page_content="Child content.", metadata={
        "chunk_type": "child",
        "embedding_prefix": "Ch1",
    })

    with patch("tools.vector_store.embed_documents_batch") as mock_embed:
        mock_embed.return_value = [[0.1] * 768]
        embeddings = embed_chunk_documents([parent, child])

    # Should only embed the child, parent gets None
    assert len(embeddings) == 2
    assert embeddings[0] is None  # parent
    assert embeddings[1] == [0.1] * 768  # child
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_parent_child_chunking.py::test_embed_chunk_documents_skips_parents -v`
Expected: FAIL

**Step 3: Update vector_store.py**

Modify `embed_chunk_documents()` to skip parents:

```python
def embed_chunk_documents(chunks: list[Any]) -> list[list[float] | None]:
    """Embed chunks, skipping parent chunks (they don't need embeddings)."""
    child_indices: list[int] = []
    child_texts: list[str] = []

    all_texts = build_embedding_texts(chunks)
    for i, chunk in enumerate(chunks):
        if (chunk.metadata or {}).get("chunk_type") == "parent":
            continue
        child_indices.append(i)
        child_texts.append(all_texts[i])

    if not child_texts:
        return [None] * len(chunks)

    child_embeddings = embed_documents_batch(child_texts)
    logger.info('ingest_embed_complete', extra={'chunks': len(chunks), 'children_embedded': len(child_embeddings)})

    # Map back: parents get None, children get their embedding
    result: list[list[float] | None] = [None] * len(chunks)
    for idx, emb in zip(child_indices, child_embeddings):
        result[idx] = emb
    return result
```

Update `ingest_pdf_to_vector_store()` to call the hierarchy builder when enabled:

```python
def ingest_pdf_to_vector_store(db: Session | None, pdf_path: str) -> IngestReport:
    parsed_doc = load_parsed_pdf(pdf_path)
    chunks = chunk_pdf_document(parsed_doc)
    doc_id = parsed_doc.doc_id or Path(pdf_path).stem
    repo = VectorRepository(db) if db is not None else InMemoryVectorRepository()

    # ... existing logging ...

    # Optional LLM-based chunk enrichment (existing code, unchanged)
    if get_settings().enrich_chunks_with_llm and chunks:
        # ... existing enrichment code ...

    # Optional parent-child hierarchy
    if get_settings().enable_parent_child_chunking and chunks:
        from tools.pdf_chunker import group_children_into_parents
        parents, chunks = group_children_into_parents(
            chunks,
            max_children_per_parent=get_settings().chunk_parent_max_children,
        )
        all_chunks = parents + chunks  # parents first, then children
        logger.info("ingest_hierarchy_complete", extra={
            "doc_id": doc_id, "parents": len(parents), "children": len(chunks),
        })
    else:
        all_chunks = chunks

    if not all_chunks:
        return IngestReport(...)

    embeddings = embed_chunk_documents(all_chunks)
    batch = build_index_batch(all_chunks, embeddings, default_doc_id=doc_id)
    inserted, total_chunks = index_chunk_batch(repo, batch, doc_id=doc_id)
    # ... rest unchanged ...
```

Also update `build_index_batch` to accept `None` embeddings:

```python
def build_index_batch(chunks: list[Any], embeddings: list[list[float] | None], default_doc_id: str) -> list[dict[str, Any]]:
    if len(chunks) != len(embeddings):
        raise ValueError(...)

    batch: list[dict[str, Any]] = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = chunk.metadata or {}
        chunk_id = metadata.get('chunk_id') or metadata.get('child_id', f'chunk-{idx}')
        batch.append({
            'chunk_id': chunk_id,
            'doc_id': metadata.get('doc_id', default_doc_id),
            'content': chunk.page_content,
            'embedding': embedding,  # None for parents
            'page': metadata.get('page'),
            'section': metadata.get('section_path') or metadata.get('section'),
            'metadata': metadata,
        })
    return batch
```

**Step 4: Run tests**

Run: `pytest tests/rag/test_parent_child_chunking.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add tools/vector_store.py tests/rag/test_parent_child_chunking.py
git commit -m "feat: update ingest pipeline to handle parent-child chunk hierarchy"
```

---

### Task 5: Update chunker to use smaller child sizes when parent-child enabled

**Files:**
- Modify: `tools/pdf_chunker.py`
- Test: `tests/rag/test_parent_child_chunking.py`

**Step 1: Write the failing test**

Append to `tests/rag/test_parent_child_chunking.py`:

```python
def test_config_from_settings_uses_child_sizes_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_PARENT_CHILD_CHUNKING", "true")
    monkeypatch.setenv("CHUNK_CHILD_TARGET", "200")
    monkeypatch.setenv("CHUNK_CHILD_MAX", "250")

    from core.settings import Settings
    settings = Settings()
    assert settings.enable_parent_child_chunking is True

    from tools.pdf_chunker import _config_from_settings_for_mode
    config = _config_from_settings_for_mode(settings)
    assert config.text_chunk_target == 200
    assert config.text_chunk_max == 250
```

**Step 2: Run to verify it fails**

Run: `pytest tests/rag/test_parent_child_chunking.py::test_config_from_settings_uses_child_sizes_when_enabled -v`
Expected: FAIL

**Step 3: Add `_config_from_settings_for_mode` to pdf_chunker.py**

```python
def _config_from_settings_for_mode(settings=None) -> ChunkConfig:
    """Build ChunkConfig, using child sizes when parent-child chunking is enabled."""
    if settings is None:
        from core.settings import get_settings
        settings = get_settings()

    if settings.enable_parent_child_chunking:
        return ChunkConfig(
            text_chunk_target=settings.chunk_child_target,
            text_chunk_max=settings.chunk_child_max,
            text_chunk_overlap=settings.chunk_text_overlap,
            table_chunk_max=settings.chunk_table_max,
        )
    return ChunkConfig(
        text_chunk_target=settings.chunk_text_target,
        text_chunk_max=settings.chunk_text_max,
        text_chunk_overlap=settings.chunk_text_overlap,
        table_chunk_max=settings.chunk_table_max,
    )
```

Also update `_config_from_settings()` to delegate to the new function:

```python
def _config_from_settings() -> ChunkConfig:
    """Build a ChunkConfig from centralised Settings (env-configurable)."""
    return _config_from_settings_for_mode()
```

**Step 4: Run tests**

Run: `pytest tests/rag/test_parent_child_chunking.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/pdf_chunker.py tests/rag/test_parent_child_chunking.py
git commit -m "feat: use smaller child chunk sizes when parent-child chunking enabled"
```

---

### Task 6: Update retriever to resolve parent chunks after retrieval

**Files:**
- Modify: `agents/retriever_agent.py`
- Create: `tests/rag/test_parent_child_retrieval.py`

**Step 1: Write the failing test**

```python
# tests/rag/test_parent_child_retrieval.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import patch
from langchain_core.documents import Document
from db.repositories import InMemoryVectorRepository


def test_resolve_parents_attaches_parent_content():
    from agents.retriever_agent import _resolve_parent_chunks

    repo = InMemoryVectorRepository()
    # Store a parent chunk
    repo.upsert_chunk(
        "doc-parent-0000", "test-doc",
        "Full parent content about aspirin dosing and side effects.",
        [0.0] * 768,
        metadata={"chunk_type": "parent"},
    )

    # Child references the parent
    children = [
        Document(page_content="Short child about aspirin dosing.", metadata={
            "chunk_id": "doc-sec01-p1-t000",
            "parent_chunk_id": "doc-parent-0000",
            "chunk_type": "child",
        }),
    ]

    resolved = _resolve_parent_chunks(children, repo)
    assert len(resolved) == 1
    assert resolved[0].metadata["parent_content"] == "Full parent content about aspirin dosing and side effects."


def test_resolve_parents_deduplicates():
    from agents.retriever_agent import _resolve_parent_chunks

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        "doc-parent-0000", "test-doc", "Parent content.",
        [0.0] * 768, metadata={"chunk_type": "parent"},
    )

    # Two children from same parent
    children = [
        Document(page_content="Child 1.", metadata={
            "chunk_id": "child-1", "parent_chunk_id": "doc-parent-0000", "chunk_type": "child",
        }),
        Document(page_content="Child 2.", metadata={
            "chunk_id": "child-2", "parent_chunk_id": "doc-parent-0000", "chunk_type": "child",
        }),
    ]

    resolved = _resolve_parent_chunks(children, repo)
    # Both children get the same parent content
    assert all(d.metadata.get("parent_content") == "Parent content." for d in resolved)


def test_resolve_parents_skips_non_child_chunks():
    from agents.retriever_agent import _resolve_parent_chunks

    repo = InMemoryVectorRepository()
    # Flat chunk (no parent-child, pre-existing behavior)
    flat = [
        Document(page_content="Flat chunk.", metadata={
            "chunk_id": "flat-1",
        }),
    ]

    resolved = _resolve_parent_chunks(flat, repo)
    assert len(resolved) == 1
    assert "parent_content" not in resolved[0].metadata
```

**Step 2: Run to verify failure**

Run: `pytest tests/rag/test_parent_child_retrieval.py -v`
Expected: FAIL with "cannot import name '_resolve_parent_chunks'"

**Step 3: Add `_resolve_parent_chunks` to retriever_agent.py**

Add this function to `agents/retriever_agent.py`:

```python
def _resolve_parent_chunks(
    docs: list[Document],
    repo: VectorRepository | InMemoryVectorRepository,
) -> list[Document]:
    """Resolve parent chunk content for child chunks.

    For each child chunk that has a parent_chunk_id, fetch the parent's
    content and attach it as 'parent_content' in metadata.
    Non-child chunks (flat or parent) pass through unchanged.
    """
    # Collect unique parent IDs
    parent_ids: set[str] = set()
    for doc in docs:
        pid = doc.metadata.get("parent_chunk_id")
        if pid and doc.metadata.get("chunk_type") == "child":
            parent_ids.add(pid)

    if not parent_ids:
        return docs

    # Batch fetch parents
    parent_docs = repo.get_chunks_by_ids(list(parent_ids))
    parent_map: dict[str, str] = {
        d.metadata["chunk_id"]: d.page_content for d in parent_docs
    }

    # Attach parent content to children
    for doc in docs:
        pid = doc.metadata.get("parent_chunk_id")
        if pid and pid in parent_map:
            doc.metadata["parent_content"] = parent_map[pid]

    return docs
```

Then wire it into `RetrieverAgent`, after reranking and before state assignment:

```python
    ranked, scores = rerank(query, docs)

    # Resolve parent chunks if parent-child chunking is active
    top_docs = ranked[:5]
    if any(d.metadata.get("chunk_type") == "child" for d in top_docs):
        top_docs = _resolve_parent_chunks(top_docs, repo)

    state['documents'] = top_docs
```

**Step 4: Run tests**

Run: `pytest tests/rag/test_parent_child_retrieval.py -v`
Expected: PASS (3 tests)

**Step 5: Run all tests to check for regressions**

Run: `pytest tests/ -v --timeout=30`
Expected: All PASS

**Step 6: Commit**

```bash
git add agents/retriever_agent.py tests/rag/test_parent_child_retrieval.py
git commit -m "feat: resolve parent chunks after retrieval for richer executor context"
```

---

### Task 7: Update executor to use parent content for context assembly

**Files:**
- Modify: `agents/executor_agent.py`
- Test: existing tests + new unit test

**Step 1: Write the failing test**

Append to `tests/rag/test_parent_child_retrieval.py`:

```python
def test_build_context_blocks_uses_parent_content():
    from agents.executor_agent import _build_context_blocks

    docs = [
        Document(page_content="Short child chunk.", metadata={
            "chunk_id": "child-1",
            "section_path": "Chapter 1",
            "context_key": "doc-sec01",
            "chunk_type": "child",
            "parent_content": "Full parent content with much more detail about the topic.",
        }),
    ]

    blocks = _build_context_blocks(docs)
    # Should use parent_content instead of child's page_content
    assert "Full parent content" in blocks[0]
    assert "Short child chunk" not in blocks[0]
```

**Step 2: Run to verify failure**

Run: `pytest tests/rag/test_parent_child_retrieval.py::test_build_context_blocks_uses_parent_content -v`
Expected: FAIL (executor still uses page_content)

**Step 3: Modify `_build_context_blocks` in executor_agent.py**

In the `_build_context_blocks` function, where it builds the chunk content, check for `parent_content`:

Find this section in the function:

```python
        parts = []
        # Prepend context summary if available
        context_summary = metadata.get('context_summary')
        if context_summary:
            parts.append(f"[Context: {context_summary}]")
        for t in before_texts:
            parts.append(f"<context>\n{t}\n</context>")
        parts.append(f"<chunk>\n{doc.page_content.strip()}\n</chunk>")
        for t in after_texts:
            parts.append(f"<context>\n{t}\n</context>")
```

Replace with:

```python
        parts = []
        # Prepend context summary if available
        context_summary = metadata.get('context_summary')
        if context_summary:
            parts.append(f"[Context: {context_summary}]")

        # Use parent content if available (parent-child chunking),
        # otherwise use child content with optional adjacent expansion
        parent_content = metadata.get('parent_content')
        if parent_content:
            parts.append(f"<chunk>\n{parent_content.strip()}\n</chunk>")
        else:
            for t in before_texts:
                parts.append(f"<context>\n{t}\n</context>")
            parts.append(f"<chunk>\n{doc.page_content.strip()}\n</chunk>")
            for t in after_texts:
                parts.append(f"<context>\n{t}\n</context>")
```

This means: when parent content is available, use it directly (it already includes the broader context). When not available (flat chunks or parent-child disabled), fall back to the existing adjacent expansion behavior.

**Step 4: Run tests**

Run: `pytest tests/rag/test_parent_child_retrieval.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add agents/executor_agent.py tests/rag/test_parent_child_retrieval.py
git commit -m "feat: executor uses parent content when available for richer context"
```

---

### Task 8: Handle NULL embeddings in repository batch upsert

**Files:**
- Modify: `db/repositories.py`
- Test: `tests/rag/test_parent_lookup.py`

**Step 1: Write the failing test**

Append to `tests/rag/test_parent_lookup.py`:

```python
def test_inmemory_upsert_batch_with_none_embedding():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    batch = [
        {"chunk_id": "parent-1", "doc_id": "doc-1", "content": "Parent", "embedding": None,
         "metadata": {"chunk_type": "parent"}},
        {"chunk_id": "child-1", "doc_id": "doc-1", "content": "Child", "embedding": [0.1] * 768,
         "metadata": {"chunk_type": "child"}},
    ]
    inserted = repo.upsert_chunks_batch(batch)
    assert inserted == 2
    assert repo.count_chunks() == 2

    # Parent should not appear in similarity search
    results = repo.similarity_search([0.1] * 768, k=5)
    chunk_ids = {d.metadata["chunk_id"] for d in results}
    assert "child-1" in chunk_ids
    # Parent may appear but with 0.0 similarity (sorted last) — that's OK


def test_inmemory_get_chunks_by_ids_fetches_parents():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    batch = [
        {"chunk_id": "parent-1", "doc_id": "doc-1", "content": "Parent content", "embedding": None,
         "metadata": {"chunk_type": "parent"}},
        {"chunk_id": "child-1", "doc_id": "doc-1", "content": "Child content", "embedding": [0.1] * 768,
         "metadata": {"chunk_type": "child", "parent_chunk_id": "parent-1"}},
    ]
    repo.upsert_chunks_batch(batch)

    parents = repo.get_chunks_by_ids(["parent-1"])
    assert len(parents) == 1
    assert parents[0].page_content == "Parent content"
```

**Step 2: Run to verify they pass (InMemory should handle None embeddings)**

Run: `pytest tests/rag/test_parent_lookup.py -v`
Expected: Check if tests pass or need fixes

**Step 3: Fix InMemoryVectorRepository if needed**

The `_cosine_distance` function already handles `None`/empty:
```python
def _cosine_distance(v1, v2):
    if not v1 or not v2:
        return 1.0
```

And `upsert_chunks_batch` stores whatever embedding is passed. So `None` embeddings should work. The similarity search would give 0.0 similarity for parents, effectively excluding them from top results.

The `VectorRepository.upsert_chunks_batch` may need a check:
```python
if use_vector:
    row.embedding = c['embedding']  # This sets NULL when embedding is None
```

This should work since the column is nullable. Verify with the test.

**Step 4: Run all tests**

Run: `pytest tests/ -v --timeout=30`
Expected: All PASS

**Step 5: Commit**

```bash
git add db/repositories.py tests/rag/test_parent_lookup.py
git commit -m "test: verify NULL embedding handling for parent chunks in repositories"
```

---

### Task 9: Update documentation

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update architecture doc**

Add under "Current important flows" or update the "Retrieval flow" section:

```markdown
### Parent-child chunk hierarchy (optional)
- When `ENABLE_PARENT_CHILD_CHUNKING=true`, the chunker creates two tiers:
  - **Child chunks** (~200 tokens): embedded and used for retrieval precision.
  - **Parent chunks** (~800–1000 tokens): stored with NULL embedding, never searched directly. Groups of 3–5 consecutive children within the same section.
- Tables are their own parent (self-contained retrieval units).
- After retrieval finds relevant children, `_resolve_parent_chunks()` fetches parent content from the DB via `get_chunks_by_ids()`.
- The executor uses parent content for context assembly, providing richer context than the small child chunk alone.
- When parent content is available, adjacent chunk expansion is skipped (parent already provides the broader context).
- Disabled by default. Requires reindex when toggling.
- Settings: `CHUNK_CHILD_TARGET=200`, `CHUNK_CHILD_MAX=250`, `CHUNK_PARENT_MAX_CHILDREN=5`.
```

**Step 2: Add implementation log entry**

```markdown
## 2026-03-16 — Parent-child chunk hierarchy

**What changed:**
- Added two-tier chunking: small child chunks (~200 tokens) for retrieval, large parent chunks (~800–1000 tokens) for executor context.
- `group_children_into_parents()` in pdf_chunker.py groups consecutive same-section children. Tables become their own parent.
- Parents stored in DB with NULL embedding (never searched). Children reference parents via `parent_chunk_id` in metadata.
- `get_chunks_by_ids()` added to both VectorRepository and InMemoryVectorRepository.
- `_resolve_parent_chunks()` in retriever_agent fetches parent content after retrieval.
- Executor uses `parent_content` metadata when available, skipping adjacent expansion.
- Feature gated by `ENABLE_PARENT_CHILD_CHUNKING=false` (default).

**Why:**
- Smaller chunks improve retrieval precision (more specific matches). Larger context improves generation quality (LLM sees full paragraph/section context). Parent-child gives both without compromise.
- Replaces the ±1 adjacent expansion approach with a cleaner, index-time solution.

**What must remain true:**
- When disabled (default), behavior is identical to pre-existing single-tier chunking.
- Parents MUST have NULL embedding (never searched).
- Toggling the feature requires reindex.
- `get_chunks_by_ids()` must exist on both repo implementations.
```

**Step 3: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: document parent-child chunk hierarchy architecture and implementation"
```

---

## Summary of changes

| Component | Change | Why |
|-----------|--------|-----|
| `core/settings.py` | 4 new settings | Feature flag + child/parent sizing |
| `tools/pdf_chunker.py` | `group_children_into_parents()`, `_config_from_settings_for_mode()` | Two-tier grouping logic |
| `tools/vector_store.py` | Skip embedding parents, handle None embeddings in batch | Only children need vectors |
| `db/repositories.py` | `get_chunks_by_ids()` on both repos | Parent content lookup |
| `agents/retriever_agent.py` | `_resolve_parent_chunks()`, wire into retrieval | Attach parent content post-retrieval |
| `agents/executor_agent.py` | Use `parent_content` in context assembly | Richer LLM context |
| Tests | 4 new test files, ~15 tests total | Full coverage of hierarchy |
| Docs | Architecture + implementation log | Track the change |

## Activation checklist (for user after implementation)

1. Set `ENABLE_PARENT_CHILD_CHUNKING=true` in `.env`
2. Optionally tune: `CHUNK_CHILD_TARGET`, `CHUNK_CHILD_MAX`, `CHUNK_PARENT_MAX_CHILDREN`
3. Reindex: `docker-compose run --rm ingest python scripts/reindex_pdf.py`
4. Compare retrieval eval results with/without the feature
