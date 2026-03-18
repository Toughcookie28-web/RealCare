# MMR Diversity Enforcement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After Cohere/BM25 reranking, apply Maximal Marginal Relevance (MMR) post-processing to ensure the top-5 retrieved chunks span multiple sections, preventing the executor from seeing 5 nearly-identical chunks from the same section.

**Architecture:** New `_mmr_filter()` function inserted in `RetrieverAgent` after `rerank()`. Uses query embedding (already computed) and chunk embeddings fetched from the vector repository. Falls back to section-quota deduplication when embeddings are unavailable. Controlled by `DIVERSITY_MMR_LAMBDA` (default 0.7) and `DIVERSITY_MMR_ENABLED` (default True) settings.

**Tech Stack:** Python, numpy (cosine similarity), existing `VectorRepository`

---

### Task 1: Add diversity settings to core/settings.py

**Files:**
- Modify: `core/settings.py`

**Step 1: Write the failing test**

In `tests/core/test_settings_contract.py`, add:
```python
def test_diversity_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.diversity_mmr_enabled is True
    assert s.diversity_mmr_lambda == 0.7
    assert s.diversity_max_per_section == 2
```

**Step 2: Run to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v -k "diversity" 2>&1`
Expected: FAIL — attribute not found

**Step 3: Add settings fields**

In `core/settings.py`, after `hybrid_dense_weight` line, add:
```python
diversity_mmr_enabled: bool = Field(default=True, alias='DIVERSITY_MMR_ENABLED')
diversity_mmr_lambda: float = Field(default=0.7, alias='DIVERSITY_MMR_LAMBDA')
diversity_max_per_section: int = Field(default=2, alias='DIVERSITY_MAX_PER_SECTION')
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_settings_contract.py -v -k "diversity" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings_contract.py
git commit -m "feat: add diversity MMR settings"
```

---

### Task 2: Add get_embeddings_for_chunk_ids to VectorRepository

**Files:**
- Modify: `db/repositories.py`

The retriever needs chunk embeddings to compute cosine similarity for MMR. The existing `get_chunks_by_ids` returns Documents without the raw embedding vector. We need a separate method that returns `{chunk_id: np.ndarray}`.

**Step 1: Write the failing test**

In `tests/rag/test_diversity.py` (create new file):
```python
import numpy as np
from db.repositories import InMemoryVectorRepository
from langchain_core.documents import Document


def _make_doc(chunk_id, embedding):
    doc = Document(page_content="test", metadata={"chunk_id": chunk_id})
    doc.metadata["embedding"] = embedding
    return doc, embedding


def test_inmemory_get_embeddings_for_chunk_ids():
    repo = InMemoryVectorRepository()
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    doc1 = Document(page_content="a", metadata={"chunk_id": "c1"})
    doc2 = Document(page_content="b", metadata={"chunk_id": "c2"})
    repo.upsert_batch([doc1, doc2], [emb1, emb2])

    result = repo.get_embeddings_for_chunk_ids(["c1", "c2"])
    assert "c1" in result
    assert "c2" in result
    np.testing.assert_array_almost_equal(result["c1"], emb1)


def test_inmemory_get_embeddings_returns_empty_for_missing():
    repo = InMemoryVectorRepository()
    result = repo.get_embeddings_for_chunk_ids(["nonexistent"])
    assert result == {}
```

**Step 2: Run to verify they fail**

Run: `pytest tests/rag/test_diversity.py -v 2>&1`
Expected: FAIL — method not found

**Step 3: Add method to InMemoryVectorRepository**

In `db/repositories.py`, inside `InMemoryVectorRepository`, add:
```python
def get_embeddings_for_chunk_ids(self, chunk_ids: list[str]) -> dict[str, Any]:
    """Return {chunk_id: embedding_array} for the requested chunk_ids."""
    result = {}
    for cid in chunk_ids:
        row = self._store.get(cid)
        if row is not None and row.get("embedding") is not None:
            result[cid] = row["embedding"]
    return result
```

**Step 4: Add method to VectorRepository**

In `db/repositories.py`, inside `VectorRepository`, add:
```python
def get_embeddings_for_chunk_ids(self, chunk_ids: list[str]) -> dict[str, Any]:
    """Return {chunk_id: raw_embedding} for the requested chunk_ids."""
    if not chunk_ids:
        return {}
    import numpy as np
    with self._session() as session:
        rows = session.query(
            DocumentChunkModel.chunk_id, DocumentChunkModel.embedding
        ).filter(DocumentChunkModel.chunk_id.in_(chunk_ids)).all()
        return {row.chunk_id: np.array(row.embedding) for row in rows if row.embedding is not None}
```

**Step 5: Run tests**

Run: `pytest tests/rag/test_diversity.py -v 2>&1`
Expected: PASS

**Step 6: Commit**

```bash
git add db/repositories.py tests/rag/test_diversity.py
git commit -m "feat: add get_embeddings_for_chunk_ids to vector repositories"
```

---

### Task 3: Implement _mmr_filter() in retriever_agent.py

**Files:**
- Modify: `agents/retriever_agent.py`
- Modify: `tests/rag/test_diversity.py`

**Step 1: Write failing tests**

Append to `tests/rag/test_diversity.py`:
```python
import numpy as np
from langchain_core.documents import Document


def _doc_with_emb(chunk_id, section, embedding):
    doc = Document(page_content=f"content {chunk_id}", metadata={
        "chunk_id": chunk_id,
        "section": section,
    })
    return doc, np.array(embedding, dtype=float)


def test_mmr_filter_selects_diverse_docs():
    from agents.retriever_agent import _mmr_filter
    # 3 docs from section A (similar), 2 docs from section B
    docs_embs = [
        _doc_with_emb("a1", "SectionA", [1.0, 0.0]),
        _doc_with_emb("a2", "SectionA", [0.99, 0.1]),
        _doc_with_emb("a3", "SectionA", [0.98, 0.2]),
        _doc_with_emb("b1", "SectionB", [0.0, 1.0]),
        _doc_with_emb("b2", "SectionB", [0.1, 0.99]),
    ]
    docs = [d for d, _ in docs_embs]
    emb_map = {d.metadata["chunk_id"]: e for d, e in docs_embs}
    query_emb = np.array([0.7, 0.7])

    result = _mmr_filter(docs, emb_map, query_emb, k=3, lambda_=0.7)

    assert len(result) == 3
    sections = [d.metadata["section"] for d in result]
    # Should not pick all 3 from SectionA
    assert sections.count("SectionA") <= 2


def test_mmr_filter_handles_missing_embeddings():
    from agents.retriever_agent import _mmr_filter
    docs = [
        Document(page_content="a", metadata={"chunk_id": "x1", "section": "S1"}),
        Document(page_content="b", metadata={"chunk_id": "x2", "section": "S2"}),
    ]
    # No embeddings available — fallback section quota
    result = _mmr_filter(docs, {}, np.array([1.0, 0.0]), k=2, lambda_=0.7)
    assert len(result) == 2


def test_mmr_filter_returns_k_or_fewer():
    from agents.retriever_agent import _mmr_filter
    docs = [Document(page_content=f"d{i}", metadata={"chunk_id": f"c{i}"}) for i in range(3)]
    emb_map = {f"c{i}": np.array([float(i), 0.0]) for i in range(3)}
    result = _mmr_filter(docs, emb_map, np.array([1.0, 0.0]), k=2, lambda_=0.7)
    assert len(result) <= 2
```

**Step 2: Run to verify they fail**

Run: `pytest tests/rag/test_diversity.py -v -k "mmr" 2>&1`
Expected: FAIL

**Step 3: Implement _mmr_filter in agents/retriever_agent.py**

Add after the existing imports and before `RetrieverAgent`:
```python
import numpy as np


def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mmr_filter(
    docs: list[Document],
    emb_map: dict[str, "np.ndarray"],
    query_emb: "np.ndarray",
    k: int = 5,
    lambda_: float = 0.7,
) -> list[Document]:
    """
    Maximal Marginal Relevance filter.
    Selects k docs that balance relevance to query and diversity from each other.
    Falls back to section-based quota when embeddings are missing.
    """
    if not docs:
        return docs

    # Check if we have embeddings for at least half the docs
    have_embs = [d for d in docs if d.metadata.get("chunk_id") in emb_map]
    if len(have_embs) < len(docs) // 2:
        # Fallback: section quota (max 2 per section)
        return _section_quota_filter(docs, k)

    selected: list[Document] = []
    remaining = list(docs)

    while remaining and len(selected) < k:
        if not selected:
            # First pick: most relevant to query
            best = max(
                remaining,
                key=lambda d: _cosine_sim(
                    emb_map.get(d.metadata.get("chunk_id", ""), query_emb),
                    query_emb,
                ),
            )
        else:
            selected_embs = [
                emb_map[d.metadata["chunk_id"]]
                for d in selected
                if d.metadata.get("chunk_id") in emb_map
            ]
            best = max(
                remaining,
                key=lambda d: (
                    lambda_ * _cosine_sim(
                        emb_map.get(d.metadata.get("chunk_id", ""), query_emb),
                        query_emb,
                    )
                    - (1 - lambda_) * (
                        max(_cosine_sim(
                            emb_map.get(d.metadata.get("chunk_id", ""), query_emb),
                            s,
                        ) for s in selected_embs)
                        if selected_embs else 0.0
                    )
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


def _section_quota_filter(docs: list[Document], k: int, max_per_section: int = 2) -> list[Document]:
    """Fallback: cap how many chunks from the same section make it to top-k."""
    seen: dict[str, int] = {}
    result = []
    for doc in docs:
        section = doc.metadata.get("section", "_unknown")
        count = seen.get(section, 0)
        if count < max_per_section:
            result.append(doc)
            seen[section] = count + 1
        if len(result) == k:
            break
    return result
```

**Step 4: Wire into RetrieverAgent**

In `RetrieverAgent`, after `ranked, scores = rerank(query, docs)` and before `top_docs = ranked[:5]`, add:

```python
settings = get_settings()
top_k = 5
if settings.diversity_mmr_enabled:
    chunk_ids = [d.metadata.get("chunk_id", "") for d in ranked[:20]]
    emb_map = repo.get_embeddings_for_chunk_ids(chunk_ids) if repo is not None else {}
    top_docs = _mmr_filter(ranked, emb_map, query_embedding, k=top_k, lambda_=settings.diversity_mmr_lambda)
else:
    top_docs = ranked[:top_k]
```

Remove the old line `top_docs = ranked[:5]`.

**Step 5: Run all diversity tests**

Run: `pytest tests/rag/test_diversity.py -v 2>&1`
Expected: all PASS

**Step 6: Run full affected tests**

Run: `pytest tests/rag/ tests/core/test_settings_contract.py -v 2>&1`
Expected: all PASS

**Step 7: Commit**

```bash
git add agents/retriever_agent.py tests/rag/test_diversity.py
git commit -m "feat: add MMR diversity enforcement in retriever post-reranking"
```

---

### Task 4: Update docs

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

Add under retrieval section: "Post-rerank MMR diversity filter enforces varied section coverage in top-5 docs. `DIVERSITY_MMR_ENABLED=true` (default), `DIVERSITY_MMR_LAMBDA=0.7`."

Log entry: `2026-03-17 — MMR diversity enforcement added to RetrieverAgent.`
