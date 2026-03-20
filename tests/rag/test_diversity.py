import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from langchain_core.documents import Document


def test_inmemory_get_embeddings_for_chunk_ids():
    from db.repositories import InMemoryVectorRepository
    repo = InMemoryVectorRepository()
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    doc1 = Document(page_content="a", metadata={"chunk_id": "c1"})
    doc2 = Document(page_content="b", metadata={"chunk_id": "c2"})
    repo.upsert_chunk("c1", "doc1", "a", list(emb1))
    repo.upsert_chunk("c2", "doc1", "b", list(emb2))

    result = repo.get_embeddings_for_chunk_ids(["c1", "c2"])
    assert "c1" in result
    assert "c2" in result
    np.testing.assert_array_almost_equal(result["c1"], emb1)


def test_inmemory_get_embeddings_returns_empty_for_missing():
    from db.repositories import InMemoryVectorRepository
    repo = InMemoryVectorRepository()
    result = repo.get_embeddings_for_chunk_ids(["nonexistent"])
    assert result == {}


def _doc_with_emb(chunk_id, section, embedding):
    doc = Document(page_content=f"content {chunk_id}", metadata={
        "chunk_id": chunk_id,
        "section": section,
    })
    return doc, np.array(embedding, dtype=float)


def test_mmr_filter_selects_diverse_docs():
    from agents.retriever_agent import _mmr_filter
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
    assert sections.count("SectionA") <= 2


def test_mmr_filter_handles_missing_embeddings():
    from agents.retriever_agent import _mmr_filter
    docs = [
        Document(page_content="a", metadata={"chunk_id": "x1", "section": "S1"}),
        Document(page_content="b", metadata={"chunk_id": "x2", "section": "S2"}),
    ]
    result = _mmr_filter(docs, {}, np.array([1.0, 0.0]), k=2, lambda_=0.7)
    assert len(result) == 2


def test_mmr_filter_returns_k_or_fewer():
    from agents.retriever_agent import _mmr_filter
    docs = [Document(page_content=f"d{i}", metadata={"chunk_id": f"c{i}"}) for i in range(3)]
    emb_map = {f"c{i}": np.array([float(i), 0.0]) for i in range(3)}
    result = _mmr_filter(docs, emb_map, np.array([1.0, 0.0]), k=2, lambda_=0.7)
    assert len(result) <= 2
