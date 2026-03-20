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
