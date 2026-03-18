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

    # Parent should not appear in top results (similarity = 0.0)
    results = repo.similarity_search([0.1] * 768, k=1)
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "child-1"


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
