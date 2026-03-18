import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document


def _make_doc(chunk_id, content, section="Ch1", chunk_index=0):
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "doc_id": "doc1",
            "page": 1,
            "section": section,
            "chunk_index": chunk_index,
        },
    )


def test_inmemory_get_adjacent_chunks():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    for i in range(5):
        repo.upsert_chunk(
            chunk_id=f"c{i}",
            doc_id="doc1",
            content=f"chunk {i}",
            embedding=[0.0] * 3,
            section="Ch1",
            metadata={"chunk_index": i, "section": "Ch1"},
        )

    neighbors = repo.get_adjacent_chunks(section="Ch1", chunk_index=2)
    neighbor_ids = [d.metadata['chunk_id'] for d in neighbors]
    assert "c1" in neighbor_ids
    assert "c3" in neighbor_ids
    assert "c2" not in neighbor_ids


def test_inmemory_get_adjacent_chunks_at_boundary():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        chunk_id="c0", doc_id="doc1", content="first",
        embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": 0, "section": "Ch1"},
    )
    repo.upsert_chunk(
        chunk_id="c1", doc_id="doc1", content="second",
        embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": 1, "section": "Ch1"},
    )

    neighbors = repo.get_adjacent_chunks(section="Ch1", chunk_index=0)
    assert len(neighbors) == 1
    assert neighbors[0].metadata['chunk_id'] == 'c1'


def test_inmemory_get_adjacent_chunks_different_section_excluded():
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        chunk_id="c0", doc_id="doc1", content="ch1 chunk",
        embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": 0, "section": "Ch1"},
    )
    repo.upsert_chunk(
        chunk_id="c1", doc_id="doc1", content="ch2 chunk",
        embedding=[0.0] * 3, section="Ch2", metadata={"chunk_index": 1, "section": "Ch2"},
    )

    neighbors = repo.get_adjacent_chunks(section="Ch1", chunk_index=0)
    assert len(neighbors) == 0


def test_build_context_blocks_with_expansion():
    from agents.executor_agent import _build_context_blocks
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    for i in range(5):
        repo.upsert_chunk(
            chunk_id=f"c{i}", doc_id="doc1", content=f"neighbor content {i}",
            embedding=[0.0] * 3, section="Ch1", metadata={"chunk_index": i, "section": "Ch1"},
        )

    docs = [_make_doc("c2", "main content", section="Ch1", chunk_index=2)]

    blocks = _build_context_blocks(docs, vector_repo=repo)
    combined = '\n'.join(blocks)
    assert "neighbor content 1" in combined
    assert "main content" in combined
    assert "neighbor content 3" in combined


def test_build_context_blocks_without_repo_no_expansion():
    from agents.executor_agent import _build_context_blocks

    docs = [_make_doc("c2", "main content", section="Ch1", chunk_index=2)]
    blocks = _build_context_blocks(docs, vector_repo=None)
    combined = '\n'.join(blocks)
    assert "main content" in combined
    assert "neighbor" not in combined
