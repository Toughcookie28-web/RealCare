import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document


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
    assert "Full parent content" in blocks[0]
    assert "Short child chunk" not in blocks[0]


def test_build_context_blocks_falls_back_to_page_content():
    from agents.executor_agent import _build_context_blocks

    docs = [
        Document(page_content="Normal flat chunk content.", metadata={
            "chunk_id": "flat-1",
            "section_path": "Chapter 1",
            "context_key": "doc-sec01",
        }),
    ]

    blocks = _build_context_blocks(docs)
    assert "Normal flat chunk content" in blocks[0]


def test_resolve_parents_attaches_parent_content():
    from agents.retriever_agent import _resolve_parent_chunks
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        "doc-parent-0000", "test-doc",
        "Full parent content about aspirin dosing and side effects.",
        [0.0] * 768,
        metadata={"chunk_type": "parent"},
    )

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
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    repo.upsert_chunk(
        "doc-parent-0000", "test-doc", "Parent content.",
        [0.0] * 768, metadata={"chunk_type": "parent"},
    )

    children = [
        Document(page_content="Child 1.", metadata={
            "chunk_id": "child-1", "parent_chunk_id": "doc-parent-0000", "chunk_type": "child",
        }),
        Document(page_content="Child 2.", metadata={
            "chunk_id": "child-2", "parent_chunk_id": "doc-parent-0000", "chunk_type": "child",
        }),
    ]

    resolved = _resolve_parent_chunks(children, repo)
    assert all(d.metadata.get("parent_content") == "Parent content." for d in resolved)


def test_resolve_parents_skips_non_child_chunks():
    from agents.retriever_agent import _resolve_parent_chunks
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()
    flat = [
        Document(page_content="Flat chunk.", metadata={
            "chunk_id": "flat-1",
        }),
    ]

    resolved = _resolve_parent_chunks(flat, repo)
    assert len(resolved) == 1
    assert "parent_content" not in resolved[0].metadata
