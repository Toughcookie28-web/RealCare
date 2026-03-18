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

    assert len(parents) == 2
    assert all(p.metadata["chunk_type"] == "parent" for p in parents)
    assert all(c.metadata.get("parent_chunk_id") for c in updated_children)
    assert all(c.metadata["chunk_type"] == "child" for c in updated_children)
    assert "Child 0" in parents[0].page_content
    assert "Child 2" in parents[0].page_content


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

    table_child = updated_children[1]
    text_child = updated_children[0]
    assert table_child.metadata["parent_chunk_id"] != text_child.metadata["parent_chunk_id"]


def test_group_children_empty_input():
    from tools.pdf_chunker import group_children_into_parents

    parents, children = group_children_into_parents([], max_children_per_parent=5)
    assert parents == []
    assert children == []


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

    assert len(embeddings) == 2
    assert embeddings[0] is None  # parent
    assert embeddings[1] == [0.1] * 768  # child
