import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_doc(chunk_id: str, content: str):
    from langchain_core.documents import Document
    return Document(page_content=content, metadata={"chunk_id": chunk_id})


def test_merge_and_deduplicate_removes_duplicates():
    from agents.retriever_agent import _merge_and_deduplicate

    primary = [_make_doc("c1", "doc one"), _make_doc("c2", "doc two")]
    stepback = [_make_doc("c2", "doc two duplicate"), _make_doc("c3", "doc three")]

    merged = _merge_and_deduplicate(primary, stepback)
    chunk_ids = [d.metadata["chunk_id"] for d in merged]
    assert chunk_ids == ["c1", "c2", "c3"]
    assert merged[1].page_content == "doc two"


def test_merge_and_deduplicate_preserves_order_primary_first():
    from agents.retriever_agent import _merge_and_deduplicate

    primary = [_make_doc("c1", "first"), _make_doc("c2", "second")]
    stepback = [_make_doc("c3", "third"), _make_doc("c4", "fourth")]

    merged = _merge_and_deduplicate(primary, stepback)
    chunk_ids = [d.metadata["chunk_id"] for d in merged]
    assert chunk_ids == ["c1", "c2", "c3", "c4"]


def test_merge_and_deduplicate_handles_empty_stepback():
    from agents.retriever_agent import _merge_and_deduplicate

    primary = [_make_doc("c1", "only")]
    merged = _merge_and_deduplicate(primary, [])
    assert len(merged) == 1
    assert merged[0].metadata["chunk_id"] == "c1"
