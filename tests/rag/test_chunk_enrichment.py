import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_parse_enrichment_response_extracts_summary():
    from tools.chunk_enrichment import _parse_enrichment_response

    raw = '{"context_summary": "This section covers pediatric acetaminophen dosing."}'
    result = _parse_enrichment_response(raw)
    assert result["context_summary"] == "This section covers pediatric acetaminophen dosing."


def test_parse_enrichment_response_handles_malformed_json():
    from tools.chunk_enrichment import _parse_enrichment_response

    result = _parse_enrichment_response("not json at all")
    assert result["context_summary"] == ""


def test_parse_enrichment_response_extracts_json_from_markdown():
    from tools.chunk_enrichment import _parse_enrichment_response

    raw = 'Here is the result:\n```json\n{"context_summary": "Summary."}\n```'
    result = _parse_enrichment_response(raw)
    assert result["context_summary"] == "Summary."


def test_build_enrichment_prompt_includes_section_and_content():
    from tools.chunk_enrichment import _build_enrichment_prompt

    prompt = _build_enrichment_prompt(
        content="Aspirin is used for pain relief.",
        section_path="Chapter 5 > Pain Management",
        content_type="text_section",
    )
    assert "Aspirin" in prompt
    assert "Chapter 5" in prompt
    assert "text_section" in prompt


def test_apply_enrichment_to_chunks():
    from langchain_core.documents import Document
    from tools.vector_store import _apply_enrichment_to_chunks

    chunks = [
        Document(page_content="Some medical content about aspirin.", metadata={
            "section_path": "Chapter 1",
            "content_type": "text_section",
            "embedding_prefix": "Chapter 1",
        }),
    ]
    enrichments = [
        {
            "context_summary": "Discusses aspirin usage in cardiology.",
        },
    ]
    _apply_enrichment_to_chunks(chunks, enrichments)

    meta = chunks[0].metadata
    assert meta["context_summary"] == "Discusses aspirin usage in cardiology."
    assert "Discusses aspirin usage in cardiology" in meta.get("embedding_text", "")
