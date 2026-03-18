"""
LLM-based chunk enrichment for RAG metadata.

Generates context summaries for document chunks at index time.
Controlled by ENRICH_CHUNKS_WITH_LLM setting.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_ENRICHMENT_SYSTEM = (
    "You are a medical document indexing assistant. "
    "Given a chunk of text from a medical textbook, extract structured metadata. "
    "Respond ONLY with valid JSON, no other text."
)

def _build_enrichment_prompt(
    content: str,
    section_path: str,
    content_type: str,
) -> str:
    return (
        f"Section: {section_path}\n"
        f"Content type: {content_type}\n"
        f"Chunk text:\n{content[:2000]}\n\n"
        "Return JSON with this field:\n"
        '- "context_summary": 1-3 sentences describing what this chunk covers. '
        "Use 1 sentence for simple factual content, 2-3 for complex or multi-topic content. "
        "Include key medical terms.\n"
    )


def _parse_enrichment_response(raw: str) -> dict[str, Any]:
    """Parse LLM enrichment response, with fallback for malformed JSON."""
    defaults: dict[str, Any] = {
        "context_summary": "",
    }
    if not raw or not raw.strip():
        return defaults

    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return defaults
        else:
            return defaults

    return {
        "context_summary": str(parsed.get("context_summary", "")).strip(),
    }


def enrich_single_chunk(
    content: str,
    section_path: str,
    content_type: str,
) -> dict[str, Any]:
    """Call the LLM to generate enrichment metadata for a single chunk."""
    from tools.llm_client import invoke_llm

    prompt = _build_enrichment_prompt(content, section_path, content_type)
    raw = invoke_llm(prompt, system=_ENRICHMENT_SYSTEM)
    return _parse_enrichment_response(raw)


def enrich_chunks_batch(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich a batch of chunks with LLM-generated metadata.

    Each item should have keys: content, section_path, content_type.
    Returns enrichment dicts in the same order.
    """
    results = []
    for i, chunk in enumerate(chunks):
        try:
            result = enrich_single_chunk(
                content=chunk["content"],
                section_path=chunk.get("section_path", ""),
                content_type=chunk.get("content_type", "text_section"),
            )
            logger.info("chunk_enriched", extra={"index": i, "has_summary": bool(result["context_summary"])})
        except Exception as exc:
            logger.warning("chunk_enrichment_failed index=%d: %s", i, exc)
            result = _parse_enrichment_response("")
        results.append(result)
    return results
