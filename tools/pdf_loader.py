from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tools.pdf_parser import ParsedDocument


def _resolve_pdf_path(pdf_path: str) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return path


def load_parsed_pdf(pdf_path: str) -> "ParsedDocument":
    path = _resolve_pdf_path(pdf_path)
    from tools.pdf_parser import parse_pdf_to_elements

    parsed_doc = parse_pdf_to_elements(str(path))
    logger.info(
        "pdf_parse_complete",
        extra={
            "pdf": path.name,
            "doc_id": parsed_doc.doc_id,
            "elements": len(parsed_doc.elements),
        },
    )
    return parsed_doc


def chunk_pdf_document(parsed_doc: "ParsedDocument") -> list[Document]:
    from tools.pdf_chunker import chunk_parsed_document

    chunks = chunk_parsed_document(parsed_doc)
    logger.info(
        "pdf_chunk_complete",
        extra={
            "doc_id": parsed_doc.doc_id,
            "elements": len(parsed_doc.elements),
            "chunks": len(chunks),
        },
    )
    return chunks


def process_pdf(pdf_path: str) -> list[Document]:
    """
    Parse and chunk a PDF using the structure-aware Docling pipeline.

    This loader is intentionally fail-fast. The project no longer keeps a
    legacy PyPDF/recursive-splitter fallback because fallback behavior hides
    parser regressions and makes chunking comparisons ambiguous.
    """
    path = _resolve_pdf_path(pdf_path)
    parsed_doc = load_parsed_pdf(str(path))
    chunks = chunk_pdf_document(parsed_doc)

    logger.info(
        "docling_pipeline_complete",
        extra={
            "pdf": path.name,
            "elements": len(parsed_doc.elements),
            "chunks": len(chunks),
        },
    )
    return chunks
