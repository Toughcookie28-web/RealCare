"""
Structure-aware PDF parser using Docling.

Converts a PDF into typed elements (sections, paragraphs, tables, captions)
with document structure metadata. This is the parsing layer — chunking is
handled separately in pdf_chunker.py.

This module is an ingest-only dependency. Install via: pip install ".[ingest]"
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.settings import get_settings
from tools.embedding_bootstrap import (
    configure_embedding_cache_env,
    ensure_docling_artifacts,
)

logger = logging.getLogger(__name__)

# Sections whose content is navigational/non-informational and should be
# excluded from the retrieval corpus. Matching is fuzzy and case-insensitive.
_EXCLUDED_SECTION_EXACT = {
    'references',
    'bibliography',
    'table of contents',
    'contents',
    'index',
    'glossary',
    'glossary of terms',
    'abbreviations',
    'acknowledgements',
    'acknowledgments',
    'appendix',
    'list of figures',
    'list of tables',
    'works cited',
    'literature cited',
}

_EXCLUDED_SECTION_PREFIXES = (
    'references',
    'further reading',
    'bibliography',
    'table of contents',
    'glossary',
    'abbreviations',
    'acknowledgements',
    'acknowledgments',
    'appendix',
    'list of figures',
    'list of tables',
)

# Short "headings" that are really inline labels, not structural hierarchy.
# Any heading whose stripped text matches one of these (case-insensitive) is
# demoted to a paragraph to prevent typographical hierarchy collapse.
_FAKE_HEADING_PREFIXES = (
    'warning',
    'note',
    'caution',
    'important',
    'tip',
    'example',
    'dosage note',
)


class ElementType(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    FIGURE = "figure"


@dataclass
class ParsedElement:
    """A single typed element extracted from a document."""
    element_type: ElementType
    text: str
    page: int | None = None
    heading_level: int | None = None
    section_path: list[str] = field(default_factory=list)
    table_markdown: str | None = None
    table_html: str | None = None
    caption_text: str | None = None
    element_index: int = 0


@dataclass
class ParsedDocument:
    """A document parsed into typed elements with structure metadata."""
    doc_id: str
    title: str | None = None
    elements: list[ParsedElement] = field(default_factory=list)
    page_count: int = 0


def parse_pdf_to_elements(pdf_path: str) -> ParsedDocument:
    """
    Parse a PDF into structured elements using Docling.

    Returns a ParsedDocument with typed elements preserving document structure.
    This parser is intentionally fail-fast; there is no legacy PDF fallback.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Docling and RapidOCR lazily download model artifacts at runtime; point them
    # at the shared writable cache root before importing the pipeline stack.
    configure_embedding_cache_env(get_settings().embedding_cache_dir)

    return _parse_with_docling(str(path))


def _parse_with_docling(pdf_path: str) -> ParsedDocument:
    """Parse using Docling's DocumentConverter for structure-aware extraction."""
    artifacts_path = ensure_docling_artifacts(get_settings().embedding_cache_dir)

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = artifacts_path
    # This corpus is a born-digital medical textbook, so prefer embedded PDF text
    # and avoid OCR unless Docling determines it is still needed internally.
    pipeline_options.do_ocr = False
    pipeline_options.force_backend_text = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(pdf_path)
    dl_doc = result.document

    doc_id = Path(pdf_path).stem
    title = None
    elements: list[ParsedElement] = []
    section_stack: list[str] = []
    in_excluded_section = False
    elem_idx = 0

    for item, _level in dl_doc.iterate_items():
        item_type = item.label if hasattr(item, 'label') else str(type(item).__name__)
        item_type_lower = str(item_type).lower()
        page = None
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            page = getattr(prov, 'page_no', None) or getattr(prov, 'page', None)

        text = _extract_item_text(item, item_type_lower)

        # Handle headings / section headers
        if 'heading' in item_type_lower or 'title' in item_type_lower or 'section' in item_type_lower:
            heading_text = text.strip()

            # Demote fake headings (inline labels) to paragraphs
            if _looks_like_fake_heading(heading_text):
                if heading_text:
                    elements.append(ParsedElement(
                        element_type=ElementType.PARAGRAPH,
                        text=heading_text,
                        page=page,
                        section_path=list(section_stack),
                        element_index=elem_idx,
                    ))
                    elem_idx += 1
                continue

            level = _level if isinstance(_level, int) else 1
            # Update section stack
            while len(section_stack) >= level:
                section_stack.pop()
            section_stack.append(heading_text)

            # Check if this section is excluded (navigational content)
            if _is_excluded_heading(heading_text):
                in_excluded_section = True
                continue
            else:
                in_excluded_section = False

            if title is None and level <= 1:
                title = heading_text

            elements.append(ParsedElement(
                element_type=ElementType.HEADING,
                text=heading_text,
                page=page,
                heading_level=level,
                section_path=list(section_stack),
                element_index=elem_idx,
            ))
            elem_idx += 1
            continue

        # Skip content inside excluded sections
        if in_excluded_section:
            continue

        # Handle tables
        if 'table' in item_type_lower:
            table_md = ""
            table_html = ""
            if hasattr(item, 'export_to_markdown'):
                try:
                    table_md = item.export_to_markdown()
                except Exception:
                    table_md = text
            if hasattr(item, 'export_to_html'):
                try:
                    table_html = item.export_to_html()
                except Exception:
                    pass

            # Look for caption from preceding element
            caption = None
            if elements and elements[-1].element_type == ElementType.CAPTION:
                caption = elements[-1].text

            elements.append(ParsedElement(
                element_type=ElementType.TABLE,
                text=table_md or text,
                page=page,
                section_path=list(section_stack),
                table_markdown=table_md,
                table_html=table_html,
                caption_text=caption,
                element_index=elem_idx,
            ))
            elem_idx += 1
            continue

        # Handle captions
        if 'caption' in item_type_lower:
            elements.append(ParsedElement(
                element_type=ElementType.CAPTION,
                text=text.strip(),
                page=page,
                section_path=list(section_stack),
                caption_text=text.strip(),
                element_index=elem_idx,
            ))
            elem_idx += 1
            continue

        # Handle list items
        if 'list' in item_type_lower:
            elements.append(ParsedElement(
                element_type=ElementType.LIST_ITEM,
                text=text.strip(),
                page=page,
                section_path=list(section_stack),
                element_index=elem_idx,
            ))
            elem_idx += 1
            continue

        # Handle figures (metadata only, no image embedding)
        if 'figure' in item_type_lower or 'picture' in item_type_lower:
            elements.append(ParsedElement(
                element_type=ElementType.FIGURE,
                text=text.strip() if text.strip() else "[Figure]",
                page=page,
                section_path=list(section_stack),
                element_index=elem_idx,
            ))
            elem_idx += 1
            continue

        # Handle page headers/footers
        if 'page_header' in item_type_lower or 'header' in item_type_lower:
            continue  # skip page headers
        if 'page_footer' in item_type_lower or 'footer' in item_type_lower:
            continue  # skip page footers

        # Default: treat as paragraph
        if text.strip():
            elements.append(ParsedElement(
                element_type=ElementType.PARAGRAPH,
                text=text.strip(),
                page=page,
                section_path=list(section_stack),
                element_index=elem_idx,
            ))
            elem_idx += 1

    page_count = 0
    if hasattr(dl_doc, 'pages') and dl_doc.pages:
        page_count = len(dl_doc.pages)

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        elements=elements,
        page_count=page_count,
    )


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _is_excluded_heading(text: str) -> bool:
    normalized = _normalize_heading(text)
    if normalized in _EXCLUDED_SECTION_EXACT:
        return True
    return any(
        normalized.startswith(f"{pattern} ")
        or normalized.startswith(f"{pattern}:")
        for pattern in _EXCLUDED_SECTION_PREFIXES
    )


def _looks_like_fake_heading(text: str) -> bool:
    stripped = " ".join(text.split())
    if not stripped:
        return False

    normalized = stripped.lower()
    if len(stripped) > 150:
        return True
    if stripped.endswith((".", ";")):
        return True

    prefix = normalized.split(":", 1)[0].strip()
    if prefix in _FAKE_HEADING_PREFIXES and ":" in normalized[:40]:
        return True

    if any(normalized.startswith(f"{candidate}:") for candidate in _FAKE_HEADING_PREFIXES):
        return True

    return False


def _extract_item_text(item, item_type_lower: str) -> str:
    """Extract item text without triggering incompatible picture exports."""
    if hasattr(item, 'text'):
        return item.text or ""

    if 'figure' in item_type_lower or 'picture' in item_type_lower:
        return ""

    export_to_markdown = getattr(item, 'export_to_markdown', None)
    if callable(export_to_markdown):
        try:
            return export_to_markdown()
        except TypeError:
            return ""

    return ""
