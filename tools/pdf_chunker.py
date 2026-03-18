"""
Structure-aware chunker for MediGenius RAG pipeline.

Takes a ParsedDocument from pdf_parser.py and produces LangChain Documents
with rich metadata suitable for embedding and retrieval.

Design principles:
- Respect section/subsection boundaries
- Treat tables as first-class retrieval units
- Generate deterministic context prefixes
- Produce stable, debuggable chunk IDs
- Token-aware sizing aligned to embedding model
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from tools.pdf_parser import ElementType, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)

try:
    import tiktoken as _tiktoken

    _enc = _tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))

except Exception:

    def _count_tokens(text: str) -> int:
        """Approximate token count when tiktoken is unavailable."""
        return len(text.split())

# Chunk sizing defaults (tokens, approximate via whitespace split)
DEFAULT_TEXT_CHUNK_TARGET = 400
DEFAULT_TEXT_CHUNK_MAX = 500
DEFAULT_TEXT_CHUNK_OVERLAP = 50
DEFAULT_TABLE_CHUNK_MAX = 450
TABLE_HEADER_REPEAT_BUDGET = 80  # tokens reserved for repeated header rows
TABLE_LARGE_TOKEN_THRESHOLD = 2000  # boundary between medium and huge table tiers


@dataclass
class ChunkConfig:
    """Configuration for the structure-aware chunker."""
    text_chunk_target: int = DEFAULT_TEXT_CHUNK_TARGET
    text_chunk_max: int = DEFAULT_TEXT_CHUNK_MAX
    text_chunk_overlap: int = DEFAULT_TEXT_CHUNK_OVERLAP
    table_chunk_max: int = DEFAULT_TABLE_CHUNK_MAX


def _config_from_settings() -> ChunkConfig:
    """Build a ChunkConfig from centralised Settings (env-configurable)."""
    return _config_from_settings_for_mode()


def chunk_parsed_document(
    parsed_doc: ParsedDocument,
    config: ChunkConfig | None = None,
) -> list[Document]:
    """
    Chunk a ParsedDocument into retrieval-ready LangChain Documents.

    Produces separate chunks for text sections and tables, with rich metadata
    including section paths, context prefixes, and content type labels.
    """
    if config is None:
        config = _config_from_settings()

    chunks: list[Document] = []
    doc_slug = _slugify(parsed_doc.doc_id)

    # Group elements by section
    sections = _group_by_section(parsed_doc.elements)

    section_idx = 0
    text_chunk_counter = 0
    table_counter = 0

    for section_path, elements in sections:
        section_idx += 1
        parent_id = f"{doc_slug}-sec{section_idx:02d}"
        section_label = " > ".join(section_path) if section_path else "Document"

        # Separate tables from text elements in this section
        text_elements: list[ParsedElement] = []
        table_elements: list[ParsedElement] = []

        for elem in elements:
            if elem.element_type == ElementType.TABLE:
                table_elements.append(elem)
            elif elem.element_type in (ElementType.HEADING, ElementType.FIGURE):
                continue  # headings are used for section structure, not chunk content
            else:
                text_elements.append(elem)

        # Chunk text elements within this section
        if text_elements:
            text_chunks = _chunk_text_elements(
                text_elements, config, doc_slug, section_idx, section_label,
                parent_id, parsed_doc,
            )
            text_chunk_counter += len(text_chunks)
            chunks.extend(text_chunks)

        # Chunk table elements
        for table_elem in table_elements:
            table_counter += 1
            table_chunks = _chunk_table_element(
                table_elem, config, doc_slug, section_idx, section_label,
                parent_id, parsed_doc, table_counter,
            )
            chunks.extend(table_chunks)

    logger.info(
        "chunking_complete",
        extra={
            "doc_id": parsed_doc.doc_id,
            "total_chunks": len(chunks),
            "text_chunks": sum(1 for c in chunks if c.metadata.get("content_type") == "text_section"),
            "table_chunks": sum(1 for c in chunks if c.metadata.get("content_type") == "table"),
        },
    )

    return chunks


def _group_by_section(elements: list[ParsedElement]) -> list[tuple[list[str], list[ParsedElement]]]:
    """Group elements into sections based on heading boundaries."""
    if not elements:
        return []

    sections: list[tuple[list[str], list[ParsedElement]]] = []
    current_path: list[str] = []
    current_elements: list[ParsedElement] = []

    for elem in elements:
        if elem.element_type == ElementType.HEADING:
            # Flush current section
            if current_elements:
                sections.append((list(current_path), current_elements))
                current_elements = []
            current_path = list(elem.section_path)
            current_elements.append(elem)
        else:
            current_elements.append(elem)

    # Flush final section
    if current_elements:
        sections.append((list(current_path), current_elements))

    return sections


def _chunk_text_elements(
    elements: list[ParsedElement],
    config: ChunkConfig,
    doc_slug: str,
    section_idx: int,
    section_label: str,
    parent_id: str,
    parsed_doc: ParsedDocument,
) -> list[Document]:
    """Chunk narrative text elements within a section."""
    # Merge element texts into paragraphs
    paragraphs: list[tuple[str, int | None]] = []
    for elem in elements:
        if elem.text.strip():
            paragraphs.append((elem.text.strip(), elem.page))

    if not paragraphs:
        return []

    # Build chunks respecting token budget
    chunks: list[Document] = []
    current_text_parts: list[str] = []
    current_tokens = 0
    current_page_start: int | None = None
    current_page_end: int | None = None
    local_idx = 0

    for para_text, para_page in paragraphs:
        para_tokens = _count_tokens(para_text)

        # If adding this paragraph exceeds max, flush current chunk
        if current_text_parts and (current_tokens + para_tokens) > config.text_chunk_max:
            chunk = _build_text_chunk(
                "\n\n".join(current_text_parts),
                doc_slug, section_idx, current_page_start or 0,
                local_idx, parent_id, section_label, parsed_doc,
                current_page_start, current_page_end,
            )
            chunks.append(chunk)
            local_idx += 1

            # Handle overlap: carry last part if it fits
            if config.text_chunk_overlap > 0 and current_text_parts:
                last_part = current_text_parts[-1]
                last_tokens = _count_tokens(last_part)
                if last_tokens <= config.text_chunk_overlap:
                    current_text_parts = [last_part]
                    current_tokens = last_tokens
                else:
                    current_text_parts = []
                    current_tokens = 0
            else:
                current_text_parts = []
                current_tokens = 0
            current_page_start = para_page
            current_page_end = para_page

        # If a single paragraph exceeds max, split it by sentences
        if para_tokens > config.text_chunk_max:
            if current_text_parts:
                chunk = _build_text_chunk(
                    "\n\n".join(current_text_parts),
                    doc_slug, section_idx, current_page_start or 0,
                    local_idx, parent_id, section_label, parsed_doc,
                    current_page_start, current_page_end,
                )
                chunks.append(chunk)
                local_idx += 1
                current_text_parts = []
                current_tokens = 0

            sentence_chunks = _split_long_text(para_text, config.text_chunk_max)
            for sc in sentence_chunks:
                chunk = _build_text_chunk(
                    sc, doc_slug, section_idx, para_page or 0,
                    local_idx, parent_id, section_label, parsed_doc,
                    para_page, para_page,
                )
                chunks.append(chunk)
                local_idx += 1
            current_page_start = para_page
            current_page_end = para_page
            continue

        current_text_parts.append(para_text)
        current_tokens += para_tokens
        if current_page_start is None:
            current_page_start = para_page
        current_page_end = para_page

    # Flush remaining
    if current_text_parts:
        chunk = _build_text_chunk(
            "\n\n".join(current_text_parts),
            doc_slug, section_idx, current_page_start or 0,
            local_idx, parent_id, section_label, parsed_doc,
            current_page_start, current_page_end,
        )
        chunks.append(chunk)

    # Tag chunk positions within section
    if chunks:
        chunks[0].metadata["chunk_position"] = "intro"
        for c in chunks[1:-1]:
            c.metadata["chunk_position"] = "body"
        if len(chunks) > 1:
            chunks[-1].metadata["chunk_position"] = "conclusion"

    return chunks


def _resolve_table_payloads(table_elem: ParsedElement, section_label: str) -> tuple[str, str, str, str]:
    """
    Return (generation_text, embedding_text, quality, serializer).

    Generation text can prefer richer HTML when Markdown is degraded.
    Embedding text prefers a normalized text form to keep vector inputs stable.
    """
    md = (table_elem.table_markdown or "").strip()
    html = (table_elem.table_html or "").strip()
    raw = (table_elem.text or "").strip()

    # Check if markdown is usable: has at least a header separator row
    md_has_header = bool(md and re.search(r'^[\s|:-]+$', md, re.MULTILINE))

    if md and md_has_header:
        return md, md, 'good', 'markdown'

    # Try HTML→simple-markdown conversion for embedding, while preserving HTML
    # for generation when Markdown would lose spans or structure.
    if html:
        converted = _html_table_to_markdown(html)
        if converted:
            conv_has_header = bool(re.search(r'^[\s|:-]+$', converted, re.MULTILINE))
            embedding_prefix = _build_embedding_prefix(section_label, table_elem.caption_text)
            embedding_text = f"{embedding_prefix}\n{converted}" if embedding_prefix else converted
            return html, embedding_text, ('good' if conv_has_header else 'degraded'), 'html'

    # Last resort: use whatever markdown/raw we have, marked degraded
    text = md or raw
    if not text:
        text = "[Empty table]"
    return text, text, 'degraded', 'text'


def _html_table_to_markdown(html: str) -> str:
    """
    Best-effort conversion of an HTML table to Markdown.

    Handles rowspan/colspan by flattening spans into repeated cells.
    Does not require external dependencies — uses regex parsing.
    """
    import re as _re

    # Extract rows
    rows: list[list[str]] = []
    for tr_match in _re.finditer(r'<tr[^>]*>(.*?)</tr>', html, _re.DOTALL | _re.IGNORECASE):
        row_html = tr_match.group(1)
        cells: list[str] = []
        for cell_match in _re.finditer(r'<(th|td)[^>]*>(.*?)</\1>', row_html, _re.DOTALL | _re.IGNORECASE):
            cell_text = _re.sub(r'<[^>]+>', '', cell_match.group(2)).strip()
            cell_text = cell_text.replace('|', '\\|')
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    return "\n".join(lines)


def _linearize_markdown_table(md_table: str) -> str | None:
    """Convert a markdown table to schema-aware embedding format.
    Returns None if the table cannot be parsed.
    """
    lines = [line.strip() for line in md_table.strip().split('\n') if line.strip()]
    if len(lines) < 2:
        return None

    sep_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^[\s|:-]+$', line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return None

    header_line = lines[sep_idx - 1]
    columns = [c.strip() for c in header_line.strip('|').split('|') if c.strip()]
    if not columns:
        return None

    data_lines = lines[sep_idx + 1:]
    out = [f"Columns = [{', '.join(columns)}]"]
    for row_line in data_lines:
        cells = [c.strip() for c in row_line.strip('|').split('|')]
        cells = [c for c in cells if c or len(cells) > len(columns)]
        while len(cells) < len(columns):
            cells.append('')
        cells = cells[:len(columns)]
        out.append(f"Row = [{', '.join(cells)}]")

    return '\n'.join(out)


def _extract_table_structure(md_text: str) -> tuple[list[str], int]:
    """Extract column names and data row count directly from markdown table text."""
    lines = [line.strip() for line in md_text.strip().split('\n') if line.strip()]
    if len(lines) < 2:
        return [], 0

    sep_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^[\s|:-]+$', line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return [], 0

    header_line = lines[sep_idx - 1]
    columns = [c.strip() for c in header_line.strip('|').split('|') if c.strip()]
    row_count = len(lines[sep_idx + 1:])
    return columns, row_count


def _chunk_table_element(
    table_elem: ParsedElement,
    config: ChunkConfig,
    doc_slug: str,
    section_idx: int,
    section_label: str,
    parent_id: str,
    parsed_doc: ParsedDocument,
    table_counter: int,
) -> list[Document]:
    """Chunk a table element, splitting large tables by row groups."""
    generation_text, embedding_text, table_quality, serializer = _resolve_table_payloads(table_elem, section_label)
    caption = table_elem.caption_text or ""

    # Schema-aware linearization for embedding text
    linearized = _linearize_markdown_table(generation_text)
    if linearized is not None:
        if caption:
            embedding_text = f"Table: {caption}\n{linearized}"
        else:
            embedding_text = linearized

    # Extract table structure metadata directly from markdown header
    table_columns, table_row_count = _extract_table_structure(generation_text)

    table_tokens = _count_tokens(generation_text)

    # Determine size tier
    if table_tokens <= config.table_chunk_max:
        table_size_tier = "small"
    elif table_tokens <= TABLE_LARGE_TOKEN_THRESHOLD:
        table_size_tier = "medium"
    else:
        table_size_tier = "huge"

    table_id = f"{doc_slug}-sec{section_idx:02d}-tbl{table_counter:02d}"
    table_parent_id = f"{parent_id}-tbl{table_counter:02d}"

    context_prefix = _build_context_prefix(
        parsed_doc.title, section_label, caption=caption,
        page=table_elem.page,
    )
    embedding_prefix = _build_embedding_prefix(section_label, caption)

    base_meta = {
        "doc_id": parsed_doc.doc_id,
        "parent_id": table_parent_id,
        "page_start": table_elem.page,
        "page_end": table_elem.page,
        "content_type": "table",
        "section_path": section_label,
        "context_prefix": context_prefix,
        "is_table": True,
        "table_id": table_id,
        "caption": caption,
        "title": caption,
        "source": parsed_doc.doc_id,
        "page": table_elem.page,
        "table_quality": table_quality,
        "table_serializer": serializer,
        "context_key": table_parent_id,
        "embedding_prefix": embedding_prefix,
        "cross_references": _extract_cross_references(generation_text),
        "table_columns": table_columns,
        "table_row_count": table_row_count,
        "table_size_tier": table_size_tier,
    }

    # If table fits in budget, return as single chunk
    if table_tokens <= config.table_chunk_max:
        return [Document(
            page_content=generation_text,
            metadata={
                **base_meta,
                "chunk_id": f"{table_id}-r00",
                "token_count_est": table_tokens,
                "table_row_range": "all",
                "embedding_text": embedding_text,
            },
        )]

    if serializer != 'markdown':
        return [Document(
            page_content=generation_text,
            metadata={
                **base_meta,
                "chunk_id": f"{table_id}-r00",
                "token_count_est": table_tokens,
                "table_row_range": "all",
                "table_quality": "degraded",
                "embedding_text": embedding_text,
            },
        )]

    # Split large table by row groups
    return _split_table_by_rows(
        generation_text, config, table_id, table_parent_id,
        section_label, context_prefix, caption, embedding_prefix,
        parsed_doc, table_elem.page, table_quality, serializer,
        table_columns=table_columns,
        table_row_count=table_row_count,
        table_size_tier=table_size_tier,
    )


def _split_table_by_rows(
    table_text: str,
    config: ChunkConfig,
    table_id: str,
    table_parent_id: str,
    section_label: str,
    context_prefix: str,
    caption: str,
    embedding_prefix: str,
    parsed_doc: ParsedDocument,
    page: int | None,
    table_quality: str = 'good',
    serializer: str = 'markdown',
    table_columns: list[str] | None = None,
    table_row_count: int = 0,
    table_size_tier: str = 'huge',
) -> list[Document]:
    """Split a Markdown table by row groups, repeating header rows."""
    lines = table_text.strip().split('\n')

    # Extract header (first line + separator line)
    header_lines: list[str] = []
    data_lines: list[str] = []
    in_header = True
    for line in lines:
        if in_header:
            header_lines.append(line)
            # Markdown table separator line (---|---|---)
            if re.match(r'^[\s|:-]+$', line):
                in_header = False
        else:
            data_lines.append(line)

    shared_meta = {
        "doc_id": parsed_doc.doc_id,
        "parent_id": table_parent_id,
        "page_start": page,
        "page_end": page,
        "content_type": "table",
        "section_path": section_label,
        "context_prefix": context_prefix,
        "is_table": True,
        "table_id": table_id,
        "caption": caption,
        "title": caption,
        "source": parsed_doc.doc_id,
        "page": page,
        "table_quality": table_quality,
        "table_serializer": serializer,
        "context_key": table_parent_id,
        "embedding_prefix": embedding_prefix,
        "table_columns": table_columns or [],
        "table_row_count": table_row_count,
        "table_size_tier": table_size_tier,
    }

    if not data_lines:
        # No data rows — return whole table as one chunk
        linearized = _linearize_markdown_table(table_text)
        if linearized is not None:
            et = f"Table: {caption}\n{linearized}" if caption else linearized
        else:
            et = f"{embedding_prefix}\n{table_text}" if embedding_prefix else table_text
        return [Document(
            page_content=table_text,
            metadata={
                **shared_meta,
                "chunk_id": f"{table_id}-r00",
                "token_count_est": _count_tokens(table_text),
                "table_row_range": "all",
                "embedding_text": et,
            },
        )]

    header_text = '\n'.join(header_lines)
    header_tokens = _count_tokens(header_text)
    budget_per_group = config.table_chunk_max - header_tokens - TABLE_HEADER_REPEAT_BUDGET

    # Extract column names for linearized embedding text
    _header_col_line = header_lines[0] if header_lines else ""
    _lin_columns = [c.strip() for c in _header_col_line.strip('|').split('|') if c.strip()]

    def _build_row_group_embedding(row_lines: list[str]) -> str:
        """Build linearized embedding text for a row-group chunk."""
        if not _lin_columns:
            # Fallback: no columns parsed
            chunk_md = header_text + '\n' + '\n'.join(row_lines)
            return f"{embedding_prefix}\n{chunk_md}" if embedding_prefix else chunk_md
        out_parts = [f"Columns = [{', '.join(_lin_columns)}]"]
        for rl in row_lines:
            cells = [c.strip() for c in rl.strip('|').split('|')]
            cells = [c for c in cells if c or len(cells) > len(_lin_columns)]
            while len(cells) < len(_lin_columns):
                cells.append('')
            cells = cells[:len(_lin_columns)]
            out_parts.append(f"Row = [{', '.join(cells)}]")
        linearized = '\n'.join(out_parts)
        if caption:
            return f"Table: {caption}\n{linearized}"
        return linearized

    chunks: list[Document] = []
    current_rows: list[str] = []
    current_tokens = 0
    row_group_idx = 0
    row_start = 0

    for i, row_line in enumerate(data_lines):
        row_tokens = _count_tokens(row_line)

        if current_rows and (current_tokens + row_tokens) > budget_per_group:
            chunk_text = header_text + '\n' + '\n'.join(current_rows)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **shared_meta,
                    "chunk_id": f"{table_id}-r{row_group_idx:02d}",
                    "token_count_est": _count_tokens(chunk_text),
                    "table_row_range": f"{row_start}-{row_start + len(current_rows) - 1}",
                    "embedding_text": _build_row_group_embedding(current_rows),
                },
            ))
            row_group_idx += 1
            row_start = i
            current_rows = []
            current_tokens = 0

        current_rows.append(row_line)
        current_tokens += row_tokens

    # Flush remaining rows
    if current_rows:
        chunk_text = header_text + '\n' + '\n'.join(current_rows)
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                **shared_meta,
                "chunk_id": f"{table_id}-r{row_group_idx:02d}",
                "token_count_est": _count_tokens(chunk_text),
                "table_row_range": f"{row_start}-{row_start + len(current_rows) - 1}",
                "embedding_text": _build_row_group_embedding(current_rows),
            },
        ))

    return chunks


def _build_text_chunk(
    text: str,
    doc_slug: str,
    section_idx: int,
    page_start_fallback: int,
    local_idx: int,
    parent_id: str,
    section_label: str,
    parsed_doc: ParsedDocument,
    page_start: int | None,
    page_end: int | None,
) -> Document:
    """Build a text section Document with full metadata."""
    context_prefix = _build_context_prefix(
        parsed_doc.title, section_label, page=page_start,
    )
    embedding_prefix = _build_embedding_prefix(section_label)
    chunk_id = f"{doc_slug}-sec{section_idx:02d}-p{page_start or page_start_fallback}-t{local_idx:03d}"

    return Document(
        page_content=text,
        metadata={
            "doc_id": parsed_doc.doc_id,
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "page_start": page_start,
            "page_end": page_end,
            "content_type": "text_section",
            "section_path": section_label,
            "token_count_est": _count_tokens(text),
            "context_prefix": context_prefix,
            "context_key": f"{doc_slug}-sec{section_idx:02d}",
            "embedding_prefix": embedding_prefix,
            "is_table": False,
            "source": parsed_doc.doc_id,
            "page": page_start,
            "cross_references": _extract_cross_references(text),
        },
    )


def _build_context_prefix(
    doc_title: str | None,
    section_label: str,
    caption: str | None = None,
    page: int | None = None,
) -> str:
    """Build a deterministic context prefix for embedding enrichment."""
    parts: list[str] = []
    if doc_title:
        parts.append(doc_title)
    if section_label and section_label != "Document":
        parts.append(section_label)
    prefix = " > ".join(parts)
    if caption:
        prefix = f"{prefix} | {caption}" if prefix else caption
    if page is not None:
        prefix = f"[p.{page}] {prefix}" if prefix else f"[p.{page}]"
    return prefix


def _build_embedding_prefix(section_label: str, caption: str | None = None) -> str:
    parts: list[str] = []
    if section_label and section_label != "Document":
        parts.append(section_label)
    if caption:
        parts.append(caption[:120].strip())
    return " | ".join([p for p in parts if p])


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """
    Split oversized text using transparent plain-Python structural fallbacks.

    Order:
    1. Markdown heading breaks (`\\n\\n### `)
    2. Paragraph breaks (`\\n\\n`)
    3. Line breaks (`\\n`)
    4. Sentence regex as absolute last resort
    """
    normalized = text.strip()
    if not normalized:
        return [text]

    hierarchy = [
        ("markdown_heading", _split_on_markdown_heading),
        ("paragraph", _split_on_double_newline),
        ("line", _split_on_single_newline),
        ("sentence", _split_on_sentences),
    ]

    for _, splitter in hierarchy:
        segments = splitter(normalized)
        # If a splitter cannot create any boundary at all, skip it so we can
        # continue down the fallback chain and eventually hard-wrap.
        if len(segments) == 1 and segments[0].strip() == normalized:
            continue
        chunks = _pack_segments(segments, max_tokens)
        if chunks and all(_count_tokens(chunk) <= max_tokens for chunk in chunks):
            return chunks

    return _hard_wrap_text(normalized, max_tokens)


def _pack_segments(segments: list[str], max_tokens: int) -> list[str]:
    """Pack ordered text segments into chunks under the token budget."""
    clean_segments = [segment for segment in segments if segment and segment.strip()]
    if not clean_segments:
        return []

    chunks: list[str] = []
    current = clean_segments[0]

    for segment in clean_segments[1:]:
        candidate = current + segment
        if _count_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        if _count_tokens(current) <= max_tokens:
            chunks.append(current.strip())
        else:
            # If even a single segment is too large, recurse to the next fallback layer.
            nested = _split_long_text(current, max_tokens)
            chunks.extend(chunk.strip() for chunk in nested if chunk.strip())
        current = segment

    if current.strip():
        if _count_tokens(current) <= max_tokens:
            chunks.append(current.strip())
        else:
            nested = _split_long_text(current, max_tokens)
            chunks.extend(chunk.strip() for chunk in nested if chunk.strip())

    return chunks


def _split_on_markdown_heading(text: str) -> list[str]:
    """Split while preserving markdown heading markers with the following block."""
    parts = re.split(r'(?=\n\n### )', text)
    return parts if len(parts) > 1 else [text]


def _split_on_double_newline(text: str) -> list[str]:
    """Split while preserving paragraph separators with the following block."""
    parts = re.split(r'(?=\n\n)', text)
    return parts if len(parts) > 1 else [text]


def _split_on_single_newline(text: str) -> list[str]:
    """Split while preserving line breaks with the following line."""
    parts = re.split(r'(?=\n)', text)
    return parts if len(parts) > 1 else [text]


def _split_on_sentences(text: str) -> list[str]:
    """Absolute last resort splitter for long prose spans."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return parts if len(parts) > 1 else [text]


def _hard_wrap_text(text: str, max_tokens: int) -> list[str]:
    """Final safety split for delimiter-free spans."""
    words = text.split()
    if not words:
        return [text]

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate_words = current_words + [word]
        candidate = " ".join(candidate_words)
        if current_words and _count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words = candidate_words

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


_CROSS_REF_PATTERNS = [
    (r'\bchapter\s+([\d]+(?:\.\d+)*)', 'chapter'),
    (r'\bsection\s+([\d]+(?:\.\d+)*)', 'section'),
    (r'\btable\s+([\d]+(?:\.\d+)*)', 'table'),
    (r'\bfigure\s+([\d]+(?:\.\d+)*)', 'figure'),
    (r'\bappendix\s+([a-zA-Z\d]+(?:\.\d+)*)', 'appendix'),
]


def _extract_cross_references(text: str) -> list[dict[str, str]]:
    """Extract structured cross-references from chunk text."""
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    lower = text.lower()
    for pattern, ref_type in _CROSS_REF_PATTERNS:
        for match in re.finditer(pattern, lower):
            number = match.group(1)
            normalized = f"{ref_type}_{number.replace('.', '_')}"
            if normalized not in seen:
                seen.add(normalized)
                refs.append({"type": ref_type, "normalized": normalized})
    return refs


def _slugify(text: str) -> str:
    """Create a filesystem/ID-safe slug from text."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    return slug.strip('_')[:50]


def _config_from_settings_for_mode(settings=None) -> ChunkConfig:
    """Build ChunkConfig, using child sizes when parent-child chunking is enabled."""
    if settings is None:
        from core.settings import get_settings
        settings = get_settings()

    if settings.enable_parent_child_chunking:
        return ChunkConfig(
            text_chunk_target=settings.chunk_child_target,
            text_chunk_max=settings.chunk_child_max,
            text_chunk_overlap=settings.chunk_text_overlap,
            table_chunk_max=settings.chunk_table_max,
        )
    return ChunkConfig(
        text_chunk_target=settings.chunk_text_target,
        text_chunk_max=settings.chunk_text_max,
        text_chunk_overlap=settings.chunk_text_overlap,
        table_chunk_max=settings.chunk_table_max,
    )


def group_children_into_parents(
    chunks: list[Document],
    max_children_per_parent: int = 5,
) -> tuple[list[Document], list[Document]]:
    """Group consecutive same-section text chunks into parent chunks.

    Tables become their own parent (they are self-contained retrieval units).
    Text chunks within the same section are grouped into parents of up to
    max_children_per_parent children.

    Returns (parent_chunks, updated_child_chunks) where each child has
    'parent_chunk_id' and 'chunk_type' set in metadata.
    """
    if not chunks:
        return [], []

    parents: list[Document] = []
    updated_children: list[Document] = []

    # Group chunks by context_key (section identifier)
    groups: list[tuple[str, list[Document]]] = []
    current_key: str | None = None
    current_group: list[Document] = []

    for chunk in chunks:
        meta = chunk.metadata
        key = meta.get("context_key", "")
        content_type = meta.get("content_type", "text_section")

        # Tables always start a new group (self-contained)
        if content_type == "table":
            if current_group:
                groups.append((current_key or "", current_group))
                current_group = []
            groups.append((key, [chunk]))
            current_key = None
            continue

        # Same section -> accumulate
        if key == current_key:
            current_group.append(chunk)
        else:
            if current_group:
                groups.append((current_key or "", current_group))
            current_key = key
            current_group = [chunk]

    if current_group:
        groups.append((current_key or "", current_group))

    # Create parents from groups
    parent_idx = 0
    for _section_key, group_chunks in groups:
        for start in range(0, len(group_chunks), max_children_per_parent):
            sub_group = group_chunks[start:start + max_children_per_parent]
            first_meta = sub_group[0].metadata

            parent_chunk_id = f"{first_meta.get('doc_id', 'doc')}-parent-{parent_idx:04d}"
            parent_idx += 1

            parent_content = "\n\n".join(c.page_content for c in sub_group)

            pages = [c.metadata.get("page_start") or c.metadata.get("page") for c in sub_group]
            pages = [p for p in pages if p is not None]
            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None

            parent = Document(
                page_content=parent_content,
                metadata={
                    "chunk_id": parent_chunk_id,
                    "doc_id": first_meta.get("doc_id", ""),
                    "section_path": first_meta.get("section_path", ""),
                    "content_type": first_meta.get("content_type", "text_section"),
                    "context_key": first_meta.get("context_key", ""),
                    "context_prefix": first_meta.get("context_prefix", ""),
                    "chunk_type": "parent",
                    "child_chunk_ids": [c.metadata["chunk_id"] for c in sub_group],
                    "page_start": page_start,
                    "page_end": page_end,
                    "page": page_start,
                    "source": first_meta.get("source", ""),
                },
            )
            parents.append(parent)

            for child in sub_group:
                child.metadata["parent_chunk_id"] = parent_chunk_id
                child.metadata["chunk_type"] = "child"
                updated_children.append(child)

    return parents, updated_children
