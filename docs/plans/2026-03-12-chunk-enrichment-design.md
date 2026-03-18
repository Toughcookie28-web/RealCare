# Chunk Enrichment & Table Linearization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve retrieval quality by enriching chunk metadata (keywords, entities, cross-references, context summaries) and replacing markdown table embedding text with a schema-aware linearized format.

**Architecture:** Three layers of enrichment: (1) static context prefix enrichment derived from document structure at chunk time — zero cost, (2) schema-aware table linearization at chunk time — zero cost, (3) optional LLM-based enrichment pass (context summary + keywords + medical entities) behind `ENRICH_CHUNKS_WITH_LLM` setting. Cross-references extracted via regex. All enrichment writes to chunk metadata and `embedding_text`; `page_content` (what the LLM reads at generation time) stays unchanged.

**Tech Stack:** Python, pdf_chunker.py, embedding_client.py, core/settings.py, tools/llm_client.py (Groq), pytest

---

### Task 1: Wire ChunkConfig to Settings (env-configurable chunk sizes for A/B testing)

**Files:**
- Modify: `core/settings.py`
- Modify: `tools/pdf_chunker.py`
- Modify: `tools/pdf_loader.py`
- Test: `tests/rag/test_pdf_chunker_contract.py`

**Step 1: Write the failing test**

```python
# tests/rag/test_pdf_chunker_contract.py

def test_chunk_config_reads_from_settings():
    """ChunkConfig should be constructable from Settings defaults."""
    from core.settings import get_settings
    settings = get_settings()
    assert hasattr(settings, 'chunk_text_target')
    assert hasattr(settings, 'chunk_text_max')
    assert hasattr(settings, 'chunk_text_overlap')
    assert hasattr(settings, 'chunk_table_max')
    assert settings.chunk_text_target == 400
    assert settings.chunk_text_max == 500
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py::test_chunk_config_reads_from_settings -v`
Expected: FAIL with AttributeError

**Step 3: Add chunk settings to `core/settings.py`**

```python
# After bm25_b field:
chunk_text_target: int = Field(default=400, alias='CHUNK_TEXT_TARGET')
chunk_text_max: int = Field(default=500, alias='CHUNK_TEXT_MAX')
chunk_text_overlap: int = Field(default=50, alias='CHUNK_TEXT_OVERLAP')
chunk_table_max: int = Field(default=450, alias='CHUNK_TABLE_MAX')
enrich_chunks_with_llm: bool = Field(default=False, alias='ENRICH_CHUNKS_WITH_LLM')
```

**Step 4: Wire ChunkConfig construction from Settings in `pdf_chunker.py`**

```python
def _config_from_settings() -> ChunkConfig:
    from core.settings import get_settings
    settings = get_settings()
    return ChunkConfig(
        text_chunk_target=settings.chunk_text_target,
        text_chunk_max=settings.chunk_text_max,
        text_chunk_overlap=settings.chunk_text_overlap,
        table_chunk_max=settings.chunk_table_max,
    )
```

Update `chunk_parsed_document()` default:
```python
def chunk_parsed_document(
    parsed_doc: ParsedDocument,
    config: ChunkConfig | None = None,
) -> list[Document]:
    if config is None:
        config = _config_from_settings()
    ...
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add core/settings.py tools/pdf_chunker.py tests/rag/test_pdf_chunker_contract.py
git commit -m "feat: wire ChunkConfig to Settings for env-configurable chunk sizes"
```

---

### Task 2: Schema-aware table linearization for embedding text

**Files:**
- Modify: `tools/pdf_chunker.py` (add `_linearize_markdown_table()`, update `_chunk_table_element()`)
- Test: `tests/rag/test_pdf_chunker_contract.py`

**Step 1: Write the failing test**

```python
def test_linearize_markdown_table_produces_schema_aware_format():
    module = _load_pdf_chunker_module()
    md = (
        "| Drug | Dose | Route |\n"
        "| --- | --- | --- |\n"
        "| Aspirin | 100mg | Oral |\n"
        "| Ibuprofen | 200mg | Oral |"
    )
    result = module._linearize_markdown_table(md)
    assert "Columns = [Drug, Dose, Route]" in result
    assert "Row = [Aspirin, 100mg, Oral]" in result
    assert "Row = [Ibuprofen, 200mg, Oral]" in result
    # No markdown pipes in output
    assert "|" not in result


def test_linearize_table_with_no_separator_returns_none():
    module = _load_pdf_chunker_module()
    result = module._linearize_markdown_table("just some text")
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py::test_linearize_markdown_table_produces_schema_aware_format -v`
Expected: FAIL with AttributeError

**Step 3: Implement `_linearize_markdown_table()` in `pdf_chunker.py`**

```python
def _linearize_markdown_table(md_table: str) -> str | None:
    """Convert a markdown table to schema-aware embedding format.

    Returns None if the table cannot be parsed (no header separator found).
    """
    lines = [line.strip() for line in md_table.strip().split('\n') if line.strip()]
    if len(lines) < 2:
        return None

    # Find header separator row
    sep_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^[\s|:-]+$', line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return None

    # Parse header columns
    header_line = lines[sep_idx - 1]
    columns = [c.strip() for c in header_line.strip('|').split('|') if c.strip()]
    if not columns:
        return None

    # Parse data rows
    data_lines = lines[sep_idx + 1:]
    out = [f"Columns = [{', '.join(columns)}]"]
    for row_line in data_lines:
        cells = [c.strip() for c in row_line.strip('|').split('|') if c.strip() or row_line.count('|') > 1]
        # Pad or trim to match column count
        while len(cells) < len(columns):
            cells.append('')
        cells = cells[:len(columns)]
        out.append(f"Row = [{', '.join(cells)}]")

    return '\n'.join(out)
```

**Step 4: Update `_chunk_table_element()` to use linearized embedding text**

In `_resolve_table_payloads()`, after building the generation text, also produce a linearized embedding text. Update the return signature to include a `linearized` field.

Alternatively, in `_chunk_table_element()`, after calling `_resolve_table_payloads()`, attempt linearization:

```python
# After line 342 in _chunk_table_element:
linearized = _linearize_markdown_table(generation_text)
if linearized is not None:
    embedding_text = linearized
    if caption:
        embedding_text = f"Table: {caption}\n{embedding_text}"
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tools/pdf_chunker.py tests/rag/test_pdf_chunker_contract.py
git commit -m "feat: schema-aware table linearization for embedding text"
```

---

### Task 3: Table metadata enrichment (columns, row count, size tier, title)

**Files:**
- Modify: `tools/pdf_chunker.py`
- Test: `tests/rag/test_pdf_chunker_contract.py`

**Step 1: Write the failing test**

```python
def test_table_chunk_metadata_includes_columns_and_row_count():
    module = _load_pdf_chunker_module()
    from tools.pdf_parser import ParsedDocument, ParsedElement, ElementType

    table_md = (
        "| Drug | Dose |\n"
        "| --- | --- |\n"
        "| Aspirin | 100mg |\n"
        "| Ibuprofen | 200mg |"
    )
    parsed = ParsedDocument(
        doc_id="test-doc",
        title="Test Book",
        elements=[
            ParsedElement(element_type=ElementType.TABLE, text=table_md,
                         page=1, section_path=["Chapter 1"],
                         table_markdown=table_md, element_index=0),
        ],
    )
    chunks = module.chunk_parsed_document(parsed)
    assert len(chunks) >= 1
    meta = chunks[0].metadata
    assert meta["table_columns"] == ["Drug", "Dose"]
    assert meta["table_row_count"] == 2
    assert meta["table_size_tier"] == "small"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py::test_table_chunk_metadata_includes_columns_and_row_count -v`
Expected: FAIL with KeyError

**Step 3: Add metadata fields in `_chunk_table_element()`**

In `_chunk_table_element()`, after linearization, extract column names and row count:

```python
# Extract table structure metadata
table_columns = []
table_row_count = 0
linearized = _linearize_markdown_table(generation_text)
if linearized is not None:
    lin_lines = linearized.split('\n')
    for line in lin_lines:
        if line.startswith('Columns = ['):
            cols_str = line[len('Columns = ['):-1]
            table_columns = [c.strip() for c in cols_str.split(',')]
        elif line.startswith('Row = ['):
            table_row_count += 1

# Determine size tier
table_tokens = _count_tokens(generation_text)
if table_tokens <= config.table_chunk_max:
    table_size_tier = "small"
elif table_tokens <= 2000:
    table_size_tier = "medium"
else:
    table_size_tier = "huge"
```

Add to `base_meta`:
```python
"table_columns": table_columns,
"table_row_count": table_row_count,
"table_size_tier": table_size_tier,
```

**Step 4: Run tests**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/pdf_chunker.py tests/rag/test_pdf_chunker_contract.py
git commit -m "feat: add table_columns, table_row_count, table_size_tier to table chunk metadata"
```

---

### Task 4: Static context prefix enrichment (chunk position)

**Files:**
- Modify: `tools/pdf_chunker.py` (update `_build_context_prefix`, `_build_embedding_prefix`)
- Test: `tests/rag/test_pdf_chunker_contract.py`

**Step 1: Write the failing test**

```python
def test_text_chunk_metadata_includes_position():
    module = _load_pdf_chunker_module()
    from tools.pdf_parser import ParsedDocument, ParsedElement, ElementType

    elements = [
        ParsedElement(element_type=ElementType.HEADING, text="Section A",
                     heading_level=1, section_path=["Section A"], page=1, element_index=0),
        ParsedElement(element_type=ElementType.PARAGRAPH, text="First paragraph. " * 30,
                     page=1, section_path=["Section A"], element_index=1),
        ParsedElement(element_type=ElementType.PARAGRAPH, text="Middle paragraph. " * 30,
                     page=1, section_path=["Section A"], element_index=2),
        ParsedElement(element_type=ElementType.PARAGRAPH, text="Last paragraph. " * 30,
                     page=2, section_path=["Section A"], element_index=3),
    ]
    parsed = ParsedDocument(doc_id="test-doc", title="Test Book", elements=elements)
    config = module.ChunkConfig(text_chunk_target=40, text_chunk_max=50, text_chunk_overlap=0)
    chunks = module.chunk_parsed_document(parsed, config=config)

    positions = [c.metadata.get("chunk_position") for c in chunks]
    assert positions[0] == "intro"
    assert positions[-1] == "conclusion"
    if len(positions) > 2:
        assert all(p == "body" for p in positions[1:-1])
```

**Step 2: Run test to verify it fails**

**Step 3: Add chunk_position to text chunk metadata**

In `_chunk_text_elements()`, after building all chunks for a section, tag them:

```python
# After all chunks are built, before returning:
if chunks:
    chunks[0].metadata["chunk_position"] = "intro"
    for c in chunks[1:-1]:
        c.metadata["chunk_position"] = "body"
    if len(chunks) > 1:
        chunks[-1].metadata["chunk_position"] = "conclusion"
    else:
        chunks[0].metadata["chunk_position"] = "intro"
```

Update `_build_embedding_prefix` to include position:

```python
def _build_embedding_prefix(section_label: str, caption: str | None = None, position: str | None = None) -> str:
    parts: list[str] = []
    if section_label and section_label != "Document":
        parts.append(section_label)
    if caption:
        parts.append(caption[:120].strip())
    if position:
        parts.append(f"[{position}]")
    return " | ".join([p for p in parts if p])
```

**Step 4: Run tests**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/pdf_chunker.py tests/rag/test_pdf_chunker_contract.py
git commit -m "feat: add chunk_position (intro/body/conclusion) to text chunk metadata"
```

---

### Task 5: Cross-reference extraction (regex-based, structured labels)

**Files:**
- Modify: `tools/pdf_chunker.py` (add `_extract_cross_references()`)
- Test: `tests/rag/test_pdf_chunker_contract.py`

**Step 1: Write the failing test**

```python
def test_extract_cross_references():
    module = _load_pdf_chunker_module()
    text = "See Chapter 8 for details. Refer to Table 5.2 and Figure 3.1 for illustration. As discussed in Section 2.4."
    refs = module._extract_cross_references(text)
    types_found = {r["type"] for r in refs}
    assert "chapter" in types_found
    assert "table" in types_found
    assert "figure" in types_found
    assert "section" in types_found
    # Check normalized format
    chapter_ref = next(r for r in refs if r["type"] == "chapter")
    assert chapter_ref["normalized"] == "chapter_8"
    table_ref = next(r for r in refs if r["type"] == "table")
    assert table_ref["normalized"] == "table_5_2"
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `_extract_cross_references()`**

```python
_CROSS_REF_PATTERNS = [
    (r'(?:see|refer(?:\s+to)?|described\s+in|discussed\s+in)\s+chapter\s+([\d]+(?:\.\d+)*)', 'chapter'),
    (r'(?:see|refer(?:\s+to)?|described\s+in|discussed\s+in)\s+section\s+([\d]+(?:\.\d+)*)', 'section'),
    (r'(?:see|refer(?:\s+to)?|in)\s+table\s+([\d]+(?:\.\d+)*)', 'table'),
    (r'(?:see|refer(?:\s+to)?|in)\s+figure\s+([\d]+(?:\.\d+)*)', 'figure'),
    (r'(?:see|refer(?:\s+to)?|in)\s+appendix\s+([a-zA-Z\d]+(?:\.\d+)*)', 'appendix'),
    # Standalone patterns (without preceding verb)
    (r'\btable\s+([\d]+(?:\.\d+)*)\b', 'table'),
    (r'\bfigure\s+([\d]+(?:\.\d+)*)\b', 'figure'),
    (r'\bchapter\s+([\d]+(?:\.\d+)*)\b', 'chapter'),
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
```

**Step 4: Wire into `_build_text_chunk()` and `_chunk_table_element()`**

Add to metadata:
```python
"cross_references": _extract_cross_references(text),
```

**Step 5: Run tests**

Run: `python -m pytest tests/rag/test_pdf_chunker_contract.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tools/pdf_chunker.py tests/rag/test_pdf_chunker_contract.py
git commit -m "feat: regex-based structured cross-reference extraction for chunk metadata"
```

---

### Task 6: LLM enrichment module (context summary + keywords + medical entities)

**Files:**
- Create: `tools/chunk_enrichment.py`
- Test: `tests/rag/test_chunk_enrichment.py`

**Step 1: Write the failing test**

```python
# tests/rag/test_chunk_enrichment.py

def test_parse_enrichment_response_extracts_all_fields():
    from tools.chunk_enrichment import _parse_enrichment_response

    raw = '''{
        "context_summary": "This section covers pediatric acetaminophen dosing.",
        "keywords": ["acetaminophen", "pediatric", "dosing"],
        "medical_entities": {
            "drugs": ["acetaminophen"],
            "conditions": ["fever"],
            "procedures": [],
            "populations": ["pediatric"]
        }
    }'''
    result = _parse_enrichment_response(raw)
    assert result["context_summary"] == "This section covers pediatric acetaminophen dosing."
    assert "acetaminophen" in result["keywords"]
    assert result["medical_entities"]["drugs"] == ["acetaminophen"]


def test_parse_enrichment_response_handles_malformed_json():
    from tools.chunk_enrichment import _parse_enrichment_response

    result = _parse_enrichment_response("not json at all")
    assert result["context_summary"] == ""
    assert result["keywords"] == []
    assert result["medical_entities"] == {"drugs": [], "conditions": [], "procedures": [], "populations": []}


def test_build_enrichment_prompt_includes_section_and_content():
    from tools.chunk_enrichment import _build_enrichment_prompt

    prompt = _build_enrichment_prompt(
        content="Aspirin is used for pain relief.",
        section_path="Chapter 5 > Pain Management",
        content_type="text_section",
    )
    assert "Aspirin" in prompt
    assert "Chapter 5" in prompt
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rag/test_chunk_enrichment.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Create `tools/chunk_enrichment.py`**

```python
"""
LLM-based chunk enrichment for RAG metadata.

Generates context summaries, keywords, and medical entity extractions
for document chunks at index time. Controlled by ENRICH_CHUNKS_WITH_LLM setting.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_ENRICHMENT_SYSTEM = (
    "You are a medical document indexing assistant. "
    "Given a chunk of text from a medical textbook, extract structured metadata. "
    "Respond ONLY with valid JSON, no other text."
)

_EMPTY_ENTITIES = {"drugs": [], "conditions": [], "procedures": [], "populations": []}


def _build_enrichment_prompt(
    content: str,
    section_path: str,
    content_type: str,
) -> str:
    return (
        f"Section: {section_path}\n"
        f"Content type: {content_type}\n"
        f"Chunk text:\n{content[:2000]}\n\n"
        "Return JSON with these fields:\n"
        '- "context_summary": 1-3 sentences describing what this chunk covers. '
        "Use 1 sentence for simple factual content, 2-3 for complex or multi-topic content. "
        "Include key medical terms.\n"
        '- "keywords": list of 3-8 important keywords or phrases from this chunk.\n'
        '- "medical_entities": object with keys "drugs", "conditions", "procedures", "populations", '
        "each a list of strings found in the chunk.\n"
    )


def _parse_enrichment_response(raw: str) -> dict[str, Any]:
    """Parse LLM enrichment response, with fallback for malformed JSON."""
    defaults: dict[str, Any] = {
        "context_summary": "",
        "keywords": [],
        "medical_entities": dict(_EMPTY_ENTITIES),
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
        "keywords": list(parsed.get("keywords", [])),
        "medical_entities": {
            "drugs": list(parsed.get("medical_entities", {}).get("drugs", [])),
            "conditions": list(parsed.get("medical_entities", {}).get("conditions", [])),
            "procedures": list(parsed.get("medical_entities", {}).get("procedures", [])),
            "populations": list(parsed.get("medical_entities", {}).get("populations", [])),
        },
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

    Each item in chunks should have keys: content, section_path, content_type.
    Returns a list of enrichment dicts in the same order.
    """
    results = []
    for i, chunk in enumerate(chunks):
        try:
            result = enrich_single_chunk(
                content=chunk["content"],
                section_path=chunk.get("section_path", ""),
                content_type=chunk.get("content_type", "text_section"),
            )
            logger.info("chunk_enriched", extra={"index": i, "keywords": len(result["keywords"])})
        except Exception as exc:
            logger.warning("chunk_enrichment_failed index=%d: %s", i, exc)
            result = _parse_enrichment_response("")
        results.append(result)
    return results
```

**Step 4: Run tests**

Run: `python -m pytest tests/rag/test_chunk_enrichment.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/chunk_enrichment.py tests/rag/test_chunk_enrichment.py
git commit -m "feat: LLM-based chunk enrichment module (summary, keywords, medical entities)"
```

---

### Task 7: Wire LLM enrichment into the ingest pipeline

**Files:**
- Modify: `tools/vector_store.py` (call enrichment after chunking, before embedding)
- Test: `tests/rag/test_chunk_enrichment.py`

**Step 1: Write the failing test**

```python
def test_enrichment_metadata_injected_into_chunk_documents(monkeypatch):
    """When ENRICH_CHUNKS_WITH_LLM=True, ingest should add enrichment metadata."""
    from unittest.mock import patch
    from tools.chunk_enrichment import _parse_enrichment_response

    fake_response = {
        "context_summary": "Test summary.",
        "keywords": ["test"],
        "medical_entities": {"drugs": [], "conditions": [], "procedures": [], "populations": []},
    }

    # Just test the apply function directly
    from tools.vector_store import _apply_enrichment_to_chunks
    from langchain_core.documents import Document

    chunks = [Document(page_content="Some content", metadata={"section_path": "Ch1", "content_type": "text_section"})]
    enrichments = [fake_response]
    _apply_enrichment_to_chunks(chunks, enrichments)

    assert chunks[0].metadata["context_summary"] == "Test summary."
    assert chunks[0].metadata["keywords"] == ["test"]
    assert "embedding_text" not in chunks[0].metadata or "Test summary" in chunks[0].metadata.get("embedding_text", "")
```

**Step 2: Run test to verify it fails**

**Step 3: Add enrichment wiring to `tools/vector_store.py`**

```python
def _apply_enrichment_to_chunks(
    chunks: list[Any],
    enrichments: list[dict[str, Any]],
) -> None:
    """Merge LLM enrichment results into chunk metadata and embedding_text."""
    for chunk, enrichment in zip(chunks, enrichments):
        meta = chunk.metadata
        summary = enrichment.get("context_summary", "")
        meta["context_summary"] = summary
        meta["keywords"] = enrichment.get("keywords", [])
        meta["medical_entities"] = enrichment.get("medical_entities", {})

        # Prepend context summary to embedding text
        if summary:
            existing_embedding = meta.get("embedding_text", "")
            prefix = f"[Context: {summary}]"
            if existing_embedding:
                meta["embedding_text"] = f"{prefix}\n{existing_embedding}"
            else:
                # Build from page_content with prefix
                embedding_prefix = meta.get("embedding_prefix", "")
                base = f"{embedding_prefix}\n{chunk.page_content}" if embedding_prefix else chunk.page_content
                meta["embedding_text"] = f"{prefix}\n{base}"
```

In `ingest_pdf_to_vector_store()`, after chunking and before embedding:

```python
from core.settings import get_settings

# After: chunks = chunk_pdf_document(parsed_doc)
if get_settings().enrich_chunks_with_llm and chunks:
    from tools.chunk_enrichment import enrich_chunks_batch
    enrichment_inputs = [
        {
            "content": c.page_content,
            "section_path": c.metadata.get("section_path", ""),
            "content_type": c.metadata.get("content_type", "text_section"),
        }
        for c in chunks
    ]
    enrichments = enrich_chunks_batch(enrichment_inputs)
    _apply_enrichment_to_chunks(chunks, enrichments)
    logger.info("ingest_enrichment_complete", extra={"doc_id": doc_id, "enriched": len(enrichments)})
```

**Step 4: Run tests**

Run: `python -m pytest tests/rag/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tools/vector_store.py tests/rag/test_chunk_enrichment.py
git commit -m "feat: wire LLM chunk enrichment into ingest pipeline behind ENRICH_CHUNKS_WITH_LLM"
```

---

### Task 8: Fix embed_documents_batch provider ordering bug

**Files:**
- Modify: `tools/embedding_client.py`
- Test: `tests/rag/test_embedding_client_contract.py`

**Step 1: Write the failing test**

```python
def test_batch_embedding_respects_provider_order_local_first():
    """When provider order is [local], batch embedding should NOT try OpenAI first."""
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        openai_api_key = "sk-test"
        openai_base_url = None
        embedding_provider = "local"
        embedding_enable_backup = False
        embedding_dim = 768
        embedding_local_model = "unused"
        embedding_local_trust_remote_code = False
        embedding_local_strategy = "plain"
        embedding_cache_dir = "/tmp/unused"
        embedding_gemini_model = "unused"
        google_api_key = None
        embedding_openai_model = "text-embedding-3-large"

        def embedding_provider_order(self, primary_override=None):
            return ["local"]

    core_settings_module.get_settings = lambda: Settings()

    bootstrap_module = types.ModuleType("tools.embedding_bootstrap")
    bootstrap_module.configure_embedding_cache_env = lambda cache_dir: cache_dir

    openai_called = {"count": 0}
    requests_module = types.ModuleType("requests")
    def _post(url, headers=None, json=None, timeout=None):
        openai_called["count"] += 1
    requests_module.post = _post

    module = _load_embedding_client(
        "embedding_client_batch_order_contract",
        {
            "core.settings": core_settings_module,
            "tools.embedding_bootstrap": bootstrap_module,
            "requests": requests_module,
        },
    )

    # Mock local model to return valid vectors
    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            import types as t
            class FakeArray:
                def __init__(self, v): self._v = v
                def tolist(self): return self._v
            return [FakeArray([0.1] * 768) for _ in texts]

    module.get_local_embedding_model = lambda: FakeModel()

    vectors = module.embed_documents_batch(["a", "b"])
    assert len(vectors) == 2
    assert openai_called["count"] == 0  # Should NOT have tried OpenAI
```

**Step 2: Run test to verify it fails**

Expected: FAIL because current code tries OpenAI first in batch path.

**Step 3: Fix `embed_documents_batch()` provider ordering**

Replace the current logic (lines 316-322) that tries OpenAI unconditionally with a proper provider-order loop:

```python
def embed_documents_batch(
    texts: list[str],
    batch_size: int = 32,
    *,
    overrides: EmbeddingOverrides | None = None,
) -> list[list[float]]:
    """Embed multiple texts in batches, respecting provider order."""
    if not texts:
        return []

    settings = get_settings()
    dim = _resolve_dim(settings, overrides)
    provider_order = _resolve_provider_order(settings, overrides)

    for provider in provider_order:
        if provider == 'openai':
            openai_vectors = _embed_batch_with_openai(texts, overrides=overrides)
            if openai_vectors is not None:
                return [
                    _validate_dimension(vec, dim, provider='openai', mode='document')
                    for vec in openai_vectors
                ]

        elif provider == 'local':
            model = get_local_embedding_model()
            if model is not None:
                strategy = _normalize_local_strategy(settings.embedding_local_strategy)
                try:
                    encoder = None
                    if strategy == 'asymmetric':
                        candidate = getattr(model, 'encode_document', None)
                        if callable(candidate):
                            encoder = candidate

                    all_vectors: list[list[float]] = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        if encoder is not None:
                            vecs = encoder(batch, normalize_embeddings=True)
                        else:
                            vecs = model.encode(batch, normalize_embeddings=True)
                        for vec in vecs:
                            v = vec.tolist() if hasattr(vec, 'tolist') else [float(x) for x in vec]
                            all_vectors.append(_validate_dimension(v, dim, provider='local', mode='document'))

                    logger.info("batch_embedding_complete", extra={"count": len(all_vectors), "batch_size": batch_size})
                    return all_vectors
                except Exception as exc:
                    logger.warning("Batch encoding failed for local, trying next provider: %s", exc)

    # Fallback: sequential embedding
    return [embed_document(t, overrides=overrides) for t in texts]
```

**Step 4: Run tests**

Run: `python -m pytest tests/rag/test_embedding_client_contract.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tools/embedding_client.py tests/rag/test_embedding_client_contract.py
git commit -m "fix: embed_documents_batch now respects provider order instead of trying OpenAI first"
```

---

### Task 9: Update .env.example and documentation

**Files:**
- Modify: `.env.example`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Add new settings to `.env.example`**

```env
# Chunk sizing (for A/B testing)
CHUNK_TEXT_TARGET=400
CHUNK_TEXT_MAX=500
CHUNK_TEXT_OVERLAP=50
CHUNK_TABLE_MAX=450

# LLM enrichment at index time (summary, keywords, medical entities)
ENRICH_CHUNKS_WITH_LLM=false
```

**Step 2: Update `docs/architecture/current-state.md`**

Add to the RAG ingest pipeline section:
- Chunker now produces schema-aware linearized embedding text for tables
- Chunks include cross-reference metadata, chunk position, and optionally LLM-generated context summaries, keywords, and medical entities
- Chunk config is externalized to env vars for A/B testing

**Step 3: Append to `docs/changes/implementation-log.md`**

Entry: "Chunk enrichment & table linearization" covering what changed, why, tradeoff, and invariants.

**Step 4: Commit**

```bash
git add .env.example docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: chunk enrichment settings, architecture, and implementation log"
```

---

### Task 10: Run full test suite and verify

**Step 1: Run all tests**

```bash
python -m pytest tests/ -x -q --timeout=30
```
Expected: All pass

**Step 2: Verify reindex works without LLM enrichment (default)**

```bash
python scripts/reindex_pdf.py
```
Expected: Completes without errors, chunks have linearized table embedding text, cross-references, chunk positions.

**Step 3: (Optional) Test with LLM enrichment**

Set `ENRICH_CHUNKS_WITH_LLM=true` in `.env`, then:
```bash
python scripts/reindex_pdf.py
```
Expected: Each chunk gets context_summary, keywords, medical_entities in metadata.

---

## Execution order and dependencies

```
Task 1 (settings) ──┐
                     ├── Task 2 (table linearization) ── Task 3 (table metadata)
                     ├── Task 4 (chunk position)
                     ├── Task 5 (cross-references)
                     │
                     ├── Task 6 (enrichment module) ── Task 7 (wire into ingest)
                     │
                     └── Task 8 (embedding bug fix)

All tasks ──── Task 9 (docs) ──── Task 10 (verify)
```

Tasks 2-6 and 8 are independent of each other and can be parallelized.
