import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_pdf_chunker_module():
    documents_module = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents_module.Document = Document

    pdf_parser_module = types.ModuleType("tools.pdf_parser")

    class ElementType(Enum):
        HEADING = "heading"
        PARAGRAPH = "paragraph"
        LIST_ITEM = "list_item"
        TABLE = "table"
        CAPTION = "caption"
        PAGE_HEADER = "page_header"
        PAGE_FOOTER = "page_footer"
        FIGURE = "figure"

    class ParsedDocument:
        pass

    class ParsedElement:
        pass

    pdf_parser_module.ElementType = ElementType
    pdf_parser_module.ParsedDocument = ParsedDocument
    pdf_parser_module.ParsedElement = ParsedElement

    original_modules = {
        "langchain_core.documents": sys.modules.get("langchain_core.documents"),
        "tools.pdf_parser": sys.modules.get("tools.pdf_parser"),
    }
    sys.modules["langchain_core.documents"] = documents_module
    sys.modules["tools.pdf_parser"] = pdf_parser_module

    try:
        spec = importlib.util.spec_from_file_location(
            "pdf_chunker_contract",
            ROOT / "tools" / "pdf_chunker.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["pdf_chunker_contract"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("pdf_chunker_contract", None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_chunk_config_reads_from_settings():
    from core.settings import get_settings
    settings = get_settings()
    assert hasattr(settings, 'chunk_text_target')
    assert hasattr(settings, 'chunk_text_max')
    assert hasattr(settings, 'chunk_text_overlap')
    assert hasattr(settings, 'chunk_table_max')
    assert settings.chunk_text_target == 400
    assert settings.chunk_text_max == 500


def test_split_long_text_hard_wraps_when_structural_splitters_do_not_progress():
    module = _load_pdf_chunker_module()
    text = " ".join(f"token{i}" for i in range(600))

    chunks = module._split_long_text(text, max_tokens=100)

    assert len(chunks) > 1
    assert all(module._count_tokens(chunk) <= 100 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == text


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
    assert "|" not in result


def test_linearize_table_with_no_separator_returns_none():
    module = _load_pdf_chunker_module()
    result = module._linearize_markdown_table("just some text")
    assert result is None


def test_extract_cross_references():
    module = _load_pdf_chunker_module()
    text = "See Chapter 8 for details. Refer to Table 5.2 and Figure 3.1 for illustration. As discussed in Section 2.4."
    refs = module._extract_cross_references(text)
    types_found = {r["type"] for r in refs}
    assert "chapter" in types_found
    assert "table" in types_found
    assert "figure" in types_found
    assert "section" in types_found
    chapter_ref = next(r for r in refs if r["type"] == "chapter")
    assert chapter_ref["normalized"] == "chapter_8"
    table_ref = next(r for r in refs if r["type"] == "table")
    assert table_ref["normalized"] == "table_5_2"
    figure_ref = next(r for r in refs if r["type"] == "figure")
    assert figure_ref["normalized"] == "figure_3_1"


def test_extract_cross_references_empty():
    module = _load_pdf_chunker_module()
    refs = module._extract_cross_references("No references here at all.")
    assert refs == []


def test_table_chunk_metadata_includes_columns_and_row_count():
    from tools.pdf_parser import ParsedDocument, ParsedElement

    module = _load_pdf_chunker_module()
    # Use the chunker module's ElementType (from the stub) for correct enum comparison
    StubElementType = module.ElementType

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
            ParsedElement(element_type=StubElementType.TABLE, text=table_md,
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


def test_text_chunk_metadata_includes_position():
    from tools.pdf_parser import ParsedDocument, ParsedElement

    module = _load_pdf_chunker_module()
    StubElementType = module.ElementType

    elements = [
        ParsedElement(element_type=StubElementType.HEADING, text="Section A",
                     heading_level=1, section_path=["Section A"], page=1, element_index=0),
        ParsedElement(element_type=StubElementType.PARAGRAPH, text="First paragraph content. " * 25,
                     page=1, section_path=["Section A"], element_index=1),
        ParsedElement(element_type=StubElementType.PARAGRAPH, text="Middle paragraph content. " * 25,
                     page=1, section_path=["Section A"], element_index=2),
        ParsedElement(element_type=StubElementType.PARAGRAPH, text="Last paragraph content. " * 25,
                     page=2, section_path=["Section A"], element_index=3),
    ]
    parsed = ParsedDocument(doc_id="test-doc", title="Test Book", elements=elements)
    config = module.ChunkConfig(text_chunk_target=30, text_chunk_max=40, text_chunk_overlap=0)
    chunks = module.chunk_parsed_document(parsed, config=config)

    assert len(chunks) >= 2
    positions = [c.metadata.get("chunk_position") for c in chunks]
    assert positions[0] == "intro"
    assert positions[-1] == "conclusion"
    if len(positions) > 2:
        assert all(p == "body" for p in positions[1:-1])

