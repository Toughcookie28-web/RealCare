# Image Chunking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** During PDF ingest, extract images (figures, diagrams, charts) and generate text captions via a vision LLM. Store captions as searchable text chunks with `chunk_type='image_caption'` and the image path in metadata, so image content becomes retrievable alongside text.

**Architecture:** Option B (caption approach). Images never get vector-embedded directly — instead a vision LLM (GPT-4V or Gemini Vision) describes each image during ingest. The caption is stored as a regular `DocumentChunkModel` row and embedded with the existing text embedding pipeline. The `image_ref` field in `metadata_json` stores the path to the extracted image file for UI rendering. Controlled by `ENABLE_IMAGE_CHUNKING=false` (default off) so existing ingest is unaffected.

**Tech Stack:** Python, Docling (already used for PDF parsing), OpenAI/Gemini vision API, existing vector store pipeline

---

### Task 1: Add image chunking settings

**Files:**
- Modify: `core/settings.py`
- Modify: `tests/core/test_settings_contract.py`

**Step 1: Write failing test**

```python
def test_image_chunking_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.enable_image_chunking is False
    assert s.image_caption_provider == "openai"
    assert s.image_output_dir == "/tmp/medigenius-images"
    assert s.image_min_size_px == 100
```

**Step 2: Run to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v -k "image" 2>&1`
Expected: FAIL

**Step 3: Add settings**

In `core/settings.py`, after `enable_parent_child_chunking` block:
```python
enable_image_chunking: bool = Field(default=False, alias='ENABLE_IMAGE_CHUNKING')
image_caption_provider: str = Field(default='openai', alias='IMAGE_CAPTION_PROVIDER')
image_output_dir: str = Field(default='/tmp/medigenius-images', alias='IMAGE_OUTPUT_DIR')
image_min_size_px: int = Field(default=100, alias='IMAGE_MIN_SIZE_PX')
```

**Step 4: Verify**

Run: `pytest tests/core/test_settings_contract.py -v -k "image" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings_contract.py
git commit -m "feat: add image chunking settings (disabled by default)"
```

---

### Task 2: Create tools/image_extractor.py

**Files:**
- Create: `tools/image_extractor.py`
- Create: `tests/rag/test_image_chunking.py`

This module extracts images from a parsed Docling document and saves them to disk.

**Step 1: Write failing tests**

Create `tests/rag/test_image_chunking.py`:
```python
from unittest.mock import MagicMock, patch
import os


def test_extract_images_returns_empty_for_no_images():
    from tools.image_extractor import extract_images_from_parsed_doc

    mock_parsed_doc = MagicMock()
    mock_parsed_doc.doc_id = "test-doc"
    mock_parsed_doc.elements = []  # no elements

    result = extract_images_from_parsed_doc(mock_parsed_doc, output_dir="/tmp/test-imgs")
    assert result == []


def test_extract_images_metadata_shape():
    from tools.image_extractor import ImageRecord
    record = ImageRecord(
        image_path="/tmp/test.png",
        doc_id="doc1",
        page=3,
        section="Pharmacology",
        image_index=0,
    )
    assert record.image_path.endswith(".png")
    assert record.page == 3


def test_extract_images_skips_tiny_images():
    """Images below min_size_px threshold should be skipped."""
    from tools.image_extractor import _is_image_large_enough
    assert _is_image_large_enough(width=50, height=50, min_size=100) is False
    assert _is_image_large_enough(width=200, height=150, min_size=100) is True
```

**Step 2: Run to verify they fail**

Run: `pytest tests/rag/test_image_chunking.py -v 2>&1`
Expected: FAIL — module not found

**Step 3: Create tools/image_extractor.py**

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Metadata for a single extracted image."""
    image_path: str
    doc_id: str
    page: int | None
    section: str | None
    image_index: int


def _is_image_large_enough(width: int, height: int, min_size: int) -> bool:
    """Filter out tiny decorative images, logos, etc."""
    return width >= min_size and height >= min_size


def extract_images_from_parsed_doc(
    parsed_doc: "ParsedDocument",
    output_dir: str,
    min_size_px: int = 100,
) -> list[ImageRecord]:
    """
    Extract images from a Docling ParsedDocument and save to output_dir.

    Returns a list of ImageRecord objects — one per extracted image.
    Images smaller than min_size_px in either dimension are skipped.
    Returns empty list if no images found or image extraction not supported.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []

    # Docling exposes images via element.image or similar — check element type
    for i, element in enumerate(getattr(parsed_doc, 'elements', [])):
        # Docling element types: 'text', 'table', 'picture', 'figure'
        el_type = getattr(element, 'element_type', None) or getattr(element, 'type', None)
        if el_type not in ('picture', 'figure', 'image'):
            continue

        image_data = getattr(element, 'image', None)
        if image_data is None:
            continue

        # Get dimensions if available
        width = getattr(image_data, 'width', 0) or 0
        height = getattr(image_data, 'height', 0) or 0
        if not _is_image_large_enough(width, height, min_size_px):
            logger.debug("Skipping small image %dx%d at element %d", width, height, i)
            continue

        # Save image to disk
        filename = f"{parsed_doc.doc_id}_img_{i:04d}.png"
        img_path = output_path / filename

        try:
            # Docling may provide bytes or a PIL image
            if hasattr(image_data, 'pil_image'):
                image_data.pil_image.save(str(img_path))
            elif hasattr(image_data, 'as_bytes'):
                img_path.write_bytes(image_data.as_bytes())
            else:
                logger.warning("Unknown image format at element %d, skipping", i)
                continue
        except Exception:
            logger.warning("Failed to save image at element %d", i, exc_info=True)
            continue

        page = getattr(element, 'page', None)
        section = getattr(element, 'section', None) or getattr(element, 'heading', None)

        records.append(ImageRecord(
            image_path=str(img_path),
            doc_id=parsed_doc.doc_id,
            page=page,
            section=str(section) if section else None,
            image_index=i,
        ))

    logger.info(
        "image_extraction_complete",
        extra={"doc_id": parsed_doc.doc_id, "images_extracted": len(records)},
    )
    return records
```

**Step 4: Run tests**

Run: `pytest tests/rag/test_image_chunking.py -v 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/image_extractor.py tests/rag/test_image_chunking.py
git commit -m "feat: add image extractor for PDF figures"
```

---

### Task 3: Create tools/image_captioner.py

**Files:**
- Create: `tools/image_captioner.py`
- Modify: `tests/rag/test_image_chunking.py`

**Step 1: Write failing tests**

Append to `tests/rag/test_image_chunking.py`:
```python
def test_generate_caption_returns_string(monkeypatch):
    from tools.image_captioner import generate_caption

    monkeypatch.setattr(
        "tools.image_captioner._caption_via_openai",
        lambda path, api_key: "Figure showing dose-response curve for metformin.",
    )

    result = generate_caption("/tmp/fake_img.png", provider="openai", api_key="fake-key")
    assert isinstance(result, str)
    assert len(result) > 10


def test_generate_caption_falls_back_on_error(monkeypatch):
    from tools.image_captioner import generate_caption

    monkeypatch.setattr(
        "tools.image_captioner._caption_via_openai",
        lambda path, api_key: (_ for _ in ()).throw(RuntimeError("API error")),
    )

    result = generate_caption("/tmp/fake_img.png", provider="openai", api_key="fake-key")
    # On error, returns empty string (caller decides whether to skip)
    assert result == ""
```

**Step 2: Run to verify they fail**

Run: `pytest tests/rag/test_image_chunking.py -v -k "caption" 2>&1`
Expected: FAIL

**Step 3: Create tools/image_captioner.py**

```python
from __future__ import annotations

import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "You are a medical education assistant. Describe this medical figure or diagram "
    "in 2-4 sentences. Focus on: what it shows, key labels or values visible, "
    "and its medical relevance. Be factual and specific. "
    "Start with 'Figure showing' or 'Diagram of'."
)


def generate_caption(
    image_path: str,
    provider: str = "openai",
    api_key: str | None = None,
) -> str:
    """
    Generate a text caption for a medical image using a vision LLM.

    Returns the caption string, or "" on failure.
    The caller decides whether to skip or use a placeholder on empty return.
    """
    try:
        if provider == "openai":
            return _caption_via_openai(image_path, api_key)
        elif provider == "gemini":
            return _caption_via_gemini(image_path, api_key)
        else:
            logger.warning("Unknown image caption provider: %s", provider)
            return ""
    except Exception:
        logger.warning("Image captioning failed for %s", image_path, exc_info=True)
        return ""


def _load_image_base64(image_path: str) -> str:
    """Load image file and return base64-encoded string."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _caption_via_openai(image_path: str, api_key: str | None) -> str:
    """Caption via OpenAI GPT-4 Vision."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    b64 = _load_image_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _caption_via_gemini(image_path: str, api_key: str | None) -> str:
    """Caption via Google Gemini Vision."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = Image.open(image_path)
    response = model.generate_content([_CAPTION_PROMPT, img])
    return response.text.strip()
```

**Step 4: Run tests**

Run: `pytest tests/rag/test_image_chunking.py -v -k "caption" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/image_captioner.py tests/rag/test_image_chunking.py
git commit -m "feat: add image captioner using vision LLM (OpenAI/Gemini)"
```

---

### Task 4: Integrate into PDF ingest pipeline (pdf_loader.py / vector_store.py)

**Files:**
- Modify: `tools/pdf_loader.py`
- Modify: `tests/rag/test_image_chunking.py`

**Step 1: Write failing test**

Append to `tests/rag/test_image_chunking.py`:
```python
def test_build_image_caption_chunks_creates_documents():
    from tools.pdf_loader import build_image_caption_chunks
    from tools.image_extractor import ImageRecord

    records = [
        ImageRecord(
            image_path="/tmp/doc1_img_0001.png",
            doc_id="doc1",
            page=5,
            section="Pharmacology",
            image_index=1,
        )
    ]
    captions = {"1": "Figure showing the mechanism of aspirin on COX enzymes."}

    docs = build_image_caption_chunks(records, captions)

    assert len(docs) == 1
    assert docs[0].page_content == "Figure showing the mechanism of aspirin on COX enzymes."
    assert docs[0].metadata["chunk_type"] == "image_caption"
    assert docs[0].metadata["image_ref"] == "/tmp/doc1_img_0001.png"
    assert docs[0].metadata["page"] == 5
    assert docs[0].metadata["section"] == "Pharmacology"


def test_build_image_caption_chunks_skips_empty_captions():
    from tools.pdf_loader import build_image_caption_chunks
    from tools.image_extractor import ImageRecord

    records = [
        ImageRecord(image_path="/tmp/img.png", doc_id="d1", page=1, section="S1", image_index=0)
    ]
    docs = build_image_caption_chunks(records, {"0": ""})  # empty caption
    assert len(docs) == 0  # skipped
```

**Step 2: Run to verify they fail**

Run: `pytest tests/rag/test_image_chunking.py -v -k "caption_chunks" 2>&1`
Expected: FAIL

**Step 3: Add build_image_caption_chunks to pdf_loader.py**

In `tools/pdf_loader.py`, add:
```python
def build_image_caption_chunks(
    image_records: list,
    captions: dict[str, str],
) -> list[Document]:
    """
    Convert extracted image records + captions into Document objects
    for ingestion into the vector store.

    captions: dict mapping str(image_index) -> caption text
    Skips images with empty captions.
    """
    docs = []
    for record in image_records:
        caption = captions.get(str(record.image_index), "").strip()
        if not caption:
            continue
        chunk_id = f"{record.doc_id}_img_{record.image_index:04d}"
        docs.append(Document(
            page_content=caption,
            metadata={
                "chunk_id": chunk_id,
                "doc_id": record.doc_id,
                "chunk_type": "image_caption",
                "image_ref": record.image_path,
                "page": record.page,
                "section": record.section,
            },
        ))
    return docs
```

**Step 4: Update process_pdf to optionally extract images**

In `tools/pdf_loader.py`, update `process_pdf`:
```python
def process_pdf(pdf_path: str, include_images: bool = False) -> list[Document]:
    from core.settings import get_settings
    settings = get_settings()

    path = _resolve_pdf_path(pdf_path)
    parsed_doc = load_parsed_pdf(str(path))
    chunks = chunk_pdf_document(parsed_doc)

    if include_images or settings.enable_image_chunking:
        from tools.image_extractor import extract_images_from_parsed_doc
        from tools.image_captioner import generate_caption

        image_records = extract_images_from_parsed_doc(
            parsed_doc,
            output_dir=settings.image_output_dir,
            min_size_px=settings.image_min_size_px,
        )
        if image_records:
            api_key = settings.openai_api_key
            captions = {
                str(r.image_index): generate_caption(
                    r.image_path,
                    provider=settings.image_caption_provider,
                    api_key=api_key,
                )
                for r in image_records
            }
            caption_chunks = build_image_caption_chunks(image_records, captions)
            chunks = chunks + caption_chunks
            logger.info(
                "image_chunks_added",
                extra={"doc_id": parsed_doc.doc_id, "image_chunks": len(caption_chunks)},
            )

    return chunks
```

**Step 5: Run all image tests**

Run: `pytest tests/rag/test_image_chunking.py -v 2>&1`
Expected: all PASS

**Step 6: Commit**

```bash
git add tools/pdf_loader.py tests/rag/test_image_chunking.py
git commit -m "feat: integrate image captioning into PDF ingest pipeline"
```

---

### Task 5: Update ExecutorAgent to render image references

**Files:**
- Modify: `agents/executor_agent.py`
- Modify: `tests/rag/test_image_chunking.py`

When a retrieved chunk has `chunk_type='image_caption'`, the executor should include the image reference in the response so the UI can render it.

**Step 1: Write failing test**

Append to `tests/rag/test_image_chunking.py`:
```python
def test_build_context_blocks_includes_image_ref():
    from agents.executor_agent import _build_context_blocks
    from langchain_core.documents import Document

    doc = Document(
        page_content="Figure showing aspirin COX inhibition pathway.",
        metadata={
            "chunk_id": "doc1_img_0001",
            "chunk_type": "image_caption",
            "image_ref": "/tmp/doc1_img_0001.png",
            "section": "Pharmacology",
        },
    )
    blocks = _build_context_blocks([doc])
    assert len(blocks) == 1
    assert "image_ref" in blocks[0] or "/tmp/doc1_img_0001.png" in blocks[0]
```

**Step 2: Run to verify it fails**

Run: `pytest tests/rag/test_image_chunking.py -v -k "image_ref" 2>&1`
Expected: FAIL

**Step 3: Update _build_context_blocks in executor_agent.py**

In the `parts` assembly section of `_build_context_blocks`, after building the `<chunk>` block, add image rendering:
```python
image_ref = metadata.get("image_ref")
if image_ref:
    parts.append(f"[Image: {image_ref}]")
```

**Step 4: Run test**

Run: `pytest tests/rag/test_image_chunking.py -v -k "image_ref" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add agents/executor_agent.py tests/rag/test_image_chunking.py
git commit -m "feat: executor includes image references in context blocks"
```

---

### Task 6: Update docs

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

Architecture note: "Image chunking (ENABLE_IMAGE_CHUNKING=false by default). When enabled, Docling extracts figures from PDFs. A vision LLM (GPT-4V or Gemini) captions each image during ingest. Captions stored as `chunk_type='image_caption'` chunks with `image_ref` in metadata. Enables semantic search over diagram/figure content."

Log entry: `2026-03-17 — Image chunking pipeline added (flag off by default).`
