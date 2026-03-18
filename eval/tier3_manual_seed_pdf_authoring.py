"""PDF-first helpers for Tier 3 manual-seed authoring.

These utilities are thin wrappers around the shared manual article-window
tooling. They preserve the manual-seed entrypoint name while keeping PDF
extraction and chunk-alignment logic reusable.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.tier3_article_documents import (
    ChunkAlignment,
    ArticleWindow as PdfArticleWindow,
    align_article_windows_to_chunks,
    build_article_windows,
    extract_pdf_pages,
    load_chunk_corpus_rows,
)


def summarize_alignment(aligned_rows: list[ChunkAlignment]) -> dict[str, int]:
    aligned_count = sum(1 for row in aligned_rows if row.chunk_ids)
    return {
        "total_windows": len(aligned_rows),
        "aligned_windows": aligned_count,
        "unaligned_windows": len(aligned_rows) - aligned_count,
    }
