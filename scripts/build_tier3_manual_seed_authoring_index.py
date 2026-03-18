from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.tier3_manual_seed_pdf_authoring import (
    align_article_windows_to_chunks,
    build_article_windows,
    extract_pdf_pages,
    load_chunk_corpus_rows,
    summarize_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the PDF-first Tier 3 manual-seed authoring index."
    )
    parser.add_argument("--pdf", default="data/medical_book.pdf", help="PDF path")
    parser.add_argument(
        "--chunk-corpus",
        default="eval/golden/v1/tier3_chunk_corpus.jsonl",
        help="Chunk corpus JSONL path",
    )
    parser.add_argument(
        "--output",
        default="eval/golden/v1/tier3_manual_seed_authoring_index.jsonl",
        help="Output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_rows = load_chunk_corpus_rows(Path(args.chunk_corpus))
    page_numbers = sorted({int(row["page"]) for row in chunk_rows})
    parsed_pages = extract_pdf_pages(Path(args.pdf), min(page_numbers), max(page_numbers))
    windows = build_article_windows(parsed_pages)
    aligned = align_article_windows_to_chunks(windows, chunk_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for window, alignment in zip(windows, aligned):
        row = {
            "article_title": window.article_title,
            "page_start": window.page_start,
            "page_end": window.page_end,
            "local_headings": window.local_headings,
            "body_preview": window.body_text[:400],
            "chunk_ids": alignment.chunk_ids,
            "chunk_pages": alignment.chunk_pages,
            "chunk_sections": alignment.chunk_sections,
        }
        lines.append(json.dumps(row, ensure_ascii=True))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = summarize_alignment(aligned)
    print(
        f"wrote {len(lines)} windows to {output_path} "
        f"({summary['aligned_windows']} aligned / {summary['total_windows']} total)"
    )


if __name__ == "__main__":
    main()
