"""Article-window extraction for Tier 3 manual-seed authoring.

This module recovers article/topic entries from the PDF and optionally aligns
those windows back to the frozen retrieval chunk corpus. It is the active
document-structure layer for the manual-seed workflow only.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


STANDARD_HEADINGS = {
    "Definition",
    "Description",
    "Causes and symptoms",
    "Causes",
    "Symptoms",
    "Diagnosis",
    "Treatment",
    "Prognosis",
    "Prevention",
    "Resources",
    "KEY TERMS",
    "Key terms",
    "Demographics",
    "Genetic profile",
    "Signs and symptoms",
    "When to call the doctor",
    "Questions to ask the doctor",
}
BACK_MATTER_MARKERS = {
    "BOOKS",
    "PERIODICALS",
    "ORGANIZATIONS",
    "OTHER",
}
FOOTER_PREFIX = "GALE ENCYCLOPEDIA OF MEDICINE"


@dataclass(eq=True)
class ArticleWindow:
    article_title: str
    page_start: int
    page_end: int
    local_headings: list[str]
    body_text: str


@dataclass(eq=True)
class ChunkAlignment:
    article_title: str
    page_span: tuple[int, int]
    chunk_ids: list[str]
    chunk_pages: list[int]
    chunk_sections: list[str]
    local_headings: list[str]


def load_chunk_corpus_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        rows.append(
            {
                "chunk_id": row["chunk_id"],
                "page": int(row["page"]),
                "section": row.get("section") or metadata.get("section"),
                "content": row.get("content", ""),
                "metadata": metadata,
            }
        )
    return rows


def extract_pdf_pages(pdf_path: Path, page_start: int, page_end: int) -> list[dict]:
    cmd = [
        "mutool",
        "draw",
        "-F",
        "txt",
        "-o",
        "-",
        str(pdf_path),
        f"{page_start}-{page_end}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw_pages = result.stdout.split("\f")
    texts = [page.strip("\n") for page in raw_pages if page.strip()]
    return [
        {"page": page_start + index, "text": text}
        for index, text in enumerate(texts)
    ]


def build_article_windows(parsed_pages: list[dict]) -> list[ArticleWindow]:
    windows: list[ArticleWindow] = []
    current_title: str | None = None

    for page_row in sorted(parsed_pages, key=lambda row: row["page"]):
        lines = _clean_lines(page_row["text"])
        if not lines:
            continue

        start_title = _extract_start_title(lines)
        footer_title = _extract_footer_title(lines)
        article_title = start_title or footer_title or current_title
        if not article_title:
            continue

        headings = _extract_headings(lines)
        body_text = _extract_body_text(
            lines,
            article_title,
            truncate_before_title=bool(start_title),
        )
        if not body_text and not headings:
            continue

        if (
            windows
            and windows[-1].article_title == article_title
            and page_row["page"] == windows[-1].page_end + 1
        ):
            windows[-1].page_end = page_row["page"]
            windows[-1].local_headings = _merge_preserving_order(
                windows[-1].local_headings,
                headings,
            )
            if body_text:
                windows[-1].body_text = _merge_body_text(windows[-1].body_text, body_text)
        else:
            windows.append(
                ArticleWindow(
                    article_title=article_title,
                    page_start=page_row["page"],
                    page_end=page_row["page"],
                    local_headings=headings,
                    body_text=body_text,
                )
            )
        current_title = article_title

    return windows

def align_article_windows_to_chunks(
    windows: list[ArticleWindow],
    chunk_rows: list[dict],
    *,
    max_chunks: int = 8,
) -> list[ChunkAlignment]:
    aligned: list[ChunkAlignment] = []
    for window in windows:
        scored = []
        title_tokens = _token_set(window.article_title)
        heading_tokens = {_normalize_text(heading) for heading in window.local_headings}
        body_tokens = _token_set(window.body_text)

        for row in chunk_rows:
            page = int(row["page"])
            metadata = row.get("metadata") or {}
            section = row.get("section") or metadata.get("section") or ""
            section_path = metadata.get("section_path") or section
            context_prefix = metadata.get("context_prefix") or ""
            chunk_text = " ".join(
                part
                for part in [row.get("content", ""), section_path, context_prefix]
                if part
            )

            page_score = _page_score(page, window.page_start, window.page_end)
            if page_score == 0:
                continue

            score = page_score
            normalized_section = _normalize_text(section)
            normalized_path = _normalize_text(section_path)
            if normalized_section in heading_tokens or normalized_path in heading_tokens:
                score += 3

            chunk_tokens = _token_set(chunk_text)
            if title_tokens and title_tokens.intersection(chunk_tokens):
                score += 2

            if body_tokens and len(body_tokens.intersection(chunk_tokens)) >= 2:
                score += 1

            scored.append((score, page, row["chunk_id"], section or section_path))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = scored[:max_chunks]
        aligned.append(
            ChunkAlignment(
                article_title=window.article_title,
                page_span=(window.page_start, window.page_end),
                chunk_ids=[item[2] for item in selected],
                chunk_pages=[item[1] for item in selected],
                chunk_sections=[item[3] for item in selected],
                local_headings=list(window.local_headings),
            )
        )

    return aligned


def _clean_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.replace("\x0c", "\n").splitlines():
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)
    return lines


def _extract_start_title(lines: list[str]) -> str | None:
    for index, line in enumerate(lines[:40]):
        if not _looks_like_title(line):
            continue
        if index + 1 >= len(lines):
            break
        next_line = lines[index + 1]
        if next_line in STANDARD_HEADINGS:
            return line
    return None


def _extract_footer_title(lines: list[str]) -> str | None:
    if len(lines) < 2:
        return None
    candidate = lines[-1]
    prefix_window = lines[-3:-1]
    has_footer_marker = any(line.isdigit() or line.startswith(FOOTER_PREFIX) for line in prefix_window)
    if has_footer_marker and _looks_like_title(candidate):
        return candidate
    return None


def _extract_headings(lines: list[str]) -> list[str]:
    return _merge_preserving_order([], [line for line in lines if line in STANDARD_HEADINGS])


def _extract_body_text(
    lines: list[str],
    article_title: str,
    *,
    truncate_before_title: bool = False,
) -> str:
    candidate_lines = list(lines)
    if truncate_before_title:
        try:
            title_index = candidate_lines.index(article_title)
        except ValueError:
            title_index = -1
        if title_index >= 0:
            candidate_lines = candidate_lines[title_index + 1 :]

    body_lines: list[str] = []
    for line in candidate_lines:
        if line == article_title:
            continue
        if line in BACK_MATTER_MARKERS:
            break
        if line in STANDARD_HEADINGS:
            continue
        if line.isdigit() or line.startswith(FOOTER_PREFIX):
            continue
        body_lines.append(line)
    return "\n".join(body_lines)


def _merge_preserving_order(existing: list[str], new_items: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(existing)
    for item in new_items:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _merge_body_text(existing: str, new_text: str) -> str:
    if not existing:
        return new_text
    if not new_text:
        return existing
    return existing + "\n" + new_text


def _looks_like_title(line: str) -> bool:
    if not line or line in STANDARD_HEADINGS:
        return False
    if line.isdigit() or line.startswith(FOOTER_PREFIX):
        return False
    if len(line) > 90:
        return False
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    title_case = line[:1].isupper() and not line.endswith(".")
    return uppercase_ratio > 0.6 or title_case

def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _token_set(value: str) -> set[str]:
    return {token for token in _normalize_text(value).split() if len(token) >= 3}


def _page_score(page: int, page_start: int, page_end: int) -> int:
    if page_start <= page <= page_end:
        return 6
    if page == page_start - 1 or page == page_end + 1:
        return 2
    return 0
