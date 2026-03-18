from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestReport:
    doc_id: str
    parsed_elements: int
    chunks: int
    embeddings: int
    inserted: int
    total_chunks: int = 0
