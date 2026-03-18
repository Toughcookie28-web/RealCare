from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from db.repositories import InMemoryVectorRepository, VectorRepository
from core.settings import get_settings
from tools.embedding_client import embed_documents_batch, embed_query
from tools.ingest_report import IngestReport
from tools.pdf_loader import chunk_pdf_document, load_parsed_pdf

logger = logging.getLogger(__name__)


def _apply_enrichment_to_chunks(
    chunks: list[Any],
    enrichments: list[dict[str, Any]],
) -> None:
    """Merge LLM enrichment results into chunk metadata and embedding_text."""
    if len(enrichments) != len(chunks):
        logger.warning(
            "enrichment_count_mismatch",
            extra={"chunks": len(chunks), "enrichments": len(enrichments)},
        )
    for chunk, enrichment in zip(chunks, enrichments):
        meta = chunk.metadata
        summary = enrichment.get("context_summary", "")
        meta["context_summary"] = summary

        # Prepend context summary to embedding text
        if summary:
            existing_embedding = meta.get("embedding_text", "")
            prefix = f"[Context: {summary}]"
            if existing_embedding:
                meta["embedding_text"] = f"{prefix}\n{existing_embedding}"
            else:
                embedding_prefix = meta.get("embedding_prefix", "")
                base = f"{embedding_prefix}\n{chunk.page_content}" if embedding_prefix else chunk.page_content
                meta["embedding_text"] = f"{prefix}\n{base}"


def build_embedding_texts(chunks: list[Any]) -> list[str]:
    embedding_texts: list[str] = []
    for c in chunks:
        metadata = c.metadata or {}
        explicit_embedding_text = metadata.get('embedding_text')
        if explicit_embedding_text:
            embedding_texts.append(explicit_embedding_text)
            continue

        prefix = metadata.get('embedding_prefix', '')
        raw = c.page_content
        embedding_texts.append(f"{prefix}\n{raw}" if prefix else raw)
    return embedding_texts


def embed_chunk_documents(chunks: list[Any]) -> list[list[float] | None]:
    """Embed chunks, skipping parent chunks (they don't need embeddings)."""
    child_indices: list[int] = []
    child_texts: list[str] = []

    all_texts = build_embedding_texts(chunks)
    for i, chunk in enumerate(chunks):
        if (chunk.metadata or {}).get("chunk_type") == "parent":
            continue
        child_indices.append(i)
        child_texts.append(all_texts[i])

    if not child_texts:
        return [None] * len(chunks)

    child_embeddings = embed_documents_batch(child_texts)
    logger.info('ingest_embed_complete', extra={'chunks': len(chunks), 'children_embedded': len(child_embeddings)})

    result: list[list[float] | None] = [None] * len(chunks)
    for idx, emb in zip(child_indices, child_embeddings):
        result[idx] = emb
    return result


def build_index_batch(chunks: list[Any], embeddings: list[list[float] | None], default_doc_id: str) -> list[dict[str, Any]]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            'Chunk and embedding counts must match before indexing. '
            f'chunks={len(chunks)} embeddings={len(embeddings)}'
        )

    batch: list[dict[str, Any]] = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        metadata = chunk.metadata or {}
        chunk_id = metadata.get('chunk_id') or metadata.get('child_id', f'chunk-{idx}')
        batch.append({
            'chunk_id': chunk_id,
            'doc_id': metadata.get('doc_id', default_doc_id),
            'content': chunk.page_content,
            'embedding': embedding,
            'page': metadata.get('page'),
            'section': metadata.get('section_path') or metadata.get('section'),
            'metadata': metadata,
        })
    return batch


def index_chunk_batch(
    repo: VectorRepository | InMemoryVectorRepository,
    batch: list[dict[str, Any]],
    doc_id: str,
) -> tuple[int, int]:
    if not batch:
        return 0, repo.count_chunks()

    inserted = repo.upsert_chunks_batch(batch, replace_doc_id=doc_id, batch_size=250)
    count = repo.count_chunks()
    logger.info(
        'ingest_index_complete',
        extra={'node': 'ingestion', 'doc_id': doc_id, 'inserted': inserted, 'total': count},
    )
    return inserted, count


def ingest_pdf_to_vector_store(db: Session | None, pdf_path: str) -> IngestReport:
    parsed_doc = load_parsed_pdf(pdf_path)
    chunks = chunk_pdf_document(parsed_doc)
    doc_id = parsed_doc.doc_id or Path(pdf_path).stem
    repo = VectorRepository(db) if db is not None else InMemoryVectorRepository()

    logger.info(
        'ingest_parse_complete',
        extra={'doc_id': doc_id, 'parsed_elements': len(parsed_doc.elements)},
    )
    logger.info(
        'ingest_chunk_complete',
        extra={'doc_id': doc_id, 'chunks': len(chunks)},
    )

    # Optional LLM-based chunk enrichment
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

    # Optional parent-child hierarchy
    if get_settings().enable_parent_child_chunking and chunks:
        from tools.pdf_chunker import group_children_into_parents
        parents, chunks = group_children_into_parents(
            chunks,
            max_children_per_parent=get_settings().chunk_parent_max_children,
        )
        all_chunks = parents + chunks  # parents first, then children
        logger.info("ingest_hierarchy_complete", extra={
            "doc_id": doc_id, "parents": len(parents), "children": len(chunks),
        })
    else:
        all_chunks = chunks

    if not all_chunks:
        return IngestReport(
            doc_id=doc_id,
            parsed_elements=len(parsed_doc.elements),
            chunks=0,
            embeddings=0,
            inserted=0,
            total_chunks=repo.count_chunks(),
        )

    embeddings = embed_chunk_documents(all_chunks)
    batch = build_index_batch(all_chunks, embeddings, default_doc_id=doc_id)
    inserted, total_chunks = index_chunk_batch(repo, batch, doc_id=doc_id)
    report = IngestReport(
        doc_id=doc_id,
        parsed_elements=len(parsed_doc.elements),
        chunks=len(all_chunks),
        embeddings=len([e for e in embeddings if e is not None]),
        inserted=inserted,
        total_chunks=total_chunks,
    )
    logger.info(
        'vector_store_ingestion_complete',
        extra={
            'node': 'ingestion',
            'doc_id': report.doc_id,
            'parsed_elements': report.parsed_elements,
            'chunks': report.chunks,
            'embeddings': report.embeddings,
            'inserted': report.inserted,
            'total': report.total_chunks,
        },
    )
    return report


def hybrid_retrieve(repo: VectorRepository | InMemoryVectorRepository, query: str, k: int = 12):
    embedding = embed_query(query)
    return repo.hybrid_search(query=query, query_embedding=embedding, k=k)
