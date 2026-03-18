"""
RAGAS-based offline utility for Tier 3 synthetic testset generation.

This command keeps a narrow contract:
- load or build article-level source documents from the medical book
- keep the frozen retrieval chunk corpus only for post-generation alignment
- run RAGAS on article-entry documents
- checkpoint the raw RAGAS rows
- convert the rows to schema_v1 draft samples

It is an offline utility for draft generation and review, not a regression gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence


GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "v1"
TIER3_PATH = GOLDEN_DIR / "tier3_rag.generated_draft.jsonl"
CHUNK_CORPUS_PATH = GOLDEN_DIR / "tier3_chunk_corpus.jsonl"
ARTICLE_CORPUS_PATH = GOLDEN_DIR / "tier3_article_corpus.jsonl"
RAW_RAGAS_ROWS_PATH = GOLDEN_DIR / "tier3_rag.raw.jsonl"
DEFAULT_PDF = Path(__file__).resolve().parent.parent / "data" / "medical_book.pdf"
DEFAULT_SIZE = 50

_DIFFICULTY_MAP = {
    "simple": "simple",
    "reasoning": "reasoning",
    "multi_context": "multi_context",
    "conditional": "reasoning",
}
_TEMP_EXPERIMENT_NAME = "temporary_cancer_family"
_TEMP_EXPERIMENT_SOURCE_TITLES = (
    "Cancer",
    "Cancer therapy, definitive",
    "Cancer therapy, palliative",
    "Cancer therapy, supportive",
)
_TEMP_EXPERIMENT_SOURCE_TITLE_SET = {title.lower() for title in _TEMP_EXPERIMENT_SOURCE_TITLES}
_TEMP_QUERY_PROFILE_NAME = "tier3_cancer_family_specific_only"
_TEMP_TRANSFORM_PROFILE_NAME = "tier3_cancer_family_document_graph"
_TEMP_BUNDLE_PROFILE_NAME = "tier3_cancer_family_bundle_docs"


@dataclass
class Tier3ConversionResult:
    samples: list[dict[str, Any]]
    raw_rows: list[dict[str, Any]]
    skipped_missing_question: int = 0
    skipped_empty_context: int = 0

    @property
    def accepted_rows(self) -> int:
        return len(self.samples)

    @property
    def raw_row_count(self) -> int:
        return len(self.raw_rows)


def _resolve_llm():
    from core.settings import get_settings
    from tools.llm_client import resolve_llm

    settings = get_settings()
    llm = resolve_llm(
        primary_provider=settings.tier3_llm_primary_provider,
        primary_model=settings.tier3_llm_primary_model,
        fallback_provider=settings.tier3_llm_fallback_provider,
        fallback_model=settings.tier3_llm_fallback_model,
    )
    if llm is not None:
        return llm
    raise RuntimeError(
        "No LLM available. Set Tier 3 LLM overrides or shared LLM provider credentials in your .env"
    )


class _ProjectEmbeddingsAdapter:
    """LangChain-compatible adapter backed by the project's embedding client."""

    def __init__(self, overrides: Any | None = None):
        from tools.embedding_client import embed_document, embed_query

        self._embed_document = embed_document
        self._embed_query = embed_query
        self._overrides = overrides

    def embed_query(self, text: str) -> list[float]:
        return self._embed_query(text, overrides=self._overrides)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_document(text, overrides=self._overrides) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _resolve_embeddings() -> tuple[Any, str]:
    from core.settings import get_settings
    from tools.embedding_client import EmbeddingOverrides

    settings = get_settings()
    provider = (settings.tier3_embedding_provider or settings.embedding_provider or "local").strip().lower()
    local_model = (settings.embedding_local_model or "").strip() or "unknown"
    gemini_model = (settings.embedding_gemini_model or "").strip() or "unknown"
    openai_model = (
        (settings.tier3_embedding_openai_model or settings.embedding_openai_model or "").strip() or "unknown"
    )
    dim = int(settings.tier3_embedding_dim or settings.embedding_dim)
    backup = "on" if settings.embedding_enable_backup else "off"
    model_label = (
        local_model
        if provider == "local"
        else (gemini_model if provider == "gemini" else openai_model)
    )
    label = (
        f"primary={provider}:{model_label} dim={dim} backup={backup}"
    )
    overrides = EmbeddingOverrides(
        provider=provider,
        dim=dim,
        openai_model=openai_model if provider == "openai" else None,
    )
    return _ProjectEmbeddingsAdapter(overrides=overrides), label


def _get_generator_model_name() -> str:
    try:
        from core.settings import get_settings

        settings = get_settings()
        provider = (settings.tier3_llm_primary_provider or settings.llm_primary_provider or "").strip().lower()
        model = (settings.tier3_llm_primary_model or settings.llm_primary_model or "").strip()
        if provider and model:
            return f"{provider}:{model}"
        if model:
            return model
    except Exception:
        pass
    return "unknown"


def _resolve_runtime_model_name(llm: Any) -> str:
    cls = llm.__class__.__name__.lower()
    provider = "groq" if "groq" in cls else ("openai" if "openai" in cls else "unknown")
    for attr in ("model_name", "model", "model_id"):
        value = getattr(llm, attr, None)
        if value:
            return f"{provider}:{value}"
    return _get_generator_model_name()


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _select_experiment_source_documents(docs: Sequence[Any]) -> list[Any]:
    selected = [
        doc
        for doc in docs
        if _clean_str((doc.metadata or {}).get("article_title")).lower() in _TEMP_EXPERIMENT_SOURCE_TITLE_SET
    ]
    return selected


def _require_experiment_source_documents(docs: Sequence[Any]) -> list[Any]:
    selected = _select_experiment_source_documents(docs)
    if len(selected) < 2:
        available_titles = sorted(
            {
                _clean_str((doc.metadata or {}).get("article_title"))
                for doc in docs
                if _clean_str((doc.metadata or {}).get("article_title"))
            }
        )
        raise RuntimeError(
            "Temporary cancer-family Tier 3 experiment requires at least two selected article docs. "
            f"Expected titles: {', '.join(_TEMP_EXPERIMENT_SOURCE_TITLES)}. "
            f"Available titles: {available_titles}"
        )
    return selected


def _dedupe_preserve_order(values: Sequence[Any]) -> list[Any]:
    seen = set()
    ordered: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _build_experiment_generation_documents(docs: Sequence[Any]) -> list[Any]:
    if not docs:
        return []

    docs_by_title = {
        _clean_str((doc.metadata or {}).get("article_title")): doc
        for doc in docs
    }
    bundle_specs = [
        ("cancer-core", ["Cancer"]),
        (
            "cancer-therapy",
            [
                "Cancer therapy, definitive",
                "Cancer therapy, palliative",
                "Cancer therapy, supportive",
            ],
        ),
    ]
    missing_titles = [
        title
        for _bundle_id, titles in bundle_specs
        for title in titles
        if title not in docs_by_title
    ]
    if missing_titles:
        raise RuntimeError(
            "Temporary cancer-family generation bundle is missing article docs: "
            + ", ".join(sorted(missing_titles))
        )

    doc_cls = docs[0].__class__
    bundled_docs: list[Any] = []
    for bundle_id, titles in bundle_specs:
        bundle_docs = [docs_by_title[title] for title in titles]
        page_content = "\n\n".join(
            f"# {_clean_str(doc.metadata.get('article_title'))}\n{doc.page_content}".strip()
            for doc in bundle_docs
        )
        bundle_metadata = {
            "source_doc_type": "tier3_generation_bundle",
            "bundle_id": bundle_id,
            "article_title": bundle_docs[0].metadata.get("article_title"),
            "bundle_article_titles": titles,
            "doc_id": _clean_str(bundle_docs[0].metadata.get("doc_id")),
            "page_start": min(int(doc.metadata.get("page_start") or 0) for doc in bundle_docs),
            "page_end": max(int(doc.metadata.get("page_end") or 0) for doc in bundle_docs),
            "local_headings": _dedupe_preserve_order(
                [
                    heading
                    for doc in bundle_docs
                    for heading in list(doc.metadata.get("local_headings") or [])
                ]
            ),
            "aligned_chunk_ids": _dedupe_preserve_order(
                [
                    chunk_id
                    for doc in bundle_docs
                    for chunk_id in list(doc.metadata.get("aligned_chunk_ids") or [])
                    if chunk_id
                ]
            ),
            "aligned_chunk_pages": _dedupe_preserve_order(
                [
                    page
                    for doc in bundle_docs
                    for page in list(doc.metadata.get("aligned_chunk_pages") or [])
                ]
            ),
            "aligned_chunk_sections": _dedupe_preserve_order(
                [
                    section
                    for doc in bundle_docs
                    for section in list(doc.metadata.get("aligned_chunk_sections") or [])
                    if section
                ]
            ),
        }
        try:
            bundled_doc = doc_cls(page_content=page_content, metadata=bundle_metadata)
        except TypeError:
            class _BundleDoc:
                def __init__(self, page_content: str, metadata: dict[str, Any]):
                    self.page_content = page_content
                    self.metadata = metadata

            bundled_doc = _BundleDoc(page_content=page_content, metadata=bundle_metadata)
        bundled_docs.append(bundled_doc)

    return bundled_docs


def _build_tier3_query_distribution(wrapped_llm: Any) -> list[tuple[Any, float]]:
    from ragas.testset.synthesizers import (
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    return [
        (SingleHopSpecificQuerySynthesizer(llm=wrapped_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=wrapped_llm), 0.75),
    ]


def _filter_query_distribution_for_knowledge_graph(
    query_distribution: Sequence[tuple[Any, float]],
    knowledge_graph: Any,
) -> list[tuple[Any, float]]:
    available: list[tuple[Any, float]] = []
    for synthesizer, weight in query_distribution:
        get_node_clusters = getattr(synthesizer, "get_node_clusters", None)
        if callable(get_node_clusters):
            clusters = list(get_node_clusters(knowledge_graph) or [])
            if not clusters:
                continue
        available.append((synthesizer, float(weight)))

    if not available:
        raise RuntimeError(
            "Tier 3 query profile produced no usable synthesizers for the current knowledge graph. "
            "Broaden the source-document slice or relax the temporary query profile."
        )

    total_weight = sum(weight for _synthesizer, weight in available)
    if total_weight <= 0:
        uniform_weight = 1.0 / len(available)
        return [(synthesizer, uniform_weight) for synthesizer, _weight in available]

    return [(synthesizer, weight / total_weight) for synthesizer, weight in available]


def _build_tier3_transforms(documents: Sequence[Any], llm: Any, embedding_model: Any) -> list[Any]:
    from ragas.testset.graph import NodeType
    from ragas.testset.transforms import (
        CosineSimilarityBuilder,
        CustomNodeFilter,
        EmbeddingExtractor,
        OverlapScoreBuilder,
        Parallel,
        SummaryExtractor,
    )
    from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor

    def _filter_docs(node: Any) -> bool:
        return getattr(node, "type", None) == NodeType.DOCUMENT

    summary_extractor = SummaryExtractor(
        llm=llm,
        filter_nodes=_filter_docs,
    )
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=_filter_docs,
    )
    theme_extractor = ThemesExtractor(
        llm=llm,
        filter_nodes=_filter_docs,
    )
    ner_extractor = NERExtractor(
        llm=llm,
        filter_nodes=_filter_docs,
    )
    node_filter = CustomNodeFilter(
        llm=llm,
        filter_nodes=_filter_docs,
    )
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.5,
        filter_nodes=_filter_docs,
    )
    ner_overlap_sim = OverlapScoreBuilder(
        property_name="entities",
        threshold=0.01,
        filter_nodes=_filter_docs,
    )
    return [
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]


def _build_knowledge_graph_for_documents(docs: Sequence[Any], transforms: Sequence[Any]) -> Any:
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    nodes = [
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        for doc in docs
    ]
    knowledge_graph = KnowledgeGraph(nodes=nodes)
    apply_transforms(knowledge_graph, list(transforms))
    return knowledge_graph


_ARTICLE_DOC_MODULE: Any | None = None


def _load_article_documents_module() -> Any:
    global _ARTICLE_DOC_MODULE
    if _ARTICLE_DOC_MODULE is not None:
        return _ARTICLE_DOC_MODULE

    try:
        from eval import tier3_article_documents as module
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parent / "tier3_article_documents.py"
        spec = importlib.util.spec_from_file_location("tier3_article_documents_runtime", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["tier3_article_documents_runtime"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("tier3_article_documents_runtime", None)

    _ARTICLE_DOC_MODULE = module
    return module


def _build_chunk_corpus_row(doc: Any) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    return {
        "chunk_id": _clean_str(metadata.get("chunk_id")),
        "doc_id": _clean_str(metadata.get("doc_id")),
        "page": metadata.get("page"),
        "section": _clean_str(metadata.get("section")),
        "content": doc.page_content,
        "metadata": metadata,
    }


def _build_article_corpus_row(doc: Any) -> dict[str, Any]:
    return {
        "content": doc.page_content,
        "metadata": dict(doc.metadata or {}),
    }


def _write_chunk_corpus_rows(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_article_corpus(docs: Sequence[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(_build_article_corpus_row(doc), ensure_ascii=False) + "\n")


def _load_chunk_rows_from_chunk_corpus(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk corpus not found: {path}")

    rows: list[dict[str, Any]] = []
    for row in _load_jsonl_rows(path):
        metadata = dict(row.get("metadata") or {})
        for key in ("chunk_id", "doc_id", "page", "section"):
            value = row.get(key)
            if value is not None and key not in metadata:
                metadata[key] = value
        rows.append(
            {
                "chunk_id": _clean_str(metadata.get("chunk_id")),
                "doc_id": _clean_str(metadata.get("doc_id")),
                "page": metadata.get("page"),
                "section": _clean_str(metadata.get("section")),
                "content": row["content"],
                "metadata": metadata,
            }
        )

    if not rows:
        raise ValueError(f"Chunk corpus is empty: {path}")

    print(f"  Reused {len(rows)} production chunks from cached corpus {path.name}")
    return rows


def _load_source_documents_from_article_corpus(path: Path) -> list[Any]:
    from langchain_core.documents import Document

    if not path.exists():
        raise FileNotFoundError(f"Article corpus not found: {path}")

    docs: list[Any] = []
    for row in _load_jsonl_rows(path):
        metadata = dict(row.get("metadata") or {})
        docs.append(Document(page_content=row["content"], metadata=metadata))

    if not docs:
        raise ValueError(f"Article corpus is empty: {path}")

    article_docs_module = _load_article_documents_module()
    eligible_windows = article_docs_module.filter_generation_eligible_article_windows(
        [
            article_docs_module.ArticleWindow(
                article_title=_clean_str(doc.metadata.get("article_title")),
                page_start=int(doc.metadata.get("page_start") or 0),
                page_end=int(doc.metadata.get("page_end") or 0),
                local_headings=list(doc.metadata.get("local_headings") or []),
                body_text=doc.page_content,
            )
            for doc in docs
        ]
    )
    eligible_keys = {
        (window.article_title, window.page_start, window.page_end)
        for window in eligible_windows
    }
    filtered_docs = [
        doc
        for doc in docs
        if (
            _clean_str(doc.metadata.get("article_title")),
            int(doc.metadata.get("page_start") or 0),
            int(doc.metadata.get("page_end") or 0),
        )
        in eligible_keys
    ]
    filtered_count = len(docs) - len(filtered_docs)
    if not filtered_docs:
        raise ValueError(
            f"Article corpus contains no eligible medical entry docs after filtering: {path}"
        )
    print(
        f"  Reused {len(filtered_docs)} article source documents from cached corpus {path.name}"
        + (f" after filtering {filtered_count} non-entry docs" if filtered_count else "")
    )
    return filtered_docs


def _load_chunk_rows_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    from tools.pdf_loader import process_pdf

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = process_pdf(str(path))
    if not docs:
        raise ValueError(f"No content extracted from PDF: {pdf_path}")

    rows = [_build_chunk_corpus_row(doc) for doc in docs]
    print(f"  Loaded {len(rows)} production chunks from {path.name}")
    return rows


def _load_chunk_rows_from_indexed_corpus(pdf_path: str) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from db.models import DocumentChunkModel
    from db.session import SessionLocal, init_database

    target_doc_id = Path(pdf_path).stem

    init_database()
    db = SessionLocal()
    try:
        rows = db.execute(
            select(DocumentChunkModel)
            .where(DocumentChunkModel.doc_id == target_doc_id)
            .order_by(DocumentChunkModel.page.asc(), DocumentChunkModel.id.asc())
        ).scalars().all()
    finally:
        db.close()

    rows = [
        {
            "chunk_id": _clean_str(row.chunk_id),
            "doc_id": _clean_str(row.doc_id),
            "page": row.page,
            "section": _clean_str(row.section),
            "content": row.content,
            "metadata": {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "page": row.page,
                "section": row.section,
                **(row.metadata_json or {}),
            },
        }
        for row in rows
    ]
    if rows:
        print(f"  Loaded {len(rows)} production chunks from indexed corpus for {target_doc_id}")
    return rows


def load_chunk_rows(
    pdf_path: str,
    *,
    source: str = "auto",
    chunk_corpus_path: Path = CHUNK_CORPUS_PATH,
) -> list[dict[str, Any]]:
    if source not in {"auto", "cache", "db", "pdf"}:
        raise ValueError(f"Unsupported source: {source}")

    if source in {"auto", "cache"} and chunk_corpus_path.exists():
        return _load_chunk_rows_from_chunk_corpus(chunk_corpus_path)
    if source == "cache":
        raise FileNotFoundError(
            f"Chunk corpus not found: {chunk_corpus_path}. "
            "Run once with --source db or --source pdf to create it."
        )

    if source in {"auto", "db"}:
        try:
            rows = _load_chunk_rows_from_indexed_corpus(pdf_path)
        except Exception:
            if source == "db":
                raise
            rows = []
        if rows:
            _write_chunk_corpus_rows(rows, chunk_corpus_path)
            print(f"  Checkpointed Step 1 chunk corpus to {chunk_corpus_path}")
            return rows
        if source == "db":
            raise RuntimeError(
                f"No indexed chunks found for {Path(pdf_path).stem}. "
                "Reindex first or run with --source pdf."
            )

    rows = _load_chunk_rows_from_pdf(pdf_path)
    _write_chunk_corpus_rows(rows, chunk_corpus_path)
    print(f"  Checkpointed Step 1 chunk corpus to {chunk_corpus_path}")
    return rows


def _page_span_chunk_rows_for_source_doc(
    source_doc: Any,
    chunk_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = source_doc.metadata or {}
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if isinstance(page_start, int) and isinstance(page_end, int):
        return [
            row
            for row in chunk_rows
            if isinstance(row.get("page"), int) and page_start <= row["page"] <= page_end
        ]
    return chunk_rows


def _candidate_chunk_rows_for_source_doc(
    source_doc: Any,
    chunk_rows: list[dict[str, Any]],
    *,
    prefer_aligned: bool = True,
) -> list[dict[str, Any]]:
    metadata = source_doc.metadata or {}
    aligned_chunk_ids = metadata.get("aligned_chunk_ids") or []
    if prefer_aligned and aligned_chunk_ids:
        aligned = {chunk_id for chunk_id in aligned_chunk_ids if chunk_id}
        aligned_rows = [row for row in chunk_rows if row.get("chunk_id") in aligned]
        if aligned_rows:
            return aligned_rows

    return _page_span_chunk_rows_for_source_doc(source_doc, chunk_rows)


def _build_article_source_documents_from_pdf(
    pdf_path: str,
    chunk_rows: list[dict[str, Any]],
    *,
    article_limit: int | None = None,
) -> list[Any]:
    article_docs_module = _load_article_documents_module()

    if not chunk_rows:
        raise ValueError("Chunk rows are required to build article source documents.")

    page_numbers = sorted(
        {
            int(row["page"])
            for row in chunk_rows
            if isinstance(row.get("page"), int) or str(row.get("page", "")).isdigit()
        }
    )
    parsed_pages = article_docs_module.extract_pdf_pages(Path(pdf_path), min(page_numbers), max(page_numbers))
    windows = article_docs_module.build_article_windows(parsed_pages)
    eligible_windows = article_docs_module.filter_generation_eligible_article_windows(windows)
    docs = article_docs_module.build_article_source_documents(
        parsed_pages,
        doc_id=Path(pdf_path).stem,
        windows=eligible_windows,
    )
    alignments = article_docs_module.align_article_windows_to_chunks(eligible_windows, chunk_rows)

    filtered_count = len(windows) - len(eligible_windows)
    if filtered_count:
        print(f"  Filtered {filtered_count} non-entry article docs before RAGAS generation")

    for doc, alignment in zip(docs, alignments):
        doc.metadata["aligned_chunk_ids"] = list(alignment.chunk_ids)
        doc.metadata["aligned_chunk_pages"] = list(alignment.chunk_pages)
        doc.metadata["aligned_chunk_sections"] = list(alignment.chunk_sections)

    if article_limit is not None:
        docs = docs[:article_limit]
    return docs


def load_source_documents(
    pdf_path: str,
    *,
    source: str = "auto",
    article_corpus_path: Path = ARTICLE_CORPUS_PATH,
    chunk_corpus_path: Path = CHUNK_CORPUS_PATH,
    chunk_rows: list[dict[str, Any]] | None = None,
    article_limit: int | None = None,
) -> list[Any]:
    if source not in {"auto", "cache", "db", "pdf"}:
        raise ValueError(f"Unsupported source: {source}")

    if source in {"auto", "cache"} and article_corpus_path.exists():
        docs = _load_source_documents_from_article_corpus(article_corpus_path)
        return docs[:article_limit] if article_limit is not None else docs
    if source == "cache":
        raise FileNotFoundError(
            f"Article corpus not found: {article_corpus_path}. "
            "Run once with --source auto, --source db, or --source pdf to create it."
        )

    if chunk_rows is None:
        chunk_rows = load_chunk_rows(
            pdf_path,
            source=source,
            chunk_corpus_path=chunk_corpus_path,
        )

    docs = _build_article_source_documents_from_pdf(
        pdf_path,
        chunk_rows,
        article_limit=article_limit,
    )
    _write_article_corpus(docs, article_corpus_path)
    print(f"  Checkpointed article corpus to {article_corpus_path}")
    return docs


def _resolve_ragas_components():
    try:
        from ragas.llms import LangchainLLMWrapper
    except Exception:
        from ragas.llms.base import LangchainLLMWrapper

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

    try:
        from ragas.testset import TestsetGenerator
    except Exception:
        from ragas.testset.generator import TestsetGenerator

    return LangchainLLMWrapper, LangchainEmbeddingsWrapper, TestsetGenerator


def generate_testset(docs, testset_size: int):
    LangchainLLMWrapper, LangchainEmbeddingsWrapper, TestsetGenerator = _resolve_ragas_components()

    llm = _resolve_llm()
    embeddings, embedding_label = _resolve_embeddings()
    llm_label = _resolve_runtime_model_name(llm)

    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    transforms = _build_tier3_transforms(docs, wrapped_llm, wrapped_embeddings)
    query_distribution = _build_tier3_query_distribution(wrapped_llm)

    try:
        ragas_version = version("ragas")
    except PackageNotFoundError:
        ragas_version = "unknown"

    print(f"  RAGAS version: {ragas_version}")
    print(f"  LLM: {llm_label}")
    print(f"  Embeddings: {embedding_label}")
    print(f"  Requested testset size: {testset_size}")
    print(f"  Query profile: {_TEMP_QUERY_PROFILE_NAME}")
    print(f"  Transform profile: {_TEMP_TRANSFORM_PROFILE_NAME}")

    try:
        generator = TestsetGenerator(
            llm=wrapped_llm,
            embedding_model=wrapped_embeddings,
        )
    except TypeError:
        if hasattr(TestsetGenerator, "from_langchain"):
            generator = TestsetGenerator.from_langchain(llm=llm, embeddings=embeddings)
        else:
            raise RuntimeError("Unsupported RAGAS TestsetGenerator constructor for installed version.")

    if not hasattr(generator, "generate"):
        raise RuntimeError("Unsupported RAGAS generator API for installed version.")

    knowledge_graph = _build_knowledge_graph_for_documents(docs, transforms)
    generator.knowledge_graph = knowledge_graph
    filtered_query_distribution = _filter_query_distribution_for_knowledge_graph(
        query_distribution,
        knowledge_graph,
    )

    if len(filtered_query_distribution) != len(query_distribution):
        active_names = [type(synthesizer).__name__ for synthesizer, _weight in filtered_query_distribution]
        print("  Filtered query profile to available synthesizers: " + ", ".join(active_names))

    dataset = generator.generate(
        testset_size=testset_size,
        query_distribution=filtered_query_distribution,
    )

    return dataset, llm_label


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _match_score(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0

    if right_norm in left_norm or left_norm in right_norm:
        return 1.0

    left_tokens = _tokenize(left_norm)
    right_tokens = _tokenize(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0

    recall_like = overlap / len(right_tokens)
    precision_like = overlap / len(left_tokens)
    jaccard = overlap / len(left_tokens | right_tokens)
    return max(recall_like, precision_like, jaccard)


def _chunk_row_search_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    section = _clean_str(row.get("section") or metadata.get("section"))
    section_path = _clean_str(metadata.get("section_path") or section)
    context_prefix = _clean_str(metadata.get("context_prefix"))
    return " ".join(
        part for part in [row.get("content", ""), section_path, context_prefix] if part
    )


def _infer_ground_truth_alignment(
    contexts: list[str],
    docs,
    chunk_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    matched_chunk_ids: list[str] = []
    matched_doc_ids: list[str] = []
    matched_pages: list[int] = []
    matched_content_types: list[str] = []
    heuristic_used = False

    for context in contexts:
        best_doc = None
        best_score = 0.0

        for doc in docs:
            score = _match_score(doc.page_content, context)
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None or best_score < 0.60:
            continue

        if best_score < 0.999:
            heuristic_used = True

        metadata = best_doc.metadata or {}
        if chunk_rows is None:
            chunk_id = _clean_str(metadata.get("chunk_id"))
            doc_id = _clean_str(metadata.get("doc_id"))
            content_type = _clean_str(metadata.get("content_type"))
            page = metadata.get("page")

            if chunk_id and chunk_id not in matched_chunk_ids:
                matched_chunk_ids.append(chunk_id)
            if doc_id and doc_id not in matched_doc_ids:
                matched_doc_ids.append(doc_id)
            if isinstance(page, int) and page not in matched_pages:
                matched_pages.append(page)
            if content_type and content_type not in matched_content_types:
                matched_content_types.append(content_type)
            continue

        candidate_rows = _candidate_chunk_rows_for_source_doc(best_doc, chunk_rows)
        best_row = None
        best_row_score = 0.0
        for row in candidate_rows:
            score = _match_score(_chunk_row_search_text(row), context)
            if score > best_row_score:
                best_row_score = score
                best_row = row

        if best_row is None or best_row_score < 0.60:
            fallback_rows = _candidate_chunk_rows_for_source_doc(
                best_doc,
                chunk_rows,
                prefer_aligned=False,
            )
            if fallback_rows != candidate_rows:
                for row in fallback_rows:
                    score = _match_score(_chunk_row_search_text(row), context)
                    if score > best_row_score:
                        best_row_score = score
                        best_row = row

        if best_row is None or best_row_score < 0.60:
            continue

        if best_row_score < 0.999:
            heuristic_used = True

        row_metadata = best_row.get("metadata") or {}
        chunk_id = _clean_str(best_row.get("chunk_id"))
        doc_id = _clean_str(best_row.get("doc_id") or row_metadata.get("doc_id"))
        content_type = _clean_str(row_metadata.get("content_type"))
        page = best_row.get("page")

        if chunk_id and chunk_id not in matched_chunk_ids:
            matched_chunk_ids.append(chunk_id)
        if doc_id and doc_id not in matched_doc_ids:
            matched_doc_ids.append(doc_id)
        if isinstance(page, int) and page not in matched_pages:
            matched_pages.append(page)
        if content_type and content_type not in matched_content_types:
            matched_content_types.append(content_type)

    return {
        "ground_truth_chunk_ids": matched_chunk_ids,
        "ground_truth_doc_ids": matched_doc_ids,
        "ground_truth_pages": matched_pages,
        "ground_truth_content_types": matched_content_types,
        "ground_truth_alignment": "heuristic_context_to_chunk" if heuristic_used else "exact_context_to_chunk",
    }


def _to_rows(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas().to_dict(orient="records")
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    if isinstance(dataset, list):
        return dataset
    if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
        return dataset.samples
    raise RuntimeError("Unsupported dataset output type from RAGAS generator.")


def _extract_contexts(row: dict[str, Any]) -> list[str]:
    for key in ("reference_contexts", "contexts", "contexts_v1", "retrieved_contexts"):
        raw = row.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    contexts = [_clean_str(item) for item in parsed if _clean_str(item)]
                    if contexts:
                        return contexts
            return [text]
        if isinstance(raw, list):
            contexts = []
            for item in raw:
                if isinstance(item, dict):
                    candidate = _clean_str(
                        item.get("page_content") or item.get("content") or item.get("text")
                    )
                else:
                    candidate = _clean_str(item)
                if candidate:
                    contexts.append(candidate)
            if contexts:
                return contexts
    return []


def _validate_ragas_metadata_contract(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("RAGAS returned no rows to convert.")

    with_metadata = 0
    for row in rows:
        evolution = _clean_str(row.get("evolution_type"))
        synthesizer = _clean_str(row.get("synthesizer_name"))
        if evolution or synthesizer:
            with_metadata += 1

    if with_metadata == 0:
        raise RuntimeError(
            "RAGAS output is missing evolution metadata for all rows. "
            "Do not trust this Tier 3 run. Rebuild the eval image before rerunning "
            "if you changed eval/generate_tier3_rag.py, then inspect "
            f"{RAW_RAGAS_ROWS_PATH} to confirm the installed RAGAS output shape."
        )

    print(f"  RAGAS rows with evolution metadata: {with_metadata}/{len(rows)}")


def _normalize_ragas_difficulty_token(raw_evolution: Any, raw_synthesizer: Any) -> str:
    evolution = _clean_str(raw_evolution).lower()
    synthesizer = _clean_str(raw_synthesizer).lower()

    if evolution in _DIFFICULTY_MAP:
        return evolution
    if synthesizer in _DIFFICULTY_MAP:
        return synthesizer
    if synthesizer.startswith("single_hop_"):
        return "simple"
    if synthesizer.startswith("multi_hop_"):
        return "multi_context"
    return "simple"


def convert_dataset_to_schema_v1(
    dataset,
    generator_model: str,
    docs,
    *,
    chunk_rows: list[dict[str, Any]] | None = None,
    requested_size: int | None = None,
) -> Tier3ConversionResult:
    rows = _to_rows(dataset)
    _validate_ragas_metadata_contract(rows)

    samples: list[dict[str, Any]] = []
    skipped_missing_question = 0
    skipped_empty_context = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for idx, row in enumerate(rows):
        question = _clean_str(row.get("user_input") or row.get("question"))
        if not question:
            skipped_missing_question += 1
            continue

        contexts = _extract_contexts(row)
        if not contexts:
            skipped_empty_context += 1
            continue

        ground_truth = _clean_str(row.get("reference") or row.get("ground_truth"))
        raw_evolution = _clean_str(row.get("evolution_type"))
        raw_synthesizer = _clean_str(row.get("synthesizer_name"))
        difficulty = _DIFFICULTY_MAP.get(
            _normalize_ragas_difficulty_token(raw_evolution, raw_synthesizer),
            "simple",
        )
        alignment = _infer_ground_truth_alignment(contexts, docs, chunk_rows)

        samples.append(
            {
                "id": f"T3-{idx + 1:04d}",
                "version": "v1",
                "tier": "rag",
                "category": f"rag_{difficulty}",
                "question": question,
                "multi_turn_context": None,
                "ground_truth": ground_truth or None,
                "ground_truth_contexts": contexts,
                "ground_truth_chunk_ids": alignment["ground_truth_chunk_ids"],
                "ground_truth_doc_ids": alignment["ground_truth_doc_ids"],
                "ground_truth_pages": alignment["ground_truth_pages"],
                "ground_truth_content_types": alignment["ground_truth_content_types"],
                "expected_planned_route": "vector",
                "expected_final_route": "vector",
                "expected_behavior": "answer_with_citations",
                "expected_guardrail_action": None,
                "difficulty": difficulty,
                "split": "test",
                "tags": ["rag", difficulty, "ragas_generated"],
                "provenance": {
                    "source_type": "ragas_generated",
                    "source_ref": f"ragas-testset-{idx + 1}",
                    "generator_model": generator_model,
                    "generator_prompt_hash": f"generated_at:{now_iso}",
                    "ragas_evolution_type": raw_evolution or None,
                    "ragas_synthesizer_name": raw_synthesizer or None,
                    "ground_truth_alignment": alignment["ground_truth_alignment"],
                    "review_status": "draft",
                    "reviewer": None,
                    "reviewed_at": None,
                },
            }
        )

    return Tier3ConversionResult(
        samples=samples,
        raw_rows=rows,
        skipped_missing_question=skipped_missing_question,
        skipped_empty_context=skipped_empty_context,
    )


def dataset_to_schema_v1(
    dataset,
    generator_model: str,
    docs,
    chunk_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return convert_dataset_to_schema_v1(
        dataset,
        generator_model,
        docs,
        chunk_rows=chunk_rows,
    ).samples


def _print_conversion_summary(result: Tier3ConversionResult) -> None:
    accepted_by_difficulty = Counter(
        sample.get("difficulty", "unknown") for sample in result.samples
    )
    print("  Raw rows:", result.raw_row_count)
    print("  Accepted rows:", result.accepted_rows)
    print("  Skipped missing-question rows:", result.skipped_missing_question)
    print("  Skipped empty-context rows:", result.skipped_empty_context)
    if accepted_by_difficulty:
        print("  Accepted by difficulty:")
        for difficulty, count in sorted(accepted_by_difficulty.items()):
            print(f"    {difficulty:<15} {count:>4}")


def write_samples(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_raw_ragas_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Tier 3 offline RAG draft samples using RAGAS.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF),
        help=f"Path to medical PDF (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Number of test samples to generate (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "cache", "db", "pdf"),
        default="auto",
        help=(
            "Alignment-corpus source used to recover article documents: cached chunk corpus, "
            "indexed DB corpus, or PDF parse (default: auto)."
        ),
    )
    parser.add_argument(
        "--chunk-corpus",
        type=str,
        default=str(CHUNK_CORPUS_PATH),
        help=f"Checkpoint path for reusable production chunks used for alignment (default: {CHUNK_CORPUS_PATH})",
    )
    parser.add_argument(
        "--article-corpus",
        type=str,
        default=str(ARTICLE_CORPUS_PATH),
        help=f"Checkpoint path for article-level source documents (default: {ARTICLE_CORPUS_PATH})",
    )
    parser.add_argument(
        "--article-limit",
        type=int,
        default=None,
        help="Optional cap on article source documents for smoke runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(TIER3_PATH),
        help=f"Draft schema_v1 output path (default: {TIER3_PATH})",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=str(RAW_RAGAS_ROWS_PATH),
        help=f"Raw RAGAS checkpoint path (default: {RAW_RAGAS_ROWS_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated samples without writing to disk.",
    )
    parser.add_argument(
        "--resume-from",
        choices=("none", "raw_samples"),
        default="none",
        help="Resume Tier 3 conversion from raw RAGAS rows. Only raw_samples is supported.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("  Tier 3 Offline RAGAS Utility")
    print(f"{'=' * 60}\n")

    print("Step 1: Loading chunk alignment corpus...")
    chunk_rows = load_chunk_rows(
        args.pdf,
        source=args.source,
        chunk_corpus_path=Path(args.chunk_corpus),
    )
    print("\nStep 2: Loading article source documents...")
    source_docs = load_source_documents(
        args.pdf,
        source=args.source,
        article_corpus_path=Path(args.article_corpus),
        chunk_corpus_path=Path(args.chunk_corpus),
        chunk_rows=chunk_rows,
        article_limit=args.article_limit,
    )
    selected_article_docs = _require_experiment_source_documents(source_docs)
    selected_titles = [
        _clean_str((doc.metadata or {}).get("article_title"))
        for doc in selected_article_docs
    ]
    _write_article_corpus(selected_article_docs, Path(args.article_corpus))
    print(
        "  Restricted Tier 3 generation to temporary cancer-family article docs: "
        + ", ".join(selected_titles)
    )
    print(f"  Re-checkpointed experiment article corpus to {args.article_corpus}")
    source_docs = _build_experiment_generation_documents(selected_article_docs)
    print(
        "  Built temporary same-family generation docs: "
        + "; ".join(
            f"{_clean_str((doc.metadata or {}).get('bundle_id'))} -> {', '.join((doc.metadata or {}).get('bundle_article_titles') or [])}"
            for doc in source_docs
        )
    )

    raw_output_path = Path(args.raw_output)
    output_path = Path(args.output)

    print("\nStep 3: Generating testset with RAGAS (this may take several minutes)...")
    if args.resume_from == "raw_samples":
        if not raw_output_path.exists():
            raise RuntimeError(
                f"Cannot resume from raw_samples: missing checkpoint {raw_output_path}"
            )
        raw_rows = _load_jsonl_rows(raw_output_path)
        generator_model = "unknown(resumed_raw_rows)"
        print(f"  Reused {len(raw_rows)} raw RAGAS rows from {raw_output_path}")
    else:
        dataset, generator_model = generate_testset(source_docs, testset_size=args.size)
        raw_rows = _to_rows(dataset)
        write_raw_ragas_rows(raw_rows, raw_output_path)
        print(f"  Checkpointed raw RAGAS rows to {raw_output_path}")

    print("\nStep 4: Converting to schema_v1 format...")
    result = convert_dataset_to_schema_v1(
        raw_rows,
        generator_model,
        source_docs,
        chunk_rows=chunk_rows,
        requested_size=args.size,
    )
    _print_conversion_summary(result)

    if not result.samples:
        print("\n  [ERROR] No valid samples generated. Check LLM and PDF configuration.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(result.samples)} samples to {output_path}")
        print("\nSample preview (first 5):")
        for sample in result.samples[:5]:
            print(f"  {sample['id']}  [{sample['difficulty']}]  {sample['question'][:80]}")
    else:
        write_samples(result.samples, output_path)
        print(f"\nWrote {len(result.samples)} samples to {output_path}")

    diff_dist = Counter(sample["difficulty"] for sample in result.samples)
    print(f"\n{'=' * 60}")
    print("  Tier 3 Offline Generation Summary")
    print(f"{'=' * 60}")
    print(f"  PDF:             {args.pdf}")
    print(f"  Requested:       {args.size}")
    print(f"  Generated:       {len(result.samples)}")
    print(f"  Generator model: {generator_model}")
    print()
    print("  Difficulty distribution:")
    for difficulty, count in sorted(diff_dist.items()):
        print(f"    {difficulty:<15} {count:>4}")
    print(f"{'=' * 60}\n")

    print("  NOTE: All generated samples have review_status='draft'.")
    print("  NOTE: Manual Tier 3 seeds are curated separately from this generator.")
    print("  Run `python -m eval.validate` to check schema and quality gates.\n")


if __name__ == "__main__":
    main()
