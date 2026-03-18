"""
Chunking component evaluation harness for comparing chunking pipeline behavior.

Scope:
- structure and metadata quality of produced chunks
- ingestion runtime
- lexical retrieval proxy over produced chunks

This script does not score final LLM answers. Use `eval/workflow_eval.py` for
end-to-end retrieval -> executor -> generation evaluation.

Usage:
    python -m eval.chunking_eval                        # evaluate current pipeline
    python -m eval.chunking_eval --pdf data/other.pdf   # custom PDF
    python -m eval.chunking_eval --save baseline        # save as named snapshot
    python -m eval.chunking_eval --compare baseline     # compare against snapshot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median, quantiles

EVAL_DIR = Path(__file__).resolve().parent / "chunking_snapshots"

try:
    import tiktoken as _tiktoken

    _enc = _tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))

except Exception:

    def _count_tokens(text: str) -> int:
        return len(text.split())


def _normalize_tokens(text: str) -> set[str]:
    import re

    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return {token for token in normalized.split() if len(token) > 2}


def analyze_chunks(chunks) -> dict:
    """Compute chunking quality metrics from a list of LangChain Documents."""
    if not chunks:
        return {"error": "no chunks produced"}

    token_counts = [_count_tokens(c.page_content) for c in chunks]
    char_counts = [len(c.page_content) for c in chunks]

    content_types = Counter()
    has_section = 0
    has_context_prefix = 0
    has_page = 0
    table_chunks = 0

    for c in chunks:
        meta = c.metadata or {}
        ct = meta.get("content_type", "unknown")
        content_types[ct] += 1

        if meta.get("section_path") or meta.get("section"):
            has_section += 1
        if meta.get("context_prefix"):
            has_context_prefix += 1
        if meta.get("page") is not None or meta.get("page_start") is not None:
            has_page += 1
        if meta.get("is_table") or ct == "table":
            table_chunks += 1

    sorted_tokens = sorted(token_counts)
    p50 = median(sorted_tokens)
    p95 = quantiles(sorted_tokens, n=20)[18] if len(sorted_tokens) >= 20 else max(sorted_tokens)

    return {
        "total_chunks": len(chunks),
        "token_stats": {
            "mean": round(mean(token_counts), 1),
            "median": round(p50, 1),
            "p95": round(p95, 1),
            "min": min(token_counts),
            "max": max(token_counts),
        },
        "char_stats": {
            "mean": round(mean(char_counts), 1),
            "min": min(char_counts),
            "max": max(char_counts),
        },
        "content_types": dict(content_types),
        "table_chunks": table_chunks,
        "pct_with_section_metadata": round(has_section / len(chunks) * 100, 1),
        "pct_with_context_prefix": round(has_context_prefix / len(chunks) * 100, 1),
        "pct_with_page_metadata": round(has_page / len(chunks) * 100, 1),
    }


def run_ingestion_benchmark(pdf_path: str) -> tuple[list, float]:
    """Run process_pdf() and measure wall-clock time."""
    from tools.pdf_loader import process_pdf

    start = time.perf_counter()
    chunks = process_pdf(pdf_path)
    elapsed = time.perf_counter() - start
    return chunks, elapsed


# -- Golden retrieval queries for recall@k --
# These are hand-crafted queries that should retrieve specific types of content.
# They don't require the full RAGAS pipeline — just check if relevant chunks
# appear in top-k results.
GOLDEN_RETRIEVAL_QUERIES = [
    {
        "query": "pathophysiology of type 2 diabetes",
        "expected_content_keywords": ["insulin", "resistance", "glucose", "beta"],
        "expected_content_type": "text",
    },
    {
        "query": "adult acetaminophen dosage",
        "expected_content_keywords": ["acetaminophen", "dose", "mg"],
        "expected_content_type": "any",
    },
    {
        "query": "contraindications of metformin",
        "expected_content_keywords": ["metformin", "contraindic", "renal"],
        "expected_content_type": "text",
    },
    {
        "query": "hypertension complications affecting kidneys",
        "expected_content_keywords": ["hypertension", "kidney", "renal", "nephro"],
        "expected_content_type": "text",
    },
    {
        "query": "drug interaction table for warfarin",
        "expected_content_keywords": ["warfarin", "interaction"],
        "expected_content_type": "table",
    },
]


def evaluate_retrieval(chunks, k: int = 5) -> dict:
    """Simple lexical retrieval proxy over produced chunks."""
    results = []
    for gq in GOLDEN_RETRIEVAL_QUERIES:
        query_tokens = _normalize_tokens(gq["query"])
        scored = []
        for chunk in chunks:
            doc_tokens = _normalize_tokens(chunk.page_content)
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            scored.append((chunk, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        # Check if any expected keywords appear in top-k
        top_k_text = " ".join(c.page_content.lower() for c, _ in top_k)
        keywords_found = sum(
            1 for kw in gq["expected_content_keywords"]
            if kw.lower() in top_k_text
        )
        keyword_recall = keywords_found / max(len(gq["expected_content_keywords"]), 1)

        results.append({
            "query": gq["query"],
            "keyword_recall": round(keyword_recall, 3),
            "top_score": round(top_k[0][1], 3) if top_k else 0.0,
        })

    avg_recall = mean(r["keyword_recall"] for r in results) if results else 0.0
    return {
        "k": k,
        "avg_keyword_recall": round(avg_recall, 3),
        "per_query": results,
    }


def save_snapshot(name: str, report: dict) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_snapshot(name: str) -> dict | None:
    path = EVAL_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(current: dict, baseline: dict) -> None:
    """Print a side-by-side comparison of two chunking eval reports."""
    print(f"\n{'='*70}")
    print(f"  Chunking Comparison: baseline vs current")
    print(f"{'='*70}\n")

    metrics = [
        ("Total chunks", "total_chunks"),
        ("Table chunks", "table_chunks"),
        ("% with section", "pct_with_section_metadata"),
        ("% with context prefix", "pct_with_context_prefix"),
        ("% with page", "pct_with_page_metadata"),
    ]

    print(f"  {'Metric':<30} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    for label, key in metrics:
        b = baseline.get(key, "N/A")
        c = current.get(key, "N/A")
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            delta = c - b
            sign = "+" if delta > 0 else ""
            print(f"  {label:<30} {b:>10} {c:>10} {sign}{delta:>9}")
        else:
            print(f"  {label:<30} {str(b):>10} {str(c):>10}")

    # Token stats
    for stat_key in ("mean", "median", "p95"):
        b = baseline.get("token_stats", {}).get(stat_key, "N/A")
        c = current.get("token_stats", {}).get(stat_key, "N/A")
        label = f"Tokens {stat_key}"
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            delta = c - b
            sign = "+" if delta > 0 else ""
            print(f"  {label:<30} {b:>10} {c:>10} {sign}{delta:>9}")

    # Retrieval
    b_recall = baseline.get("retrieval", {}).get("avg_keyword_recall", "N/A")
    c_recall = current.get("retrieval", {}).get("avg_keyword_recall", "N/A")
    if isinstance(b_recall, (int, float)) and isinstance(c_recall, (int, float)):
        delta = c_recall - b_recall
        sign = "+" if delta > 0 else ""
        print(f"  {'Avg keyword recall@5':<30} {b_recall:>10} {c_recall:>10} {sign}{delta:>9}")

    # Ingestion time
    b_time = baseline.get("ingestion_seconds", "N/A")
    c_time = current.get("ingestion_seconds", "N/A")
    if isinstance(b_time, (int, float)) and isinstance(c_time, (int, float)):
        delta = c_time - b_time
        sign = "+" if delta > 0 else ""
        print(f"  {'Ingestion time (s)':<30} {b_time:>10.2f} {c_time:>10.2f} {sign}{delta:>9.2f}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Chunking pipeline evaluation harness.")
    parser.add_argument("--pdf", default="./data/medical_book.pdf", help="PDF path")
    parser.add_argument("--save", metavar="NAME", help="Save results as a named snapshot")
    parser.add_argument("--compare", metavar="NAME", help="Compare against a saved snapshot")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"PDF not found: {args.pdf}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Chunking Evaluation Harness")
    print(f"{'='*60}")
    print(f"  PDF: {pdf.name}")

    # Run ingestion benchmark
    print("\n  Running ingestion...")
    chunks, elapsed = run_ingestion_benchmark(str(pdf))
    print(f"  Produced {len(chunks)} chunks in {elapsed:.2f}s")

    # Analyze chunks
    print("\n  Analyzing chunk quality...")
    analysis = analyze_chunks(chunks)

    # Run retrieval eval
    print("  Running retrieval recall test...")
    retrieval = evaluate_retrieval(chunks)

    # Build full report
    report = {
        **analysis,
        "ingestion_seconds": round(elapsed, 3),
        "pdf": str(pdf),
        "scope": "component_only",
        "retrieval": retrieval,
    }

    # Print results
    print(f"\n  {'='*50}")
    print(f"  Results")
    print(f"  {'='*50}")
    print(f"  Total chunks:          {analysis['total_chunks']}")
    print(f"  Token mean/p50/p95:    {analysis['token_stats']['mean']} / {analysis['token_stats']['median']} / {analysis['token_stats']['p95']}")
    print(f"  Table chunks:          {analysis['table_chunks']}")
    print(f"  % with section:        {analysis['pct_with_section_metadata']}%")
    print(f"  % with context prefix: {analysis['pct_with_context_prefix']}%")
    print(f"  % with page:           {analysis['pct_with_page_metadata']}%")
    print(f"  Content types:         {analysis['content_types']}")
    print(f"  Ingestion time:        {elapsed:.2f}s")
    print(f"  Avg recall@5:          {retrieval['avg_keyword_recall']}")
    print()

    # Save if requested
    if args.save:
        path = save_snapshot(args.save, report)
        print(f"  Saved snapshot: {path}")

    # Compare if requested
    if args.compare:
        baseline = load_snapshot(args.compare)
        if baseline is None:
            print(f"  Snapshot '{args.compare}' not found in {EVAL_DIR}")
        else:
            compare_reports(report, baseline)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
