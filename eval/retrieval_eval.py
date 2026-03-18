"""
Retrieval evaluation harness for the indexed medical corpus.

This script measures the actual retrieval layer against Tier 3 gold labels
using ranked-retrieval metrics over the real indexed corpus with graded
chunk-ID-only relevance scoring.

Modes:
- dense
- sparse
- hybrid
- hybrid_rerank
- pipeline (runs query through rewriter first, then hybrid_rerank with step-back merge + metadata boost)
- all (runs dense/sparse/hybrid/hybrid_rerank, excludes pipeline)

Metrics:
- HitRate@k
- Recall@k
- Precision@k
- MaxPrecision@k (theoretical ceiling given ground truth size)
- MRR@k
- MAP@k
- nDCG@k
- section_diversity@k
- latency mean / median / p95
- difficulty/content-type slices
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from agents.retriever_agent import (
    _merge_and_deduplicate,
    rerank,
)
from db.repositories import InMemoryVectorRepository, VectorRepository
from db.session import SessionLocal
from tools.embedding_client import embed_query

DEFAULT_TIER3_PATH = Path(__file__).resolve().parent / "golden" / "v1" / "tier3_rag.jsonl"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "retrieval_snapshots"
PIPELINE_SNAPSHOT_DIR = SNAPSHOT_DIR / "pipeline"
DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_FETCH_K = 20
SUPPORTED_MODES = ("dense", "sparse", "hybrid", "hybrid_rerank", "pipeline")
SUPPORTED_SPLITS = ("dev", "test", "all")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((p / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _load_tier3_samples(path: Path, split: str = "test") -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Tier 3 file not found: {path}. The active Tier 3 benchmark is the curated manual-seed file "
            "`eval/golden/v1/tier3_rag.jsonl`."
        )
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS}.")

    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        contexts = row.get("ground_truth_contexts") or []
        chunk_ids = row.get("ground_truth_chunk_ids") or []
        evidence = row.get("evidence") or {}
        row_split = row.get("split")
        if split != "all" and row_split not in (None, split):
            continue
        if row.get("question") and (contexts or chunk_ids or evidence):
            samples.append(row)
    if not samples:
        raise RuntimeError(f"No valid Tier 3 retrieval samples found in {path}")
    return samples


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_fixture_repo(path: str | Path) -> InMemoryVectorRepository:
    repo = InMemoryVectorRepository()
    for row in _load_jsonl_rows(Path(path)):
        repo.upsert_chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            content=row["content"],
            embedding=row["embedding"],
            page=row.get("page"),
            section=row.get("section"),
            metadata=row.get("metadata") or {},
        )
    return repo


def _retrieve(
    repo: VectorRepository | InMemoryVectorRepository,
    query: str,
    mode: str,
    fetch_k: int,
    *,
    query_embedder=embed_query,
    rerank_fn=rerank,
):
    query_embedding = query_embedder(query)

    if mode == "dense":
        return repo.similarity_search(query_embedding, k=fetch_k)
    if mode == "sparse":
        return repo.keyword_search(query, k=fetch_k)
    if mode == "hybrid":
        return repo.hybrid_search(query=query, query_embedding=query_embedding, k=fetch_k)
    if mode == "hybrid_rerank":
        docs = repo.hybrid_search(query=query, query_embedding=query_embedding, k=fetch_k)
        ranked, _ = rerank_fn(query, docs)
        return ranked[:fetch_k]
    if mode == "pipeline":
        return _retrieve_pipeline(repo, query, fetch_k, query_embedder=query_embedder, rerank_fn=rerank_fn)
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def _rewrite_query(question: str) -> dict[str, Any]:
    """Run the query rewriter LLM and return structured fields.

    Returns dict with: optimized_query, stepback_query, slots, turn_intent.
    Falls back to original question on any failure.
    """
    try:
        from agents.query_rewriter_agent import (
            RewriteResult,
            _REWRITER_PROMPT,
            _REWRITER_SYSTEM,
        )
        from tools.llm_client import invoke_json

        prompt = _REWRITER_PROMPT.format(
            history_text="(no prior conversation)",
            question=question,
        )
        raw_json = invoke_json(prompt, system=_REWRITER_SYSTEM)

        if raw_json:
            try:
                result = RewriteResult.model_validate(raw_json)
            except Exception:
                result = RewriteResult(
                    optimized_query=str(raw_json.get("optimized_query", question)).strip() or question
                )
        else:
            result = RewriteResult(optimized_query=question)

        return {
            "optimized_query": result.optimized_query or question,
            "stepback_query": result.stepback_query,
            "slots": result.slots,
            "turn_intent": result.turn_intent,
        }
    except Exception:
        return {
            "optimized_query": question,
            "stepback_query": "",
            "slots": {},
            "turn_intent": "",
        }


def _retrieve_pipeline(
    repo: VectorRepository | InMemoryVectorRepository,
    question: str,
    fetch_k: int,
    *,
    query_embedder=embed_query,
    rerank_fn=rerank,
):
    """Pipeline-aware retrieval: rewrite → hybrid search → step-back merge → metadata boost → rerank.

    Mirrors the real RAG pipeline (QueryRewriterAgent → RetrieverAgent) but without
    route classification or workflow orchestration.
    """
    # Step 1: Rewrite query (LLM call)
    rewrite = _rewrite_query(question)
    query = rewrite["optimized_query"]
    stepback = rewrite["stepback_query"]
    slots = rewrite["slots"]

    # Step 2: Primary hybrid search with optimized query
    query_embedding = query_embedder(query)
    docs = repo.hybrid_search(query=query, query_embedding=query_embedding, k=fetch_k)

    # Step 3: Step-back merge if available
    if stepback:
        sb_embedding = query_embedder(stepback)
        sb_docs = repo.hybrid_search(query=stepback, query_embedding=sb_embedding, k=10)
        docs = _merge_and_deduplicate(docs, sb_docs)

    # Step 4: Cross-encoder rerank (or BM25 fallback)
    ranked, _ = rerank_fn(query, docs)

    return ranked[:fetch_k]


def _extract_evidence_ids(sample: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """Extract required, supporting, and hard_negative chunk_id sets from sample evidence."""
    evidence = sample.get("evidence", {})
    if evidence:
        required = {e["chunk_id"] for e in evidence.get("required", []) if "chunk_id" in e}
        supporting = {e["chunk_id"] for e in evidence.get("supporting", []) if "chunk_id" in e}
        hard_negative = {e["chunk_id"] for e in evidence.get("hard_negative_candidates", []) if "chunk_id" in e}
    else:
        # Fallback: treat ground_truth_chunk_ids as required
        required = set(sample.get("ground_truth_chunk_ids") or [])
        supporting = set()
        hard_negative = set()
    return required, supporting, hard_negative


def _graded_relevance_for_doc(
    doc,
    required_ids: set[str],
    supporting_ids: set[str],
    hard_negative_ids: set[str],
) -> int:
    """Return graded relevance: 3=essential, 2=supportive, 1=hard_negative, 0=irrelevant."""
    chunk_id = (doc.metadata or {}).get("chunk_id")
    if not chunk_id:
        return 0
    if chunk_id in required_ids:
        return 3
    if chunk_id in supporting_ids:
        return 2
    if chunk_id in hard_negative_ids:
        return 1
    return 0


def _average_precision(binary_relevance: list[int], k: int) -> float:
    if not binary_relevance:
        return 0.0
    running_hits = 0
    precision_sum = 0.0
    for rank, rel in enumerate(binary_relevance[:k], start=1):
        if rel:
            running_hits += 1
            precision_sum += running_hits / rank
    if running_hits == 0:
        return 0.0
    return precision_sum / running_hits


def _score_ranked_docs(
    docs,
    required_ids: set[str],
    supporting_ids: set[str],
    hard_negative_ids: set[str],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    grades: list[int] = []
    for doc in docs:
        grade = _graded_relevance_for_doc(doc, required_ids, supporting_ids, hard_negative_ids)
        grades.append(grade)

    metrics: dict[str, Any] = {}
    required_count = max(len(required_ids), 1)

    # Build ideal gains for nDCG: all required (3) first, then supporting (2), then hard_neg (1)
    all_ideal_gains = (
        [3] * len(required_ids)
        + [2] * len(supporting_ids)
        + [1] * len(hard_negative_ids)
    )
    all_ideal_gains.sort(reverse=True)

    for k in ks:
        top_grades = grades[:k]

        # recall@k: unique required IDs found in top-k / total required
        found_required: set[str] = set()
        for doc in docs[:k]:
            chunk_id = (doc.metadata or {}).get("chunk_id")
            if chunk_id and chunk_id in required_ids:
                found_required.add(chunk_id)
        recall = len(found_required) / required_count

        # precision@k: docs with grade >= 2 (essential + supportive) / k
        precision = sum(1 for g in top_grades if g >= 2) / k if k > 0 else 0.0

        # hit_rate@k: 1 if any grade-3 doc in top-k
        hit_rate = 1.0 if any(g == 3 for g in top_grades) else 0.0

        # mrr@k: 1/rank of first grade-3 doc
        first_essential_rank = None
        for rank_idx, g in enumerate(top_grades):
            if g == 3:
                first_essential_rank = rank_idx + 1
                break
        mrr = 1.0 / first_essential_rank if first_essential_rank else 0.0

        # map@k: average precision using grade >= 2 as hits
        binary_hits = [1 if g >= 2 else 0 for g in top_grades]
        ap = _average_precision(binary_hits, k)

        # ndcg@k: use actual grades as gain values
        dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(top_grades, start=1))
        ideal_gains_k = all_ideal_gains[:k]
        # Pad with zeros if fewer ideal items than k
        while len(ideal_gains_k) < k:
            ideal_gains_k.append(0)
        idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains_k, start=1))
        ndcg = dcg / idcg if idcg else 0.0

        # section_diversity@k: unique sections in top-k / k
        sections: set[str] = set()
        for doc in docs[:k]:
            section = (doc.metadata or {}).get("section_path") or (doc.metadata or {}).get("section")
            if section:
                sections.add(section)
        section_diversity = len(sections) / k if k > 0 else 0.0

        # max_precision@k: theoretical ceiling = min(relevant_count, k) / k
        relevant_count = len(required_ids) + len(supporting_ids)
        max_precision = min(relevant_count, k) / k if k > 0 else 0.0

        metrics[f"hit_rate@{k}"] = round(hit_rate, 4)
        metrics[f"recall@{k}"] = round(recall, 4)
        metrics[f"precision@{k}"] = round(precision, 4)
        metrics[f"max_precision@{k}"] = round(max_precision, 4)
        metrics[f"mrr@{k}"] = round(mrr, 4)
        metrics[f"map@{k}"] = round(ap, 4)
        metrics[f"ndcg@{k}"] = round(ndcg, 4)
        metrics[f"section_diversity@{k}"] = round(section_diversity, 4)

    return metrics


def _slice_key(sample: dict[str, Any]) -> tuple[str, str]:
    difficulty = sample.get("difficulty") or "unknown"
    content_types = sample.get("ground_truth_content_types") or []
    if content_types:
        content_family = "text"
    else:
        content_family = "mixed_or_unknown"
    return difficulty, content_family


def _aggregate_sample_metrics(items: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    if not items:
        return {"sample_count": 0}

    out: dict[str, Any] = {
        "sample_count": len(items),
        "latency_ms_mean": round(mean(item["latency_ms"] for item in items), 2),
        "latency_ms_median": round(median(item["latency_ms"] for item in items), 2),
        "latency_ms_p95": round(_percentile([item["latency_ms"] for item in items], 95), 2),
    }
    for k in ks:
        for metric in ("hit_rate", "recall", "precision", "max_precision", "mrr", "map", "ndcg", "section_diversity"):
            key = f"{metric}@{k}"
            out[key] = round(mean(item[key] for item in items), 4)
    return out


def _run_retrieval_eval_against_repo(
    *,
    repo: VectorRepository | InMemoryVectorRepository,
    samples: list[dict[str, Any]],
    mode: str,
    ks: tuple[int, ...],
    fetch_k: int,
    query_embedder=embed_query,
    rerank_fn=rerank,
) -> dict[str, Any]:
    if repo.count_chunks() == 0:
        raise RuntimeError("No indexed chunks found in the repository. Reindex before retrieval evaluation.")

    per_sample: list[dict[str, Any]] = []
    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        start = time.perf_counter()
        docs = _retrieve(
            repo,
            sample["question"],
            mode=mode,
            fetch_k=fetch_k,
            query_embedder=query_embedder,
            rerank_fn=rerank_fn,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        required_ids, supporting_ids, hard_negative_ids = _extract_evidence_ids(sample)

        metrics = _score_ranked_docs(
            docs=docs,
            required_ids=required_ids,
            supporting_ids=supporting_ids,
            hard_negative_ids=hard_negative_ids,
            ks=ks,
        )

        sample_result = {
            "id": sample["id"],
            "question": sample["question"],
            "difficulty": sample.get("difficulty"),
            "category": sample.get("category"),
            "latency_ms": round(latency_ms, 2),
            **metrics,
            "top_chunks": [
                {
                    "chunk_id": (doc.metadata or {}).get("chunk_id"),
                    "doc_id": (doc.metadata or {}).get("doc_id"),
                    "page": (doc.metadata or {}).get("page"),
                    "section": (doc.metadata or {}).get("section_path") or (doc.metadata or {}).get("section"),
                    "content_type": (doc.metadata or {}).get("content_type"),
                }
                for doc in docs[: max(ks)]
            ],
        }
        per_sample.append(sample_result)

        difficulty, content_family = _slice_key(sample)
        slices[f"difficulty:{difficulty}"].append(sample_result)
        slices[f"content:{content_family}"].append(sample_result)

    return {
        "scope": "retrieval_only",
        "mode": mode,
        "sample_count": len(per_sample),
        "fetch_k": fetch_k,
        **_aggregate_sample_metrics(per_sample, ks),
        "slices": {
            name: _aggregate_sample_metrics(items, ks)
            for name, items in sorted(slices.items())
        },
        "results": per_sample,
    }


def run_retrieval_eval(
    samples: list[dict[str, Any]],
    mode: str,
    ks: tuple[int, ...],
    fetch_k: int,
    *,
    repo: VectorRepository | InMemoryVectorRepository | None = None,
    query_embedder=None,
    rerank_fn=None,
) -> dict[str, Any]:
    query_embedder = query_embedder or embed_query
    rerank_fn = rerank_fn or rerank

    if repo is not None:
        return _run_retrieval_eval_against_repo(
            repo=repo,
            samples=samples,
            mode=mode,
            ks=ks,
            fetch_k=fetch_k,
            query_embedder=query_embedder,
            rerank_fn=rerank_fn,
        )

    db = SessionLocal()
    try:
        return _run_retrieval_eval_against_repo(
            repo=VectorRepository(db),
            samples=samples,
            mode=mode,
            ks=ks,
            fetch_k=fetch_k,
            query_embedder=query_embedder,
            rerank_fn=rerank_fn,
        )
    finally:
        db.close()


def _snapshot_path(name: str, mode: str | None = None) -> Path:
    base_dir = PIPELINE_SNAPSHOT_DIR if mode == "pipeline" else SNAPSHOT_DIR
    return base_dir / f"{name}.json"


def _git_commit_short() -> str:
    """Return short git commit hash, or 'unknown' if not in a repo."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        ).decode().strip()
    except Exception:
        # Inside container: .git not mounted. Try reading from a build-time stamp.
        stamp = Path(__file__).resolve().parent.parent / ".git_commit"
        if stamp.exists():
            return stamp.read_text().strip()
        return "unknown"


def save_snapshot(name: str, report: dict[str, Any], note: str = "") -> Path:
    report["snapshot_meta"] = {
        "git_commit": _git_commit_short(),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "note": note,
    }
    path = _snapshot_path(name, mode=report.get("mode"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_snapshot(name: str, mode: str | None = None) -> dict[str, Any] | None:
    candidate_paths = [_snapshot_path(name, mode=mode)]
    if mode == "pipeline":
        # Backward-compatible fallback for older pipeline snapshots saved flat.
        candidate_paths.append(_snapshot_path(name, mode=None))

    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def compare_reports(current: dict[str, Any], baseline: dict[str, Any], ks: tuple[int, ...]) -> None:
    print(f"\n{'=' * 90}")
    print("  Retrieval Evaluation Comparison")
    print(f"{'=' * 90}")
    print(f"{'Metric':<24} {'Baseline':>12} {'Current':>12} {'Delta':>12}")
    print("-" * 90)

    summary_keys = ["latency_ms_mean", "latency_ms_median", "latency_ms_p95"]
    for k in ks:
        summary_keys.extend(
            [f"hit_rate@{k}", f"recall@{k}", f"precision@{k}", f"max_precision@{k}", f"mrr@{k}", f"map@{k}", f"ndcg@{k}", f"section_diversity@{k}"]
        )

    for key in summary_keys:
        baseline_value = baseline.get(key, "N/A")
        current_value = current.get(key, "N/A")
        if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
            delta = current_value - baseline_value
            sign = "+" if delta > 0 else ""
            print(f"{key:<24} {baseline_value:>12} {current_value:>12} {sign}{delta:>11}")
        else:
            print(f"{key:<24} {str(baseline_value):>12} {str(current_value):>12}")


def _parse_ks(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one k value is required.")
    return tuple(sorted(set(values)))


def _run_mode_set(
    samples: list[dict[str, Any]],
    mode: str,
    ks: tuple[int, ...],
    fetch_k: int,
) -> dict[str, Any]:
    if mode == "all":
        # "all" runs the non-LLM modes only; use --mode pipeline separately
        non_llm_modes = [m for m in SUPPORTED_MODES if m != "pipeline"]
        runs = {
            candidate: run_retrieval_eval(
                samples=samples,
                mode=candidate,
                ks=ks,
                fetch_k=fetch_k,
            )
            for candidate in non_llm_modes
        }
        return {
            "scope": "retrieval_only",
            "mode": "all",
            "dataset_sample_count": len(samples),
            "runs": runs,
        }
    return run_retrieval_eval(
        samples=samples,
        mode=mode,
        ks=ks,
        fetch_k=fetch_k,
    )


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against Tier 3 gold labels.")
    parser.add_argument("--dataset", default=str(DEFAULT_TIER3_PATH), help="Tier 3 JSONL dataset path")
    parser.add_argument("--mode", choices=(*SUPPORTED_MODES, "all"), default="hybrid_rerank")
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="test", help="Benchmark split to score")
    parser.add_argument("--ks", default="1,3,5,10", help="Comma-separated k values, e.g. 1,3,5,10")
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K, help="Candidate retrieval depth before scoring")
    parser.add_argument("--limit", type=int, help="Optional max number of samples to score")
    parser.add_argument("--save", metavar="NAME", help="Save results as a named snapshot")
    parser.add_argument("--compare", metavar="NAME", help="Compare against a saved snapshot")
    parser.add_argument("--note", default="", help="Description of code changes for this snapshot")
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    samples = _load_tier3_samples(Path(args.dataset), split=args.split)
    if args.limit:
        samples = samples[: args.limit]

    report = _run_mode_set(
        samples=samples,
        mode=args.mode,
        ks=ks,
        fetch_k=max(args.fetch_k, max(ks)),
    )

    print(f"\n{'=' * 78}")
    print("  Retrieval Evaluation")
    print(f"{'=' * 78}")
    print(f"  Dataset:              {args.dataset}")
    print(f"  Split:                {args.split}")
    print(f"  Requested mode:       {args.mode}")
    print(f"  Samples:              {len(samples)}")

    if args.mode == "pipeline":
        print(f"  NOTE: pipeline mode runs the LLM query rewriter before retrieval.")
        print(f"        This measures rewriter + step-back + metadata boost + rerank combined.")

    if args.mode == "all":
        print("  NOTE: 'all' excludes pipeline mode (LLM calls). Run --mode pipeline separately.")
        print()
        non_llm_modes = [m for m in SUPPORTED_MODES if m != "pipeline"]
        for mode_name in non_llm_modes:
            run = report["runs"][mode_name]
            print(f"- {mode_name}: latency_mean={run['latency_ms_mean']}")
            for k in ks:
                print(
                    f"    @ {k:<2} hit={run[f'hit_rate@{k}']:<6} "
                    f"recall={run[f'recall@{k}']:<6} precision={run[f'precision@{k}']:<6} "
                    f"max_prec={run[f'max_precision@{k}']:<6} "
                    f"mrr={run[f'mrr@{k}']:<6} map={run[f'map@{k}']:<6} ndcg={run[f'ndcg@{k}']:<6} "
                    f"sec_div={run[f'section_diversity@{k}']:<6}"
                )
    else:
        print(f"  Mode:                 {report['mode']}")
        print(f"  Fetch depth:          {report['fetch_k']}")
        print(f"  Mean latency (ms):    {report['latency_ms_mean']}")
        print(f"  Median latency (ms):  {report['latency_ms_median']}")
        print(f"  P95 latency (ms):     {report['latency_ms_p95']}")
        print()
        for k in ks:
            print(
                f"  @ {k:<2}  "
                f"hit={report[f'hit_rate@{k}']:<6} "
                f"recall={report[f'recall@{k}']:<6} "
                f"precision={report[f'precision@{k}']:<6} "
                f"max_prec={report[f'max_precision@{k}']:<6} "
                f"mrr={report[f'mrr@{k}']:<6} "
                f"map={report[f'map@{k}']:<6} "
                f"ndcg={report[f'ndcg@{k}']:<6} "
                f"sec_div={report[f'section_diversity@{k}']:<6}"
            )
        print()
        print("  Slices:")
        for name, slice_report in sorted(report.get("slices", {}).items()):
            print(
                f"    {name:<24} samples={slice_report['sample_count']:<3} "
                f"recall@5={slice_report.get('recall@5', 0.0):<6} "
                f"mrr@5={slice_report.get('mrr@5', 0.0):<6} "
                f"sec_div@5={slice_report.get('section_diversity@5', 0.0):<6}"
            )

    if args.save:
        path = save_snapshot(args.save, report, note=args.note)
        print(f"\nSaved snapshot: {path}")

    if args.compare:
        baseline = load_snapshot(args.compare, mode=args.mode)
        if baseline is None:
            expected_dir = PIPELINE_SNAPSHOT_DIR if args.mode == "pipeline" else SNAPSHOT_DIR
            raise SystemExit(f"Snapshot '{args.compare}' not found in {expected_dir}")
        if args.mode == "all":
            print("\nComparison for --mode all is not auto-rendered. Compare saved JSON snapshots per mode.")
        else:
            compare_reports(report, baseline, ks)


if __name__ == "__main__":
    main()
