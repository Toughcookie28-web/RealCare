"""
RAGAS LLM-as-judge evaluation harness for the MediGenius RAG pipeline.

This script runs RAGAS evaluation metrics (faithfulness, answer relevancy,
context precision, context recall) on top of the existing retrieval + workflow
pipeline. It uses your Tier 3 golden dataset and evaluates the full
retrieve-then-generate path.

Metrics:
- Faithfulness:       Is the answer grounded in retrieved context?
- Answer Relevancy:   Does the answer address the question?
- Context Precision:  Are retrieved chunks relevant to the question?
- Context Recall:     Do retrieved chunks cover the ground truth?

Usage:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --limit 5               # quick test with 5 samples
    python -m eval.ragas_eval --save baseline_v1       # save snapshot
    python -m eval.ragas_eval --compare baseline_v1    # compare against snapshot
    python -m eval.ragas_eval --judge-model gpt-4o-mini  # cheaper judge
    python -m eval.ragas_eval --retrieval-only          # skip LLM generation, eval retrieval only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from agents.retriever_agent import rerank
from db.repositories import VectorRepository
from db.session import SessionLocal
from eval.run_tracker import persist_saved_run
from tools.embedding_client import embed_query

DEFAULT_TIER3_PATH = Path(__file__).resolve().parent / "golden" / "v1" / "tier3_rag.jsonl"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "ragas_snapshots"
SUPPORTED_SPLITS = ("dev", "test", "all")

# Retrieval-only metrics (no LLM generation needed)
RETRIEVAL_METRICS = [context_precision, context_recall]
# Full pipeline metrics (need LLM generation)
FULL_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _load_env() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _load_tier3_samples(path: Path, split: str = "test") -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Tier 3 file not found: {path}")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS}.")
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row_split = row.get("split")
        if split != "all" and row_split not in (None, split):
            continue
        if row.get("question") and row.get("ground_truth_contexts"):
            samples.append(row)
    if not samples:
        raise RuntimeError(f"No valid samples in {path}")
    return samples


def _retrieve_for_query(
    repo: VectorRepository,
    query: str,
    fetch_k: int = 20,
    top_k: int = 5,
) -> tuple[list[str], list[dict[str, Any]], float]:
    """Retrieve and rerank, return (context_texts, chunk_metadata, latency_ms)."""
    start = time.perf_counter()
    query_embedding = embed_query(query)
    docs = repo.hybrid_search(query=query, query_embedding=query_embedding, k=fetch_k)
    ranked, _scores, _rerank_method = rerank(query, docs)
    latency_ms = (time.perf_counter() - start) * 1000.0

    top_docs = ranked[:top_k]
    contexts = [doc.page_content for doc in top_docs]
    metadata = [
        {
            "chunk_id": (doc.metadata or {}).get("chunk_id"),
            "page": (doc.metadata or {}).get("page"),
            "section": (doc.metadata or {}).get("section_path") or (doc.metadata or {}).get("section"),
            "content_type": (doc.metadata or {}).get("content_type"),
        }
        for doc in top_docs
    ]
    return contexts, metadata, latency_ms


def _generate_answer(query: str, contexts: list[str]) -> str:
    """Generate answer using the project LLM client."""
    from tools.llm_client import invoke_llm

    context_block = "\n\n---\n\n".join(contexts)
    prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
    system = (
        "You are a medical education assistant. Answer the question based strictly "
        "on the provided context. If the context does not contain enough information, "
        "say so. Cite relevant parts of the context."
    )
    return invoke_llm(prompt, system=system)


def run_ragas_eval(
    samples: list[dict[str, Any]],
    *,
    retrieval_only: bool = False,
    judge_model: str = "gpt-4o-mini",
    fetch_k: int = 20,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run RAGAS evaluation on Tier 3 samples against the live DB."""
    _load_env()
    db = SessionLocal()
    try:
        repo = VectorRepository(db)
        if repo.count_chunks() == 0:
            raise RuntimeError("No indexed chunks. Reindex before evaluation.")

        ragas_samples = []
        per_sample_results = []

        for i, sample in enumerate(samples):
            question = sample["question"]
            ground_truth = sample.get("ground_truth", "")
            reference_contexts = sample.get("ground_truth_contexts", [])

            print(f"  [{i + 1}/{len(samples)}] {sample['id']}: {question[:60]}...")

            # Retrieve
            retrieved_contexts, chunk_meta, latency_ms = _retrieve_for_query(
                repo,
                question,
                fetch_k=fetch_k,
                top_k=top_k,
            )

            # Generate answer (skip if retrieval-only)
            response = ""
            if not retrieval_only:
                try:
                    response = _generate_answer(question, retrieved_contexts)
                except Exception as e:
                    print(f"    LLM generation failed: {e}")
                    response = f"[generation_error: {e}]"

            # Build RAGAS sample
            ragas_sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
                response=response if not retrieval_only else "N/A",
                reference=ground_truth,
            )
            ragas_samples.append(ragas_sample)

            per_sample_results.append(
                {
                    "id": sample["id"],
                    "question": question,
                    "difficulty": sample.get("difficulty"),
                    "category": sample.get("category"),
                    "response": response[:500] if not retrieval_only else "[retrieval_only]",
                    "retrieved_chunk_count": len(retrieved_contexts),
                    "latency_ms": round(latency_ms, 2),
                    "top_chunks": chunk_meta,
                }
            )

        # Run RAGAS evaluation
        metrics = RETRIEVAL_METRICS if retrieval_only else FULL_METRICS
        dataset = EvaluationDataset(samples=ragas_samples)

        print(f"\n  Running RAGAS evaluation with {judge_model}...")
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        # Clear empty OPENAI_BASE_URL which confuses the client
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ.pop("OPENAI_BASE_URL", None)

        judge_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model))

        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
        )

        # Extract per-sample scores from RAGAS result
        scores_df = ragas_result.to_pandas()
        metric_names = [m.name for m in metrics]

        for i, row in scores_df.iterrows():
            for metric_name in metric_names:
                if metric_name in row:
                    per_sample_results[i][metric_name] = (
                        round(float(row[metric_name]), 4)
                        if row[metric_name] is not None and str(row[metric_name]) != "nan"
                        else None
                    )

        # Aggregate scores
        aggregate = {}
        for metric_name in metric_names:
            values = [r[metric_name] for r in per_sample_results if r.get(metric_name) is not None]
            aggregate[metric_name] = round(mean(values), 4) if values else None
            aggregate[f"{metric_name}_scored_count"] = len(values)

        # Aggregate latency
        latencies = [r["latency_ms"] for r in per_sample_results]
        aggregate["latency_ms_mean"] = round(mean(latencies), 2) if latencies else 0.0

        # Slice by difficulty
        difficulty_slices: dict[str, list[dict]] = {}
        for r in per_sample_results:
            d = r.get("difficulty", "unknown")
            difficulty_slices.setdefault(d, []).append(r)

        slices = {}
        for diff, items in sorted(difficulty_slices.items()):
            slice_agg = {"sample_count": len(items)}
            for metric_name in metric_names:
                values = [r[metric_name] for r in items if r.get(metric_name) is not None]
                slice_agg[metric_name] = round(mean(values), 4) if values else None
            slices[f"difficulty:{diff}"] = slice_agg

        report = {
            "scope": "ragas_llm_judge",
            "mode": "retrieval_only" if retrieval_only else "full_pipeline",
            "judge_model": judge_model,
            "sample_count": len(per_sample_results),
            "fetch_k": fetch_k,
            "top_k": top_k,
            **aggregate,
            "slices": slices,
            "results": per_sample_results,
        }
        return report
    finally:
        db.close()


def _git_commit_short() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:
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
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_snapshot(name: str) -> dict[str, Any] | None:
    path = SNAPSHOT_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(current: dict, baseline: dict, metric_names: list[str]) -> None:
    print(f"\n{'=' * 78}")
    print("  RAGAS Evaluation Comparison")
    print(f"{'=' * 78}")
    print(f"{'Metric':<28} {'Baseline':>12} {'Current':>12} {'Delta':>12}")
    print("-" * 78)

    keys = metric_names + ["latency_ms_mean", "sample_count"]
    for key in keys:
        b = baseline.get(key, "N/A")
        c = current.get(key, "N/A")
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            delta = c - b
            sign = "+" if delta > 0 else ""
            print(f"{key:<28} {b:>12.4f} {c:>12.4f} {sign}{delta:>11.4f}")
        else:
            print(f"{key:<28} {str(b):>12} {str(c):>12}")

    # Slice comparison
    for slice_name in sorted(set(list(current.get("slices", {})) + list(baseline.get("slices", {})))):
        b_slice = baseline.get("slices", {}).get(slice_name, {})
        c_slice = current.get("slices", {}).get(slice_name, {})
        if b_slice or c_slice:
            print(f"\n  {slice_name}:")
            for key in metric_names:
                b = b_slice.get(key, "N/A")
                c = c_slice.get(key, "N/A")
                if isinstance(b, (int, float)) and isinstance(c, (int, float)):
                    delta = c - b
                    sign = "+" if delta > 0 else ""
                    print(f"    {key:<24} {b:>12.4f} {c:>12.4f} {sign}{delta:>11.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS LLM-as-judge evaluation.")
    parser.add_argument("--dataset", default=str(DEFAULT_TIER3_PATH), help="Tier 3 JSONL path")
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="test", help="Benchmark split to score")
    parser.add_argument("--limit", type=int, help="Max samples to evaluate")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model for RAGAS judge")
    parser.add_argument("--retrieval-only", action="store_true", help="Only evaluate retrieval (skip generation)")
    parser.add_argument("--fetch-k", type=int, default=20, help="Retrieval depth")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks to keep after rerank")
    parser.add_argument("--save", metavar="NAME", help="Save results as named snapshot")
    parser.add_argument("--compare", metavar="NAME", help="Compare against saved snapshot")
    parser.add_argument("--note", default="", help="Description of code changes for this snapshot")
    args = parser.parse_args()

    samples = _load_tier3_samples(Path(args.dataset), split=args.split)
    if args.limit:
        samples = samples[: args.limit]

    print(f"\n{'=' * 60}")
    print("  RAGAS Evaluation")
    print(f"{'=' * 60}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  Split:          {args.split}")
    print(f"  Samples:        {len(samples)}")
    print(f"  Judge model:    {args.judge_model}")
    print(f"  Mode:           {'retrieval_only' if args.retrieval_only else 'full_pipeline'}")
    print(f"  Fetch-k:        {args.fetch_k}")
    print(f"  Top-k:          {args.top_k}")
    print()

    report = run_ragas_eval(
        samples,
        retrieval_only=args.retrieval_only,
        judge_model=args.judge_model,
        fetch_k=args.fetch_k,
        top_k=args.top_k,
    )

    # Print results
    metric_names = [m.name for m in (RETRIEVAL_METRICS if args.retrieval_only else FULL_METRICS)]

    print(f"\n{'=' * 60}")
    print("  Results")
    print(f"{'=' * 60}")
    for metric_name in metric_names:
        val = report.get(metric_name, "N/A")
        scored = report.get(f"{metric_name}_scored_count", 0)
        val_str = "N/A" if val is None else f"{val:.4f}"
        print(f"  {metric_name:<24} {val_str:<10} ({scored}/{report['sample_count']} scored)")
    print(f"  {'latency_ms_mean':<24} {report['latency_ms_mean']:.2f}")

    print("\n  Slices:")
    for name, slice_report in sorted(report.get("slices", {}).items()):
        parts = [f"n={slice_report['sample_count']}"]
        for mn in metric_names:
            v = slice_report.get(mn)
            parts.append(f"{mn}={v:.4f}" if v is not None else f"{mn}=N/A")
        print(f"    {name:<28} {' '.join(parts)}")

    # Per-sample detail
    print("\n  Per-sample:")
    for r in report["results"]:
        scores = " ".join(
            f"{mn}={r.get(mn, 'N/A')}" if not isinstance(r.get(mn), float) else f"{mn}={r[mn]:.3f}"
            for mn in metric_names
        )
        print(f"    {r['id']:<20} {scores}")

    if args.save:
        path = save_snapshot(args.save, report, note=args.note)
        print(f"\n  Saved snapshot: {path}")
        try:
            persist_saved_run(
                eval_type="ragas",
                report=report,
                split=args.split,
                dataset_path=args.dataset,
                snapshot_path=str(path),
                session_factory=SessionLocal,
                judge_model=args.judge_model,
            )
        except Exception as exc:
            print(f"  Warning: benchmark run tracking failed: {exc}")

    if args.compare:
        baseline = load_snapshot(args.compare)
        if baseline is None:
            raise SystemExit(f"Snapshot '{args.compare}' not found in {SNAPSHOT_DIR}")
        compare_reports(report, baseline, metric_names)

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
