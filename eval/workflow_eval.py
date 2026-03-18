"""
End-to-end workflow evaluation harness.

This script runs the real LangGraph workflow with the configured vector store,
retriever, executor, prompt budgeting, and LLM call path. Use it to compare
final answer behavior after chunking/indexing changes.

Usage:
    python -m eval.workflow_eval
    python -m eval.workflow_eval --semantic-cache
    python -m eval.workflow_eval --save baseline_old
    python -m eval.workflow_eval --compare baseline_old
    python -m eval.workflow_eval --cases eval/workflow_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from statistics import mean
from typing import overload

from core.workflow_service import generate_trace_id, get_workflow_service
from db.repositories import InMemoryChatRepository, VectorRepository
from db.session import SessionLocal
from tools.cache import semantic_cache_override

EVAL_DIR = Path(__file__).resolve().parent / "workflow_snapshots"

DEFAULT_CASES = [
    {
        "id": "dka_signs",
        "query": "What are the signs of diabetic ketoacidosis?",
        "expected_keywords": ["ketoacidosis", "ketones", "dehydration"],
        "expect_table_hit": False,
    },
    {
        "id": "metformin_contra",
        "query": "What are the contraindications of metformin?",
        "expected_keywords": ["metformin", "contraind", "renal"],
        "expect_table_hit": False,
    },
    {
        "id": "acetaminophen_dose",
        "query": "What is the adult acetaminophen dosage?",
        "expected_keywords": ["acetaminophen", "dose", "mg"],
        "expect_table_hit": True,
    },
    {
        "id": "warfarin_interactions",
        "query": "Show the interaction information for warfarin.",
        "expected_keywords": ["warfarin", "interaction"],
        "expect_table_hit": True,
    },
]


def _normalize_tokens(text: str) -> set[str]:
    import re

    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return {token for token in normalized.split() if len(token) > 2}


def _keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return hits / len(expected_keywords)


def _load_cases(path: str | None) -> list[dict]:
    if path is None:
        return list(DEFAULT_CASES)

    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Workflow eval cases not found: {path}")

    if eval_path.suffix == ".json":
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON workflow case file must contain a list of objects.")
        return data

    cases = []
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def _build_live_workflow_runner(repo: VectorRepository):
    workflow = get_workflow_service()

    def _runner(case: dict) -> dict:
        session_id = f"eval-{uuid.uuid4()}"
        return workflow.run(
            question=case["query"],
            session_id=session_id,
            trace_id=generate_trace_id(),
            history=[],
            summary="",
            facts=[],
            chat_repo=InMemoryChatRepository(),
            vector_repo=repo,
        )

    return _runner


def _run_cases_with_runner(
    cases: list[dict],
    *,
    use_semantic_cache: bool,
    workflow_runner,
) -> dict:
    results = []
    with semantic_cache_override(enabled=use_semantic_cache, version="workflow-eval", clear=True):
        for case in cases:
            result = workflow_runner(case)
            docs = result.get("documents", [])
            retrieved_tables = sum(
                1
                for doc in docs
                if (doc.metadata or {}).get("is_table")
                or (doc.metadata or {}).get("content_type") == "table"
            )
            answer = result.get("generation", "")

            results.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "route": result.get("route"),
                    "source": result.get("source"),
                    "response": answer,
                    "response_keyword_recall": round(
                        _keyword_recall(answer, case.get("expected_keywords", [])),
                        3,
                    ),
                    "retrieved_doc_count": len(docs),
                    "retrieved_table_count": retrieved_tables,
                    "table_hit_ok": (
                        True
                        if not case.get("expect_table_hit")
                        else retrieved_tables > 0
                    ),
                    "top_chunks": [
                        {
                            "chunk_id": (doc.metadata or {}).get("chunk_id"),
                            "content_type": (doc.metadata or {}).get("content_type"),
                            "page": (doc.metadata or {}).get("page"),
                            "section": (doc.metadata or {}).get("section_path") or (doc.metadata or {}).get("section"),
                        }
                        for doc in docs[:5]
                    ],
                }
            )

    avg_recall = mean(item["response_keyword_recall"] for item in results) if results else 0.0
    table_hit_rate = mean(1.0 if item["table_hit_ok"] else 0.0 for item in results) if results else 0.0

    return {
        "scope": "end_to_end_workflow",
        "semantic_cache_enabled": use_semantic_cache,
        "case_count": len(results),
        "avg_response_keyword_recall": round(avg_recall, 3),
        "table_hit_rate": round(table_hit_rate, 3),
        "results": results,
    }


@overload
def run_cases(cases: list[dict], *, use_semantic_cache: bool = False) -> dict: ...


def run_cases(
    cases: list[dict],
    *,
    use_semantic_cache: bool = False,
    workflow_runner=None,
) -> dict:
    if workflow_runner is not None:
        return _run_cases_with_runner(
            cases,
            use_semantic_cache=use_semantic_cache,
            workflow_runner=workflow_runner,
        )

    db = SessionLocal()
    try:
        repo = VectorRepository(db)
        chunk_count = repo.count_chunks()
        if chunk_count == 0:
            raise RuntimeError("No indexed chunks found in the database. Reindex before workflow evaluation.")

        return _run_cases_with_runner(
            cases,
            use_semantic_cache=use_semantic_cache,
            workflow_runner=_build_live_workflow_runner(repo),
        )
    finally:
        db.close()


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
    print(f"\n{'='*72}")
    print("  Workflow Evaluation Comparison")
    print(f"{'='*72}")
    metrics = [
        ("Case count", "case_count"),
        ("Avg response keyword recall", "avg_response_keyword_recall"),
        ("Table hit rate", "table_hit_rate"),
    ]
    print(f"{'Metric':<34} {'Baseline':>12} {'Current':>12} {'Delta':>12}")
    print("-" * 72)
    for label, key in metrics:
        baseline_value = baseline.get(key, "N/A")
        current_value = current.get(key, "N/A")
        if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
            delta = current_value - baseline_value
            sign = "+" if delta > 0 else ""
            print(f"{label:<34} {baseline_value:>12} {current_value:>12} {sign}{delta:>11}")
        else:
            print(f"{label:<34} {str(baseline_value):>12} {str(current_value):>12}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end workflow evaluation.")
    parser.add_argument("--cases", help="JSON or JSONL file containing workflow eval cases")
    parser.add_argument("--save", metavar="NAME", help="Save results as a named snapshot")
    parser.add_argument("--compare", metavar="NAME", help="Compare against a saved snapshot")
    parser.add_argument(
        "--semantic-cache",
        action="store_true",
        help="Enable semantic cache during workflow eval. Disabled by default for trustworthy debugging.",
    )
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    report = run_cases(cases, use_semantic_cache=args.semantic_cache)

    print(f"\n{'='*60}")
    print("  Workflow Evaluation")
    print(f"{'='*60}")
    print(f"  Cases:                    {report['case_count']}")
    print(f"  Avg response recall:      {report['avg_response_keyword_recall']}")
    print(f"  Table hit rate:           {report['table_hit_rate']}")
    print(f"  Semantic cache enabled:   {report['semantic_cache_enabled']}")
    print()

    for item in report["results"]:
        print(f"- {item['id']}: route={item['route']} source={item['source']} recall={item['response_keyword_recall']} table_hit={item['table_hit_ok']}")

    if args.save:
        path = save_snapshot(args.save, report)
        print(f"\nSaved snapshot: {path}")

    if args.compare:
        baseline = load_snapshot(args.compare)
        if baseline is None:
            raise SystemExit(f"Snapshot '{args.compare}' not found in {EVAL_DIR}")
        compare_reports(report, baseline)


if __name__ == "__main__":
    main()
