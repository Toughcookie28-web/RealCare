from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Callable

from core.settings import get_settings
from observability.metrics import (
    LIVE_ANSWER_RELEVANCE,
    LIVE_CONTEXT_COVERAGE_PROXY,
    LIVE_CONTEXT_PRECISION_PROXY,
    LIVE_GROUNDEDNESS,
    LIVE_JUDGE_FAILURES,
    LIVE_JUDGE_LATENCY,
    LIVE_JUDGED_REQUESTS,
)
from tools.llm_client import invoke_json

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _overlap_recall(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _f1_like(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(right)
    recall = overlap / len(left)
    return (2 * precision * recall) / (precision + recall)


def _cap(v: Any) -> float:
    return min(1.0, max(0.0, float(v)))


def build_live_judge_payload(question: str, trace_id: str, result: dict[str, Any]) -> dict[str, Any]:
    contexts: list[str] = []
    for doc in result.get("documents", []) or []:
        page_content = getattr(doc, "page_content", "")
        if page_content:
            contexts.append(str(page_content))

    return {
        "question": question,
        "answer": str(result.get("generation", "") or result.get("response", "") or ""),
        "route": str(result.get("route", "") or ""),
        "trace_id": trace_id,
        "contexts": contexts,
    }


def score_live_payload(payload: dict[str, Any]) -> dict[str, float]:
    question_tokens = _tokenize(str(payload.get("question", "")))
    answer_tokens = _tokenize(str(payload.get("answer", "")))
    contexts = [str(item) for item in payload.get("contexts", [])]
    context_union_tokens = _tokenize(" ".join(contexts))

    context_hits = 0
    combined_query_answer = question_tokens | answer_tokens
    for context in contexts:
        if _tokenize(context) & combined_query_answer:
            context_hits += 1

    context_precision_proxy = (context_hits / len(contexts)) if contexts else 0.0

    return {
        "answer_relevance": _f1_like(question_tokens, answer_tokens),
        "groundedness": _overlap_recall(answer_tokens, context_union_tokens),
        "context_precision_proxy": context_precision_proxy,
        "context_coverage_proxy": _overlap_recall(question_tokens, context_union_tokens),
    }


_LLM_SYSTEM_PROMPT = (
    "You are a medical RAG quality evaluator. "
    "Score the quality of a question-answer pair given retrieved context. "
    "Return JSON only."
)

_LLM_PROMPT_TEMPLATE = """\
Evaluate the following medical RAG response and return a JSON object with exactly these keys:
  answer_relevance, groundedness, context_precision, context_coverage, reasoning

All scores are floats from 0.0 to 1.0.

Scoring rubric:
- answer_relevance: Does the answer semantically address the question?
  1.0 = fully answers, 0.5 = partial/drifts, 0.0 = off-topic
- groundedness: Are medical claims consistent with the retrieved context?
  1.0 = all claims consistent, 0.5 = most supported with some from general knowledge, 0.0 = contradicts context
- context_precision: What fraction of retrieved chunks are actually relevant to the question?
  1.0 = every chunk useful, 0.5 = half relevant, 0.0 = no chunks relevant
- context_coverage: Does the context contain enough info to fully answer the question?
  1.0 = fully covers, 0.5 = partial coverage, 0.0 = missing key information

Question: {question}

Answer: {answer}

Retrieved context chunks:
{contexts}

Return ONLY valid JSON: {{"answer_relevance": 0.85, "groundedness": 0.90, "context_precision": 0.75, "context_coverage": 0.80, "reasoning": "..."}}
"""

_MAX_CHUNKS = 5
_MAX_CHARS_PER_CHUNK = 300
_MAX_ANSWER_CHARS = 500
_MAX_QUESTION_CHARS = 300


def score_live_payload_llm(payload: dict[str, Any]) -> dict[str, float]:
    """Score a live judge payload using a single LLM call.

    Falls back to score_live_payload (BM25) on any exception or missing keys.
    Scores are capped to [0.0, 1.0].
    """
    try:
        question = str(payload.get("question", ""))[:_MAX_QUESTION_CHARS]
        answer = str(payload.get("answer", ""))[:_MAX_ANSWER_CHARS]
        raw_contexts = [str(c) for c in payload.get("contexts", [])]
        truncated_contexts = [c[:_MAX_CHARS_PER_CHUNK] for c in raw_contexts[:_MAX_CHUNKS]]

        # No documents retrieved — skip LLM call; BM25 scorer handles the empty-context case.
        if not truncated_contexts:
            return score_live_payload(payload)

        contexts_text = "\n".join(
            f"[{i + 1}] {chunk}" for i, chunk in enumerate(truncated_contexts)
        )

        prompt = _LLM_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            contexts=contexts_text,
        )

        result = invoke_json(prompt, system=_LLM_SYSTEM_PROMPT)

        # "reasoning" is intentionally excluded — it is a free-text debug field, not a scored metric.
        required_keys = {"answer_relevance", "groundedness", "context_precision", "context_coverage"}
        if not required_keys.issubset(result.keys()):
            logger.warning(
                "live_judge_llm_missing_keys",
                extra={"got_keys": list(result.keys()), "expected": list(required_keys)},
            )
            return score_live_payload(payload)

        return {
            "answer_relevance": _cap(result["answer_relevance"]),
            "groundedness": _cap(result["groundedness"]),
            "context_precision_proxy": _cap(result["context_precision"]),
            "context_coverage_proxy": _cap(result["context_coverage"]),
        }

    except Exception:
        logger.warning("live_judge_llm_failed_falling_back_to_bm25", exc_info=True)
        return score_live_payload(payload)


class LiveJudgeService:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        sampling_ratio: float | None = None,
        scorer: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    ):
        settings = None
        if enabled is None or sampling_ratio is None:
            settings = get_settings()
        self.enabled = enabled if enabled is not None else bool(settings.live_judge_enabled)
        self.sampling_ratio = (
            sampling_ratio if sampling_ratio is not None else float(settings.live_judge_sampling_ratio)
        )
        self.scorer = scorer or score_live_payload_llm

    def should_judge(self, trace_id: str) -> bool:
        if not self.enabled:
            return False
        if self.sampling_ratio <= 0.0:
            return False
        if self.sampling_ratio >= 1.0:
            return True

        digest = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:8]
        bucket = int(digest, 16) / 0xFFFFFFFF
        return bucket <= self.sampling_ratio

    def judge(self, payload: dict[str, Any]) -> dict[str, float] | None:
        trace_id = str(payload.get("trace_id", "") or "")
        route = str(payload.get("route", "") or "unknown")
        if not self.should_judge(trace_id):
            return None

        start = time.perf_counter()
        try:
            scores = self.scorer(payload)
            LIVE_JUDGED_REQUESTS.labels(route=route).inc()
            LIVE_ANSWER_RELEVANCE.labels(route=route).observe(scores["answer_relevance"])
            LIVE_GROUNDEDNESS.labels(route=route).observe(scores["groundedness"])
            LIVE_CONTEXT_PRECISION_PROXY.labels(route=route).observe(scores["context_precision_proxy"])
            LIVE_CONTEXT_COVERAGE_PROXY.labels(route=route).observe(scores["context_coverage_proxy"])
            LIVE_JUDGE_LATENCY.labels(route=route).observe((time.perf_counter() - start) * 1000.0)
            return scores
        except Exception:
            LIVE_JUDGE_FAILURES.labels(route=route).inc()
            logger.exception("live_judge_failed", extra={"trace_id": trace_id, "route": route})
            return None
