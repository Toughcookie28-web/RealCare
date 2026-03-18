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
        self.scorer = scorer or score_live_payload

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
