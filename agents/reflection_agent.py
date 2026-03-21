from __future__ import annotations

import logging

from pydantic import BaseModel, field_validator

from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_json

logger = logging.getLogger(__name__)

_REFLECTION_SYSTEM = "You are a strict medical QA reviewer. Evaluate answer quality and safety."

_REFLECTION_PROMPT = """Evaluate this medical Q&A pair with access to the source chunks used to generate the answer.

Question: {question}
Answer: {answer}

Source chunks used for this answer:
{chunk_context}

Judge the response on these criteria:
1. Is the answer relevant to the question?
2. Does it contain hallucinations? Specifically: are there claims in the answer that CANNOT be traced to any of the source chunks above?
3. Is it complete enough to be useful?
4. Is it safe (no dangerous unsupported medical advice)?

Return JSON ONLY with these keys:
- "is_relevant": boolean
- "has_hallucinations": boolean — true if any answer claim is absent from the source chunks
- "failure_category": one of "none", "irrelevant", "hallucination", "incomplete", "unsafe"
- "suggested_focus": if the answer is incomplete or irrelevant, write a specific search query that would find the missing information. Otherwise empty string.
- "confidence": 0.0-1.0 score for overall answer quality
- "grounding_score": 0.0-1.0 — what fraction of the answer's key claims are supported by the source chunks (1.0 = fully grounded, 0.0 = no grounding)
- "feedback": brief explanation of your judgment
"""


class ReflectionResult(BaseModel):
    """Structured output from the reflection judge."""
    is_relevant: bool = True
    has_hallucinations: bool = False
    failure_category: str = "none"
    suggested_focus: str = ""
    confidence: float = 0.0
    grounding_score: float = 1.0
    feedback: str = ""

    @field_validator('failure_category', mode='before')
    @classmethod
    def coerce_category(cls, v):
        valid = {'none', 'irrelevant', 'hallucination', 'incomplete', 'unsafe'}
        val = str(v or 'none').strip().lower()
        return val if val in valid else 'none'

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator('grounding_score', mode='before')
    @classmethod
    def coerce_grounding_score(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 1.0

    @field_validator('is_relevant', 'has_hallucinations', mode='before')
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes')
        return bool(v)

    @field_validator('suggested_focus', 'feedback', mode='before')
    @classmethod
    def coerce_str(cls, v):
        return str(v or '').strip()


def _parse_reflection_response(raw: dict) -> ReflectionResult:
    """Parse LLM judge output into structured ReflectionResult."""
    if not raw:
        return ReflectionResult()

    if 'failure_category' not in raw:
        is_rel = raw.get('is_relevant', True)
        has_hal = raw.get('has_hallucinations', False)
        if isinstance(is_rel, str):
            is_rel = is_rel.strip().lower() in ('true', '1', 'yes')
        if isinstance(has_hal, str):
            has_hal = has_hal.strip().lower() in ('true', '1', 'yes')
        if has_hal:
            raw['failure_category'] = 'hallucination'
        elif not is_rel:
            raw['failure_category'] = 'irrelevant'
        else:
            raw['failure_category'] = 'none'

    try:
        return ReflectionResult.model_validate(raw)
    except Exception:
        return ReflectionResult()


def _build_chunk_context(docs: list, max_chars: int = 2000) -> str:
    """Build a condensed view of retrieved chunks for the reflection judge."""
    if not docs:
        return "(no source chunks — answer was generated from memory or cache)"
    parts = []
    total = 0
    for i, doc in enumerate(docs[:5]):
        snippet = doc.page_content[:400].strip()
        section = doc.metadata.get("section", "unknown")
        part = f"[Chunk {i+1} — {section}]\n{snippet}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def ReflectionAgent(state: AgentStateV2) -> AgentStateV2:
    # Clarify turns carry no source chunks — skip the judge to avoid
    # a false needs_retry that would route back to retrieval (ADR-0007).
    if state.get('route') == 'clarify' or state.get('source') == 'Clarification Request':
        state['needs_retry'] = False
        return state

    question = state.get('question', '')
    answer = state.get('generation', '')
    docs = state.get('documents', [])

    chunk_context = _build_chunk_context(docs)
    judge_prompt = _REFLECTION_PROMPT.format(question=question, answer=answer, chunk_context=chunk_context)
    raw = invoke_json(judge_prompt, system=_REFLECTION_SYSTEM)
    result = _parse_reflection_response(raw)

    attempts = state.get('attempts', {'reflection': 0, 'executor': 0})
    attempts['reflection'] = attempts.get('reflection', 0) + 1
    state['attempts'] = attempts

    needs_retry = result.failure_category != 'none' and attempts['reflection'] < 2
    state['needs_retry'] = needs_retry
    state['reflection_feedback'] = result.feedback
    state['reflection_suggested_focus'] = result.suggested_focus
    state['reflection_confidence'] = result.confidence
    state['reflection_failure_category'] = result.failure_category
    state['grounding_score'] = result.grounding_score

    logger.info(
        "reflection_complete",
        extra={
            "failure_category": result.failure_category,
            "confidence": result.confidence,
            "grounding_score": result.grounding_score,
            "needs_retry": needs_retry,
            "suggested_focus": result.suggested_focus[:80] if result.suggested_focus else "",
            "attempt": attempts['reflection'],
        },
    )

    return state
