from __future__ import annotations

import logging

from pydantic import BaseModel, field_validator

from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_json

logger = logging.getLogger(__name__)

# Strip citation block before judging — evaluate the medical content only
_CITATION_STRIP_MARKERS = ('\n\nSource:', '\n\n**Sources:**', '\n\n[Source:')


def _strip_citations(text: str) -> str:
    for marker in _CITATION_STRIP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            return text[:idx]
    return text


_REFLECTION_SYSTEM = "You are a strict medical QA reviewer. Evaluate answer quality and safety."

_REFLECTION_PROMPT = """Evaluate this medical Q&A pair.

Question: {question}
Answer: {answer}

Source chunks retrieved for this answer:
{chunk_context}

Judge on these criteria:
1. RELEVANCE: Does the answer directly address what was asked in the question? An answer that discusses a related topic but misses the actual question is irrelevant.
2. HALLUCINATION: Does the answer contain claims that CONTRADICT the source chunks, or claims that are medically implausible given the question context? Note: well-established medical facts not explicitly in the chunks are acceptable — only flag genuine contradictions or implausible claims.
3. COMPLETENESS: Is the answer sufficiently complete to be useful for the question asked?
4. SAFETY: Does the answer contain dangerous or unsupported medical advice?

Return JSON ONLY with these keys:
- "is_relevant": boolean
- "has_hallucinations": boolean — true ONLY if claims contradict source chunks or are medically implausible, NOT just because they are absent from chunks
- "failure_category": one of "none", "irrelevant", "hallucination", "incomplete", "unsafe"
- "suggested_focus": if the answer is incomplete, irrelevant, or has hallucinations, write a specific search query that would find the missing or corrected information. Otherwise empty string.
- "confidence": 0.0-1.0 overall answer quality score
- "grounding_score": 0.0-1.0 — fraction of answer claims consistent with source chunks (1.0 = fully consistent, 0.0 = contradicts chunks)
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


def _build_chunk_context(docs: list, max_chars: int = 4000) -> str:
    """Build a condensed view of retrieved chunks for the reflection judge."""
    if not docs:
        return "(no source chunks — answer was generated from memory or cache)"
    parts = []
    total = 0
    for i, doc in enumerate(docs[:5]):
        snippet = doc.page_content[:800].strip()
        section = doc.metadata.get("section", "unknown")
        part = f"[Chunk {i+1} — {section}]\n{snippet}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def ReflectionAgent(state: AgentStateV2) -> AgentStateV2:
    # Skip the judge for routes that carry no retrieved source chunks.
    # - clarify: no retrieval was performed (ADR-0007)
    # - chitchat: no documents retrieved; judge would always flag poor groundedness
    # - memory: answer is from conversation history, not chunks; grounding is meaningless
    if state.get('route') in ('clarify', 'chitchat', 'memory') or state.get('source') in (
        'Clarification Request', 'LLM Medical Reasoning',
        'Memory (Conversation History)', 'Memory (no history)'
    ):
        state['needs_retry'] = False
        return state

    question = state.get('question', '')
    answer = _strip_citations(state.get('generation', ''))
    docs = state.get('documents', [])

    chunk_context = _build_chunk_context(docs)
    judge_prompt = _REFLECTION_PROMPT.format(question=question, answer=answer, chunk_context=chunk_context)
    raw = invoke_json(judge_prompt, system=_REFLECTION_SYSTEM)
    result = _parse_reflection_response(raw)

    attempts = dict(state.get('attempts', {'reflection': 0, 'executor': 0}))
    attempts['reflection'] = attempts.get('reflection', 0) + 1
    state['attempts'] = attempts

    # Only retry when the judge has a concrete suggested focus to improve retrieval.
    # Hallucination without a focus = same query again = no improvement.
    has_focus = bool(result.suggested_focus.strip())
    needs_retry = (
        result.failure_category in ('incomplete', 'irrelevant')
        or (result.failure_category == 'hallucination' and has_focus)
    ) and attempts['reflection'] < 2
    state['needs_retry'] = needs_retry

    _CAVEAT = (
        '\n\n⚠️ *Note: parts of this answer may not be fully supported by the '
        'retrieved medical sources. Please verify with a qualified healthcare professional.*'
    )
    _UNSAFE_CAVEAT = (
        '\n\n⚠️ *Warning: this answer may contain unsupported or potentially unsafe medical '
        'information. Please consult a qualified healthcare professional before acting on it.*'
    )

    # Inject caveat when the judge flags a problem but no retry will fix it:
    # - hallucination (no retry OR retry exhausted): answer may contain incorrect claims
    # - unsafe: answer may contain dangerous advice — always inject, never retry
    if result.failure_category == 'unsafe':
        generation = state.get('generation', '')
        if _UNSAFE_CAVEAT not in generation:
            state['generation'] = generation + _UNSAFE_CAVEAT
    elif result.failure_category == 'hallucination' and not needs_retry:
        generation = state.get('generation', '')
        if _CAVEAT not in generation:
            state['generation'] = generation + _CAVEAT

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
