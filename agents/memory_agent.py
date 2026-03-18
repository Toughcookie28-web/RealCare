from __future__ import annotations

import logging
import re

from pydantic import BaseModel, field_validator

from core.settings import get_settings
from core.state_v2 import AgentStateV2
from tools.embedding_client import embed_query
from tools.llm_client import invoke_json, invoke_llm

logger = logging.getLogger(__name__)


class MedicalContext(BaseModel):
    """Structured medical entities mentioned in conversation."""
    conditions: list[str] = []
    drugs: list[str] = []
    procedures: list[str] = []
    populations: list[str] = []

    @field_validator('conditions', 'drugs', 'procedures', 'populations', mode='before')
    @classmethod
    def coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        return []


class StructuredSummary(BaseModel):
    """Structured conversation summary for downstream query understanding."""
    session_goal: str = ""
    medical_context: MedicalContext = MedicalContext()
    key_findings: list[str] = []
    open_questions: list[str] = []
    conversation_trajectory: str = ""

    @field_validator('session_goal', 'conversation_trajectory', mode='before')
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return ""
        return str(v).strip()

    @field_validator('key_findings', 'open_questions', mode='before')
    @classmethod
    def coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        return []


_SUMMARY_SYSTEM = (
    "You are a medical conversation summarizer. "
    "You extract structured context from clinical conversations. "
    "Respond ONLY with valid JSON, no other text."
)

_SUMMARY_PROMPT = """Conversation:
{conversation_text}

STEP 1 — Read the conversation above and identify:
- What is the user's overall research goal?
- What medical entities (conditions, drugs, procedures, patient populations) were discussed?
- What key facts or answers were established?
- What questions remain unanswered?
- How did the conversation topic evolve?

STEP 2 — Extract your findings into the following JSON fields:
- "session_goal": The user's overall research goal (1 sentence)
- "medical_context": Object with:
  - "conditions": list of diseases, symptoms, diagnoses mentioned
  - "drugs": list of medications, supplements mentioned
  - "procedures": list of treatments, tests, procedures mentioned
  - "populations": list of patient groups, age ranges, risk factors mentioned
- "key_findings": List of important facts established so far (2-5 items)
- "open_questions": Unresolved questions from the user (0-3 items)
- "conversation_trajectory": How the topic evolved (1 sentence, e.g., "Started with X, narrowed to Y, now exploring Z")

STEP 3 — Return JSON as final output:
{{"session_goal": "...", "medical_context": {{"conditions": [], "drugs": [], "procedures": [], "populations": []}}, "key_findings": [], "open_questions": [], "conversation_trajectory": "..."}}"""


def _build_structured_summary(older: list[dict]) -> str:
    """Generate a structured summary from older conversation turns."""
    conversation_text = '\n'.join(
        [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in older]
    )
    prompt = _SUMMARY_PROMPT.format(conversation_text=conversation_text)
    raw = invoke_json(prompt, system=_SUMMARY_SYSTEM, pydantic_model=StructuredSummary)

    if raw:
        try:
            summary = StructuredSummary.model_validate(raw)
        except Exception:
            logger.warning("Structured summary validation failed, using flat summary")
            return _flat_summary_fallback(conversation_text)
        return _format_structured_summary(summary)

    return _flat_summary_fallback(conversation_text)


def _format_structured_summary(summary: StructuredSummary) -> str:
    """Format a StructuredSummary into a text block for the rewriter."""
    parts: list[str] = []

    if summary.session_goal:
        parts.append(f"Session goal: {summary.session_goal}")

    ctx = summary.medical_context
    entities: list[str] = []
    if ctx.conditions:
        entities.append(f"conditions: {', '.join(ctx.conditions)}")
    if ctx.drugs:
        entities.append(f"drugs: {', '.join(ctx.drugs)}")
    if ctx.procedures:
        entities.append(f"procedures: {', '.join(ctx.procedures)}")
    if ctx.populations:
        entities.append(f"populations: {', '.join(ctx.populations)}")
    if entities:
        parts.append(f"Medical context: {'; '.join(entities)}")

    if summary.key_findings:
        findings = '; '.join(summary.key_findings)
        parts.append(f"Key findings: {findings}")

    if summary.open_questions:
        questions = '; '.join(summary.open_questions)
        parts.append(f"Open questions: {questions}")

    if summary.conversation_trajectory:
        parts.append(f"Trajectory: {summary.conversation_trajectory}")

    return '\n'.join(parts)


def _flat_summary_fallback(conversation_text: str) -> str:
    """Fallback to flat summary when structured summary fails."""
    summary = invoke_llm(
        f"Summarize the prior medical conversation in <= 150 words with key topics, "
        f"drugs, conditions, and unresolved questions.\n\n{conversation_text}",
        system="You are a concise clinical conversation summarizer.",
    )
    return summary or ""


def MemoryAgent(state: AgentStateV2) -> AgentStateV2:
    history = state.get('conversation_history', [])

    # Keep recent 3 turns exact in active prompt window.
    recent = history[-6:]
    older = history[:-6]

    if len(history) > 8:
        summary = _build_structured_summary(older)
        if summary:
            state['summary'] = summary
            logger.info("structured_summary_generated", extra={"length": len(summary)})

    state['conversation_history'] = recent

    facts = _extract_facts(state.get('question', ''))
    merged = {fact['key']: fact for fact in state.get('facts', [])}
    for fact in facts:
        merged[fact['key']] = fact
    state['facts'] = list(merged.values())

    # Long-term memory loading
    settings = get_settings()
    user_id = state.get('user_id')
    lt_repo = state.get('long_term_memory_repo')

    if settings.long_term_memory_enabled and user_id and lt_repo is not None:
        _load_long_term_memory(state, user_id, lt_repo, settings)

    return state


def _load_long_term_memory(state, user_id: str, lt_repo, settings) -> None:
    """Load user-scoped facts and episodic memories into state."""
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    # 1. Load structured long-term facts — merge into existing facts
    try:
        lt_facts = lt_repo.get_user_facts(user_id)
        if lt_facts:
            merged = {f['key']: f for f in state.get('facts', [])}
            for f in lt_facts:
                existing = merged.get(f['key'])
                if not existing or f['confidence'] >= existing.get('confidence', 0):
                    merged[f['key']] = f
            state['facts'] = list(merged.values())
            _logger.info(
                "long_term_facts_loaded",
                extra={"user_id": user_id, "count": len(lt_facts)},
            )
    except Exception:
        _logger.warning("long_term_facts_load_failed", extra={"user_id": user_id}, exc_info=True)

    # 2. Load episodic memories — vector search for similar past conversations
    try:
        question = state.get('question', '')
        if question:
            query_emb = embed_query(question)
            episodes = lt_repo.search_similar_memories(
                user_id, query_emb, k=settings.long_term_memory_recall_k
            )
            state['episodic_memories'] = episodes
            _logger.info(
                "episodic_memories_loaded",
                extra={"user_id": user_id, "count": len(episodes)},
            )
    except Exception:
        _logger.warning("episodic_memories_load_failed", extra={"user_id": user_id}, exc_info=True)
        state['episodic_memories'] = []


def _extract_facts(question: str) -> list[dict]:
    text = question.lower()
    facts = []

    age_match = re.search(r"\b(i am|i'm)\s+(\d{1,3})\b", text)
    if age_match:
        facts.append({'key': 'age', 'value': age_match.group(2), 'confidence': 0.9})

    allergy_match = re.search(r"allerg(y|ic)\s+to\s+([a-z\s-]+)", text)
    if allergy_match:
        facts.append({'key': 'allergy', 'value': allergy_match.group(2).strip(), 'confidence': 0.85})

    condition_match = re.search(r"i have\s+([a-z0-9\s-]{3,40})", text)
    if condition_match:
        facts.append({'key': 'condition', 'value': condition_match.group(1).strip(), 'confidence': 0.65})

    return facts
