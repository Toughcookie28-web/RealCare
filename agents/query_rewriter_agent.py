from __future__ import annotations

import json
import logging

from pydantic import BaseModel, field_validator

from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_json

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    """Structured output from the query rewriter."""
    optimized_query: str
    stepback_query: str = ""
    route: str = "vector"
    session_intent: str = ""
    turn_intent: str = ""
    slots: dict[str, str] = {}
    query_context: str = ""
    reasoning: str = ""

    @field_validator('optimized_query', 'stepback_query', 'route',
                     'session_intent', 'turn_intent', 'query_context',
                     'reasoning', mode='before')
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return ""
        return str(v).strip()

    @field_validator('slots', mode='before')
    @classmethod
    def coerce_slots(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return {}


_REWRITER_SYSTEM = (
    "You are a medical query understanding assistant for a textbook retrieval system. "
    "You analyze user questions, rewrite them for optimal retrieval, classify intent, "
    "and extract structured information. "
    "Respond ONLY with valid JSON, no other text."
)

_REWRITER_PROMPT = """Prior conversation summary:
{summary_text}

Recent conversation history:
{history_text}

User question:
{question}

STEP 1 — REASON about the user's true intent.
Think through these questions and write your analysis in the "reasoning" field:
- What is the user actually asking? If they use pronouns ("it", "that drug", "the same condition"), resolve them using the prior conversation summary and recent history.
- Check the medical context in the summary (conditions, drugs, procedures) — does the current question relate to entities already discussed?
- Are there medical abbreviations to expand? (MI → myocardial infarction, BP → blood pressure, CHF → congestive heart failure, DVT → deep vein thrombosis, PE → pulmonary embolism, COPD → chronic obstructive pulmonary disease, DM → diabetes mellitus, HTN → hypertension, CAD → coronary artery disease, etc.)
- Is this a follow-up to an earlier topic (check conversation trajectory), or a new question?
- Does this relate to any open questions from the summary?
- What type of information does the user need? (definition, mechanism, dosage, comparison, side effects, etc.)

STEP 2 — Based on your reasoning, produce the structured output.
Follow these rules:
1. OPTIMIZED QUERY: Rewrite the question with resolved references, expanded abbreviations, and medical synonyms in parentheses where helpful. Preserve specificity (dosages, populations, timeframes, drug names).
2. STEP-BACK QUERY: For mechanisms of action, drug comparisons, treatment rationale, dosage lookups, or timeline questions, generate a broader principle-level query. Otherwise leave empty.
3. ROUTE: Where to search.
   - "vector": medical/clinical questions answerable from a textbook (most queries)
   - "web": current events, latest guidelines, recent outbreaks, news
   - "literature": explicit requests for research papers, studies, PubMed, systematic reviews
   - "memory": questions about previous conversation ("what did we discuss", "remind me")
   - "chitchat": greetings, thanks, non-medical small talk with NO medical content
4. SESSION INTENT: Overall goal of this conversation based on history. Empty if no history.
5. TURN INTENT: What this specific message asks for (e.g., "dosage_lookup", "drug_comparison", "definition", "side_effects", "mechanism_of_action", "greeting").
6. SLOTS: Structured medical entities. Keys: drug, condition, population, aspect, timeframe, comparison_target. Add others if relevant.
7. QUERY CONTEXT: 1-2 sentence description of what the user needs, for the downstream answer generator.

STEP 3 - Return JSON as final OUTPUT:
{{"reasoning": "...", "optimized_query": "...", "stepback_query": "...", "route": "...", "session_intent": "...", "turn_intent": "...", "slots": {{}}, "query_context": "..."}}

Final output examples:
- "what does aspirin do for heart attacks?" → {{"reasoning": "User asks about aspirin's role in heart attacks. Expand to include synonym myocardial infarction. This is a mechanism question, so generate a step-back query about the broader drug class.", "optimized_query": "aspirin mechanism of action for myocardial infarction (heart attack)", "stepback_query": "pharmacology of antiplatelet agents in cardiovascular disease", "route": "vector", "session_intent": "", "turn_intent": "mechanism_of_action", "slots": {{"drug": "aspirin", "condition": "myocardial infarction", "aspect": "mechanism"}}, "query_context": "User wants to understand how aspirin works in treating heart attacks"}}
- "what is hypertension?" → {{"reasoning": "Simple definition lookup for hypertension. Add synonym high blood pressure. No step-back needed.", "optimized_query": "hypertension (high blood pressure) definition and overview", "stepback_query": "", "route": "vector", "session_intent": "", "turn_intent": "definition", "slots": {{"condition": "hypertension"}}, "query_context": "User wants a basic definition of hypertension"}}
- "what did we talk about last time?" → {{"reasoning": "User wants to recall previous conversation. Route to memory, no retrieval needed.", "optimized_query": "what did we talk about last time", "stepback_query": "", "route": "memory", "session_intent": "", "turn_intent": "conversation_recall", "slots": {{}}, "query_context": "User wants to recall previous conversation topics"}}
- "hello!" → {{"reasoning": "Simple greeting with no medical content.", "optimized_query": "hello", "stepback_query": "", "route": "chitchat", "session_intent": "", "turn_intent": "greeting", "slots": {{}}, "query_context": "User is greeting"}}"""


def parse_rewrite_response(raw: str, fallback_query: str) -> RewriteResult:
    """Parse LLM rewrite response with tiered graceful degradation."""
    if not raw or not raw.strip():
        return RewriteResult(optimized_query=fallback_query)

    text = raw.strip()

    # Tier 1: Try JSON parsing + Pydantic validation
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    if parsed is not None:
        try:
            return RewriteResult.model_validate(parsed)
        except Exception:
            if isinstance(parsed, dict) and parsed.get('optimized_query'):
                return RewriteResult(optimized_query=str(parsed['optimized_query']).strip())

    # Tier 3: JSON parsing failed, use raw text as query
    return RewriteResult(optimized_query=text)


def QueryRewriterAgent(state: AgentStateV2) -> AgentStateV2:
    question = state.get('question', '')
    history = state.get('conversation_history', [])[-6:]
    summary = state.get('summary', '')

    history_text = '\n'.join(
        [f"{item.get('role')}: {item.get('content')}" for item in history]
    ) if history else '(no prior conversation)'

    summary_text = summary if summary else '(no prior summary)'

    prompt = _REWRITER_PROMPT.format(
        history_text=history_text,
        question=question,
        summary_text=summary_text,
    )

    raw_json = invoke_json(
        prompt, system=_REWRITER_SYSTEM, pydantic_model=RewriteResult,
    )

    if raw_json:
        try:
            result = RewriteResult.model_validate(raw_json)
        except Exception:
            result = RewriteResult(
                optimized_query=str(raw_json.get('optimized_query', question)).strip() or question
            )
    else:
        result = RewriteResult(optimized_query=question)

    state['optimized_query'] = result.optimized_query or question
    state['stepback_query'] = result.stepback_query
    state['route'] = result.route or 'vector'
    state['session_intent'] = result.session_intent
    state['turn_intent'] = result.turn_intent
    state['slots'] = result.slots
    state['query_context'] = result.query_context
    logger.info(
        "query_rewrite_complete",
        extra={
            "original": question,
            "optimized": result.optimized_query,
            "stepback": result.stepback_query,
            "stepback_generated": bool(result.stepback_query),
            "reasoning": result.reasoning,
            "route": result.route,
            "session_intent": result.session_intent,
            "turn_intent": result.turn_intent,
            "slots": result.slots,
            "query_context": result.query_context,
        },
    )
    return state
