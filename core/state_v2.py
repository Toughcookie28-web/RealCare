from __future__ import annotations

from typing import Any, TypedDict
from langchain_core.documents import Document


class AgentStateV2(TypedDict):
    trace_id: str
    session_id: str
    question: str
    raw_query: str
    optimized_query: str
    stepback_query: str
    route: str
    planned_route: str
    route_decision_reason: str
    post_retrieval_route: str
    generation: str
    source: str
    citations: list[dict[str, Any]]
    documents: list[Document]
    retrieval_confidence: float
    retrieval_candidate_count: int
    conversation_history: list[dict[str, Any]]
    summary: str
    facts: list[dict[str, Any]]
    safety_flags: dict[str, Any]
    attempts: dict[str, int]
    current_tool: str | None
    status_events: list[dict[str, Any]]
    semantic_cache_hit: bool
    needs_retry: bool
    reflection_feedback: str
    reflection_suggested_focus: str
    reflection_confidence: float
    reflection_failure_category: str
    grounding_score: float
    session_intent: str
    turn_intent: str
    slots: dict[str, str]
    slot_coverage: float
    query_context: str
    sub_questions: list[dict[str, str]]
    decomposition_enabled: bool
    chat_repo: Any | None
    vector_repo: Any | None
    user_id: str | None
    episodic_memories: list[dict[str, Any]]
    long_term_memory_repo: Any | None
    intent_confidence: float
    needs_clarification: bool
    clarification_question: str


def initialize_state(session_id: str, trace_id: str) -> AgentStateV2:
    return {
        'trace_id': trace_id,
        'session_id': session_id,
        'question': '',
        'raw_query': '',
        'optimized_query': '',
        'stepback_query': '',
        'route': '',
        'planned_route': '',
        'route_decision_reason': '',
        'post_retrieval_route': '',
        'generation': '',
        'source': '',
        'citations': [],
        'documents': [],
        'retrieval_confidence': 0.0,
        'retrieval_candidate_count': 0,
        'conversation_history': [],
        'summary': '',
        'facts': [],
        'safety_flags': {'blocked': False, 'reason': None, 'risk_level': 'low'},
        'attempts': {'reflection': 0, 'executor': 0},
        'current_tool': None,
        'status_events': [],
        'semantic_cache_hit': False,
        'needs_retry': False,
        'reflection_feedback': '',
        'reflection_suggested_focus': '',
        'reflection_confidence': 0.0,
        'reflection_failure_category': '',
        'grounding_score': 0.0,
        'session_intent': '',
        'turn_intent': '',
        'slots': {},
        'slot_coverage': 0.0,
        'query_context': '',
        'sub_questions': [],
        'decomposition_enabled': False,
        'chat_repo': None,
        'vector_repo': None,
        'user_id': None,
        'episodic_memories': [],
        'long_term_memory_repo': None,
        'intent_confidence': 1.0,
        'needs_clarification': False,
        'clarification_question': '',
    }


def reset_query_state(state: AgentStateV2, question: str) -> AgentStateV2:
    state.update(
        {
            'question': question,
            'raw_query': question,
            'optimized_query': '',
            'stepback_query': '',
            'route': '',
            'planned_route': '',
            'route_decision_reason': '',
            'post_retrieval_route': '',
            'generation': '',
            'source': '',
            'citations': [],
            'documents': [],
            'retrieval_confidence': 0.0,
            'retrieval_candidate_count': 0,
            'attempts': {'reflection': 0, 'executor': 0},
            'current_tool': None,
            'status_events': [],
            'semantic_cache_hit': False,
            'safety_flags': {'blocked': False, 'reason': None, 'risk_level': 'low'},
            'needs_retry': False,
            'reflection_feedback': '',
            'reflection_suggested_focus': '',
            'reflection_confidence': 0.0,
            'reflection_failure_category': '',
            'grounding_score': 0.0,
            'session_intent': '',
            'turn_intent': '',
            'slots': {},
            'slot_coverage': 0.0,
            'query_context': '',
            'sub_questions': [],
            'episodic_memories': [],
            'intent_confidence': 1.0,
            'needs_clarification': False,
            'clarification_question': '',
        }
    )
    return state
