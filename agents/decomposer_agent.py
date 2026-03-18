"""Plan-and-Execute: query decomposition agent (STUBBED).

This agent detects multi-part questions and decomposes them into
sub-questions for independent retrieval and answer generation.

Currently DISABLED by default (ENABLE_QUERY_DECOMPOSITION=false).
The decompose-execute-synthesize loop is not yet wired into the
LangGraph workflow. This module provides the interface and detection
logic for future activation when multi-PDF ingestion is ready.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from core.state_v2 import AgentStateV2

logger = logging.getLogger(__name__)

# Compound question indicators
_COMPOUND_PATTERNS = [
    r'\band\b.*\b(?:what|how|why|when|which|compare|explain|describe)\b',
    r'\b(?:compare|contrast|difference between)\b',
    r'\b(?:also|additionally|furthermore)\b',
    r'(?:1\.|2\.|a\)|b\))',
    r'^(?:what|how|why|when|which)\b.*\band\b',
]
_COMPOUND_RE = [re.compile(p, re.IGNORECASE) for p in _COMPOUND_PATTERNS]

# Conjunction splitters
_SPLIT_CONJUNCTIONS = re.compile(
    r'\b(?:and also|and what|and how|and explain|and describe|and compare)\b',
    re.IGNORECASE,
)


def is_multi_part_question(question: str) -> bool:
    """Heuristic check: does this question contain multiple distinct sub-questions?"""
    if not question:
        return False
    matches = sum(1 for p in _COMPOUND_RE if p.search(question))
    return matches >= 1 and len(question.split()) >= 8


def decompose_question(question: str) -> list[dict[str, str]]:
    """Split a compound question into sub-questions.

    Returns a list of dicts with 'question' and 'route' keys.
    This is a rule-based stub — future versions will use LLM decomposition.
    """
    if not is_multi_part_question(question):
        return [{"question": question, "route": "vector"}]

    parts = _SPLIT_CONJUNCTIONS.split(question)
    parts = [p.strip().rstrip('?').strip() + '?' for p in parts if p.strip()]

    if len(parts) <= 1:
        return [{"question": question, "route": "vector"}]

    return [{"question": part, "route": "vector"} for part in parts]


def DecomposerAgent(state: AgentStateV2) -> AgentStateV2:
    """Plan-and-Execute decomposer node.

    When disabled (default), passes through without modification.
    When enabled, decomposes multi-part questions into sub_questions.
    """
    enabled = state.get('decomposition_enabled', False)

    if not enabled:
        state['sub_questions'] = state.get('sub_questions', [])
        return state

    question = state.get('question', '')
    if is_multi_part_question(question):
        sub_qs = decompose_question(question)
        state['sub_questions'] = sub_qs
        logger.info(
            "query_decomposed",
            extra={"original": question[:80], "sub_count": len(sub_qs)},
        )
    else:
        state['sub_questions'] = []

    return state
