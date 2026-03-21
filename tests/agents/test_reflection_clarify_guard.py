"""
Regression tests for ReflectionAgent clarify-route guard.

Bug: executor→reflection edge is unconditional in the LangGraph workflow.
ReflectionAgent received clarify-turn state (empty documents, clarification
question as 'generation') and could mark needs_retry=True, routing back to
retrieval — contradicting ADR-0007 which states clarify turns skip retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clarify_state(**overrides):
    return {
        "question": "what should I take",
        "optimized_query": "what should I take",
        "generation": "Could you clarify whether you're asking about OTC pain relief or a prescription?",
        "source": "Clarification Request",
        "route": "clarify",
        "needs_clarification": True,
        "documents": [],
        "attempts": {"reflection": 0, "executor": 0},
        "session_id": "s1",
        "trace_id": "t1",
        "status_events": [],
        **overrides,
    }


def test_reflection_agent_skips_llm_call_for_clarify_route():
    """
    ReflectionAgent must NOT call the LLM judge for clarify-route turns.
    Calling the judge with empty docs and a clarification question as the
    answer risks flagging needs_retry=True and routing back to retrieval.
    """
    with patch("agents.reflection_agent.invoke_json") as mock_invoke:
        from agents.reflection_agent import ReflectionAgent

        state = _clarify_state()
        ReflectionAgent(state)

    mock_invoke.assert_not_called(), (
        "ReflectionAgent called the LLM judge for a clarify-route turn. "
        "This wastes tokens and risks triggering a retry loop."
    )


def test_reflection_agent_does_not_set_needs_retry_for_clarify_route():
    """
    needs_retry must remain False after ReflectionAgent processes a clarify turn.
    """
    with patch("agents.reflection_agent.invoke_json") as mock_invoke:
        mock_invoke.return_value = {
            "failure_category": "irrelevant",
            "is_relevant": False,
            "has_hallucinations": False,
            "confidence": 0.0,
            "grounding_score": 0.0,
            "suggested_focus": "",
            "feedback": "answer is not relevant",
        }

        from agents.reflection_agent import ReflectionAgent

        state = _clarify_state()
        result = ReflectionAgent(state)

    assert result.get("needs_retry") is False, (
        f"needs_retry={result.get('needs_retry')} — ReflectionAgent should not "
        "set needs_retry=True for a clarify-route turn."
    )
