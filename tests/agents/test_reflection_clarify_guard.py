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


# ---------------------------------------------------------------------------
# Route-based and source-based skip guard tests (extended coverage)
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.parametrize("route", ["chitchat", "memory"])
def test_reflection_agent_skips_llm_call_for_no_retrieval_routes(route):
    """ReflectionAgent must not invoke the LLM judge for chitchat/memory routes
    which produce no retrieved source chunks."""
    with patch("agents.reflection_agent.invoke_json") as mock_invoke:
        from agents.reflection_agent import ReflectionAgent

        state = _clarify_state(route=route, source="some source")
        ReflectionAgent(state)

    mock_invoke.assert_not_called()


@pytest.mark.parametrize("source", [
    "LLM Medical Reasoning",
    "Memory (Conversation History)",
    "Memory (no history)",
])
def test_reflection_agent_skips_llm_call_for_no_chunk_sources(source):
    """ReflectionAgent must not judge answers from memory/LLM-only sources
    where retrieved chunks are absent or irrelevant."""
    with patch("agents.reflection_agent.invoke_json") as mock_invoke:
        from agents.reflection_agent import ReflectionAgent

        state = _clarify_state(route="vector", source=source)
        ReflectionAgent(state)

    mock_invoke.assert_not_called()


@pytest.mark.parametrize("route,source", [
    ("chitchat", "LLM Medical Reasoning"),
    ("memory", "Memory (Conversation History)"),
])
def test_reflection_agent_needs_retry_false_for_skipped_routes(route, source):
    """needs_retry must be False for all skipped route/source combinations."""
    with patch("agents.reflection_agent.invoke_json"):
        from agents.reflection_agent import ReflectionAgent

        state = _clarify_state(route=route, source=source)
        result = ReflectionAgent(state)

    assert result.get("needs_retry") is False


# ---------------------------------------------------------------------------
# _strip_citations tests
# ---------------------------------------------------------------------------

def test_strip_citations_removes_source_block():
    from agents.reflection_agent import _strip_citations

    text = "Metformin lowers blood sugar.\n\n**Sources:**\n- Section: Treatment"
    assert _strip_citations(text) == "Metformin lowers blood sugar."


def test_strip_citations_removes_inline_source_marker():
    from agents.reflection_agent import _strip_citations

    text = "Take 500mg twice daily.\n\nSource: Drug guide p.12"
    assert _strip_citations(text) == "Take 500mg twice daily."


def test_strip_citations_no_marker_returns_original():
    from agents.reflection_agent import _strip_citations

    text = "No citations here."
    assert _strip_citations(text) == "No citations here."


# ---------------------------------------------------------------------------
# Hallucination caveat injection
# ---------------------------------------------------------------------------

def _rag_state(**overrides):
    """Minimal state representing a completed vector-route answer."""
    return {
        "question": "What is the dose of aspirin?",
        "generation": "Aspirin is typically dosed at 81mg daily for cardioprotection.",
        "source": "LLM + Retrieved Evidence",
        "route": "vector",
        "documents": [],
        "attempts": {"reflection": 0, "executor": 0},
        "session_id": "s2",
        "trace_id": "t2",
        "status_events": [],
        **overrides,
    }


def test_hallucination_caveat_injected_when_no_focus():
    """When judge flags hallucination but provides no suggested_focus,
    a safety caveat must be appended to the generation."""
    hallucination_response = {
        "failure_category": "hallucination",
        "is_relevant": True,
        "has_hallucinations": True,
        "confidence": 0.4,
        "grounding_score": 0.3,
        "suggested_focus": "",   # no focus → no retry, inject caveat
        "feedback": "claim contradicts source",
    }
    with patch("agents.reflection_agent.invoke_json", return_value=hallucination_response):
        from agents.reflection_agent import ReflectionAgent

        state = _rag_state()
        result = ReflectionAgent(state)

    assert result.get("needs_retry") is False
    assert "⚠️" in result.get("generation", ""), (
        "Safety caveat should be injected when hallucination detected but no focus for retry"
    )


def test_hallucination_caveat_not_duplicated():
    """Caveat must not be appended if it already exists in the generation."""
    caveat = (
        '\n\n⚠️ *Note: parts of this answer may not be fully supported by the '
        'retrieved medical sources. Please verify with a qualified healthcare professional.*'
    )
    hallucination_response = {
        "failure_category": "hallucination",
        "is_relevant": True,
        "has_hallucinations": True,
        "confidence": 0.4,
        "grounding_score": 0.3,
        "suggested_focus": "",
        "feedback": "claim contradicts source",
    }
    with patch("agents.reflection_agent.invoke_json", return_value=hallucination_response):
        from agents.reflection_agent import ReflectionAgent

        state = _rag_state(generation="Some answer" + caveat)
        result = ReflectionAgent(state)

    assert result["generation"].count("⚠️") == 1, "Caveat must not be duplicated"
