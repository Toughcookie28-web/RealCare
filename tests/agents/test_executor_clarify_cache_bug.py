"""
Regression tests for the clarify-route + semantic-cache interaction.

Bug: _should_use_semantic_cache() returned True even when route='clarify',
so repeat clarification queries hit the cache and served a stale medical
answer instead of the clarification question.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _base_state(**overrides):
    return {
        "question": "what should I take",
        "optimized_query": "what should I take",
        "session_id": "s1",
        "trace_id": "t1",
        "attempts": {"reflection": 0, "executor": 0},
        "status_events": [],
        **overrides,
    }


# ---------------------------------------------------------------------------
# P1: cache must not intercept clarify-route turns
# ---------------------------------------------------------------------------

def test_clarify_route_not_served_from_semantic_cache(monkeypatch):
    """
    When route='clarify' and the semantic cache holds a prior medical answer
    for the same query, ExecutorAgent must return the clarification question,
    NOT the cached answer.
    """
    fake_cache = MagicMock()
    fake_cache.get.return_value = "Metformin is used for type-2 diabetes."  # stale answer

    monkeypatch.setattr("agents.executor_agent.get_semantic_cache", lambda: fake_cache)

    from agents.executor_agent import ExecutorAgent

    state = _base_state(
        route="clarify",
        clarification_question="Are you asking about OTC pain relief or a prescription?",
        needs_clarification=True,
        intent_confidence=0.3,
    )
    result = ExecutorAgent(state)

    assert result["source"] == "Clarification Request", (
        f"Expected 'Clarification Request', got '{result['source']}'. "
        "Cache is bypassing the clarify guard."
    )
    assert "Metformin" not in result["generation"], (
        "Cached medical answer leaked into a clarification turn."
    )


def test_should_use_semantic_cache_returns_false_for_clarify_route():
    """
    _should_use_semantic_cache() must return False when route='clarify'.
    This is the unit-level guard.
    """
    from agents.executor_agent import _should_use_semantic_cache

    state = _base_state(
        route="clarify",
        needs_clarification=True,
        attempts={"reflection": 0, "executor": 0},
    )
    assert _should_use_semantic_cache(state) is False, (
        "_should_use_semantic_cache should return False for route='clarify'"
    )
