import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import patch, call


def test_retry_uses_suggested_focus_as_query():
    """On retry, the retriever should use reflection's suggested_focus, not the original query."""
    from agents.retriever_agent import RetrieverAgent
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()

    state = {
        "question": "What is the pediatric dosage of aspirin?",
        "optimized_query": "pediatric aspirin dosing guidelines",
        "needs_retry": True,
        "reflection_suggested_focus": "aspirin dose milligrams children weight-based",
        "attempts": {"reflection": 1, "executor": 0},
        "vector_repo": repo,
        "stepback_query": "general pharmacology of aspirin",
        "documents": [],
        "retrieval_confidence": 0.0,
        "retrieval_candidate_count": 0,
        "route": "vector",
        "planned_route": "vector",
        "post_retrieval_route": "executor",
        "route_decision_reason": "",
    }

    with patch("agents.retriever_agent.embed_query") as mock_embed, \
         patch.object(repo, "hybrid_search", return_value=[]) as mock_search:
        mock_embed.return_value = [0.0] * 768
        RetrieverAgent(state)

        # The first (and only) call should use the suggested_focus
        assert mock_search.call_count == 1
        first_call_kwargs = mock_search.call_args
        query_used = first_call_kwargs[1].get("query", "") if first_call_kwargs[1] else first_call_kwargs[0][0]
        assert query_used == "aspirin dose milligrams children weight-based"


def test_retry_clears_stepback_query():
    """On retry, stepback_query should not be used (it was for the original query)."""
    from agents.retriever_agent import RetrieverAgent
    from db.repositories import InMemoryVectorRepository

    repo = InMemoryVectorRepository()

    state = {
        "question": "What is the pediatric dosage of aspirin?",
        "optimized_query": "pediatric aspirin dosing guidelines",
        "needs_retry": True,
        "reflection_suggested_focus": "aspirin dose children",
        "stepback_query": "general pharmacology of aspirin",
        "attempts": {"reflection": 1, "executor": 0},
        "vector_repo": repo,
        "documents": [],
        "retrieval_confidence": 0.0,
        "retrieval_candidate_count": 0,
        "route": "vector",
        "planned_route": "vector",
        "post_retrieval_route": "executor",
        "route_decision_reason": "",
    }

    with patch("agents.retriever_agent.embed_query") as mock_embed, \
         patch.object(repo, "hybrid_search", return_value=[]) as mock_search:
        mock_embed.return_value = [0.0] * 768
        RetrieverAgent(state)

        # Should only be called once (no stepback search on retry)
        assert mock_search.call_count == 1


def test_retry_flow_documented():
    """Document the expected ReAct retry flow:

    1. executor generates answer
    2. reflection judges -> failure_category != 'none', suggested_focus set
    3. needs_retry=True -> route_after_reflection returns 'retry'
    4. retriever uses suggested_focus as query (not original optimized_query)
    5. executor generates new answer from new context
    6. reflection judges again (attempt 2)
    7. If still failing, needs_retry=False (max attempts reached) -> finalize
    """
    from core.langgraph_workflow import route_after_reflection

    state_retry = {"needs_retry": True}
    assert route_after_reflection(state_retry) == "retry"

    state_done = {"needs_retry": False}
    assert route_after_reflection(state_done) == "finalize"
