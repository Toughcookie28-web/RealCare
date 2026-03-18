import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_route_after_planner_routes_memory_to_memory():
    from core.langgraph_workflow import route_after_planner

    state = {'route': 'memory'}
    assert route_after_planner(state) == 'memory'


def test_route_after_planner_routes_chitchat_unchanged():
    from core.langgraph_workflow import route_after_planner

    state = {'route': 'chitchat'}
    assert route_after_planner(state) == 'chitchat'


def test_route_after_planner_routes_vector_to_retriever():
    from core.langgraph_workflow import route_after_planner

    state = {'route': 'vector'}
    assert route_after_planner(state) == 'retriever'


def test_executor_memory_prompt_uses_conversation_history():
    from agents.executor_agent import _render_memory_prompt

    prompt = _render_memory_prompt(
        question="what did we discuss?",
        summary="We discussed metformin dosage and side effects.",
        facts="condition=diabetes",
        history_text="user: what is metformin?\nassistant: Metformin is...",
    )
    assert "what did we discuss?" in prompt
    assert "metformin" in prompt.lower()
    assert "conversation" in prompt.lower() or "history" in prompt.lower()
