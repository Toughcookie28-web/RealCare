import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_planner_reads_route_from_state():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='web')
    result = PlannerAgent(state)
    assert result['route'] == 'web'
    assert result['planned_route'] == 'web'


def test_planner_accepts_memory_route():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='memory')
    result = PlannerAgent(state)
    assert result['route'] == 'memory'


def test_planner_falls_back_on_invalid_route():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='invalid_garbage')
    result = PlannerAgent(state)
    assert result['route'] in {'vector', 'chitchat', 'web', 'literature', 'memory'}


def test_planner_falls_back_on_empty_route():
    from agents.planner_agent import PlannerAgent

    state = _make_state(route='')
    result = PlannerAgent(state)
    assert result['route'] in {'vector', 'chitchat', 'web', 'literature', 'memory'}


def _make_state(**overrides):
    base = {
        'trace_id': 't1',
        'session_id': 's1',
        'question': 'test question',
        'raw_query': 'test question',
        'optimized_query': 'test question',
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
        'chat_repo': None,
        'vector_repo': None,
        'session_intent': '',
        'turn_intent': '',
        'slots': {},
        'query_context': '',
    }
    base.update(overrides)
    return base
