import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_state_v2_has_new_pipeline_fields():
    from core.state_v2 import initialize_state

    state = initialize_state(session_id='s1', trace_id='t1')
    assert state['session_intent'] == ''
    assert state['turn_intent'] == ''
    assert state['slots'] == {}
    assert state['query_context'] == ''


def test_reset_query_state_clears_new_fields():
    from core.state_v2 import initialize_state, reset_query_state

    state = initialize_state(session_id='s1', trace_id='t1')
    state['session_intent'] = 'medical_consultation'
    state['turn_intent'] = 'dosage_lookup'
    state['slots'] = {'drug': 'metformin'}
    state['query_context'] = 'user wants dosage info'

    reset_query_state(state, 'new question')
    assert state['session_intent'] == ''
    assert state['turn_intent'] == ''
    assert state['slots'] == {}
    assert state['query_context'] == ''


def test_state_has_slot_coverage():
    from core.state_v2 import initialize_state

    state = initialize_state(session_id='s1', trace_id='t1')
    assert 'slot_coverage' in state
    assert state['slot_coverage'] == 0.0


def test_reset_clears_slot_coverage():
    from core.state_v2 import initialize_state, reset_query_state

    state = initialize_state(session_id='s1', trace_id='t1')
    state['slot_coverage'] = 0.8
    reset_query_state(state, 'new question')
    assert state['slot_coverage'] == 0.0
