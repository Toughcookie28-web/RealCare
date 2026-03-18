from agents.executor_agent import _intent_instruction, _render_prompt_optimized


def test_intent_instruction_definition():
    instr = _intent_instruction('definition', {})
    assert 'definition' in instr.lower() or 'summary' in instr.lower()


def test_intent_instruction_mechanism():
    instr = _intent_instruction('mechanism_of_action', {})
    assert 'step' in instr.lower() or 'mechanism' in instr.lower()


def test_intent_instruction_drug_comparison():
    instr = _intent_instruction('drug_comparison', {})
    assert 'compare' in instr.lower() or 'difference' in instr.lower()


def test_intent_instruction_dosage():
    instr = _intent_instruction('dosage_lookup', {})
    assert 'dosage' in instr.lower()


def test_intent_instruction_side_effects():
    instr = _intent_instruction('side_effects', {})
    assert 'side effect' in instr.lower() or 'frequency' in instr.lower()


def test_intent_instruction_unknown_returns_empty():
    instr = _intent_instruction('unknown_intent', {})
    assert instr == ''


def test_intent_instruction_empty_returns_empty():
    instr = _intent_instruction('', {})
    assert instr == ''


def test_render_prompt_includes_query_context():
    prompt = _render_prompt_optimized(
        question='what is aspirin?',
        query='aspirin mechanism',
        summary='',
        facts='',
        context='some context',
        query_context='User wants to understand aspirin mechanism for heart attacks',
        turn_intent='mechanism_of_action',
        session_intent='',
    )
    assert 'User wants to understand aspirin mechanism' in prompt


def test_render_prompt_includes_session_intent():
    prompt = _render_prompt_optimized(
        question='dosage?',
        query='aspirin dosage',
        summary='',
        facts='',
        context='some context',
        query_context='',
        turn_intent='dosage_lookup',
        session_intent='researching cardiovascular pharmacology',
    )
    assert 'cardiovascular pharmacology' in prompt


def test_render_prompt_includes_intent_instruction():
    prompt = _render_prompt_optimized(
        question='what is hypertension?',
        query='hypertension definition',
        summary='',
        facts='',
        context='some context',
        query_context='',
        turn_intent='definition',
        session_intent='',
    )
    assert 'definition' in prompt.lower() or 'summary' in prompt.lower()
