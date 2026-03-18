from agents.executor_agent import _render_prompt_optimized


def test_render_prompt_includes_query_context():
    prompt = _render_prompt_optimized(
        question="what is aspirin?",
        query="aspirin mechanism",
        summary="",
        facts="",
        context="some context",
        query_context="User wants to understand aspirin mechanism for heart attacks",
        session_intent="",
    )
    assert "User wants to understand aspirin mechanism" in prompt


def test_render_prompt_includes_session_intent():
    prompt = _render_prompt_optimized(
        question="dosage?",
        query="aspirin dosage",
        summary="",
        facts="",
        context="some context",
        query_context="",
        session_intent="researching cardiovascular pharmacology",
    )
    assert "cardiovascular pharmacology" in prompt


def test_render_prompt_includes_episodic_memories():
    prompt = _render_prompt_optimized(
        question="when should I take metformin?",
        query="metformin timing",
        summary="",
        facts="",
        context="some context",
        session_intent="",
        query_context="",
        episodic_memories="- User previously asked about metformin dosage.",
    )
    assert "User previously asked about metformin dosage." in prompt


def test_render_prompt_omits_empty_optional_lines():
    prompt = _render_prompt_optimized(
        question="what is hypertension?",
        query="hypertension definition",
        summary="",
        facts="",
        context="some context",
        query_context="",
        session_intent="",
        episodic_memories="",
    )
    assert "Session goal:" not in prompt
    assert "Query context:" not in prompt
