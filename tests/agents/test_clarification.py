def test_rewrite_result_has_clarification_fields():
    from agents.query_rewriter_agent import RewriteResult
    r = RewriteResult(optimized_query="test")
    assert r.needs_clarification is False
    assert r.clarification_question == ""
    assert r.intent_confidence == 1.0


def test_rewrite_result_with_clarification():
    from agents.query_rewriter_agent import RewriteResult
    r = RewriteResult(
        optimized_query="aspirin dosage",
        needs_clarification=True,
        clarification_question="Are you asking about adult or pediatric dosage?",
        intent_confidence=0.3,
    )
    assert r.needs_clarification is True
    assert "dosage" in r.clarification_question


def test_parse_rewrite_response_handles_clarification():
    from agents.query_rewriter_agent import parse_rewrite_response
    raw = """{
        "optimized_query": "aspirin dosage",
        "route": "vector",
        "needs_clarification": true,
        "clarification_question": "Are you asking about aspirin for pain or heart disease?",
        "intent_confidence": 0.3,
        "stepback_query": "", "session_intent": "", "turn_intent": "", "slots": {}, "query_context": ""
    }"""
    result = parse_rewrite_response(raw, fallback_query="aspirin dosage")
    assert result.needs_clarification is True
    assert result.intent_confidence == 0.3


def test_clarify_route_generates_question_without_retrieval(monkeypatch):
    from core.state_v2 import initialize_state

    monkeypatch.setattr(
        "agents.query_rewriter_agent.invoke_json",
        lambda *a, **kw: {
            "optimized_query": "aspirin",
            "route": "vector",
            "needs_clarification": True,
            "clarification_question": "Are you asking about aspirin for pain or heart disease?",
            "intent_confidence": 0.3,
            "stepback_query": "", "session_intent": "dosage", "turn_intent": "dosage_lookup",
            "slots": {}, "query_context": "",
        },
    )

    from agents.query_rewriter_agent import QueryRewriterAgent
    state = initialize_state("s1", "t1")
    state["question"] = "what should I take"
    result = QueryRewriterAgent(state)

    assert result["needs_clarification"] is True
    assert "pain or heart" in result["clarification_question"]
