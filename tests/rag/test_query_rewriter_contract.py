import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_rewrite_result_validates_well_formed_json():
    from agents.query_rewriter_agent import RewriteResult

    data = {
        "optimized_query": "aspirin mechanism of action for myocardial infarction",
        "stepback_query": "pharmacology of antiplatelet agents",
        "reasoning": "expanded medical terms",
    }
    result = RewriteResult.model_validate(data)
    assert result.optimized_query == "aspirin mechanism of action for myocardial infarction"
    assert result.stepback_query == "pharmacology of antiplatelet agents"
    assert result.reasoning == "expanded medical terms"


def test_rewrite_result_provides_defaults_for_missing_fields():
    from agents.query_rewriter_agent import RewriteResult

    data = {"optimized_query": "test query"}
    result = RewriteResult.model_validate(data)
    assert result.optimized_query == "test query"
    assert result.stepback_query == ""
    assert result.reasoning == ""


def test_rewrite_result_requires_optimized_query():
    from agents.query_rewriter_agent import RewriteResult
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError):
        RewriteResult.model_validate({})


def test_parse_rewrite_response_handles_valid_json():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = '{"optimized_query": "test", "stepback_query": "broad test"}'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.optimized_query == "test"
    assert result.stepback_query == "broad test"


def test_parse_rewrite_response_handles_malformed_json():
    from agents.query_rewriter_agent import parse_rewrite_response

    result = parse_rewrite_response("not json at all", fallback_query="original question")
    assert result.optimized_query == "not json at all"
    assert result.stepback_query == ""


def test_parse_rewrite_response_handles_empty_string():
    from agents.query_rewriter_agent import parse_rewrite_response

    result = parse_rewrite_response("", fallback_query="original question")
    assert result.optimized_query == "original question"
    assert result.stepback_query == ""


def test_parse_rewrite_response_handles_partial_json():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = '{"optimized_query": "good query", "stepback_query": 123}'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.optimized_query == "good query"


def test_parse_rewrite_response_extracts_json_from_markdown():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = 'Here is the result:\n```json\n{"optimized_query": "extracted query"}\n```'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.optimized_query == "extracted query"


def test_rewrite_result_validates_route_and_intent_fields():
    from agents.query_rewriter_agent import RewriteResult

    data = {
        "optimized_query": "metformin dosage",
        "route": "vector",
        "session_intent": "medical_consultation",
        "turn_intent": "dosage_lookup",
        "slots": {"drug": "metformin", "aspect": "dosage"},
        "query_context": "user wants metformin dosage information",
    }
    result = RewriteResult.model_validate(data)
    assert result.route == "vector"
    assert result.session_intent == "medical_consultation"
    assert result.turn_intent == "dosage_lookup"
    assert result.slots == {"drug": "metformin", "aspect": "dosage"}
    assert result.query_context == "user wants metformin dosage information"


def test_rewrite_result_defaults_new_fields():
    from agents.query_rewriter_agent import RewriteResult

    result = RewriteResult.model_validate({"optimized_query": "test"})
    assert result.route == "vector"
    assert result.session_intent == ""
    assert result.turn_intent == ""
    assert result.slots == {}
    assert result.query_context == ""


def test_parse_rewrite_response_preserves_route_and_slots():
    from agents.query_rewriter_agent import parse_rewrite_response

    raw = '{"optimized_query": "test", "route": "web", "slots": {"drug": "aspirin"}}'
    result = parse_rewrite_response(raw, fallback_query="original")
    assert result.route == "web"
    assert result.slots == {"drug": "aspirin"}


def test_parse_rewrite_response_defaults_route_on_failure():
    from agents.query_rewriter_agent import parse_rewrite_response

    result = parse_rewrite_response("not json", fallback_query="original")
    assert result.route == "vector"
    assert result.slots == {}
