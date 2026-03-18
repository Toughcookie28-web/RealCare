import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_parse_reflection_response_extracts_structured_fields():
    from agents.reflection_agent import _parse_reflection_response

    raw = {
        "is_relevant": True,
        "has_hallucinations": False,
        "failure_category": "none",
        "suggested_focus": "",
        "confidence": 0.85,
        "feedback": "Good answer.",
    }
    result = _parse_reflection_response(raw)
    assert result.failure_category == "none"
    assert result.confidence == 0.85
    assert result.suggested_focus == ""
    assert result.is_relevant is True


def test_parse_reflection_response_defaults_on_missing_fields():
    from agents.reflection_agent import _parse_reflection_response

    raw = {"is_relevant": False, "has_hallucinations": True}
    result = _parse_reflection_response(raw)
    assert result.failure_category == "hallucination"
    assert result.confidence == 0.0
    assert result.suggested_focus == ""


def test_parse_reflection_response_coerces_types():
    from agents.reflection_agent import _parse_reflection_response

    raw = {
        "is_relevant": "true",
        "has_hallucinations": "false",
        "failure_category": "incomplete",
        "suggested_focus": "dosage for children",
        "confidence": "0.7",
        "feedback": "Missing pediatric info.",
    }
    result = _parse_reflection_response(raw)
    assert result.is_relevant is True
    assert result.confidence == 0.7
    assert result.suggested_focus == "dosage for children"
    assert result.failure_category == "incomplete"


from unittest.mock import patch


def test_reflection_agent_sets_structured_state():
    from agents.reflection_agent import ReflectionAgent

    mock_result = {
        "is_relevant": True,
        "has_hallucinations": False,
        "failure_category": "none",
        "suggested_focus": "",
        "confidence": 0.9,
        "feedback": "Accurate and complete.",
    }

    state = {
        "question": "What is the dosage of aspirin?",
        "generation": "Aspirin is typically dosed at 81-325mg daily.",
        "attempts": {"reflection": 0, "executor": 0},
        "needs_retry": False,
        "reflection_feedback": "",
    }

    with patch("agents.reflection_agent.invoke_json", return_value=mock_result):
        result = ReflectionAgent(state)

    assert result["needs_retry"] is False
    assert result["reflection_feedback"] == "Accurate and complete."
    assert result.get("reflection_suggested_focus") == ""
    assert result.get("reflection_confidence") == 0.9
    assert result.get("reflection_failure_category") == "none"


def test_reflection_agent_triggers_retry_with_focus():
    from agents.reflection_agent import ReflectionAgent

    mock_result = {
        "is_relevant": False,
        "has_hallucinations": False,
        "failure_category": "incomplete",
        "suggested_focus": "pediatric aspirin dosing guidelines",
        "confidence": 0.3,
        "feedback": "Answer lacks pediatric dosing information.",
    }

    state = {
        "question": "What is the pediatric dosage of aspirin?",
        "generation": "Aspirin is used for pain relief.",
        "attempts": {"reflection": 0, "executor": 0},
        "needs_retry": False,
        "reflection_feedback": "",
    }

    with patch("agents.reflection_agent.invoke_json", return_value=mock_result):
        result = ReflectionAgent(state)

    assert result["needs_retry"] is True
    assert result["reflection_suggested_focus"] == "pediatric aspirin dosing guidelines"
    assert result["reflection_failure_category"] == "incomplete"
