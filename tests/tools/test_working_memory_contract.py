import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.working_memory import (
    build_empty_core_state,
    merge_core_state,
    normalize_recent_turn,
    trim_recent_turns,
)


def test_build_empty_core_state_has_versioned_shape():
    state = build_empty_core_state()

    assert state["version"] == "v1"
    assert state["medical_facts"] == {
        "age": None,
        "allergies": [],
        "conditions": [],
    }
    assert state["session_intent"] == ""
    assert state["session_goal"] == ""
    assert state["medical_context"] == {
        "conditions": [],
        "drugs": [],
        "procedures": [],
        "populations": [],
    }
    assert state["key_findings"] == []
    assert state["open_questions"] == []
    assert "updated_at" in state


def test_normalize_recent_turn_keeps_prompt_relevant_fields():
    turn = normalize_recent_turn(
        {
            "role": "user",
            "content": "I am allergic to penicillin",
            "timestamp": "2026-03-17T10:00:00Z",
            "trace_id": "trace-123",
            "route": "vector",
            "ignored": "value",
        }
    )

    assert turn == {
        "role": "user",
        "content": "I am allergic to penicillin",
        "timestamp": "2026-03-17T10:00:00Z",
        "trace_id": "trace-123",
        "route": "vector",
    }


def test_trim_recent_turns_keeps_most_recent_items():
    turns = [{"content": f"turn-{i}"} for i in range(5)]

    trimmed = trim_recent_turns(turns, max_length=3)

    assert trimmed == [
        {"content": "turn-2"},
        {"content": "turn-3"},
        {"content": "turn-4"},
    ]


def test_merge_core_state_updates_age_allergies_conditions_and_session_intent():
    state = build_empty_core_state()

    merged = merge_core_state(
        state,
        facts=[
            {"key": "age", "value": "45", "confidence": 0.9},
            {"key": "allergy", "value": "penicillin", "confidence": 0.85},
            {"key": "condition", "value": "type 2 diabetes", "confidence": 0.8},
            {"key": "condition", "value": "hypertension", "confidence": 0.8},
        ],
        session_intent="compare diabetes medications",
        summary_text="",
    )

    assert merged["medical_facts"]["age"] == "45"
    assert merged["medical_facts"]["allergies"] == ["penicillin"]
    assert merged["medical_facts"]["conditions"] == ["type 2 diabetes", "hypertension"]
    assert merged["session_intent"] == "compare diabetes medications"
    assert merged["version"] == "v1"
