import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.working_memory import build_empty_core_state, merge_core_state


def test_merge_core_state_promotes_summary_fields():
    summary_text = "\n".join(
        [
            "Session goal: compare diabetes medications for side effects",
            "Medical context: conditions: type 2 diabetes, hypertension; drugs: metformin, insulin; procedures: A1C testing; populations: older adults",
            "Key findings: Metformin commonly causes gastrointestinal upset; Insulin can cause hypoglycemia",
            "Open questions: Which option is safer in older adults?; How should dose changes be monitored?",
            "Trajectory: Started with diabetes overview, narrowed to medication comparison",
        ]
    )

    merged = merge_core_state(
        build_empty_core_state(),
        facts=[],
        session_intent="drug_comparison",
        summary_text=summary_text,
    )

    assert merged["session_goal"] == "compare diabetes medications for side effects"
    assert merged["medical_context"] == {
        "conditions": ["type 2 diabetes", "hypertension"],
        "drugs": ["metformin", "insulin"],
        "procedures": ["A1C testing"],
        "populations": ["older adults"],
    }
    assert merged["key_findings"] == [
        "Metformin commonly causes gastrointestinal upset",
        "Insulin can cause hypoglycemia",
    ]
    assert merged["open_questions"] == [
        "Which option is safer in older adults?",
        "How should dose changes be monitored?",
    ]
    assert merged["session_intent"] == "drug_comparison"
