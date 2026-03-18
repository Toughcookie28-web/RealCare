import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_decomposer_disabled_by_default():
    """When decomposition_enabled is False, decomposer passes through."""
    from agents.decomposer_agent import DecomposerAgent

    state = {
        "question": "Compare aspirin and ibuprofen for pain",
        "optimized_query": "compare aspirin ibuprofen pain relief",
        "sub_questions": [],
        "decomposition_enabled": False,
    }
    result = DecomposerAgent(state)
    assert result.get("sub_questions") == []
    assert result.get("optimized_query") == "compare aspirin ibuprofen pain relief"


def test_decompose_multi_part_question():
    """Decomposer should split a multi-part question into sub-questions."""
    from agents.decomposer_agent import decompose_question

    sub_qs = decompose_question(
        "What are aspirin's side effects and what is the recommended dosage for children?"
    )
    assert isinstance(sub_qs, list)
    for sq in sub_qs:
        assert "question" in sq
        assert "route" in sq


def test_is_multi_part_detects_compound_questions():
    from agents.decomposer_agent import is_multi_part_question

    assert is_multi_part_question("What are side effects and dosage of aspirin?") is True
    assert is_multi_part_question("What is aspirin?") is False
    assert is_multi_part_question("Compare drug A with drug B and explain mechanisms") is True
