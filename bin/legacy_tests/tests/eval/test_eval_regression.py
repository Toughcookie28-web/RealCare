import pytest

pytest.importorskip('fastapi')
from tests.eval.run_eval import run_eval


def test_eval_regression_thresholds():
    result = run_eval()
    assert result['num_samples'] >= 3
    assert result['answer_relevance'] >= 0.05
    assert result['faithfulness_proxy'] >= 0.5
