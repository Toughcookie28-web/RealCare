from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ci_uses_frozen_regression_suite_not_live_generation():
    workflow = (ROOT / ".github" / "workflows" / "eval.yml").read_text(encoding="utf-8")

    assert "python3 -m pytest tests/eval tests/rag tests/smoke -v" in workflow
    assert "python3 -m eval.generate_tier3_rag" not in workflow
    assert "python3 scripts/preflight_rag.py" not in workflow


def test_readme_marks_tier3_synthetic_generation_as_retired():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "retired" in readme.lower()
    assert not (ROOT / "eval" / "generate_tier3_rag.py").exists()
