import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.generate_tier3_rag import RAW_RAGAS_ROWS_PATH
from scripts.run_tier3_article_generation_smoke import build_command_plan, resolve_python_bin


def test_smoke_runner_uses_article_document_generation_with_bounded_outputs():
    plan = build_command_plan()

    assert len(plan) == 1
    command = plan[0].command
    assert command[0] == resolve_python_bin()
    assert command[1:3] == ("-m", "eval.generate_tier3_rag")
    assert "--article-limit" in command
    assert "--article-corpus" in command
    assert "--raw-output" in command
    assert "--output" in command
    assert "--resume-from" in command
    assert "raw_samples" in command
    assert "/tmp/tier3_article_corpus.smoke.jsonl" in command
    assert str(RAW_RAGAS_ROWS_PATH) in command
    assert "/tmp/tier3_rag.smoke.jsonl" in command


def test_smoke_runner_supports_small_override_values():
    plan = build_command_plan(
        size=2,
        article_limit=6,
        output_path="/tmp/custom-smoke.jsonl",
        raw_output_path="/tmp/custom-smoke.raw.jsonl",
        article_corpus_path="/tmp/custom-article-corpus.jsonl",
        resume_from_raw_samples=False,
    )

    command = plan[0].command
    assert "--size" in command
    assert "2" in command
    assert "--article-limit" in command
    assert "6" in command
    assert "/tmp/custom-smoke.jsonl" in command
    assert "/tmp/custom-smoke.raw.jsonl" in command
    assert "/tmp/custom-article-corpus.jsonl" in command
    assert "--resume-from" not in command


def test_smoke_runner_supports_explicit_python_override():
    assert resolve_python_bin("/tmp/fake-python") == "/tmp/fake-python"

    plan = build_command_plan(python_bin="/tmp/fake-python")
    assert plan[0].command[0] == "/tmp/fake-python"
