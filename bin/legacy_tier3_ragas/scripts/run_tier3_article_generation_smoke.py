from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure repository root is importable when executed as a file path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.generate_tier3_rag import DEFAULT_PDF, RAW_RAGAS_ROWS_PATH


@dataclass(frozen=True)
class PlannedCommand:
    label: str
    command: tuple[str, ...]
    description: str


def resolve_python_bin(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def build_command_plan(
    *,
    python_bin: str | None = None,
    pdf_path: str | None = None,
    size: int = 4,
    article_limit: int = 12,
    output_path: str = "/tmp/tier3_rag.smoke.jsonl",
    raw_output_path: str | None = None,
    article_corpus_path: str = "/tmp/tier3_article_corpus.smoke.jsonl",
    resume_from_raw_samples: bool = True,
) -> list[PlannedCommand]:
    resolved_python = resolve_python_bin(python_bin)
    resolved_raw_output_path = raw_output_path
    if resolved_raw_output_path is None:
        if resume_from_raw_samples:
            resolved_raw_output_path = str(RAW_RAGAS_ROWS_PATH)
        else:
            resolved_raw_output_path = "/tmp/tier3_rag.smoke.raw.jsonl"
    command = [
        resolved_python,
        "-m",
        "eval.generate_tier3_rag",
        "--pdf",
        pdf_path or str(DEFAULT_PDF),
        "--size",
        str(size),
        "--article-limit",
        str(article_limit),
        "--output",
        output_path,
        "--raw-output",
        resolved_raw_output_path,
        "--article-corpus",
        article_corpus_path,
    ]
    if resume_from_raw_samples:
        command.extend(["--resume-from", "raw_samples"])
    return [
        PlannedCommand(
            label="tier3_article_generation_smoke",
            command=tuple(command),
            description=(
                "Run the bounded article-document Tier 3 generation smoke path "
                "without touching the curated benchmark."
            ),
        )
    ]


def run_tier3_article_generation_smoke(**kwargs) -> int:
    plan = build_command_plan(**kwargs)
    _ensure_smoke_dependencies(plan[0].command[0])
    for step in plan:
        print(f"\n[{step.label}] {' '.join(step.command)}")
        subprocess.run(step.command, cwd=ROOT, check=True)
    return 0


def _iter_runbook_lines(plan: list[PlannedCommand]) -> list[str]:
    lines = [f"Repo root: {ROOT}", ""]
    for step in plan:
        lines.append(f"- {step.label}: {' '.join(step.command)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    dry_run = "--dry-run" in argv
    if dry_run:
        argv = [arg for arg in argv if arg != "--dry-run"]

    plan = build_command_plan()
    if dry_run:
        print("\n".join(_iter_runbook_lines(plan)))
        return 0
    return run_tier3_article_generation_smoke()


def _ensure_smoke_dependencies(python_bin: str) -> None:
    probe = subprocess.run(
        [python_bin, "-c", "import langchain_core.documents"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "Tier 3 article-generation smoke requires the project Python environment. "
            f"Selected interpreter {python_bin} is missing langchain_core. "
            "Install the project eval/runtime dependencies or run the smoke script from the project env."
        )


if __name__ == "__main__":
    raise SystemExit(main())
