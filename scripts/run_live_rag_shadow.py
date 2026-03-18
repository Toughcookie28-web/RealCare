from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure repository root is importable when executed as a file path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_runtime_smoke import resolve_compose_bin


@dataclass(frozen=True)
class ShadowStep:
    label: str
    command: tuple[str, ...]
    env: dict[str, str]
    blocking: bool = False
    artifact_hint: str | None = None


def build_shadow_steps(compose_bin: str = "docker compose") -> list[ShadowStep]:
    compose_parts = tuple(compose_bin.split())
    eval_env = {"COMPOSE_PROFILES": "eval"}
    return [
        ShadowStep(
            label="reindex_pdf",
            command=compose_parts + ("run", "--rm", "eval", "python3", "scripts/reindex_pdf.py"),
            env=eval_env,
            artifact_hint="terminal stage counts from reindex",
        ),
        ShadowStep(
            label="retrieval_eval_live",
            command=compose_parts
            + (
                "run",
                "--rm",
                "eval",
                "python3",
                "-m",
                "eval.retrieval_eval",
                "--save",
                "live-shadow",
            ),
            env=eval_env,
            artifact_hint="eval/retrieval_snapshots/live-shadow.json",
        ),
        ShadowStep(
            label="chunking_eval_live",
            command=compose_parts
            + (
                "run",
                "--rm",
                "eval",
                "python3",
                "-m",
                "eval.chunking_eval",
                "--save",
                "live-shadow",
            ),
            env=eval_env,
            artifact_hint="eval/chunking_snapshots/live-shadow.json",
        ),
    ]


def _run_step(step: ShadowStep) -> None:
    env = os.environ.copy()
    env.update(step.env)
    print(f"\n[{step.label}] {' '.join(step.command)}")
    subprocess.run(step.command, check=True, env=env, cwd=ROOT)


def run_live_rag_shadow(compose_bin: str | None = None) -> int:
    resolved = resolve_compose_bin(compose_bin)
    for step in build_shadow_steps(compose_bin=resolved):
        _run_step(step)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    compose_bin = None
    for idx, arg in enumerate(argv):
        if arg == "--compose-bin" and idx + 1 < len(argv):
            compose_bin = argv[idx + 1]
    return run_live_rag_shadow(compose_bin=compose_bin)


if __name__ == "__main__":
    raise SystemExit(main())
