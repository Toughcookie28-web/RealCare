from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ensure repository root is importable when executed as a file path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.live_runtime_contract import LiveRuntimeCheck, build_live_runtime_checks


@dataclass(frozen=True)
class PlannedCommand:
    label: str
    command: tuple[str, ...]
    env: dict[str, str]
    blocking: bool
    description: str


def resolve_compose_bin(preferred: str | None = None) -> str:
    if preferred:
        return preferred

    docker_bin = shutil.which("docker")
    if docker_bin:
        probe = subprocess.run(
            [docker_bin, "compose", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return "docker compose"

    if shutil.which("docker-compose"):
        return "docker-compose"

    raise RuntimeError("No compose command found. Install Docker Compose or pass --compose-bin explicitly.")


def build_command_plan(compose_bin: str = "docker compose") -> list[PlannedCommand]:
    checks = build_live_runtime_checks(compose_bin=compose_bin)
    return [
        PlannedCommand(
            label=check.name,
            command=check.command,
            env=check.env,
            blocking=check.blocking,
            description=check.description,
        )
        for check in checks
    ]


def _run_step(step: PlannedCommand) -> None:
    env = os.environ.copy()
    env.update(step.env)
    print(f"\n[{step.label}] {' '.join(step.command)}")
    subprocess.run(step.command, check=True, env=env, cwd=ROOT)


def run_live_runtime_smoke(compose_bin: str | None = None) -> int:
    resolved = resolve_compose_bin(compose_bin)
    plan = build_command_plan(compose_bin=resolved)
    for step in plan:
        _run_step(step)
    return 0


def _iter_runbook_lines(plan: Iterable[PlannedCommand], compose_bin: str) -> list[str]:
    lines = [f"Repo root: {ROOT}", f"Compose command: {compose_bin}", ""]
    for step in plan:
        lines.append(f"- {step.label}: {' '.join(step.command)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    compose_bin = None
    dry_run = False
    for idx, arg in enumerate(argv):
        if arg == "--compose-bin" and idx + 1 < len(argv):
            compose_bin = argv[idx + 1]
        if arg == "--dry-run":
            dry_run = True

    resolved = resolve_compose_bin(compose_bin)
    plan = build_command_plan(compose_bin=resolved)
    if dry_run:
        print("\n".join(_iter_runbook_lines(plan, resolved)))
        return 0
    return run_live_runtime_smoke(compose_bin=resolved)


if __name__ == "__main__":
    raise SystemExit(main())
