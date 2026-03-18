from __future__ import annotations

from dataclasses import dataclass, field
from shlex import split as shlex_split


@dataclass(frozen=True)
class LiveRuntimeCheck:
    name: str
    description: str
    command: tuple[str, ...]
    blocking: bool = True
    env: dict[str, str] = field(default_factory=dict)
    artifact_hint: str | None = None


def build_live_runtime_checks(compose_bin: str = "docker compose") -> list[LiveRuntimeCheck]:
    compose = tuple(shlex_split(compose_bin))
    eval_env = {"COMPOSE_PROFILES": "eval"}

    return [
        LiveRuntimeCheck(
            name="build_app_image",
            description="Build the app image from the canonical runtime install contract.",
            command=compose + ("build", "app"),
        ),
        LiveRuntimeCheck(
            name="build_eval_image",
            description="Build the eval image with eval+ingest extras enabled.",
            command=compose + ("build", "eval"),
            env=eval_env,
        ),
        LiveRuntimeCheck(
            name="db_healthy",
            description="Start db and app so later checks run against the live compose stack.",
            command=compose + ("up", "-d", "db", "app"),
        ),
        LiveRuntimeCheck(
            name="alembic_head",
            description="Upgrade the live database schema to Alembic head before runtime checks.",
            command=compose
            + ("exec", "-T", "app", "python3", "-m", "alembic", "-c", "/app/alembic.ini", "upgrade", "head"),
        ),
        LiveRuntimeCheck(
            name="app_live",
            description="Verify the live health endpoint responds from the running containerized app.",
            command=("curl", "-fsS", "http://localhost:8000/health/live"),
        ),
        LiveRuntimeCheck(
            name="app_ready",
            description="Verify the ready endpoint reports a usable DB-backed runtime.",
            command=("curl", "-fsS", "http://localhost:8000/health/ready"),
        ),
        LiveRuntimeCheck(
            name="eval_preflight",
            description="Run the RAG runtime preflight inside the eval container.",
            command=compose + ("run", "--rm", "eval", "python3", "scripts/preflight_rag.py"),
            env=eval_env,
        ),
        LiveRuntimeCheck(
            name="cache_paths_writable",
            description="Confirm the eval container can create and validate writable cache paths.",
            command=compose
            + (
                "run",
                "--rm",
                "eval",
                "python3",
                "-c",
                "from core.runtime_contracts import validate_rag_runtime_contract; "
                "raise SystemExit(0 if not validate_rag_runtime_contract() else 1)",
            ),
            env=eval_env,
        ),
    ]
