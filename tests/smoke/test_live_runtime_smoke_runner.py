import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_runtime_smoke import build_command_plan, resolve_compose_bin


def test_command_plan_runs_blocking_checks_in_contract_order():
    plan = build_command_plan(compose_bin="docker compose")
    assert plan[0].label == "build_app_image"
    assert plan[-1].label == "cache_paths_writable"


def test_command_plan_supports_legacy_compose_binary():
    plan = build_command_plan(compose_bin="docker-compose")
    assert plan[0].command[0] == "docker-compose"
    assert "docker-compose" == resolve_compose_bin("docker-compose")


def test_readme_documents_the_live_runtime_smoke_entrypoint():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python3 scripts/run_live_runtime_smoke.py --dry-run" in readme
    assert "python3 scripts/run_live_runtime_smoke.py --compose-bin docker-compose" in readme


def test_dockerfile_copies_alembic_assets_into_the_image():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY --chown=appuser:appuser alembic /app/alembic" in dockerfile
    assert "COPY --chown=appuser:appuser alembic.ini /app/alembic.ini" in dockerfile
