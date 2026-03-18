import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.live_runtime_contract import build_live_runtime_checks


def test_live_runtime_checks_cover_minimal_docker_contract():
    checks = build_live_runtime_checks()
    assert [check.name for check in checks] == [
        "build_app_image",
        "build_eval_image",
        "db_healthy",
        "alembic_head",
        "app_live",
        "app_ready",
        "eval_preflight",
        "cache_paths_writable",
    ]


def test_eval_checks_use_the_eval_profile_env():
    checks = {check.name: check for check in build_live_runtime_checks()}

    assert checks["build_eval_image"].env == {"COMPOSE_PROFILES": "eval"}
    assert checks["eval_preflight"].env == {"COMPOSE_PROFILES": "eval"}
    assert checks["cache_paths_writable"].env == {"COMPOSE_PROFILES": "eval"}


def test_alembic_check_uses_explicit_container_config_path():
    checks = {check.name: check for check in build_live_runtime_checks()}

    assert checks["alembic_head"].command == (
        "docker",
        "compose",
        "exec",
        "-T",
        "app",
        "python3",
        "-m",
        "alembic",
        "-c",
        "/app/alembic.ini",
        "upgrade",
        "head",
    )
