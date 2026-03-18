#!/usr/bin/env python3
"""Bring up the full local stack, generate dashboard traffic, and persist final eval snapshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_SNAPSHOT_DIR = ROOT / "eval" / "retrieval_snapshots" / "pipeline"
RAGAS_SNAPSHOT_DIR = ROOT / "eval" / "ragas_snapshots"
DEFAULT_APP_URL = "http://localhost:8000"
DEFAULT_GRAFANA_URL = "http://localhost:3000/d/medigenius-overview/medigenius-overview"
DEFAULT_WORKLOAD = [
    "What are the signs of diabetic ketoacidosis?",
    "What are the contraindications of metformin?",
    "What is the adult acetaminophen dosage?",
    "Show the interaction information for warfarin.",
]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _wait_for_http(url: str, *, label: str, timeout_s: int = 180, interval_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    print(f"{label} ready: {url}")
                    return
                last_error = f"unexpected status {response.status}"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(interval_s)
    raise RuntimeError(f"Timed out waiting for {label} at {url}: {last_error}")


def _post_chat(app_url: str, message: str) -> dict:
    payload = json.dumps(
        {
            "message": message,
            "conversation_id": f"final-pipeline-{uuid.uuid4()}",
        }
    ).encode("utf-8")
    request = Request(
        f"{app_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _replay_workload(app_url: str, workload: list[str]) -> None:
    print("\nReplaying app traffic for Grafana metrics...")
    for index, query in enumerate(workload, start=1):
        result = _post_chat(app_url, query)
        print(
            f"  [{index}/{len(workload)}] route={result.get('route') or 'unknown'} "
            f"trace_id={result.get('trace_id')} success={result.get('success')}"
        )


def _open_urls(app_url: str, grafana_url: str) -> None:
    opened_any = False
    for url in (app_url, grafana_url):
        opened = webbrowser.open_new_tab(url)
        opened_any = opened_any or opened
        print(f"Open request sent: {url}")
    if not opened_any:
        print("Browser did not acknowledge an open request. Open these manually if needed:")
        print(f"  Frontend: {app_url}")
        print(f"  Grafana:  {grafana_url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bring up the local app/eval/observability stack, replay app traffic for Grafana, "
            "and save both the pipeline retrieval snapshot and the full RAGAS snapshot."
        )
    )
    parser.add_argument("--prefix", help="Snapshot prefix. Defaults to final-YYYYmmdd-HHMMSS.")
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--grafana-url", default=DEFAULT_GRAFANA_URL)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the current images instead of rebuilding app/eval.",
    )
    parser.add_argument(
        "--skip-open",
        action="store_true",
        help="Do not open the frontend or Grafana in a browser after the stack is ready.",
    )
    parser.add_argument(
        "--skip-workload",
        action="store_true",
        help="Do not replay HTTP workload against the app for Grafana dashboard traffic.",
    )
    args = parser.parse_args()

    prefix = args.prefix or f"final-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    retrieval_name = f"{prefix}_retrieval_pipeline"
    ragas_name = f"{prefix}_ragas_full"

    services = ["db", "redis", "app", "eval", "prometheus", "grafana"]
    up_cmd = ["docker-compose", "up", "-d"]
    if not args.skip_build:
        up_cmd.append("--build")
    up_cmd.extend(services)
    _run(up_cmd)

    _wait_for_http(f"{args.app_url.rstrip('/')}/health/ready", label="app")
    _wait_for_http(f"{args.grafana_url.split('/d/')[0].rstrip('/')}/login", label="grafana")

    if not args.skip_open:
        _open_urls(args.app_url, args.grafana_url)

    if not args.skip_workload:
        _replay_workload(args.app_url, DEFAULT_WORKLOAD)
        print("Waiting for Prometheus to scrape the replayed traffic...")
        time.sleep(12)

    _run(
        [
            "docker-compose",
            "exec",
            "-T",
            "eval",
            "python3",
            "-m",
            "eval.retrieval_eval",
            "--mode",
            "pipeline",
            "--split",
            args.split,
            "--save",
            retrieval_name,
        ]
    )
    _run(
        [
            "docker-compose",
            "exec",
            "-T",
            "eval",
            "python3",
            "-m",
            "eval.ragas_eval",
            "--split",
            args.split,
            "--judge-model",
            args.judge_model,
            "--save",
            ragas_name,
        ]
    )

    retrieval_path = RETRIEVAL_SNAPSHOT_DIR / f"{retrieval_name}.json"
    ragas_path = RAGAS_SNAPSHOT_DIR / f"{ragas_name}.json"
    print("\nArtifacts")
    print(f"  Retrieval snapshot: {retrieval_path}")
    print(f"  RAGAS snapshot:     {ragas_path}")
    print(f"  Frontend:           {args.app_url}")
    print(f"  Grafana dashboard:  {args.grafana_url}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)
