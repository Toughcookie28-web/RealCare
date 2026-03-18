import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_final_pipeline.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("final_pipeline_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_wait_for_http_retries_after_connection_reset(monkeypatch):
    module = _load_runner_module()
    timeline = iter([0.0, 0.0, 0.1])
    calls = iter([ConnectionResetError(104, "reset by peer"), _DummyResponse()])

    def fake_urlopen(url, timeout=5):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "time", lambda: next(timeline))
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    module._wait_for_http("http://localhost:8000/health/ready", label="app", timeout_s=5)
