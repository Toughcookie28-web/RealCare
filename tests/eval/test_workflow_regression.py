import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class _FakeDocument:
    def __init__(self, metadata: dict):
        self.page_content = metadata.get("content", "")
        self.metadata = metadata


def _load_module():
    core_workflow_service_module = types.ModuleType("core.workflow_service")
    core_workflow_service_module.generate_trace_id = lambda: "trace-1"
    core_workflow_service_module.get_workflow_service = lambda: None

    db_repositories_module = types.ModuleType("db.repositories")
    db_repositories_module.InMemoryChatRepository = object
    db_repositories_module.VectorRepository = object

    db_session_module = types.ModuleType("db.session")
    db_session_module.SessionLocal = lambda: None

    tools_cache_module = types.ModuleType("tools.cache")

    @contextmanager
    def semantic_cache_override(*, enabled=None, version=None, clear=False):
        yield None

    tools_cache_module.semantic_cache_override = semantic_cache_override

    injected = {
        "core.workflow_service": core_workflow_service_module,
        "db.repositories": db_repositories_module,
        "db.session": db_session_module,
        "tools.cache": tools_cache_module,
    }
    originals = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)

    try:
        spec = importlib.util.spec_from_file_location("workflow_eval_module", ROOT / "eval" / "workflow_eval.py")
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["workflow_eval_module"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("workflow_eval_module", None)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_workflow_eval_can_run_against_frozen_fixture_cases():
    module = _load_module()
    cases = _load_jsonl(ROOT / "tests" / "fixtures" / "rag" / "workflow_cases.jsonl")

    def fake_runner(case):
        docs = [
            _FakeDocument(
                {
                    "chunk_id": chunk["chunk_id"],
                    "content_type": chunk["content_type"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "is_table": chunk["is_table"],
                }
            )
            for chunk in case["top_chunks"]
        ]
        return {
            "route": case["expected_route"],
            "source": case["expected_source"],
            "generation": case["fixture_response"],
            "documents": docs,
        }

    report = module.run_cases(cases, workflow_runner=fake_runner)

    assert report["case_count"] == 2
    assert report["semantic_cache_enabled"] is False
    assert report["table_hit_rate"] == 1.0
    assert report["avg_response_keyword_recall"] == 1.0
