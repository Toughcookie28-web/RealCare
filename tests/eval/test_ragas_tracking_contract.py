import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class _Metric:
    def __init__(self, name: str):
        self.name = name


def _load_module():
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None

    ragas_module = types.ModuleType("ragas")
    ragas_module.EvaluationDataset = object
    ragas_module.SingleTurnSample = object
    ragas_module.evaluate = lambda *args, **kwargs: None

    ragas_metrics_module = types.ModuleType("ragas.metrics")
    ragas_metrics_module.answer_relevancy = _Metric("answer_relevancy")
    ragas_metrics_module.context_precision = _Metric("context_precision")
    ragas_metrics_module.context_recall = _Metric("context_recall")
    ragas_metrics_module.faithfulness = _Metric("faithfulness")

    agents_retriever_module = types.ModuleType("agents.retriever_agent")
    agents_retriever_module.rerank = lambda query, docs: (docs, [1.0 for _ in docs])

    db_repositories_module = types.ModuleType("db.repositories")
    db_repositories_module.VectorRepository = object

    db_session_module = types.ModuleType("db.session")
    db_session_module.SessionLocal = lambda: None

    tools_embedding_client_module = types.ModuleType("tools.embedding_client")
    tools_embedding_client_module.embed_query = lambda text: [0.0, 0.0]

    injected = {
        "dotenv": dotenv_module,
        "ragas": ragas_module,
        "ragas.metrics": ragas_metrics_module,
        "agents.retriever_agent": agents_retriever_module,
        "db.repositories": db_repositories_module,
        "db.session": db_session_module,
        "tools.embedding_client": tools_embedding_client_module,
    }
    originals = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)

    try:
        spec = importlib.util.spec_from_file_location("ragas_eval_module", ROOT / "eval" / "ragas_eval.py")
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["ragas_eval_module"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("ragas_eval_module", None)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_ragas_main_tracks_saved_snapshot_metadata(tmp_path: Path):
    module = _load_module()
    module.SNAPSHOT_DIR = tmp_path
    module._load_tier3_samples = lambda path, split="test": [{"id": "sample-1"}]
    module.run_ragas_eval = lambda *args, **kwargs: {
        "mode": "retrieval_only",
        "sample_count": 1,
        "context_precision": 0.9,
        "context_precision_scored_count": 1,
        "context_recall": 1.0,
        "context_recall_scored_count": 1,
        "latency_ms_mean": 50.0,
        "slices": {},
        "results": [
            {
                "id": "sample-1",
                "context_precision": 0.9,
                "context_recall": 1.0,
            }
        ],
    }
    tracked_calls = []
    module.persist_saved_run = lambda **kwargs: tracked_calls.append(kwargs) or "run-1"

    original_argv = sys.argv
    try:
        sys.argv = [
            "ragas_eval.py",
            "--split",
            "test",
            "--retrieval-only",
            "--judge-model",
            "gpt-4o-mini",
            "--save",
            "ragas-baseline",
        ]
        module.main()
    finally:
        sys.argv = original_argv

    assert len(tracked_calls) == 1
    call = tracked_calls[0]
    assert call["eval_type"] == "ragas"
    assert call["split"] == "test"
    assert call["judge_model"] == "gpt-4o-mini"
    assert call["snapshot_path"] == str(tmp_path / "ragas-baseline.json")
    assert call["session_factory"] is module.SessionLocal
    assert "snapshot_meta" in call["report"]


def test_ragas_main_skips_tracking_without_save():
    module = _load_module()
    module._load_tier3_samples = lambda path, split="test": [{"id": "sample-1"}]
    module.run_ragas_eval = lambda *args, **kwargs: {
        "mode": "full_pipeline",
        "sample_count": 1,
        "faithfulness": 0.8,
        "faithfulness_scored_count": 1,
        "answer_relevancy": 0.9,
        "answer_relevancy_scored_count": 1,
        "context_precision": 1.0,
        "context_precision_scored_count": 1,
        "context_recall": 1.0,
        "context_recall_scored_count": 1,
        "latency_ms_mean": 50.0,
        "slices": {},
        "results": [
            {
                "id": "sample-1",
                "faithfulness": 0.8,
                "answer_relevancy": 0.9,
                "context_precision": 1.0,
                "context_recall": 1.0,
            }
        ],
    }
    tracked_calls = []
    module.persist_saved_run = lambda **kwargs: tracked_calls.append(kwargs) or "run-1"

    original_argv = sys.argv
    try:
        sys.argv = ["ragas_eval.py", "--split", "test"]
        module.main()
    finally:
        sys.argv = original_argv

    assert tracked_calls == []
