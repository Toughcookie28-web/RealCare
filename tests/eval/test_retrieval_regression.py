import importlib.util
import json
import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class _FakeDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class _FakeInMemoryVectorRepository:
    def __init__(self):
        self._chunks: dict[str, dict] = {}

    def upsert_chunk(self, chunk_id, doc_id, content, embedding, page=None, section=None, metadata=None):
        self._chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "embedding": list(embedding),
            "page": page,
            "section": section,
            "metadata": metadata or {},
        }

    def count_chunks(self):
        return len(self._chunks)

    def similarity_search(self, query_embedding, k=8):
        rows = sorted(
            self._chunks.values(),
            key=lambda row: _cosine_distance(row["embedding"], query_embedding),
        )[:k]
        return [_row_to_doc(row) for row in rows]

    def keyword_search(self, query, k=8):
        tokens = [t for t in query.lower().split() if len(t) > 2]
        scored = []
        for row in self._chunks.values():
            text = row["content"].lower()
            score = sum(token in text for token in tokens)
            if score:
                scored.append((row, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [_row_to_doc(row) for row, _ in scored[:k]]

    def hybrid_search(self, query, query_embedding, k=8):
        dense = self.similarity_search(query_embedding, k=k * 2)
        sparse = self.keyword_search(query, k=k * 2)
        scored: dict[str, tuple[_FakeDocument, float]] = {}
        for rank, doc in enumerate(dense):
            key = doc.metadata["chunk_id"]
            scored[key] = (doc, scored.get(key, (doc, 0.0))[1] + (1.0 / (rank + 1)))
        for rank, doc in enumerate(sparse):
            key = doc.metadata["chunk_id"]
            scored[key] = (doc, scored.get(key, (doc, 0.0))[1] + (0.75 / (rank + 1)))
        return [doc for doc, _ in sorted(scored.values(), key=lambda item: item[1], reverse=True)[:k]]


def _row_to_doc(row: dict) -> _FakeDocument:
    metadata = {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "page": row.get("page"),
        "section": row.get("section"),
        **(row.get("metadata") or {}),
    }
    return _FakeDocument(page_content=row["content"], metadata=metadata)


def _cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - (numerator / (left_norm * right_norm))


def _load_module():
    agents_retriever_module = types.ModuleType("agents.retriever_agent")
    agents_retriever_module.rerank = lambda query, docs: (docs, [1.0 for _ in docs])
    agents_retriever_module._merge_and_deduplicate = lambda primary, stepback: primary + [d for d in stepback if d not in primary]

    db_repositories_module = types.ModuleType("db.repositories")
    db_repositories_module.VectorRepository = _FakeInMemoryVectorRepository
    db_repositories_module.InMemoryVectorRepository = _FakeInMemoryVectorRepository

    db_session_module = types.ModuleType("db.session")
    db_session_module.SessionLocal = lambda: None

    tools_embedding_client_module = types.ModuleType("tools.embedding_client")
    tools_embedding_client_module.embed_query = lambda text: [0.0, 0.0]

    injected = {
        "agents.retriever_agent": agents_retriever_module,
        "db.repositories": db_repositories_module,
        "db.session": db_session_module,
        "tools.embedding_client": tools_embedding_client_module,
    }
    originals = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)

    try:
        spec = importlib.util.spec_from_file_location("retrieval_eval_module", ROOT / "eval" / "retrieval_eval.py")
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["retrieval_eval_module"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("retrieval_eval_module", None)
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_retrieval_eval_can_run_against_frozen_fixture_corpus():
    module = _load_module()
    repo = module.load_fixture_repo(ROOT / "tests" / "fixtures" / "rag" / "mini_chunks.jsonl")
    samples = _load_jsonl(ROOT / "tests" / "fixtures" / "rag" / "retrieval_cases.jsonl")
    vectors = {
        "adult acetaminophen dose": [1.0, 0.0],
        "signs of diabetic ketoacidosis": [0.0, 1.0],
    }

    report = module.run_retrieval_eval(
        samples=samples,
        mode="hybrid",
        ks=(1, 3),
        fetch_k=3,
        repo=repo,
        query_embedder=lambda query: vectors[query],
    )

    assert report["sample_count"] == 2
    assert report["hit_rate@1"] == 1.0


def test_tier3_loader_filters_dev_and_test_splits_but_keeps_unsplit_rows(tmp_path: Path):
    module = _load_module()
    dataset = tmp_path / "tier3.jsonl"
    rows = [
        {"id": "DEV", "split": "dev", "question": "q1", "ground_truth_contexts": ["a"], "ground_truth_chunk_ids": ["c1"]},
        {"id": "TEST", "split": "test", "question": "q2", "ground_truth_contexts": ["b"], "ground_truth_chunk_ids": ["c2"]},
        {"id": "LEGACY", "question": "q3", "ground_truth_contexts": ["c"], "ground_truth_chunk_ids": ["c3"]},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert [row["id"] for row in module._load_tier3_samples(dataset, split="dev")] == ["DEV", "LEGACY"]
    assert [row["id"] for row in module._load_tier3_samples(dataset, split="test")] == ["TEST", "LEGACY"]
    assert [row["id"] for row in module._load_tier3_samples(dataset, split="all")] == ["DEV", "TEST", "LEGACY"]


def test_pipeline_snapshots_save_to_dedicated_subfolder_and_load_with_fallback(tmp_path: Path):
    module = _load_module()
    module.SNAPSHOT_DIR = tmp_path / "retrieval_snapshots"
    module.PIPELINE_SNAPSHOT_DIR = module.SNAPSHOT_DIR / "pipeline"

    path = module.save_snapshot("pipeline-run", {"mode": "pipeline", "latency_ms_mean": 123.4})
    assert path == module.PIPELINE_SNAPSHOT_DIR / "pipeline-run.json"
    assert path.exists()

    loaded = module.load_snapshot("pipeline-run", mode="pipeline")
    assert loaded is not None
    assert loaded["mode"] == "pipeline"

    legacy_path = module.SNAPSHOT_DIR / "legacy-flat.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps({"mode": "pipeline", "latency_ms_mean": 55.0}), encoding="utf-8")

    legacy_loaded = module.load_snapshot("legacy-flat", mode="pipeline")
    assert legacy_loaded is not None
    assert legacy_loaded["latency_ms_mean"] == 55.0
