import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_cache_module():
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        semantic_cache_enabled = True
        semantic_cache_threshold = 0.95
        semantic_cache_size = 500
        semantic_cache_version = "default-version"

        def use_semantic_cache(self):
            return bool(self.semantic_cache_enabled)

    core_settings_module.get_settings = lambda: Settings()

    observability_metrics_module = types.ModuleType("observability.metrics")

    class Counter:
        def __init__(self):
            self.value = 0

        def inc(self):
            self.value += 1

    observability_metrics_module.CACHE_HITS = Counter()
    observability_metrics_module.CACHE_MISSES = Counter()

    tools_embedding_client_module = types.ModuleType("tools.embedding_client")
    tools_embedding_client_module.embed_query = lambda text: [1.0, 0.0]

    injected_modules = {
        "core.settings": core_settings_module,
        "observability.metrics": observability_metrics_module,
        "tools.embedding_client": tools_embedding_client_module,
    }
    original_modules = {name: sys.modules.get(name) for name in injected_modules}
    sys.modules.update(injected_modules)

    try:
        spec = importlib.util.spec_from_file_location("semantic_cache_module", ROOT / "tools" / "cache.py")
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["semantic_cache_module"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("semantic_cache_module", None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_semantic_cache_can_be_disabled():
    module = _load_cache_module()

    cache = module.SemanticCache(enabled=False, embedder=lambda query: [1.0, 0.0])
    cache.set("what is dka", "cached answer")

    assert cache.get("what is dka") is None


def test_semantic_cache_respects_version_boundaries():
    module = _load_cache_module()

    cache = module.SemanticCache(
        enabled=True,
        version="executor-v1",
        embedder=lambda query: [1.0, 0.0],
    )
    cache.set("what is dka", "cached answer")

    assert cache.get("what is dka") == "cached answer"
    assert cache.get("what is dka", namespace="executor-v2") is None


def test_eval_workflow_disables_semantic_cache_by_default():
    source = (ROOT / "eval" / "workflow_eval.py").read_text(encoding="utf-8")

    assert "def run_cases(cases: list[dict], *, use_semantic_cache: bool = False)" in source
    assert "semantic_cache_override(enabled=use_semantic_cache" in source
    assert "--semantic-cache" in source


def test_semantic_cache_falls_back_to_inmemory_without_redis():
    """SemanticCache works with in-memory backend when Redis is unavailable."""
    module = _load_cache_module()

    cache = module.SemanticCache(
        enabled=True,
        version="test-v1",
        embedder=lambda query: [1.0, 0.0],
    )
    cache.set("what is dka", "cached answer")
    assert cache.get("what is dka") == "cached answer"
    assert cache.get("what is dka", namespace="other") is None


def test_semantic_cache_escapes_namespace_for_redis_tag_filter():
    module = _load_cache_module()

    cache = module.SemanticCache(
        enabled=True,
        version="executor-v1",
        embedder=lambda query: [1.0, 0.0],
    )

    assert cache._redis_namespace_filter("executor-v1:medical") == "(@namespace:{executor-v1\\:medical})"
