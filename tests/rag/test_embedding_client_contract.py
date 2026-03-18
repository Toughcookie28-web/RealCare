import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_embedding_client(module_name: str, injected_modules: dict[str, types.ModuleType]):
    original_modules = {name: sys.modules.get(name) for name in injected_modules}
    sys.modules.update(injected_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / "embedding_client.py")
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_settings_support_openai_embedding_provider_contract():
    source = (ROOT / "core" / "settings.py").read_text(encoding="utf-8")

    assert "embedding_openai_model" in source
    assert "{'local', 'gemini', 'openai'}" in source


def test_openai_embedding_uses_text_embedding_3_large_with_schema_dimensions():
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        openai_api_key = "sk-test"
        openai_base_url = None
        embedding_provider = "openai"
        embedding_enable_backup = False
        embedding_dim = 768
        embedding_local_model = "unused"
        embedding_local_trust_remote_code = False
        embedding_local_strategy = "plain"
        embedding_cache_dir = "/tmp/unused"
        embedding_gemini_model = "unused"
        google_api_key = None
        embedding_openai_model = "text-embedding-3-large"

        def embedding_provider_order(self):
            return ["openai"]

    core_settings_module.get_settings = lambda: Settings()

    bootstrap_module = types.ModuleType("tools.embedding_bootstrap")
    bootstrap_module.configure_embedding_cache_env = lambda cache_dir: cache_dir

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.25] * 768}]}

    requests_module = types.ModuleType("requests")

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    requests_module.post = _post

    module = _load_embedding_client(
        "embedding_client_openai_contract",
        {
            "core.settings": core_settings_module,
            "tools.embedding_bootstrap": bootstrap_module,
            "requests": requests_module,
        },
    )

    vector = module.embed_document("article text")

    assert len(vector) == 768
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "text-embedding-3-large"
    assert captured["json"]["dimensions"] == 768
    assert captured["json"]["input"] == "article text"


def test_openai_batch_embedding_uses_single_request():
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        openai_api_key = "sk-test"
        openai_base_url = "https://example.test/v1"
        embedding_provider = "openai"
        embedding_enable_backup = False
        embedding_dim = 768
        embedding_local_model = "unused"
        embedding_local_trust_remote_code = False
        embedding_local_strategy = "plain"
        embedding_cache_dir = "/tmp/unused"
        embedding_gemini_model = "unused"
        google_api_key = None
        embedding_openai_model = "text-embedding-3-large"

        def embedding_provider_order(self):
            return ["openai"]

    core_settings_module.get_settings = lambda: Settings()

    bootstrap_module = types.ModuleType("tools.embedding_bootstrap")
    bootstrap_module.configure_embedding_cache_env = lambda cache_dir: cache_dir

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {"embedding": [0.1] * 768},
                    {"embedding": [0.2] * 768},
                ]
            }

    requests_module = types.ModuleType("requests")

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    requests_module.post = _post

    module = _load_embedding_client(
        "embedding_client_openai_batch_contract",
        {
            "core.settings": core_settings_module,
            "tools.embedding_bootstrap": bootstrap_module,
            "requests": requests_module,
        },
    )

    vectors = module.embed_documents_batch(["a", "b"], batch_size=16)

    assert len(vectors) == 2
    assert all(len(vec) == 768 for vec in vectors)
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["json"]["input"] == ["a", "b"]
    assert captured["json"]["dimensions"] == 768


def test_openai_embedding_overrides_allow_generation_only_3072_dimensions():
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        openai_api_key = "sk-test"
        openai_base_url = None
        embedding_provider = "local"
        embedding_enable_backup = False
        embedding_dim = 768
        embedding_local_model = "unused"
        embedding_local_trust_remote_code = False
        embedding_local_strategy = "plain"
        embedding_cache_dir = "/tmp/unused"
        embedding_gemini_model = "unused"
        google_api_key = None
        embedding_openai_model = "text-embedding-3-small"

        def embedding_provider_order(self):
            return ["local"]

    core_settings_module.get_settings = lambda: Settings()

    bootstrap_module = types.ModuleType("tools.embedding_bootstrap")
    bootstrap_module.configure_embedding_cache_env = lambda cache_dir: cache_dir

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.5] * 3072}]}

    requests_module = types.ModuleType("requests")

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    requests_module.post = _post

    module = _load_embedding_client(
        "embedding_client_openai_override_contract",
        {
            "core.settings": core_settings_module,
            "tools.embedding_bootstrap": bootstrap_module,
            "requests": requests_module,
        },
    )

    overrides = module.EmbeddingOverrides(
        provider="openai",
        dim=3072,
        openai_model="text-embedding-3-large",
    )
    vector = module.embed_document("article text", overrides=overrides)

    assert len(vector) == 3072
    assert captured["json"]["model"] == "text-embedding-3-large"
    assert captured["json"]["dimensions"] == 3072


def test_batch_embedding_respects_provider_order_local_first():
    """When provider order is [local], batch embedding should NOT try OpenAI first."""
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        openai_api_key = "sk-test"
        openai_base_url = None
        embedding_provider = "local"
        embedding_enable_backup = False
        embedding_dim = 768
        embedding_local_model = "unused"
        embedding_local_trust_remote_code = False
        embedding_local_strategy = "plain"
        embedding_cache_dir = "/tmp/unused"
        embedding_gemini_model = "unused"
        google_api_key = None
        embedding_openai_model = "text-embedding-3-large"

        def embedding_provider_order(self, primary_override=None):
            return ["local"]

    core_settings_module.get_settings = lambda: Settings()

    bootstrap_module = types.ModuleType("tools.embedding_bootstrap")
    bootstrap_module.configure_embedding_cache_env = lambda cache_dir: cache_dir

    openai_called = {"count": 0}
    requests_module = types.ModuleType("requests")

    def _post(url, headers=None, json=None, timeout=None):
        openai_called["count"] += 1

    requests_module.post = _post

    module = _load_embedding_client(
        "embedding_client_batch_order_contract",
        {
            "core.settings": core_settings_module,
            "tools.embedding_bootstrap": bootstrap_module,
            "requests": requests_module,
        },
    )

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            class FakeArray:
                def __init__(self, v):
                    self._v = v
                def tolist(self):
                    return self._v
            return [FakeArray([0.1] * 768) for _ in texts]

    module.get_local_embedding_model = lambda: FakeModel()

    vectors = module.embed_documents_batch(["a", "b"])
    assert len(vectors) == 2
    assert openai_called["count"] == 0  # Should NOT have tried OpenAI
