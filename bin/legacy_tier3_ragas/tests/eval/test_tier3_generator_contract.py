import asyncio
import importlib.util
import json
import sys
import tomllib
import types
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_source_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


def _load_generator_module():
    return _load_source_module(
        "tier3_generator_contract",
        ROOT / "eval" / "generate_tier3_rag.py",
    )


def _load_validate_module():
    return _load_source_module(
        "tier3_validate_contract",
        ROOT / "eval" / "validate.py",
    )


def _install_fake_langchain_document():
    class _FakeDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    langchain_core = types.ModuleType("langchain_core")
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = _FakeDocument
    original_core = sys.modules.get("langchain_core")
    original_documents = sys.modules.get("langchain_core.documents")
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.documents"] = documents_module
    return original_core, original_documents


def _restore_fake_langchain_document(original_core, original_documents):
    if original_core is None:
        sys.modules.pop("langchain_core", None)
    else:
        sys.modules["langchain_core"] = original_core

    if original_documents is None:
        sys.modules.pop("langchain_core.documents", None)
    else:
        sys.modules["langchain_core.documents"] = original_documents


def _install_fake_embedding_client():
    embedding_client_module = types.ModuleType("tools.embedding_client")
    embedding_client_module.EmbeddingOverrides = type("EmbeddingOverrides", (), {})
    embedding_client_module.embed_document = lambda text, *, overrides=None: [float(len(text))]
    embedding_client_module.embed_query = lambda text, *, overrides=None: [float(len(text) + 1)]
    original_module = sys.modules.get("tools.embedding_client")
    sys.modules["tools.embedding_client"] = embedding_client_module
    return original_module


def _restore_fake_embedding_client(original_module):
    if original_module is None:
        sys.modules.pop("tools.embedding_client", None)
    else:
        sys.modules["tools.embedding_client"] = original_module


def _install_fake_ragas_profile_modules():
    originals = {name: sys.modules.get(name) for name in (
        "ragas",
        "ragas.testset",
        "ragas.testset.graph",
        "ragas.testset.synthesizers",
        "ragas.testset.transforms",
        "ragas.testset.transforms.extractors",
        "ragas.testset.transforms.extractors.llm_based",
    )}

    ragas_module = types.ModuleType("ragas")
    testset_module = types.ModuleType("ragas.testset")
    graph_module = types.ModuleType("ragas.testset.graph")
    synthesizers_module = types.ModuleType("ragas.testset.synthesizers")
    transforms_module = types.ModuleType("ragas.testset.transforms")
    extractors_module = types.ModuleType("ragas.testset.transforms.extractors")
    llm_based_module = types.ModuleType("ragas.testset.transforms.extractors.llm_based")

    class NodeType(Enum):
        DOCUMENT = "document"

    class _Named:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class SingleHopSpecificQuerySynthesizer(_Named):
        pass

    class MultiHopSpecificQuerySynthesizer(_Named):
        pass

    class SummaryExtractor(_Named):
        pass

    class EmbeddingExtractor(_Named):
        pass

    class ThemesExtractor(_Named):
        pass

    class NERExtractor(_Named):
        pass

    class CosineSimilarityBuilder(_Named):
        pass

    class OverlapScoreBuilder(_Named):
        pass

    class CustomNodeFilter(_Named):
        pass

    class Parallel:
        def __init__(self, *transforms):
            self.transforms = transforms

    graph_module.NodeType = NodeType
    synthesizers_module.SingleHopSpecificQuerySynthesizer = SingleHopSpecificQuerySynthesizer
    synthesizers_module.MultiHopSpecificQuerySynthesizer = MultiHopSpecificQuerySynthesizer
    transforms_module.SummaryExtractor = SummaryExtractor
    transforms_module.EmbeddingExtractor = EmbeddingExtractor
    transforms_module.CosineSimilarityBuilder = CosineSimilarityBuilder
    transforms_module.OverlapScoreBuilder = OverlapScoreBuilder
    transforms_module.CustomNodeFilter = CustomNodeFilter
    transforms_module.Parallel = Parallel
    llm_based_module.ThemesExtractor = ThemesExtractor
    llm_based_module.NERExtractor = NERExtractor

    sys.modules["ragas"] = ragas_module
    sys.modules["ragas.testset"] = testset_module
    sys.modules["ragas.testset.graph"] = graph_module
    sys.modules["ragas.testset.synthesizers"] = synthesizers_module
    sys.modules["ragas.testset.transforms"] = transforms_module
    sys.modules["ragas.testset.transforms.extractors"] = extractors_module
    sys.modules["ragas.testset.transforms.extractors.llm_based"] = llm_based_module
    return originals


def _restore_fake_ragas_profile_modules(originals):
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_eval_optional_dependencies_include_rapidfuzz_for_ragas_default_transforms():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    eval_deps = pyproject["project"]["optional-dependencies"]["eval"]
    assert "rapidfuzz" in eval_deps


def test_generator_source_removes_retired_seed_gating_cli():
    source = (ROOT / "eval" / "generate_tier3_rag.py").read_text(encoding="utf-8")

    assert "--manual-seed-policy" not in source
    assert "--checkpoint-upstream-stages" not in source
    assert "tier3_rag.benchmark.jsonl" not in source


def test_generator_uses_neutral_article_tooling_instead_of_manual_seed_helper_name():
    source = (ROOT / "eval" / "generate_tier3_rag.py").read_text(encoding="utf-8")

    assert "from eval.tier3_manual_seed_pdf_authoring import" not in source
    assert "tier3_article_documents.py" in source
    assert "_load_article_documents_module" in source


def test_generator_default_output_no_longer_targets_curated_tier3_benchmark():
    module = _load_generator_module()

    assert module.TIER3_PATH.name == "tier3_rag.generated_draft.jsonl"


def test_load_chunk_rows_prefers_cached_chunk_corpus(tmp_path):
    module = _load_generator_module()
    corpus_path = tmp_path / "tier3_chunk_corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-001",
                "doc_id": "medical_book",
                "page": 1,
                "section": "Intro",
                "content": "cached content",
                "metadata": {"content_type": "text_section"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module.load_chunk_rows(
        "missing.pdf",
        source="auto",
        chunk_corpus_path=corpus_path,
    )

    assert len(rows) == 1
    assert rows[0]["content"] == "cached content"
    assert rows[0]["chunk_id"] == "chunk-001"


def test_load_chunk_rows_uses_indexed_corpus_and_checkpoints_when_cache_missing(tmp_path, monkeypatch):
    module = _load_generator_module()
    corpus_path = tmp_path / "tier3_chunk_corpus.jsonl"

    monkeypatch.setattr(
        module,
        "_load_chunk_rows_from_indexed_corpus",
        lambda pdf_path: [
            {
                "chunk_id": "chunk-002",
                "doc_id": "medical_book",
                "page": 2,
                "section": "Indexed",
                "content": "indexed content",
                "metadata": {"content_type": "text_section"},
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_load_chunk_rows_from_pdf",
        lambda pdf_path: (_ for _ in ()).throw(AssertionError("pdf fallback should not run")),
    )

    rows = module.load_chunk_rows(
        "medical_book.pdf",
        source="auto",
        chunk_corpus_path=corpus_path,
    )

    saved = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert saved[0]["content"] == "indexed content"
    assert saved[0]["chunk_id"] == "chunk-002"


def test_load_chunk_rows_falls_back_to_pdf_and_checkpoints_when_index_is_unavailable(tmp_path, monkeypatch):
    module = _load_generator_module()
    corpus_path = tmp_path / "tier3_chunk_corpus.jsonl"
    pdf_path = tmp_path / "medical_book.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(module, "_load_chunk_rows_from_indexed_corpus", lambda pdf_path: [])
    monkeypatch.setattr(
        module,
        "_load_chunk_rows_from_pdf",
        lambda pdf_path: [
            {
                "chunk_id": "chunk-003",
                "doc_id": "medical_book",
                "page": 3,
                "section": "PDF",
                "content": "pdf content",
                "metadata": {"content_type": "text_section"},
            }
        ],
    )

    rows = module.load_chunk_rows(
        str(pdf_path),
        source="auto",
        chunk_corpus_path=corpus_path,
    )

    saved = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert saved[0]["content"] == "pdf content"
    assert saved[0]["chunk_id"] == "chunk-003"


def test_project_embeddings_adapter_supports_async_ragas_methods():
    original_embedding_client = _install_fake_embedding_client()
    try:
        module = _load_generator_module()
        adapter = module._ProjectEmbeddingsAdapter()

        assert adapter.embed_documents(["aa", "bbb"]) == [[2.0], [3.0]]
        assert asyncio.run(adapter.aembed_documents(["aa", "bbb"])) == [[2.0], [3.0]]
        assert adapter.embed_query("q") == [2.0]
        assert asyncio.run(adapter.aembed_query("q")) == [2.0]
    finally:
        _restore_fake_embedding_client(original_embedding_client)


def test_resolve_embeddings_uses_tier3_generation_embedding_overrides():
    calls: dict[str, object] = {}
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        embedding_provider = "local"
        embedding_local_model = "Alibaba-NLP/gte-base-en-v1.5"
        embedding_gemini_model = "gemini-embedding-001"
        embedding_openai_model = "text-embedding-3-small"
        embedding_enable_backup = False
        embedding_dim = 768
        tier3_embedding_provider = "openai"
        tier3_embedding_openai_model = "text-embedding-3-large"
        tier3_embedding_dim = 3072

    core_settings_module.get_settings = lambda: Settings()

    embedding_client_module = types.ModuleType("tools.embedding_client")

    class EmbeddingOverrides:
        def __init__(self, *, provider=None, dim=None, openai_model=None):
            self.provider = provider
            self.dim = dim
            self.openai_model = openai_model

    def embed_document(text, *, overrides=None):
        calls["document_overrides"] = overrides
        return [1.0] * int(overrides.dim)

    def embed_query(text, *, overrides=None):
        calls["query_overrides"] = overrides
        return [2.0] * int(overrides.dim)

    embedding_client_module.EmbeddingOverrides = EmbeddingOverrides
    embedding_client_module.embed_document = embed_document
    embedding_client_module.embed_query = embed_query

    original_settings_module = sys.modules.get("core.settings")
    original_embedding_module = sys.modules.get("tools.embedding_client")
    sys.modules["core.settings"] = core_settings_module
    sys.modules["tools.embedding_client"] = embedding_client_module
    try:
        module = _load_generator_module()
        adapter, label = module._resolve_embeddings()

        assert "primary=openai:text-embedding-3-large" in label
        assert "dim=3072" in label
        assert len(adapter.embed_documents(["article"])[0]) == 3072
        assert len(adapter.embed_query("q")) == 3072
        assert calls["document_overrides"].provider == "openai"
        assert calls["document_overrides"].dim == 3072
        assert calls["document_overrides"].openai_model == "text-embedding-3-large"
    finally:
        if original_settings_module is None:
            sys.modules.pop("core.settings", None)
        else:
            sys.modules["core.settings"] = original_settings_module
        if original_embedding_module is None:
            sys.modules.pop("tools.embedding_client", None)
        else:
            sys.modules["tools.embedding_client"] = original_embedding_module


def test_resolve_llm_uses_tier3_generation_llm_overrides():
    calls: dict[str, object] = {}
    core_settings_module = types.ModuleType("core.settings")

    class Settings:
        llm_primary_provider = "groq"
        llm_primary_model = "moonshotai/kimi-k2-instruct-0905"
        llm_fallback_provider = "groq"
        llm_fallback_model = "llama-3.3-70b-versatile"
        tier3_llm_primary_provider = "openai"
        tier3_llm_primary_model = "gpt-4o"
        tier3_llm_fallback_provider = "openai"
        tier3_llm_fallback_model = "gpt-4o"

    core_settings_module.get_settings = lambda: Settings()

    llm_client_module = types.ModuleType("tools.llm_client")

    class FakeLLM:
        model_name = "gpt-4o"

    def resolve_llm(*, primary_provider=None, primary_model=None, fallback_provider=None, fallback_model=None):
        calls["primary_provider"] = primary_provider
        calls["primary_model"] = primary_model
        calls["fallback_provider"] = fallback_provider
        calls["fallback_model"] = fallback_model
        return FakeLLM()

    llm_client_module.resolve_llm = resolve_llm
    original_settings_module = sys.modules.get("core.settings")
    original_llm_module = sys.modules.get("tools.llm_client")
    sys.modules["core.settings"] = core_settings_module
    sys.modules["tools.llm_client"] = llm_client_module
    try:
        module = _load_generator_module()
        llm = module._resolve_llm()

        assert isinstance(llm, FakeLLM)
        assert calls == {
            "primary_provider": "openai",
            "primary_model": "gpt-4o",
            "fallback_provider": "openai",
            "fallback_model": "gpt-4o",
        }
        assert module._get_generator_model_name() == "openai:gpt-4o"
    finally:
        if original_settings_module is None:
            sys.modules.pop("core.settings", None)
        else:
            sys.modules["core.settings"] = original_settings_module
        if original_llm_module is None:
            sys.modules.pop("tools.llm_client", None)
        else:
            sys.modules["tools.llm_client"] = original_llm_module


def test_write_raw_ragas_rows_persists_jsonl(tmp_path):
    module = _load_generator_module()
    rows = [
        {"question": "What is cardiac tamponade?", "evolution_type": "simple"},
        {"question": "How is fever diagnosed?", "synthesizer_name": "multi_context"},
    ]
    output_path = tmp_path / "tier3_rag.raw.jsonl"

    module.write_raw_ragas_rows(rows, output_path)

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert written == rows


def test_load_source_documents_prefers_cached_article_corpus(tmp_path):
    module = _load_generator_module()
    original_core, original_documents = _install_fake_langchain_document()
    article_corpus_path = tmp_path / "tier3_article_corpus.jsonl"
    try:
        article_corpus_path.write_text(
            json.dumps(
                {
                    "content": "Campylobacteriosis is a food-borne infection.",
                    "metadata": {
                        "source_doc_type": "tier3_article",
                        "article_doc_id": "medical_book-article-0001-campylobacteriosis",
                        "doc_id": "medical_book",
                        "article_title": "Campylobacteriosis",
                        "page_start": 18,
                        "page_end": 19,
                        "local_headings": ["Definition", "Description"],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        docs = module.load_source_documents(
            "missing.pdf",
            source="auto",
            article_corpus_path=article_corpus_path,
        )

        assert len(docs) == 1
        assert docs[0].metadata["source_doc_type"] == "tier3_article"
        assert docs[0].metadata["article_title"] == "Campylobacteriosis"
    finally:
        _restore_fake_langchain_document(original_core, original_documents)


def test_load_source_documents_filters_non_entry_cached_article_docs(tmp_path):
    module = _load_generator_module()
    original_core, original_documents = _install_fake_langchain_document()
    article_corpus_path = tmp_path / "tier3_article_corpus.jsonl"
    try:
        rows = [
            {
                "content": "Nancy Ross-Flanigan Science Writer Belleville, MI",
                "metadata": {
                    "source_doc_type": "tier3_article",
                    "article_doc_id": "medical_book-article-0001-contributors",
                    "doc_id": "medical_book",
                    "article_title": "Contributors",
                    "page_start": 9,
                    "page_end": 13,
                    "local_headings": [],
                },
            },
            {
                "content": "1. Smith J. Medical reference text.",
                "metadata": {
                    "source_doc_type": "tier3_article",
                    "article_doc_id": "medical_book-article-0002-bibliography",
                    "doc_id": "medical_book",
                    "article_title": "Bibliography",
                    "page_start": 750,
                    "page_end": 752,
                    "local_headings": [],
                },
            },
            {
                "content": "Campylobacteriosis is a food-borne infection.",
                "metadata": {
                    "source_doc_type": "tier3_article",
                    "article_doc_id": "medical_book-article-0003-campylobacteriosis",
                    "doc_id": "medical_book",
                    "article_title": "Campylobacteriosis",
                    "page_start": 18,
                    "page_end": 19,
                    "local_headings": ["Definition", "Description"],
                },
            },
        ]
        article_corpus_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        docs = module.load_source_documents(
            "missing.pdf",
            source="auto",
            article_corpus_path=article_corpus_path,
        )

        assert [doc.metadata["article_title"] for doc in docs] == ["Campylobacteriosis"]
    finally:
        _restore_fake_langchain_document(original_core, original_documents)


def test_load_source_documents_builds_article_docs_from_pdf_when_cache_missing(tmp_path, monkeypatch):
    module = _load_generator_module()
    original_core, original_documents = _install_fake_langchain_document()
    article_corpus_path = tmp_path / "tier3_article_corpus.jsonl"
    pdf_path = tmp_path / "medical_book.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")
    try:
        fake_chunk_rows = [
            {
                "chunk_id": "medical_book-sec55-p20-t000",
                "doc_id": "medical_book",
                "page": 20,
                "section": "Definition",
                "content": "Campylobacteriosis refers to infection by Campylobacter bacteria.",
                "metadata": {"content_type": "text_section"},
            }
        ]

        class _Doc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        monkeypatch.setattr(module, "load_chunk_rows", lambda *args, **kwargs: fake_chunk_rows)
        monkeypatch.setattr(
            module,
            "_build_article_source_documents_from_pdf",
            lambda pdf_path, chunk_rows, article_limit=None: [
                _Doc(
                    "Campylobacteriosis refers to infection by Campylobacter bacteria.",
                    {
                        "source_doc_type": "tier3_article",
                        "article_doc_id": "medical_book-article-0001-campylobacteriosis",
                        "doc_id": "medical_book",
                        "article_title": "Campylobacteriosis",
                        "page_start": 18,
                        "page_end": 19,
                        "local_headings": ["Definition", "Description"],
                        "aligned_chunk_ids": ["medical_book-sec55-p20-t000"],
                    },
                )
            ],
        )

        docs = module.load_source_documents(
            str(pdf_path),
            source="auto",
            article_corpus_path=article_corpus_path,
        )

        assert len(docs) == 1
        assert docs[0].metadata["article_title"] == "Campylobacteriosis"
        saved = [json.loads(line) for line in article_corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert saved[0]["metadata"]["source_doc_type"] == "tier3_article"
    finally:
        _restore_fake_langchain_document(original_core, original_documents)


def test_context_alignment_maps_article_context_back_to_underlying_chunk_ids():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    article_docs = [
        _Doc(
            "The Pap test is a screening tool rather than a diagnostic tool. Following an abnormal Pap test, a colposcopy is usually performed.",
            {
                "source_doc_type": "tier3_article",
                "article_doc_id": "medical_book-article-0100-cervical-cancer",
                "doc_id": "medical_book",
                "article_title": "Cervical cancer",
                "page_start": 100,
                "page_end": 105,
                "aligned_chunk_ids": ["chunk-pap", "chunk-colposcopy"],
            },
        )
    ]
    chunk_rows = [
        {
            "chunk_id": "chunk-pap",
            "doc_id": "medical_book",
            "page": 101,
            "section": "Diagnosis",
            "content": "The Pap test is a screening tool rather than a diagnostic tool.",
            "metadata": {"content_type": "text_section"},
        },
        {
            "chunk_id": "chunk-colposcopy",
            "doc_id": "medical_book",
            "page": 101,
            "section": "Diagnosis",
            "content": "Following an abnormal Pap test, a colposcopy is usually performed.",
            "metadata": {"content_type": "text_section"},
        },
    ]

    alignment = module._infer_ground_truth_alignment(
        ["Following an abnormal Pap test, a colposcopy is usually performed."],
        article_docs,
        chunk_rows,
    )

    assert alignment["ground_truth_chunk_ids"] == ["chunk-colposcopy"]
    assert alignment["ground_truth_pages"] == [101]


def test_context_alignment_falls_back_beyond_prealigned_chunk_ids_when_needed():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    article_docs = [
        _Doc(
            "The Pap test is a screening tool rather than a diagnostic tool. Following an abnormal Pap test, a colposcopy is usually performed.",
            {
                "source_doc_type": "tier3_article",
                "article_doc_id": "medical_book-article-0100-cervical-cancer",
                "doc_id": "medical_book",
                "article_title": "Cervical cancer",
                "page_start": 100,
                "page_end": 105,
                "aligned_chunk_ids": ["chunk-pap"],
            },
        )
    ]
    chunk_rows = [
        {
            "chunk_id": "chunk-pap",
            "doc_id": "medical_book",
            "page": 101,
            "section": "Diagnosis",
            "content": "The Pap test is a screening tool rather than a diagnostic tool.",
            "metadata": {"content_type": "text_section"},
        },
        {
            "chunk_id": "chunk-colposcopy",
            "doc_id": "medical_book",
            "page": 101,
            "section": "Diagnosis",
            "content": "Following an abnormal Pap test, a colposcopy is usually performed.",
            "metadata": {"content_type": "text_section"},
        },
    ]

    alignment = module._infer_ground_truth_alignment(
        ["Following an abnormal Pap test, a colposcopy is usually performed."],
        article_docs,
        chunk_rows,
    )

    assert alignment["ground_truth_chunk_ids"] == ["chunk-colposcopy"]
    assert alignment["ground_truth_pages"] == [101]


def test_generator_supports_smoke_safe_overrides():
    module = _load_generator_module()

    parser = module._build_arg_parser()
    args = parser.parse_args(
        [
            "--size",
            "4",
            "--article-limit",
            "12",
            "--article-corpus",
            "tmp/article_corpus.jsonl",
            "--raw-output",
            "tmp/raw_rows.jsonl",
            "--output",
            "tmp/draft_rows.jsonl",
        ]
    )

    assert args.article_limit == 12
    assert args.article_corpus.endswith("article_corpus.jsonl")
    assert args.raw_output.endswith("raw_rows.jsonl")
    assert args.output.endswith("draft_rows.jsonl")


def test_dataset_to_schema_v1_maps_ragas_synthesizer_names_and_alignment():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    docs = [
        _Doc(
            "Cardiac tamponade reduces ventricular filling and lowers cardiac output.",
            {"chunk_id": "chunk-tamponade", "doc_id": "medical_book", "page": 73, "content_type": "text_section"},
        ),
        _Doc(
            "Colposcopy is performed after an abnormal Pap test to inspect the cervix and obtain biopsies.",
            {"chunk_id": "chunk-colposcopy", "doc_id": "medical_book", "page": 101, "content_type": "text_section"},
        ),
    ]
    rows = [
        {
            "question": "What happens to the ventricles during cardiac tamponade?",
            "ground_truth": "ventricles fill less",
            "contexts": [docs[0].page_content],
            "synthesizer_name": "single_hop_specifc_query_synthesizer",
        },
        {
            "question": "Why is colposcopy performed after an abnormal Pap test?",
            "ground_truth": "to inspect the cervix",
            "contexts": [docs[1].page_content],
            "synthesizer_name": "multi_hop_specific_query_synthesizer",
        },
    ]

    samples = module.dataset_to_schema_v1(rows, "openai:gpt-4o", docs)

    assert len(samples) == 2
    assert samples[0]["difficulty"] == "simple"
    assert samples[0]["ground_truth_chunk_ids"] == ["chunk-tamponade"]
    assert samples[1]["difficulty"] == "multi_context"
    assert samples[1]["ground_truth_chunk_ids"] == ["chunk-colposcopy"]


def test_convert_dataset_to_schema_v1_skips_missing_question_and_empty_context():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    docs = [
        _Doc(
            "Fever is diagnosed with repeated temperature measurements and a clinical history.",
            {"chunk_id": "chunk-fever", "doc_id": "medical_book", "page": 120, "content_type": "text_section"},
        )
    ]
    rows = [
        {
            "question": "",
            "ground_truth": "missing question",
            "contexts": [docs[0].page_content],
            "synthesizer_name": "single_hop_specifc_query_synthesizer",
        },
        {
            "question": "How is fever diagnosed?",
            "ground_truth": "history and temperature checks",
            "contexts": [],
            "synthesizer_name": "single_hop_specifc_query_synthesizer",
        },
        {
            "question": "How is fever diagnosed?",
            "ground_truth": "history and temperature checks",
            "contexts": [docs[0].page_content],
            "synthesizer_name": "single_hop_specifc_query_synthesizer",
        },
    ]

    result = module.convert_dataset_to_schema_v1(
        rows,
        "openai:gpt-4o",
        docs,
        requested_size=10,
    )

    assert len(result.samples) == 1
    assert result.skipped_missing_question == 1
    assert result.skipped_empty_context == 1


def test_convert_dataset_to_schema_v1_keeps_rows_even_when_chunk_alignment_is_empty():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    article_docs = [
        _Doc(
            "Carbon monoxide poisoning may cause headache and confusion.",
            {
                "source_doc_type": "tier3_article",
                "article_doc_id": "medical_book-article-0040-carbon-monoxide-poisoning",
                "doc_id": "medical_book",
                "article_title": "Carbon monoxide poisoning",
                "page_start": 40,
                "page_end": 41,
                "aligned_chunk_ids": ["unhelpful-chunk"],
            },
        )
    ]
    chunk_rows = [
        {
            "chunk_id": "unhelpful-chunk",
            "doc_id": "medical_book",
            "page": 40,
            "section": "Definition",
            "content": "Carbon monoxide is a toxic gas.",
            "metadata": {"content_type": "text_section"},
        }
    ]
    rows = [
        {
            "question": "What symptoms can this poisoning cause?",
            "ground_truth": "headache and confusion",
            "contexts": ["Carbon monoxide poisoning may cause headache and confusion."],
            "synthesizer_name": "single_hop_specifc_query_synthesizer",
        }
    ]

    samples = module.dataset_to_schema_v1(
        rows,
        "openai:gpt-4o",
        article_docs,
        chunk_rows=chunk_rows,
    )

    assert len(samples) == 1
    assert samples[0]["ground_truth_contexts"] == ["Carbon monoxide poisoning may cause headache and confusion."]
    assert samples[0]["ground_truth_chunk_ids"] == []


def test_dataset_to_schema_v1_raises_when_all_ragas_metadata_is_missing():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    docs = [
        _Doc(
            "Carbon monoxide binds to hemoglobin and reduces oxygen transport.",
            {"chunk_id": "chunk-co", "doc_id": "medical_book", "page": 40, "content_type": "text_section"},
        )
    ]
    rows = [
        {
            "question": "What is carboxyhemoglobin?",
            "ground_truth": "cohb",
            "contexts": [docs[0].page_content],
        }
    ]

    try:
        module.dataset_to_schema_v1(rows, "openai:gpt-4o", docs)
    except RuntimeError as exc:
        message = str(exc)
        assert "missing evolution metadata" in message.lower()
        assert "rebuild" in message.lower()
    else:
        raise AssertionError("expected RuntimeError for missing RAGAS evolution metadata")


def test_print_conversion_summary_reports_raw_accepted_and_skips(capsys):
    module = _load_generator_module()
    result = module.Tier3ConversionResult(
        samples=[
            {"difficulty": "simple"},
            {"difficulty": "multi_context"},
        ],
        raw_rows=[{}, {}, {}, {}],
        skipped_missing_question=1,
        skipped_empty_context=1,
    )

    module._print_conversion_summary(result)
    output = capsys.readouterr().out

    assert "Raw rows: 4" in output
    assert "Accepted rows: 2" in output
    assert "Skipped missing-question rows: 1" in output
    assert "Skipped empty-context rows: 1" in output
    assert "simple" in output
    assert "multi_context" in output


def test_repo_final_curated_tier3_fixture_is_schema_valid_and_approved():
    validate_module = _load_validate_module()
    rows = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden" / "v1" / "tier3_rag.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    errors = []
    for row in rows:
        errors.extend(validate_module.validate_sample(row, "rag"))

    assert len(rows) == 53
    assert errors == []
    assert all((row.get("provenance") or {}).get("source_type") == "human_seed" for row in rows)
    assert all((row.get("provenance") or {}).get("review_status") == "approved" for row in rows)
    assert all("benchmark_set:final_curated_v1" in row.get("tags", []) for row in rows)


def test_select_experiment_source_documents_keeps_only_hardcoded_cancer_family_titles():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, title):
            self.page_content = f"Body for {title}"
            self.metadata = {"article_title": title}

    docs = [
        _Doc("Caffeine"),
        _Doc("Cancer"),
        _Doc("Cancer therapy, definitive"),
        _Doc("Campylobacteriosis"),
        _Doc("Cancer therapy, palliative"),
        _Doc("Cancer therapy, supportive"),
    ]

    selected = module._select_experiment_source_documents(docs)

    assert [doc.metadata["article_title"] for doc in selected] == [
        "Cancer",
        "Cancer therapy, definitive",
        "Cancer therapy, palliative",
        "Cancer therapy, supportive",
    ]


def test_build_tier3_query_distribution_excludes_abstract_multihop_and_biases_specific():
    module = _load_generator_module()
    originals = _install_fake_ragas_profile_modules()
    try:
        distribution = module._build_tier3_query_distribution(object())

        synthesizer_names = [type(synth).__name__ for synth, _weight in distribution]
        assert synthesizer_names == [
            "SingleHopSpecificQuerySynthesizer",
            "MultiHopSpecificQuerySynthesizer",
        ]
        assert distribution[1][1] > distribution[0][1]
    finally:
        _restore_fake_ragas_profile_modules(originals)


def test_filter_query_distribution_for_knowledge_graph_drops_unavailable_synthesizers_and_renormalizes():
    module = _load_generator_module()

    class _Synth:
        def __init__(self, name, clusters):
            self.name = name
            self._clusters = clusters

        def get_node_clusters(self, knowledge_graph):
            return self._clusters

    distribution = [
        (_Synth("single", ["node-a"]), 0.25),
        (_Synth("multi", []), 0.75),
    ]

    filtered = module._filter_query_distribution_for_knowledge_graph(distribution, object())

    assert len(filtered) == 1
    assert filtered[0][0].name == "single"
    assert filtered[0][1] == 1.0


def test_build_tier3_transforms_uses_explicit_narrow_profile_without_headline_splitters():
    module = _load_generator_module()
    originals = _install_fake_ragas_profile_modules()
    try:
        transforms = module._build_tier3_transforms(
            documents=[type("Doc", (), {"page_content": "Cancer therapy uses surgery and radiation to control disease."})()],
            llm=object(),
            embedding_model=object(),
        )

        transform_names = [type(transform).__name__ for transform in transforms]
        assert "SummaryExtractor" in transform_names
        assert "HeadlineSplitter" not in transform_names
        assert "HeadlinesExtractor" not in transform_names
    finally:
        _restore_fake_ragas_profile_modules(originals)


def test_generate_testset_passes_explicit_ragas_profiles():
    module = _load_generator_module()
    captured: dict[str, object] = {}

    class FakeLLMWrapper:
        def __init__(self, llm):
            self.llm = llm

    class FakeEmbeddingsWrapper:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    class FakeGenerator:
        def __init__(self, llm=None, embedding_model=None):
            self.llm = llm
            self.embedding_model = embedding_model
            self.knowledge_graph = None

        def generate(self, *, testset_size, query_distribution=None):
            captured["size"] = testset_size
            captured["query_distribution"] = query_distribution
            captured["knowledge_graph"] = self.knowledge_graph
            return [{"question": "Q", "ground_truth": "A", "contexts": ["ctx"], "synthesizer_name": "single_hop_specifc_query_synthesizer"}]

    module._resolve_ragas_components = lambda: (FakeLLMWrapper, FakeEmbeddingsWrapper, FakeGenerator)
    module._resolve_llm = lambda: object()
    module._resolve_embeddings = lambda: (object(), "primary=openai:text-embedding-3-large dim=3072 backup=off")
    module._resolve_runtime_model_name = lambda llm: "openai:gpt-4o"
    module._build_tier3_query_distribution = lambda llm: "QUERY_PROFILE"
    module._build_tier3_transforms = lambda documents, llm, embedding_model: "TRANSFORM_PROFILE"
    module._build_knowledge_graph_for_documents = lambda docs, transforms: "KG"
    module._filter_query_distribution_for_knowledge_graph = (
        lambda query_distribution, knowledge_graph: [("FILTERED_SYNTH", 1.0)]
    )

    dataset, generator_model = module.generate_testset(["doc-a", "doc-b"], testset_size=4)

    assert generator_model == "openai:gpt-4o"
    assert dataset[0]["question"] == "Q"
    assert captured == {
        "size": 4,
        "query_distribution": [("FILTERED_SYNTH", 1.0)],
        "knowledge_graph": "KG",
    }


def test_build_experiment_generation_documents_bundles_cancer_family_into_two_docs():
    module = _load_generator_module()

    class _Doc:
        def __init__(self, title, content, page_start, page_end, chunk_ids):
            self.page_content = content
            self.metadata = {
                "article_title": title,
                "page_start": page_start,
                "page_end": page_end,
                "aligned_chunk_ids": list(chunk_ids),
                "aligned_chunk_pages": list(range(page_start, page_end + 1)),
                "aligned_chunk_sections": ["Description"],
                "doc_id": "medical_book",
                "local_headings": ["Description"],
                "source_doc_type": "tier3_article",
            }

    docs = [
        _Doc("Cancer", "Cancer overview body.", 20, 26, ["chunk-cancer"]),
        _Doc("Cancer therapy, definitive", "Definitive therapy body.", 28, 29, ["chunk-definitive"]),
        _Doc("Cancer therapy, palliative", "Palliative therapy body.", 30, 30, ["chunk-palliative"]),
        _Doc("Cancer therapy, supportive", "Supportive therapy body.", 31, 33, ["chunk-supportive"]),
    ]

    bundled = module._build_experiment_generation_documents(docs)

    assert len(bundled) == 2
    assert bundled[0].metadata["bundle_article_titles"] == ["Cancer"]
    assert bundled[0].metadata["source_doc_type"] == "tier3_generation_bundle"
    assert bundled[0].metadata["aligned_chunk_ids"] == ["chunk-cancer"]
    assert bundled[1].metadata["bundle_article_titles"] == [
        "Cancer therapy, definitive",
        "Cancer therapy, palliative",
        "Cancer therapy, supportive",
    ]
    assert bundled[1].metadata["aligned_chunk_ids"] == [
        "chunk-definitive",
        "chunk-palliative",
        "chunk-supportive",
    ]
    assert "# Cancer therapy, definitive" in bundled[1].page_content
    assert "# Cancer therapy, supportive" in bundled[1].page_content
