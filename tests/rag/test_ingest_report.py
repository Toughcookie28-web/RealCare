import importlib.util
import sys
import types
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


def _load_module(module_name: str, relative_path: str, injected_modules: dict[str, types.ModuleType]):
    original_modules = {name: sys.modules.get(name) for name in injected_modules}
    sys.modules.update(injected_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
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


def test_ingest_report_preserves_stage_counts():
    module = _load_source_module("ingest_report", ROOT / "tools" / "ingest_report.py")

    report = module.IngestReport(
        doc_id="medical_book",
        parsed_elements=10,
        chunks=5,
        embeddings=5,
        inserted=5,
    )

    assert report.chunks == 5
    assert report.inserted == 5


def test_pdf_loader_exposes_parse_and_chunk_boundaries():
    source = (ROOT / "tools" / "pdf_loader.py").read_text(encoding="utf-8")

    assert "def load_parsed_pdf(" in source
    assert "def chunk_pdf_document(" in source
    assert "def process_pdf(" in source


def test_vector_store_ingest_returns_stage_report():
    sqlalchemy_orm_module = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm_module.Session = object

    db_repositories_module = types.ModuleType("db.repositories")

    class InMemoryVectorRepository:
        def __init__(self):
            self.batch = []

        def upsert_chunks_batch(self, chunks, replace_doc_id=None, batch_size=250):
            self.batch = list(chunks)
            return len(chunks)

        def count_chunks(self):
            return len(self.batch)

        def hybrid_search(self, query, query_embedding, k=12):
            return []

    class VectorRepository(InMemoryVectorRepository):
        pass

    db_repositories_module.InMemoryVectorRepository = InMemoryVectorRepository
    db_repositories_module.VectorRepository = VectorRepository

    parsed_doc = types.SimpleNamespace(doc_id="medical_book", elements=[1, 2, 3, 4])
    fake_chunk = types.SimpleNamespace(
        page_content="chunk text",
        metadata={"chunk_id": "chunk-0", "doc_id": "medical_book", "page": 1, "section": "Intro"},
    )

    tools_pdf_loader_module = types.ModuleType("tools.pdf_loader")
    tools_pdf_loader_module.load_parsed_pdf = lambda pdf_path: parsed_doc
    tools_pdf_loader_module.chunk_pdf_document = lambda doc: [fake_chunk, fake_chunk]
    tools_pdf_loader_module.process_pdf = lambda pdf_path: [fake_chunk, fake_chunk]

    tools_embedding_client_module = types.ModuleType("tools.embedding_client")
    tools_embedding_client_module.embed_documents_batch = lambda texts, batch_size=32: [[0.1, 0.2] for _ in texts]
    tools_embedding_client_module.embed_query = lambda text: [0.1, 0.2]

    tools_ingest_report_module = types.ModuleType("tools.ingest_report")
    ingest_report_module = _load_source_module("ingest_report_module", ROOT / "tools" / "ingest_report.py")
    tools_ingest_report_module.IngestReport = ingest_report_module.IngestReport

    module = _load_module(
        "batch_d_vector_store",
        "tools/vector_store.py",
        {
            "sqlalchemy.orm": sqlalchemy_orm_module,
            "db.repositories": db_repositories_module,
            "tools.pdf_loader": tools_pdf_loader_module,
            "tools.embedding_client": tools_embedding_client_module,
            "tools.ingest_report": tools_ingest_report_module,
        },
    )

    report = module.ingest_pdf_to_vector_store(db=None, pdf_path="/tmp/medical_book.pdf")

    assert report.doc_id == "medical_book"
    assert report.parsed_elements == 4
    assert report.chunks == 2
    assert report.embeddings == 2
    assert report.inserted == 2
