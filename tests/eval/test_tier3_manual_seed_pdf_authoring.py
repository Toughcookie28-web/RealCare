import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_shared_module():
    path = ROOT / "eval" / "tier3_article_documents.py"
    spec = importlib.util.spec_from_file_location(
        "tier3_article_documents_shared_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["tier3_article_documents_shared_test"] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("tier3_article_documents_shared_test", None)


def _load_module():
    path = ROOT / "eval" / "tier3_manual_seed_pdf_authoring.py"
    spec = importlib.util.spec_from_file_location(
        "tier3_manual_seed_pdf_authoring_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["tier3_manual_seed_pdf_authoring_test"] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("tier3_manual_seed_pdf_authoring_test", None)


def test_build_article_windows_groups_pages_by_article_title_and_headings():
    module = _load_module()
    pages = [
        {
            "page": 290,
            "text": "\n".join(
                [
                    "Congenital heart disease",
                    "Definition",
                    "Congenital heart disease is present at birth.",
                    "Diagnosis",
                    "Echocardiography is used to confirm it.",
                    "GALE ENCYCLOPEDIA OF MEDICINE 2",
                    "901",
                    "Congenital heart disease",
                ]
            ),
        },
        {
            "page": 291,
            "text": "\n".join(
                [
                    "Treatment",
                    "Some children need surgery and lifelong monitoring.",
                    "GALE ENCYCLOPEDIA OF MEDICINE 2",
                    "902",
                    "Congenital heart disease",
                ]
            ),
        },
        {
            "page": 292,
            "text": "\n".join(
                [
                    "Conjunctivitis",
                    "Definition",
                    "Conjunctivitis is inflammation of the conjunctiva.",
                    "Causes and symptoms",
                    "Infection or allergy may cause redness.",
                    "GALE ENCYCLOPEDIA OF MEDICINE 2",
                    "903",
                    "Conjunctivitis",
                ]
            ),
        },
    ]

    windows = module.build_article_windows(pages)

    assert windows == [
        module.PdfArticleWindow(
            article_title="Congenital heart disease",
            page_start=290,
            page_end=291,
            local_headings=["Definition", "Diagnosis", "Treatment"],
            body_text=(
                "Congenital heart disease is present at birth.\n"
                "Echocardiography is used to confirm it.\n"
                "Some children need surgery and lifelong monitoring."
            ),
        ),
        module.PdfArticleWindow(
            article_title="Conjunctivitis",
            page_start=292,
            page_end=292,
            local_headings=["Definition", "Causes and symptoms"],
            body_text=(
                "Conjunctivitis is inflammation of the conjunctiva.\n"
                "Infection or allergy may cause redness."
            ),
        ),
    ]


def test_align_article_windows_to_chunks_prefers_page_and_heading_overlap():
    module = _load_module()
    windows = [
        module.PdfArticleWindow(
            article_title="Congenital heart disease",
            page_start=290,
            page_end=291,
            local_headings=["Diagnosis", "Treatment"],
            body_text="Echocardiography confirms the diagnosis and surgery may be required.",
        ),
    ]
    chunk_rows = [
        {
            "chunk_id": "medical_book-sec1724-p290-t000",
            "page": 290,
            "section": "Diagnosis",
            "content": "Echocardiography and MRI are used to confirm congenital heart disease.",
            "metadata": {"section_path": "Diagnosis", "context_prefix": "[p.290] Congenital heart disease > Diagnosis"},
        },
        {
            "chunk_id": "medical_book-sec1725-p291-t000",
            "page": 291,
            "section": "Treatment",
            "content": "Children may need surgery and lifelong monitoring.",
            "metadata": {"section_path": "Treatment", "context_prefix": "[p.291] Congenital heart disease > Treatment"},
        },
        {
            "chunk_id": "medical_book-sec2200-p410-t000",
            "page": 410,
            "section": "Diagnosis",
            "content": "Diagnosis of dermatitis is clinical.",
            "metadata": {"section_path": "Diagnosis", "context_prefix": "[p.410] Dermatitis > Diagnosis"},
        },
    ]

    aligned = module.align_article_windows_to_chunks(windows, chunk_rows, max_chunks=4)

    assert aligned == [
        module.ChunkAlignment(
            article_title="Congenital heart disease",
            page_span=(290, 291),
            chunk_ids=["medical_book-sec1724-p290-t000", "medical_book-sec1725-p291-t000"],
            chunk_pages=[290, 291],
            chunk_sections=["Diagnosis", "Treatment"],
            local_headings=["Diagnosis", "Treatment"],
        )
    ]


def test_summarize_alignment_counts_aligned_and_unaligned_windows():
    module = _load_module()
    aligned_rows = [
        module.ChunkAlignment(
            article_title="Congenital heart disease",
            page_span=(290, 291),
            chunk_ids=["medical_book-sec1724-p290-t000"],
            chunk_pages=[290],
            chunk_sections=["Diagnosis"],
            local_headings=["Diagnosis"],
        ),
        module.ChunkAlignment(
            article_title="Conjunctivitis",
            page_span=(292, 292),
            chunk_ids=[],
            chunk_pages=[],
            chunk_sections=[],
            local_headings=["Definition"],
        ),
    ]

    assert module.summarize_alignment(aligned_rows) == {
        "total_windows": 2,
        "aligned_windows": 1,
        "unaligned_windows": 1,
    }


def test_manual_seed_authoring_helper_reuses_shared_article_window_logic():
    module = _load_module()
    shared = _load_shared_module()
    pages = [
        {
            "page": 290,
            "text": "\n".join(
                [
                    "Congenital heart disease",
                    "Definition",
                    "Congenital heart disease is present at birth.",
                    "Diagnosis",
                    "Echocardiography is used to confirm it.",
                    "GALE ENCYCLOPEDIA OF MEDICINE 2",
                    "901",
                    "Congenital heart disease",
                ]
            ),
        },
        {
            "page": 291,
            "text": "\n".join(
                [
                    "Treatment",
                    "Some children need surgery and lifelong monitoring.",
                    "GALE ENCYCLOPEDIA OF MEDICINE 2",
                    "902",
                    "Congenital heart disease",
                ]
            ),
        },
    ]

    assert [window.__dict__ for window in module.build_article_windows(pages)] == [
        window.__dict__ for window in shared.build_article_windows(pages)
    ]


def test_manual_seed_authoring_helper_reuses_shared_chunk_alignment_logic():
    module = _load_module()
    shared = _load_shared_module()
    windows = [
        module.PdfArticleWindow(
            article_title="Congenital heart disease",
            page_start=290,
            page_end=291,
            local_headings=["Diagnosis", "Treatment"],
            body_text="Echocardiography confirms the diagnosis and surgery may be required.",
        ),
    ]
    chunk_rows = [
        {
            "chunk_id": "medical_book-sec1724-p290-t000",
            "page": 290,
            "section": "Diagnosis",
            "content": "Echocardiography and MRI are used to confirm congenital heart disease.",
            "metadata": {"section_path": "Diagnosis", "context_prefix": "[p.290] Congenital heart disease > Diagnosis"},
        },
        {
            "chunk_id": "medical_book-sec1725-p291-t000",
            "page": 291,
            "section": "Treatment",
            "content": "Children may need surgery and lifelong monitoring.",
            "metadata": {"section_path": "Treatment", "context_prefix": "[p.291] Congenital heart disease > Treatment"},
        },
    ]

    assert [row.__dict__ for row in module.align_article_windows_to_chunks(windows, chunk_rows)] == [
        row.__dict__ for row in shared.align_article_windows_to_chunks(windows, chunk_rows)
    ]
