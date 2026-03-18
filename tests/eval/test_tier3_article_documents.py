import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "eval" / "tier3_article_documents.py"
    spec = importlib.util.spec_from_file_location("tier3_article_documents_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["tier3_article_documents_test"] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("tier3_article_documents_test", None)


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
        module.ArticleWindow(
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
        module.ArticleWindow(
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
        module.ArticleWindow(
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


def test_build_article_windows_drops_leading_carryover_and_intra_article_back_matter():
    module = _load_module()
    parsed_pages = [
        {
            "page": 20,
            "text": "\n".join(
                [
                    "Campylobacter is only one of many causes of acute diarrhea.",
                    "ORGANIZATIONS",
                    "Centers for Disease Control and Prevention.",
                    "Cancer",
                    "Definition",
                    "Cancer is not just one disease, but a large group of diseases.",
                    "Description",
                    "Cancer involves uncontrolled cell growth and spread.",
                    "BOOKS",
                    "National Cancer Institute Monograph.",
                    "GALE ENCYCLOPEDIA OF MEDICINE 2",
                    "20",
                    "Cancer",
                ]
            ),
        }
    ]

    windows = module.build_article_windows(parsed_pages)

    assert windows == [
        module.ArticleWindow(
            article_title="Cancer",
            page_start=20,
            page_end=20,
            local_headings=["Definition", "Description"],
            body_text=(
                "Cancer is not just one disease, but a large group of diseases.\n"
                "Cancer involves uncontrolled cell growth and spread."
            ),
        )
    ]
