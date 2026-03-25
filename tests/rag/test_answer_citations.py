from langchain_core.documents import Document
from agents.executor_agent import _build_citations


def _doc(section=None, page=None, doc_id='book1'):
    meta = {'doc_id': doc_id}
    if section is not None:
        meta['section'] = section
    if page is not None:
        meta['page'] = page
    return Document(page_content='content', metadata=meta)


def test_citations_basic():
    docs = [
        _doc(section='Cardiovascular', page=142),
        _doc(section='Antiplatelet', page=145),
    ]
    citations = _build_citations(docs)
    assert len(citations) == 2
    assert citations[0]['section'] == 'Cardiovascular'
    assert citations[0]['page'] == 142


def test_citations_dedup_same_section_page():
    docs = [
        _doc(section='Cardiovascular', page=142),
        _doc(section='Cardiovascular', page=142),
    ]
    citations = _build_citations(docs)
    assert len(citations) == 1


def test_citations_missing_section():
    docs = [_doc(page=100)]
    citations = _build_citations(docs)
    assert len(citations) == 0


def test_citations_missing_page():
    docs = [_doc(section='Cardiovascular')]
    citations = _build_citations(docs)
    assert len(citations) == 1
    assert citations[0].get('page') is None


def test_citations_empty_docs():
    citations = _build_citations([])
    assert citations == []
