from langchain_core.documents import Document
from agents.executor_agent import _build_context_blocks


def _doc(content, section='sec1', context_summary=None, chunk_position=None,
         section_path=None, chunk_index=0):
    meta = {'section': section, 'chunk_index': chunk_index}
    if context_summary is not None:
        meta['context_summary'] = context_summary
    if chunk_position is not None:
        meta['chunk_position'] = chunk_position
    if section_path is not None:
        meta['section_path'] = section_path
    return Document(page_content=content, metadata=meta)


def test_context_summary_prepended():
    docs = [_doc('Aspirin inhibits COX-1.',
                 context_summary='Overview of aspirin pharmacology')]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    assert '[Context: Overview of aspirin pharmacology]' in block_text
    assert 'Aspirin inhibits COX-1.' in block_text


def test_no_context_summary_unchanged():
    docs = [_doc('Aspirin inhibits COX-1.')]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    assert '[Context:' not in block_text
    assert 'Aspirin inhibits COX-1.' in block_text


def test_intro_chunks_ordered_first():
    docs = [
        _doc('Body content', section='pharmacology', chunk_position='body', chunk_index=1),
        _doc('Intro content', section='pharmacology', chunk_position='intro', chunk_index=0),
    ]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    intro_pos = block_text.index('Intro content')
    body_pos = block_text.index('Body content')
    assert intro_pos < body_pos


def test_section_path_in_header():
    docs = [_doc('content', section_path='Ch5 > Cardiovascular > Pharmacology')]
    blocks = _build_context_blocks(docs)
    block_text = '\n'.join(blocks)
    assert 'Ch5 > Cardiovascular > Pharmacology' in block_text
