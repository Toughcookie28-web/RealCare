from __future__ import annotations

import hashlib
import logging

from core.settings import get_settings
from core.state_v2 import AgentStateV2
from tools.cache import get_semantic_cache
from tools.llm_client import invoke_llm

logger = logging.getLogger(__name__)

# Conservative prompt budgeting for mixed prose + table retrieval context.
_MAX_TOTAL_TOKENS = 16000
_GENERATION_HEADROOM = 1200
_MAX_PROMPT_TOKENS = _MAX_TOTAL_TOKENS - _GENERATION_HEADROOM
_MAX_CONTEXT_DOCS = 5
_SYSTEM_PROMPT = (
    'You are a clinical education assistant. '
    'Do not provide definitive diagnosis. '
    'Give educational guidance and advise professional medical consultation where appropriate.'
)
_PROMPT_TEMPLATE = """
User question: {question}
Rewritten query: {query}
Conversation summary: {summary}
Known user facts: {facts}

Retrieved context:
{context}

The retrieved context may contain prose, Markdown tables, or HTML-like table
structure. Pay careful attention to section headers, table titles/captions,
column headers, and row alignment when extracting facts.

If user asks diagnosis or dangerous dosage, decline diagnosis and perform educational pivot.
Provide concise, practical, medically safe educational guidance in 3-6 bullet points.
""".strip()

_MEMORY_PROMPT_TEMPLATE = """
User question: {question}
Conversation summary: {summary}
Known user facts: {facts}

Recent conversation:
{history_text}

The user is asking about a previous conversation. Answer based ONLY on the conversation
history and summary above. If the information they're asking about is not in the history,
say so honestly. Do not make up or infer information that wasn't discussed.
""".strip()


_PROMPT_TEMPLATE_V2 = """
User question: {question}
Rewritten query: {query}
Conversation summary: {summary}
Known user facts: {facts}
{session_line}
{query_context_line}
{episodic_block}

Retrieved context:
{context}

The retrieved context may contain prose, Markdown tables, or HTML-like table
structure. Pay careful attention to section headers, table titles/captions,
column headers, and row alignment when extracting facts.

If user asks diagnosis or dangerous dosage, decline diagnosis and perform educational pivot.
Provide concise, practical, medically safe educational guidance in 3-6 bullet points.
""".strip()


def _format_episodic_memories(episodes: list[dict]) -> str:
    """Format episodic memories as a readable block for the executor prompt."""
    if not episodes:
        return ""
    lines = ["**Relevant past context from prior sessions:**"]
    for ep in episodes:
        date_str = ""
        created = ep.get("created_at")
        if created:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created)
                date_str = f"[{dt.strftime('%b %d')}] "
            except Exception:
                pass
        lines.append(f"- {date_str}{ep['summary']}")
    return "\n".join(lines)


def _render_prompt_optimized(
    question: str, query: str, summary: str, facts: str, context: str,
    query_context: str = '', session_intent: str = '',
    episodic_memories: str = '',
) -> str:
    return _PROMPT_TEMPLATE_V2.format(
        question=question,
        query=query,
        summary=summary,
        facts=facts,
        context=context,
        session_line=f'Session goal: {session_intent}' if session_intent else '',
        query_context_line=f'Query context: {query_context}' if query_context else '',
        episodic_block=episodic_memories,
    )


def _render_memory_prompt(question: str, summary: str, facts: str, history_text: str) -> str:
    return _MEMORY_PROMPT_TEMPLATE.format(
        question=question,
        summary=summary,
        facts=facts,
        history_text=history_text,
    )


try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text.split())


def _render_prompt(question: str, query: str, summary: str, facts: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(
        question=question,
        query=query,
        summary=summary,
        facts=facts,
        context=context,
    )


def _build_context_blocks(docs: list, vector_repo=None) -> list[str]:
    result_chunk_ids = {doc.metadata.get('chunk_id') for doc in docs}

    # Sort: intro chunks first, then body, then conclusion
    def _position_sort_key(d):
        pos = (d.metadata or {}).get('chunk_position', 'body')
        return 0 if pos == 'intro' else 1 if pos == 'body' else 2
    docs = sorted(docs, key=_position_sort_key)

    grouped: dict[str, dict[str, list[str] | str]] = {}
    order: list[str] = []

    for doc in docs:
        metadata = doc.metadata or {}
        header = (
            metadata.get('context_prefix')
            or metadata.get('section_path')
            or metadata.get('section')
            or metadata.get('doc_id')
            or 'Retrieved evidence'
        )
        key = metadata.get('context_key') or header
        if key not in grouped:
            grouped[key] = {'header': header, 'chunks': []}
            order.append(key)

        # Expand: fetch neighbors if repo available
        before_texts = []
        after_texts = []
        if vector_repo is not None:
            section = metadata.get('section')
            chunk_index = metadata.get('chunk_index')
            if section and chunk_index is not None:
                try:
                    neighbors = vector_repo.get_adjacent_chunks(section, chunk_index)
                    for nb in neighbors:
                        nb_id = nb.metadata.get('chunk_id')
                        if nb_id in result_chunk_ids:
                            continue
                        nb_idx = nb.metadata.get('chunk_index', 0)
                        if nb_idx < chunk_index:
                            before_texts.append(nb.page_content.strip())
                        else:
                            after_texts.append(nb.page_content.strip())
                except Exception:
                    pass

        parts = []
        # Prepend context summary if available
        context_summary = metadata.get('context_summary')
        if context_summary:
            parts.append(f"[Context: {context_summary}]")
        # Use parent content if available (parent-child chunking),
        # otherwise use child content with optional adjacent expansion
        parent_content = metadata.get('parent_content')
        if parent_content:
            parts.append(f"<chunk>\n{parent_content.strip()}\n</chunk>")
        else:
            for t in before_texts:
                parts.append(f"<context>\n{t}\n</context>")
            parts.append(f"<chunk>\n{doc.page_content.strip()}\n</chunk>")
            for t in after_texts:
                parts.append(f"<context>\n{t}\n</context>")

        grouped[key]['chunks'].append('\n'.join(parts))

    blocks: list[str] = []
    for key in order:
        group = grouped[key]
        block_parts = [f"### {group['header']}"]
        block_parts.extend(group['chunks'])
        blocks.append('\n\n'.join(block_parts))
    return blocks


def _executor_cache_namespace() -> str:
    settings = get_settings()
    contract = '\n'.join(
        (
            settings.semantic_cache_version,
            _SYSTEM_PROMPT,
            _PROMPT_TEMPLATE,
            f'context_doc_limit={_MAX_CONTEXT_DOCS}',
            f'max_prompt_tokens={_MAX_PROMPT_TOKENS}',
            'context_builder=section_grouping_v1',
        )
    )
    digest = hashlib.sha256(contract.encode('utf-8')).hexdigest()[:12]
    return f'executor:{digest}'


def _should_use_semantic_cache(state: AgentStateV2) -> bool:
    if state.get('route') == 'clarify':
        return False
    if state.get('needs_retry'):
        return False
    if state.get('reflection_feedback'):
        return False
    attempts = state.get('attempts', {})
    return attempts.get('reflection', 0) == 0 and attempts.get('executor', 0) == 0


def _build_citations(docs: list) -> list[dict]:
    """Extract deduplicated section/page citations from retrieved docs."""
    seen: set[tuple] = set()
    citations: list[dict] = []
    for doc in docs:
        meta = doc.metadata or {}
        section = meta.get('section')
        if not section:
            continue
        page = meta.get('page')
        key = (section, page)
        if key in seen:
            continue
        seen.add(key)
        citation: dict = {'section': section, 'doc_id': meta.get('doc_id', '')}
        if page is not None:
            citation['page'] = page
        citations.append(citation)
    return citations


def _format_citations_block(citations: list[dict]) -> str:
    """Format citations as a readable Sources block."""
    if not citations:
        return ''
    lines = ['**Sources:**']
    for c in citations:
        parts = [f"Section: \"{c['section']}\""]
        if c.get('page') is not None:
            parts.append(f"Page {c['page']}")
        lines.append(f"- {', '.join(parts)}")
    return '\n'.join(lines)


def ExecutorAgent(state: AgentStateV2) -> AgentStateV2:
    cache = get_semantic_cache()
    query = state.get('optimized_query') or state.get('question', '')
    cache_namespace = _executor_cache_namespace()

    cached = cache.get(query, namespace=cache_namespace) if _should_use_semantic_cache(state) else None
    if cached is not None:
        state['generation'] = cached
        state['source'] = 'Semantic Cache'
        state['semantic_cache_hit'] = True
        return state

    # Clarify route: return clarification question directly, no retrieval or LLM generation
    if state.get('route') == 'clarify':
        clarification_q = state.get('clarification_question', '')
        state['generation'] = clarification_q if clarification_q else (
            "Could you clarify what you're looking for? "
            "Your question could be interpreted in a few different ways."
        )
        state['source'] = 'Clarification Request'
        return state

    question = state.get('question', '')
    summary = state.get('summary', '')
    facts = ', '.join([f"{f.get('key')}={f.get('value')}" for f in state.get('facts', [])])

    # Memory route: generate from conversation history, not retrieved docs
    if state.get('route') == 'memory':
        history = state.get('conversation_history', [])
        history_text = '\n'.join(
            [f"{item.get('role')}: {item.get('content')}" for item in history]
        ) if history else ''

        if not history_text and not summary:
            state['generation'] = (
                "I don't have any previous conversation history to reference. "
                "This appears to be the start of our conversation."
            )
            state['source'] = 'Memory (no history)'
            return state

        prompt = _render_memory_prompt(question, summary, facts, history_text)
        answer = invoke_llm(prompt, system=_SYSTEM_PROMPT, use_fallback=False)
        if not answer:
            answer = invoke_llm(prompt, system=_SYSTEM_PROMPT, use_fallback=True)
        if not answer:
            answer = "I couldn't retrieve our previous conversation details at the moment."

        state['generation'] = answer
        state['source'] = 'Memory (Conversation History)'
        return state

    docs = state.get('documents', [])[:_MAX_CONTEXT_DOCS]

    episodic_block = _format_episodic_memories(state.get('episodic_memories', []))

    vector_repo = state.get('vector_repo')
    context_blocks: list[str] = []
    for block in _build_context_blocks(docs, vector_repo=vector_repo):
        candidate_context = '\n\n'.join(context_blocks + [block])
        candidate_prompt = _render_prompt_optimized(
            question, query, summary, facts, candidate_context,
            query_context=state.get('query_context', ''),
            session_intent=state.get('session_intent', ''),
            episodic_memories=episodic_block,
        )
        prompt_tokens = _count_tokens(_SYSTEM_PROMPT) + _count_tokens(candidate_prompt)
        if prompt_tokens > _MAX_PROMPT_TOKENS:
            break
        context_blocks.append(block)

    context = '\n\n'.join(context_blocks)

    prompt = _render_prompt_optimized(
        question, query, summary, facts, context,
        query_context=state.get('query_context', ''),
        session_intent=state.get('session_intent', ''),
        episodic_memories=episodic_block,
    )

    answer = invoke_llm(prompt, system=_SYSTEM_PROMPT, use_fallback=False)
    if not answer:
        answer = invoke_llm(prompt, system=_SYSTEM_PROMPT, use_fallback=True)

    if not answer:
        answer = (
            'I could not safely generate a medical answer at the moment. '
            'Please consult a qualified healthcare professional for direct guidance.'
        )
        state['source'] = 'System Fallback'
    else:
        # Always set source for a successful execution. Do not preserve 'System Message'
        # which is set by run_node()'s error handler when an upstream node failed — that
        # error state is stale once the executor produces a real answer.
        if state.get('source') not in {'Semantic Cache', 'Clarification Request',
                                        'Memory (no history)', 'Memory (Conversation History)'}:
            state['source'] = 'LLM + Retrieved Evidence'

    state['generation'] = answer

    # Build and append citations
    citations = _build_citations(docs)
    state['citations'] = citations
    if citations:
        citation_block = _format_citations_block(citations)
        state['generation'] = f"{answer}\n\n{citation_block}"

    cache.set(query, state['generation'], namespace=cache_namespace)

    return state
