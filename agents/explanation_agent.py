from __future__ import annotations

from core.state_v2 import AgentStateV2


def ExplanationAgent(state: AgentStateV2) -> AgentStateV2:
    citations = []
    for doc in state.get('documents', [])[:5]:
        metadata = doc.metadata or {}
        citations.append(
            {
                'source': metadata.get('doc_id') or metadata.get('source') or 'medical_source',
                'page': metadata.get('page'),
                'section': metadata.get('section'),
                'chunk_id': metadata.get('chunk_id'),
                'url': metadata.get('url'),
            }
        )

    state['citations'] = citations
    if citations:
        citation_text = '\n'.join(
            [
                f"[Source: {c['source']}, Page {c.get('page')}, Section: {c.get('section')}]"
                for c in citations
            ]
        )
        state['generation'] = f"{state.get('generation', '').strip()}\n\n{citation_text}"
    return state
