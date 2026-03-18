from __future__ import annotations

from langchain_core.documents import Document

from core.state_v2 import AgentStateV2
from tools.search_tools import get_wikipedia_wrapper


def WikipediaAgent(state: AgentStateV2) -> AgentStateV2:
    wiki = get_wikipedia_wrapper()
    query = state.get('optimized_query') or state.get('question', '')

    content = wiki.run(f'{query} medical') or wiki.run(query)
    if content and len(content.strip()) > 120:
        state['documents'] = [Document(page_content=content, metadata={'source': 'wikipedia'})]
        state['source'] = 'Wikipedia Medical Information'
    else:
        state['documents'] = []
    return state
