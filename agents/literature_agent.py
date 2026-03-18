from __future__ import annotations

from langchain_core.documents import Document

from core.state_v2 import AgentStateV2
from tools.search_tools import fetch_pubmed_abstracts


def LiteratureAgent(state: AgentStateV2) -> AgentStateV2:
    query = state.get('optimized_query') or state.get('question', '')
    abstracts = fetch_pubmed_abstracts(query, retmax=5)
    docs = [Document(page_content=item['content'], metadata={'source': item.get('source', 'PubMed')}) for item in abstracts]
    state['documents'] = docs
    if docs:
        state['source'] = 'PubMed Literature'
    return state
