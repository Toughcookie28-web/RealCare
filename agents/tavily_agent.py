from __future__ import annotations

import json
import logging

from langchain_core.documents import Document

from core.state_v2 import AgentStateV2
from tools.redis_client import ExactCache
from tools.search_tools import get_tavily_search

logger = logging.getLogger(__name__)
_tavily_cache = ExactCache(prefix="tavily", ttl=86400)  # 24 hour TTL


def TavilyAgent(state: AgentStateV2) -> AgentStateV2:
    tavily = get_tavily_search()
    if tavily is None:
        state['documents'] = []
        return state

    query = state.get('optimized_query') or state.get('question', '')

    # Check cache first
    cached = _tavily_cache.get(query)
    if cached is not None:
        try:
            results = json.loads(cached)
            logger.info("Tavily cache hit (query=%s...)", query[:40])
        except Exception:
            logger.warning("Tavily cache entry corrupt, calling API")
            results = tavily.invoke(query)
            if results:
                _tavily_cache.set(query, json.dumps(results))
    else:
        results = tavily.invoke(query)
        if results:
            _tavily_cache.set(query, json.dumps(results))

    docs = []
    for result in results or []:
        content = result.get('content', '') if isinstance(result, dict) else ''
        if len(content.strip()) > 80:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'source': 'tavily',
                    },
                )
            )

    state['documents'] = docs
    if docs:
        state['source'] = 'Trusted Medical Web Search'
    return state
