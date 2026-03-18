from __future__ import annotations

import os
from typing import Any

import requests
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
try:
    from langchain_tavily import TavilySearch as TavilySearchTool
except Exception:  # pragma: no cover - optional new package path
    TavilySearchTool = None
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except Exception:  # pragma: no cover - optional legacy fallback
    TavilySearchResults = None

from core.settings import get_settings


_wiki_wrapper = None
_tavily_search = None


def _clean_secret(value: str | None) -> str:
    if value is None:
        return ''
    value = value.strip()
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1].strip()
    return value


class TavilySearchAdapter:
    def __init__(self, tool: Any):
        self.tool = tool

    def invoke(self, query: str) -> list[dict[str, Any]]:
        try:
            payload = self.tool.invoke(query)
        except Exception:
            return []

        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get('results'), list):
                return [item for item in payload['results'] if isinstance(item, dict)]
            if isinstance(payload.get('data'), list):
                return [item for item in payload['data'] if isinstance(item, dict)]
            return [payload]
        return []


def get_wikipedia_wrapper() -> WikipediaAPIWrapper:
    global _wiki_wrapper
    if _wiki_wrapper is None:
        _wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=2000,
            load_all_available_meta=True,
        )
    return _wiki_wrapper


def get_tavily_search() -> TavilySearchAdapter | None:
    global _tavily_search
    settings = get_settings()
    if _tavily_search is None:
        api_key = _clean_secret(settings.tavily_api_key)
        if not api_key:
            return None
        if TavilySearchTool is not None:
            os.environ['TAVILY_API_KEY'] = api_key
            for kwargs in (
                {'max_results': 5, 'include_domains': settings.trusted_domains_list},
                {'max_results': 5},
                {},
            ):
                try:
                    _tavily_search = TavilySearchAdapter(TavilySearchTool(**kwargs))
                    break
                except TypeError:
                    continue
                except Exception:
                    continue

        if _tavily_search is None and TavilySearchResults is not None:
            _tavily_search = TavilySearchAdapter(
                TavilySearchResults(
                    api_key=api_key,
                    max_results=5,
                    include_domains=settings.trusted_domains_list,
                )
            )

    return _tavily_search


def fetch_pubmed_abstracts(query: str, retmax: int = 3) -> list[dict]:
    # NCBI ESearch -> EFetch pipeline
    search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    fetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

    search_resp = requests.get(
        search_url,
        params={'db': 'pubmed', 'term': query, 'retmode': 'json', 'retmax': retmax},
        timeout=10,
    )
    if search_resp.status_code != 200:
        return []
    ids = search_resp.json().get('esearchresult', {}).get('idlist', [])
    if not ids:
        return []

    fetch_resp = requests.get(
        fetch_url,
        params={'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'text', 'rettype': 'abstract'},
        timeout=12,
    )
    if fetch_resp.status_code != 200:
        return []

    text = fetch_resp.text.strip()
    if not text:
        return []

    abstracts = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    return [{'source': 'PubMed', 'content': a[:2000]} for a in abstracts[:retmax]]
