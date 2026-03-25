from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.common import run_node
from agents.executor_agent import ExecutorAgent
from agents.guardrail_agent import GuardrailAgent
from agents.literature_agent import LiteratureAgent
from agents.llm_agent import LLMAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.query_rewriter_agent import QueryRewriterAgent
from agents.reflection_agent import ReflectionAgent
from agents.retriever_agent import RetrieverAgent
from agents.tavily_agent import TavilyAgent
from agents.wikipedia_agent import WikipediaAgent
from core.state_v2 import AgentStateV2


def _guardrail_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('guardrail', GuardrailAgent, state)


def _memory_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('memory', MemoryAgent, state)


def _rewriter_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('query_rewriter', QueryRewriterAgent, state)


def _planner_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('planner', PlannerAgent, state)


def _retriever_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('retriever', RetrieverAgent, state)


def _web_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('tavily', TavilyAgent, state)


def _wiki_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('wikipedia', WikipediaAgent, state)


def _lit_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('literature', LiteratureAgent, state)


def _llm_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('llm', LLMAgent, state)


def _executor_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('executor', ExecutorAgent, state)


def _reflection_node(state: AgentStateV2) -> AgentStateV2:
    return run_node('reflection', ReflectionAgent, state)


def route_after_guardrail(state: AgentStateV2) -> str:
    return 'blocked' if state.get('safety_flags', {}).get('blocked') else 'continue'


def route_after_planner(state: AgentStateV2) -> str:
    route = state.get('route', 'vector')
    if route == 'clarify':
        return 'executor'
    if route == 'chitchat':
        return 'chitchat'
    if route == 'memory':
        return 'memory'
    # Retriever-first policy for all non-chitchat, non-memory routes.
    return 'retriever'


def route_after_retriever(state: AgentStateV2) -> str:
    route = state.get('post_retrieval_route', 'executor')
    if route not in {'executor', 'web', 'literature'}:
        return 'executor'
    return route


def route_after_web(state: AgentStateV2) -> str:
    return 'executor' if state.get('documents') else 'wikipedia'


def route_after_reflection(state: AgentStateV2) -> str:
    return 'retry' if state.get('needs_retry') else 'finalize'


def create_workflow():
    workflow = StateGraph(AgentStateV2)

    workflow.add_node('guardrail', _guardrail_node)
    workflow.add_node('memory', _memory_node)
    workflow.add_node('rewriter', _rewriter_node)
    workflow.add_node('planner', _planner_node)

    workflow.add_node('retriever', _retriever_node)
    workflow.add_node('tavily', _web_node)
    workflow.add_node('wikipedia', _wiki_node)
    workflow.add_node('literature', _lit_node)
    workflow.add_node('llm', _llm_node)

    workflow.add_node('executor', _executor_node)
    workflow.add_node('reflection', _reflection_node)

    workflow.set_entry_point('guardrail')

    workflow.add_conditional_edges(
        'guardrail',
        route_after_guardrail,
        {
            'blocked': END,
            'continue': 'memory',
        },
    )

    workflow.add_edge('memory', 'rewriter')
    workflow.add_edge('rewriter', 'planner')

    workflow.add_conditional_edges(
        'planner',
        route_after_planner,
        {
            'executor': 'executor',
            'chitchat': 'llm',
            'memory': 'executor',
            'retriever': 'retriever',
        },
    )

    workflow.add_conditional_edges(
        'retriever',
        route_after_retriever,
        {
            'executor': 'executor',
            'web': 'tavily',
            'literature': 'literature',
        },
    )
    workflow.add_conditional_edges(
        'tavily',
        route_after_web,
        {
            'executor': 'executor',
            'wikipedia': 'wikipedia',
        },
    )
    workflow.add_edge('wikipedia', 'executor')
    workflow.add_edge('literature', 'executor')
    workflow.add_edge('llm', 'executor')

    workflow.add_edge('executor', 'reflection')
    workflow.add_conditional_edges(
        'reflection',
        route_after_reflection,
        {
            'retry': 'retriever',
            'finalize': END,
        },
    )

    return workflow.compile()
