from __future__ import annotations

from core.state_v2 import AgentStateV2
from tools.llm_client import invoke_llm


def LLMAgent(state: AgentStateV2) -> AgentStateV2:
    history = '\n'.join([f"{m.get('role')}: {m.get('content')}" for m in state.get('conversation_history', [])[-6:]])
    summary = state.get('summary', '')
    facts = ', '.join([f"{f.get('key')}: {f.get('value')}" for f in state.get('facts', [])])

    prompt = f"""
Conversation summary: {summary}
Known patient traits: {facts}
Recent turns:\n{history}

User question: {state.get('question')}
""".strip()

    system = (
        'You are an empathetic medical education assistant. '
        'Never diagnose. If asked for diagnosis, perform an educational pivot and suggest professional care.'
    )
    answer = invoke_llm(prompt, system=system)
    if answer:
        state['generation'] = answer
        state['source'] = 'LLM Medical Reasoning'
    return state
