from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Generator

from core.langgraph_workflow import create_workflow
from core.state_v2 import initialize_state, reset_query_state


class WorkflowService:
    def __init__(self):
        self.workflow = create_workflow()

    def run(
        self,
        question: str,
        session_id: str,
        trace_id: str,
        history: list[dict[str, Any]],
        summary: str,
        facts: list[dict[str, Any]],
        chat_repo=None,
        vector_repo=None,
        user_id: str | None = None,
        long_term_memory_repo=None,
    ) -> dict[str, Any]:
        state = initialize_state(session_id=session_id, trace_id=trace_id)
        state = reset_query_state(state, question)
        state['conversation_history'] = history
        state['summary'] = summary
        state['facts'] = facts
        state['chat_repo'] = chat_repo
        state['vector_repo'] = vector_repo
        state['user_id'] = user_id
        state['long_term_memory_repo'] = long_term_memory_repo

        result = self.workflow.invoke(state)
        return result

    def stream(
        self,
        question: str,
        session_id: str,
        trace_id: str,
        history: list[dict[str, Any]],
        summary: str,
        facts: list[dict[str, Any]],
        chat_repo=None,
        vector_repo=None,
        user_id: str | None = None,
        long_term_memory_repo=None,
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        state = initialize_state(session_id=session_id, trace_id=trace_id)
        state = reset_query_state(state, question)
        state['conversation_history'] = history
        state['summary'] = summary
        state['facts'] = facts
        state['chat_repo'] = chat_repo
        state['vector_repo'] = vector_repo
        state['user_id'] = user_id
        state['long_term_memory_repo'] = long_term_memory_repo

        final_result = None
        emitted = 0
        try:
            for current_state in self.workflow.stream(state, stream_mode='values'):
                if not isinstance(current_state, dict):
                    continue
                status_events = current_state.get('status_events', [])
                while emitted < len(status_events):
                    node_event = status_events[emitted]
                    emitted += 1
                    yield {
                        'event': 'status',
                        'trace_id': trace_id,
                        'data': {'node': node_event.get('node'), 'status': node_event.get('status')},
                    }
                final_result = current_state
        except Exception as exc:  # pragma: no cover - runtime safety path
            logging.getLogger(__name__).exception('Workflow stream error', extra={'trace_id': trace_id})
            yield {
                'event': 'error',
                'trace_id': trace_id,
                'data': {'message': str(exc)},
            }
            return {}

        if not isinstance(final_result, dict):
            final_result = self.workflow.invoke(state)

        yield {
            'event': 'final',
            'trace_id': trace_id,
            'data': {
                'response': final_result.get('generation', ''),
                'source': final_result.get('source', 'Unknown'),
                'timestamp': datetime.now().strftime('%I:%M %p'),
                'success': bool(final_result.get('generation')),
                'route': final_result.get('route'),
                'citations': final_result.get('citations', []),
                'summary': final_result.get('summary'),
                'facts': final_result.get('facts', []),
            },
            '_internal': {
                'generation': final_result.get('generation', ''),
                'documents': final_result.get('documents', []),
                'session_intent': final_result.get('session_intent', ''),
                'route': final_result.get('route'),
                'summary': final_result.get('summary'),
                'facts': final_result.get('facts', []),
            },
        }
        return final_result


_workflow_service = WorkflowService()


def get_workflow_service() -> WorkflowService:
    return _workflow_service


def generate_trace_id() -> str:
    return str(uuid.uuid4())
