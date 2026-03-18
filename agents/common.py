from __future__ import annotations

import time
from typing import Callable

from observability.metrics import NODE_ERRORS, NODE_LATENCY


def run_node(node_name: str, fn: Callable, state):
    start = time.perf_counter()
    try:
        updated = fn(state)
        updated.setdefault('status_events', []).append({'node': node_name, 'status': 'ok'})
        return updated
    except Exception as exc:  # pragma: no cover - runtime robustness
        state.setdefault('status_events', []).append({'node': node_name, 'status': 'error', 'error': str(exc)})
        state['source'] = 'System Message'
        state['generation'] = 'An internal error occurred while processing your request.'
        NODE_ERRORS.labels(node=node_name, error_code='internal_error').inc()
        return state
    finally:
        NODE_LATENCY.labels(node=node_name).observe(int((time.perf_counter() - start) * 1000))
