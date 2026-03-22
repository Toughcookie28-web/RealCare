from __future__ import annotations

import time
from typing import Callable

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from observability.metrics import (
    NODE_ERRORS,
    NODE_LATENCY,
    ROUTE_COUNTER,
    INTENT_CONFIDENCE,
    CLARIFICATION_TRIGGERED,
    REFLECTION_RETRY,
    REFLECTION_GROUNDING_SCORE,
)

# Per-node OTel span attribute extractors.
# Each lambda takes the updated state dict returned by the node and returns
# a dict of {attribute_name: value} to attach to the span.
# All string values are truncated to 500 chars to avoid OTel's 128KB attribute limit.
_NODE_ATTRS: dict[str, Callable] = {
    'guardrail': lambda s: {
        'blocked': bool((s.get('safety_flags') or {}).get('blocked', False)),
        'risk_level': str((s.get('safety_flags') or {}).get('risk_level', 'low')),
        'safety_reason': str((s.get('safety_flags') or {}).get('reason') or 'none'),
    },
    'memory': lambda s: {
        'history_len': len(s.get('conversation_history', [])),
        'facts_extracted': len(s.get('facts', [])),
        'summary_generated': bool(s.get('summary')),
        'long_term_enabled': bool(s.get('long_term_memory_repo') is not None),
        'episodic_memories_loaded': len(s.get('episodic_memories', [])),
    },
    'query_rewriter': lambda s: {
        'optimized_query': str(s.get('optimized_query', ''))[:500],
        'stepback_query': str(s.get('stepback_query', ''))[:300],
        'route': str(s.get('route', '')),
        'turn_intent': str(s.get('turn_intent', '')),
        'slots': str(s.get('slots', {}))[:300],
        'intent_confidence': float(s.get('intent_confidence', 1.0)),
        'needs_clarification': bool(s.get('needs_clarification', False)),
    },
    'planner': lambda s: {
        'route': str(s.get('route', '')),
        'reason': str(s.get('route_decision_reason', ''))[:500],
    },
    'retriever': lambda s: {
        'num_docs': len(s.get('documents', [])),
        'mmr_enabled': True,
        'sections': ','.join(
            list({
                d.metadata.get('section', '')
                for d in s.get('documents', [])
                if hasattr(d, 'metadata')
            })[:5]
        )[:500],
    },
    'reflection': lambda s: {
        'grounding_score': float(s.get('grounding_score', 0.0)),
        'failure_category': str(s.get('reflection_failure_category', 'none')),
        'needs_retry': bool(s.get('needs_retry', False)),
    },
    'executor': lambda s: {
        'source': str(s.get('source', ''))[:200],
        'generation_length': len(s.get('generation', '')),
        'citations_count': len(s.get('citations', [])),
    },
    'explanation': lambda s: {
        'citations_count': len(s.get('citations', [])),
        'docs_available': len(s.get('documents', [])),
        'generation_length': len(s.get('generation', '')),
    },
}


def _emit_node_metrics(node_name: str, state) -> None:
    """Emit Prometheus metrics for a completed node based on its output state.

    Centralized here (rather than in each agent file) so agent modules stay
    free of metrics imports and all metric emission is in one place.
    """
    if node_name == 'planner':
        route = state.get('route') or 'unknown'
        ROUTE_COUNTER.labels(route=route).inc()
    elif node_name == 'query_rewriter':
        confidence = state.get('intent_confidence')
        if isinstance(confidence, (int, float)):
            INTENT_CONFIDENCE.observe(float(confidence))
        if state.get('needs_clarification'):
            CLARIFICATION_TRIGGERED.inc()
    elif node_name == 'reflection':
        if state.get('needs_retry'):
            REFLECTION_RETRY.inc()
        grounding = state.get('grounding_score')
        if isinstance(grounding, (int, float)):
            REFLECTION_GROUNDING_SCORE.observe(float(grounding))


def run_node(node_name: str, fn: Callable, state):
    # Lazy tracer lookup: called after configure_tracing() runs in app lifespan.
    # OTel caches tracers by name — this is an O(1) dict lookup, not object creation.
    tracer = trace.get_tracer('medigenius.pipeline')

    # Re-attach the pipeline.request span context captured by WorkflowService.
    # LangGraph may run node functions in a different thread/async context, which
    # breaks Python's contextvars-based OTel propagation.  Explicitly re-attaching
    # ensures each node span is a child of the root pipeline.request span.
    parent_ctx = state.get('_otel_context')
    token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
    try:
        with tracer.start_as_current_span(node_name) as span:
            span.set_attribute('trace_id', str(state.get('trace_id', '')))
            span.set_attribute('session_id', str(state.get('session_id', '')))
            # gen_ai.* semantic convention attributes for LLM observability
            span.set_attribute('gen_ai.system', 'medigenius')
            span.set_attribute('gen_ai.operation.name', node_name)

            start = time.perf_counter()
            try:
                updated = fn(state)
                # Attach per-node output attributes (truncated to avoid OTel limits)
                for k, v in (_NODE_ATTRS.get(node_name, lambda s: {})(updated)).items():
                    span.set_attribute(k, v)
                span.set_status(StatusCode.OK)
                _emit_node_metrics(node_name, updated)
                updated.setdefault('status_events', []).append({'node': node_name, 'status': 'ok'})
                return updated
            except Exception as exc:  # pragma: no cover - runtime robustness
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                state.setdefault('status_events', []).append({'node': node_name, 'status': 'error', 'error': str(exc)})
                state['source'] = 'System Message'
                state['generation'] = 'An internal error occurred while processing your request.'
                NODE_ERRORS.labels(node=node_name, error_code='internal_error').inc()
                return state  # DO NOT reraise — preserves graceful degradation behavior
            finally:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                # Attach exemplar linking this latency observation to the current OTel trace.
                # Requires OpenMetrics scrape format in prometheus.yml (already configured).
                try:
                    span_ctx = span.get_span_context()
                    trace_id_hex = format(span_ctx.trace_id, '032x') if span_ctx and span_ctx.trace_id else ''
                    NODE_LATENCY.labels(node=node_name).observe(
                        elapsed_ms,
                        {'TraceID': trace_id_hex} if trace_id_hex else None,
                    )
                except Exception:
                    NODE_LATENCY.labels(node=node_name).observe(elapsed_ms)
    finally:
        if token is not None:
            otel_context.detach(token)
