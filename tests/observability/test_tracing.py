from __future__ import annotations

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- _NODE_ATTRS extractor tests ---

def test_node_attrs_query_rewriter():
    from agents.common import _NODE_ATTRS
    state = {
        'optimized_query': 'what is metformin',
        'route': 'vector',
        'intent_confidence': 0.85,
        'needs_clarification': False,
    }
    attrs = _NODE_ATTRS['query_rewriter'](state)
    assert attrs['optimized_query'] == 'what is metformin'
    assert attrs['route'] == 'vector'
    assert attrs['intent_confidence'] == 0.85
    assert attrs['needs_clarification'] is False


def test_node_attrs_query_rewriter_truncates_long_query():
    from agents.common import _NODE_ATTRS
    long_query = 'x' * 1000
    state = {'optimized_query': long_query, 'route': 'vector', 'intent_confidence': 0.9, 'needs_clarification': False}
    attrs = _NODE_ATTRS['query_rewriter'](state)
    assert len(attrs['optimized_query']) == 500


def test_node_attrs_reflection():
    from agents.common import _NODE_ATTRS
    state = {
        'grounding_score': 0.72,
        'reflection_failure_category': 'hallucination',
        'needs_retry': True,
    }
    attrs = _NODE_ATTRS['reflection'](state)
    assert attrs['grounding_score'] == 0.72
    assert attrs['failure_category'] == 'hallucination'
    assert attrs['needs_retry'] is True


# --- _emit_node_metrics tests ---

def _get_counter_value(counter):
    """Read current value from a prometheus_client Counter (unlabeled)."""
    try:
        return counter._value.get()
    except Exception:
        return [s.value for s in counter.collect()[0].samples if s.name.endswith('_total')][0]


def _get_labeled_counter_value(counter_child):
    """Read current value from a labeled prometheus_client Counter child."""
    try:
        return counter_child._value.get()
    except Exception:
        return counter_child._value.get()


def _get_histogram_sum(histogram):
    """Read current _sum from a prometheus_client Histogram."""
    try:
        return histogram._sum.get()
    except Exception:
        metric = histogram.collect()[0]
        return [s.value for s in metric.samples if s.name.endswith('_sum')][0]


def test_emit_node_metrics_route_counter():
    from agents.common import _emit_node_metrics
    import observability.metrics as m
    before = _get_labeled_counter_value(m.ROUTE_COUNTER.labels(route='vector'))
    _emit_node_metrics('planner', {'route': 'vector'})
    after = _get_labeled_counter_value(m.ROUTE_COUNTER.labels(route='vector'))
    assert after == before + 1


def test_emit_node_metrics_intent_confidence():
    """INTENT_CONFIDENCE histogram receives a sample when node is query_rewriter."""
    from agents.common import _emit_node_metrics
    import observability.metrics as m
    before = _get_histogram_sum(m.INTENT_CONFIDENCE)
    _emit_node_metrics('query_rewriter', {'intent_confidence': 0.3, 'needs_clarification': False})
    after = _get_histogram_sum(m.INTENT_CONFIDENCE)
    assert after == pytest.approx(before + 0.3, abs=1e-6)


def test_emit_node_metrics_clarification_triggered():
    from agents.common import _emit_node_metrics
    import observability.metrics as m
    before = _get_counter_value(m.CLARIFICATION_TRIGGERED)
    _emit_node_metrics('query_rewriter', {'intent_confidence': 0.2, 'needs_clarification': True})
    after = _get_counter_value(m.CLARIFICATION_TRIGGERED)
    assert after == before + 1


def test_emit_node_metrics_reflection_grounding():
    from agents.common import _emit_node_metrics
    import observability.metrics as m
    before = _get_histogram_sum(m.REFLECTION_GROUNDING_SCORE)
    _emit_node_metrics('reflection', {'needs_retry': False, 'grounding_score': 0.65})
    after = _get_histogram_sum(m.REFLECTION_GROUNDING_SCORE)
    assert after == pytest.approx(before + 0.65, abs=1e-6)
