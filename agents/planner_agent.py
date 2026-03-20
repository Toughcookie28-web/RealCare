from __future__ import annotations

import logging

from core.settings import get_settings
from core.state_v2 import AgentStateV2
from tools.knn_router import knn_fallback_route

logger = logging.getLogger(__name__)

VALID_ROUTES = {'chitchat', 'vector', 'web', 'future_clinical_db', 'literature', 'memory'}


def PlannerAgent(state: AgentStateV2) -> AgentStateV2:
    settings = get_settings()
    if (
        settings.hitl_clarification_enabled
        and state.get('needs_clarification')
        and state.get('intent_confidence', 1.0) < settings.hitl_clarification_confidence_threshold
    ):
        state['route'] = 'clarify'
        state['planned_route'] = 'clarify'
        state['route_decision_reason'] = 'planner_low_confidence_clarify'
        state['current_tool'] = 'clarify'
        logger.info("planner_clarify_route", extra={"intent_confidence": state.get('intent_confidence')})
        return state

    route = (state.get('route') or '').strip().lower()

    if route not in VALID_ROUTES:
        route = knn_fallback_route(state)
        state['route_decision_reason'] = 'knn_fallback'
        logger.info("planner_knn_fallback", extra={"resolved_route": route})

    state['route'] = route
    state['planned_route'] = route
    state['current_tool'] = route
    return state
