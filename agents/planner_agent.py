from __future__ import annotations

import logging

from core.state_v2 import AgentStateV2
from tools.knn_router import knn_fallback_route

logger = logging.getLogger(__name__)

VALID_ROUTES = {'chitchat', 'vector', 'web', 'future_clinical_db', 'literature', 'memory'}


def PlannerAgent(state: AgentStateV2) -> AgentStateV2:
    route = (state.get('route') or '').strip().lower()

    if route not in VALID_ROUTES:
        route = knn_fallback_route(state)
        state['route_decision_reason'] = 'knn_fallback'
        logger.info("planner_knn_fallback", extra={"resolved_route": route})

    state['route'] = route
    state['planned_route'] = route
    state['current_tool'] = route
    return state
