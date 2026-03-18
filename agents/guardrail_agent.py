from __future__ import annotations

from core.state_v2 import AgentStateV2
from core.settings import get_settings
from core.hitl import create_pending_review


BLOCK_PATTERNS = [
    'synthesize meth',
    'make bomb',
    'illicit drug synthesis',
    'how to kill',
    'suicide method',
    'fatal dose',
]

HIGH_RISK_PATTERNS = [
    'dosage adjustment',
    'insulin dose',
    'warfarin dose',
    'stop medication',
]


def GuardrailAgent(state: AgentStateV2) -> AgentStateV2:
    settings = get_settings()
    text = state.get('question', '').lower()
    if any(pattern in text for pattern in BLOCK_PATTERNS):
        state['safety_flags'] = {'blocked': True, 'reason': 'unsafe_content', 'risk_level': 'high'}
        state['generation'] = (
            'I cannot help with harmful or illegal medical requests. '
            'If this is an emergency or safety concern, please contact local emergency services.'
        )
        state['source'] = 'Safety Guardrail'
        return state

    risk_level = 'high' if any(pattern in text for pattern in HIGH_RISK_PATTERNS) else 'low'
    if risk_level == 'high' and settings.enable_hitl_interrupts:
        approval_id = state.get('trace_id', 'pending-review')
        chat_repo = state.get('chat_repo')
        db = getattr(chat_repo, 'db', None)
        create_pending_review(
            approval_id=approval_id,
            session_id=state.get('session_id', 'unknown'),
            trace_id=state.get('trace_id', 'unknown'),
            question=state.get('question', ''),
            db=db,
        )
        state['safety_flags'] = {
            'blocked': True,
            'reason': 'hitl_review_required',
            'risk_level': 'high',
            'approval_id': approval_id,
        }
        state['generation'] = (
            'This request is marked high risk and requires human clinical review before response. '
            f'Please submit for clinician/admin approval. Approval ID: {approval_id}'
        )
        state['source'] = 'HITL Guardrail'
        return state
    state['safety_flags'] = {'blocked': False, 'reason': None, 'risk_level': risk_level}
    return state
