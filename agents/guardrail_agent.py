from __future__ import annotations

from core.state_v2 import AgentStateV2


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
    state['safety_flags'] = {'blocked': False, 'reason': None, 'risk_level': risk_level}
    return state
