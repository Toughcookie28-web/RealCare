import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.working_memory_service import WorkingMemoryService
import tools.redis_client as redis_client


class _FakeChatRepo:
    def __init__(self):
        self._history = [
            {"role": "user", "content": "What is metformin?", "timestamp": "2026-03-17T09:00:00Z"},
        ]
        self._summary = "Session goal: understand diabetes treatment"
        self._facts = [{"key": "condition", "value": "diabetes", "confidence": 0.8}]

    def get_history(self, session_id):
        return list(self._history)

    def get_summary(self, session_id):
        return self._summary

    def list_facts(self, session_id):
        return list(self._facts)


def test_load_session_memory_uses_redis_when_present():
    redis_client._local_json.clear()
    redis_client._redis_client = None
    redis_client._redis_initialised = True
    service = WorkingMemoryService(recent_turns_ttl=10, core_state_ttl=20, recent_turns_max=3)
    chat_repo = _FakeChatRepo()

    service.set_recent_turns("session-1", [{"role": "user", "content": "redis turn"}])
    service.set_core_state(
        "session-1",
        {
            "version": "v1",
            "updated_at": "2026-03-17T10:00:00Z",
            "medical_facts": {"age": None, "allergies": ["penicillin"], "conditions": []},
            "session_intent": "compare antibiotics",
            "session_goal": "",
            "medical_context": {"conditions": [], "drugs": [], "procedures": [], "populations": []},
            "key_findings": [],
            "open_questions": [],
        },
    )

    bundle = service.load_session_memory("session-1", chat_repo)

    assert bundle["history"] == [
        {"role": "user", "content": "redis turn", "timestamp": "", "trace_id": "", "route": ""}
    ]
    assert bundle["core_state"]["session_intent"] == "compare antibiotics"
    assert "Persisted session intent: compare antibiotics" in bundle["summary"]
    assert "Session goal: understand diabetes treatment" in bundle["summary"]
    assert {"key": "condition", "value": "diabetes", "confidence": 0.8} in bundle["facts"]
    assert {"key": "allergy", "value": "penicillin", "confidence": 1.0} in bundle["facts"]


def test_load_session_memory_falls_back_to_chat_repo_when_redis_is_empty():
    redis_client._local_json.clear()
    redis_client._redis_client = None
    redis_client._redis_initialised = True
    service = WorkingMemoryService(recent_turns_ttl=10, core_state_ttl=20, recent_turns_max=3)
    chat_repo = _FakeChatRepo()

    bundle = service.load_session_memory("session-2", chat_repo)

    assert bundle["history"] == chat_repo.get_history("session-2")
    assert bundle["summary"] == chat_repo.get_summary("session-2")
    assert bundle["facts"] == chat_repo.list_facts("session-2")
    assert bundle["core_state"]["medical_facts"]["conditions"] == ["diabetes"]


def test_append_recent_turn_normalizes_and_trims_history():
    redis_client._local_json.clear()
    redis_client._redis_client = None
    redis_client._redis_initialised = True
    service = WorkingMemoryService(recent_turns_ttl=10, core_state_ttl=20, recent_turns_max=2)

    service.append_recent_turn("session-3", {"role": "user", "content": "turn-1", "ignored": "x"})
    service.append_recent_turn("session-3", {"role": "assistant", "content": "turn-2"})
    service.append_recent_turn("session-3", {"role": "user", "content": "turn-3"})

    assert service.get_recent_turns("session-3") == [
        {"role": "assistant", "content": "turn-2", "timestamp": "", "trace_id": "", "route": ""},
        {"role": "user", "content": "turn-3", "timestamp": "", "trace_id": "", "route": ""},
    ]
