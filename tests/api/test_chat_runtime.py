import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.chat_runtime import load_memory_bundle, post_response_updates


class _FakeWorkingMemoryService:
    def __init__(self):
        self.loaded = []
        self.appended = []
        self.saved_core_state = None

    def load_session_memory(self, session_id, chat_repo):
        self.loaded.append((session_id, chat_repo))
        return {
            "history": [{"role": "user", "content": "redis turn"}],
            "summary": "Persisted session intent: compare antibiotics",
            "facts": [{"key": "allergy", "value": "penicillin", "confidence": 1.0}],
            "core_state": {
                "version": "v1",
                "updated_at": "2026-03-17T10:00:00Z",
                "medical_facts": {"age": None, "allergies": ["penicillin"], "conditions": []},
                "session_intent": "compare antibiotics",
                "session_goal": "",
                "medical_context": {"conditions": [], "drugs": [], "procedures": [], "populations": []},
                "key_findings": [],
                "open_questions": [],
            },
        }

    def append_recent_turn(self, session_id, turn):
        self.appended.append((session_id, turn))

    def set_core_state(self, session_id, core_state):
        self.saved_core_state = (session_id, core_state)


class _FakeLiveJudgeService:
    def __init__(self):
        self.payloads = []

    def judge(self, payload):
        self.payloads.append(payload)
        return {"answer_relevance": 0.8}


class _FakeDocument:
    def __init__(self, page_content):
        self.page_content = page_content


def test_load_memory_bundle_delegates_to_working_memory_service():
    working_memory_service = _FakeWorkingMemoryService()
    chat_repo = object()

    bundle = load_memory_bundle("session-1", chat_repo, working_memory_service=working_memory_service)

    assert bundle["history"] == [{"role": "user", "content": "redis turn"}]
    assert working_memory_service.loaded == [("session-1", chat_repo)]


def test_post_response_updates_appends_turns_promotes_core_state_and_judges():
    working_memory_service = _FakeWorkingMemoryService()
    live_judge_service = _FakeLiveJudgeService()

    post_response_updates(
        session_id="session-2",
        trace_id="trace-2",
        question="Which antibiotic is safer if I am allergic to penicillin?",
        result={
            "generation": "Macrolides may be considered depending on the case.",
            "route": "vector",
            "facts": [{"key": "condition", "value": "pneumonia", "confidence": 0.8}],
            "summary": "Session goal: choose a safe antibiotic",
            "session_intent": "treatment_comparison",
            "documents": [_FakeDocument("Macrolides are an alternative in some cases.")],
        },
        current_core_state=working_memory_service.load_session_memory("session-2", object())["core_state"],
        working_memory_service=working_memory_service,
        live_judge_service=live_judge_service,
    )

    assert working_memory_service.appended == [
        ("session-2", {"role": "user", "content": "Which antibiotic is safer if I am allergic to penicillin?", "trace_id": "trace-2", "route": "vector"}),
        ("session-2", {"role": "assistant", "content": "Macrolides may be considered depending on the case.", "trace_id": "trace-2", "route": "vector"}),
    ]
    saved_session_id, saved_core_state = working_memory_service.saved_core_state
    assert saved_session_id == "session-2"
    assert saved_core_state["medical_facts"]["allergies"] == ["penicillin"]
    assert saved_core_state["medical_facts"]["conditions"] == ["pneumonia"]
    assert saved_core_state["session_intent"] == "treatment_comparison"
    assert live_judge_service.payloads[0]["trace_id"] == "trace-2"
    assert live_judge_service.payloads[0]["contexts"] == ["Macrolides are an alternative in some cases."]
