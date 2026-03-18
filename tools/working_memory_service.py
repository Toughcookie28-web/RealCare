from __future__ import annotations

from typing import Any

from core.settings import get_settings
from tools.redis_client import append_json_list, get_json, set_json
from tools.working_memory import build_empty_core_state, merge_core_state, normalize_recent_turn, trim_recent_turns


class WorkingMemoryService:
    def __init__(
        self,
        *,
        recent_turns_ttl: int | None = None,
        core_state_ttl: int | None = None,
        recent_turns_max: int | None = None,
    ):
        settings = None
        if recent_turns_ttl is None or core_state_ttl is None or recent_turns_max is None:
            settings = get_settings()
        self.recent_turns_ttl = (
            recent_turns_ttl if recent_turns_ttl is not None else settings.working_memory_recent_turns_ttl
        )
        self.core_state_ttl = (
            core_state_ttl if core_state_ttl is not None else settings.working_memory_core_state_ttl
        )
        self.recent_turns_max = (
            recent_turns_max if recent_turns_max is not None else settings.working_memory_recent_turns_max
        )

    @staticmethod
    def _recent_turns_key(session_id: str) -> str:
        return f"recent_turns:{session_id}"

    @staticmethod
    def _core_state_key(session_id: str) -> str:
        return f"core_state:{session_id}"

    def get_recent_turns(self, session_id: str) -> list[dict[str, Any]]:
        return list(get_json(self._recent_turns_key(session_id), default=[]))

    def set_recent_turns(self, session_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = [normalize_recent_turn(turn) for turn in turns]
        trimmed = trim_recent_turns(normalized, self.recent_turns_max)
        set_json(self._recent_turns_key(session_id), trimmed, ttl=self.recent_turns_ttl)
        return trimmed

    def append_recent_turn(self, session_id: str, turn: dict[str, Any]) -> list[dict[str, Any]]:
        return append_json_list(
            self._recent_turns_key(session_id),
            normalize_recent_turn(turn),
            max_length=self.recent_turns_max,
            ttl=self.recent_turns_ttl,
        )

    def get_core_state(self, session_id: str) -> dict[str, Any] | None:
        value = get_json(self._core_state_key(session_id), default=None)
        if value is None:
            return None
        return dict(value)

    def set_core_state(self, session_id: str, core_state: dict[str, Any]) -> None:
        set_json(self._core_state_key(session_id), core_state, ttl=self.core_state_ttl)

    @staticmethod
    def _core_state_facts(core_state: dict[str, Any]) -> list[dict[str, Any]]:
        medical_facts = core_state.get("medical_facts", {})
        facts: list[dict[str, Any]] = []
        age = medical_facts.get("age")
        if age:
            facts.append({"key": "age", "value": str(age), "confidence": 1.0})
        for allergy in medical_facts.get("allergies", []):
            facts.append({"key": "allergy", "value": str(allergy), "confidence": 1.0})
        for condition in medical_facts.get("conditions", []):
            facts.append({"key": "condition", "value": str(condition), "confidence": 1.0})
        return facts

    @staticmethod
    def _merge_facts(existing: list[dict[str, Any]], overlay: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for fact in [*existing, *overlay]:
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            if not key or not value:
                continue
            dedupe_key = (key, value)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append({"key": key, "value": value, "confidence": float(fact.get("confidence", 1.0))})
        return merged

    @staticmethod
    def _core_state_summary(core_state: dict[str, Any]) -> str:
        parts: list[str] = []
        session_intent = str(core_state.get("session_intent", "")).strip()
        if session_intent:
            parts.append(f"Persisted session intent: {session_intent}")

        session_goal = str(core_state.get("session_goal", "")).strip()
        if session_goal:
            parts.append(f"Session goal: {session_goal}")

        medical_context = core_state.get("medical_context", {})
        context_parts: list[str] = []
        for key in ("conditions", "drugs", "procedures", "populations"):
            values = [str(item).strip() for item in medical_context.get(key, []) if str(item).strip()]
            if values:
                context_parts.append(f"{key}: {', '.join(values)}")
        if context_parts:
            parts.append(f"Medical context: {'; '.join(context_parts)}")

        key_findings = [str(item).strip() for item in core_state.get("key_findings", []) if str(item).strip()]
        if key_findings:
            parts.append(f"Key findings: {'; '.join(key_findings)}")

        open_questions = [str(item).strip() for item in core_state.get("open_questions", []) if str(item).strip()]
        if open_questions:
            parts.append(f"Open questions: {'; '.join(open_questions)}")

        return "\n".join(parts)

    def load_session_memory(self, session_id: str, chat_repo: Any) -> dict[str, Any]:
        history = self.get_recent_turns(session_id)
        if not history:
            history = chat_repo.get_history(session_id)

        db_summary = chat_repo.get_summary(session_id)
        db_facts = chat_repo.list_facts(session_id)
        core_state = self.get_core_state(session_id)
        if core_state is None:
            core_state = merge_core_state(
                build_empty_core_state(),
                facts=db_facts,
                session_intent="",
                summary_text=db_summary,
            )

        core_summary = self._core_state_summary(core_state)
        summary_parts: list[str] = []
        for part in (core_summary, db_summary):
            if part and part not in summary_parts:
                summary_parts.append(part)
        summary = "\n".join(summary_parts)
        facts = self._merge_facts(db_facts, self._core_state_facts(core_state))

        return {
            "history": history,
            "summary": summary,
            "facts": facts,
            "core_state": core_state,
        }
