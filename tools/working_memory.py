from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_empty_core_state() -> dict[str, Any]:
    return {
        "version": "v1",
        "updated_at": _now_iso(),
        "medical_facts": {
            "age": None,
            "allergies": [],
            "conditions": [],
        },
        "session_intent": "",
        "session_goal": "",
        "medical_context": {
            "conditions": [],
            "drugs": [],
            "procedures": [],
            "populations": [],
        },
        "key_findings": [],
        "open_questions": [],
    }


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _split_semicolon_items(text: str) -> list[str]:
    return [part.strip() for part in text.split(";") if part.strip()]


def _merge_unique(existing: list[str], new_values: list[str]) -> list[str]:
    merged = list(existing)
    for value in new_values:
        if value not in merged:
            merged.append(value)
    return merged


def _promote_summary_fields(merged: dict[str, Any], summary_text: str) -> None:
    if not summary_text.strip():
        return

    medical_context = merged.setdefault(
        "medical_context",
        {"conditions": [], "drugs": [], "procedures": [], "populations": []},
    )

    for raw_line in summary_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Session goal:"):
            merged["session_goal"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("Medical context:"):
            payload = line.split(":", 1)[1].strip()
            for section in _split_semicolon_items(payload):
                if ":" not in section:
                    continue
                key, values = section.split(":", 1)
                normalized_key = key.strip().lower()
                if normalized_key in medical_context:
                    medical_context[normalized_key] = _merge_unique(
                        list(medical_context.get(normalized_key, [])),
                        _split_csv(values),
                    )
            continue

        if line.startswith("Key findings:"):
            values = _split_semicolon_items(line.split(":", 1)[1].strip())
            merged["key_findings"] = _merge_unique(list(merged.get("key_findings", [])), values)
            continue

        if line.startswith("Open questions:"):
            values = _split_semicolon_items(line.split(":", 1)[1].strip())
            merged["open_questions"] = _merge_unique(list(merged.get("open_questions", [])), values)


def normalize_recent_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": str(turn.get("role", "")).strip(),
        "content": str(turn.get("content", "")).strip(),
        "timestamp": str(turn.get("timestamp", "")).strip(),
        "trace_id": str(turn.get("trace_id", "")).strip(),
        "route": str(turn.get("route", "")).strip(),
    }


def trim_recent_turns(turns: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
    if max_length <= 0:
        return []
    return list(turns[-max_length:])


def merge_core_state(
    existing: dict[str, Any] | None,
    *,
    facts: list[dict[str, Any]],
    session_intent: str,
    summary_text: str,
) -> dict[str, Any]:
    merged = deepcopy(existing) if existing else build_empty_core_state()
    medical_facts = merged.setdefault(
        "medical_facts",
        {"age": None, "allergies": [], "conditions": []},
    )

    allergies = list(medical_facts.get("allergies", []))
    conditions = list(medical_facts.get("conditions", []))

    for fact in facts:
        key = str(fact.get("key", "")).strip().lower()
        value = str(fact.get("value", "")).strip()
        if not value:
            continue
        if key == "age":
            medical_facts["age"] = value
        elif key == "allergy" and value not in allergies:
            allergies.append(value)
        elif key == "condition" and value not in conditions:
            conditions.append(value)

    medical_facts["allergies"] = allergies
    medical_facts["conditions"] = conditions

    if session_intent.strip():
        merged["session_intent"] = session_intent.strip()

    _promote_summary_fields(merged, summary_text)

    merged["version"] = "v1"
    merged["updated_at"] = _now_iso()
    return merged
