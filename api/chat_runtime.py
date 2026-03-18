from __future__ import annotations

import logging
from typing import Any

from observability.live_judge import LiveJudgeService, build_live_judge_payload
from tools.working_memory import merge_core_state
from tools.working_memory_service import WorkingMemoryService


logger = logging.getLogger(__name__)

_working_memory_service: WorkingMemoryService | None = None
_live_judge_service: LiveJudgeService | None = None


def get_working_memory_service() -> WorkingMemoryService:
    global _working_memory_service
    if _working_memory_service is None:
        _working_memory_service = WorkingMemoryService()
    return _working_memory_service


def get_live_judge_service() -> LiveJudgeService:
    global _live_judge_service
    if _live_judge_service is None:
        _live_judge_service = LiveJudgeService()
    return _live_judge_service


def load_memory_bundle(
    session_id: str,
    chat_repo: Any,
    *,
    working_memory_service: WorkingMemoryService | None = None,
) -> dict[str, Any]:
    service = working_memory_service or get_working_memory_service()
    return service.load_session_memory(session_id, chat_repo)


def write_long_term_memory(
    *,
    user_id: str | None,
    session_id: str,
    question: str,
    result: dict,
    lt_repo: Any,
) -> None:
    """
    Persist extracted facts and episodic memory for the user after each turn.
    Runs inside BackgroundTask — failures are logged, never raised.
    Skips entirely for anonymous users (user_id=None) and chitchat turns.
    """
    from core.settings import get_settings
    settings = get_settings()

    if not settings.long_term_memory_enabled:
        return
    if not user_id:
        return

    route = str(result.get("route", "") or "")
    if route in ("chitchat", "clarify", "memory"):
        return  # no medical content worth storing

    facts = result.get("facts", []) or []
    generation = str(result.get("generation", "") or "")

    # 1. Upsert extracted facts to long_term_facts
    for fact in facts:
        key = fact.get("key", "").strip()
        value = fact.get("value", "").strip()
        confidence = float(fact.get("confidence", 0.7))
        if key and value and confidence >= settings.long_term_memory_min_confidence:
            try:
                lt_repo.upsert_long_term_fact(
                    user_id, key, value,
                    confidence=confidence,
                    source_session=session_id,
                )
            except Exception:
                logger.warning(
                    "lt_fact_write_failed",
                    extra={"user_id": user_id, "key": key},
                    exc_info=True,
                )

    # 2. Store episodic memory if turn has meaningful medical content
    has_enough_facts = len(facts) >= settings.long_term_memory_episode_min_facts
    if not has_enough_facts or not generation:
        return

    try:
        from tools.embedding_client import embed_query

        facts_str = ', '.join(f'{f["key"]}={f["value"]}' for f in facts)
        turn_summary = (
            f"User asked: {question[:200]}\n"
            f"Key facts mentioned: {facts_str}\n"
            f"Answer summary: {generation[:300]}"
        )
        embedding = embed_query(turn_summary)

        medical_entities = {
            "drugs": [f["value"] for f in facts if f.get("key") == "drug"],
            "conditions": [f["value"] for f in facts if f.get("key") == "condition"],
        }

        lt_repo.add_conversation_memory(
            user_id=user_id,
            session_id=session_id,
            summary=turn_summary,
            embedding=embedding,
            medical_entities=medical_entities,
        )
        logger.info(
            "episodic_memory_stored",
            extra={"user_id": user_id, "session_id": session_id},
        )
    except Exception:
        logger.warning(
            "episodic_memory_write_failed",
            extra={"user_id": user_id, "session_id": session_id},
            exc_info=True,
        )


def post_response_updates(
    *,
    session_id: str,
    user_id: str | None = None,
    trace_id: str,
    question: str,
    result: dict[str, Any],
    current_core_state: dict[str, Any] | None,
    lt_repo: Any | None = None,
    working_memory_service: WorkingMemoryService | None = None,
    live_judge_service: LiveJudgeService | None = None,
) -> None:
    memory_service = working_memory_service or get_working_memory_service()
    judge_service = live_judge_service or get_live_judge_service()
    route = str(result.get("route", "") or "")
    answer = str(result.get("generation", "") or result.get("response", "") or "")

    try:
        memory_service.append_recent_turn(
            session_id,
            {
                "role": "user",
                "content": question,
                "trace_id": trace_id,
                "route": route,
            },
        )
        if answer:
            memory_service.append_recent_turn(
                session_id,
                {
                    "role": "assistant",
                    "content": answer,
                    "trace_id": trace_id,
                    "route": route,
                },
            )

        merged_core_state = merge_core_state(
            current_core_state,
            facts=result.get("facts", []) or [],
            session_intent=str(result.get("session_intent", "") or ""),
            summary_text=str(result.get("summary", "") or ""),
        )
        memory_service.set_core_state(session_id, merged_core_state)
    except Exception:
        logger.exception("working_memory_post_response_failed", extra={"trace_id": trace_id, "session_id": session_id})

    try:
        judge_service.judge(build_live_judge_payload(question=question, trace_id=trace_id, result=result))
    except Exception:
        logger.exception("live_judge_post_response_failed", extra={"trace_id": trace_id, "session_id": session_id})

    if lt_repo is not None:
        write_long_term_memory(
            user_id=user_id,
            session_id=session_id,
            question=question,
            result=result,
            lt_repo=lt_repo,
        )
