from unittest.mock import MagicMock, patch


def test_write_long_term_memory_persists_facts(monkeypatch):
    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    mock_lt_repo = MagicMock()

    from api.chat_runtime import write_long_term_memory
    write_long_term_memory(
        user_id="user-abc",
        session_id="sess-1",
        question="I am 45 years old and have diabetes",
        result={
            "facts": [
                {"key": "age", "value": "45", "confidence": 0.9},
                {"key": "condition", "value": "diabetes", "confidence": 0.8},
            ],
            "route": "vector",
            "generation": "Here is information about diabetes management...",
        },
        lt_repo=mock_lt_repo,
    )

    assert mock_lt_repo.upsert_long_term_fact.call_count == 2


def test_write_long_term_memory_skips_chitchat(monkeypatch):
    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    mock_lt_repo = MagicMock()

    from api.chat_runtime import write_long_term_memory
    write_long_term_memory(
        user_id="user-abc",
        session_id="sess-1",
        question="hello!",
        result={
            "facts": [],
            "route": "chitchat",
            "generation": "Hello! How can I help you?",
        },
        lt_repo=mock_lt_repo,
    )

    mock_lt_repo.upsert_long_term_fact.assert_not_called()
    mock_lt_repo.add_conversation_memory.assert_not_called()


def test_write_long_term_memory_skips_when_no_user_id(monkeypatch):
    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    mock_lt_repo = MagicMock()

    from api.chat_runtime import write_long_term_memory
    write_long_term_memory(
        user_id=None,
        session_id="sess-1",
        question="what is aspirin?",
        result={"facts": [{"key": "age", "value": "30", "confidence": 0.9}], "route": "vector", "generation": "..."},
        lt_repo=mock_lt_repo,
    )

    mock_lt_repo.upsert_long_term_fact.assert_not_called()
