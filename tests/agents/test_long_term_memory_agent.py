def test_memory_agent_loads_user_facts_when_user_id_present(monkeypatch):
    from agents.memory_agent import MemoryAgent
    from unittest.mock import MagicMock

    mock_repo = MagicMock()
    mock_repo.get_user_facts.return_value = [
        {"key": "age", "value": "45", "confidence": 0.9},
        {"key": "condition", "value": "diabetes", "confidence": 0.85},
    ]
    mock_repo.search_similar_memories.return_value = []

    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    # find the initialize_state import — try core.state first, then core.state_v2
    try:
        from core.state import initialize_state
    except ImportError:
        from core.state_v2 import initialize_state  # type: ignore

    state = initialize_state("s1", "t1")
    state["question"] = "what is metformin?"
    state["user_id"] = "user-abc"
    state["long_term_memory_repo"] = mock_repo

    result = MemoryAgent(state)

    mock_repo.get_user_facts.assert_called_once_with("user-abc")
    fact_keys = [f["key"] for f in result["facts"]]
    assert "age" in fact_keys
    assert "condition" in fact_keys


def test_memory_agent_skips_long_term_when_no_user_id(monkeypatch):
    from agents.memory_agent import MemoryAgent

    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    try:
        from core.state import initialize_state
    except ImportError:
        from core.state_v2 import initialize_state  # type: ignore

    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin?"
    state["user_id"] = None
    state["long_term_memory_repo"] = None

    result = MemoryAgent(state)  # should not raise
    assert result["episodic_memories"] == []


def test_memory_agent_loads_episodic_memories(monkeypatch):
    from agents.memory_agent import MemoryAgent
    from unittest.mock import MagicMock

    mock_repo = MagicMock()
    mock_repo.get_user_facts.return_value = []
    mock_repo.search_similar_memories.return_value = [
        {
            "summary": "User asked about metformin dosage. Prefers morning dosing.",
            "similarity": 0.88,
            "created_at": "2026-03-01T10:00:00",
            "medical_entities": {"drugs": ["metformin"]},
        }
    ]

    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    monkeypatch.setattr(
        "agents.memory_agent.embed_query",
        lambda q: [0.1] * 768,
    )

    try:
        from core.state import initialize_state
    except ImportError:
        from core.state_v2 import initialize_state  # type: ignore

    state = initialize_state("s1", "t1")
    state["question"] = "what is the best time to take metformin?"
    state["user_id"] = "user-abc"
    state["long_term_memory_repo"] = mock_repo

    result = MemoryAgent(state)

    assert len(result["episodic_memories"]) == 1
    assert "metformin" in result["episodic_memories"][0]["summary"]


def test_format_episodic_memories_block():
    from agents.executor_agent import _format_episodic_memories

    episodes = [
        {
            "summary": "User asked about metformin dosage. Takes 500mg twice daily.",
            "created_at": "2026-03-01T10:00:00",
            "similarity": 0.91,
        }
    ]
    result = _format_episodic_memories(episodes)
    assert "metformin" in result
    assert "Relevant past context" in result


def test_format_episodic_memories_empty():
    from agents.executor_agent import _format_episodic_memories
    assert _format_episodic_memories([]) == ""
