import pytest
from unittest.mock import MagicMock


def test_get_or_create_user_returns_existing():
    from db.long_term_memory_repository import LongTermMemoryRepository
    from db.models import UserModel

    mock_session = MagicMock()
    existing_user = UserModel(user_id="user-abc")
    mock_session.get.return_value = existing_user

    repo = LongTermMemoryRepository(session=mock_session)
    user_id = repo.get_or_create_user("user-abc")

    assert user_id == "user-abc"
    mock_session.add.assert_not_called()


def test_get_or_create_user_creates_new():
    from db.long_term_memory_repository import LongTermMemoryRepository

    mock_session = MagicMock()
    mock_session.get.return_value = None

    repo = LongTermMemoryRepository(session=mock_session)
    user_id = repo.get_or_create_user("user-new")

    assert user_id == "user-new"
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


def test_upsert_long_term_fact_inserts_new():
    from db.long_term_memory_repository import LongTermMemoryRepository

    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = None
    mock_session.query.return_value = mock_query

    repo = LongTermMemoryRepository(session=mock_session)
    repo.upsert_long_term_fact("u1", "age", "45", confidence=0.9, source_session="s1")

    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


def test_upsert_long_term_fact_updates_existing_higher_confidence():
    from db.long_term_memory_repository import LongTermMemoryRepository
    from db.models import LongTermFactModel

    mock_session = MagicMock()
    existing = LongTermFactModel(user_id="u1", fact_key="age", fact_value="40", confidence=0.6)
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = existing
    mock_session.query.return_value = mock_query

    repo = LongTermMemoryRepository(session=mock_session)
    repo.upsert_long_term_fact("u1", "age", "45", confidence=0.9)

    assert existing.fact_value == "45"
    assert existing.confidence == 0.9
    mock_session.commit.assert_called_once()


def test_upsert_long_term_fact_ignores_lower_confidence():
    from db.long_term_memory_repository import LongTermMemoryRepository
    from db.models import LongTermFactModel

    mock_session = MagicMock()
    existing = LongTermFactModel(user_id="u1", fact_key="age", fact_value="45", confidence=0.9)
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = existing
    mock_session.query.return_value = mock_query

    repo = LongTermMemoryRepository(session=mock_session)
    repo.upsert_long_term_fact("u1", "age", "40", confidence=0.5)

    # Should NOT update because new confidence is lower
    assert existing.fact_value == "45"
    assert existing.confidence == 0.9


def test_get_user_facts_returns_list():
    from db.long_term_memory_repository import LongTermMemoryRepository
    from db.models import LongTermFactModel

    mock_session = MagicMock()
    facts = [
        LongTermFactModel(user_id="u1", fact_key="age", fact_value="45", confidence=0.9),
        LongTermFactModel(user_id="u1", fact_key="condition", fact_value="diabetes", confidence=0.8),
    ]
    mock_query = MagicMock()
    mock_query.filter.return_value.all.return_value = facts
    mock_session.query.return_value = mock_query

    repo = LongTermMemoryRepository(session=mock_session)
    result = repo.get_user_facts("u1")

    assert len(result) == 2
    assert result[0]["key"] == "age"
    assert result[1]["value"] == "diabetes"


def test_get_user_facts_returns_empty_for_unknown_user():
    from db.long_term_memory_repository import LongTermMemoryRepository

    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.all.return_value = []
    mock_session.query.return_value = mock_query

    repo = LongTermMemoryRepository(session=mock_session)
    result = repo.get_user_facts("unknown-user")
    assert result == []
