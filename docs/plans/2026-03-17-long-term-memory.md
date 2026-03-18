# Long-Term Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give users persistent memory across sessions — structured facts (age, conditions, medications) survive session boundaries, and episodic conversation memories are vector-embedded for semantic recall when relevant.

**Architecture:** Three additions: (1) A persistent `user_token` UUID stored in browser localStorage, sent as `X-User-Token` header, mapped server-side to a `user_id` in a new `users` table. (2) Two new PostgreSQL tables — `user_facts` (user-scoped, replaces session-scoped facts) and `conversation_memories` (vector-embedded summaries of past turns). (3) Extended `MemoryAgent` that loads user-scoped facts + top-K similar past episodes at turn start; extended `post_response_updates` that persists new facts and episodic memories asynchronously after each response. No new background workers — episodic writes happen in the existing `BackgroundTask`.

**Tech Stack:** Python, PostgreSQL + pgvector, Redis, SQLAlchemy, Alembic, FastAPI BackgroundTasks, existing `embed_query` from `tools/embedding_client.py`

---

### Task 1: Add long-term memory settings

**Files:**
- Modify: `core/settings.py`
- Modify: `tests/core/test_settings_contract.py`

**Step 1: Write failing test**

```python
def test_long_term_memory_settings_have_correct_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.long_term_memory_enabled is False
    assert s.long_term_memory_recall_k == 3
    assert s.long_term_memory_min_confidence == 0.6
    assert s.long_term_memory_episode_min_facts == 1
```

**Step 2: Run to verify it fails**

Run: `pytest tests/core/test_settings_contract.py -v -k "long_term" 2>&1`
Expected: FAIL

**Step 3: Add settings**

In `core/settings.py`, after `hitl_clarification_confidence_threshold`, add:
```python
long_term_memory_enabled: bool = Field(default=False, alias='LONG_TERM_MEMORY_ENABLED')
long_term_memory_recall_k: int = Field(default=3, alias='LONG_TERM_MEMORY_RECALL_K')
long_term_memory_min_confidence: float = Field(default=0.6, alias='LONG_TERM_MEMORY_MIN_CONFIDENCE')
long_term_memory_episode_min_facts: int = Field(default=1, alias='LONG_TERM_MEMORY_EPISODE_MIN_FACTS')
```

**Step 4: Verify**

Run: `pytest tests/core/test_settings_contract.py -v -k "long_term" 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings_contract.py
git commit -m "feat: add long-term memory settings (disabled by default)"
```

---

### Task 2: Add new DB models

**Files:**
- Modify: `db/models.py`
- Modify: `db/schema_contract.py`

**Step 1: Add CONVERSATION_MEMORY_VECTOR_DIM to schema_contract.py**

In `db/schema_contract.py`, after existing constants:
```python
CONVERSATION_MEMORY_VECTOR_DIM = 768   # same embedding model as document_chunks
CONVERSATION_MEMORY_VECTOR_SQL_TYPE = f"vector({CONVERSATION_MEMORY_VECTOR_DIM})"
```

**Step 2: Add models to db/models.py**

Add three new model classes. Read `db/models.py` first to understand existing import pattern, then add:

```python
class UserModel(Base):
    __tablename__ = 'users'

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
```

```python
class LongTermFactModel(Base):
    """User-scoped facts that persist across all sessions."""
    __tablename__ = 'long_term_facts'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.user_id'), index=True)
    fact_key: Mapped[str] = mapped_column(String(128), index=True)
    fact_value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(default=0.6)
    source_session: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('ix_long_term_facts_user_key', 'user_id', 'fact_key'),
    )
```

```python
from db.schema_contract import DOCUMENT_CHUNK_VECTOR_DIM, CONVERSATION_MEMORY_VECTOR_DIM

class ConversationMemoryModel(Base):
    """Vector-embedded episodic memory of past conversation turns."""
    __tablename__ = 'conversation_memories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.user_id'), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any | None] = mapped_column(
        Vector(CONVERSATION_MEMORY_VECTOR_DIM), nullable=True
    )
    medical_entities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('ix_conversation_memories_user_id_created', 'user_id', 'created_at'),
    )
```

**Step 3: Verify models import cleanly**

Run: `python -c "from db.models import UserModel, LongTermFactModel, ConversationMemoryModel; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add db/models.py db/schema_contract.py
git commit -m "feat: add UserModel, LongTermFactModel, ConversationMemoryModel"
```

---

### Task 3: Create Alembic migration

**Files:**
- Create: `alembic/versions/0005_add_long_term_memory.py`

**Step 1: Generate migration**

Run: `alembic revision --autogenerate -m "add_long_term_memory"`

This creates a new file in `alembic/versions/`. Open it and verify the `upgrade()` function creates:
- `users` table
- `long_term_facts` table with composite index
- `conversation_memories` table with vector column and index

**Step 2: Review and fix the generated migration**

The autogenerated migration may not correctly handle the pgvector type on `conversation_memories.embedding`. Manually ensure the `upgrade()` uses:

```python
import pgvector.sqlalchemy

def upgrade() -> None:
    op.create_table('users',
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_seen', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('user_id'),
    )
    op.create_table('long_term_facts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('fact_key', sa.String(128), nullable=False),
        sa.Column('fact_value', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('source_session', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_long_term_facts_user_key', 'long_term_facts', ['user_id', 'fact_key'])
    op.create_index(op.f('ix_long_term_facts_user_id'), 'long_term_facts', ['user_id'])
    op.create_index(op.f('ix_long_term_facts_fact_key'), 'long_term_facts', ['fact_key'])
    op.create_table('conversation_memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('session_id', sa.String(64), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(768), nullable=True),
        sa.Column('medical_entities', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_conversation_memories_user_id_created', 'conversation_memories', ['user_id', 'created_at'])
    op.create_index(op.f('ix_conversation_memories_user_id'), 'conversation_memories', ['user_id'])
    op.create_index(op.f('ix_conversation_memories_session_id'), 'conversation_memories', ['session_id'])


def downgrade() -> None:
    op.drop_table('conversation_memories')
    op.drop_table('long_term_facts')
    op.drop_table('users')
```

**Step 3: Apply migration**

Run: `alembic upgrade head`
Expected: `Running upgrade ... -> 0005...`

**Step 4: Verify tables exist**

Run:
```bash
python -c "
from db.session import engine
from sqlalchemy import text, inspect
with engine.connect() as conn:
    tables = inspect(engine).get_table_names()
    print([t for t in tables if t in ('users','long_term_facts','conversation_memories')])
"
```
Expected: `['conversation_memories', 'long_term_facts', 'users']`

**Step 5: Commit**

```bash
git add alembic/versions/0005_add_long_term_memory.py
git commit -m "feat: alembic migration for long-term memory tables"
```

---

### Task 4: Create LongTermMemoryRepository

**Files:**
- Create: `db/long_term_memory_repository.py`
- Create: `tests/db/test_long_term_memory_repository.py`

**Step 1: Write failing tests**

Create `tests/db/test_long_term_memory_repository.py`:
```python
import pytest
from unittest.mock import MagicMock, patch


def _make_repo():
    from db.long_term_memory_repository import LongTermMemoryRepository
    mock_session = MagicMock()
    return LongTermMemoryRepository(session=mock_session), mock_session


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
    mock_session.get.return_value = None  # user does not exist

    repo = LongTermMemoryRepository(session=mock_session)
    user_id = repo.get_or_create_user("user-new")

    assert user_id == "user-new"
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


def test_upsert_long_term_fact_inserts_new():
    from db.long_term_memory_repository import LongTermMemoryRepository

    mock_session = MagicMock()
    # No existing fact
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
```

**Step 2: Run to verify they fail**

Run: `pytest tests/db/test_long_term_memory_repository.py -v 2>&1`
Expected: FAIL — module not found

**Step 3: Create db/long_term_memory_repository.py**

```python
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import ConversationMemoryModel, LongTermFactModel, UserModel

logger = logging.getLogger(__name__)


class LongTermMemoryRepository:
    """
    Handles all long-term, user-scoped memory operations.
    Separate from ChatRepository which is session-scoped.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── User identity ────────────────────────────────────────────────────────

    def get_or_create_user(self, user_token: str) -> str:
        """
        Resolve a persistent user_token to a user_id (they are the same value).
        Creates the user row on first visit. Returns user_id.
        """
        existing = self._session.get(UserModel, user_token)
        if existing:
            return existing.user_id

        user = UserModel(user_id=user_token)
        self._session.add(user)
        self._session.commit()
        logger.info("long_term_user_created", extra={"user_id": user_token})
        return user_token

    # ── Long-term facts ──────────────────────────────────────────────────────

    def upsert_long_term_fact(
        self,
        user_id: str,
        fact_key: str,
        fact_value: str,
        confidence: float = 0.7,
        source_session: str | None = None,
    ) -> None:
        """
        Upsert a user-scoped fact. Updates if existing confidence is lower.
        """
        existing = (
            self._session.query(LongTermFactModel)
            .filter(
                LongTermFactModel.user_id == user_id,
                LongTermFactModel.fact_key == fact_key,
            )
            .first()
        )
        if existing:
            if confidence >= existing.confidence:
                existing.fact_value = fact_value
                existing.confidence = confidence
                if source_session:
                    existing.source_session = source_session
            # Lower confidence update: ignore
        else:
            row = LongTermFactModel(
                user_id=user_id,
                fact_key=fact_key,
                fact_value=fact_value,
                confidence=confidence,
                source_session=source_session,
            )
            self._session.add(row)

        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.warning(
                "long_term_fact_upsert_failed",
                extra={"user_id": user_id, "fact_key": fact_key},
                exc_info=True,
            )

    def get_user_facts(self, user_id: str) -> list[dict[str, Any]]:
        """Return all long-term facts for a user as [{key, value, confidence}]."""
        rows = (
            self._session.query(LongTermFactModel)
            .filter(LongTermFactModel.user_id == user_id)
            .all()
        )
        return [
            {"key": r.fact_key, "value": r.fact_value, "confidence": r.confidence}
            for r in rows
        ]

    # ── Episodic conversation memories ───────────────────────────────────────

    def add_conversation_memory(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        embedding: list[float],
        medical_entities: dict[str, Any] | None = None,
    ) -> None:
        """Store a vector-embedded episodic memory of a past turn."""
        row = ConversationMemoryModel(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            embedding=embedding,
            medical_entities=medical_entities or {},
        )
        self._session.add(row)
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.warning(
                "conversation_memory_add_failed",
                extra={"user_id": user_id, "session_id": session_id},
                exc_info=True,
            )

    def search_similar_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Return top-k episodic memories for this user, ranked by cosine similarity
        to the query embedding. Uses pgvector's <=> operator.
        """
        if not query_embedding:
            return []

        try:
            results = self._session.execute(
                text(
                    """
                    SELECT summary, medical_entities, created_at,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                    FROM conversation_memories
                    WHERE user_id = :user_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :k
                    """
                ),
                {
                    "user_id": user_id,
                    "embedding": str(query_embedding),
                    "k": k,
                },
            ).fetchall()

            return [
                {
                    "summary": row.summary,
                    "medical_entities": row.medical_entities,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "similarity": float(row.similarity),
                }
                for row in results
            ]
        except Exception:
            logger.warning(
                "conversation_memory_search_failed",
                extra={"user_id": user_id},
                exc_info=True,
            )
            return []
```

**Step 4: Run tests**

Run: `pytest tests/db/test_long_term_memory_repository.py -v 2>&1`
Expected: all PASS

**Step 5: Commit**

```bash
git add db/long_term_memory_repository.py tests/db/test_long_term_memory_repository.py
git commit -m "feat: add LongTermMemoryRepository for user-scoped facts and episodic memories"
```

---

### Task 5: Persistent user token (frontend + API)

**Files:**
- Modify: `static/js/main.js`
- Modify: `core/contracts.py`
- Modify: `api/routes/chat.py`
- Modify: `api/deps.py`
- Create: `tests/api/test_user_token.py`

**Step 1: Write failing tests**

Create `tests/api/test_user_token.py`:
```python
def test_resolve_user_id_returns_none_when_no_token():
    from api.routes.chat import _resolve_user_id
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers.get.return_value = None
    payload = MagicMock()
    payload.user_token = None

    result = _resolve_user_id(request, payload)
    assert result is None


def test_resolve_user_id_from_header():
    from api.routes.chat import _resolve_user_id
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers.get.return_value = "tok-abc-123"
    payload = MagicMock()
    payload.user_token = None

    result = _resolve_user_id(request, payload)
    assert result == "tok-abc-123"


def test_resolve_user_id_from_payload():
    from api.routes.chat import _resolve_user_id
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers.get.return_value = None
    payload = MagicMock()
    payload.user_token = "tok-xyz-456"

    result = _resolve_user_id(request, payload)
    assert result == "tok-xyz-456"
```

**Step 2: Run to verify they fail**

Run: `pytest tests/api/test_user_token.py -v 2>&1`
Expected: FAIL

**Step 3: Add user_token to ChatRequest**

In `core/contracts.py`, update `ChatRequest`:
```python
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    stream: bool = False
    user_token: str | None = None   # ADD: persistent anonymous identity token
```

**Step 4: Add _resolve_user_id to api/routes/chat.py**

```python
def _resolve_user_id(request: Request, payload: "ChatRequest") -> str | None:
    """Resolve persistent user identity from header or payload. Returns None for anonymous."""
    # Prefer header (set by frontend localStorage logic)
    token = request.headers.get('X-User-Token')
    if not token:
        token = getattr(payload, 'user_token', None)
    return token.strip() if token and token.strip() else None
```

**Step 5: Add get_long_term_memory_repository to api/deps.py**

In `api/deps.py`, add:
```python
def get_long_term_memory_repository(db: Session = Depends(get_db)):
    from db.long_term_memory_repository import LongTermMemoryRepository
    return LongTermMemoryRepository(session=db)
```

**Step 6: Thread user_id through chat endpoint**

In `api/routes/chat.py`, update the `chat` function signature and body:
```python
@router.post('/chat', response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # ... existing code ...
    user_id = _resolve_user_id(request, payload)

    # Pass user_id into background task
    background_tasks.add_task(
        post_response_updates,
        session_id=session_id,
        user_id=user_id,           # ADD
        trace_id=trace_id,
        question=payload.message,
        result=result,
        current_core_state=memory_bundle.get('core_state'),
        db=db,                     # ADD — for long-term memory writes
    )
```

Also pass `user_id` into `load_memory_bundle` (Task 6 will wire it in).

**Step 7: Add localStorage token logic to static/js/main.js**

Find where the fetch call is made for `/api/chat` and add:
```javascript
// Persistent user identity — generated once, stored in localStorage
function getUserToken() {
    let token = localStorage.getItem('mg_user_token');
    if (!token) {
        token = 'u-' + crypto.randomUUID();
        localStorage.setItem('mg_user_token', token);
    }
    return token;
}

// Add to fetch headers wherever /api/chat or /api/chat/stream is called:
headers: {
    'Content-Type': 'application/json',
    'X-User-Token': getUserToken(),   // ADD THIS LINE
},
```

**Step 8: Run tests**

Run: `pytest tests/api/test_user_token.py -v 2>&1`
Expected: PASS

**Step 9: Commit**

```bash
git add core/contracts.py api/routes/chat.py api/deps.py static/js/main.js tests/api/test_user_token.py
git commit -m "feat: add persistent user_token for cross-session identity"
```

---

### Task 6: Extend MemoryAgent to load long-term memory

**Files:**
- Modify: `agents/memory_agent.py`
- Modify: `core/state_v2.py`
- Create: `tests/agents/test_long_term_memory_agent.py`

**Step 1: Add user_id and episodic_memories to state**

In `core/state_v2.py`, add to `AgentStateV2`:
```python
user_id: str | None
episodic_memories: list[dict[str, Any]]
long_term_memory_repo: Any | None
```

In `initialize_state()`:
```python
'user_id': None,
'episodic_memories': [],
'long_term_memory_repo': None,
```

In `reset_query_state()`:
```python
'episodic_memories': [],
```

**Step 2: Write failing tests**

Create `tests/agents/test_long_term_memory_agent.py`:
```python
def test_memory_agent_loads_user_facts_when_user_id_present(monkeypatch):
    from agents.memory_agent import MemoryAgent
    from core.state_v2 import initialize_state

    mock_repo = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock()
    mock_repo.get_user_facts.return_value = [
        {"key": "age", "value": "45", "confidence": 0.9},
        {"key": "condition", "value": "diabetes", "confidence": 0.85},
    ]
    mock_repo.search_similar_memories.return_value = []

    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    state = initialize_state("s1", "t1")
    state["question"] = "what is metformin?"
    state["user_id"] = "user-abc"
    state["long_term_memory_repo"] = mock_repo

    result = MemoryAgent(state)

    mock_repo.get_user_facts.assert_called_once_with("user-abc")
    # Long-term facts merged into state facts
    fact_keys = [f["key"] for f in result["facts"]]
    assert "age" in fact_keys
    assert "condition" in fact_keys


def test_memory_agent_skips_long_term_when_no_user_id(monkeypatch):
    from agents.memory_agent import MemoryAgent
    from core.state_v2 import initialize_state

    monkeypatch.setenv("LONG_TERM_MEMORY_ENABLED", "true")
    from core.settings import get_settings
    get_settings.cache_clear()

    state = initialize_state("s1", "t1")
    state["question"] = "what is aspirin?"
    state["user_id"] = None
    state["long_term_memory_repo"] = None

    result = MemoryAgent(state)  # should not raise
    assert result["episodic_memories"] == []


def test_memory_agent_loads_episodic_memories(monkeypatch):
    from agents.memory_agent import MemoryAgent
    from core.state_v2 import initialize_state

    mock_repo = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock()
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

    state = initialize_state("s1", "t1")
    state["question"] = "what is the best time to take metformin?"
    state["user_id"] = "user-abc"
    state["long_term_memory_repo"] = mock_repo

    result = MemoryAgent(state)

    assert len(result["episodic_memories"]) == 1
    assert "metformin" in result["episodic_memories"][0]["summary"]
```

**Step 3: Run to verify they fail**

Run: `pytest tests/agents/test_long_term_memory_agent.py -v 2>&1`
Expected: FAIL

**Step 4: Update MemoryAgent**

In `agents/memory_agent.py`, add at top:
```python
from tools.embedding_client import embed_query
from core.settings import get_settings
```

Update `MemoryAgent` function to add long-term memory loading block after existing fact extraction:

```python
def MemoryAgent(state: AgentStateV2) -> AgentStateV2:
    # --- existing code unchanged ---
    history = state.get('conversation_history', [])
    recent = history[-6:]
    older = history[:-6]

    if len(history) > 8:
        summary = _build_structured_summary(older)
        if summary:
            state['summary'] = summary
            logger.info("structured_summary_generated", extra={"length": len(summary)})

    state['conversation_history'] = recent

    facts = _extract_facts(state.get('question', ''))
    merged = {fact['key']: fact for fact in state.get('facts', [])}
    for fact in facts:
        merged[fact['key']] = fact
    state['facts'] = list(merged.values())

    # --- NEW: long-term memory loading ---
    settings = get_settings()
    user_id = state.get('user_id')
    lt_repo = state.get('long_term_memory_repo')

    if settings.long_term_memory_enabled and user_id and lt_repo is not None:
        _load_long_term_memory(state, user_id, lt_repo, settings)

    return state


def _load_long_term_memory(state: AgentStateV2, user_id: str, lt_repo, settings) -> None:
    """Load user-scoped facts and episodic memories into state."""
    # 1. Load structured long-term facts — merge into existing facts
    try:
        lt_facts = lt_repo.get_user_facts(user_id)
        if lt_facts:
            merged = {f['key']: f for f in state.get('facts', [])}
            for f in lt_facts:
                # Long-term facts override session facts only if higher confidence
                existing = merged.get(f['key'])
                if not existing or f['confidence'] >= existing.get('confidence', 0):
                    merged[f['key']] = f
            state['facts'] = list(merged.values())
            logger.info(
                "long_term_facts_loaded",
                extra={"user_id": user_id, "count": len(lt_facts)},
            )
    except Exception:
        logger.warning("long_term_facts_load_failed", extra={"user_id": user_id}, exc_info=True)

    # 2. Load episodic memories — vector search for similar past conversations
    try:
        question = state.get('question', '')
        if question:
            query_emb = embed_query(question)
            episodes = lt_repo.search_similar_memories(
                user_id, query_emb, k=settings.long_term_memory_recall_k
            )
            state['episodic_memories'] = episodes
            logger.info(
                "episodic_memories_loaded",
                extra={"user_id": user_id, "count": len(episodes)},
            )
    except Exception:
        logger.warning("episodic_memories_load_failed", extra={"user_id": user_id}, exc_info=True)
        state['episodic_memories'] = []
```

**Step 5: Run tests**

Run: `pytest tests/agents/test_long_term_memory_agent.py -v 2>&1`
Expected: PASS

**Step 6: Commit**

```bash
git add agents/memory_agent.py core/state_v2.py tests/agents/test_long_term_memory_agent.py
git commit -m "feat: MemoryAgent loads long-term user facts and episodic memories"
```

---

### Task 7: Inject episodic memories into executor prompt

**Files:**
- Modify: `agents/executor_agent.py`
- Modify: `tests/agents/test_long_term_memory_agent.py`

**Step 1: Write failing test**

Append to `tests/agents/test_long_term_memory_agent.py`:
```python
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
```

**Step 2: Run to verify they fail**

Run: `pytest tests/agents/test_long_term_memory_agent.py -v -k "episodic" 2>&1`
Expected: FAIL

**Step 3: Add _format_episodic_memories to executor_agent.py**

```python
def _format_episodic_memories(episodes: list[dict]) -> str:
    """Format episodic memories as a readable block for the executor prompt."""
    if not episodes:
        return ""
    lines = ["**Relevant past context from prior sessions:**"]
    for ep in episodes:
        date_str = ""
        created = ep.get("created_at")
        if created:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created)
                date_str = f"[{dt.strftime('%b %d')}] "
            except Exception:
                pass
        lines.append(f"- {date_str}{ep['summary']}")
    return "\n".join(lines)
```

**Step 4: Inject into executor prompt**

In `ExecutorAgent`, after computing `facts`, add:
```python
episodic_block = _format_episodic_memories(state.get('episodic_memories', []))
```

Update `_render_prompt_optimized` call to pass episodic block — add `{episodic_line}` to `_PROMPT_TEMPLATE_V2` after `{query_context_line}`:
```python
_PROMPT_TEMPLATE_V2 = """
...
{episodic_line}

Retrieved context:
...
""".strip()
```

And in `_render_prompt_optimized`:
```python
def _render_prompt_optimized(
    question, query, summary, facts, context,
    query_context='', session_intent='', episodic_memories='',
) -> str:
    return _PROMPT_TEMPLATE_V2.format(
        ...
        episodic_line=episodic_memories if episodic_memories else '',
    )
```

**Step 5: Run all executor tests**

Run: `pytest tests/agents/test_long_term_memory_agent.py -v 2>&1`
Expected: PASS

**Step 6: Commit**

```bash
git add agents/executor_agent.py tests/agents/test_long_term_memory_agent.py
git commit -m "feat: executor injects episodic memories into prompt"
```

---

### Task 8: Write long-term memory after each response

**Files:**
- Modify: `api/chat_runtime.py`
- Modify: `api/routes/chat.py`
- Create: `tests/api/test_long_term_memory_write.py`

This task wires the write path: after each response, extract facts → upsert to `long_term_facts`; if turn has medical content → embed + store to `conversation_memories`.

**Step 1: Write failing tests**

Create `tests/api/test_long_term_memory_write.py`:
```python
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
```

**Step 2: Run to verify they fail**

Run: `pytest tests/api/test_long_term_memory_write.py -v 2>&1`
Expected: FAIL

**Step 3: Add write_long_term_memory to api/chat_runtime.py**

```python
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

        # Build a short turn summary for embedding
        turn_summary = (
            f"User asked: {question[:200]}\n"
            f"Key facts mentioned: {', '.join(f'{f[\"key\"]}={f[\"value\"]}' for f in facts)}\n"
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
```

**Step 4: Call write_long_term_memory from post_response_updates**

Update signature and body of `post_response_updates` in `api/chat_runtime.py`:
```python
def post_response_updates(
    *,
    session_id: str,
    user_id: str | None = None,    # ADD
    trace_id: str,
    question: str,
    result: dict[str, Any],
    current_core_state: dict[str, Any] | None,
    lt_repo: Any | None = None,    # ADD
    working_memory_service: WorkingMemoryService | None = None,
    live_judge_service: LiveJudgeService | None = None,
) -> None:
    # ... existing working memory + live judge code unchanged ...

    # ADD at end:
    if lt_repo is not None:
        write_long_term_memory(
            user_id=user_id,
            session_id=session_id,
            question=question,
            result=result,
            lt_repo=lt_repo,
        )
```

**Step 5: Update chat endpoint to pass lt_repo and user_id to background task**

In `api/routes/chat.py`, update the `background_tasks.add_task` call:
```python
lt_repo = get_long_term_memory_repository(db) if user_id else None

background_tasks.add_task(
    post_response_updates,
    session_id=session_id,
    user_id=user_id,
    trace_id=trace_id,
    question=payload.message,
    result=result,
    current_core_state=memory_bundle.get('core_state'),
    lt_repo=lt_repo,
)
```

**Step 6: Run tests**

Run: `pytest tests/api/test_long_term_memory_write.py -v 2>&1`
Expected: PASS

**Step 7: Run full affected suite**

Run: `pytest tests/api/ tests/agents/ tests/db/ -v -q 2>&1 | tail -30`
Expected: no failures

**Step 8: Commit**

```bash
git add api/chat_runtime.py api/routes/chat.py tests/api/test_long_term_memory_write.py
git commit -m "feat: write long-term facts and episodic memories after each turn"
```

---

### Task 9: Wire user_id and lt_repo into WorkflowService

**Files:**
- Modify: `core/workflow_service.py`
- Modify: `api/routes/chat.py`

The `user_id` and `long_term_memory_repo` need to reach `AgentStateV2` so `MemoryAgent` can use them.

**Step 1: Update WorkflowService.run() and stream()**

In `core/workflow_service.py`, add `user_id` and `long_term_memory_repo` params:
```python
def run(
    self,
    question: str,
    session_id: str,
    trace_id: str,
    history: list[dict[str, Any]],
    summary: str,
    facts: list[dict[str, Any]],
    chat_repo=None,
    vector_repo=None,
    user_id: str | None = None,             # ADD
    long_term_memory_repo=None,              # ADD
) -> dict[str, Any]:
    state = initialize_state(session_id=session_id, trace_id=trace_id)
    state = reset_query_state(state, question)
    state['conversation_history'] = history
    state['summary'] = summary
    state['facts'] = facts
    state['chat_repo'] = chat_repo
    state['vector_repo'] = vector_repo
    state['user_id'] = user_id                        # ADD
    state['long_term_memory_repo'] = long_term_memory_repo  # ADD
    result = self.workflow.invoke(state)
    return result
```

Apply same changes to `stream()`.

**Step 2: Update chat endpoint to pass them**

In `api/routes/chat.py`:
```python
lt_repo = get_long_term_memory_repository(db) if user_id else None

result = get_workflow_service().run(
    question=payload.message,
    session_id=session_id,
    trace_id=trace_id,
    history=history,
    summary=summary,
    facts=facts,
    chat_repo=chat_repo,
    vector_repo=vector_repo,
    user_id=user_id,                 # ADD
    long_term_memory_repo=lt_repo,   # ADD
)
```

**Step 3: Verify app starts cleanly**

Run: `python -c "from app import app; print('ok')"`
Expected: `ok`

**Step 4: Run full test suite**

Run: `pytest tests/ -x -q 2>&1 | tail -30`
Expected: no new failures

**Step 5: Commit**

```bash
git add core/workflow_service.py api/routes/chat.py
git commit -m "feat: thread user_id and long_term_memory_repo into workflow state"
```

---

### Task 10: Add ADR and update docs

**Files:**
- Create: `docs/decisions/ADR-0006-long-term-memory.md`
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**ADR key points:**
- **Problem:** All memory is session-scoped. Users who return after closing the browser lose all context (conditions, medications, past conversations). The system always starts from zero.
- **Decision:** Three-layer addition: (1) Persistent anonymous `user_token` (localStorage UUID) maps sessions to a `user_id`. (2) `long_term_facts` table: user-scoped structured facts (age, conditions, drugs) persist across sessions, always loaded. (3) `conversation_memories` table: vector-embedded turn summaries, semantically recalled at query time.
- **Write path:** Runs inside existing `BackgroundTask` after each response — no new background workers needed. Embedding call is the only latency, fully async from user perspective.
- **Read path:** `MemoryAgent` loads facts + searches episodic memories before pipeline runs. Episodic recall uses same `embed_query` as document retrieval, separate table namespace.
- **Default off:** `LONG_TERM_MEMORY_ENABLED=false`. Enable when ready to test cross-session recall.
- **Anonymous degradation:** `user_id=None` → all long-term memory paths silently skip. Existing session behavior unchanged.

**Architecture note to add:**

```
Memory Stack (revised):
  Request scope    → AgentStateV2
  Session (hot)    → Redis working memory (recent_turns TTL=12h, core_state TTL=7d)
  Session (cold)   → PostgreSQL messages, conversation_summaries, user_facts (session-scoped)
  User (always)    → PostgreSQL long_term_facts (user-scoped, always loaded when user_id present)
  User (recalled)  → PostgreSQL conversation_memories + pgvector (semantic recall, top-K)
```

Log entry: `2026-03-17 — Long-term memory layer added: persistent user_token, user-scoped facts, episodic conversation memories with vector recall.`
