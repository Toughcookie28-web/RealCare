import sys
import types
from pathlib import Path

from sqlalchemy.types import UserDefinedType


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pgvector_module = types.ModuleType("pgvector")
pgvector_sqlalchemy_module = types.ModuleType("pgvector.sqlalchemy")


class _Vector(UserDefinedType):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_col_spec(self, **_kwargs):
        return "VECTOR"


pgvector_sqlalchemy_module.Vector = _Vector
pgvector_module.sqlalchemy = pgvector_sqlalchemy_module
sys.modules.setdefault("pgvector", pgvector_module)
sys.modules.setdefault("pgvector.sqlalchemy", pgvector_sqlalchemy_module)

from db.models import ConversationSummaryModel, MessageModel, SessionModel, UserFactModel
from db.repositories import ChatRepository


class _FakeResult:
    def __init__(self, value=None):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeSession:
    def __init__(self):
        self.sessions: dict[str, SessionModel] = {}
        self.added: list[object] = []
        self.commit_count = 0

    def get(self, model, key):
        if model is SessionModel:
            return self.sessions.get(key)
        return None

    def add(self, obj):
        self.added.append(obj)
        if isinstance(obj, SessionModel):
            self.sessions[obj.session_id] = obj

    def execute(self, _query):
        return _FakeResult(None)

    def commit(self):
        self.commit_count += 1


def test_ensure_session_sets_required_timestamps_on_create():
    db = _FakeSession()
    repo = ChatRepository(db)

    repo.ensure_session("session-1")

    created = db.sessions["session-1"]
    assert created.created_at is not None
    assert created.last_active is not None


def test_add_message_sets_created_at_before_commit():
    db = _FakeSession()
    repo = ChatRepository(db)

    repo.add_message("session-1", "user", "hello")

    message = next(obj for obj in db.added if isinstance(obj, MessageModel))
    assert message.created_at is not None


def test_summary_and_fact_creation_set_required_timestamps():
    db = _FakeSession()
    repo = ChatRepository(db)

    repo.upsert_summary("session-1", "summary")
    repo.upsert_fact("session-1", "allergy", "penicillin")

    summary = next(obj for obj in db.added if isinstance(obj, ConversationSummaryModel))
    fact = next(obj for obj in db.added if isinstance(obj, UserFactModel))
    assert summary.updated_at is not None
    assert fact.created_at is not None
