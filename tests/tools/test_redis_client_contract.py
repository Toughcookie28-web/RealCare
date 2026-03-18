import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.redis_client as redis_client


class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.expiry = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.expiry[key] = ex


def test_json_helpers_round_trip_through_redis(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(redis_client, "_redis_client", fake_redis)
    monkeypatch.setattr(redis_client, "_redis_initialised", True)

    redis_client.set_json("core_state:abc", {"session_intent": "compare drugs"}, ttl=30)

    assert redis_client.get_json("core_state:abc") == {"session_intent": "compare drugs"}
    assert fake_redis.expiry["core_state:abc"] == 30


def test_append_json_list_trims_to_max_length(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(redis_client, "_redis_client", fake_redis)
    monkeypatch.setattr(redis_client, "_redis_initialised", True)

    redis_client.append_json_list("recent_turns:abc", {"content": "turn-1"}, max_length=2, ttl=20)
    redis_client.append_json_list("recent_turns:abc", {"content": "turn-2"}, max_length=2, ttl=20)
    redis_client.append_json_list("recent_turns:abc", {"content": "turn-3"}, max_length=2, ttl=20)

    assert redis_client.get_json("recent_turns:abc") == [
        {"content": "turn-2"},
        {"content": "turn-3"},
    ]
    assert fake_redis.expiry["recent_turns:abc"] == 20


def test_json_helpers_fall_back_to_local_store_when_redis_is_unavailable(monkeypatch):
    monkeypatch.setattr(redis_client, "_redis_client", None)
    monkeypatch.setattr(redis_client, "_redis_initialised", True)
    redis_client._local_json.clear()

    redis_client.set_json("core_state:local", {"medical_facts": {"age": "45"}}, ttl=15)

    assert redis_client.get_json("core_state:local") == {"medical_facts": {"age": "45"}}
