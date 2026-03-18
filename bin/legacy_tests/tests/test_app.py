import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_live_and_ready_health():
    live = client.get('/health/live')
    ready = client.get('/health/ready')

    assert live.status_code == 200
    assert ready.status_code == 200
    assert live.json()['status'] == 'alive'
    assert ready.json()['status'] == 'ready'


def test_chat_endpoint_basic_response():
    response = client.post('/api/chat', json={'message': 'What are symptoms of flu?'})
    assert response.status_code == 200

    payload = response.json()
    assert 'response' in payload
    assert 'source' in payload
    assert 'trace_id' in payload
    assert payload['success'] is True


def test_session_listing_and_history_endpoints():
    client.post('/api/chat', json={'message': 'hello'})

    sessions = client.get('/api/sessions')
    history = client.get('/api/history')

    assert sessions.status_code == 200
    assert history.status_code == 200
    assert sessions.json()['success'] is True
    assert history.json()['success'] is True
