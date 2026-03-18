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
