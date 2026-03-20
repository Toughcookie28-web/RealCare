from __future__ import annotations

import logging
import re
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from observability.metrics import REQUEST_COUNT, REQUEST_LATENCY


_PATH_NORMALIZERS = [
    (re.compile(r'/api/session/[^/]+'), '/api/session/{id}'),
]


def _normalize_path(raw: str) -> str:
    for pattern, replacement in _PATH_NORMALIZERS:
        raw = pattern.sub(replacement, raw)
    return raw


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get('x-trace-id') or str(uuid.uuid4())
        request.state.trace_id = trace_id

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            method = request.method
            path = _normalize_path(request.url.path)
            REQUEST_COUNT.labels(method=method, path=path, status=str(status_code)).inc()
            REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
            logging.getLogger('medigenius.http').info(
                'request_complete',
                extra={
                    'trace_id': trace_id,
                    'path': path,
                    'method': method,
                    'latency_ms': int(elapsed * 1000),
                },
            )
