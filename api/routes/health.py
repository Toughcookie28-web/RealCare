from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_chat_repository, get_db, get_vector_repository
from observability.metrics import metrics_payload


router = APIRouter(tags=['health'])


@router.get('/health/live')
def live():
    return {'status': 'alive', 'service': 'MediGenius'}


@router.get('/health/ready')
def ready(db: Session = Depends(get_db)):
    checks: dict = {'service': 'MediGenius'}
    try:
        db.execute(text('SELECT 1'))
        get_chat_repository(db)
        get_vector_repository(db)
        checks['database'] = 'ok'
        checks['repositories'] = 'ok'
    except Exception:
        checks['database'] = 'unavailable'
        checks['status'] = 'degraded'
        return JSONResponse(status_code=503, content=checks)
    checks['status'] = 'ready'
    return checks


@router.get('/api/health')
def legacy_health():
    return {'status': 'healthy', 'service': 'MediGenius'}


@router.get('/metrics')
def metrics():
    return Response(content=metrics_payload(), media_type='text/plain; version=0.0.4')
