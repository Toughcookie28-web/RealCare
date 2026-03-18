from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db
from core.hitl import list_pending_reviews, update_review_status


router = APIRouter(prefix='/api/hitl', tags=['hitl'])


class HITLDecision(BaseModel):
    approval_id: str
    decision: str
    reviewer_note: str | None = None


@router.get('/pending')
def pending_reviews(db: Session = Depends(get_db)):
    return {'items': list_pending_reviews(db=db), 'success': True}


@router.post('/decision')
def review_decision(payload: HITLDecision, db: Session = Depends(get_db)):
    if payload.decision not in {'approved', 'rejected'}:
        raise HTTPException(status_code=400, detail='decision must be approved or rejected')
    ok = update_review_status(
        payload.approval_id,
        payload.decision,
        db=db,
        reviewer_note=payload.reviewer_note,
    )
    if not ok:
        raise HTTPException(status_code=404, detail='approval id not found')
    return {'success': True, 'approval_id': payload.approval_id, 'decision': payload.decision}
