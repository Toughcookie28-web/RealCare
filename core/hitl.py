from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session


def create_pending_review(
    approval_id: str,
    session_id: str,
    trace_id: str,
    question: str,
    db: Session | None = None,
) -> dict[str, Any]:
    if db is None:
        raise RuntimeError('persistent HITL database required')

    record = {
        'approval_id': approval_id,
        'session_id': session_id,
        'trace_id': trace_id,
        'question': question,
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }

    try:
        from db.models import HitlReviewModel
        row = HitlReviewModel(
            approval_id=approval_id,
            session_id=session_id,
            trace_id=trace_id,
            question=question,
        )
        db.add(row)
        db.commit()
        return record
    except Exception as exc:
        db.rollback()
        raise RuntimeError('persistent HITL database unavailable') from exc


def list_pending_reviews(db: Session | None = None) -> list[dict[str, Any]]:
    if db is None:
        raise RuntimeError('persistent HITL database required')

    try:
        from db.models import HitlReviewModel
        rows = db.query(HitlReviewModel).filter(
            HitlReviewModel.status == 'pending'
        ).order_by(HitlReviewModel.created_at.desc()).all()
        return [
            {
                'approval_id': r.approval_id,
                'session_id': r.session_id,
                'trace_id': r.trace_id,
                'question': r.question,
                'status': r.status,
                'created_at': r.created_at.isoformat() if r.created_at else None,
                'reviewed_at': r.reviewed_at.isoformat() if r.reviewed_at else None,
                'reviewer_note': r.reviewer_note,
            }
            for r in rows
        ]
    except Exception as exc:
        raise RuntimeError('persistent HITL database unavailable') from exc


def update_review_status(
    approval_id: str,
    status: str,
    db: Session | None = None,
    reviewer_note: str | None = None,
) -> bool:
    if db is None:
        raise RuntimeError('persistent HITL database required')

    try:
        from db.models import HitlReviewModel
        row = db.get(HitlReviewModel, approval_id)
        if row is None:
            return False
        row.status = status
        row.reviewed_at = datetime.now(timezone.utc)
        if reviewer_note:
            row.reviewer_note = reviewer_note
        db.commit()
        return True
    except Exception as exc:
        db.rollback()
        raise RuntimeError('persistent HITL database unavailable') from exc
