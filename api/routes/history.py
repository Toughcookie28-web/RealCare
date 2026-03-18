from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from api.deps import get_chat_repository, get_db


router = APIRouter(prefix='/api', tags=['history'])


@router.get('/history')
def get_history(request: Request, db: Session = Depends(get_db)):
    session_id = request.cookies.get('session_id')
    if not session_id:
        return {'messages': [], 'success': True}

    repo = get_chat_repository(db)
    return {'messages': repo.get_history(session_id), 'success': True}


@router.get('/sessions')
def get_sessions(db: Session = Depends(get_db)):
    repo = get_chat_repository(db)
    sessions = [
        {
            'session_id': item.session_id,
            'created_at': item.created_at.isoformat() if item.created_at else None,
            'last_active': item.last_active.isoformat() if item.last_active else None,
            'preview': item.preview,
        }
        for item in repo.list_sessions()
    ]
    return {'sessions': sessions, 'success': True}


@router.get('/session/{session_id}')
def load_session(session_id: str, response: Response, db: Session = Depends(get_db)):
    repo = get_chat_repository(db)
    response.set_cookie('session_id', session_id, httponly=True, samesite='lax')
    return {'messages': repo.get_history(session_id), 'session_id': session_id, 'success': True}


@router.delete('/session/{session_id}')
def delete_session(session_id: str, request: Request, response: Response, db: Session = Depends(get_db)):
    repo = get_chat_repository(db)
    repo.delete_session(session_id)

    active = request.cookies.get('session_id')
    if active == session_id:
        response.set_cookie('session_id', str(uuid.uuid4()), httponly=True, samesite='lax')

    return {'message': 'Session deleted', 'success': True}


@router.post('/clear')
def clear_session(request: Request, db: Session = Depends(get_db)):
    session_id = request.cookies.get('session_id')
    if session_id:
        repo = get_chat_repository(db)
        repo.delete_session(session_id)
    return {'message': 'Conversation cleared', 'success': True}


@router.post('/new-chat')
def new_chat(response: Response):
    new_session_id = str(uuid.uuid4())
    response.set_cookie('session_id', new_session_id, httponly=True, samesite='lax')
    return {'message': 'New chat created', 'session_id': new_session_id, 'success': True}
