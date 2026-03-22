from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from api.chat_runtime import load_memory_bundle, post_response_updates
from api.deps import get_chat_repository, get_db, get_long_term_memory_repository, get_vector_repository
from core.contracts import ChatRequest, ChatResponse
from core.workflow_service import generate_trace_id, get_workflow_service


router = APIRouter(prefix='/api', tags=['chat'])


def _resolve_session_id(request: Request, conversation_id: str | None) -> str:
    if conversation_id:
        return conversation_id
    cookie_id = request.cookies.get('session_id')
    if cookie_id:
        return cookie_id
    return str(uuid.uuid4())


def _resolve_user_id(request: "Request", payload: "ChatRequest") -> str | None:
    """Resolve persistent user identity from header or payload. Returns None for anonymous."""
    token = request.headers.get('X-User-Token')
    if not token:
        token = getattr(payload, 'user_token', None)
    return token.strip() if token and token.strip() else None


@router.post('/chat', response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail='No message provided')

    trace_id = getattr(request.state, 'trace_id', generate_trace_id())
    session_id = _resolve_session_id(request, payload.conversation_id)

    chat_repo = get_chat_repository(db)
    vector_repo = get_vector_repository(db)

    memory_bundle = load_memory_bundle(session_id, chat_repo)
    history = memory_bundle['history']
    summary = memory_bundle['summary']
    facts = memory_bundle['facts']

    user_id = _resolve_user_id(request, payload)
    lt_repo = get_long_term_memory_repository(db) if user_id else None

    chat_repo.add_message(session_id, 'user', payload.message)

    result = get_workflow_service().run(
        question=payload.message,
        session_id=session_id,
        trace_id=trace_id,
        history=history,
        summary=summary,
        facts=facts,
        chat_repo=chat_repo,
        vector_repo=vector_repo,
        user_id=user_id,
        long_term_memory_repo=lt_repo,
    )

    answer = result.get('generation', 'Unable to generate response.')
    source = result.get('source', 'Unknown')
    timestamp = datetime.now().strftime('%I:%M %p')

    chat_repo.add_message(session_id, 'assistant', answer, source=source)
    if result.get('summary'):
        chat_repo.upsert_summary(session_id, result['summary'])
    for fact in result.get('facts', []):
        if fact.get('key') and fact.get('value'):
            chat_repo.upsert_fact(session_id, fact['key'], str(fact['value']), float(fact.get('confidence', 0.7)))

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

    response.set_cookie('session_id', session_id, httponly=True, samesite='lax')

    return ChatResponse(
        response=answer,
        source=source,
        timestamp=timestamp,
        success=bool(answer),
        trace_id=trace_id,
        otel_trace_id=result.get('otel_trace_id'),
        route=result.get('route'),
        citations=result.get('citations', []),
    )


@router.post('/chat/stream')
def chat_stream(
    payload: ChatRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail='No message provided')

    trace_id = getattr(request.state, 'trace_id', generate_trace_id())
    session_id = _resolve_session_id(request, payload.conversation_id)

    chat_repo = get_chat_repository(db)
    vector_repo = get_vector_repository(db)

    memory_bundle = load_memory_bundle(session_id, chat_repo)
    history = memory_bundle['history']
    summary = memory_bundle['summary']
    facts = memory_bundle['facts']

    user_id = _resolve_user_id(request, payload)
    lt_repo = get_long_term_memory_repository(db) if user_id else None

    chat_repo.add_message(session_id, 'user', payload.message)

    post_response_holder: dict[str, dict | None] = {'final_data': None}

    def event_stream():
        final_data = None
        for event in get_workflow_service().stream(
            question=payload.message,
            session_id=session_id,
            trace_id=trace_id,
            history=history,
            summary=summary,
            facts=facts,
            chat_repo=chat_repo,
            vector_repo=vector_repo,
            user_id=user_id,
            long_term_memory_repo=lt_repo,
        ):
            if event.get('event') == 'final':
                final_data = {
                    **event['data'],
                    **(event.get('_internal') or {}),
                }
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event['data'])}\n\n"

        if final_data:
            post_response_holder['final_data'] = final_data
            chat_repo.add_message(
                session_id,
                'assistant',
                final_data.get('response', 'Unable to generate response.'),
                source=final_data.get('source', 'Unknown'),
            )
            if final_data.get('summary'):
                chat_repo.upsert_summary(session_id, final_data['summary'])
            for fact in final_data.get('facts', []):
                if fact.get('key') and fact.get('value'):
                    chat_repo.upsert_fact(
                        session_id, fact['key'], str(fact['value']),
                        float(fact.get('confidence', 0.7)),
                    )

    def _run_stream_post_response():
        final_data = post_response_holder.get('final_data')
        if final_data:
            post_response_updates(
                session_id=session_id,
                user_id=user_id,
                trace_id=trace_id,
                question=payload.message,
                result=final_data,
                current_core_state=memory_bundle.get('core_state'),
                lt_repo=lt_repo,
            )

    stream_response = StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
        background=BackgroundTask(_run_stream_post_response),
    )
    stream_response.set_cookie('session_id', session_id, httponly=True, samesite='lax')
    return stream_response
