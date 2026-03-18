from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    NONE = 'none'
    VALIDATION_ERROR = 'validation_error'
    SAFETY_BLOCKED = 'safety_blocked'
    LLM_TIMEOUT = 'llm_timeout'
    LLM_PROVIDER_ERROR = 'llm_provider_error'
    RETRIEVAL_ERROR = 'retrieval_error'
    DATABASE_ERROR = 'database_error'
    INTERNAL_ERROR = 'internal_error'


class Citation(BaseModel):
    source: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    url: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    stream: bool = False
    user_token: str | None = None  # persistent anonymous identity token


class ChatResponse(BaseModel):
    response: str
    source: str
    timestamp: str
    success: bool
    trace_id: str
    route: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ChatHistoryItem(BaseModel):
    role: Literal['user', 'assistant']
    content: str
    source: str | None = None
    timestamp: datetime | str | None = None


class NodeResult(BaseModel):
    status: Literal['ok', 'error', 'blocked']
    payload: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    error: str | None = None
    error_code: ErrorCode = ErrorCode.NONE
    retryable: bool = False


class StreamEvent(BaseModel):
    event: Literal['status', 'token', 'final', 'error']
    trace_id: str
    data: dict[str, Any]
