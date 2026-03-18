from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional provider
    ChatGroq = None

from core.settings import get_settings

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional fallback provider
    ChatOpenAI = None


def _clean_secret(value: str | None) -> str:
    if value is None:
        return ''
    value = value.strip()
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1].strip()
    return value


@lru_cache(maxsize=1)
def get_primary_llm() -> BaseChatModel | None:
    settings = get_settings()
    provider = _normalize_provider(settings.llm_primary_provider)
    return _build_model(provider, settings.llm_primary_model)


@lru_cache(maxsize=1)
def get_fallback_llm() -> BaseChatModel | None:
    settings = get_settings()
    provider = _normalize_provider(settings.llm_fallback_provider)
    return _build_model(provider, settings.llm_fallback_model)


def _normalize_provider(provider: str | None) -> str:
    value = (provider or 'groq').strip().lower()
    if value not in {'groq', 'openai'}:
        return 'groq'
    return value


def _build_model(provider: str, model_name: str) -> BaseChatModel | None:
    settings = get_settings()
    groq_key = _clean_secret(settings.groq_api_key)
    openai_key = _clean_secret(settings.openai_api_key)
    openai_base_url = _clean_secret(settings.openai_base_url)
    if provider == 'groq':
        if not groq_key or ChatGroq is None:
            return None
        return ChatGroq(
            api_key=groq_key,
            model_name=model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == 'openai':
        if not openai_key or ChatOpenAI is None:
            return None
        kwargs = {
            'api_key': openai_key,
            'model': model_name,
            'temperature': settings.llm_temperature,
            'max_tokens': settings.llm_max_tokens,
        }
        if openai_base_url:
            kwargs['base_url'] = openai_base_url
        return ChatOpenAI(**kwargs)

    return None


def resolve_llm(
    *,
    primary_provider: str | None = None,
    primary_model: str | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> BaseChatModel | None:
    settings = get_settings()

    resolved_primary_provider = _normalize_provider(primary_provider or settings.llm_primary_provider)
    resolved_primary_model = (primary_model or settings.llm_primary_model or '').strip()
    resolved_fallback_provider = _normalize_provider(fallback_provider or settings.llm_fallback_provider)
    resolved_fallback_model = (fallback_model or settings.llm_fallback_model or '').strip()

    primary = _build_model(resolved_primary_provider, resolved_primary_model)
    if primary is not None:
        return primary

    return _build_model(resolved_fallback_provider, resolved_fallback_model)

def invoke_llm(prompt: str, system: str | None = None, use_fallback: bool = False) -> str:
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    def _call(model: BaseChatModel | None) -> str:
        if model is None:
            return ''
        response = model.invoke(messages)
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()

    if use_fallback:
        try:
            return _call(get_fallback_llm())
        except Exception:
            return ''

    primary = get_primary_llm()
    if primary is not None:
        try:
            return _call(primary)
        except Exception:
            pass

    try:
        return _call(get_fallback_llm())
    except Exception:
        return ''


def _invoke_json_mode(
    messages: list,
    use_fallback: bool = False,
) -> str:
    """Invoke LLM with response_format=json_object when supported."""
    json_kwargs = {"response_format": {"type": "json_object"}}

    def _call(model: BaseChatModel | None) -> str:
        if model is None:
            return ''
        try:
            bound = model.bind(**json_kwargs)
        except Exception:
            bound = model  # model doesn't support response_format, use plain
        response = bound.invoke(messages)
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()

    if use_fallback:
        try:
            return _call(get_fallback_llm())
        except Exception:
            return ''

    primary = get_primary_llm()
    if primary is not None:
        try:
            return _call(primary)
        except Exception:
            pass

    try:
        return _call(get_fallback_llm())
    except Exception:
        return ''


def _parse_json_text(text: str) -> dict:
    """Try to parse JSON from text, with brace-extraction fallback."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def invoke_json(
    prompt: str,
    system: str | None = None,
    use_fallback: bool = False,
    pydantic_model: type | None = None,
    retry_on_validation: bool = True,
) -> dict:
    """Invoke LLM expecting JSON output.

    Layers:
    1. response_format=json_object (API-level enforcement)
    2. Brace-extraction fallback for malformed responses
    3. Pydantic validation + one retry if pydantic_model is provided
    """
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    # Layer 1: API-level JSON mode
    text = _invoke_json_mode(messages, use_fallback=use_fallback)

    # Layer 2: Parse JSON (with brace-extraction fallback)
    result = _parse_json_text(text)

    # Layer 3: Pydantic validation + retry
    if pydantic_model is not None and result:
        try:
            pydantic_model.model_validate(result)
        except Exception as exc:
            if retry_on_validation:
                retry_prompt = (
                    f"Your previous JSON response had a validation error:\n"
                    f"{exc}\n\n"
                    f"Original response:\n{text}\n\n"
                    f"Please fix the JSON and return ONLY valid JSON."
                )
                retry_messages = list(messages) + [HumanMessage(content=retry_prompt)]
                retry_text = _invoke_json_mode(retry_messages, use_fallback=use_fallback)
                retry_result = _parse_json_text(retry_text)
                if retry_result:
                    try:
                        pydantic_model.model_validate(retry_result)
                        return retry_result
                    except Exception:
                        pass  # retry also failed, return original result

    return result
