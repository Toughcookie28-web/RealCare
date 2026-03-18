from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Sequence

import requests

from core.settings import get_settings
from tools.embedding_bootstrap import configure_embedding_cache_env

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingOverrides:
    provider: str | None = None
    dim: int | None = None
    openai_model: str | None = None


def _validate_dimension(values: Sequence[float], dim: int, provider: str, mode: str) -> list[float]:
    expected = max(int(dim), 1)
    vec = [float(v) for v in values]
    actual = len(vec)
    if actual != expected:
        raise ValueError(
            f'Embedding dimension mismatch for provider={provider} mode={mode}: '
            f'expected={expected}, actual={actual}. '
            'Align EMBEDDING_DIM with the selected embedding model output.'
        )
    return vec


def _clean_secret(value: str | None) -> str:
    if value is None:
        return ''
    value = value.strip()
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1].strip()
    return value


def _embed_with_gemini(text: str, task_type: str) -> list[float] | None:
    settings = get_settings()
    api_key = _clean_secret(settings.google_api_key)
    if not api_key:
        return None

    model = settings.embedding_gemini_model.strip() or 'gemini-embedding-001'
    model_path = model if model.startswith('models/') else f'models/{model}'
    url = f'https://generativelanguage.googleapis.com/v1beta/{model_path}:embedContent'
    payload = {
        'content': {
            'parts': [{'text': text}],
        },
        'taskType': task_type,
    }
    try:
        response = requests.post(url, params={'key': api_key}, json=payload, timeout=20)
        if response.status_code != 200:
            logger.warning(
                'Gemini embedding request failed with status=%s model=%s task_type=%s',
                response.status_code,
                model,
                task_type,
            )
            return None
        values = response.json().get('embedding', {}).get('values', [])
        if not values:
            logger.warning('Gemini embedding returned no values for model=%s task_type=%s', model, task_type)
            return None
        return [float(v) for v in values]
    except Exception as exc:
        logger.warning('Gemini embedding failed for model=%s task_type=%s: %s', model, task_type, exc, exc_info=True)
        return None


def _openai_embeddings_url(base_url: str | None = None) -> str:
    resolved_base_url = _clean_secret(base_url)
    if not resolved_base_url:
        settings = get_settings()
        resolved_base_url = _clean_secret(settings.openai_base_url) or 'https://api.openai.com/v1'
    base_url = resolved_base_url.rstrip('/')
    base_url = base_url.rstrip('/')
    return base_url if base_url.endswith('/embeddings') else f'{base_url}/embeddings'


def _build_openai_embedding_payload(
    input_value: str | list[str],
    *,
    model: str,
    dim: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'input': input_value,
        'model': model,
    }
    if model.startswith('text-embedding-3-'):
        payload['dimensions'] = max(int(dim), 1)
    return payload


def _resolve_dim(settings, overrides: EmbeddingOverrides | None = None) -> int:
    override_dim = None if overrides is None else overrides.dim
    return max(int(override_dim if override_dim is not None else settings.embedding_dim), 1)


def _resolve_provider_order(settings, overrides: EmbeddingOverrides | None = None) -> list[str]:
    primary_override = None if overrides is None else overrides.provider
    try:
        return settings.embedding_provider_order(primary_override=primary_override)
    except TypeError:
        primary = (primary_override or getattr(settings, 'embedding_provider', 'local') or 'local').strip().lower()
        if primary in {'huggingface', 'hf'}:
            primary = 'local'
        if primary not in {'local', 'gemini', 'openai'}:
            primary = 'local'

        order = [primary]
        if bool(getattr(settings, 'embedding_enable_backup', False)):
            backup = 'gemini' if primary == 'local' else 'local'
            if backup not in order:
                order.append(backup)
        return order


def _resolve_openai_model(settings, overrides: EmbeddingOverrides | None = None) -> str:
    override_model = None if overrides is None else overrides.openai_model
    return (
        _clean_secret(override_model)
        or _clean_secret(getattr(settings, 'embedding_openai_model', None))
        or 'text-embedding-3-large'
    )


def _embed_with_openai(text: str, *, overrides: EmbeddingOverrides | None = None) -> list[float] | None:
    vectors = _embed_batch_with_openai([text], overrides=overrides)
    if not vectors:
        return None
    return vectors[0]


def _embed_batch_with_openai(
    texts: list[str],
    *,
    overrides: EmbeddingOverrides | None = None,
) -> list[list[float]] | None:
    settings = get_settings()
    api_key = _clean_secret(settings.openai_api_key)
    if not api_key:
        return None
    model = _resolve_openai_model(settings, overrides)
    dim = _resolve_dim(settings, overrides)

    try:
        response = requests.post(
            _openai_embeddings_url(settings.openai_base_url),
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json=_build_openai_embedding_payload(
                texts if len(texts) > 1 else texts[0],
                model=model,
                dim=dim,
            ),
            timeout=30,
        )
        if response.status_code != 200:
            logger.warning(
                'OpenAI embedding request failed with status=%s model=%s',
                response.status_code,
                model,
            )
            return None

        rows = response.json().get('data', [])
        if not rows:
            logger.warning('OpenAI embedding returned no data for model=%s', model)
            return None

        return [[float(v) for v in row.get('embedding', [])] for row in rows]
    except Exception as exc:
        logger.warning(
            'OpenAI embedding failed for model=%s: %s',
            model,
            exc,
            exc_info=True,
        )
        return None


@lru_cache(maxsize=1)
def _get_sentence_transformer_class():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_local_embedding_model():
    cache_folder = configure_embedding_cache_env(get_settings().embedding_cache_dir)
    sentence_transformer_cls = _get_sentence_transformer_class()
    if sentence_transformer_cls is None:
        return None
    settings = get_settings()
    try:
        return sentence_transformer_cls(
            settings.embedding_local_model,
            trust_remote_code=settings.embedding_local_trust_remote_code,
            cache_folder=cache_folder,
        )
    except Exception as exc:
        logger.warning('Failed to load local embedding model: %s', exc, exc_info=True)
        return None


def _normalize_local_strategy(value: str | None) -> str:
    strategy = (value or 'plain').strip().lower()
    if strategy == 'auto':
        logger.info('EMBEDDING_LOCAL_STRATEGY=auto is deprecated; treating it as asymmetric.')
        return 'asymmetric'
    if strategy not in {'plain', 'asymmetric'}:
        logger.warning('Unknown EMBEDDING_LOCAL_STRATEGY=%r; falling back to plain.', value)
        return 'plain'
    return strategy


def _embed_with_local(text: str, mode: str) -> list[float] | None:
    settings = get_settings()
    model = get_local_embedding_model()
    if model is None:
        return None
    strategy = _normalize_local_strategy(settings.embedding_local_strategy)
    try:
        encoder = None
        if strategy == 'asymmetric':
            # SentenceTransformer always exposes encode(...), but query/document
            # specific methods are model- and library-specific. When changing
            # local embedding models, check the model docs/API before assuming
            # encode_query(...) / encode_document(...) exist or have the right
            # retrieval semantics.
            encoder_name = 'encode_query' if mode == 'query' else 'encode_document'
            candidate = getattr(model, encoder_name, None)
            if callable(candidate):
                encoder = candidate
        vec = encoder(text, normalize_embeddings=True) if encoder is not None else model.encode(
            text,
            normalize_embeddings=True,
        )

        if hasattr(vec, 'tolist'):
            vec = vec.tolist()
        return [float(v) for v in vec]
    except Exception as exc:
        logger.warning(
            'Local embedding failed for model=%s mode=%s strategy=%s: %s',
            settings.embedding_local_model,
            mode,
            strategy,
            exc,
            exc_info=True,
        )
        return None


def _embed(text: str, mode: str, *, overrides: EmbeddingOverrides | None = None) -> list[float]:
    settings = get_settings()
    dim = _resolve_dim(settings, overrides)
    for provider in _resolve_provider_order(settings, overrides):
        if provider == 'local':
            local_vector = _embed_with_local(text, mode=mode)
            if local_vector is not None:
                return _validate_dimension(local_vector, dim, provider='local', mode=mode)
        elif provider == 'openai':
            openai_vector = _embed_with_openai(text, overrides=overrides)
            if openai_vector is not None:
                return _validate_dimension(openai_vector, dim, provider='openai', mode=mode)
        elif provider == 'gemini':
            task_type = 'RETRIEVAL_QUERY' if mode == 'query' else 'RETRIEVAL_DOCUMENT'
            gemini_vector = _embed_with_gemini(text, task_type=task_type)
            if gemini_vector is not None:
                return _validate_dimension(gemini_vector, dim, provider='gemini', mode=mode)

    raise RuntimeError(
        'No embedding provider returned a vector. '
        'Check EMBEDDING_PROVIDER / backup settings and provider credentials.'
    )


def embed_query(text: str, *, overrides: EmbeddingOverrides | None = None) -> list[float]:
    return _embed(text, mode='query', overrides=overrides)


def embed_document(text: str, *, overrides: EmbeddingOverrides | None = None) -> list[float]:
    return _embed(text, mode='document', overrides=overrides)


def embed_documents_batch(
    texts: list[str],
    batch_size: int = 32,
    *,
    overrides: EmbeddingOverrides | None = None,
) -> list[list[float]]:
    """Embed multiple texts in batches, respecting provider order."""
    if not texts:
        return []

    settings = get_settings()
    dim = _resolve_dim(settings, overrides)
    provider_order = _resolve_provider_order(settings, overrides)

    for provider in provider_order:
        if provider == 'openai':
            openai_vectors = _embed_batch_with_openai(texts, overrides=overrides)
            if openai_vectors is not None:
                return [
                    _validate_dimension(vec, dim, provider='openai', mode='document')
                    for vec in openai_vectors
                ]

        elif provider == 'local':
            model = get_local_embedding_model()
            if model is not None:
                strategy = _normalize_local_strategy(settings.embedding_local_strategy)
                try:
                    encoder = None
                    if strategy == 'asymmetric':
                        candidate = getattr(model, 'encode_document', None)
                        if callable(candidate):
                            encoder = candidate

                    all_vectors: list[list[float]] = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        if encoder is not None:
                            vecs = encoder(batch, normalize_embeddings=True)
                        else:
                            vecs = model.encode(batch, normalize_embeddings=True)
                        for vec in vecs:
                            v = vec.tolist() if hasattr(vec, 'tolist') else [float(x) for x in vec]
                            all_vectors.append(_validate_dimension(v, dim, provider='local', mode='document'))

                    logger.info("batch_embedding_complete", extra={"count": len(all_vectors), "batch_size": batch_size})
                    return all_vectors
                except Exception as exc:
                    logger.warning("Batch encoding failed for local, trying next provider: %s", exc)

    # Fallback: sequential embedding
    return [embed_document(t, overrides=overrides) for t in texts]
