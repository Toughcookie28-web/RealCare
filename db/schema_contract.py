from __future__ import annotations


SCHEMA_AUTHORITY = "alembic"
DOCUMENT_CHUNK_VECTOR_DIM = 768
DOCUMENT_CHUNK_VECTOR_SQL_TYPE = f"vector({DOCUMENT_CHUNK_VECTOR_DIM})"
CONVERSATION_MEMORY_VECTOR_DIM = 768   # same embedding model as document_chunks
CONVERSATION_MEMORY_VECTOR_SQL_TYPE = f"vector({CONVERSATION_MEMORY_VECTOR_DIM})"


def default_embedding_dim() -> int:
    return DOCUMENT_CHUNK_VECTOR_DIM


def require_document_chunk_vector_dim(value: int) -> int:
    value = int(value)
    if value != DOCUMENT_CHUNK_VECTOR_DIM:
        raise ValueError(
            "EMBEDDING_DIM must match the document chunk schema contract: "
            f"{DOCUMENT_CHUNK_VECTOR_DIM}."
        )
    return value
