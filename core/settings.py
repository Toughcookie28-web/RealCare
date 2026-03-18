from functools import lru_cache
import secrets
from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from db.schema_contract import default_embedding_dim, require_document_chunk_vector_dim


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'MediGenius'
    env: str = 'development'
    debug: bool = False

    api_host: str = '0.0.0.0'
    api_port: int = 8000

    database_url: str = Field(
        default='postgresql+psycopg://postgres:postgres@localhost:5432/medigenius',
        alias='DATABASE_URL',
    )
    pgvector_enabled: bool = Field(default=True, alias='PGVECTOR_ENABLED')
    database_strict_mode: bool | None = Field(default=None, alias='DATABASE_STRICT_MODE')

    groq_api_key: str | None = Field(default=None, alias='GROQ_API_KEY')
    tavily_api_key: str | None = Field(default=None, alias='TAVILY_API_KEY')
    openai_api_key: str | None = Field(default=None, alias='OPENAI_API_KEY')
    cohere_api_key: str | None = Field(default=None, alias='COHERE_API_KEY')
    cohere_rerank_model: str = Field(default='rerank-v3.5', alias='COHERE_RERANK_MODEL')

    redis_url: str | None = Field(default=None, alias='REDIS_URL')
    redis_cache_ttl: int = Field(default=86400, alias='REDIS_CACHE_TTL')  # 24 hours
    working_memory_recent_turns_ttl: int = Field(default=43200, alias='WORKING_MEMORY_RECENT_TURNS_TTL')
    working_memory_core_state_ttl: int = Field(default=604800, alias='WORKING_MEMORY_CORE_STATE_TTL')
    working_memory_recent_turns_max: int = Field(default=12, alias='WORKING_MEMORY_RECENT_TURNS_MAX')
    live_judge_enabled: bool = Field(default=False, alias='LIVE_JUDGE_ENABLED')
    live_judge_sampling_ratio: float = Field(default=0.2, alias='LIVE_JUDGE_SAMPLING_RATIO')

    llm_primary_model: str = Field(default='moonshotai/kimi-k2-instruct-0905', alias='LLM_PRIMARY_MODEL')
    llm_fallback_model: str = Field(default='llama-3.3-70b-versatile', alias='LLM_FALLBACK_MODEL')
    llm_primary_provider: str = Field(default='groq', alias='LLM_PRIMARY_PROVIDER')
    llm_fallback_provider: str = Field(default='groq', alias='LLM_FALLBACK_PROVIDER')
    tier3_llm_primary_provider: str | None = Field(default=None, alias='TIER3_LLM_PRIMARY_PROVIDER')
    tier3_llm_primary_model: str | None = Field(default=None, alias='TIER3_LLM_PRIMARY_MODEL')
    tier3_llm_fallback_provider: str | None = Field(default=None, alias='TIER3_LLM_FALLBACK_PROVIDER')
    tier3_llm_fallback_model: str | None = Field(default=None, alias='TIER3_LLM_FALLBACK_MODEL')
    llm_temperature: float = Field(default=0.2, alias='LLM_TEMPERATURE')
    llm_max_tokens: int = Field(default=1200, alias='LLM_MAX_TOKENS')
    openai_base_url: str | None = Field(default=None, alias='OPENAI_BASE_URL')

    embedding_provider: str = Field(default='local', alias='EMBEDDING_PROVIDER')
    embedding_local_model: str = Field(
        default='Alibaba-NLP/gte-base-en-v1.5',
        alias='EMBEDDING_LOCAL_MODEL',
    )
    embedding_local_strategy: str = Field(default='plain', alias='EMBEDDING_LOCAL_STRATEGY')
    embedding_local_trust_remote_code: bool = Field(default=True, alias='EMBEDDING_LOCAL_TRUST_REMOTE_CODE')
    embedding_cache_dir: str = Field(default='/tmp/medigenius-embeddings', alias='EMBEDDING_CACHE_DIR')
    embedding_gemini_model: str = Field(
        default='gemini-embedding-001',
        alias='EMBEDDING_GEMINI_MODEL',
    )
    embedding_openai_model: str = Field(
        default='text-embedding-3-large',
        alias='EMBEDDING_OPENAI_MODEL',
    )
    embedding_enable_backup: bool = Field(default=False, alias='EMBEDDING_ENABLE_BACKUP')
    google_api_key: str | None = Field(default=None, alias='GOOGLE_API_KEY')
    embedding_dim: int = Field(default_factory=default_embedding_dim, alias='EMBEDDING_DIM')
    tier3_embedding_provider: str | None = Field(default=None, alias='TIER3_EMBEDDING_PROVIDER')
    tier3_embedding_openai_model: str | None = Field(default=None, alias='TIER3_EMBEDDING_OPENAI_MODEL')
    tier3_embedding_dim: int | None = Field(default=None, alias='TIER3_EMBEDDING_DIM')

    trusted_medical_domains: str = Field(
        default='nih.gov,mayoclinic.org,who.int,cdc.gov,pubmed.ncbi.nlm.nih.gov',
        alias='TRUSTED_MEDICAL_DOMAINS',
    )

    semantic_cache_threshold: float = Field(default=0.92, alias='SEMANTIC_CACHE_THRESHOLD')
    semantic_cache_size: int = Field(default=5000, alias='SEMANTIC_CACHE_SIZE')
    semantic_cache_enabled: bool | None = Field(default=None, alias='SEMANTIC_CACHE_ENABLED')
    semantic_cache_version: str = Field(default='executor-v1', alias='SEMANTIC_CACHE_VERSION')
    bm25_k1: float = Field(default=1.2, alias='BM25_K1')
    bm25_b: float = Field(default=0.75, alias='BM25_B')
    hybrid_dense_weight: float = Field(default=0.7, alias='HYBRID_DENSE_WEIGHT')

    chunk_text_target: int = Field(default=400, alias='CHUNK_TEXT_TARGET')
    chunk_text_max: int = Field(default=500, alias='CHUNK_TEXT_MAX')
    chunk_text_overlap: int = Field(default=50, alias='CHUNK_TEXT_OVERLAP')
    chunk_table_max: int = Field(default=450, alias='CHUNK_TABLE_MAX')
    enrich_chunks_with_llm: bool = Field(default=False, alias='ENRICH_CHUNKS_WITH_LLM')
    enable_parent_child_chunking: bool = Field(default=False, alias='ENABLE_PARENT_CHILD_CHUNKING')
    chunk_child_target: int = Field(default=200, alias='CHUNK_CHILD_TARGET')
    chunk_child_max: int = Field(default=250, alias='CHUNK_CHILD_MAX')
    chunk_parent_max_children: int = Field(default=5, alias='CHUNK_PARENT_MAX_CHILDREN')

    routing_rag_confidence_threshold: float = Field(default=0.12, alias='ROUTING_RAG_CONFIDENCE_THRESHOLD')
    auto_ingest_on_startup: bool = Field(default=False, alias='AUTO_INGEST_ON_STARTUP')

    enable_hitl_interrupts: bool = Field(default=True, alias='ENABLE_HITL_INTERRUPTS')
    enable_query_decomposition: bool = Field(default=False, alias='ENABLE_QUERY_DECOMPOSITION')

    long_term_memory_enabled: bool = Field(default=False, alias='LONG_TERM_MEMORY_ENABLED')
    long_term_memory_recall_k: int = Field(default=3, alias='LONG_TERM_MEMORY_RECALL_K')
    long_term_memory_min_confidence: float = Field(default=0.6, alias='LONG_TERM_MEMORY_MIN_CONFIDENCE')
    long_term_memory_episode_min_facts: int = Field(default=1, alias='LONG_TERM_MEMORY_EPISODE_MIN_FACTS')

    otel_enabled: bool = Field(default=True, alias='OTEL_ENABLED')

    session_secret: str | None = Field(default=None, alias='SESSION_SECRET')
    allowed_origins: str = Field(default='http://localhost:8000', alias='ALLOWED_ORIGINS')

    @property
    def trusted_domains_list(self) -> list[str]:
        return [d.strip() for d in self.trusted_medical_domains.split(',') if d.strip()]

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(',') if o.strip()]

    def embedding_provider_order(self, primary_override: str | None = None) -> list[str]:
        """
        Resolve embedding provider order.
        Primary is explicit (`local`, `gemini`, or `openai`), optional backup
        keeps the existing local fallback behavior for remote providers.
        """
        primary = (primary_override or self.embedding_provider or 'local').strip().lower()
        if primary in {'huggingface', 'hf'}:
            primary = 'local'
        if primary not in {'local', 'gemini', 'openai'}:
            primary = 'local'

        order = [primary]
        if self.embedding_enable_backup:
            backup = 'gemini' if primary == 'local' else 'local'
            if backup not in order:
                order.append(backup)
        return order

    @field_validator('semantic_cache_enabled', 'database_strict_mode', mode='before')
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == '':
            return None
        return v

    @field_validator('pgvector_enabled')
    @classmethod
    def validate_pgvector_enabled(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "PGVECTOR_ENABLED=false is not supported. "
                "The schema contract requires pgvector-managed embeddings."
            )
        return value

    @field_validator('embedding_dim')
    @classmethod
    def validate_embedding_dim(cls, value: int) -> int:
        return require_document_chunk_vector_dim(value)

    @field_validator('live_judge_sampling_ratio')
    @classmethod
    def validate_live_judge_sampling_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError('LIVE_JUDGE_SAMPLING_RATIO must be between 0.0 and 1.0')
        return value

    def use_strict_database_mode(self) -> bool:
        """
        Explicit strict-mode toggle for DB startup behavior.
        If not provided, defaults to strict mode outside development-like environments.
        """
        if self.database_strict_mode is not None:
            return bool(self.database_strict_mode)
        return self.env.lower() not in ('development', 'dev', 'local')

    def use_semantic_cache(self) -> bool:
        if self.semantic_cache_enabled is not None:
            return bool(self.semantic_cache_enabled)
        if self.debug:
            return False
        return self.env.lower() not in ('development', 'dev', 'local', 'test', 'eval')

    def resolved_session_secret(self) -> str:
        """
        Resolve a safe session secret.
        - In development-like environments, generate an ephemeral secret if missing.
        - In non-development environments, require SESSION_SECRET to be set.
        """
        if self.session_secret and self.session_secret.strip():
            return self.session_secret.strip()

        if self.env.lower() in ('development', 'dev', 'local'):
            return secrets.token_urlsafe(48)

        raise RuntimeError(
            'SESSION_SECRET must be set in non-development environments.'
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
