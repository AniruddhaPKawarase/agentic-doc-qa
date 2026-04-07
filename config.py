"""
Configuration — pydantic-settings with .env file support.
──────────────────────────────────────────────────────────────────────────────
Single source of truth for all tunables. Access via `get_settings()`.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────
    docqa_host: str = "0.0.0.0"
    docqa_port: int = 8006

    # ── OpenAI ────────────────────────────────────────
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o"  # Vision model for image/drawing analysis (gpt-4o has better vision than gpt-4o-mini)
    openai_max_output_tokens: int = 2048

    # ── Retrieval ─────────────────────────────────────
    retrieval_top_k: int = 8
    retrieval_score_threshold: float = 0.15
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    embedding_batch_size: int = 100
    max_context_tokens: int = 80000

    # ── Upload ────────────────────────────────────────
    max_file_size_mb: int = 20
    max_files_per_upload: int = 10
    allowed_extensions: str = ".pdf,.docx,.xlsx,.xls,.txt,.csv,.json,.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.tif,.xml,.html,.htm,.svg,.md,.yaml,.yml,.log,.ini,.cfg"

    # ── Session ───────────────────────────────────────
    session_ttl_hours: int = 168
    max_history_messages: int = 50
    session_cleanup_interval_min: int = 30

    # ── Hallucination Guard ───────────────────────────
    hallucination_guard_enabled: bool = True
    groundedness_threshold: float = 0.35
    min_followup_questions: int = 3

    # ── Cache ─────────────────────────────────────────
    cache_l1_maxsize: int = 500
    cache_l1_ttl: int = 3600
    redis_url: str = ""
    cache_semantic_enabled: bool = True

    # ── Token Budget ──────────────────────────────────
    max_input_tokens: int = 100000
    token_warning_threshold: float = 0.85

    # ── Cookie (for /api/converse unified endpoint) ────
    cookie_secure: bool = False

    # S3 Storage Migration (Phase 7)
    storage_backend: str = "local"
    s3_bucket_name: str = ""
    s3_region: str = "us-east-1"
    s3_agent_prefix: str = "document-qa-agent"
    s3_restore_sessions_on_startup: bool = True
    s3_max_restore_sessions: int = 50

    # ── v2 Pipeline ──────────────────────────────────────
    pipeline_version: str = "v2"
    primary_model: str = "gpt-4o"
    secondary_model: str = "gpt-4o-mini"
    primary_max_output_tokens: int = 4096

    # ── v2 Feature Flags ─────────────────────────────────
    enable_full_context: bool = True
    enable_bm25: bool = True
    enable_summary_generation: bool = True
    enable_llm_guard: bool = True

    # ── v2 Context Strategy ──────────────────────────────
    full_context_threshold: int = 80000
    summary_threshold: int = 200000
    bm25_weight: float = 0.30

    # ── v2 Hallucination Guard ───────────────────────────
    guard_general_threshold: float = 0.20
    guard_specific_threshold: float = 0.30
    guard_marginal_low: float = 0.25
    guard_marginal_high: float = 0.50

    # ── v2 Monitoring ────────────────────────────────────
    enable_metrics: bool = True
    alert_latency_warning_ms: int = 15000
    alert_latency_critical_ms: int = 30000
    alert_cost_warning_usd: float = 0.50
    alert_cost_critical_usd: float = 1.00
    alert_groundedness_warning: float = 0.40
    alert_groundedness_critical: float = 0.20

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
