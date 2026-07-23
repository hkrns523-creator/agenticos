"""
Central, validated configuration for AgenticOS.

All tunables live here and are overridable via environment variables or a
local `.env` file. Using pydantic-settings (instead of a handful of
`os.getenv` calls) gives us type coercion, validation, and a single object
that's easy to override wholesale in tests (`Settings(db_path=...)`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTICOS_",
        extra="ignore",
    )

    # LLM backend 
    -----------------------------------------------------
    # "ollama" = local/self-hosted model (default, used for local dev/demo).
    # "groq" = hosted OpenAI-compatible endpoint, used for public deployment
    # where nothing is available to run Ollama continuously.
    llm_backend: Literal["ollama", "groq"] = Field(default="ollama")
    ollama_model: str = Field(default="qwen2.5:3b-instruct")
    ollama_base_url: str = Field(default="http://localhost:11434")
    groq_model: str = Field(default="llama-3.1-8b-instant")
    groq_api_key: str | None = Field(default=None)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_request_timeout: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_min_wait: float = Field(default=0.5, ge=0)
    llm_retry_max_wait: float = Field(default=4.0, ge=0)

    # Data layer --------------------------------------------------------
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "agenticos.db")
    docs_dir: Path = Field(default=PROJECT_ROOT / "docs")
    vector_db_dir: Path = Field(default=PROJECT_ROOT / "vector_db")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    rag_top_k: int = Field(default=3, ge=1, le=20)
    rag_enabled: bool = Field(default=True, description="Whether the documentation_agent (RAG) is registered and available to the planner.")

    #  RAG retrieval tuning ----------------------------------------------
    # Raw PDF pages are frequently too coarse a retrieval unit (a manual page
    # can cover several unrelated procedures); chunking gives the retriever
    # smaller, more topically-focused units to match against.
    rag_chunk_size: int = Field(default=1000, ge=100, description="Characters per chunk when splitting source documents.")
    rag_chunk_overlap: int = Field(default=150, ge=0, description="Character overlap between adjacent chunks.")
    # "mmr" (Maximal Marginal Relevance) trades a little pure-similarity for
    # diversity, avoiding a top_k list of near-duplicate chunks from the same
    # page — usually a better default for SOP/manual retrieval than plain
    # similarity search.
    rag_search_type: Literal["similarity", "mmr"] = Field(default="mmr")
    rag_fetch_k: int = Field(default=10, ge=1, description="Candidate pool size for MMR before reranking down to top_k.")
    rag_score_threshold: float | None = Field(
        default=None,
        description="Optional minimum similarity score (0-1) for a chunk to be returned; None disables filtering.",
    )

    # --- Conversation memory ------------------------------------------------
    memory_enabled: bool = Field(default=True, description="Persist and reuse conversation history across turns.")
    memory_max_turns: int = Field(default=6, ge=0, description="Number of most recent past turns fed back to the planner/supervisor as context.")
    memory_db_path: Path | None = Field(
        default=None,
        description="SQLite path for conversation history. Defaults to db_path (same database, separate table).",
    )

    # --- Runtime behavior --------------------------------------------------
    specialist_concurrency: bool = Field(
        default=True,
        description="Run specialist data-fetch nodes concurrently instead of sequentially.",
    )
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False, description="Emit structured JSON logs (production) vs. plain text (dev).")

    # --- API server (for containerized/AWS deployment) ----------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_cors_origins: list[str] = Field(default_factory=lambda: ["*"])


    @property
    def resolved_memory_db_path(self) -> Path:
        """`memory_db_path` if set, otherwise the same database `db_path` points at
        (conversation history lives in its own table, so sharing the file is fine)."""
        return self.memory_db_path or self.db_path


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Use `get_settings.cache_clear()` in tests
    that need to override environment variables mid-run."""
    return Settings()