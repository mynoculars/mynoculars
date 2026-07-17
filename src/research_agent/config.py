"""
config.py — Central configuration for the research agent.

Purpose:
    Load every tunable value from environment variables (.env supported),
    validate them once at startup, and expose a single typed Settings object.

Responsibilities:
    - Define all configuration keys with sensible defaults.
    - Fail fast with a clear error when required values are missing/invalid.
    - Never let secrets or endpoints be hardcoded anywhere else in the codebase.

Design decision (why pydantic-settings):
    One declarative class gives us env parsing, type coercion, defaults and
    validation in ~50 lines. Alternatives considered: raw os.environ (no
    validation, silent typos) and dynaconf (more features than a reference
    implementation needs). Tradeoff: adds a small dependency; worth it for
    the fail-fast behavior.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the process environment.

    Every field maps 1:1 to an environment variable of the same (uppercased)
    name. See .env.example for documentation of each value.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers -----------------------------------------------------
    # Both providers speak the OpenAI-compatible chat API. The primary is a
    # local llama-server hosting Qwen Cogito; the fallback is Gemini Flash via
    # Google's OpenAI-compatibility endpoint. One client class serves both —
    # a deliberate simplification (see llm/client.py header).
    llm_mode: Literal["live", "stub"] = "stub"
    llm_primary_base_url: str = "http://127.0.0.1:8080/v1"
    llm_primary_api_key: str = "not-needed-for-local"
    llm_primary_model: str = "qwen-cogito"
    llm_fallback_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = "gemini-2.0-flash"
    llm_quality_threshold: float = Field(0.6, ge=0.0, le=1.0)
    llm_timeout_seconds: float = 60.0

    # --- Storage endpoints -------------------------------------------------
    postgres_dsn: str = "postgresql://agent:agent@localhost:5432/agent"
    qdrant_url: str = "http://localhost:6333"
    opensearch_url: str = "http://localhost:9200"
    memory_collection: str = "agent_semantic_memory"
    corpus_index: str = "agent_corpus"

    # --- Graph bounds (see design doc D-3/D-4/D-9/D-13/D-17/D-22) ----------
    max_fanout: int = Field(6, ge=1)          # D-13: producers cap task output
    max_depth: int = Field(3, ge=1)           # D-3: gather-loop bound
    recall_target: float = Field(0.85, ge=0.0, le=1.0)   # D-4
    min_evidence_score: float = Field(0.0, ge=0.0)       # D-17: 0.0 = inert
    max_revisions: int = Field(2, ge=0)       # D-22: critique-loop bound
    recursion_limit: int = Field(60, ge=10)   # D-8: invoke-time backstop

    # --- Memory decay (D-24, Python-side reranking in this core build) -----
    memory_top_k: int = Field(5, ge=1)
    decay_half_life_days_semi_stable: float = 90.0
    decay_half_life_days_volatile: float = 14.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first load)."""
    return Settings()
