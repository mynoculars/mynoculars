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

Python mechanics used in this file, if any of this is new to you:
    BaseSettings (from pydantic_settings)
        A special kind of Pydantic model (see state.py's module docstring
        for what a Pydantic model is) whose fields are populated FROM THE
        PROCESS ENVIRONMENT rather than from constructor arguments. Every
        field declared below, e.g. "llm_mode: Literal[...] = 'stub'",
        automatically reads the environment variable LLM_MODE (Pydantic
        upper-cases the field name) if it's set, and falls back to the
        written default ("stub") if it isn't. This is the ONLY place in the
        entire codebase that reads os.environ, directly or indirectly —
        every other file receives values through a Settings object, never
        by reading the environment itself.
    Literal["live", "stub"]
        A type hint meaning "this field's value must be exactly one of
        these strings" — see state.py's docstring for the same construct.
        Here it restricts llm_mode so a typo like LLM_MODE=Live (capital L)
        would be REJECTED at startup with a clear validation error, rather
        than silently misbehaving later.
    Field(0.6, ge=0.0, le=1.0)
        Field() lets you attach VALIDATION RULES to a value, not just a
        plain default. "0.6" is the default; ge/le mean "greater-or-equal"
        and "less-or-equal" — so llm_quality_threshold is required to be a
        number between 0.0 and 1.0 inclusive. Passing an out-of-range value
        via the environment raises a clear error at startup instead of
        misbehaving quietly deep inside the LLM router.
    @lru_cache (from functools, stdlib — nothing pydantic-specific)
        A decorator (see agents/gathering.py's module docstring for what a
        decorator is) that makes a function remember its own return value:
        the FIRST time get_settings() is called, it actually constructs a
        new Settings() object (which reads the environment); every
        subsequent call in the SAME process just returns that same cached
        object instantly, without re-reading the environment. This is what
        makes Settings effectively a "singleton" — one shared instance for
        the whole running process — without writing any singleton
        boilerplate by hand.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the process environment.

    Every field maps 1:1 to an environment variable of the same (uppercased)
    name. See .env.example for documentation of each value.

    Nothing in this class DOES anything by itself — it is a pure data
    container. Every field below is read somewhere else in the codebase
    (search the field name to find every call site); this class's only job
    is to be the one place that decides where each value comes from and
    what its default and valid range are.
    """

    # SettingsConfigDict is BaseSettings' equivalent of a normal Pydantic
    # model's ConfigDict (see state.py). env_file=".env" means: in addition
    # to real environment variables, also read a ".env" file in the current
    # working directory if one exists (handy for local development so you
    # don't have to `export` a dozen variables by hand). extra="ignore"
    # means an unrecognized environment variable (e.g. a typo like HITL=true
    # instead of HITL_ENABLED=true) is SILENTLY DISCARDED rather than
    # raising an error — worth knowing if a setting you set doesn't seem to
    # take effect: check the exact field name below first.
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
    # Fallback chain, in order: Mistral first, then Gemini. Each is included
    # in the chain only if its API key is set (see FallbackRouter.from_settings).
    llm_mistral_base_url: str = "https://api.mistral.ai/v1"
    llm_mistral_api_key: str = ""
    llm_mistral_model: str = "mistral-small-latest"
    llm_fallback_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = "gemini-2.0-flash"
    llm_quality_threshold: float = Field(0.6, ge=0.0, le=1.0)
    # Two SEPARATE timeouts, not one shared value (this used to be a single
    # llm_timeout_seconds applied to every provider identically — see
    # PHASE2_PLAN.md / a live debug trace for why that was wrong in
    # practice: a local llama-server-hosted model like Cogito can spend its
    # first request of a session just loading the model into memory before
    # it ever starts on your actual prompt, and its largest prompts (e.g.
    # compiler's ~4500-token report-writing call) can genuinely need more
    # wall-clock time than a fast cloud API ever would. Giving Cogito the
    # SAME short timeout as Mistral/Gemini meant it was being given up on
    # for being slow, not for being wrong — these two fields let it have
    # more room while keeping the cloud fallbacks quick to fail over.
    llm_primary_timeout_seconds: float = Field(120.0, gt=0.0)   # local Cogito
    llm_timeout_seconds: float = Field(90.0, gt=0.0)            # Mistral, Gemini

    # --- Storage endpoints -------------------------------------------------
    postgres_dsn: str = "postgresql://agent:agent@localhost:5432/agent"
    qdrant_url: str = "http://localhost:6333"
    opensearch_url: str = "http://localhost:9200"
    # OpenSearch auth/SSL (blank username = anonymous plain-HTTP, unchanged
    # default). Set these when your cluster has the security plugin enabled.
    opensearch_username: str = ""
    opensearch_password: str = ""
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    memory_collection: str = "agent_semantic_memory"
    corpus_index: str = "agent_corpus"

    # --- Graph bounds (see design doc D-3/D-4/D-9/D-13/D-17/D-22) ----------
    # ge=1 / ge=0 below means "must be greater-or-equal to 1 / 0" — same
    # Field() validation mechanism explained in the module docstring above.
    max_fanout: int = Field(6, ge=1)          # D-13: producers cap task output
    max_depth: int = Field(3, ge=1)           # D-3: gather-loop bound
    recall_target: float = Field(0.85, ge=0.0, le=1.0)   # D-4
    min_evidence_score: float = Field(0.0, ge=0.0)       # D-17: 0.0 = inert
    max_revisions: int = Field(2, ge=0)       # D-22: critique-loop bound
    # D-23: human-in-the-loop escalation. Off by default — the graceful-
    # degradation posture: shipping inert, enabled deliberately.
    hitl_enabled: bool = False
    recursion_limit: int = Field(60, ge=10)   # D-8: invoke-time backstop

    # --- Memory decay (D-24, Python-side reranking in this core build) -----
    memory_top_k: int = Field(5, ge=1)
    decay_half_life_days_semi_stable: float = 90.0
    decay_half_life_days_volatile: float = 14.0

    # Debug tracing: when true (or --debug on the CLI), dump the exact prompt,
    # raw response, provider, tokens and latency of every LLM call, plus every
    # retrieval engine's hits, to logs/trace-<run_id>.txt. Off by default.
    debug_trace: bool = False

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first load).

    CALLED BY   cli.py::build_app_and_settings (and cli.py::main, once,
                before the tracer is built), api/server.py at import time.
                Every other module that needs a config value receives a
                Settings OBJECT as a function/constructor argument (see
                agents/planning.py's closure explanation) rather than
                calling get_settings() itself — this function is the one
                and only place Settings() gets constructed.
    RETURNS     the same Settings instance on every call within one process
                (see the @lru_cache explanation in the module docstring).
    """
    return Settings()
