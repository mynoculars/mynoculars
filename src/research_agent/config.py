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

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


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
    # D-17: coverage gate. 0.0 was the ORIGINAL default, and it made this
    # check inert — `score >= 0.0` is true for every item ever returned,
    # including one scored exactly 0.0. Combined with the fact that a dense
    # index always returns its top-k nearest neighbours regardless of actual
    # relevance, recall degenerated to "did ANY document come back", which
    # is why the HITL escalation paths (E2/E3) could never fire while
    # retrieval was up. 0.5 is a starting point based on a real debug trace
    # showing off-topic corpus hits landing at 0.48-0.50 after the
    # RRF_SQUASH scaling in tools/corpus_search.py — NOT independently
    # calibrated against every corpus. Tune against your own data.
    min_evidence_score: float = Field(0.5, ge=0.0)
    # P2-01, the other half of the same fix: a floor applied at RETRIEVAL
    # time (retrieval/hybrid.py), before a hit ever becomes Evidence at
    # all — as opposed to min_evidence_score above, which filters AFTER
    # retrieval, at the coverage-checking step. Without this, a dense
    # index still always returns its k nearest neighbours no matter how
    # irrelevant, and min_evidence_score alone was the only gate; this
    # adds a second, earlier one so an out-of-domain query can actually
    # produce zero evidence rather than three confident wrong answers.
    min_similarity: float = Field(0.35, ge=0.0, le=1.0)
    max_revisions: int = Field(2, ge=0)       # D-22: critique-loop bound
    # D-23: human-in-the-loop escalation. Off by default — the graceful-
    # degradation posture: shipping inert, enabled deliberately.
    hitl_enabled: bool = False
    recursion_limit: int = Field(60, ge=10)   # D-8: invoke-time backstop
    # D-18/P2-12: LLM-based contradiction detection in merger_node. Off by
    # default — costs one extra LLM call per merger execution (i.e. once
    # per gather cycle, up to max_depth times per run) whenever any goal
    # has 2+ evidence items. When off, merger_node keeps the ORIGINAL
    # marker-only behaviour (honours an explicit Evidence.contradicts, which
    # no tool in this build sets — see agents/gathering.py). Turning this
    # on is what makes E2 reachable in a real run for the first time.
    contradiction_detection_enabled: bool = False

    # --- Memory decay (D-24/D-27) -------------------------------------------
    memory_top_k: int = Field(5, ge=1)
    decay_half_life_days_semi_stable: float = 90.0
    decay_half_life_days_volatile: float = 14.0
    # P2-10: server-side (Qdrant FormulaQuery) decay reranking instead of
    # the Python over-fetch-then-rerank path. Off by default — requires
    # points to carry the "created_at_iso" payload field, which only
    # points written AFTER this shipped will have (see
    # storage/qdrant_store.py::ensure_payload_indexes's docstring). Points
    # from before this flag existed will simply never match the server-side
    # formula's datetime filter; wipe and re-ingest (scripts/reset_stores.py)
    # rather than trying to run mixed old/new points through this path.
    # The Python decay_factor() path (memory/semantic_memory.py) remains
    # the default AND the permanent parity oracle — never deleted.
    memory_server_side_decay: bool = False

    # --- MCP tool seam (P2-13, implements D-26/D-30) ------------------------
    # Off by default: corpus_search.py (the original function-registry tool)
    # remains cli.py's DEFAULT worker regardless of this flag. When enabled,
    # cli.py ADDITIONALLY wires tools/mcp_client.py's tool in as a second,
    # addressable specialist (P2-14, D-25) -- reachable only via a task's
    # tool_hint == "mcp" (orchestration/graph.py::dispatch_tasks); the
    # default corpus worker keeps handling every task without a hint. This
    # is also the ONLY signal task_expander_node/gap_generator_node use to
    # decide whether the LLM is even told "mcp" is an option to hint at
    # (see agents/planning.py and agents/gathering.py) -- no separate
    # "which hints are available" setting exists, deliberately, since
    # "mcp" is the only specialist this build can ever wire in.
    mcp_enabled: bool = False
    # The command to launch a LOCAL MCP server over stdio (D-30 -- the only
    # transport this build implements; never SSE, which D-30 prohibits
    # outright). Empty string is a deliberately invalid default: turning
    # mcp_enabled on without also setting a real command is a configuration
    # mistake this should surface early, not silently do nothing.
    mcp_server_command: str = ""
    # Comma-separated argv for the command above (e.g. "-m,my_mcp_server").
    # A plain string, not a List[str] Settings field, so a person editing
    # .env doesn't need to know pydantic-settings' JSON-for-lists env
    # convention -- split_and_strip below does the parsing this codebase's
    # other comma-separated-feeling settings don't otherwise need.
    mcp_server_args: str = ""
    # Comma-separated env VAR NAMES allowed to reach the MCP server
    # subprocess (D-30: never a blanket os.environ passthrough -- see
    # tools/mcp_client.py::_build_subprocess_env for exactly why that
    # matters). Empty by default: an MCP server gets NO inherited
    # environment variables unless each one is explicitly named here.
    mcp_server_env_allowlist: str = ""
    # The server-side tool name this build calls (a server can expose more
    # than one tool; this build only ever calls exactly one).
    mcp_tool_name: str = "search"
    # The argument name the server's tool schema expects for the search
    # string -- not every server will call it "query".
    mcp_query_arg_name: str = "query"
    mcp_call_timeout_seconds: float = 30.0
    mcp_max_workers: int = Field(6, ge=1)
    # Debug tracing: when true (or --debug on the CLI), dump the exact prompt,
    # raw response, provider, tokens and latency of every LLM call, plus every
    # retrieval engine's hits, to logs/trace-<run_id>.txt. Off by default.
    debug_trace: bool = False

    log_level: str = "INFO"


# P2-09: known-typo list. Each key here is a plausible mistyped env var name
# someone might set, mapped to the CORRECT field-backed name they probably
# meant. Settings.model_config uses extra="ignore" (see above), so any of
# these left-hand names would previously be silently discarded with NO
# indication the intended setting never took effect — e.g. HITL=true in
# .env does nothing; HITL_ENABLED=true is the real field. This is a fixed,
# reviewed list (not fuzzy matching) — deliberately conservative: it only
# warns on names someone plausibly typed, never blocks a genuinely unknown
# environment variable a shell might have set for unrelated reasons.
_KNOWN_ENV_TYPOS = {
    "HITL": "HITL_ENABLED",
    "LLM_TIMEOUT": "LLM_TIMEOUT_SECONDS",
    "LLM_PRIMARY_TIMEOUT": "LLM_PRIMARY_TIMEOUT_SECONDS",
    "MIN_EVIDENCE": "MIN_EVIDENCE_SCORE",
    "MIN_SIM": "MIN_SIMILARITY",
    "RECALL": "RECALL_TARGET",
    "FANOUT": "MAX_FANOUT",
    "DEPTH": "MAX_DEPTH",
    "REVISIONS": "MAX_REVISIONS",
    "DEBUG": "DEBUG_TRACE",
    "MEMORY_TOPK": "MEMORY_TOP_K",
    "CONTRADICTION_DETECTION": "CONTRADICTION_DETECTION_ENABLED",
    "SERVER_SIDE_DECAY": "MEMORY_SERVER_SIDE_DECAY",
    "MCP": "MCP_ENABLED",
    "MCP_COMMAND": "MCP_SERVER_COMMAND",
    "MCP_ARGS": "MCP_SERVER_ARGS",
    "MCP_ENV_ALLOWLIST": "MCP_SERVER_ENV_ALLOWLIST",
}


def split_csv(value: str) -> list:
    """Split a comma-separated Settings string into a clean list.

    CALLED BY   cli.py::build_app_and_settings, for
                settings.mcp_server_args and
                settings.mcp_server_env_allowlist -- both plain comma-
                separated strings rather than pydantic-settings' native
                List[str] fields (see config.py's MCP settings comments
                for why: a person editing .env by hand shouldn't need to
                know pydantic-settings' JSON-for-lists env convention).
    Empty entries (from a trailing comma, double comma, or an entirely
    empty input string) are dropped, and every remaining entry is
    whitespace-stripped -- "a, b ,,c" -> ["a", "b", "c"].
    """
    return [part.strip() for part in value.split(",") if part.strip()]


def warn_on_likely_env_typos() -> None:
    """Log a WARNING for every real environment variable matching a known
    typo pattern (P2-09).

    CALLED BY   get_settings(), below, once per process (immediately after
                Settings() is constructed) — this never blocks startup,
                it only makes a previously-silent misconfiguration visible
                in the log the first time it happens.
    READS       os.environ directly — the one exception to config.py's own
                rule that this is the only place the process environment
                is read (see the module docstring); Settings itself never
                sees these keys at all, precisely because they don't match
                any declared field, which is the whole problem being
                flagged here.
    """
    for wrong, right in _KNOWN_ENV_TYPOS.items():
        if wrong in os.environ and right not in os.environ:
            log_event(logger, "config.likely_typo", level=logging.WARNING,
                      set_key=wrong, probably_meant=right)


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
    warn_on_likely_env_typos()  # P2-09: surface likely misconfiguration
    return Settings()
