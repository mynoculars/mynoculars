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
import pathlib
import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_agent.logging_setup import configure_logging, log_event

logger = logging.getLogger(__name__)

# The repository root, derived from THIS FILE's location and never from the
# current working directory:
#
#     <repo>/src/research_agent/config.py  ->  parents[2] == <repo>
#
# Why this exists (Phase 4 / D-58), and why CWD is not good enough:
# tools/mcp_client.py::MCPBridge builds an mcp.StdioServerParameters WITHOUT
# setting its `cwd` field, so the server subprocess inherits the CWD of
# whichever process launched the agent. Any relative path in MCP_SERVER_ARGS
# or WEB_MCP_SERVER_COMMAND is therefore resolved against THAT directory,
# not against the repo.
#
# Verified directly rather than assumed, by spawning real subprocesses:
#
#   command=<abs>  args=["scripts/x.py"]  cwd=<repo>  -> OK
#   command=<abs>  args=["scripts/x.py"]  cwd=/tmp    -> FAIL, McpError
#   command="./python"                    cwd=<repo>  -> OK
#   command="./python"                    cwd=/tmp    -> FAIL, FileNotFoundError
#
# So the shipped `MCP_SERVER_ARGS=scripts\\mcp_corpus_server.py` has always
# worked only because every documented invocation happens to start in the
# repo root. Running the CLI from anywhere else, or under a service manager,
# scheduled task or IDE that sets its own working directory, breaks it -- and
# breaks it as an opaque "Connection closed", never as "file not found".
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str) -> str:
    """Resolve a possibly-relative path against REPO_ROOT, not the CWD.

    CALLED BY   assembly.py, for BOTH MCP bridges' command and args.

    Rules, in order:

    1. Empty stays empty. "is this configured at all" is the caller's
       decision to make, not this function's.
    2. Backslashes are normalized to forward slashes FIRST. The shipped
       .env.example uses Windows separators, and pathlib on POSIX treats
       "scripts\\x.py" as a single filename containing a backslash rather
       than as a path -- so the same .env would silently behave differently
       on the two platforms. Normalizing makes one .env portable across
       both. Safe because no path this project references legitimately
       contains a backslash in a filename.
    3. An ABSOLUTE path is returned unchanged. Someone who wrote an
       absolute path meant it.
    4. A relative path is resolved against REPO_ROOT -- but ONLY IF the
       result actually EXISTS. Otherwise the original string is returned
       untouched.

       That existence check is what keeps a BARE COMMAND NAME working.
       "python3", "uvx" and "npx" are not paths at all; they are names the
       OS resolves through PATH. <repo>/python3 does not exist, so the name
       passes through unchanged and PATH lookup still happens. Blindly
       prefixing REPO_ROOT would turn every bare command into a
       guaranteed FileNotFoundError.

    Deliberately does NOT raise on a path that resolves nowhere. A wrong
    path is the caller's problem to report with its own context -- MCPBridge
    already produces a clear error naming the command, and
    scripts/check_services.py exists to surface exactly this before a real
    run. Raising here would turn a configuration mistake into an import-time
    crash of the whole application.
    """
    if not value:
        return value
    normalized = value.replace("\\", "/")
    candidate = pathlib.Path(normalized)
    if candidate.is_absolute():
        return str(candidate)
    resolved = REPO_ROOT / candidate
    return str(resolved) if resolved.exists() else value


def resolve_server_command(value: str) -> str:
    """The command to launch an MCP server subprocess with.

    CALLED BY   assembly.py, for both bridges.

    AN EMPTY VALUE MEANS sys.executable -- the interpreter currently running
    the agent -- and that is the RECOMMENDED configuration, not a fallback
    for the careless.

    Why empty-means-sys.executable is the most portable answer available:

      - It is correct by construction on every machine. No absolute path to
        get wrong, no path to update when a colleague clones to a different
        drive, no CI runner needing its own override, no difference between
        Windows and POSIX layouts (Scripts/python.exe vs bin/python).
      - It GUARANTEES the server runs in the same virtualenv as the agent.
        That matters concretely here: scripts/mcp_web_search_server.py
        imports `ddgs`, and a bare "python" resolved through PATH can easily
        be a system interpreter that has never seen this project's
        dependencies -- failing as an opaque "Connection closed" rather than
        as a readable ImportError, because the subprocess dies before the
        MCP handshake completes.
      - It keeps the checked-in .env.example free of machine-specific
        absolute paths, which is the thing that makes a config file
        non-portable in the first place.

    A non-empty value is honoured and passed through resolve_repo_path, so
    an absolute path, a repo-relative path (".venv/Scripts/python.exe") and
    a bare PATH name ("python3") all work. Use one only when the server
    genuinely must run under a DIFFERENT interpreter than the agent -- a
    real case, just not the common one.
    """
    if not value or not value.strip():
        return sys.executable
    return resolve_repo_path(value.strip())


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
    # Separate timeouts per tier (D-54's llm_max_tokens neighbor): a local
    # llama-server-hosted model can spend its first request just loading
    # into memory, and the compiler's largest prompts need real wall-clock
    # time no fast cloud API would — giving it the same short timeout as
    # the cloud fallbacks meant it was given up on for being slow, not
    # wrong.
    llm_primary_timeout_seconds: float = Field(120.0, gt=0.0)   # local Cogito
    llm_timeout_seconds: float = Field(90.0, gt=0.0)            # Mistral, Gemini
    llm_max_tokens: int = Field(4096, ge=1)  # D-54: request-level generation cap


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
    # M-3: a separate, lower floor for what may enter long-term memory
    # (memory/semantic_memory.py::store_run), decoupled from
    # min_evidence_score above. Single-leg RRF fusion caps a score at
    # exactly SINGLE_LEG_SCORE_CEILING (0.5, guardrails/retrieval.py), so
    # gating memory writes at the SAME floor as coverage made every
    # single-leg hit ineligible for memory. Tune alongside
    # min_evidence_score if you change either.
    memory_write_min_score: float = Field(0.4, ge=0.0)
    # The other half of the same fix: a floor at RETRIEVAL time
    # (retrieval/hybrid.py), before a hit ever becomes Evidence — without
    # it a dense index still returns its k nearest neighbours regardless
    # of relevance, and min_evidence_score alone was the only gate.
    min_similarity: float = Field(0.35, ge=0.0, le=1.0)
    retrieval_floor_warn_ratio: float = Field(0.8, ge=0.0, le=1.0)  # D-48
    quality_judge_warn_ratio: float = Field(0.5, ge=0.0, le=1.0)  # D-53
    max_revisions: int = Field(2, ge=0)       # D-22: critique-loop bound
    # Guardrail G2: the OTHER half of convergence, alongside recall_target.
    # recall_target answers "is every goal covered by SOMETHING"; this
    # answers "how much of that coverage is a real document, not the
    # model's own recollection" — route_convergence only accepts full
    # convergence when both clear their floor (D-47). 0.5 is a starting
    # point; tune against your own corpus like min_evidence_score.
    grounded_recall_target: float = Field(0.5, ge=0.0, le=1.0)
    # D-38: the retrieval escalation ladder's terminal tier. On by default
    # — a missing corpus document is a retrieval limitation, not an
    # absence of knowledge, and reporting the former as the latter was
    # this system's single worst failure mode. False restores
    # corpus-only behaviour.
    model_knowledge_enabled: bool = True
    # Must exceed min_evidence_score or model evidence can never mark a
    # goal covered and the gather loop cannot converge; must stay well
    # below the ~1.0 of a document both retrieval legs agreed on so a real
    # document always outranks recollection.
    model_knowledge_score: float = Field(0.6, ge=0.0, le=1.0)
    query_reformulation_enabled: bool = True
    # D-23: human-in-the-loop escalation. Off by default — the graceful-
    # degradation posture: shipping inert, enabled deliberately.
    hitl_enabled: bool = False
    # CURRENTLY UNREAD -- kept, not deleted, and labelled rather than left
    # looking live. No node, tool or producer consults it; text-searching
    # this field name across the repo returns only this line. The INTENT
    # was: how many times one query formulation may be retried across
    # later gather cycles once every tier came back empty, with D-16's
    # depth gate still applying on top. Either wire it or drop it -- but
    # not silently.
    max_task_retries: int = Field(2, ge=0)
    # D-23 (bound): the maximum number of times ONE run may pause for a
    # human. Without this, HITL removes the loop bounds it is layered on
    # top of -- route_convergence and dispatch_tasks both test
    # escalation_trigger before their normal exits, so a re-raised E2/E3
    # re-enters human_escalation instead of terminating, and an unbounded
    # human-nagging loop ends only at recursion_limit (exit code 2, no
    # report). Once this budget is spent, the raising nodes take their
    # existing HITL-OFF branch instead: log loudly and fall through to
    # the compiler with whatever WAS retrieved. 0 disables pausing
    # entirely without needing to also flip hitl_enabled.
    max_escalations: int = Field(2, ge=0)
    recursion_limit: int = Field(60, ge=10)   # D-8: invoke-time backstop
    run_call_budget_warn: int = Field(40, ge=1)  # D-54: observational only
    # D-18/P2-12: LLM-based contradiction detection in merger_node. Off by
    # default — costs one extra LLM call per merger execution (up to
    # max_depth times per run) whenever any goal has 2+ evidence items.
    # When off, merger_node keeps the ORIGINAL marker-only behaviour
    # (honours an explicit Evidence.contradicts, which no tool in this
    # build sets). Turning this on is what makes E2 reachable in a real
    # run for the first time.
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

    # --- Phase 4: web search (D-57) ---------------------------------------
    # The gap this closes: corpus_search, mcp_client and semantic memory all
    # resolve to the SAME ingested documents, so "the corpus does not
    # contain it" had nowhere left to escalate except the model's own
    # recollection (D-38 tier 4). Web search is a real retrieval tier
    # between those two.
    #
    # EVERY SETTING IN THIS BLOCK IS READ BY THE SEARCH SERVER SUBPROCESS
    # (scripts/mcp_web_search_server.py), not by the agent. The agent never
    # imports research_agent.websearch and never issues a search request of
    # its own -- it talks to the server over stdio through
    # tools/mcp_client.py, exactly as it already does for the corpus MCP
    # server. The settings that wire the AGENT side (WEB_MCP_*) are a
    # separate block, added with that wiring.
    #
    # The server reads these from ITS OWN environment/.env at startup (same
    # as scripts/mcp_corpus_server.py does with get_settings()), which is
    # why WEB_MCP_SERVER_ENV_ALLOWLIST can stay minimal: the subprocess does
    # not need these forwarded to it.
    web_search_provider: str = "ddgs"
    # How many results to ask the engine for per query. Modest on purpose:
    # DDGS is an unofficial client against an endpoint that promises it
    # nothing, and aggressive querying is what triggers throttling. Also
    # bounds how much untrusted third-party text can enter one compile
    # prompt at a time.
    web_search_max_results: int = Field(5, ge=1, le=25)
    # The score band a rank is mapped onto (websearch/scoring.py).
    #
    # FLOOR must EXCEED min_evidence_score, or the tier is inert: D-17's
    # coverage predicate is a strict `>`, so a web hit scoring at or below
    # the floor can never mark a goal covered, and the whole feature runs
    # while contributing nothing -- the same silent-inertness failure
    # MIN_EVIDENCE_SCORE=0.0 was, and that make_mcp_tool's `unscored_score`
    # parameter exists to prevent from the other direction.
    # warn_on_web_search_band below WARNs when it does not.
    #
    # CEILING must stay well below the ~1.0 a document both retrieval legs
    # ranked first reaches after tools/corpus_search.py's RRF_SQUASH.
    # D-38's ordering invariant is that a real document always beats weaker
    # provenance; a snippet allowed to score 0.95 would sit above genuinely
    # fused corpus evidence in the compiler's context and invert it. 0.75
    # also sits above model_knowledge_score (0.60), which is the intended
    # order: a live retrieved snippet is better provenance than
    # recollection, worse than a curated document.
    web_search_min_score: float = Field(0.60, ge=0.0, le=1.0)
    web_search_max_score: float = Field(0.75, ge=0.0, le=1.0)
    # At most this many hits from any one registrable domain
    # (websearch/filtering.py). Not tidiness: five hits from one site read
    # to the compiler as five independent sources agreeing -- corroboration
    # that does not exist -- and telemetry's web_search_results count looks
    # identical either way. 0 or less disables the cap, the documented way
    # to reproduce uncapped behaviour deliberately (same posture as
    # min_similarity=0.0 for pre-P2-01 retrieval).
    web_search_max_per_domain: int = 2
    # Engine-specific request shaping. Configurable rather than hardcoded
    # for the same reason every other endpoint in this file is: a run from
    # Bengaluru and a run from Frankfurt should not be forced to one result
    # set. "wt-wt" is DDGS's own no-region default.
    web_search_region: str = "wt-wt"
    web_search_safesearch: str = "moderate"
    # Per-HTTP-request timeout INSIDE the provider. Distinct from the MCP
    # call timeout on the agent side: this one stops a single hung request
    # from occupying a thread-pool slot in the server forever; that one
    # stops the agent waiting on a wedged server subprocess. Both are
    # needed -- neither substitutes for the other.
    web_search_provider_timeout_seconds: float = Field(20.0, gt=0.0)
    # The search server's own thread pool, mirroring mcp_max_workers for the
    # corpus server. Separate field, not a reuse: these pools front totally
    # different work (outbound internet vs. local Qdrant/OpenSearch) and
    # sizing one should never silently resize the other.
    web_search_max_workers: int = Field(6, ge=1)

    # --- Phase 4: the AGENT side of web search (D-57) ---------------------
    # Everything above is read by the search server subprocess. Everything
    # HERE is read by assembly.py, in the agent process, to wire the second
    # MCPBridge.
    #
    # A SECOND, SEPARATE NAMESPACE, not a reuse of MCP_*. The existing MCP_*
    # block describes exactly one server, and in every deployment of this
    # repo so far that server is scripts/mcp_corpus_server.py -- i.e. the
    # corpus, reached a second way. Web search is a genuinely different
    # server, with a different command, a different tool name, a different
    # timeout profile (outbound internet, not local stores) and a different
    # env allowlist. Overloading one namespace would force a choice between
    # them; two namespaces let both run at once, which is the point.
    #
    # Off by default, same posture as MCP_ENABLED and HITL_ENABLED: no
    # surprise egress and no surprise third-party text in a prompt. With
    # this false, assembly.py builds no second bridge, make_web_search_tool
    # is never called, and the retrieval ladder is byte-identical to every
    # run before Phase 4.
    web_search_enabled: bool = False
    # EMPTY IS THE RECOMMENDED VALUE, not an unset one: resolve_server_command
    # (above) turns it into sys.executable, the interpreter already running
    # the agent. That is correct on every machine with no configuration at
    # all, and it guarantees the server runs in the SAME virtualenv -- which
    # matters because the server imports ddgs, and a wrong interpreter dies
    # before the MCP handshake and surfaces as "Connection closed" rather
    # than as a readable ImportError.
    #
    # Set this only when the server must genuinely run under a DIFFERENT
    # interpreter than the agent. Absolute paths, repo-relative paths and
    # bare PATH names are all accepted -- see resolve_server_command.
    web_mcp_server_command: str = ""
    # Relative to the REPO ROOT, resolved by resolve_repo_path -- NOT to the
    # current working directory, which is what mcp.StdioServerParameters
    # would otherwise use (D-58).
    web_mcp_server_args: str = "scripts/mcp_web_search_server.py"
    # THE ENTRY THAT MATTERS, and the reason this defaults NON-EMPTY while
    # mcp_server_env_allowlist defaults empty: this subprocess makes
    # OUTBOUND INTERNET requests. tools/mcp_client.py::_build_subprocess_env
    # never forwards os.environ (D-30), so behind a corporate proxy the
    # server would receive no HTTPS_PROXY/HTTP_PROXY/NO_PROXY and every
    # search would fail as an opaque timeout with nothing in the log
    # explaining why. The corpus server needs none of this because it talks
    # only to Qdrant/OpenSearch on URLs it reads from its own .env.
    #
    # Naming a variable here does NOT leak it unless it is actually set in
    # this process -- see _build_subprocess_env. On a machine with no proxy
    # configured this default forwards nothing at all.
    web_mcp_server_env_allowlist: str = "HTTPS_PROXY,HTTP_PROXY,NO_PROXY"
    web_mcp_tool_name: str = "web_search"
    web_mcp_query_arg_name: str = "query"
    # Higher than mcp_call_timeout_seconds' 30.0 default: a corpus lookup
    # is a local round trip, a web search is a third-party HTTP request that
    # may be retried or throttled upstream. Still bounded -- the ladder must
    # be able to give up on this tier and reach the model tier within a
    # sensible wall-clock budget.
    web_mcp_call_timeout_seconds: float = 45.0
    # Debug tracing: when true (or --debug on the CLI), dump the exact prompt,
    # raw response, provider, tokens and latency of every LLM call, plus every
    # retrieval engine's hits, to logs/trace-<run_id>.txt. Off by default.
    debug_trace: bool = False

    log_level: str = "INFO"

    # Phase 3 -- Langfuse observability. Off by default; when off, the
    # langfuse/ package never imports the langfuse SDK, never constructs a
    # client, and makes zero network calls (see langfuse/client.py). This
    # config block only PARSES the env vars -- every actual SDK call lives
    # inside research_agent/langfuse/, never here and never in any business
    # module. LANGFUSE_HOST is deliberately a plain configurable string, not
    # a Literal of known hosts, so this same code works unmodified against
    # Langfuse Cloud, a self-hosted instance, or an enterprise install --
    # only the URL changes.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_project: str = ""
    langfuse_environment: str = "development"
    langfuse_release: str = ""
    # What leaves the process for Langfuse. "patterns" (the default)
    # redacts four distinctive PII/secret shapes and keeps everything
    # else; "all" replaces every string leaf; "off" sends payloads
    # verbatim. Defaults to ON deliberately: an observability feature
    # that ships inert is the same mistake min_evidence_score=0.0 was.
    # The redaction itself lives in langfuse/masking.py.
    langfuse_mask_mode: str = "patterns"
    # What to send Langfuse as the cost of a generation whose configured
    # rates work out to exactly $0. "explicit" (the default) asserts the
    # zero, which is correct for a genuinely free local model but also
    # OVERRIDES Langfuse's own model pricing table for a cloud provider
    # whose rate was simply never set. "infer" omits the cost instead and
    # lets Langfuse price it from the model name. Code cannot tell those
    # two cases apart -- see langfuse/pricing.py::resolve_cost_mode.
    langfuse_cost_mode: str = "explicit"
    # Per-provider $ cost per 1M tokens, input/output. Never hardcoded in
    # the SDK-facing code (see langfuse/pricing.py) -- a provider with no
    # entry here costs $0, which is the correct default for a local model
    # and an honest "unknown" for anything else rather than a silently
    # wrong guessed number.
    langfuse_price_primary_in_per_1m: float = 0.0
    langfuse_price_primary_out_per_1m: float = 0.0
    langfuse_price_mistral_in_per_1m: float = 0.0
    langfuse_price_mistral_out_per_1m: float = 0.0
    langfuse_price_gemini_in_per_1m: float = 0.0
    langfuse_price_gemini_out_per_1m: float = 0.0


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
    "WEB_SEARCH": "WEB_SEARCH_ENABLED",
    "WEB_SEARCH_SCORE": "WEB_SEARCH_MIN_SCORE",
    "WEB_SEARCH_RESULTS": "WEB_SEARCH_MAX_RESULTS",
    "WEB_SEARCH_TIMEOUT": "WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS",
    "WEB_MCP": "WEB_MCP_SERVER_COMMAND",
    "WEB_MCP_COMMAND": "WEB_MCP_SERVER_COMMAND",
    "WEB_MCP_ARGS": "WEB_MCP_SERVER_ARGS",
    "WEB_MCP_ENV_ALLOWLIST": "WEB_MCP_SERVER_ENV_ALLOWLIST",
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
                Settings() is constructed and configure_logging() has run,
                S-11 — so this and the other two warn_on_* checks log
                against an already-configured root logger).
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


def warn_on_inert_coverage_gate(s: "Settings") -> None:
    """Log a WARNING when settings would make the D-17 coverage gate inert.

    CALLED BY   get_settings(), below, once per process.
    WHY THIS EXISTS: min_evidence_score defaults to 0.5 here, but a .env
    carrying the ORIGINAL 0.0 silently reverts P2-01 entirely — the
    coverage predicate in agents/gathering.py::progress_checker_node
    (`e.score > settings.min_evidence_score`) then passes essentially every
    hit a dense index returns, recall pins at 1.0 on the first cycle, the
    gap loop never runs, and E2/E3 become unreachable. That is a config
    value undoing a code fix, and it is completely silent: the run looks
    like a fast, confident success. Same for min_similarity, the
    retrieval-time half of the same two-stage gate.

    A warning, never a hard failure: 0.0 stays legal (it is the documented
    way to reproduce pre-P2-01 behaviour deliberately), it just can no
    longer happen without anyone noticing.
    """
    if s.min_evidence_score <= 0.0:
        log_event(logger, "config.coverage_gate_inert", level=logging.WARNING,
                  setting="MIN_EVIDENCE_SCORE", value=s.min_evidence_score,
                  effect="every retrieved item satisfies coverage; recall "
                         "will pin at 1.0 and E2/E3 cannot fire")
    if s.min_similarity <= 0.0:
        log_event(logger, "config.relevance_floor_inert", level=logging.WARNING,
                  setting="MIN_SIMILARITY", value=s.min_similarity,
                  effect="every dense hit enters fusion regardless of relevance")


def warn_on_web_search_band(s: "Settings") -> None:
    """Log a WARNING when the web-search score band is misconfigured.

    CALLED BY   get_settings(), below, once per process -- alongside
                warn_on_likely_env_typos and warn_on_inert_coverage_gate,
                which this deliberately mirrors in both shape and posture.

    Two distinct misconfigurations, two distinct silent failures:

    1. FLOOR AT OR BELOW THE COVERAGE GATE. D-17's predicate
       (agents/gathering.py::progress_checker_node) is a strict
       `e.score > settings.min_evidence_score`. With
       web_search_min_score <= min_evidence_score, not one web result can
       ever mark a goal covered -- so the tier fires, spends real network
       time, returns real evidence, and contributes nothing to convergence.
       The run looks normal. This is the same class of config-undoing-code
       defect warn_on_inert_coverage_gate exists for.

    2. INVERTED BAND. floor > ceiling would mean the engine's BEST result
       scores lowest. websearch/scoring.py::rank_to_score normalizes this
       rather than running backwards, so it is not a correctness bug --
       but it is certainly not what anyone meant, and silently doing the
       right thing with the wrong config teaches nothing.

    A warning, never a hard failure -- same reasoning as every other check
    in this file: an unusual value stays legal, it just cannot happen
    without anyone noticing.
    """
    if s.web_search_min_score <= s.min_evidence_score:
        log_event(logger, "config.web_search_tier_inert", level=logging.WARNING,
                  setting="WEB_SEARCH_MIN_SCORE", value=s.web_search_min_score,
                  min_evidence_score=s.min_evidence_score,
                  effect="no web result can clear the D-17 coverage gate; the "
                         "web tier will retrieve evidence but never cover a goal")
    if s.web_search_min_score > s.web_search_max_score:
        log_event(logger, "config.web_search_band_inverted", level=logging.WARNING,
                  min_score=s.web_search_min_score,
                  max_score=s.web_search_max_score,
                  effect="floor exceeds ceiling; rank_to_score normalizes the "
                         "band, but the configured values are not what was meant")


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

    S-11: configure_logging() runs HERE, immediately after Settings() is
    built and before any of the three warn_on_* checks below, rather than
    later in assembly.py (its previous call site). Those checks fire at
    WARNING and exist specifically to surface misconfiguration a person
    needs to see — logging them against an unconfigured root logger (no
    handler attached yet, format unset) risked exactly that message being
    the one that got lost. assembly.py's own configure_logging call is
    now redundant (get_settings() always runs first, every real entry
    point calls it before assembly) and has been removed.
    """
    settings = Settings()
    configure_logging(settings.log_level)
    warn_on_likely_env_typos()  # P2-09: surface likely misconfiguration
    warn_on_inert_coverage_gate(settings)
    warn_on_web_search_band(settings)
    return settings
