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
import re
import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_agent.logging_setup import configure_logging, log_event

logger = logging.getLogger(__name__)

# The repository root, derived from THIS FILE's location and never from the
# current working directory. A general path anchor.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# D-148: every numeric value this process normalised on the way in, as
# (SETTING, raw, parsed). Drained and logged by get_settings() rather than
# logged where it happens, for S-11's reason exactly: Settings() is built
# BEFORE configure_logging(), so a warning emitted inside the validator
# below would go to an unconfigured root logger and be the one message
# that got lost.
_NUMERIC_NORMALISATIONS: list = []

# A number written with thousands separators, and nothing else: an optional
# sign, then groups of exactly three digits after the first, then an
# optional decimal part. Both the en-US comma and Python's own numeric
# underscore, but not mixed within one value.
#
# THE GROUPING MUST BE VALID. An earlier, looser version of this stripped
# every separator and kept the result if it parsed -- which turned "1,2,3"
# into 123. That is guessing, and this function must never guess: "1,2,3"
# is not a number anyone meant, so it is handed back untouched for pydantic
# to reject with its own message. Caught by its own test, not in review.
_GROUPED_NUMBER_RE = re.compile(
    r"^[+-]?\d{1,3}(?:(?P<sep>[,_])\d{3})+(?:\.\d+)?$")

# Quotes and whitespace a .env line picks up when a value is copied out of
# documentation. Stripping these is always safe -- they carry no meaning
# around a number -- so it happens before the grouping test above.
_NUMERIC_WRAPPERS = " \t'\""


def _normalise_numeric(raw: str) -> str:
    """Make a human-written number parseable. Pure; returns `raw` unchanged
    unless the result is unambiguously the same number.

    CALLED BY   Settings._accept_grouped_numbers, for int and float fields
                only.

    Two steps, in order, and both conservative:

      1. strip surrounding quotes and whitespace -- ' "8 192" ' has picked
         those up from a copy-paste and they mean nothing;
      2. remove thousands separators, but ONLY when what is left of them is
         a validly grouped number.

        "8,876"   -> "8876"      grouped correctly
        "8_876"   -> "8876"      Python's own separator
        "1,2,3"   -> "1,2,3"     NOT valid grouping; left for pydantic
        "8,876.5" -> "8876.5"    decimal part preserved
        "abc"     -> "abc"       untouched
        "8192"    -> "8192"      nothing to do
    """
    core = raw.strip(_NUMERIC_WRAPPERS).replace(" ", "")
    match = _GROUPED_NUMBER_RE.match(core)
    if match:
        core = core.replace(match.group("sep"), "")
    return core if core and core != raw else raw


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

    @model_validator(mode="before")
    @classmethod
    def _accept_grouped_numbers(cls, values):
        """Let a numeric setting carry a thousands separator (D-148).

        WHY THIS EXISTS. `LLM_PRIMARY_CONTEXT_TOKENS=8,876` -- a model's
        real context window, written the way a person writes a number --
        made Settings() raise, and because 43 tests reach Settings()
        through get_settings() (see tests/conftest.py::_no_ambient_config)
        that one line reported as `26 failed, 1087 passed, 17 errors`,
        none of it near the field in question.

        The isolation fix in conftest is what stops a .env typo reaching
        the suite at all. This is the other half: for a REAL run, "8,876"
        is not ambiguous and refusing it teaches nothing. Accepted, and
        recorded so get_settings() can say it happened -- silently
        rewriting a person's configuration would be the worse failure.

        Only int and float fields are touched, and only when the cleaned
        string still parses as a number; anything else falls through to
        pydantic's own error, unchanged.
        """
        if not isinstance(values, dict):
            return values
        for name, field in cls.model_fields.items():
            if field.annotation not in (int, float):
                continue
            for key in (name, name.upper()):
                raw = values.get(key)
                if not isinstance(raw, str):
                    continue
                cleaned = _normalise_numeric(raw)
                if cleaned != raw:
                    values[key] = cleaned
                    _NUMERIC_NORMALISATIONS.append(
                        (name.upper(), raw, cleaned))
        return values

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
    # D-114: the THIRD chain position is "the cloud fallback", not "Gemini".
    # Its base URL, key and model have always been generic -- only the
    # NAME was hardwired, which meant pointing these three at another
    # OpenAI-compatible provider produced a working chain whose every log
    # line, telemetry counter, health-check row and D-110 error message
    # said "gemini" while calling something else. A label that asserts a
    # property the code does not have is the defect this project keeps
    # finding (D-99 14.3, D-109); naming the slot is what stops this
    # switch from creating one.
    #
    # The name is used verbatim as the provider name everywhere: the
    # `llm.*` log lines, `llm.chain_built`, the D-110 `llm.http_error`,
    # `quality.score_failed`'s judge field, D-111's health-check row, and
    # pricing.py's rate lookup. Free-form rather than a Literal so a
    # fourth provider needs no code change here either -- an unknown name
    # costs only its pricing row, and warn_on_unpriced_fallback below says
    # so at startup rather than leaving the cost silently absent.
    llm_fallback_name: str = "gemini"
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
    # D-131 (P6-2): how many characters of EVIDENCE may enter one prompt.
    # The other half of llm_max_tokens above -- that one bounds what a
    # provider may GENERATE, this bounds what we may SEND, and until it
    # existed nothing did: compile_report inlined every evidence item and
    # critique kept a bare `evidence[-60:]` tail slice.
    #
    # 12000 IS DERIVED, NOT GUESSED. Measured on a p205.267-check-shaped
    # request (4 goals, 97 items): the evidence block alone is 30,199 of
    # 32,873 characters. p205.246's compile that actually SUCCEEDED ran
    # at 4,023 prompt tokens; 12,000 characters of evidence is ~3,000
    # estimated tokens, which with compile_report's own ~2,700-character
    # instruction body lands a compile prompt at roughly that same shape.
    # Re-derive it for your own corpus and provider the way
    # OPERATIONS.md's floor-calibration section does for min_similarity --
    # a 32k-context cloud model and a `-c 1536` local one do not want the
    # same number.
    #
    # 0 DISABLES the budget entirely and restores the pre-D-131 prompt,
    # the same documented escape hatch MIN_SIMILARITY=0.0 and
    # WEB_SEARCH_MAX_PER_DOMAIN=0 already provide --
    # warn_on_unbounded_prompt_budget below says so at startup rather
    # than letting a config value silently undo a code fix.
    prompt_evidence_max_chars: int = Field(12000, ge=0)
    # D-93: the PRIMARY provider's context window, in tokens -- the
    # `-c` value llama-server was started with. 0 (the default) means
    # "not configured", and with it every routing decision is
    # byte-identical to before this setting existed.
    #
    # WHY ONLY THE PRIMARY: this is the constrained hop. Live
    # (p205.246-check) the local model served 216- and 444-token
    # prompts and rejected 4,023- and 7,198-token ones with an
    # HTTPStatusError in 95ms and 29ms -- an immediate server-side
    # refusal, not a slow response, against a 120s timeout.
    # OPERATIONS.md's own recommended invocation for that machine is
    # `-c 1536`. So on that deployment the primary can NEVER serve
    # compiler or critic, and every run burns two guaranteed-failed
    # provider calls before falling back. Cloud fallbacks have
    # context windows orders of magnitude larger and have never been
    # observed to refuse on size, so giving them a knob would be
    # configuration nobody needs.
    #
    # Note this ALSO makes llm_max_tokens (4096) legible: a
    # generation cap larger than the whole context window is inert.
    llm_primary_context_tokens: int = Field(0, ge=0)


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
    # D-82: `max_task_retries` used to sit here, labelled CURRENTLY UNREAD
    # and inviting a future reader to "either wire it or drop it". Dropped,
    # deliberately, rather than wired: D-16's retry rule already exists and
    # is DEPTH-scoped -- agents/task_utils.py::cap_and_filter drops any key
    # whose recorded failure depth is >= the depth currently producing
    # tasks, so one query formulation may be re-emitted only by a strictly
    # later gather cycle. Adding a second, TASK-scoped retry counter would
    # have created two retry policies with no stated precedence between
    # them, and the depth-scoped one is the right shape for this
    # architecture: it is bounded by max_depth, which is one of the four
    # independent termination bounds the whole graph rests on (D-3/D-14).
    # A typed setting that does nothing is a trap for the next reader; a
    # second retry policy would have been worse than the trap.
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
    # D-132 (P6-4): the two bounds this graph did not have. max_depth,
    # max_revisions, max_escalations and recursion_limit above all bound
    # a run in STEPS; neither TIME nor SPEND had a bound anywhere. Live
    # (p205.267-check): a run inside every one of those four took 237
    # seconds and 9 provider calls with nothing misbehaving.
    #
    # 0 DISABLES each, and both ship disabled -- the same posture
    # hitl_enabled, mcp_enabled, contradiction_detection_enabled and
    # claim_verification_enabled all take, and it matters more here than
    # for any of them: this is the first setting in this codebase that
    # can end a run early. With both at 0 the graph is byte-identical to
    # before D-132.
    #
    # WHAT HAPPENS WHEN ONE IS SPENT: a soft stop, never a cancellation.
    # The checking node sets state.budget_exhausted, routing sends the
    # run to the compiler instead of another gather lap or another
    # revision, and the report says it was cut short
    # (guardrails/truncation.py). Every path still reaches telemetry and
    # still produces an answer, which is what D-1/D-21/D-22 already do
    # for every other stop condition here.
    #
    # Wall-clock seconds of RESEARCH time -- time spent paused for a
    # human review is subtracted (limits.py), so a reviewer who takes
    # four minutes to answer does not spend the run's budget. 600 is a
    # sensible starting point against the 237-second run above; there is
    # no measured "right" value and this one is deliberately not
    # defaulted to it.
    run_deadline_seconds: float = Field(0.0, ge=0.0)
    # Prompt + completion tokens across every provider call, the SAME
    # total telemetry reports as llm_total_tokens (D-86). The honest
    # complement to run_call_budget_warn, which counts REQUESTS and only
    # warns: three cheap classify calls and three 7,000-token compiles
    # are the same number of requests and not remotely the same spend.
    run_token_budget: int = Field(0, ge=0)
    run_call_budget_warn: int = Field(40, ge=1)  # D-54: observational only
    # D-18/P2-12: LLM-based contradiction detection in merger_node. Off by
    # default — costs one extra LLM call per merger execution (up to
    # max_depth times per run) whenever any goal has 2+ evidence items.
    # When off, merger_node keeps the ORIGINAL marker-only behaviour
    # (honours an explicit Evidence.contradicts, which no tool in this
    # build sets). Turning this on is what makes E2 reachable in a real
    # run for the first time.
    contradiction_detection_enabled: bool = False
    # D-95: the D-91-triggered semantic judge in critic_node. Off by
    # default, exactly like contradiction_detection_enabled above and for
    # the same reasons.
    #
    # COSTS at most ONE extra LLM call per critique pass, and only when
    # guardrails/claims.py has already flagged a figure -- a clean report
    # never pays for it. When it fires and CONFIRMS, the critique fails
    # with notes naming the specific figures, which the compiler can
    # actually act on by dropping or hedging them.
    #
    # WHY THIS ONE MAY ENFORCE WHEN D-85's NOTICE MAY NOT: a rewrite
    # genuinely can fix an unsupported figure. It cannot fix ungrounded
    # evidence, which is a property of what retrieval found -- that
    # asymmetry is the whole reason D-85 is a notice and this is a gate.
    #
    # Still OFF by default because the false-positive rate has not been
    # measured on real reports (D-54). Turn it on once
    # cited_figures_unsupported has a track record you trust.
    claim_verification_enabled: bool = False

    # --- HTTP API (api/server.py) -----------------------------------------
    # D-133 (P6-5): the shared secret /research, /resume and
    # /state/{thread_id} require. EMPTY (the default) means NO
    # AUTHENTICATION, exactly as this project has always shipped -- the
    # README has said "put it behind a gateway that terminates auth"
    # since the API existed, and that stays the right answer for a
    # reference implementation. What changes is that the repo now offers
    # a lock for the person who clones it and skips that step.
    #
    # Empty is not silent: api/server.py logs `api.unauthenticated` at
    # WARNING on startup. Deliberately logged THERE and not in
    # get_settings(), because a CLI run has no HTTP surface to protect
    # and a warning it can do nothing about is how real ones get
    # scrolled past (D-107).
    #
    # ONE key, no rotation, no per-caller identity, no scopes. That is
    # honest about what this is: a deployment-hygiene lock, not an
    # authorization model. The graph still has no notion of a caller, so
    # anything needing per-tenant isolation needs a gateway in front,
    # unchanged.
    api_key: str = ""

    # --- Memory decay (D-24/D-27) -------------------------------------------
    memory_top_k: int = Field(5, ge=1)
    # D-142: the relevance floor for RECALL. Until this existed, memory had
    # none: SemanticMemory.retrieve took scored[:top_k] unconditionally, so
    # every run inherited five remembered items no matter how unrelated
    # they were. memory_write_min_score below gates what goes IN; nothing
    # gated what came back OUT.
    #
    # Live shape (run p205.280-check, "Compare the Armies of China and
    # India"): five Redis-vs-Memcached items from an unrelated earlier run
    # were recalled at similarity 0.45-0.47 and led the compile prompt --
    # while the CORPUS floor at min_similarity 0.55 dropped 72 of 72 dense
    # candidates and the prompt budget dropped 47 real items to make room.
    # The lowest bar in the system was being applied to its least
    # trustworthy source.
    #
    # Defaults to the same 0.60 as min_similarity, for the same corpus and
    # the same embedding model, because it is the same question asked of
    # the same vector space. RE-DERIVE IT for your own corpus rather than
    # copying the number -- OPERATIONS.md's "Calibrate the retrieval floor"
    # now covers both floors. 0.0 disables it and restores the pre-D-142
    # behaviour exactly.
    memory_min_similarity: float = Field(0.60, ge=0.0, le=1.0)
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

    # --- MCP tool seam (P2-13, implements D-26; D-76: Streamable HTTP only) -
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
    # D-76: the already-running, standalone MCP server's endpoint, e.g.
    # "http://127.0.0.1:8765/mcp". This process NEVER spawns a server --
    # it only connects to one you started yourself, separately, and
    # nothing in this codebase can start or stop it. Empty is a
    # deliberately invalid default: turning mcp_enabled on with no URL
    # should surface early, not silently do nothing. (D-75 briefly made
    # this config-selectable between a spawned stdio server and a
    # standalone HTTP one; D-76 removed the stdio path entirely -- see
    # DECISIONS.md for why. This is the only connection setting now.)
    mcp_server_url: str = ""
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
    # D-76: the already-running, standalone web-search MCP server's
    # endpoint, e.g. "http://127.0.0.1:8766/mcp" -- this process never
    # spawns it. Configured entirely independently from mcp_server_url
    # above; the two standalone servers are unrelated processes with
    # different ports.
    web_mcp_server_url: str = ""
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
    # D-114: the second provider that can occupy the cloud-fallback slot.
    # Exactly the extension path pricing.py's own comment describes --
    # "one line there and two fields in config.py, never a new code path".
    langfuse_price_grok_in_per_1m: float = 0.0
    langfuse_price_grok_out_per_1m: float = 0.0


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
    "RUN_DEADLINE": "RUN_DEADLINE_SECONDS",
    "RUN_TIMEOUT": "RUN_DEADLINE_SECONDS",
    "TOKEN_BUDGET": "RUN_TOKEN_BUDGET",
    "API_TOKEN": "API_KEY",
    "RESEARCH_API_KEY": "API_KEY",
    "DEBUG": "DEBUG_TRACE",
    "MEMORY_TOPK": "MEMORY_TOP_K",
    "PROMPT_EVIDENCE_MAX": "PROMPT_EVIDENCE_MAX_CHARS",
    "EVIDENCE_MAX_CHARS": "PROMPT_EVIDENCE_MAX_CHARS",
    "CONTRADICTION_DETECTION": "CONTRADICTION_DETECTION_ENABLED",
    "SERVER_SIDE_DECAY": "MEMORY_SERVER_SIDE_DECAY",
    "MCP": "MCP_ENABLED",
    "MCP_URL": "MCP_SERVER_URL",
    "WEB_SEARCH": "WEB_SEARCH_ENABLED",
    "WEB_SEARCH_SCORE": "WEB_SEARCH_MIN_SCORE",
    "WEB_SEARCH_RESULTS": "WEB_SEARCH_MAX_RESULTS",
    "WEB_SEARCH_TIMEOUT": "WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS",
    "WEB_MCP": "WEB_MCP_SERVER_URL",
    "WEB_MCP_URL": "WEB_MCP_SERVER_URL",
    # D-79: six settings D-76 removed outright when stdio was deleted from
    # MCPBridge -- a .env carried over from before that change (or a person
    # remembering the old shape) silently does nothing now, and the
    # process that actually NEEDS a value (MCP_SERVER_URL /
    # WEB_MCP_SERVER_URL) is left empty, which raises at startup with no
    # hint that these old names are the reason. Live-evidenced, not
    # hypothetical: two separate real runs hit exactly this, one via the
    # CLI (MCP_SERVER_URL empty) and one via uvicorn (WEB_MCP_SERVER_URL
    # empty, WEB_MCP_SERVER_COMMAND set instead -- the exact old name).
    "MCP_TRANSPORT": "MCP_SERVER_URL",
    "MCP_SERVER_COMMAND": "MCP_SERVER_URL",
    "MCP_SERVER_ARGS": "MCP_SERVER_URL",
    "MCP_SERVER_ENV_ALLOWLIST": "MCP_SERVER_URL",
    "WEB_MCP_TRANSPORT": "WEB_MCP_SERVER_URL",
    "WEB_MCP_SERVER_COMMAND": "WEB_MCP_SERVER_URL",
    "WEB_MCP_SERVER_ARGS": "WEB_MCP_SERVER_URL",
    "WEB_MCP_SERVER_ENV_ALLOWLIST": "WEB_MCP_SERVER_URL",
}


def _env_file_keys() -> set:
    """Keys actually SET in the .env file itself, as opposed to
    os.environ (D-79).

    CALLED BY   warn_on_likely_env_typos, below, to close a real blind
                spot: confirmed empirically (constructing a Settings
                instance from a temp .env file and checking os.environ
                immediately after) that pydantic-settings' env_file
                support parses the file and feeds it into ITS OWN field
                resolution WITHOUT ever writing those values into the
                real process environment. warn_on_likely_env_typos'
                original `wrong in os.environ` check therefore only ever
                caught a typo in a genuinely EXPORTED shell variable --
                never a typo made by editing .env directly, which is how
                the overwhelming majority of this project's users
                actually configure it (every setup example in
                OPERATIONS.md/README.md edits .env, not `$env:`/`export`).
                The mechanism had been silently inert for its main
                intended audience since it was written (P2-09).
    READS       the SAME relative path Settings.model_config declares
                (env_file=".env") -- resolved against the CWD, same as
                Settings() itself, so this checks the exact file that
                would actually be read, not a guess at where it might be.
    RETURNS     an empty set if the file does not exist or cannot be
                parsed -- dotenv_values() itself already degrades to an
                empty dict on a missing path, confirmed directly, so no
                try/except is needed here to match that behavior.
    """
    from dotenv import dotenv_values
    return set(dotenv_values(".env").keys())


def warn_on_likely_env_typos() -> None:
    """Log a WARNING for every real environment variable matching a known
    typo pattern (P2-09).

    CALLED BY   get_settings(), below, once per process (immediately after
                Settings() is constructed and configure_logging() has run,
                S-11 — so this and the other two warn_on_* checks log
                against an already-configured root logger).
    READS       os.environ AND the .env file itself (D-79 -- see
                _env_file_keys' own docstring for why the file must be
                checked separately from os.environ, not assumed to be
                reflected in it). Settings itself never sees these keys
                at all, precisely because they don't match any declared
                field, which is the whole problem being flagged here.
    """
    env_keys = set(os.environ) | _env_file_keys()
    for wrong, right in _KNOWN_ENV_TYPOS.items():
        if wrong in env_keys and right not in env_keys:
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


def warn_on_unpriced_fallback(s: "Settings") -> None:
    """WARN when the configured fallback provider has no pricing row (D-114).

    Same shape and same reason as the two inert-setting warnings above: a
    thing that silently does nothing is worse than a thing that fails.
    LLM_FALLBACK_NAME is free-form so a fourth provider needs no code
    change, but a name pricing.py does not know -- a typo, or a genuinely
    new provider -- means calculate_cost returns None for every call it
    makes and the run's cost figure quietly loses a provider.

    Only fires when that provider is actually IN the chain: with no API
    key set, FallbackRouter.from_settings omits it entirely and its price
    is irrelevant. Nothing here fails a run -- an unpriced provider is a
    reporting gap, not an outage, exactly as LANGFUSE_PRICE_* defaulting
    to 0.0 has always been.
    """
    # Imported here, not at module import: pricing.py imports nothing from
    # config, and keeping it that way means this warning cannot introduce
    # a cycle between the two.
    from research_agent.langfuse.pricing import _PROVIDER_RATE_FIELDS
    if s.llm_fallback_api_key and s.llm_fallback_name not in _PROVIDER_RATE_FIELDS:
        log_event(logger, "config.fallback_provider_unpriced",
                  level=logging.WARNING,
                  setting="LLM_FALLBACK_NAME", value=s.llm_fallback_name,
                  known=sorted(_PROVIDER_RATE_FIELDS),
                  effect="this provider's calls report no cost; add a row to "
                         "pricing.py::_PROVIDER_RATE_FIELDS and two "
                         "LANGFUSE_PRICE_* fields to Settings")


def warn_on_unbounded_prompt_budget(s: "Settings") -> None:
    """WARN when the D-131 evidence budget is switched off.

    CALLED BY   get_settings(), below, once per process -- same shape and
                same posture as the three checks around it.

    0 stays legal: it is the documented way to reproduce the pre-D-131
    prompt deliberately. What it must not be is SILENT. An unbounded
    evidence block is what put 30,199 characters into one compile prompt
    on run p205.267-check, and a run configured back into that state
    looks, from every other number on screen, exactly like one that was
    not -- the same class of config-undoing-code defect
    warn_on_inert_coverage_gate exists for.
    """
    if s.prompt_evidence_max_chars <= 0:
        log_event(logger, "config.prompt_budget_unbounded", level=logging.WARNING,
                  setting="PROMPT_EVIDENCE_MAX_CHARS",
                  value=s.prompt_evidence_max_chars,
                  effect="every evidence item enters the compile and critique "
                         "prompts; one run measured 30,199 characters of "
                         "evidence in a single compile call")


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


def warn_on_primary_context_below_prompt_budget(s: "Settings") -> None:
    """WARN when the primary can never serve a compile or critique (D-143).

    CALLED BY   get_settings(), below, alongside the other warn_on_* checks
                it deliberately mirrors in shape and posture.

    THE INCONSISTENCY THIS CATCHES. Two settings decide whether the local
    model can ever see the two prompts that produce the deliverable, and
    nothing compared them:

      - PROMPT_EVIDENCE_MAX_CHARS bounds the evidence block (12,000 by
        default, ~3,000 tokens at this file's ~4 chars/token estimate);
      - LLM_PRIMARY_CONTEXT_TOKENS declares the window the local server was
        started with, and OPERATIONS.md recommends `-c 1536` for an 8 GB
        card, which is what people then configure.

    12,000 characters of evidence cannot fit in 1,536 tokens. Not
    sometimes -- never, by arithmetic, before a single goal, critique note
    or instruction is added. So D-93 skips the primary on every compile and
    every critique, correctly, and the three-provider chain silently
    becomes a two-provider cloud chain for exactly the two node types that
    write the report.

    Live (p205.280-check): llm_context_skips 2, llm_fallback_hops 2, and
    the compiler's answer came from a Gemini hop the local model was never
    allowed to attempt -- which then hit its own token limit. Every
    individual number was correct and nothing said the configuration was
    self-contradictory.

    A WARNING, not a failure, and not a clamp: a deliberately small local
    window is a legitimate choice (D-93 exists to make it cheap). What it
    must not be is invisible. The remedy is one of two things and the
    message says both: raise the server's -c to the model's real window, or
    lower PROMPT_EVIDENCE_MAX_CHARS to something that fits it.
    """
    if s.llm_primary_context_tokens <= 0:
        return  # unconfigured; D-93 is inert and there is nothing to compare
    # The same ~4 chars/token estimate llm/client.py::estimate_prompt_tokens
    # uses, reused rather than reinvented -- a second, subtly different
    # notion of "how big is this prompt" is exactly the drift D-99 records.
    evidence_tokens = s.prompt_evidence_max_chars / 4.0
    if evidence_tokens < s.llm_primary_context_tokens:
        return
    log_event(logger, "config.primary_context_below_prompt_budget",
              level=logging.WARNING,
              setting="LLM_PRIMARY_CONTEXT_TOKENS",
              value=s.llm_primary_context_tokens,
              prompt_evidence_max_chars=s.prompt_evidence_max_chars,
              evidence_tokens_estimate=int(evidence_tokens),
              effect="the evidence block alone cannot fit the primary's "
                     "window, so D-93 will skip it on EVERY compile and "
                     "critique and the chain is cloud-only for the two "
                     "nodes that write the report; raise the local "
                     "server's -c to the model's real window, or lower "
                     "PROMPT_EVIDENCE_MAX_CHARS to fit it")


def warn_on_normalised_numerics() -> None:
    """Report every numeric value that needed cleaning up (D-148).

    CALLED BY   get_settings(), below, immediately after
                configure_logging() and before the other warn_on_* checks
                -- this one describes what was READ, so it belongs ahead
                of the checks that judge what was read.

    DRAINS the record, so a second Settings() in the same process (every
    test that builds one) cannot make this report the same line twice.

    A WARNING rather than silence because the alternative is a system that
    quietly rewrites a person's configuration and never says so. The value
    IS accepted -- see Settings._accept_grouped_numbers -- and the message
    shows both forms so the operator can decide whether 8,876 was what
    they meant.
    """
    while _NUMERIC_NORMALISATIONS:
        setting, raw, parsed = _NUMERIC_NORMALISATIONS.pop(0)
        log_event(logger, "config.numeric_normalised", level=logging.WARNING,
                  setting=setting, raw=raw, parsed=parsed,
                  effect="the value was read as %s; remove the grouping "
                         "characters from .env to silence this" % parsed)


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
    warn_on_normalised_numerics()  # D-148: what was cleaned up on the way in
    warn_on_likely_env_typos()  # P2-09: surface likely misconfiguration
    warn_on_inert_coverage_gate(settings)
    warn_on_web_search_band(settings)
    warn_on_unbounded_prompt_budget(settings)  # D-131
    warn_on_unpriced_fallback(settings)  # D-114
    warn_on_primary_context_below_prompt_budget(settings)  # D-143
    return settings
