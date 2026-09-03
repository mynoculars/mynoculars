"""
api/server.py — Minimal HTTP interface.

Purpose:
    One POST endpoint running the same graph as the CLI — demonstrates that
    the assembly function (build_app_and_settings) is interface-agnostic:
    this file adds zero new wiring, it only adds HTTP verbs on top of the
    exact same _graph.invoke() calls cli.py makes.

Authentication (D-133):
    POST /research, POST /resume and GET /state/{thread_id} require
    API_KEY when one is configured -- sent as `X-API-Key: <key>` or
    `Authorization: Bearer <key>`, compared in constant time. With no
    key configured every endpoint is open, exactly as this project has
    always shipped, and startup logs `api.unauthenticated` at WARNING so
    the posture is stated rather than assumed. GET /health stays open
    either way; it withholds only its build-error detail from an
    unauthenticated caller, and only when a key is set.

    ONE key, no rotation, no caller identity, no scopes, no rate
    limiting, and no CORS middleware -- FastAPI sends no CORS headers by
    default, which is the RESTRICTIVE state; adding a policy here could
    only loosen it. This is deployment hygiene for a repo that gets
    cloned and run, not an authorization model. The graph still has no
    notion of a caller, so per-tenant isolation still needs a gateway.

Responsibilities:
    - POST /research {"query": "..."} -> {"report": ..., "telemetry": ...}
      or, if the graph pauses for a human, {"status": "interrupted", ...}.
    - POST /resume — the HTTP equivalent of cli.py's input() loop: supplies
      the human's decision and continues a paused run.
    - GET /health for liveness.

Run:
    uvicorn research_agent.api.server:app --reload

Relationship to cli.py's HITL loop:
    cli.py blocks on stdin in a while loop until the run finishes. This
    file cannot block an HTTP request that way, so it turns each pause
    into a distinct response instead: /research returns "interrupted" and
    hands back a thread_id; the caller is expected to show the review
    payload to a human, then hit /resume with that same thread_id once a
    decision exists. Both endpoints share _respond() so "done" and
    "interrupted" look identical regardless of which endpoint produced
    them — a caller only needs to branch on the one status field.

Run history: _respond() calls record_run() on every completed run (P2-08) --
FAILED API runs are still not recorded, because nothing here calls
record_failed_run (D-103's CLI-only half; see D-121),
so API-driven runs get an agent_runs row exactly like CLI runs do. (This
paragraph previously claimed the opposite -- it predated P2-08 and was
never updated.) Checkpointing itself, the thing that makes /resume work,
comes from the graph's own checkpointer, not from this app.
"""

import hmac
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_agent import langfuse as lf
from research_agent.assembly import AppBundle, build_app_and_settings, reject_if_thread_in_use
from research_agent.logging_setup import log_event, run_id_var
from research_agent.reporting.scores import emit_run_scores
from research_agent.state import ResearchState
from research_agent.storage.postgres import close_checkpointer, record_run

logger = logging.getLogger(__name__)

# D-78: built LAZILY, in _lifespan's startup phase below -- NOT at import
# time, and NOT unconditionally. Every name below starts as None/empty and
# is only ever populated (or left None, on failure) once uvicorn actually
# starts serving, never while the module is still being imported.
#
# WHY THIS CHANGED (was: `_bundle = build_app_and_settings()` at plain
# module level, i.e. the very first thing this file did on import): a
# real run hit this directly -- MCP_ENABLED=true with an empty
# MCP_SERVER_URL (D-76 made that combination raise immediately, correctly,
# instead of silently half-working) took down the ENTIRE uvicorn worker
# before it could bind its port or serve /health, with a raw multiprocess
# traceback as the only feedback. cli.py never had this problem: it calls
# build_app_and_settings() inside main(), per invocation, so the exact
# same misconfiguration fails as one clean, readable error at the moment
# you try to run something -- everything else about the CLI still works.
# This file now matches that: a bad config degrades /health and every
# other endpoint to a clear 503, rather than preventing the process from
# starting at all. See DECISIONS.md D-78 for the full account.
_bundle: Optional[AppBundle] = None
_graph = None
_settings = None
_durable = False
_checkpointer = None
_mcp_bridge = None
_web_mcp_bridge = None
# Set ONLY on a failed build -- str(exc), not the exception object itself,
# so /health can return it as plain JSON without needing to know how to
# serialize whatever build_app_and_settings happened to raise.
_build_error: Optional[str] = None


def _ensure_built() -> None:
    """Raise HTTPException(503) if startup's build attempt failed.

    CALLED BY   research() and resume() below, as their first line --
                every endpoint that actually needs _graph/_settings to do
                anything, so a caller gets one clear, actionable error
                instead of an AttributeError on None.invoke(...) (a 500
                with a traceback pointing at the wrong problem entirely).
                /health does NOT call this -- it reports the same
                _build_error directly, since liveness must stay reachable
                even when the deeper build failed (that is the entire
                point of this file's D-78 rework).
    """
    if _build_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Server started but its app bundle failed to build: "
                   f"{_build_error} -- fix the underlying config (see the "
                   f"startup log for the exact error) and restart the "
                   f"server. GET /health reports this same detail.")


def _configured_key() -> str:
    """The API key this process was started with, or "" for none (D-133).

    getattr, not attribute access: a FAILED build leaves _settings as
    None, and this must answer "no key configured" then rather than
    raising. Nothing is left unprotected by that answer -- every
    endpoint the key guards returns 503 in that state without touching
    the graph, the checkpointer or any run (see _ensure_built).
    """
    return getattr(_settings, "api_key", "") or ""


def _presented_key(x_api_key: Optional[str], authorization: Optional[str]) -> str:
    """The key the caller sent, from either accepted header.

    TWO HEADERS, ONE SECRET. `X-API-Key` is the conventional shape for a
    shared key; `Authorization: Bearer` is what most HTTP clients reach
    for by habit, and refusing it would produce a 401 that looks like a
    wrong key rather than a wrong header. Accepting both costs three
    lines and removes a whole class of support question.

    Anything else -- a Basic credential, a malformed Authorization
    value -- yields "" and is rejected by the comparison, never parsed
    further. This function does not authenticate; it only reads.
    """
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _key_is_valid(presented: str) -> bool:
    """Constant-time comparison against the configured key (D-133).

    hmac.compare_digest, never `==`: a plain string comparison returns
    as soon as two bytes differ, and that timing difference is
    measurable across enough requests. This is a shared secret sent on
    every call -- exactly the shape that comparison mode is for. It is
    cheap here and the alternative is a real, if slow, oracle.

    A configured key is required for anything to be valid: with none
    set, callers are not authenticated, they are UNGATED, and the two
    states must not be confused (see require_api_key).
    """
    configured = _configured_key()
    if not configured or not presented:
        return False
    return hmac.compare_digest(presented, configured)


def require_api_key(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency guarding /research, /resume and /state (D-133).

    A NO-OP WHEN NO KEY IS CONFIGURED, which is the default and which
    keeps this repo's documented posture unchanged: the README has
    always said to put the API behind a gateway that terminates auth,
    and that is still the right answer. This adds a lock for the person
    who clones the repo and skips that step -- it does not pretend to be
    an authorization model. There is one key, no rotation, no caller
    identity, and no scopes; the graph still has no notion of who is
    asking.

    Runs BEFORE the handler, and therefore before _ensure_built's 503:
    an unauthenticated caller should not learn whether this deployment's
    app bundle built, what its MCP configuration is, or that it exists
    in a degraded state at all.

    401 rather than 403 -- the caller has not identified itself, which
    is precisely what 401 means. No WWW-Authenticate challenge is sent:
    this is not HTTP Basic and there is no interactive flow to invite.
    """
    if not _configured_key():
        return
    if _key_is_valid(_presented_key(x_api_key, authorization)):
        return
    log_event(logger, "api.rejected_unauthenticated", level=logging.WARNING,
              presented=bool(_presented_key(x_api_key, authorization)),
              effect="401 returned; the request never reached the graph")
    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key. Send it as 'X-API-Key: <key>' "
               "or 'Authorization: Bearer <key>'. This deployment sets "
               "API_KEY; a deployment that does not is open by design.")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Build the app bundle at STARTUP (not import time, D-78 — see the
    module-level comment above for the full story), then own the
    shutdown of everything it opened.

    A build failure here is caught and stored in _build_error rather than
    left to propagate: propagating it would make FastAPI/uvicorn treat
    lifespan startup itself as failed, which -- confirmed against the
    actual behavior, not assumed -- still prevents the server from
    serving ANY request, /health included. Catching it here is what
    lets /health report the failure as a normal 200 (with an error body,
    not an error status) instead of the port never opening at all.

    Replaces the deprecated @app.on_event("shutdown") hook. Closes BOTH
    the checkpointer connection (P2-08) and, when MCP or web search is
    enabled, BOTH MCPBridges -- each owning a real background thread
    that were previously left running past shutdown here, even though
    cli.py has always closed them in its own finally block. Every close
    call below is guarded: a failed build leaves these as None/False,
    and closing something that was never built would itself raise.

    Also shuts the Langfuse Observer down, which cli.py's own finally
    block has always done and this file previously never did: without it
    a uvicorn process could exit with buffered observations still in the
    SDK's queue, and any root span left open by a request that died
    mid-flight would never be exported at all (an un-.end()ed span is
    not "incomplete" in the OTel model -- it is never sent). Deliberately
    LAST, after the checkpointer and bridges, so an exception closing
    either of those cannot skip the flush.
    """
    global _bundle, _graph, _settings, _durable, _checkpointer
    global _mcp_bridge, _web_mcp_bridge, _build_error
    try:
        _bundle = build_app_and_settings()
        _graph = _bundle.app
        _settings = _bundle.settings
        _durable = _bundle.durable
        _checkpointer = _bundle.checkpointer
        _mcp_bridge = _bundle.mcp_bridge
        _web_mcp_bridge = _bundle.web_mcp_bridge
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: ANY
        # build failure (bad MCP config, an unreachable required store,
        # a settings validation error) must degrade to a reachable
        # /health, never take the whole process down. The exception is
        # still logged in full here, at ERROR, so it is not silently
        # swallowed -- only its propagation past this point is stopped.
        _build_error = f"{type(exc).__name__}: {exc}"
        log_event(logger, "api.startup_build_failed", level=logging.ERROR,
                  error=_build_error)
    # D-133: say plainly which posture this process started in. Logged
    # HERE, at API startup, and never in get_settings() -- a CLI run has
    # no HTTP surface to protect, and a warning it can do nothing about
    # is how the ones that matter get scrolled past (D-107).
    if _configured_key():
        log_event(logger, "api.authenticated", key_configured=True)
    else:
        log_event(logger, "api.unauthenticated", level=logging.WARNING,
                  effect="/research, /resume and /state/{thread_id} accept "
                         "any caller; set API_KEY, or put this behind a "
                         "gateway that terminates auth")
    yield
    if _checkpointer is not None:
        close_checkpointer(_checkpointer)
    if _mcp_bridge is not None:
        _mcp_bridge.close()
    if _web_mcp_bridge is not None:
        _web_mcp_bridge.close()
    lf.shutdown()


app = FastAPI(title="Agentic Research Agent (core build)", lifespan=_lifespan)


class ResearchRequest(BaseModel):
    """Request body for POST /research.

    query      — the research question, same string cli.py's positional
                 argument takes.
    thread_id  — optional. Omit it to start a brand-new run (a fresh id is
                 generated below); supply an OLD id only if you intend to
                 re-invoke a thread that is not currently paused — passing
                 an id belonging to a run that already finished starts a
                 second, independent run under that same checkpoint key,
                 same as reusing --thread-id on the CLI.

    Guardrail G5 (P205 Phase 2): `query` previously had no length
    constraint at all -- confirmed absent in the codebase, not just
    theorized, while reviewing this exact class. Every OTHER user-facing
    numeric setting in this project (config.py's Field(..., ge=...)
    entries) already carries a bound; this was the one place external
    text enters the graph with none. min_length=1 rejects an empty
    string outright (an empty query still reaches classify_node today
    and produces a meaningless LLM call); max_length bounds how much
    text a single request can push into goal_manager's prompt -- chosen
    generously (a real research question is a sentence or two, not a
    document) rather than tightly, since the failure mode being guarded
    against is unbounded input, not merely long input.
    """

    query: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


@app.get("/health")
def health(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None),
) -> dict:
    """Liveness probe — proves the process is up, and reports whether the
    app bundle actually finished building.

    READS   _settings, _durable, _build_error (module-level, see above).
    CALLS   nothing external — no store or LLM reachability is checked
            here. A 200 from this endpoint means "the FastAPI process is
            up and serving," not "the app bundle built successfully" --
            check the response body's own "status" field for that (D-78:
            previously a build failure meant the process never started
            at all, so /health was unreachable either way; now it is
            always reachable, and tells you WHY if something failed).
            Each storage module does its own liveness probe independently
            at build_app_and_settings() time, and degrades silently if
            unreachable — see storage/qdrant_store.py and
            storage/opensearch_store.py for that behaviour.
    ⚠ DELIBERATELY UNAUTHENTICATED, even when API_KEY is set (D-133): a
            liveness probe that needs credentials is a liveness probe
            that fails for the wrong reason, and every orchestrator that
            calls this expects an open endpoint. `llm_mode` and
            `durable` stay visible for the same reason -- they are what
            a readiness check reads and they name nothing secret. Only
            the build-error `detail` is gated, and only when a key is
            configured; see the branch below.
    RETURNS on a successful build:
                {"status": "ok", "llm_mode": "stub"|"live", "durable": bool}
            P2-08: `durable` — False means checkpointing degraded to
            an in-memory MemorySaver at startup (Postgres was unreachable),
            so any paused (interrupted) run will NOT survive a process
            restart.
            on a FAILED build (D-78):
                {"status": "error", "detail": "<exception type and message>"}
            -- every other endpoint (/research, /resume) returns
            HTTPException(503) with this same detail until the server is
            restarted with a corrected config.
    """
    if _build_error is not None:
        # D-133: the DETAIL is the only part of this response that can
        # quote a configured value back at a stranger -- it is
        # f"{type(exc).__name__}: {exc}", and the exceptions that reach
        # it name MCP URLs, DSN fragments and file paths. Withheld from
        # an unauthenticated caller when, and only when, this deployment
        # actually set a key: with none set, D-78's diagnosability is
        # exactly as it was, because that is the posture that deployment
        # chose. The full detail is always in the startup log either way.
        if _configured_key() and not _key_is_valid(
                _presented_key(x_api_key, authorization)):
            return {"status": "error"}
        return {"status": "error", "detail": _build_error}
    return {"status": "ok", "llm_mode": _settings.llm_mode, "durable": _durable}


def _config(thread_id: str) -> dict:
    """Build the LangGraph invoke config shared by both endpoints below.

    This is the SAME shape cli.py passes to app.invoke(): a thread_id
    (which checkpoint row this run's state lives under — see D-20) plus
    the recursion_limit backstop from settings (one of the four
    independent termination bounds; see orchestration/graph.py). Neither
    endpoint below constructs this dict manually, which keeps the two
    call sites from silently drifting apart on config shape.
    """
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": _settings.recursion_limit}


def _respond(thread_id: str, result: dict) -> dict:
    """Shared response shape for BOTH /research and /resume (D-23).

    READS   result — whatever _graph.invoke(...) just returned. LangGraph
            puts a "__interrupt__" key in that dict precisely when a node
            called interrupt() during this invoke (see
            agents/escalation.py::human_escalation) — its presence is the
            ONLY signal this function uses to decide which shape to return;
            there is no separate flag anywhere else to check. result also
            carries every other ResearchState field flattened at the top
            level — including "raw_query" — which is how this function gets
            the original query text on the /resume path below, where the
            request body itself (ResumeRequest) never carries one.
    CALLS   storage/postgres.py::record_run — ONLY on the "done" branch
            (P2-08). Previously NOTHING in this file ever called
            record_run, so API-driven runs produced no agent_runs row at
            all, unlike every CLI run (cli.py::main calls it unconditionally
            after printing the report) — a real asymmetry between the two
            interfaces this closes.
    RETURNS
      paused    {"thread_id", "status": "interrupted",
                 "review": <the payload human_escalation built for a
                            person to read — see escalation.py's
                            _payload_for for exactly what's in it>}
      finished  {"thread_id", "status": "done",
                 "report": <the markdown string, same as cli.py prints>,
                 "telemetry": <the same aggregate dict telemetry_node
                              produced — see compilation.py>}

    Both call sites below (research() and resume()) feed their own
    _graph.invoke(...) result through this exact function, which is why a
    caller of this API only ever needs to branch on one field ("status")
    regardless of whether the run just started or was resumed from a
    pause — the two endpoints are otherwise producing identical shapes.
    """
    if "__interrupt__" in result:
        return {"thread_id": thread_id, "status": "interrupted",
                "review": result["__interrupt__"][0].value}
    # .get(), not [] -- a run that ends without reaching telemetry_node
    # (recursion limit, an aborted resume) would otherwise raise KeyError
    # here and turn a degraded run into a 500 with no diagnostic.
    telemetry = result.get("telemetry") or {}
    # D-121: .get("recall") without a default, matching cli.py. D-103
    # removed the 0.0 fallback there -- a run that reached this line with
    # no recall in its telemetry wrote a literal 0.0, a number nothing
    # measured and indistinguishable in the column from a run that
    # genuinely retrieved nothing -- and did not touch this call site,
    # which had the identical defect. The column is nullable; NULL is what
    # "not measured" looks like.
    record_run(_settings.postgres_dsn, thread_id, result.get("raw_query", ""),
              telemetry.get("recall"), telemetry)
    return {"thread_id": thread_id, "status": "done",
            "report": result.get("final_report", ""),
            "telemetry": telemetry}


@contextmanager
def _traced_request(thread_id: str, name: str, *, input: dict):
    """Open and close ONE Langfuse root span around ONE HTTP request.

    WHY A CONTEXT MANAGER AND NOT start_trace/end_trace INLINE, and why
    the pairing is per-REQUEST rather than per-RUN: Observer.start_trace
    enters a `propagate_attributes(...)` context (that is what carries
    session_id/environment onto every span the run produces) and
    end_trace exits it. That context manager is SYNC-ONLY -- the
    installed SDK exposes no __aenter__/__aexit__ -- and it attaches to
    the OTel context of the CALLING THREAD.

    Both endpoints below are plain `def`, so FastAPI runs each one in a
    threadpool worker, and threadpool workers are REUSED across
    requests. If a context were entered on the thread serving /research
    and only exited on whichever thread later served /resume, the
    detach would target a context that thread never had: the SDK
    swallows that failure silently, and the original thread keeps the
    attached session_id FOREVER -- bleeding it into whatever unrelated
    request that worker picks up next. With one external consumer that
    is a confusing trace; with several it is one caller's session id
    stamped on another caller's run.

    So: enter and exit inside the SAME handler invocation, always, via
    the finally below. A HITL run spanning /research + /resume therefore
    produces TWO root spans rather than one -- but both land on the SAME
    Langfuse trace, because Observer derives trace_id deterministically
    from thread_id (see observer.py's SDK VERSION NOTE). Two HTTP
    requests showing up as two spans on one trace is an honest
    representation of what actually happened, and it is the version that
    cannot leak.

    Yields a dict the caller fills in: set `output` (and optionally
    `metadata`) before the block exits and they are attached to the root
    span on the way out.
    """
    lf.start_trace(thread_id, name, input=input)
    holder: dict = {"output": None, "metadata": None}
    try:
        yield holder
    except Exception as exc:
        # Same reasoning as cli.py::_run's own except/raise: a span that
        # never gets .end()ed is never exported, so a request that blew
        # up would otherwise produce NO trace -- precisely the request
        # you most want to look at. Record the error, then re-raise
        # untouched so FastAPI still returns its 500.
        holder["metadata"] = {**(holder["metadata"] or {}),
                              "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        raise
    finally:
        lf.end_trace(thread_id, output=holder["output"],
                     metadata=holder["metadata"])


def _record_scores(thread_id: str, response: dict) -> None:
    """Emit the same four Langfuse scores cli.py emits at end-of-run, so
    an API-served run is not silently less observable than a CLI one.

    Reads ONLY the telemetry dict the graph already produced -- D-12's
    "aggregate, never invent" rule, exactly as cli.py applies it. A
    still-interrupted response has no telemetry to score yet and is
    skipped entirely rather than scored as zeros.
    """
    if response.get("status") != "done":
        return
    # S-17: the five scores themselves live in reporting/scores.py. This
    # file and cli.py::_run each carried an identical copy until now, which
    # meant adding a sixth score to one interface silently made runs served
    # by the other incomparable. What stays HERE is the only thing that is
    # genuinely API-specific: a still-interrupted response has no telemetry
    # to score yet and is skipped entirely rather than scored as zeros.
    emit_run_scores(thread_id, response.get("telemetry") or {})


class ResumeRequest(BaseModel):
    """Request body for POST /resume — the human's escalation decision.

    thread_id  — MUST match the thread_id a prior /research (or /resume)
                 response returned with status "interrupted". Sending an
                 id for a run that finished normally, or that was never
                 paused, resumes nothing meaningful — LangGraph will
                 simply re-invoke that checkpoint from wherever it last
                 left off, which for a completed run is nowhere useful.
    action     — one of "approve" | "redirect" | "abort". See
                 agents/escalation.py::human_escalation for exactly what
                 each does per trigger (E1/E2/E3/E4) — the mapping is not
                 duplicated here to avoid the two copies drifting apart.
    guidance   — free text, read ONLY when action == "redirect". Ignored
                 for "approve" and "abort". This is the human's actual
                 input into the next planning/gathering/critique pass —
                 see escalation.py for where it lands in state.
    """

    thread_id: str
    action: str            # approve | redirect | abort
    guidance: str = ""


@app.get("/state/{thread_id}", dependencies=[Depends(require_api_key)])
def read_state(thread_id: str) -> dict:
    """Inspect a thread's current state without running anything (D-94).

    READS   the checkpointer, via _graph.get_state(config) -- the SAME
            call assembly.py::reject_if_thread_in_use already makes, so
            this adds no new mechanism, only a way to see what it sees.
    RETURNS a BOUNDED PROJECTION, never the raw state. See below.
    RAISES  503 if the app bundle failed to build (D-78); 404 if the
            thread holds no run.

    WHY A PROJECTION AND NOT THE WHOLE STATE: `ResearchState.evidence` is
    unbounded -- a live run reached 37 items of up to 800 characters each,
    and every one of them is verbatim corpus or third-party web text.
    Returning it would make this endpoint an unauthenticated full-text
    export of the operator's ingested corpus, over an interface the README
    already flags as having no auth. Counts and identifiers answer "where
    is this run, and what has it found" without becoming an exfiltration
    route; anyone who needs the evidence itself has the report, the
    narrative log and the database.

    ⚠ PUBLIC API SURFACE. D-37 names the HTTP shapes as this repo's
    declared public interface, owed a MAJOR bump if they change. This
    endpoint is part of that surface from the moment it ships -- it is a
    versioned commitment, not a debugging convenience that can be
    reshaped later.

    ⚠ GUARDED BY API_KEY WHEN ONE IS SET, and open when none is (D-133) --
    see require_api_key, which this endpoint declares as a dependency.
    This paragraph read "NO AUTH, like every other endpoint here" until
    now: it predated D-133 and contradicted this function's own
    decorator, which is the worst shape a comment can take -- a reader
    checking whether /state is protected would have believed it was not.
    The DEFAULT posture is still open, and the deployment advice is
    unchanged: behind a gateway that terminates auth this is a progress
    view; open to the internet it is a live feed of what a caller is
    researching.
    """
    _ensure_built()
    snapshot = _graph.get_state(_config(thread_id))
    values = getattr(snapshot, "values", None) or {}
    if not values.get("raw_query"):
        raise HTTPException(
            status_code=404,
            detail=f"thread_id '{thread_id}' holds no run. A thread exists "
                   f"only once /research has been called with it.")

    # `next` is LangGraph's own "which node(s) would run next" -- empty on
    # a finished run, populated on one paused at an interrupt().
    pending = list(getattr(snapshot, "next", ()) or ())
    goals = values.get("goals") or []
    evidence = values.get("evidence") or []
    by_source: dict = {}
    for item in evidence:
        source = getattr(item, "source", None) or "?"
        by_source[source] = by_source.get(source, 0) + 1

    return {
        "thread_id": thread_id,
        "raw_query": values.get("raw_query"),
        "status": "interrupted" if pending else "idle",
        "next": pending,
        "iteration_depth": values.get("iteration_depth", 0),
        "recall_score": values.get("recall_score", 0.0),
        "grounded_score": values.get("grounded_score", 0.0),
        "revision_count": values.get("revision_count", 0),
        "critique_passed": values.get("critique_passed", False),
        "escalation_trigger": values.get("escalation_trigger"),
        # Trigger/action pairs only -- the human's free-text guidance is
        # deliberately omitted for the same reason the evidence is.
        "escalations": [
            {"trigger": h.get("trigger"), "action": h.get("action")}
            for h in (values.get("escalation_history") or [])
        ],
        # Identifiers and coverage flags, never the descriptions' evidence.
        "goals": [
            {"goal_id": getattr(g, "goal_id", None),
             "description": getattr(g, "description", None),
             "covered": getattr(g, "covered", False),
             "contested": getattr(g, "contested", False)}
            for g in goals
        ],
        "evidence_items": len(evidence),
        "evidence_by_source": by_source,
        "report_chars": len(values.get("final_report") or ""),
        # Present only once telemetry_node has run; it is already a
        # counts-only dict by construction (D-12).
        "telemetry": values.get("telemetry") or {},
    }


@app.post("/research", dependencies=[Depends(require_api_key)])

def research(req: ResearchRequest) -> dict:
    """Start a new run (or restart under a caller-supplied thread_id).

    READS   req.query, req.thread_id (optional).
    CALLS   _graph.invoke(ResearchState(raw_query=req.query), config) —
            this is the ENTIRE run: PLAN -> GATHER -> COMPILE -> PERSIST,
            exactly as cli.py's main() drives it, just without cli.py's
            surrounding argparse/print/record_run scaffolding. If a node
            calls interrupt() anywhere along that path, invoke() returns
            immediately with "__interrupt__" in the result instead of
            running to completion — see _respond() above for how that
            shows up in this endpoint's response.
    WRITES  run_id_var (a ContextVar — see logging_setup.py) is set to
            this thread_id BEFORE invoking, so every log line this run
            produces, however deeply nested, carries the same id. This
            has no effect on graph state; it exists purely so `grep` on
            one thread_id in the logs reconstructs one run even with
            other requests' log lines interleaved (uvicorn typically
            serves requests concurrently).
    RETURNS see _respond() — "done" with a report, or "interrupted" with
            a review payload and the thread_id the caller must send back
            to /resume.

    If req.thread_id is omitted, a fresh id is generated here
    ("api-<12 hex chars>", mirroring cli.py's own "run-<12 hex chars>"
    default) — this is the only place an API-driven run's identity is
    decided.

    RAISES  HTTPException(503) if startup's app-bundle build failed
            (D-78) -- see _ensure_built()'s own docstring; checked FIRST,
            before the D-20 check below, since neither _graph nor
            reject_if_thread_in_use has anything valid to check against
            otherwise.
            HTTPException(409) if the caller supplied a thread_id that
            already holds a run (M-2 / D-20). An HTTP client can reuse
            a thread_id far more casually than a human retyping
            --thread-id, and without this check the graph's reducers
            silently blend the old run's evidence and counters into
            the new one -- see assembly.reject_if_thread_in_use.
    """
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    _ensure_built()
    run_id_var.set(thread_id)
    prior_query = reject_if_thread_in_use(_graph, _config(thread_id))
    if prior_query:
        raise HTTPException(
            status_code=409,
            detail=f"thread_id '{thread_id}' already holds a run for "
                   f"\"{prior_query}\". Re-invoking it with a new query "
                   f"ACCUMULATES the old run's evidence and counters "
                   f"instead of replacing them (D-20). Use a fresh "
                   f"thread_id, or omit it to get a generated one.")
    with _traced_request(thread_id, "research_run",
                         input={"query": req.query}) as trace:
        result = _graph.invoke(ResearchState(raw_query=req.query),
                               config=_config(thread_id))
        response = _respond(thread_id, result)
        trace["output"] = response
        _record_scores(thread_id, response)
        return response


@app.post("/resume", dependencies=[Depends(require_api_key)])
def resume(req: ResumeRequest) -> dict:
    """Continue a run that /research (or a prior /resume) reported as
    "interrupted" — the HTTP equivalent of cli.py's input() prompt.

    READS   req.thread_id, req.action, req.guidance.
    CALLS   _graph.invoke(Command(resume={"action": ..., "guidance": ...}),
            config) under the SAME thread_id the paused run used. LangGraph
            resolves that thread_id against the checkpointer (Postgres, if
            reachable — see storage/postgres.py::get_checkpointer) to find
            exactly where the run paused, then re-executes the escalation
            node from its top (see agents/escalation.py::human_escalation
            and its D-28 note for why "from its top" matters here) with
            interrupt() now RETURNING this request's action/guidance
            instead of pausing again. From there the graph proceeds
            wherever that node's Command(goto=...) sends it — could be
            straight to compiler and finish, or back to an earlier node
            (e.g. gap_generator on a "redirect") and pause AGAIN later on
            a subsequent trigger.
    WRITES  run_id_var set to req.thread_id, same reasoning as research().
    RAISES  HTTPException(503) if startup's app-bundle build failed
            (D-78) -- see _ensure_built()'s own docstring.
    RETURNS see _respond() — "done" if this resume ran the graph to
            completion, or another "interrupted" if a later check paused
            it again (e.g. a "redirect" that leads to a second failed
            critique). Callers should keep polling /resume in a loop
            exactly the way cli.py's while "__interrupt__" in result loop
            does, until they see status "done".

    ⚠ If Postgres was unreachable when this process started,
    build_app_and_settings() silently fell back to an in-memory
    checkpointer (see storage/postgres.py) — in that case a paused
    thread_id only survives as long as THIS PROCESS stays running. A
    restart between /research and /resume loses the checkpoint entirely,
    and this endpoint will find nothing to resume.
    """
    run_id_var.set(req.thread_id)
    _ensure_built()
    with _traced_request(req.thread_id, "research_resume",
                         input={"action": req.action,
                                "guidance": req.guidance}) as trace:
        result = _graph.invoke(
            Command(resume={"action": req.action, "guidance": req.guidance}),
            config=_config(req.thread_id))
        response = _respond(req.thread_id, result)
        trace["output"] = response
        _record_scores(req.thread_id, response)
        return response
