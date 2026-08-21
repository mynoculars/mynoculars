"""
api/server.py — Minimal HTTP interface.

Purpose:
    One POST endpoint running the same graph as the CLI — demonstrates that
    the assembly function (build_app_and_settings) is interface-agnostic:
    this file adds zero new wiring, it only adds HTTP verbs on top of the
    exact same _graph.invoke() calls cli.py makes.

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

Run history: _respond() calls record_run() on every completed run (P2-08),
so API-driven runs get an agent_runs row exactly like CLI runs do. (This
paragraph previously claimed the opposite -- it predated P2-08 and was
never updated.) Checkpointing itself, the thing that makes /resume work,
comes from the graph's own checkpointer, not from this app.
"""

import uuid
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_agent import langfuse as lf
from research_agent.assembly import build_app_and_settings
from research_agent.logging_setup import run_id_var
from research_agent.state import ResearchState
from research_agent.storage.postgres import close_checkpointer, record_run

# Built ONCE, at import time (i.e. when uvicorn loads this module) — not
# per-request. This is the exact same build_app_and_settings() call cli.py
# makes per invocation; here it happens once and _graph/_settings are then
# shared across every request the process ever serves. _graph is a
# compiled LangGraph app (already bound to its checkpointer); _settings is
# the process-wide config singleton (see config.py::get_settings).
# P2-08: build_app_and_settings now returns an AppBundle (not a bare
# 2-tuple) — _durable and _checkpointer were previously unreachable from
# this file at all, which is what made both the /health gap and the
# leaked-connection-on-shutdown gap possible.
_bundle = build_app_and_settings()
# Field-by-field, NEVER tuple-unpacking. AppBundle gained a fifth field
# (mcp_bridge) in P2-13 while this line still unpacked four, which raised
# ValueError at import time and made the entire API unstartable -- every
# endpoint below, plus /health and the P2-08 record_run parity, unreachable.
# Named access cannot break that way when a future field is added.
_graph = _bundle.app
_settings = _bundle.settings
_durable = _bundle.durable
_checkpointer = _bundle.checkpointer
_mcp_bridge = _bundle.mcp_bridge
_web_mcp_bridge = _bundle.web_mcp_bridge


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Own the shutdown of everything build_app_and_settings opened.

    Replaces the deprecated @app.on_event("shutdown") hook. Closes BOTH
    the checkpointer connection (P2-08) and, when MCP or web search is
    enabled, BOTH MCPBridges -- each owning a real subprocess and a
    background thread
    that were previously left running past shutdown here, even though
    cli.py has always closed them in its own finally block.

    Also shuts the Langfuse Observer down, which cli.py's own finally
    block has always done and this file previously never did: without it
    a uvicorn process could exit with buffered observations still in the
    SDK's queue, and any root span left open by a request that died
    mid-flight would never be exported at all (an un-.end()ed span is
    not "incomplete" in the OTel model -- it is never sent). Deliberately
    LAST, after the checkpointer and bridge, so an exception closing
    either of those cannot skip the flush.
    """
    yield
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
def health() -> dict:
    """Liveness probe — proves the process is up and the graph was built.

    READS   _settings.llm_mode, _durable (module-level, see above).
    CALLS   nothing external — no store or LLM reachability is checked
            here. A 200 from this endpoint means "the FastAPI app started
            successfully," not "Postgres/Qdrant/OpenSearch are reachable."
            Each storage module does its own liveness probe independently
            at build_app_and_settings() time, and degrades silently if
            unreachable — see storage/qdrant_store.py and
            storage/opensearch_store.py for that behaviour.
    RETURNS {"status": "ok", "llm_mode": "stub"|"live", "durable": bool}
            P2-08: `durable` is new — False means checkpointing degraded to
            an in-memory MemorySaver at startup (Postgres was unreachable),
            so any paused (interrupted) run will NOT survive a process
            restart. Previously this was visible only in a startup log
            line; a caller polling /health had no way to see it at all.
    """
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
    record_run(_settings.postgres_dsn, thread_id, result.get("raw_query", ""),
              telemetry.get("recall", 0.0), telemetry)
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
    telemetry = response.get("telemetry") or {}
    if "recall" in telemetry:
        lf.score(thread_id, "recall", telemetry["recall"])
    if "critique_passed" in telemetry:
        lf.score(thread_id, "critique_passed", bool(telemetry["critique_passed"]))
    if telemetry.get("evidence_items", 0) and telemetry.get("goals", 0):
        lf.score(thread_id, "evidence_per_goal",
                 telemetry["evidence_items"] / telemetry["goals"])
    if telemetry.get("search_calls", 0):
        lf.score(thread_id, "memory_hit_rate",
                 telemetry.get("memory_hits", 0) / telemetry["search_calls"])
    if "grounding_ratio" in telemetry:
        unevidenced = telemetry.get("goals_without_evidence") or []
        lf.score(thread_id, "grounding_ratio", telemetry["grounding_ratio"],
                 comment=f"unevidenced={','.join(unevidenced) or 'none'}")


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


@app.post("/research")
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
    """
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    run_id_var.set(thread_id)
    with _traced_request(thread_id, "research_run",
                         input={"query": req.query}) as trace:
        result = _graph.invoke(ResearchState(raw_query=req.query),
                               config=_config(thread_id))
        response = _respond(thread_id, result)
        trace["output"] = response
        _record_scores(thread_id, response)
        return response


@app.post("/resume")
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
