"""
cli.py — Command-line entry point.

Purpose:
    Wire real dependencies, run one research query, print the report and
    telemetry. The only place (besides api/server.py) where everything is
    assembled — nodes stay wiring-free.

Responsibilities:
    - Argument parsing, dependency assembly, single invoke, run-history row.

Usage:
    python -m research_agent.cli "Compare Redis and Memcached for session caching"
    python -m research_agent.cli "..." --thread-id my-run-1

If you are reading this codebase for the first time, THIS is where
execution actually begins. Everything else in src/research_agent/ is a
function or class waiting to be called; this file is what calls the first
of them. Read build_app_and_settings() first, then main() — together they
are the complete story of "what happens between typing a command and seeing
a report," with every other file's role explained by which of these two
functions calls into it.

Python mechanics used in this file, if any of this is new to you:
    if __name__ == "__main__": sys.exit(main())
        A near-universal Python idiom. When you RUN a file directly (e.g.
        `python -m research_agent.cli "..."`), Python sets that module's
        special __name__ variable to the string "__main__". When a file is
        instead IMPORTED by some other code (e.g. api/server.py doing
        `from research_agent.cli import build_app_and_settings`), __name__
        is set to the module's dotted path instead, so this line's
        condition is False and main() does NOT run automatically. This is
        what lets api/server.py reuse build_app_and_settings from this file
        without also triggering an argument-parsing command-line run every
        time it's imported.
    sys.exit(main())
        main() returns an integer (0 for success); sys.exit(...) terminates
        the Python process with that integer as its exit code — the value
        a shell script or CI pipeline could check to know whether the run
        succeeded.
"""

import argparse
import json
import logging
import sys
import uuid
from typing import NamedTuple

from langgraph.types import Command

from research_agent.config import get_settings, split_csv
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import configure_logging, log_event, run_id_var
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import ResearchState
from research_agent.tracing import NullTracer, Tracer
from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.postgres import close_checkpointer, get_checkpointer, record_run
from research_agent.storage.qdrant_store import QdrantStore
from research_agent.tools.corpus_search import make_corpus_tool
from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool


class AppBundle(NamedTuple):
    """Everything build_app_and_settings assembles (P2-08).

    Replaces the old bare (app, settings) 2-tuple with named fields so
    `durable` and `checkpointer` are no longer silently dropped by callers
    who only unpack the first two — the exact gap this item exists to
    close (`durable` was previously computed inside build_app_and_settings
    and never returned at all; see get_checkpointer in storage/postgres.py
    for what durable=False actually means operationally).
    """

    app: object
    settings: object
    durable: bool
    checkpointer: object
    mcp_bridge: object = None  # P2-13: an MCPBridge if settings.mcp_enabled,
                               # else None -- see build_app_and_settings and
                               # main()'s finally block, which closes this
                               # exactly like close_checkpointer(checkpointer)
                               # does, when it's not None.


def build_app_and_settings(tracer=None):
    """Assemble the full application (shared by CLI and API).

    This function is the ENTIRE dependency graph of this project laid out
    in one place: every object every node in agents/*.py eventually uses
    (the LLM router, both retrieval stores, the memory wrapper, the
    checkpointer) is constructed HERE and nowhere else. Nothing inside
    orchestration/graph.py or agents/*.py ever imports config.py, httpx,
    qdrant_client, or opensearchpy directly — they receive already-built
    objects as arguments instead (the closure pattern explained across
    agents/*.py's docstrings). This is what makes swapping in fake objects
    for tests, or reusing the exact same wiring from api/server.py, as
    simple as calling this one function.

    CALLED BY   main() below, and api/server.py at module-import time (so
                the FastAPI app's _graph and _settings are built once, when
                uvicorn loads the module, and reused across every incoming
                HTTP request).

    Parameters:
        tracer: optional Tracer for debug tracing (threaded into the LLM chain
            and both retrieval stores). None -> no tracing overhead.

    Returns:
        AppBundle(app, settings, durable, checkpointer) — every dependency
        wired from Settings, with graceful degradation applied by each
        storage module. P2-08: `durable` and `checkpointer` are now part of
        the return value (previously `durable` was computed here and
        silently discarded, and `checkpointer` wasn't returned at all,
        making close_checkpointer's cleanup unreachable from either
        caller).
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    # `tracer or NullTracer()` — if the caller passed a real Tracer, use it;
    # if they passed None (the default), fall back to a NullTracer instead,
    # so every line below can hand `tracer` to a constructor unconditionally
    # without an `if tracer is not None` check at every call site.
    tracer = tracer or NullTracer()
    # debug reuses tracer.enabled rather than being a second, separately-
    # supplied flag: a real Tracer means --debug (or DEBUG_TRACE=true) was
    # on, and that's exactly when we also want every node's "node.enter"
    # line to fire. Deriving it here means graph.py and every agents/*.py
    # builder gets one unambiguous boolean, with no risk of it drifting out
    # of sync with whether tracing is actually on.
    debug = tracer.enabled

    router = FallbackRouter.from_settings(settings, tracer=tracer)
    # Two SEPARATE QdrantStore instances get created in this function — one
    # here (for the corpus, used inside HybridRetriever below) and one
    # further down (for semantic memory) — pointed at two different
    # collection names on the SAME Qdrant server. See storage/
    # qdrant_store.py's docstring for why one class serves both roles.
    dense = QdrantStore(settings.qdrant_url, settings.corpus_index, tracer=tracer)
    keyword = OpenSearchStore(
        settings.opensearch_url, settings.corpus_index,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        tracer=tracer)
    # P2-01: settings.min_similarity is the retrieval-time floor on the
    # dense leg — see retrieval/hybrid.py for what it filters and why.
    # P2-14 (D-25) follow-up to P2-13: the corpus tool is now ALWAYS
    # built and is ALWAYS the default -- settings.mcp_enabled no longer
    # swaps it out wholesale (that was P2-13's original, simpler
    # behavior, before P2-14's tool_hint routing existed to make a
    # genuine choice possible). Instead, when mcp_enabled, the MCP tool
    # is built ADDITIONALLY, as a second, ADDRESSABLE specialist --
    # reachable only when a task's tool_hint == "mcp" (see
    # orchestration/graph.py::dispatch_tasks and
    # agents/planning.py::task_expander_node /
    # agents/gathering.py::gap_generator_node, which are the only two
    # places that ever set that hint, and only when settings.mcp_enabled
    # is what allowed it in the first place). With mcp_enabled off (the
    # default), mcp_tool stays None and build_graph registers no extra
    # node at all -- behavior is unchanged from every run before this.
    tool = make_corpus_tool(
        HybridRetriever(dense, keyword, min_similarity=settings.min_similarity))
    mcp_bridge = None
    mcp_tool = None
    if settings.mcp_enabled:
        mcp_bridge = MCPBridge(
            command=settings.mcp_server_command,
            args=split_csv(settings.mcp_server_args),
            env_allowlist=split_csv(settings.mcp_server_env_allowlist),
        )
        mcp_tool = make_mcp_tool(
            mcp_bridge, settings.mcp_tool_name,
            query_arg_name=settings.mcp_query_arg_name,
            call_timeout_seconds=settings.mcp_call_timeout_seconds)
    memory = SemanticMemory(
        QdrantStore(settings.qdrant_url, settings.memory_collection, tracer=tracer,
                    trace_label="QDRANT (semantic memory)"),
        settings.memory_top_k,
        settings.decay_half_life_days_semi_stable,
        settings.decay_half_life_days_volatile,
        server_side_decay=settings.memory_server_side_decay,  # P2-10
    )
    checkpointer, durable = get_checkpointer(settings.postgres_dsn)
    if not durable:
        # P2-08: previously this was visible only as a WARNING log line
        # from get_checkpointer itself (checkpointer.memory_fallback) — a
        # caller that doesn't read logs had no way to know. Surfacing it
        # here too means both cli.py and api/server.py can act on it
        # (print a banner, put it in /health) without re-deriving it.
        log_event(logging.getLogger(__name__), "app.degraded_checkpointing",
                  level=logging.WARNING)
    app = build_graph(router, tool, memory, settings, checkpointer, debug=debug,
                     mcp_tool=mcp_tool)  # P2-14: None unless settings.mcp_enabled
    return AppBundle(app=app, settings=settings, durable=durable,
                     checkpointer=checkpointer, mcp_bridge=mcp_bridge)


def main(argv=None) -> int:
    """Run one query end to end; returns process exit code.

    This function is the CLI's version of a "controller": it handles
    everything OUTSIDE the graph itself — reading the command line,
    building the app, calling invoke(), handling a human-escalation
    pause/resume loop if one occurs, and printing/persisting the result.
    Nothing in here contains any research LOGIC; all of that lives inside
    the graph built by build_app_and_settings() above.

    CALLED BY   the `if __name__ == "__main__":` guard at the bottom of
                this file, when this module is run directly as a script.
    """
    parser = argparse.ArgumentParser(description="Agentic research agent (core build)")
    # nargs="?" makes this positional argument OPTIONAL (default None) —
    # needed so `--print-graph` can be used on its own, with no question,
    # just to inspect the topology. The manual check a few lines below
    # (`if args.query is None and not args.print_graph`) is what still
    # enforces "a query is required for an actual run."
    parser.add_argument("query", nargs="?", default=None, help="The research question")
    parser.add_argument("--thread-id", default=None,
                        help="Run identity (fresh UUID per run by default; "
                             "reuse an old id to resume — design D-20)")
    # action="store_true" means this flag takes NO value — its mere
    # presence on the command line (`--debug`) sets args.debug to True;
    # its absence leaves args.debug as False. Contrast with --thread-id
    # above, which expects a value right after it.
    parser.add_argument("--debug", action="store_true",
                        help="Dump exact prompts, raw responses, provider, "
                             "tokens/latency, and every retrieval engine's hits "
                             "to logs/trace-<run_id>.txt (also enabled by "
                             "DEBUG_TRACE=true in .env).")
    parser.add_argument("--print-graph", action="store_true",
                        help="Print the graph's TOPOLOGY (13 node names and "
                             "how they're wired) — not a run's output. This "
                             "is static structure, unrelated to telemetry, "
                             "which summarizes what happened during a run. "
                             "Combine with a query to see it before running; "
                             "omit the query to just inspect the wiring and "
                             "exit without running anything.")
    # parser.parse_args(argv) reads sys.argv (the actual command-line
    # arguments) by default, or the `argv` list passed into this function —
    # the latter is what lets tests call main(["some", "query", "--debug"])
    # directly without needing to spawn a real subprocess.
    args = parser.parse_args(argv)
    if args.query is None and not args.print_graph:
        parser.error("query is required unless --print-graph is given alone")

    thread_id = args.thread_id or f"run-{uuid.uuid4().hex[:12]}"
    run_id_var.set(thread_id)  # correlate every log line to this run

    # get_settings() is called TWICE across this file's flow: once here,
    # just to peek at debug_trace before the tracer exists, and again
    # inside build_app_and_settings() below. That's harmless — remember
    # from config.py that get_settings() is cached (@lru_cache), so both
    # calls return the exact same Settings object rather than re-reading
    # the environment twice.
    settings_peek = get_settings()
    trace_on = args.debug or settings_peek.debug_trace
    tracer = Tracer(thread_id) if trace_on else NullTracer()

    bundle = build_app_and_settings(tracer=tracer)
    app, settings = bundle.app, bundle.settings
    if not bundle.durable:
        print("[warning: Postgres unreachable — running with an in-memory "
              "checkpointer; a process restart loses any paused/resumable run]")

    try:
        return _run(app, settings, args, thread_id, tracer)
    finally:
        # P2-08: close whatever connection get_checkpointer opened, even if
        # _run raised — a CLI process is short-lived, but leaving this to
        # the OS was never a design decision, just an oversight this item
        # closes.
        close_checkpointer(bundle.checkpointer)
        # P2-13: same reasoning as close_checkpointer above -- if MCP is
        # enabled, bundle.mcp_bridge owns a real subprocess and background
        # thread that should not just be left to the OS. None when MCP is
        # off (the default), so this is a no-op for every existing run.
        if bundle.mcp_bridge is not None:
            bundle.mcp_bridge.close()


def _run(app, settings, args, thread_id, tracer) -> int:
    """The actual run, factored out of main() so P2-08's finally/close
    wraps it cleanly without one giant try block."""
    if args.print_graph:
        # app.get_graph() is a LangGraph/LangChain introspection call — it
        # walks the SAME compiled graph object we're about to (maybe) run,
        # and returns its static topology: node names and edges, with no
        # dependency on any particular query or run. This is why it is NOT
        # the same thing as telemetry: telemetry is a dict of COUNTS
        # summarizing what happened during one specific invoke() call
        # (llm_calls, recall, etc — see agents/compilation.py::
        # telemetry_node); this is the WIRING itself, unchanged run to run.
        graph_repr = app.get_graph()
        try:
            # draw_ascii() needs the optional `grandalf` package (not in
            # requirements.txt) — try it first since it's the most readable
            # in a terminal, and fall back to Mermaid text (needs nothing
            # extra) if that package isn't installed.
            print(graph_repr.draw_ascii())
        except ImportError:
            print("[install 'grandalf' for ASCII art — falling back to Mermaid text]")
            print(graph_repr.draw_mermaid())
        if args.query is None:
            return 0  # inspecting the wiring only — nothing to run

    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.recursion_limit}
    # This is the single line that actually RUNS the entire graph: PLAN,
    # GATHER (looping as many times as needed), COMPILE, PERSIST — all of
    # it happens inside this one call, unless a node calls interrupt()
    # along the way (see agents/escalation.py), in which case invoke()
    # returns EARLY with "__interrupt__" in the result instead of a
    # finished report.
    result = app.invoke(ResearchState(raw_query=args.query), config=config)

    # HITL loop (D-23): an interrupted run surfaces "__interrupt__" instead
    # of finishing. Show the review payload, collect a decision, resume under
    # the SAME thread_id (D-20). Blocking stdin IS the timeout policy for a
    # CLI — the deferred operational decision, resolved per-interface.
    #
    # `while "__interrupt__" in result:` — result is a plain dict here;
    # this checks for the presence of that one special key, which LangGraph
    # adds only when a node paused via interrupt(). The loop keeps calling
    # invoke() again (each time with a human's decision) until a call
    # finally returns a result WITHOUT that key — i.e. the run has actually
    # finished.
    while "__interrupt__" in result:
        # result["__interrupt__"] is a list (LangGraph supports multiple
        # simultaneous interrupts in more advanced graphs, though this
        # project's graph only ever produces one at a time); [0] takes the
        # first one, and .value is the actual payload dict
        # agents/escalation.py::_payload_for built.
        payload = result["__interrupt__"][0].value
        print("\n=== HUMAN REVIEW REQUIRED ===")
        print(json.dumps(payload, indent=2, default=str))
        action = ""
        # This loop keeps asking until the user types one of exactly three
        # valid words — input(...) blocks (pauses this Python process
        # entirely) until the person at the keyboard presses Enter.
        while action not in ("approve", "redirect", "abort"):
            action = input("action [approve/redirect/abort]: ").strip().lower()
        guidance = input("guidance: ").strip() if action == "redirect" else ""
        # Command(resume={...}) is how you tell LangGraph "continue the
        # paused run, and this is what interrupt() should return this
        # time" — see agents/escalation.py's docstring for exactly how that
        # resume value flows back into the escalation node.
        result = app.invoke(Command(resume={"action": action, "guidance": guidance}),
                            config=config)

    trace_path = tracer.flush()
    if trace_path:
        print(f"[debug trace written to {trace_path}]")

    print(result["final_report"])
    print("\n--- telemetry ---")
    print(json.dumps(result["telemetry"], indent=2))
    record_run(settings.postgres_dsn, thread_id, args.query,
               result["telemetry"].get("recall", 0.0), result["telemetry"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
