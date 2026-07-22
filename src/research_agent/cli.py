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
import sys
import uuid

from langgraph.types import Command

from research_agent.config import get_settings
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import configure_logging, run_id_var
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import ResearchState
from research_agent.tracing import NullTracer, Tracer
from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.postgres import get_checkpointer, record_run
from research_agent.storage.qdrant_store import QdrantStore
from research_agent.tools.corpus_search import make_corpus_tool


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
        (compiled_graph, settings) — every dependency wired from Settings,
        with graceful degradation applied by each storage module.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    # `tracer or NullTracer()` — if the caller passed a real Tracer, use it;
    # if they passed None (the default), fall back to a NullTracer instead,
    # so every line below can hand `tracer` to a constructor unconditionally
    # without an `if tracer is not None` check at every call site.
    tracer = tracer or NullTracer()

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
    tool = make_corpus_tool(HybridRetriever(dense, keyword))
    memory = SemanticMemory(
        QdrantStore(settings.qdrant_url, settings.memory_collection, tracer=tracer,
                    trace_label="QDRANT (semantic memory)"),
        settings.memory_top_k,
        settings.decay_half_life_days_semi_stable,
        settings.decay_half_life_days_volatile,
    )
    checkpointer, durable = get_checkpointer(settings.postgres_dsn)
    app = build_graph(router, tool, memory, settings, checkpointer)
    return app, settings


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
    # parser.add_argument("query", ...) with no leading "--" makes this a
    # POSITIONAL argument — the user must supply it directly, e.g.
    # `python -m research_agent.cli "my question"`, with no flag name.
    parser.add_argument("query", help="The research question")
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
    # parser.parse_args(argv) reads sys.argv (the actual command-line
    # arguments) by default, or the `argv` list passed into this function —
    # the latter is what lets tests call main(["some", "query", "--debug"])
    # directly without needing to spawn a real subprocess.
    args = parser.parse_args(argv)

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

    app, settings = build_app_and_settings(tracer=tracer)

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
