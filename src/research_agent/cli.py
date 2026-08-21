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
import time
import uuid

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from research_agent import langfuse as lf
# Re-exported, not just imported: build_app_and_settings and AppBundle
# moved to assembly.py so api/server.py no longer has to import its
# startup path from a module named "cli". Keeping both names bound
# here means every existing `from research_agent.cli import ...` call
# site -- including any outside this repo -- keeps working unchanged.
from research_agent.assembly import AppBundle, build_app_and_settings
from research_agent.config import get_settings
from research_agent.logging_setup import run_id_var
from research_agent.state import ResearchState
from research_agent.tracing import NullTracer, Tracer
from research_agent.storage.postgres import close_checkpointer, record_run


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
    except GraphRecursionError as exc:
        # One of the four termination bounds (D-8's invoke-time backstop)
        # actually firing is an operational event, not a crash: print a
        # diagnosable message and return a NON-ZERO exit code instead of a
        # raw traceback. main() previously returned 0 unconditionally, so
        # no script or CI step could detect a failed run at all.
        print(f"[run hit the recursion limit ({settings.recursion_limit}) — "
              f"the graph did not terminate within budget: {exc}]",
              file=sys.stderr)
        return 2
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
        # Phase 4 (D-57): the SECOND bridge, owning the web-search server
        # subprocess. None unless WEB_SEARCH_ENABLED, so a no-op for every
        # existing run. Closed separately rather than folded into the line
        # above so a failure to close one is attributable to that one.
        if bundle.web_mcp_bridge is not None:
            bundle.web_mcp_bridge.close()
        # Same reasoning again for the LLM providers' httpx clients, one
        # per configured provider, none of which was ever closed.
        if bundle.router is not None:
            bundle.router.close()
        # Phase 3: flush and release the Langfuse client last, same
        # pattern as every other closeable resource above -- a no-op
        # when observability is disabled or was never reachable.
        lf.shutdown()


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
    # Phase 3: one root trace per user query (Phase 3's own requirement),
    # keyed by the SAME thread_id already used for Postgres checkpointing
    # and structured-log correlation. No-op when observability is off.
    lf.start_trace(thread_id, "research_run", input={"query": args.query})
    # This is the single line that actually RUNS the entire graph: PLAN,
    # GATHER (looping as many times as needed), COMPILE, PERSIST — all of
    # it happens inside this one call, unless a node calls interrupt()
    # along the way (see agents/escalation.py), in which case invoke()
    # returns EARLY with "__interrupt__" in the result instead of a
    # finished report.
    # Phase 3 (#2): guarantee end_trace() fires even on a genuine
    # crash, not just the normal-completion path below. Before this,
    # only GraphRecursionError was caught (in main(), not even here),
    # and ANY other exception left the root span dangling -- which,
    # in v4's OTel model, means the trace is never exported at all,
    # not just "incomplete". This is the primary fix; Observer.
    # shutdown()'s end-open-roots loop is the backstop for whatever
    # this doesn't catch.
    try:
        # D-20 guard: a thread_id IDENTIFIES one run. Starting a FRESH
        # query on a thread that already holds state does not replace that
        # state -- `evidence` is Annotated[..., operator.add] and `counters`
        # merges, so both ACCUMULATE, while reducerless fields like
        # iteration_depth reset to 0. Found live (run p205.70-check, second
        # invocation): search_calls 18 = 12 + 6, memory_writes 31 = 15 + 16,
        # revision_cycles 3 = 1 + 2, and the previous run's E3 escalation
        # still in telemetry. Worse, the previous run's evidence was still
        # marking goals covered, so a run that retrieved ONE item reported
        # recall 1.0 at depth 1. Refuse rather than silently blend two runs.
        snapshot = app.get_state(config)
        prior = getattr(snapshot, "values", None) or {}
        if prior.get("raw_query"):
            print(
                f"[thread-id '{thread_id}' already holds a run for "
                f"\"{prior['raw_query']}\". Re-invoking it with a new query "
                f"ACCUMULATES the old run's evidence and counters instead of "
                f"replacing them (D-20). Use a fresh --thread-id, or omit the "
                f"flag to get a generated one.]",
                file=sys.stderr)
            return 3
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
            # Phase 3: HITL event -- trigger, and (once resumed) approval/
            # rejection and how long the human took. `pause_started` is this
            # process's own wall-clock, not a durable timestamp -- fine here
            # since resume latency is only meaningful within one CLI session
            # anyway (a resume from a NEW process, per Thread IDs in
            # OPERATIONS.md, would need a durable timestamp to measure this
            # honestly, which this event does not attempt).
            lf.event(thread_id, "hitl.escalation_raised",
                     input=payload, metadata={"trigger": payload.get("trigger")})
            pause_started = time.time()
            print("\n=== HUMAN REVIEW REQUIRED ===")
            print(json.dumps(payload, indent=2, default=str))
            action = ""
            # This loop keeps asking until the user types one of exactly three
            # valid words — input(...) blocks (pauses this Python process
            # entirely) until the person at the keyboard presses Enter.
            while action not in ("approve", "redirect", "abort"):
                action = input("action [approve/redirect/abort]: ").strip().lower()
                if action not in ("approve", "redirect", "abort"):
                    # Silence here cost two live runs (p205.80/.81-check):
                    # the reviewer typed their GUIDANCE at this prompt --
                    # a natural mistake, since the payload's own "hint"
                    # field talks about guidance -- got no feedback at
                    # all, and then typed "abort". The redirect they
                    # intended never happened, in either run, and the
                    # transcript gave no clue why.
                    print(f"  '{action}' is not one of the three actions. "
                          f"Type 'redirect' first -- you will be asked for "
                          f"your guidance text on the NEXT line.")
            guidance = input("guidance: ").strip() if action == "redirect" else ""
            if action == "redirect" and not guidance:
                print("  [empty guidance -- gap generation will re-run with "
                      "no new direction, which usually re-raises the same "
                      "escalation]")
            lf.event(thread_id, "hitl.resumed",
                     metadata={"trigger": payload.get("trigger"), "action": action,
                               "resume_latency_s": round(time.time() - pause_started, 2)})
            # Phase 3 (#11): the event above carries the point-in-time
            # detail (trigger, timing); this score is the aggregatable
            # form of the SAME decision -- "what fraction of runs get
            # approved vs redirected vs aborted" is a dashboard-shaped
            # question the event alone doesn't answer as cleanly.
            lf.score(thread_id, "human_review", action,
                    comment=guidance if action == "redirect" else None)
            # Command(resume={...}) is how you tell LangGraph "continue the
            # paused run, and this is what interrupt() should return this
            # time" — see agents/escalation.py's docstring for exactly how that
            # resume value flows back into the escalation node.
            result = app.invoke(Command(resume={"action": action, "guidance": guidance}),
                                config=config)

        trace_path = tracer.flush()
        if trace_path:
            print(f"[debug trace written to {trace_path}]")

        # .get(), not [] — an interrupted-then-abandoned run, or any path that
        # ends without reaching telemetry_node, left these keys absent and
        # turned a degraded run into a bare KeyError traceback.
        telemetry = result.get("telemetry") or {}
        report = result.get("final_report", "(no report was produced)")
        print(report)
        print("\n--- telemetry ---")
        print(json.dumps(telemetry, indent=2))
        record_run(settings.postgres_dsn, thread_id, args.query,
                   telemetry.get("recall", 0.0), telemetry)

        # Phase 3: custom scores pulled straight from the SAME telemetry dict
        # the report already printed above -- D-12's own rule ("aggregate,
        # never invent") applies here too: these scores repeat numbers
        # telemetry already computed, they never derive new ones.
        if "recall" in telemetry:
            lf.score(thread_id, "recall", telemetry["recall"])
        if "critique_passed" in telemetry:
            lf.score(thread_id, "critique_passed", bool(telemetry["critique_passed"]))
        if telemetry.get("evidence_items", 0) and telemetry.get("goals", 0):
            lf.score(thread_id, "evidence_per_goal",
                     telemetry["evidence_items"] / telemetry["goals"])
        if telemetry.get("search_calls", 0):
            memory_hit_rate = telemetry.get("memory_hits", 0) / telemetry["search_calls"]
            lf.score(thread_id, "memory_hit_rate", memory_hit_rate)
        # Trendable across prompt revisions -- which is what item 5's
        # prompt_name/prompt_version tagging on every generation was for.
        # The comment carries WHICH goals were unevidenced, so a low score
        # in the Langfuse UI is actionable without opening the run's logs.
        if "grounding_ratio" in telemetry:
            unevidenced = telemetry.get("goals_without_evidence") or []
            lf.score(thread_id, "grounding_ratio", telemetry["grounding_ratio"],
                     comment=f"unevidenced={','.join(unevidenced) or 'none'}")
        lf.end_trace(thread_id, output={"final_report": report, "telemetry": telemetry})

        return 0 if telemetry else 1
    except Exception as exc:
        lf.end_trace(thread_id, metadata={"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        raise


if __name__ == "__main__":
    sys.exit(main())
