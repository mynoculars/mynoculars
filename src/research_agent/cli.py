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
from research_agent.assembly import (AppBundle, build_app_and_settings,
                                     reject_if_thread_in_use)
from research_agent.config import get_settings
from research_agent.llm.router import ProviderChainExhausted
from research_agent.logging_setup import drain_problems, run_id_var
from research_agent.reporting.confidence import format_line as format_confidence
from research_agent.reporting.metrics import count_sections
from research_agent.state import ResearchState
from research_agent.tracing import NullTracer, Tracer
from research_agent.storage.postgres import (close_checkpointer,
                                             record_failed_run, record_run)


def _fmt_hitl_wall_time_line(hitl_triggered: bool, elapsed_s: float) -> str:
    """One line reporting whether HITL fired and the total wall-clock
    time from the first invoke() to the run actually finishing.

    Pure and side-effect-free (matches _fmt_result_summary's own shape
    just below) so it is directly unit-testable without running main()'s
    interactive loop.

    elapsed_s deliberately INCLUDES any time spent blocked on a human at
    the "action [approve/redirect/abort]:" prompt -- the person asked how
    long THEY waited for an answer, and a human review pause is part of
    that, not separate from it.
    """
    minutes, seconds = divmod(elapsed_s, 60)
    status = "HITL triggered" if hitl_triggered else "HITL Not triggered"
    return f"{status} | Total wall time: {int(minutes)}min {seconds:.2f}secs"


def _fmt_problems(records, dropped: int) -> str:
    """The run's WARNINGs and ERRORs, once, in plain text (D-118).

    CALLED BY   main()'s finally block, so it prints on the normal path,
                the diagnosable-failure paths (exit 2/4) and the
                unrecognised-exception path alike -- the requirement is
                that the same diagnostic is available wherever the
                failure propagates, and a summary that only prints on
                success would fail exactly the case it exists for.

    Returns "" when the run logged nothing above INFO, so a clean run
    prints no section at all rather than an empty banner.

    Repeats are collapsed: three identical context skips are one problem
    seen three times, and listing them separately would push a singular
    failure off the screen. `provider`, `status`, `kind` and `hint` are
    named first and in a fixed order because they are what an operator
    acts on; `body` prints last and whole, because that is the field that
    said "this team has no credits" in run p205.265-check while the
    status alone said only 403.
    """
    if not records:
        return ""
    order, groups = [], {}
    for level, name, fields in records:
        if name not in groups:
            order.append(name)
            groups[name] = [level, fields, 0]
        groups[name][2] += 1

    lines = ["", "=== PROBLEMS ===",
             f"{len(records)} warning(s)/error(s) logged during this run"
             + (f"; {dropped} more not shown (collector cap)" if dropped else "")]
    for name in order:
        level, fields, count = groups[name]
        lines.append("")
        lines.append(f"[{level}] {name}" + (f"  (x{count})" if count > 1 else ""))
        shown = {k: v for k, v in fields.items() if k != "run_id"}
        width = max([len("detail")] + [len(k) for k in shown])
        for key in ("provider", "model", "node", "status", "kind", "hint",
                    "effect"):
            if shown.get(key) not in (None, ""):
                lines.append(f"    {key:<{width}} : {shown.pop(key)}")
        body = shown.pop("body", None)
        rest = "  ".join(f"{k}={v}" for k, v in shown.items())
        if rest:
            lines.append(f"    {'detail':<{width}} : {rest}")
        if body:
            lines.append(f"    {'body':<{width}} : {body}")
    return "\n".join(lines)


def _fmt_judge_line(telemetry: dict) -> str:
    """One line describing what the quality judge actually said (D-108).

    CALLED BY   _fmt_result_summary below, and covered by its own tests
                because the three states it distinguishes are the whole
                point and each is a different sentence.

    D-106 recorded the mean, the rejections and the distribution, and
    routed them to the agent_runs row and the cross-run report -- but the
    RESULT block a human reads after EVERY run still showed only
    "0/2 quality check(s) failed", i.e. the failure ratio and nothing
    about the judgement. That is the same shape of defect this whole
    series keeps finding: PHASE5-HONESTY 14.6 and 16.5 both record a
    signal that existed where nobody looked. Recording the distribution
    and then not showing it here would have been the third instance.

    Three distinguishable states, deliberately worded so they cannot be
    confused with each other:
      - the judge scored things       -> mean, rejections, distribution
      - the judge was asked and failed every time -> say THAT, and point
        at the counter that explains it, because a fail-open run looks
        identical to a clean one in every other number on screen
      - the judge was never asked     -> the common case (a single
        provider, or a chain that never needed a second opinion)
    """
    judged = int(telemetry.get("llm_quality_scores_judged") or 0)
    attempted = int(telemetry.get("llm_quality_calls") or 0)
    failed = int(telemetry.get("llm_quality_calls_failed") or 0)
    if judged:
        bands = telemetry.get("llm_quality_bands") or {}
        band_text = ", ".join(f"{name} {count}"
                              for name, count in bands.items()) or "none"
        return (f"Quality judge: {judged} judgement(s), mean "
                f"{telemetry.get('llm_quality_score_mean')}, "
                f"{int(telemetry.get('llm_quality_rejections') or 0)} below "
                f"threshold  [{band_text}]"
                + (f"   ({failed} scoring call(s) failed open)"
                   if failed else ""))
    if attempted:
        # The state D-53's WARNING exists for, stated here in the summary
        # rather than only in a log line -- a run whose judge failed every
        # time had NO quality gate, and every other number on this screen
        # looks exactly as it would have if the gate had passed everything.
        return (f"Quality judge: no judgement -- all {attempted} scoring "
                f"call(s) failed open (the gate was inert this run)")
    return "Quality judge: not asked (no fallback hop needed a second opinion)"


def _fmt_result_summary(telemetry: dict, report: str) -> str:
    """Render the human-readable verdict block printed after the report.

    CALLED BY   main(), between the report and the raw telemetry dump.

    WHY THIS EXISTS: the deliverable was the only output this file did not
    label. `=== HUMAN REVIEW REQUIRED ===` had a banner and
    `--- telemetry ---` had a marker, but `print(report)` had neither, so
    after several hundred lines of stderr JSON the report simply began --
    opening on a markdown `#` heading that reads as more noise in a
    terminal rather than as the answer.

    Worse, the run's VERDICT was legible only by reading 45 lines of JSON.
    Runs p205.211 and p205.212 differed by `critique_passed`,
    `revision_cycles` and a 652-vs-9,603-character report, and none of that
    was visible at a glance; the difference was only found by diffing two
    log files. Every number below is one telemetry already computed --
    D-12's "aggregate, never invent" applies here exactly as it does in the
    telemetry node, and this function derives nothing new. The raw JSON is
    still printed in full underneath, unchanged, for anything that parses
    it.

    `.get` with defaults throughout, never `[]`: an interrupted or degraded
    run reaches this line with a partial (or empty) telemetry dict, and the
    whole point of the summary is to stay readable exactly then. A missing
    key renders as "n/a", never a KeyError -- the same reasoning that made
    main() read `result.get("telemetry")` in the first place.
    """
    def num(key, default="n/a"):
        v = telemetry.get(key)
        return default if v is None else v

    passed = telemetry.get("critique_passed")
    cycles = telemetry.get("revision_cycles", 0)
    if passed is True:
        verdict = "PASSED"
    elif passed is False:
        verdict = "FAILED"
    else:
        verdict = "n/a (run did not reach the critic)"
    if cycles:
        verdict += f" after {cycles} revision cycle(s)"

    # "E4 -> approve, E1 -> redirect", or "none". Reads the same list the
    # JSON shows; the arrow is the only thing added.
    esc = telemetry.get("escalations") or []
    esc_line = ", ".join(f"{e.get('trigger', '?')} -> {e.get('action', '?')}"
                         for e in esc) or "none"

    by_source = telemetry.get("evidence_by_source") or {}
    src_line = " / ".join(f"{k} {v}" for k, v in by_source.items()) or "none"

    # Report shape is measured here rather than pulled from telemetry
    # because telemetry does not carry it: the count of Markdown
    # sections and the character length are the two figures that made
    # p205.211's failure obvious on sight (3 sections / 2,737 chars, then
    # 0 sections / 652 chars on the revision). S-10: shared with
    # compiler_node's "node.compiled" log line via reporting/metrics.py
    # -- previously two different regexes (this one only matched exactly
    # "## ", compiler_node's matched "#" through "######") reported two
    # different counts for the same report.
    sections = count_sections(report)

    # D-145: the composed verdict FIRST, because it is the one line that
    # answers the question every other line contributes to. p205.280-check
    # printed six raw signals -- recall 1.0, grounding_ratio 1.0,
    # corpus_recall 0.0, grounded 0.0, a 0.067 judge mean and a failed
    # critique -- and left the reader to integrate them; the report shipped
    # because a human approved an E4 and the prose read fine.
    confidence = telemetry.get("confidence") or {}
    # D-144: two attribution facts that WERE computed and never printed.
    # "0 listed / 58 retrieved" is the single most useful line in the block
    # on a run like p205.280-check, and finding it previously meant reading
    # 45 lines of JSON.
    listed = telemetry.get("web_sources_listed")
    web_items = telemetry.get("web_sourced_items")
    domains = telemetry.get("web_source_domains")
    cited_goals = telemetry.get("evidence_cited")
    attached = int(telemetry.get("citations_attached") or 0)

    lines = [
        "",
        "=== RESULT ===",
        f"Confidence   : {format_confidence(confidence)}"
        if confidence else "Confidence   : n/a",
        f"Citations    : {num('evidence_cited')} goal(s) cited in the prose"
        + (f", {attached} attached deterministically" if attached else "")
        + f"   [{num('evidence_items')} evidence item(s) available]",
        f"Sources      : {num('web_sources_listed')} listed"
        + (f" / {int(web_items)} web item(s) across {num('web_source_domains')} domain(s)"
           if web_items else " (no web evidence retrieved)"),
        f"Critique     : {verdict}",
        f"Escalations  : {esc_line}",
        f"Report       : {sections} section(s), {len(report):,} chars",
        f"Goals        : {num('goals')} ({len(telemetry.get('goals_without_evidence') or [])} without evidence)",
        f"Evidence     : {num('evidence_items')} item(s) -- {src_line}",
        f"Recall       : {num('recall')}   grounding_ratio {num('grounding_ratio')}"
        f"   grounded {num('grounded_score')}   corpus_recall {num('corpus_recall')}",
        f"Providers    : {num('llm_provider_calls')} call(s), "
        f"{num('llm_fallback_hops')} fallback hop(s), "
        f"{num('llm_quality_calls_failed')}/{num('llm_quality_calls')} quality check(s) failed",
        # D-108: the line above counts how often the judge was ASKED and
        # how often asking failed. It says nothing about what the judge
        # decided -- which, across five live runs, is the thing that
        # actually determined which answer shipped.
        _fmt_judge_line(telemetry),
        f"Search       : {num('search_calls')} call(s), {num('search_failures')} failed",
    ]
    if telemetry.get("planning_error"):
        lines.append(f"Planning     : {telemetry['planning_error']}")
    return "\n".join(lines)


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
    except ProviderChainExhausted as exc:
        # D-101: the SAME class of event as GraphRecursionError above, and
        # this file already states the policy in that handler's comment --
        # an operational event gets a diagnosable message and a non-zero
        # exit code, never a raw traceback. D-78 made the identical
        # argument for the API. Provider-chain exhaustion had no such
        # handler, so run p205.254-check ended in 40 lines of LangGraph
        # internals whose only real content was the last provider's
        # exception -- which by itself does not say the other two failed,
        # or how they failed.
        #
        # Exit codes in this file: 0 success, 1 ran but produced no
        # telemetry, 2 recursion limit, 3 thread-id already in use,
        # 4 provider chain exhausted. Distinct from 2 because the
        # operator action differs: 2 means look at the graph's budgets,
        # 4 means look at the providers.
        chain = " → ".join(f"{name} {how}" for name, how in exc.attempts)
        where = f" at the {exc.node} node" if exc.node else ""
        print(f"[all {len(exc.attempts)} providers in the chain failed"
              f"{where} ({exc.mode} call):\n  {chain}]", file=sys.stderr)
        # The last provider's own message carries the detail the chain
        # summary cannot (which ceiling truncated it, D-102; what the HTTP
        # status was), and __cause__ is where `raise ... from` put it.
        if exc.__cause__ is not None:
            print(f"[last failure: {type(exc.__cause__).__name__}: "
                  f"{exc.__cause__}]", file=sys.stderr)
        return 4
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
        # D-100: the tracer was the ONE resource this block did not
        # release. tracer.flush() sits at the end of _run's HAPPY path
        # (below), so the run that most needs a debug trace -- one that
        # raised -- was precisely the one that never wrote one. Live
        # (p205.254-check): a provider-chain exhaustion at the compiler
        # produced a 40-line traceback and no logs/run-*.txt at all.
        # Same oversight and same shape as close_checkpointer's own
        # comment above describes for itself: leaving this to the process
        # exiting was never a design decision.
        #
        # LAST in this block, after every close above, because those
        # closes emit real events (checkpointer.closed, llm.close_failed,
        # mcp.bridge_closed) that belong in the narrative this writes.
        #
        # Safe on the happy path too, where _run already flushed:
        # flush_narrative POPS the run's buffered events
        # (reporting/narrative.py::flush_narrative), so a second call
        # finds nothing and returns None. NullTracer.flush() returns None
        # unconditionally, so a non-debug run pays nothing here.
        #
        # D-118: BEFORE the tracer flush below, so the last line on the
        # console stays the trace path rather than a wall of problems,
        # and AFTER every close above, so a failure while closing a
        # resource is included rather than missed by one line.
        #
        # Drained HERE, in the finally, means it prints exactly once per
        # run on every path -- success, exit 2, exit 3, exit 4, or an
        # exception nobody recognised. A summary that only printed on
        # success would fail exactly the case it exists for.
        problems, problems_dropped = drain_problems()
        problem_text = _fmt_problems(problems, problems_dropped)
        if problem_text:
            print(problem_text, file=sys.stderr)
        # stderr, not stdout: on the crash path stdout may hold a
        # half-written report, and this is diagnostic output.
        trace_path = tracer.flush()
        if trace_path:
            print(f"[debug trace written to {trace_path}]", file=sys.stderr)


def _failure_record(exc: BaseException) -> dict:
    """Describe one failed run for its agent_runs row (D-103).

    Pure and exception-shaped rather than a formatted string, so
    analyze_runs.py can GROUP by failure type instead of grepping prose —
    "how often do we lose a run to provider exhaustion" is a count, and a
    count needs a field.

    ProviderChainExhausted gets its chain, node and mode recorded, because
    those are precisely what the bare exception could not carry (D-101)
    and precisely what makes a run of these rows worth reading: a history
    where `primary` fails every time says something a single run cannot.
    The underlying `__cause__` comes along for the same reason main()
    prints it — D-102's cap attribution lives in that string.

    Messages are truncated at 500 characters. A row is a record, not a log
    file, and an unbounded provider error would be the one field able to
    bloat the table.
    """
    record = {"type": type(exc).__name__, "message": str(exc)[:500]}
    if isinstance(exc, ProviderChainExhausted):
        record["node"] = exc.node
        record["mode"] = exc.mode
        # list(), not the tuples themselves: this is about to become JSON,
        # where a tuple would round-trip as a list anyway. Converting here
        # keeps what is written equal to what is read back.
        record["chain"] = [list(pair) for pair in exc.attempts]
        if exc.__cause__ is not None:
            record["cause"] = {"type": type(exc.__cause__).__name__,
                               "message": str(exc.__cause__)[:500]}
    return record


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
        # D-20 guard (M-2: shared with api/server.py via
        # assembly.reject_if_thread_in_use — see its docstring for why).
        prior_query = reject_if_thread_in_use(app, config)
        if prior_query:
            print(
                f"[thread-id '{thread_id}' already holds a run for "
                f"\"{prior_query}\". Re-invoking it with a new query "
                f"ACCUMULATES the old run's evidence and counters instead of "
                f"replacing them (D-20). Use a fresh --thread-id, or omit the "
                f"flag to get a generated one.]",
                file=sys.stderr)
            return 3
        # Wall-clock timer for the "HITL triggered | Total wall time: ..."
        # line printed once the run finishes, below. time.monotonic(), not
        # time.time(): a wall-clock elapsed measurement must never go
        # backwards or jump if the system clock is adjusted mid-run (NTP
        # sync, DST) -- the same reasoning pause_started below already
        # uses time.time() for, since THAT measurement is reported on its
        # own (human review latency), not accumulated into a total.
        run_started = time.monotonic()
        hitl_triggered = False
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
            hitl_triggered = True
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

        # D-113: the flush that used to sit here has moved into main()'s
        # finally block, which is now the SINGLE flush site. D-100 added
        # the finally flush and left this one in place, reasoning that
        # flush_narrative POPS the buffer so a second call finds nothing.
        # That is true only if nothing is buffered BETWEEN the two calls
        # -- and the finally block deliberately emits events
        # (checkpointer.closed, llm.close_failed, mcp.bridge_closed),
        # which is the very reason D-100 put its flush last.
        #
        # So on a SUCCESSFUL run this one wrote the complete narrative and
        # the finally one reopened the file in "w" mode and overwrote it
        # with the single close event. Live (runs p205.260/.261): both
        # logs/run-*.txt were 25 lines containing nothing but
        # "Checkpointer closed", and the console printed the same trace
        # path twice. D-100 fixed the failure path and broke the success
        # path in the same change; the two halves of its own comment
        # contradicted each other and neither test caught it, because
        # every test used a fake tracer that counted calls instead of a
        # real buffer that could be emptied.

        # .get(), not [] — an interrupted-then-abandoned run, or any path that
        # ends without reaching telemetry_node, left these keys absent and
        # turned a degraded run into a bare KeyError traceback.
        telemetry = result.get("telemetry") or {}
        report = result.get("final_report", "(no report was produced)")
        # The banner matches the "=== ... ===" convention this file already
        # uses for HUMAN REVIEW REQUIRED above; the report was the only
        # output printed with no label at all. Leading newline so it
        # separates cleanly from whatever stderr last wrote to the terminal.
        print("\n=== FINAL REPORT ===")
        print(report)
        print(_fmt_result_summary(telemetry, report))
        # Total wall-clock time from the first app.invoke() call to the
        # run actually finishing; hitl_triggered reports whether the run
        # paused for a human at all -- a fast HITL run (instant approve)
        # and a slow non-HITL run (a genuinely long gather loop) are both
        # real, distinct facts this line surfaces. See
        # _fmt_hitl_wall_time_line's own docstring for why elapsed_s
        # includes time spent blocked on the human.
        elapsed_s = time.monotonic() - run_started
        print(_fmt_hitl_wall_time_line(hitl_triggered, elapsed_s))
        print("\n--- telemetry (full) ---")
        print(json.dumps(telemetry, indent=2))
        # D-103: .get("recall") without a default. The old 0.0 fallback
        # wrote a number nothing measured into the recall column, and a
        # run that genuinely retrieved nothing recorded the identical
        # value. The column is nullable; NULL is what "not measured"
        # looks like.
        record_run(settings.postgres_dsn, thread_id, args.query,
                   telemetry.get("recall"), telemetry)

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
        # D-103: the single point every failing run passes through exactly
        # once. Deliberately HERE and not in main()'s except clauses: a
        # ProviderChainExhausted would pass through this block AND the
        # handler above, double-recording the same run, and an exception
        # main() does not recognise would never be recorded at all.
        #
        # Guarded on args.query because the column is TEXT NOT NULL and
        # `--print-graph` with no query is a legitimate way to reach this
        # function. Nothing ran in that case, so there is nothing to
        # record.
        #
        # record_failed_run swallows its own database errors exactly as
        # record_run does, so this cannot mask the original exception --
        # which is re-raised on the next line, unchanged.
        if args.query is not None:
            record_failed_run(settings.postgres_dsn, thread_id, args.query,
                              _failure_record(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
