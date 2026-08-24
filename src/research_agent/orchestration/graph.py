"""
orchestration/graph.py — The workflow topology: nodes, edges, routing.

Purpose:
    Wire every node into the fixed LangGraph topology from the design doc
    (v3.1 §2.1, core subset) and expose one build function.

Responsibilities:
    - The three routing functions (goal validation D-21, task dispatch D-1,
      convergence D-14, critique loop D-22) — small, pure, unit-testable.
    - build_graph(): dependency-injected assembly, so tests wire fakes and
      the CLI wires real services through the same single entry point.

Topology (core build):

    START -> classify -> memory_retrieve -> goal_manager
        goal_manager --(zero goals, D-21)--------------------> compiler
        goal_manager --(goals present)--> task_expander
        task_expander --(dispatch D-1: empty)----------------> compiler
        task_expander --(tasks)--> search_worker (xN via Send)
        search_worker -> merger -> progress_checker
        progress_checker --(converged/depth, D-14)-----------> compiler
        progress_checker --(expand)--> gap_generator
        gap_generator --(dispatch D-1: empty)----------------> compiler
        gap_generator --(tasks)--> search_worker (loop)
        compiler -> critic
        critic --(fail, budget remains, D-22)----------------> compiler
        critic --(pass)--> memory_writer -> telemetry -> END
        critic --(fail, exhausted: E4 stub)------------------> telemetry

    Termination is guaranteed by four independent bounds (design §6.3):
    depth counter, dedup-finite task supply, empty-backlog fallthrough,
    and the invoke-time recursion_limit backstop.

This file is the single best place to start reading this codebase, because
it is a literal, executable version of the ASCII diagram above: every node
name, every arrow, and every branch condition in that diagram corresponds
to exactly one line below.

Python mechanics used in this file, if any of this is new to you:
    Literal["a", "b", "c"]  /  Union[X, Y]
        Both are TYPE HINTS, not executable logic — see agents/escalation.py
        for what Literal means. Union[X, Y] means "a value of type X, OR a
        value of type Y" — e.g. dispatch_tasks below is declared to return
        EITHER a list of Send objects OR one of two literal strings,
        because it genuinely does one or the other depending on state.
    g.add_conditional_edges("node_name", routing_fn, [...])
        This is how LangGraph wires up a DECISION POINT rather than a fixed
        "always go to the next node" edge. After "node_name" finishes,
        LangGraph calls routing_fn(state) and uses whatever STRING that
        function returns as the name of the next node to run. The list
        argument (e.g. ["task_expander", "compiler", "human_escalation"])
        is just a declaration, for LangGraph's own validation and for
        anyone reading the graph, of every possible destination that
        routing_fn is allowed to return — it does not itself decide
        anything.
    lambda s: route_convergence(s, settings)
        A LAMBDA is a tiny, unnamed, inline function — "lambda s: EXPR"
        means "given one argument named s, evaluate and return EXPR".
        This one exists purely for a plumbing reason: LangGraph calls a
        routing function with exactly one argument (the current state),
        but route_convergence (defined below) needs TWO arguments — state
        AND settings. The lambda "bakes in" `settings` (captured from the
        surrounding build_graph() function, the same closure mechanism
        explained in agents/planning.py) so that LangGraph only ever has
        to supply the one argument it knows how to supply.
    [Send(...) for t in state.pending_tasks]
        A LIST COMPREHENSION: it builds a new list by looping over
        state.pending_tasks and, for each task `t`, constructing one
        Send(...) object. It is exactly equivalent to writing:
            result = []
            for t in state.pending_tasks:
                result.append(Send("search_worker", WorkerPayload(task=t)))
        just written on one line. Send(node_name, payload) is a LangGraph
        primitive meaning "run `node_name` once with this exact payload,
        as one of possibly many PARALLEL invocations in the same
        superstep" — this is the mechanism that turns N pending tasks into
        N simultaneous search_worker executions.
"""

from typing import Any, List, Literal, Optional, Union
import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_agent.agents.escalation import build_escalation_node
from research_agent.agents.compilation import (build_compiler_node, build_critic_node,
                                               build_memory_writer_node, build_telemetry_node)
from research_agent.agents.gathering import (ToolFn, build_gap_generator_node,
                                             build_merger_node, build_progress_checker_node,
                                             build_search_worker)
from research_agent.agents.planning import (build_classify_node, build_goal_manager_node,
                                            build_memory_retrieve_node, build_task_expander_node)
from research_agent import langfuse as lf
from research_agent.config import Settings
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import log_event
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.state import ResearchState, WorkerPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing functions (pure — they READ state, they never write it)
#
# Every function in this section has the same job: look at a few fields on
# ResearchState and return the NAME (a string) of the node that should run
# next. None of them ever modify `state` or call any LLM/store — they are
# ordinary, side-effect-free Python functions, which is exactly why they can
# be unit-tested directly (construct a ResearchState by hand, call the
# function, assert on the string it returns) without running the graph at
# all.
# ---------------------------------------------------------------------------


def route_after_goals(state: ResearchState
                      ) -> Literal["task_expander", "compiler", "human_escalation"]:
    """Runs immediately after goal_manager. D-21 + D-23: zero goals ->
    human review (HITL on) or straight to an error report.

    READS   state.escalation_trigger (set by goal_manager_node when it
            produced zero goals AND HITL is enabled — see agents/
            planning.py), state.planning_error (set whenever it produced
            zero goals, regardless of HITL).
    RETURNS "human_escalation" if the E1 trigger fired; otherwise
            "compiler" if there was a planning error (so the run still
            ends with an explicit report instead of silently vanishing);
            otherwise "task_expander" — the normal, everyday path.
    """
    if state.escalation_trigger == "E1":
        to_node, reason = "human_escalation", "E1 escalation trigger fired (zero goals, HITL on)"
    elif state.planning_error:
        to_node, reason = "compiler", "goal composition produced zero goals (no escalation)"
    else:
        to_node, reason = "task_expander", "goals present"
    log_event(logger, "route.decision", from_node="goal_manager", to_node=to_node,
              reason=reason, escalation_trigger=state.escalation_trigger,
              goal_count=len(state.goals))
    return to_node


def dispatch_tasks(state: ResearchState, hint_to_node: dict, from_node: str = "task_expander"
                   ) -> Union[List[Send], Literal["compiler", "human_escalation"]]:
    """Wired to TWO different conditional edges below — after
    task_expander (the first pass) and after gap_generator (every later
    pass). Same backlog-to-workers logic serves both, so the two producers
    can never accidentally be dispatched differently.

    READS   state.escalation_trigger, state.pending_tasks, each task's own
            .tool_hint (P2-14, D-25).
    CALLED  via a lambda in build_graph, below (`lambda s: dispatch_tasks(s,
            hint_to_node, from_node="task_expander")` /
            `..., from_node="gap_generator")`) — same "pre-fill an extra
            argument LangGraph itself never passes" pattern route_convergence's
            own call site already uses. from_node exists ONLY so the
            route.decision log line below can say which of the two call
            sites fired (see logging_setup.py's NarrativeFormatter — a
            "why did we go to search_worker" question should never need
            the reader to check which edge led here); it plays no role in
            the actual dispatch decision.
    RETURNS "human_escalation" if gap_generator just raised E2/E3 (it ran
            out of new tasks to try — see agents/gathering.py); otherwise
            "compiler" if the backlog is empty (D-1 — a graph with nothing
            left to search still needs to produce a report); otherwise a
            LIST of Send objects, one per pending task, which LangGraph
            executes as a batch of PARALLEL worker invocations, all in the
            same superstep — POSSIBLY split across more than one node
            NAME now (P2-14): hint_to_node.get(t.tool_hint, "search_worker")
            sends a task to its requested specialist if one is wired into
            THIS graph, else the default "search_worker", covering both
            "no hint" (the overwhelming common case, t.tool_hint == "") and
            "hint given but this graph doesn't have that specialist" —
            the second case shouldn't be reachable in practice
            (task_utils.py::cap_and_filter already only ever sets a hint
            to something present in hint_to_node's own key set), but the
            fallback costs nothing and means a mismatch degrades instead
            of crashing, matching this codebase's consistent posture
            everywhere else a lookup could plausibly miss.

    Producers already capped/ranked the backlog (D-13), so dispatch is
    always total — no truncation decisions happen here by design.
    """
    if state.escalation_trigger in ("E2", "E3"):
        to_node, reason = "human_escalation", f"{state.escalation_trigger} raised (task supply exhausted)"
    elif not state.pending_tasks:
        to_node, reason = "compiler", "empty backlog (D-1)"
    else:
        to_node, reason = "search_worker (parallel)", f"{len(state.pending_tasks)} task(s) to dispatch"
    log_event(logger, "route.decision", from_node=from_node, to_node=to_node,
              reason=reason, escalation_trigger=state.escalation_trigger,
              pending_tasks=len(state.pending_tasks))
    # S-4: branch on the SAME decision just logged, rather than
    # re-evaluating the identical conditions a second time -- one place
    # decides the routing policy, this just acts on it.
    if to_node == "human_escalation":
        return "human_escalation"  # gap generator exhausted its supply (D-23)
    if to_node == "compiler":
        return "compiler"
    return [Send(hint_to_node.get(t.tool_hint, "search_worker"), WorkerPayload(task=t))
            for t in state.pending_tasks]


def route_convergence(state: ResearchState, settings: Settings
                      ) -> Literal["compiler", "gap_generator", "human_escalation"]:
    """Runs immediately after progress_checker, every single gather-loop
    cycle. This is the fork that decides whether the loop continues.

    READS   state.escalation_trigger, state.recall_score,
            state.grounded_score, state.grounded_score_prev,
            state.iteration_depth, settings.recall_target,
            settings.grounded_recall_target, settings.max_depth.
    RETURNS "human_escalation" if progress_checker just raised E2/E3;
            otherwise "compiler" if recall has reached target AND that
            coverage is adequately grounded (Guardrail G2), OR grounding
            has stalled across a cycle (S-8, below), OR the depth budget
            is spent; otherwise "gap_generator" — go round again.

    D-14 point 1: recall/depth only — NEVER the backlog, which is stale
    here (it still holds the just-dispatched tasks). The backlog is judged
    at dispatch time, on fresh data. Two termination points, two truths.

    Guardrail G2 (grounded convergence): recall alone answers "is every
    goal covered by SOMETHING"; it does not distinguish a real document
    from the model's own recollection (source="model"). Live evidence
    (run p205.131-check) shows recall reaching 1.0 with corpus_recall at
    0.0 — every goal "covered" by MCP/model tiers, nothing from the
    ingested corpus — which then shipped straight to the compiler and
    was rejected twice by the critic on fabricated figures, burning the
    full revision budget before escalating. This adds a THIRD truth
    alongside recall/depth: if recall has reached target but
    grounded_score has not, and depth budget still remains, spend it on
    another gather cycle before compiling — same as the "recall below
    target" branch below, just gated on a different signal. Once depth
    IS spent, this still falls through to "compiler" regardless of
    grounding — there is no budget left to spend chasing it further, and
    an ungrounded-but-complete draft is still better data for the critic
    (and, if it fails, for a human at E4) than no draft at all.

    S-8 (grounding-stall exit): the branch above sends the run back to
    gap_generator for ANOTHER cycle every time grounding is short — but
    against an off-topic corpus (armies query, Redis corpus is the
    live-evidenced case) grounded_score cannot move no matter how many
    times gap_generator is asked, and nothing previously noticed. Live
    trace: `grounded 0.00` at depth 1, still `0.00` at depth 2, still
    `0.00` at depth 3 — three consecutive laps, 6 extra LLM calls, 57 web
    fetches, 223.7s spent arriving exactly where the run already was at
    depth 1. Grounding gets exactly ONE gap_generator attempt (the branch
    above, unchanged) before this stall check applies: if grounded_score
    did NOT increase between the previous cycle and this one (and there
    WAS a previous cycle — grounded_score_prev's -1.0 sentinel excludes
    the very first below-target measurement, which deserves its one
    attempt), route to compiler instead of spending remaining depth
    budget on a condition already shown not to move.
    """
    if state.escalation_trigger in ("E2", "E3"):
        to_node, reason = "human_escalation", f"{state.escalation_trigger} raised by progress_checker"
    elif state.iteration_depth >= settings.max_depth:
        to_node, reason = "compiler", f"depth {state.iteration_depth}/{settings.max_depth} budget spent, recall {state.recall_score:.2f} (grounded {state.grounded_score:.2f})"
    elif (state.recall_score >= settings.recall_target
          and state.grounded_score < settings.grounded_recall_target
          and state.grounded_score_prev >= 0.0
          and state.grounded_score <= state.grounded_score_prev):
        to_node, reason = "compiler", f"grounding stalled: {state.grounded_score:.2f} did not improve on {state.grounded_score_prev:.2f}, stopping rather than spending remaining depth {state.iteration_depth}/{settings.max_depth} chasing it"
        log_event(logger, "convergence.grounding_stalled", level=logging.WARNING,
                  grounded=round(state.grounded_score, 3),
                  grounded_prev=round(state.grounded_score_prev, 3),
                  depth=state.iteration_depth, max_depth=settings.max_depth)
    elif state.recall_score >= settings.recall_target and state.grounded_score < settings.grounded_recall_target:
        to_node, reason = "gap_generator", f"recall {state.recall_score:.2f} reached target but grounded {state.grounded_score:.2f} below {settings.grounded_recall_target:.2f}, depth {state.iteration_depth}/{settings.max_depth} remains"
    elif state.recall_score >= settings.recall_target:
        to_node, reason = "compiler", f"recall {state.recall_score:.2f} reached target {settings.recall_target:.2f}, grounded {state.grounded_score:.2f}"
    else:
        to_node, reason = "gap_generator", f"recall {state.recall_score:.2f} below target {settings.recall_target:.2f}, depth {state.iteration_depth}/{settings.max_depth} remains"
    log_event(logger, "route.decision", from_node="progress_checker", to_node=to_node,
              reason=reason, escalation_trigger=state.escalation_trigger,
              recall=round(state.recall_score, 3), grounded=round(state.grounded_score, 3),
              depth=state.iteration_depth, max_depth=settings.max_depth,
              recall_target=settings.recall_target,
              grounded_recall_target=settings.grounded_recall_target)
    return to_node


def route_after_critique(state: ResearchState, settings: Settings
                         ) -> Literal["compiler", "memory_writer", "telemetry",
                                      "human_escalation"]:
    """Runs immediately after critic, every pass through the compile/
    critique loop.

    READS   state.escalation_trigger, state.critique_passed,
            state.revision_count, settings.max_revisions.
    RETURNS "human_escalation" if critic just raised E4; otherwise
            "memory_writer" if the critique passed (report is good enough
            to learn from); otherwise "compiler" if there is still
            revision budget left (grounded rewrite, D-22 — the critic's
            notes travel with it, see agents/compilation.py); otherwise
            "telemetry" directly — the run ends WITHOUT ever reaching
            memory_writer, because a report that failed its own quality
            bar is never fed into long-term memory (D-24).
    """
    if state.escalation_trigger == "E4":
        to_node, reason = "human_escalation", "E4 raised by critic"
    elif state.critique_passed:
        to_node, reason = "memory_writer", "critique passed"
    elif state.revision_count < settings.max_revisions:
        to_node, reason = "compiler", f"critique failed, revision {state.revision_count}/{settings.max_revisions} budget remains"
    else:
        to_node, reason = "telemetry", f"critique failed, revision budget {settings.max_revisions} exhausted (report NOT sent to memory_writer, D-24)"
    log_event(logger, "route.decision", from_node="critic", to_node=to_node,
              reason=reason, escalation_trigger=state.escalation_trigger,
              critique_passed=state.critique_passed, revision_count=state.revision_count,
              max_revisions=settings.max_revisions)
    return to_node


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_graph(router: FallbackRouter, tool: ToolFn, memory: SemanticMemory,
                settings: Settings, checkpointer: Any, debug: bool = False,
                mcp_tool: Optional[ToolFn] = None):
    """Assemble and compile the workflow — the ONE function that turns
    independently-testable node functions into a single runnable graph.

    CALLED BY   cli.py::build_app_and_settings — the only call site in the
                whole codebase. Both the CLI and the API import and call
                that same function, so they end up with an identical graph;
                this file is never invoked with different wiring for the
                two interfaces.
    RETURNS     a compiled LangGraph app — an object with an .invoke(state,
                config) method. Nothing runs yet at the point this function
                returns; invoke() is what actually executes nodes.

    Parameters:
        router: LLM routing (real or stub).
        tool: the DEFAULT retrieval tool every plain SearchTask (no
            tool_hint) is dispatched to — always required, unchanged from
            before P2-14.
        memory: semantic memory (may be degraded/off).
        settings: graph bounds and thresholds.
        checkpointer: LangGraph checkpointer (Postgres or MemorySaver).
        debug: when True, every node logs a "node.enter" line the instant
            it starts running — including merger/progress_checker, which
            make no LLM or store call and so never appear in a --debug
            TRACE FILE (tracing.py's Tracer only ever records LLM calls and
            store searches, never "a node ran"). This flag is a second,
            independent signal from that one, threaded down from
            cli.py::build_app_and_settings, where it is set to
            tracer.enabled — so in practice both turn on together, but this
            one reaches every node, not just the ones that touch an LLM or
            a store.
        mcp_tool: (P2-14, D-25) an OPTIONAL second tool, routed to only
            when a SearchTask's tool_hint == "mcp" -- see cli.py, which
            builds this (tools/mcp_client.py::make_mcp_tool) exactly when
            settings.mcp_enabled, and None otherwise (the default). None
            here means this build_graph call registers NO extra node at
            all -- the graph is byte-for-byte the same shape it was
            before P2-14 existed, not just "the extra node happens to go
            unused." A task with tool_hint="mcp" arriving at dispatch_tasks
            when this is None still degrades safely to the default
            worker (see dispatch_tasks's own docstring) -- but that
            shouldn't happen in practice, since cap_and_filter only ever
            sets a hint the SAME settings.mcp_enabled check already
            confirmed is active for this run.
    """
    # StateGraph(ResearchState) creates a new, empty graph builder whose
    # shared state will be validated against the ResearchState Pydantic
    # model at every step — this is where LangGraph learns which fields
    # exist and, via the Annotated[...] hints on some of them (see
    # state.py), which reducer function to use if two parallel nodes both
    # write the same field in one superstep.
    g = StateGraph(ResearchState)
    log_event(logger, "graph.state_created", state_model="ResearchState")

    # Each line below registers one node under a string name. The object
    # passed in (e.g. build_classify_node(router)) is the ACTUAL function
    # LangGraph will call when it's that node's turn to run — remember from
    # agents/planning.py's docstring that build_classify_node(router)
    # doesn't run classify_node itself; it returns the ready-to-call
    # classify_node function, already holding onto `router` via closure.
    # Phase 3: every node is wrapped with a Langfuse span via
    # lf.traced_node -- ONE change point here, not thirteen changes across
    # agents/*.py, and it changes nothing about what any node itself does
    # or returns (see langfuse/helpers.py::traced_node's own docstring).
    # `lf.get_observer` is passed as a callable, not `lf.get_observer()`,
    # so the wrapper always uses whichever Observer is active at CALL
    # time -- important for tests, which build a fresh Observer per test.
    def _tn(name, fn):
        return lf.traced_node(lf.get_observer, name, fn)

    # Three tiny wrappers around g.add_node/g.add_edge/g.add_conditional_edges
    # — this is the fix for the biggest gap the logging design identified:
    # build_graph() previously had ZERO log_event calls of its own, so
    # nothing ever recorded which nodes/edges a given build actually wired
    # (13 vs 14 nodes depending on mcp_tool, worker_destinations varying the
    # same way). One log_event call per registration, here, covers every
    # g.add_node/g.add_edge/g.add_conditional_edges call below without
    # repeating a log_event call at each of the ~20 individual call sites.
    node_count = 0
    edge_count = 0
    conditional_count = 0

    def _add_node(name, fn):
        nonlocal node_count
        g.add_node(name, _tn(name, fn))
        node_count += 1
        log_event(logger, "graph.node_registered", node=name)

    def _edge(from_node, to_node):
        nonlocal edge_count
        g.add_edge(from_node, to_node)
        edge_count += 1
        log_event(logger, "graph.edge_registered", edge_type="fixed",
                  from_node=str(from_node), to_node=str(to_node))

    def _cedge(from_node, router_name, fn, destinations):
        nonlocal edge_count, conditional_count
        g.add_conditional_edges(from_node, fn, destinations)
        edge_count += 1
        conditional_count += 1
        log_event(logger, "graph.edge_registered", edge_type="conditional",
                  from_node=from_node, router=router_name, destinations=destinations)

    _add_node("classify", build_classify_node(router, debug))
    _add_node("memory_retrieve", build_memory_retrieve_node(memory, debug))
    _add_node("goal_manager", build_goal_manager_node(router, settings, debug))
    _add_node("task_expander", build_task_expander_node(router, settings, debug))
    _add_node("search_worker", build_search_worker(tool, debug))
    # P2-14 (D-25): the ONE additional specialist this build can wire in,
    # registered ONLY when cli.py actually built one (mcp_tool is not
    # None) -- build_search_worker is already fully tool-agnostic (see
    # its own docstring), so this is just a second call to the SAME
    # function with a DIFFERENT tool closure, not new worker logic.
    if mcp_tool is not None:
        _add_node("mcp_search_worker", build_search_worker(mcp_tool, debug))
    _add_node("merger", build_merger_node(router, settings, debug))  # P2-12
    _add_node("progress_checker", build_progress_checker_node(settings, debug))
    _add_node("gap_generator", build_gap_generator_node(router, settings, debug))
    _add_node("compiler", build_compiler_node(router, debug))
    _add_node("critic", build_critic_node(router, settings, debug))
    _add_node("memory_writer", build_memory_writer_node(memory, settings, debug))
    _add_node("telemetry", build_telemetry_node(settings, debug))
    # D-23/D-28: single parametrized escalation node. It returns Command
    # (goto inferred from its type hint), so no static edges are added.
    _add_node("human_escalation", build_escalation_node(settings, debug))

    # _edge/_cedge wire the SAME topology the module docstring's ASCII
    # diagram describes — see it for what a fixed vs conditional edge means.
    _edge(START, "classify")
    _edge("classify", "memory_retrieve")
    _edge("memory_retrieve", "goal_manager")
    _cedge("goal_manager", "route_after_goals", route_after_goals,
          ["task_expander", "compiler", "human_escalation"])
    # P2-14 (D-25): hint_to_node and worker_destinations are both built
    # HERE, once, from whatever specialists this particular build_graph
    # call actually registered above -- never a fixed, hardcoded set.
    # With mcp_tool=None (every run before P2-14, and every run today
    # with settings.mcp_enabled off), both are exactly what they always
    # were: hint_to_node={} and worker_destinations lists only
    # "search_worker" -- so dispatch_tasks and these two
    # _cedge calls are BYTE-IDENTICAL in behavior to before P2-14 existed
    # in that case.
    hint_to_node = {"mcp": "mcp_search_worker"} if mcp_tool is not None else {}
    worker_destinations = ["search_worker", "compiler", "human_escalation"]
    if mcp_tool is not None:
        worker_destinations.append("mcp_search_worker")
    # from_node="task_expander"/"gap_generator" below exists ONLY so
    # dispatch_tasks's own route.decision log line can say which of its two
    # call sites fired — see dispatch_tasks's docstring.
    _cedge("task_expander", "dispatch_tasks(task_expander)",
          lambda s: dispatch_tasks(s, hint_to_node, from_node="task_expander"),
          worker_destinations)
    _edge("search_worker", "merger")
    if mcp_tool is not None:
        _edge("mcp_search_worker", "merger")
    _edge("merger", "progress_checker")
    # route_convergence needs BOTH state and settings, but LangGraph only
    # ever passes ONE argument (state) to a routing function — the lambda
    # here "pre-fills" settings from this build_graph() call's own
    # argument, so LangGraph can call `lambda s: ...` with just `s` and get
    # the full two-argument call it actually needs underneath. See the
    # module docstring's lambda explanation for more detail.
    _cedge("progress_checker", "route_convergence",
          lambda s: route_convergence(s, settings),
          ["compiler", "gap_generator", "human_escalation"])
    _cedge("gap_generator", "dispatch_tasks(gap_generator)",
          lambda s: dispatch_tasks(s, hint_to_node, from_node="gap_generator"),
          worker_destinations)
    _edge("compiler", "critic")
    _cedge("critic", "route_after_critique",
          lambda s: route_after_critique(s, settings),
          ["compiler", "memory_writer", "telemetry", "human_escalation"])
    _edge("memory_writer", "telemetry")
    _edge("telemetry", END)

    # g.compile(...) turns the builder (which has just been describing
    # nodes and edges so far) into an actual runnable app. `checkpointer`
    # is what makes state durable across invoke() calls under the same
    # thread_id — without it, a paused (interrupted) run would have
    # nowhere to save its progress and could never be resumed.
    app = g.compile(checkpointer=checkpointer)
    log_event(logger, "graph.compiled", nodes=node_count, edges=edge_count,
              conditional_edges=conditional_count)
    return app
