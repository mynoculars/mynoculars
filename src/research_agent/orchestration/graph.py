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

from typing import Any, List, Literal, Union

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
from research_agent.config import Settings
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.state import ResearchState, WorkerPayload

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
        return "human_escalation"
    return "compiler" if state.planning_error else "task_expander"


def dispatch_tasks(state: ResearchState
                   ) -> Union[List[Send], Literal["compiler", "human_escalation"]]:
    """Wired to TWO different conditional edges below — after
    task_expander (the first pass) and after gap_generator (every later
    pass). Same backlog-to-workers logic serves both, so the two producers
    can never accidentally be dispatched differently.

    READS   state.escalation_trigger, state.pending_tasks.
    RETURNS "human_escalation" if gap_generator just raised E2/E3 (it ran
            out of new tasks to try — see agents/gathering.py); otherwise
            "compiler" if the backlog is empty (D-1 — a graph with nothing
            left to search still needs to produce a report); otherwise a
            LIST of Send objects, one per pending task, which LangGraph
            executes as a batch of PARALLEL search_worker invocations, all
            in the same superstep.

    Producers already capped/ranked the backlog (D-13), so dispatch is
    always total — no truncation decisions happen here by design.
    """
    if state.escalation_trigger in ("E2", "E3"):
        return "human_escalation"  # gap generator exhausted its supply (D-23)
    if not state.pending_tasks:
        return "compiler"
    return [Send("search_worker", WorkerPayload(task=t)) for t in state.pending_tasks]


def route_convergence(state: ResearchState, settings: Settings
                      ) -> Literal["compiler", "gap_generator", "human_escalation"]:
    """Runs immediately after progress_checker, every single gather-loop
    cycle. This is the fork that decides whether the loop continues.

    READS   state.escalation_trigger, state.recall_score,
            state.iteration_depth, settings.recall_target,
            settings.max_depth.
    RETURNS "human_escalation" if progress_checker just raised E2/E3;
            otherwise "compiler" if recall has reached target OR the depth
            budget is spent (either way, there's nothing more useful to
            gain by looping again); otherwise "gap_generator" — go round
            again.

    D-14 point 1: recall/depth only — NEVER the backlog, which is stale
    here (it still holds the just-dispatched tasks). The backlog is judged
    at dispatch time, on fresh data. Two termination points, two truths.
    """
    if state.escalation_trigger in ("E2", "E3"):
        return "human_escalation"  # D-23; checker set the trigger
    if state.recall_score >= settings.recall_target:
        return "compiler"
    if state.iteration_depth >= settings.max_depth:
        return "compiler"
    return "gap_generator"


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
        return "human_escalation"  # D-23; critic set the trigger
    if state.critique_passed:
        return "memory_writer"
    if state.revision_count < settings.max_revisions:
        return "compiler"
    return "telemetry"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_graph(router: FallbackRouter, tool: ToolFn, memory: SemanticMemory,
                settings: Settings, checkpointer: Any):
    """Assemble and compile the workflow — the ONE function that turns 13
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
        tool: the retrieval tool workers invoke.
        memory: semantic memory (may be degraded/off).
        settings: graph bounds and thresholds.
        checkpointer: LangGraph checkpointer (Postgres or MemorySaver).
    """
    # StateGraph(ResearchState) creates a new, empty graph builder whose
    # shared state will be validated against the ResearchState Pydantic
    # model at every step — this is where LangGraph learns which fields
    # exist and, via the Annotated[...] hints on some of them (see
    # state.py), which reducer function to use if two parallel nodes both
    # write the same field in one superstep.
    g = StateGraph(ResearchState)

    # Each line below registers one node under a string name. The object
    # passed in (e.g. build_classify_node(router)) is the ACTUAL function
    # LangGraph will call when it's that node's turn to run — remember from
    # agents/planning.py's docstring that build_classify_node(router)
    # doesn't run classify_node itself; it returns the ready-to-call
    # classify_node function, already holding onto `router` via closure.
    g.add_node("classify", build_classify_node(router))
    g.add_node("memory_retrieve", build_memory_retrieve_node(memory))
    g.add_node("goal_manager", build_goal_manager_node(router, settings))
    g.add_node("task_expander", build_task_expander_node(router, settings))
    g.add_node("search_worker", build_search_worker(tool))
    g.add_node("merger", build_merger_node())
    g.add_node("progress_checker", build_progress_checker_node(settings))
    g.add_node("gap_generator", build_gap_generator_node(router, settings))
    g.add_node("compiler", build_compiler_node(router))
    g.add_node("critic", build_critic_node(router, settings))
    g.add_node("memory_writer", build_memory_writer_node(memory))
    g.add_node("telemetry", build_telemetry_node())
    # D-23/D-28: single parametrized escalation node. It returns Command
    # (goto inferred from its type hint), so no static edges are added.
    g.add_node("human_escalation", build_escalation_node(settings))

    # g.add_edge(from_node, to_node) wires a FIXED, unconditional edge:
    # whenever `from_node` finishes, always run `to_node` next, no decision
    # involved. Contrast with g.add_conditional_edges further down, which
    # is used wherever the diagram in the module docstring shows a fork.
    g.add_edge(START, "classify")
    g.add_edge("classify", "memory_retrieve")
    g.add_edge("memory_retrieve", "goal_manager")
    # See the module docstring for exactly what add_conditional_edges does:
    # after "goal_manager" runs, call route_after_goals(state) and go to
    # whichever of the three listed node names it returns.
    g.add_conditional_edges("goal_manager", route_after_goals,
                            ["task_expander", "compiler", "human_escalation"])
    g.add_conditional_edges("task_expander", dispatch_tasks,
                            ["search_worker", "compiler", "human_escalation"])
    g.add_edge("search_worker", "merger")
    g.add_edge("merger", "progress_checker")
    # route_convergence needs BOTH state and settings, but LangGraph only
    # ever passes ONE argument (state) to a routing function — the lambda
    # here "pre-fills" settings from this build_graph() call's own
    # argument, so LangGraph can call `lambda s: ...` with just `s` and get
    # the full two-argument call it actually needs underneath. See the
    # module docstring's lambda explanation for more detail.
    g.add_conditional_edges("progress_checker",
                            lambda s: route_convergence(s, settings),
                            ["compiler", "gap_generator", "human_escalation"])
    g.add_conditional_edges("gap_generator", dispatch_tasks,
                            ["search_worker", "compiler", "human_escalation"])
    g.add_edge("compiler", "critic")
    g.add_conditional_edges("critic",
                            lambda s: route_after_critique(s, settings),
                            ["compiler", "memory_writer", "telemetry",
                             "human_escalation"])
    g.add_edge("memory_writer", "telemetry")
    g.add_edge("telemetry", END)

    # g.compile(...) turns the builder (which has just been describing
    # nodes and edges so far) into an actual runnable app. `checkpointer`
    # is what makes state durable across invoke() calls under the same
    # thread_id — without it, a paused (interrupted) run would have
    # nowhere to save its progress and could never be resumed.
    return g.compile(checkpointer=checkpointer)
