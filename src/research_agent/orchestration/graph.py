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
# ---------------------------------------------------------------------------


def route_after_goals(state: ResearchState
                      ) -> Literal["task_expander", "compiler", "human_escalation"]:
    """D-21 + D-23: zero goals -> human review (HITL on) or error report."""
    if state.escalation_trigger == "E1":
        return "human_escalation"
    return "compiler" if state.planning_error else "task_expander"


def dispatch_tasks(state: ResearchState
                   ) -> Union[List[Send], Literal["compiler", "human_escalation"]]:
    """D-1: fan out one Send per pending task; empty backlog -> compiler.

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
    """D-14 point 1: recall/depth only — NEVER the backlog, which is stale
    here (it still holds the just-dispatched tasks). The backlog is judged
    at dispatch time, on fresh data. Two termination points, two truths."""
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
    """D-22: pass -> persist; fail with budget -> grounded rewrite;
    fail exhausted -> telemetry directly (E4 stub — memory is NEVER fed
    from a report that failed its own quality bar)."""
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
    """Assemble and compile the workflow.

    Parameters:
        router: LLM routing (real or stub).
        tool: the retrieval tool workers invoke.
        memory: semantic memory (may be degraded/off).
        settings: graph bounds and thresholds.
        checkpointer: LangGraph checkpointer (Postgres or MemorySaver).

    Returns:
        A compiled LangGraph app; invoke with a thread_id config.
    """
    g = StateGraph(ResearchState)

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

    g.add_edge(START, "classify")
    g.add_edge("classify", "memory_retrieve")
    g.add_edge("memory_retrieve", "goal_manager")
    g.add_conditional_edges("goal_manager", route_after_goals,
                            ["task_expander", "compiler", "human_escalation"])
    g.add_conditional_edges("task_expander", dispatch_tasks,
                            ["search_worker", "compiler", "human_escalation"])
    g.add_edge("search_worker", "merger")
    g.add_edge("merger", "progress_checker")
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

    return g.compile(checkpointer=checkpointer)
