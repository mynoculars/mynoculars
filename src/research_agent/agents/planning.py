"""
agents/planning.py — Phase 1 nodes: classify, remember, compose goals, expand.

Purpose:
    The four Plan-phase node functions. Each is a plain function taking
    ResearchState and returning a partial-update dict (LangGraph convention)
    — no classes, no hidden state, deliberately boring.

Responsibilities:
    - classify_node: intent classification (feeds goal composition).
    - memory_retrieve_node: pull relevant long-term memory into evidence
      (D-24) BEFORE goals are composed so past findings can shape them.
    - goal_manager_node: compose the goal set; zero goals is a legal output
      handled by graph routing (D-21), never an exception here.
    - task_expander_node: decompose goals into a ranked, capped, dedup-
      filtered task backlog (D-13/D-2/D-16).

Design decision (nodes as closures over dependencies):
    Each build_* function binds its dependencies (router, memory, settings)
    and returns the node fn. Alternatives: a DI container (magic) or module
    globals (untestable). Closures keep wiring visible in graph.py and make
    every node trivially testable with fakes.
"""

import logging
from typing import Any, Callable, Dict

from research_agent.agents.task_utils import cap_and_filter
from research_agent.config import Settings
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import log_event
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.prompts import templates
from research_agent.state import Goal, ResearchState, SearchTask

logger = logging.getLogger(__name__)

NodeFn = Callable[[ResearchState], Dict[str, Any]]


def build_classify_node(router: FallbackRouter) -> NodeFn:
    """Node: classify the query intent. Writes classification + counters."""

    def classify_node(state: ResearchState) -> Dict[str, Any]:
        result = router.complete_json(templates.classify(state.raw_query))
        log_event(logger, "node.classify", intent=result.get("intent"))
        return {"classification": result, "counters": {"llm_calls": 1}}

    return classify_node


def build_memory_retrieve_node(memory: SemanticMemory) -> NodeFn:
    """Node: retrieve long-term memory as evidence (source='memory')."""

    def memory_retrieve_node(state: ResearchState) -> Dict[str, Any]:
        recalled = memory.retrieve(state.raw_query)
        return {"evidence": recalled,
                "counters": {"memory_hits": float(len(recalled))}}

    return memory_retrieve_node


def build_goal_manager_node(router: FallbackRouter, settings: Settings) -> NodeFn:
    """Node: compose research goals, informed by memory and any human
    redirect guidance (E1 escalation, D-23)."""

    def goal_manager_node(state: ResearchState) -> Dict[str, Any]:
        hints = [e.content[:150] for e in state.evidence if e.source == "memory"]
        result = router.complete_json(templates.compose_goals(
            state.raw_query, state.classification.get("intent", "Unknown"), hints,
            guidance=state.human_guidance))
        goals = [Goal(goal_id=g["goal_id"], description=g["description"])
                 for g in result.get("goals", [])]
        update: Dict[str, Any] = {"goals": goals,
                                  "human_guidance": "",  # consumed; never reused stale
                                  "counters": {"llm_calls": 1,
                                               "composed_goals": float(len(goals))}}
        if not goals:
            # D-21: record, don't raise — routing sends this to the compiler
            # for an explicit error report. (Human escalation is the full
            # design's response; stubbed to a log line in this core build.)
            update["planning_error"] = "Goal composition produced zero goals."
            if settings.hitl_enabled:
                # D-23: the CHECK sets the trigger — routing functions are
                # read-only in LangGraph and cannot write state.
                update["escalation_trigger"] = "E1"
            else:
                log_event(logger, "escalation.stub", level=logging.WARNING,
                          trigger="E1", reason="zero_goals")
        return update

    return goal_manager_node



def build_task_expander_node(router: FallbackRouter, settings: Settings) -> NodeFn:
    """Node: expand goals into the initial ranked, capped task backlog."""

    def task_expander_node(state: ResearchState) -> Dict[str, Any]:
        result = router.complete_json(
            templates.expand_tasks(state.goals, settings.max_fanout))
        tasks = cap_and_filter(result.get("tasks", []), state, depth=0,
                                max_fanout=settings.max_fanout)
        log_event(logger, "node.expand", produced=len(tasks))
        # D-2: replace-on-write — this IS the whole backlog for next dispatch.
        return {"pending_tasks": tasks, "counters": {"llm_calls": 1}}

    return task_expander_node
