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

If "closure" is a new word: every function below named build_something_node
does NOT do the actual work. It takes a few dependencies as arguments (e.g.
a `router` object that talks to an LLM), defines a SECOND, smaller function
INSIDE itself (e.g. `classify_node`), and returns that inner function
WITHOUT calling it. Because the inner function was defined inside the
outer one, it "remembers" the outer function's arguments even after the
outer function has already finished running — that remembered value is
what "closure" refers to. Concretely:

    router_instance = FallbackRouter(...)             # made once, at startup
    classify_node = build_classify_node(router_instance)
    # classify_node is now a function of ONE argument (state) that still
    # has access to router_instance, even though build_classify_node's own
    # local variables are long gone. LangGraph calls classify_node(state)
    # later, whenever that node's turn comes up in the running graph.

This is the entire "dependency injection" story in this codebase: no
framework, no container — just an outer function handing a few objects to
an inner function before returning it.
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

# Another type ALIAS (see gathering.py for the full explanation of what this
# syntax means) — "NodeFn" is shorthand for "a function that takes a
# ResearchState and returns a dict." Every LangGraph node in this codebase
# has that exact shape; naming it once here avoids repeating it everywhere.
NodeFn = Callable[[ResearchState], Dict[str, Any]]


def build_classify_node(router: FallbackRouter) -> NodeFn:
    """Node: classify the query intent. Writes classification + counters."""

    def classify_node(state: ResearchState) -> Dict[str, Any]:
        """First node the graph runs. No prior state to react to yet.

        READS   state.raw_query — the user's question, verbatim.
        CALLS   one LLM JSON call: "what kind of question is this?"
                (Comparison / Survey / Explanation / Diagnosis / Recommendation
                + a confidence score — see templates.classify for the schema).
        WRITES  state.classification = {"intent": ..., "confidence": ...}
                state.counters["llm_calls"] += 1
        NEXT    graph.py routes unconditionally to memory_retrieve.

        This is a cheap, low-stakes call: its only job is to give
        goal_manager (two nodes downstream) a label to shape its prompt
        with. Nothing downstream reads `confidence` — only `intent` is
        consumed later.
        """
        router.set_node("classify")
        result = router.complete_json(templates.classify(state.raw_query))
        log_event(logger, "node.classify", intent=result.get("intent"))
        return {"classification": result, "counters": {"llm_calls": 1}}

    return classify_node


def build_memory_retrieve_node(memory: SemanticMemory) -> NodeFn:
    """Node: retrieve long-term memory as evidence (source='memory')."""

    def memory_retrieve_node(state: ResearchState) -> Dict[str, Any]:
        """Runs right after classify, BEFORE any goal is composed.

        READS   state.raw_query.
        CALLS   memory.retrieve(query) — no LLM call. Internally this embeds
                the query, does a Qdrant similarity search against past runs'
                stored evidence, and reranks by (similarity x staleness-decay)
                — see memory/semantic_memory.py for that math. On a fresh
                install, or if Qdrant is unreachable, this returns [].
        WRITES  state.evidence += recalled items, each tagged source="memory"
                state.counters["memory_hits"] += len(recalled)
                (evidence uses operator.add as its reducer, so this is safe
                to accumulate even though later nodes also append to it)
        NEXT    graph.py routes unconditionally to goal_manager.

        WHY BEFORE GOALS: goal_manager reads memory evidence as a hint when
        composing this run's goals, so what a PAST run learned can steer
        what THIS run decides to research. That is the whole point of
        putting this node here rather than after goal composition.

        Recalled items later behave EXACTLY like fresh corpus evidence in
        every downstream rule (coverage, contradiction, context-building) —
        there is no separate code path for "memory" evidence past this
        point. See the memory goal_id caveat in memory/semantic_memory.py.
        """
        recalled = memory.retrieve(state.raw_query)
        return {"evidence": recalled,
                "counters": {"memory_hits": float(len(recalled))}}

    return memory_retrieve_node


def build_goal_manager_node(router: FallbackRouter, settings: Settings) -> NodeFn:
    """Node: compose research goals, informed by memory and any human
    redirect guidance (E1 escalation, D-23)."""

    def goal_manager_node(state: ResearchState) -> Dict[str, Any]:
        """The first genuinely agentic step: nobody wrote these goals down.

        READS   state.raw_query, state.classification["intent"],
                up to 5 memory-sourced evidence snippets (150 chars each,
                pulled out of state.evidence — the ones memory_retrieve
                just added),
                state.human_guidance — non-empty ONLY if we are re-entering
                this node after an E1 "redirect" from human_escalation.
        CALLS   one LLM JSON call asking for 2-5 concrete research goals.
        WRITES  state.goals = [Goal(goal_id="g1", description=...), ...]
                state.human_guidance = ""      (consumed; must not leak into
                                                 a later, unrelated call)
                state.counters["llm_calls"] += 1, ["composed_goals"] = n
                IF ZERO GOALS COME BACK (D-21 — this is a legal, expected
                outcome, not an exception):
                    state.planning_error = "Goal composition produced..."
                    IF hitl_enabled: state.escalation_trigger = "E1"
                        (the CHECK sets this; graph.py's routing function
                        only ever READS it — routers can't write state)
                    ELSE: just log a WARNING and carry on
        NEXT    graph.py's route_after_goals reads escalation_trigger and
                planning_error to decide: human_escalation (if E1) ->
                compiler (if planning_error, no escalation) ->
                task_expander (the normal path, if goals is non-empty).

        Caution if you touch this: `g["goal_id"]` / `g["description"]` below
        index the model's JSON directly with no validation. A live model
        that omits either key raises KeyError here and takes the whole run
        down — there is no producer-side equivalent of the worker's
        try/except-as-data pattern (see gathering.py's search_worker).
        """
        hints = [e.content[:150] for e in state.evidence if e.source == "memory"]
        router.set_node("goal_manager")
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
        """Turns the abstract goals into concrete search strings to execute.

        This is the FIRST task-producing node — depth=0. Its twin,
        gap_generator (gathering.py), produces tasks at every later depth
        using the same cap_and_filter hygiene function, so the two stay in
        lockstep by construction rather than by convention.

        READS   state.goals, settings.max_fanout.
        CALLS   one LLM JSON call: "write search queries for these goals,
                ranked by priority" (schema in templates.expand_tasks).
        WRITES  state.pending_tasks = [SearchTask, ...]
                    NOTE: this REPLACES the whole backlog (D-2), it does not
                    append — pending_tasks has no reducer, so only one node
                    is ever allowed to write it per superstep.
                state.counters["llm_calls"] += 1
        NEXT    graph.py's dispatch_tasks reads pending_tasks: empty ->
                compiler (D-1, never an empty Send list); non-empty -> one
                parallel search_worker per task.

        Before tasks reach pending_tasks they pass through
        task_utils.cap_and_filter, which applies three rules in one place:
            D-13  keep only the top settings.max_fanout by priority — the
                  PRODUCER decides what to drop, never the dispatcher.
            D-2   drop anything already in state.completed_task_keys.
            D-16  drop anything that failed at this depth or deeper
                  (irrelevant on this first pass; matters on retries).

        Caution: `t['goal_id']` / `t['query']` in cap_and_filter index the
        model's JSON directly — a missing key is a KeyError that aborts the
        run, same risk as goal_manager_node above.
        """
        router.set_node("task_expander")
        result = router.complete_json(
            templates.expand_tasks(state.goals, settings.max_fanout))
        tasks = cap_and_filter(result.get("tasks", []), state, depth=0,
                                max_fanout=settings.max_fanout)
        log_event(logger, "node.expand", produced=len(tasks))
        # D-2: replace-on-write — this IS the whole backlog for next dispatch.
        return {"pending_tasks": tasks, "counters": {"llm_calls": 1}}

    return task_expander_node
