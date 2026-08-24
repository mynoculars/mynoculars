"""
agents/task_utils.py — Shared task hygiene for the two task producers.

Contract:
    cap_and_filter() is the one seam task_expander_node (planning.py) and
    gap_generator_node (gathering.py) both call before turning raw LLM
    output into SearchTasks, so the two producers cannot drift apart on
    validation, dedup, retry, or capping (D-2, D-13, D-16, D-65).
"""

import logging
from typing import List, Tuple

from pydantic import BaseModel, Field, ValidationError

from research_agent.logging_setup import log_event
from research_agent.state import ResearchState, SearchTask

logger = logging.getLogger(__name__)


class RawTask(BaseModel):
    """The shape a task-producing LLM call must return per item, validated
    before any dict indexing happens (D-65).

    Contract: query/goal_id are required, non-empty strings; priority
    defaults to 0; tool_hint defaults to "" and is validated against what
    is actually wired into THIS run by cap_and_filter below, not here.
    """

    query: str = Field(min_length=1)
    goal_id: str = Field(min_length=1)
    priority: int = 0
    tool_hint: str = ""


def cap_and_filter(raw_tasks: list, state: ResearchState, depth: int,
                   max_fanout: int,
                   allowed_tool_hints: frozenset = frozenset()) -> Tuple[List[SearchTask], int]:
    """Validate, dedup, retry-gate, and cap one producer's raw task output.

    CALLED BY   task_expander_node (depth=0, first pass) and
                gap_generator_node (depth=state.iteration_depth, every
                later pass) — right after each gets raw
                {"query", "goal_id", "priority"} dicts back from an LLM
                call, before anything becomes a SearchTask.
    READS       state.goals, state.completed_task_keys,
                state.failed_task_keys — read-only; this function never
                writes state itself.
    RETURNS     (tasks, rejected_count): tasks is already validated,
                deduped, depth-filtered, ranked, and capped — the caller
                does nothing further before writing it to
                state.pending_tasks. rejected_count folds into
                state.counters["producer_rejects"].

    Applied in order, each rule doing one job:
        D-65 VALIDATE — each raw dict must satisfy RawTask (non-empty
              query/goal_id). A malformed item is dropped and counted,
              never raises.
        UNKNOWN-GOAL GUARD — drop a well-formed task whose goal_id is not
              one of state.goals' actual ids (D-50).
        D-65 BATCH DEDUP — drop a (goal_id, query) pair repeated within
              THIS producer response, before the cross-cycle check below.
        D-2  DEDUP — drop any key already in state.completed_task_keys.
        D-16 RETRY GATE — drop any key that failed at this depth or later
              (state.failed_task_keys[key] >= depth); a task that failed
              at depth d may be re-emitted only by a call at depth > d.
        D-13 CAP-AT-PRODUCTION — sort survivors by priority, keep the top
              max_fanout. Always done HERE, by the producer that can rank;
              graph.py's dispatch_tasks sends everything it is given and
              never ranks.

    Parameters:
        raw_tasks: model output, one dict per task candidate.
        state: current graph state — read-only in this function.
        depth: the iteration depth these tasks are produced at.
        max_fanout: settings.max_fanout, passed through by the caller.
        allowed_tool_hints: (D-25) the set of specialist hint names wired
            into THIS run's graph, e.g. frozenset({"mcp"}) when
            settings.mcp_enabled, else empty — the default for every
            caller that omits this, giving every SearchTask tool_hint="".
    """
    tasks: List[SearchTask] = []
    rejected = 0
    known_goal_ids = {g.goal_id for g in state.goals}
    batch_keys: set = set()
    for t in raw_tasks:
        try:
            raw = RawTask.model_validate(t)
        except ValidationError as exc:
            rejected += 1
            log_event(logger, "producer.reject", level=logging.WARNING,
                      depth=depth, errors=str(exc.errors()))
            continue
        if known_goal_ids and raw.goal_id not in known_goal_ids:
            rejected += 1
            log_event(logger, "producer.reject_unknown_goal_id",
                      level=logging.WARNING, depth=depth,
                      goal_id=raw.goal_id,
                      known_goal_ids=sorted(known_goal_ids))
            continue
        key = f"{raw.goal_id}::{raw.query.strip().lower()}"
        if key in batch_keys:
            continue  # D-65: same key twice in one producer response
        batch_keys.add(key)
        if key in state.completed_task_keys:
            continue  # D-2: already run in an earlier cycle
        failed_at = state.failed_task_keys.get(key)
        if failed_at is not None and depth <= failed_at:
            continue  # D-16: retry gate
        hint = raw.tool_hint if raw.tool_hint in allowed_tool_hints else ""
        tasks.append(SearchTask(key=key, query=raw.query, goal_id=raw.goal_id,
                                priority=raw.priority, depth=depth, tool_hint=hint))
    tasks.sort(key=lambda t: t.priority, reverse=True)  # D-13: highest priority first
    return tasks[:max_fanout], rejected  # D-13: cap at max_fanout
