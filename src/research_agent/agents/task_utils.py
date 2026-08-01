"""
agents/task_utils.py — Shared task hygiene for the two task producers.

Purpose:
    One visible home for the three production rules both the task expander
    and the gap generator must apply. Previously a private helper inside
    planning.py imported across modules — promoted here per review.

Responsibilities:
    - cap_and_filter(): D-13 cap-at-production (rank, keep top max_fanout),
      D-2 dedup against completed keys, D-16 depth-gated retry of failed keys.
"""

import logging
from typing import List, Tuple

from pydantic import BaseModel, Field, ValidationError

from research_agent.logging_setup import log_event
from research_agent.state import ResearchState, SearchTask

logger = logging.getLogger(__name__)


class RawTask(BaseModel):
    """P2-06: the shape a task-producing LLM call is REQUIRED to return per
    item, validated BEFORE any dict indexing happens.

    CALLED BY   cap_and_filter, below — the seam BOTH task_expander_node
                and gap_generator_node already share (see this module's
                original docstring). Before this fix, `t['goal_id']` /
                `t['query']` indexed the model's raw JSON directly; a live
                model omitting either key raised KeyError and aborted the
                whole run — the one place in this codebase where a
                producer's malformed output had no D-16-style "failure is
                data" handling. This model makes that failure mode data
                too: a bad item is dropped and counted, not fatal.

    query/goal_id are required, non-empty strings (Pydantic rejects a
    missing key OR an empty string via min_length=1). priority is optional
    with the same default (0) cap_and_filter always assumed via
    t.get("priority", 0).
    """

    query: str = Field(min_length=1)
    goal_id: str = Field(min_length=1)
    priority: int = 0
    # P2-14 (D-25): optional -- a producer MAY ask for a specific
    # specialist worker (today, only ever "mcp" -- see
    # tools/mcp_client.py). Validated against what's ACTUALLY wired into
    # THIS run inside cap_and_filter below, not here: RawTask's job is
    # "is this a well-formed string", not "is this hint currently
    # available" -- those are different questions, checked in different
    # places, matching this file's existing division of labor (RawTask =
    # shape, cap_and_filter's own logic = the D-2/D-13/D-16 business
    # rules). Absent/empty (the default -- every task before P2-14, and
    # every task a producer emits without a reason to ask for a
    # specialist) always means "the default corpus worker", unchanged.
    tool_hint: str = ""


def cap_and_filter(raw_tasks: list, state: ResearchState, depth: int,
                   max_fanout: int,
                   allowed_tool_hints: frozenset = frozenset()) -> Tuple[List[SearchTask], int]:
    """Shared hygiene applied by BOTH task-producing nodes before their
    output ever becomes state.pending_tasks — task_expander_node calls this
    with depth=0 (planning.py, the first pass); gap_generator_node
    (gathering.py) calls it with depth=state.iteration_depth on every
    later pass. Same function, same three rules, both call sites — so the
    two producers can never quietly drift apart on how they filter or cap.

    CALLED BY   task_expander_node, gap_generator_node — right after each
                gets raw {"query", "goal_id", "priority"} dicts back from
                an LLM call, before anything is turned into a SearchTask.
    READS       state.completed_task_keys, state.failed_task_keys — both
                read-only here; this function never writes state itself,
                it only returns a list for the CALLER to put into
                pending_tasks.
    RETURNS     a List[SearchTask] — already deduped, depth-filtered,
                ranked, and capped. The caller does nothing further to it
                before writing it to state.pending_tasks.

    The three rules, applied in this order, each doing one job:

        D-2  DEDUP  — drop any (goal_id, query) pair whose key is already
                      in state.completed_task_keys. Once a task has
                      actually produced evidence, it is never re-asked.

        D-16 RETRY GATE — drop any key that FAILED at this depth or a
                      later one (state.failed_task_keys[key] >= depth).
                      A task that failed at depth 1 CAN be re-emitted by a
                      call with depth=2, but not by another call still at
                      depth=1 — this is what turns a transient backend
                      error into "wait one cycle," not "never retry" or
                      "retry immediately in a tight loop."

        D-13 CAP-AT-PRODUCTION — sort what survives by priority, keep only
                      the top `max_fanout`. This is deliberately done HERE,
                      by the producer that knows the ranking, and NEVER by
                      graph.py's dispatch_tasks — the dispatcher always
                      sends everything it is given; it is not allowed to
                      make a ranking judgement it has no way to make well.

    Parameters:
        raw_tasks: model output — dicts that SHOULD look like
            {"query": ..., "goal_id": ..., "priority": ...}. P2-06: each
            dict is now validated against RawTask (this module) BEFORE any
            key is ever indexed. A dict missing "query"/"goal_id", or
            carrying an empty string for either, is dropped and counted —
            it no longer raises KeyError and takes the whole run down, the
            same "failure is data" philosophy D-16 already applies on the
            worker side of this same loop.
        state: current graph state — read-only in this function.
        depth: the iteration depth these tasks are being produced AT (0
            for task_expander_node's first pass; state.iteration_depth for
            every gap_generator_node pass after that).
        max_fanout: the cap — settings.max_fanout, passed through by the
            caller.
        allowed_tool_hints: (P2-14, D-25) the set of specialist hint names
            actually wired into THIS run's graph -- e.g. frozenset({"mcp"})
            when settings.mcp_enabled, else the empty frozenset (the
            default -- every existing caller that doesn't pass this at
            all gets byte-identical pre-P2-14 behavior: every SearchTask's
            tool_hint ends up "").

    Returns:
        (tasks, rejected_count) — rejected_count is how many raw_tasks
        entries failed RawTask validation, so callers can fold it into
        state.counters["producer_rejects"] (D-13 hygiene extended to cover
        malformed input, not just dedup/depth/cap).
    """
    tasks: List[SearchTask] = []
    rejected = 0
    # D-2, second half: dedup WITHIN this batch, not only against history.
    # completed_task_keys answers "have we already run this query in an
    # EARLIER cycle"; it says nothing about the model emitting the same
    # query twice in the SAME response. Found live (run p205.70-check):
    # gap_generator returned six tasks that were two distinct queries
    # repeated three times each, all for g2 -- "produced": 6,
    # "rejected": 0, then six search_worker invocations doing two distinct
    # searches. Four of six MAX_FANOUT slots were wasted, and the goal's
    # evidence list was multiplied 3x by identical items, which is half of
    # what drove an uncovered goal to recall 1.0 on irrelevant documents
    # (the other half was fusion double-credit -- see
    # retrieval/hybrid.py::rrf_fuse).
    batch_keys: set = set()
    for t in raw_tasks:
        try:
            raw = RawTask.model_validate(t)
        except ValidationError as exc:
            # P2-06: a malformed item is DATA, not a crash — log which
            # fields were the problem (without ever raising KeyError) and
            # move on to the next raw task, exactly like D-16's failed-task
            # handling on the worker side of this same seam.
            rejected += 1
            log_event(logger, "producer.reject", level=logging.WARNING,
                      depth=depth, errors=str(exc.errors()))
            continue
        # f"{...}::{...}" builds a plain string by substituting variables
        # into the {} placeholders — this is an f-string, Python's way of
        # formatting text. Here it builds one unique identifier per
        # (goal, query) pair, e.g. "g1::key differences".
        key = f"{raw.goal_id}::{raw.query.strip().lower()}"
        if key in batch_keys:
            continue  # D-2: same key twice in one producer response
        batch_keys.add(key)
        if key in state.completed_task_keys:
            # "continue" skips the rest of THIS loop iteration and moves on
            # to the next raw task — it does not exit the whole loop, just
            # this one pass through it. Effect: this task is silently
            # dropped from the output list (D-2 dedup).
            continue
        failed_at = state.failed_task_keys.get(key)
        if failed_at is not None and depth <= failed_at:
            continue  # D-16 retry gate — see the docstring above
        # P2-14 (D-25): a raw tool_hint is only ever HONORED if it names a
        # specialist actually wired into THIS run (allowed_tool_hints,
        # passed down from settings.mcp_enabled at the two call sites --
        # see agents/planning.py::task_expander_node and
        # agents/gathering.py::gap_generator_node). Anything else --
        # empty (the overwhelming common case), unrecognized, or a real
        # hint name that just isn't active this run -- silently resets
        # to "" (the default route) rather than ever reaching
        # dispatch_tasks as something it would have to guess about. This
        # is the ONLY place in the whole codebase that ever sets
        # SearchTask.tool_hint to anything other than "" -- see that
        # field's own docstring in state.py.
        hint = raw.tool_hint if raw.tool_hint in allowed_tool_hints else ""
        tasks.append(SearchTask(key=key, query=raw.query, goal_id=raw.goal_id,
                                priority=raw.priority, depth=depth, tool_hint=hint))
    # .sort(key=..., reverse=True) sorts the list IN PLACE (it modifies
    # `tasks` directly and returns nothing). `key=lambda t: t.priority` tells
    # Python "for sorting purposes, look at each task's .priority field" — a
    # lambda is just a tiny, unnamed function written inline: `lambda t: t.priority`
    # means "given a task t, return t.priority". reverse=True means highest
    # priority first.
    tasks.sort(key=lambda t: t.priority, reverse=True)
    # tasks[:max_fanout] is a SLICE: "give me the first max_fanout items of
    # this list". Combined with the sort above, this keeps only the
    # highest-priority tasks — the D-13 cap.
    return tasks[:max_fanout], rejected
