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

from typing import List

from research_agent.state import ResearchState, SearchTask


def cap_and_filter(raw_tasks: list, state: ResearchState, depth: int,
                   max_fanout: int) -> List[SearchTask]:
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
        raw_tasks: model output — dicts with query/goal_id/priority. No
            validation happens here: `t['goal_id']` / `t['query']` index
            directly, so a dict missing either key raises KeyError and
            aborts the run. This function trusts its caller's LLM output.
        state: current graph state — read-only in this function.
        depth: the iteration depth these tasks are being produced AT (0
            for task_expander_node's first pass; state.iteration_depth for
            every gap_generator_node pass after that).
        max_fanout: the cap — settings.max_fanout, passed through by the
            caller.
    """
    tasks: List[SearchTask] = []
    for t in raw_tasks:
        # f"{...}::{...}" builds a plain string by substituting variables
        # into the {} placeholders — this is an f-string, Python's way of
        # formatting text. Here it builds one unique identifier per
        # (goal, query) pair, e.g. "g1::key differences".
        key = f"{t['goal_id']}::{t['query'].strip().lower()}"
        if key in state.completed_task_keys:
            # "continue" skips the rest of THIS loop iteration and moves on
            # to the next raw task — it does not exit the whole loop, just
            # this one pass through it. Effect: this task is silently
            # dropped from the output list (D-2 dedup).
            continue
        failed_at = state.failed_task_keys.get(key)
        if failed_at is not None and depth <= failed_at:
            continue  # D-16 retry gate — see the docstring above
        tasks.append(SearchTask(key=key, query=t["query"], goal_id=t["goal_id"],
                                priority=int(t.get("priority", 0)), depth=depth))
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
    return tasks[:max_fanout]
