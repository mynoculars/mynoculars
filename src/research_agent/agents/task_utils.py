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
    """Apply production-side task hygiene; returns the ranked, capped backlog.

    Rules (design doc references):
        D-13 cap-at-production: rank by priority, keep top max_fanout —
             overflow is the PRODUCER's decision; dispatch is always total.
        D-2  dedup: drop keys already completed.
        D-16 failure retry: a previously FAILED key may return only at a
             strictly greater depth than where it failed.

    Parameters:
        raw_tasks: model output — dicts with query/goal_id/priority.
        state: current graph state (read-only here).
        depth: the iteration depth these tasks are produced at.
        max_fanout: the cap.
    """
    tasks: List[SearchTask] = []
    for t in raw_tasks:
        key = f"{t['goal_id']}::{t['query'].strip().lower()}"
        if key in state.completed_task_keys:
            continue
        failed_at = state.failed_task_keys.get(key)
        if failed_at is not None and depth <= failed_at:
            continue
        tasks.append(SearchTask(key=key, query=t["query"], goal_id=t["goal_id"],
                                priority=int(t.get("priority", 0)), depth=depth))
    tasks.sort(key=lambda t: t.priority, reverse=True)
    return tasks[:max_fanout]
