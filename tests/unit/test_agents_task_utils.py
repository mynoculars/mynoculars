"""
tests/unit/test_agents_task_utils.py — agents/task_utils.py's
cap_and_filter.

Covers: malformed-task rejection with a counted reject rather than a
KeyError (P2-06), and the tool_hint allowlist/reset behavior (P2-14,
D-25) — a hint the LLM emitted but that isn't wired into THIS run is
silently reset to the default, not treated as a validation failure.
"""

from research_agent.agents.task_utils import cap_and_filter
from research_agent.state import ResearchState


def test_cap_and_filter_drops_malformed_tasks_and_counts_them(settings):
    raw = [
        {"query": "good query", "goal_id": "g1", "priority": 1},
        {"query": "", "goal_id": "g1"},           # empty query -> rejected
        {"goal_id": "g1", "priority": 1},          # missing query -> rejected
        {"query": "another", "goal_id": ""},       # empty goal_id -> rejected
    ]
    state = ResearchState(raw_query="q")
    tasks, rejected = cap_and_filter(raw, state, depth=0, max_fanout=6)
    assert len(tasks) == 1
    assert tasks[0].query == "good query"
    assert rejected == 3


def test_cap_and_filter_never_raises_on_missing_keys():
    # Before P2-06, a dict missing "goal_id"/"query" raised KeyError here
    # and took the whole run down. Now it's just a counted rejection.
    state = ResearchState(raw_query="q")
    tasks, rejected = cap_and_filter([{}], state, depth=0, max_fanout=6)
    assert tasks == []
    assert rejected == 1


def test_cap_and_filter_keeps_a_hint_that_is_in_allowed_tool_hints():
    raw = [{"query": "q1", "goal_id": "g1", "priority": 1, "tool_hint": "mcp"}]
    tasks, rejected = cap_and_filter(raw, ResearchState(raw_query="q"), depth=0,
                                     max_fanout=5, allowed_tool_hints=frozenset({"mcp"}))
    assert rejected == 0
    assert tasks[0].tool_hint == "mcp"


def test_cap_and_filter_resets_a_hint_not_in_allowed_tool_hints():
    """The core validation this item exists for: a hint the LLM emitted
    but that isn't actually wired into THIS run must never survive into
    a SearchTask -- it's silently reset to the default, not rejected as
    malformed (the request itself was well-formed; the hint just isn't
    available right now)."""
    raw = [{"query": "q1", "goal_id": "g1", "priority": 1, "tool_hint": "mcp"}]
    tasks, rejected = cap_and_filter(raw, ResearchState(raw_query="q"), depth=0,
                                     max_fanout=5, allowed_tool_hints=frozenset())
    assert rejected == 0
    assert tasks[0].tool_hint == ""


def test_cap_and_filter_default_call_with_no_allowed_tool_hints_arg_is_unchanged():
    """Backward compatibility: a caller that doesn't even know about
    allowed_tool_hints yet (the exact old call signature) still gets
    tool_hint="" on everything -- byte-identical pre-P2-14 behavior."""
    raw = [{"query": "q1", "goal_id": "g1", "priority": 1}]
    tasks, rejected = cap_and_filter(raw, ResearchState(raw_query="q"), depth=0, max_fanout=5)
    assert tasks[0].tool_hint == ""


def test_cap_and_filter_dedups_identical_tasks_within_one_batch():
    """P205 regression (run p205.70-check). The gap generator returned six
    tasks that were two distinct queries repeated three times each, all for
    g2. completed_task_keys could not catch it -- none of them had run yet
    -- so all six dispatched, burning four of six MAX_FANOUT slots and
    multiplying that goal's evidence 3x with identical items, which is half
    of what pushed an uncovered goal to recall 1.0 on irrelevant documents.
    """
    state = ResearchState(raw_query="Compare India and US")
    raw = [
        {"query": "Analyze US and India economic systems", "goal_id": "g2", "priority": 1},
        {"query": "Compare US and India economic systems", "goal_id": "g2", "priority": 1},
        {"query": "Analyze US and India economic systems", "goal_id": "g2", "priority": 1},
        {"query": "Compare US and India economic systems", "goal_id": "g2", "priority": 1},
        {"query": "Analyze US and India economic systems", "goal_id": "g2", "priority": 1},
        {"query": "Compare US and India economic systems", "goal_id": "g2", "priority": 1},
    ]
    tasks, rejected = cap_and_filter(raw, state, depth=1, max_fanout=6)
    assert len(tasks) == 2, "six raw tasks were only two distinct queries"
    assert len({t.key for t in tasks}) == 2
    # Duplicates are not malformed input -- they must not inflate the
    # producer_rejects counter, which means something different (D-13).
    assert rejected == 0


def test_cap_and_filter_keeps_same_query_under_different_goals():
    """Dedup is on (goal_id, query), not query alone -- the same phrasing
    genuinely serving two goals is not a duplicate."""
    state = ResearchState(raw_query="q")
    raw = [{"query": "same text", "goal_id": "g1", "priority": 1},
           {"query": "same text", "goal_id": "g2", "priority": 1}]
    tasks, _ = cap_and_filter(raw, state, depth=0, max_fanout=6)
    assert len(tasks) == 2
