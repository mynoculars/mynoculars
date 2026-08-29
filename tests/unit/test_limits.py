"""
tests/unit/test_limits.py — limits.py's run-budget predicate (D-132, P6-4).

Covers the predicate ONLY: what counts as elapsed research time, what
counts as spent, and that both budgets are inert at their defaults.
Where the answer is ACTED on lives elsewhere -- see
test_orchestration_graph.py for the routing, test_agents_gathering.py and
test_agents_compilation.py for the two nodes that set the flag.
"""

from research_agent.config import Settings
from research_agent.limits import (elapsed_seconds, run_budget_exhausted,
                                   tokens_used)
from research_agent.state import ResearchState


def _state(**kw):
    return ResearchState(raw_query="q", **kw)


def _settings(**kw):
    return Settings(_env_file=None, **kw)


# ---------------------------------------------------------------------------
# elapsed time
# ---------------------------------------------------------------------------


def test_an_unstamped_run_has_no_elapsed_time():
    """Every offline test builds a bare ResearchState, and classify_node
    is what stamps the clock. An unstamped run must never be judged over
    budget -- 0.0, not "since the epoch"."""
    assert elapsed_seconds(_state()) == 0.0


def test_elapsed_is_measured_from_the_stamp():
    assert elapsed_seconds(_state(run_started_at=1000.0), now=1042.0) == 42.0


def test_time_spent_waiting_for_a_human_is_not_research_time():
    """p205.267-check paused 68 seconds at one E4 and 7 at another. A
    budget that charged a reviewer's reading time to the run would make
    HITL and deadlines mutually exclusive."""
    state = _state(run_started_at=1000.0, paused_seconds=75.0)
    assert elapsed_seconds(state, now=1100.0) == 25.0


def test_elapsed_never_goes_negative():
    """A clock adjustment mid-run is the cost of an epoch (limits.py says
    so plainly); it must not produce a negative age."""
    state = _state(run_started_at=1000.0, paused_seconds=500.0)
    assert elapsed_seconds(state, now=1100.0) == 0.0


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_tokens_used_reads_the_same_counters_telemetry_reports():
    """Not a second tally -- llm_total_tokens (D-86) is the sum of these
    exact two counters, and a budget enforced against a different number
    than the one in the run record would be indefensible."""
    state = _state(counters={"llm_prompt_tokens": 4023.0,
                             "llm_completion_tokens": 2068.0})
    assert tokens_used(state) == 6091


def test_tokens_used_is_zero_on_a_run_that_made_no_calls():
    assert tokens_used(_state()) == 0


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------


def test_both_budgets_are_inert_at_their_defaults():
    """THE property that makes this safe to ship: with both at 0 the
    graph is byte-identical to before D-132, however long or expensive
    the run gets."""
    state = _state(run_started_at=1.0, paused_seconds=0.0,
                   counters={"llm_prompt_tokens": 10_000_000.0})
    assert run_budget_exhausted(state, _settings(), now=10 ** 9) is None


def test_a_spent_deadline_reports_deadline():
    state = _state(run_started_at=1000.0)
    settings = _settings(run_deadline_seconds=600.0)

    assert run_budget_exhausted(state, settings, now=1599.0) is None
    assert run_budget_exhausted(state, settings, now=1601.0) == "deadline"


def test_a_spent_token_budget_reports_tokens():
    state = _state(counters={"llm_prompt_tokens": 30_000.0,
                             "llm_completion_tokens": 5_000.0})
    settings = _settings(run_token_budget=40_000)
    assert run_budget_exhausted(state, settings) is None

    settings = _settings(run_token_budget=30_000)
    assert run_budget_exhausted(state, settings) == "tokens"


def test_the_deadline_is_reported_first_when_both_are_spent():
    """Not arbitrary: a run past its wall clock is late whatever it
    spent, and "late" is the answer an operator waiting on it needs."""
    state = _state(run_started_at=1000.0,
                   counters={"llm_prompt_tokens": 99_000.0})
    settings = _settings(run_deadline_seconds=10.0, run_token_budget=10)

    assert run_budget_exhausted(state, settings, now=2000.0) == "deadline"


def test_a_pause_can_keep_a_run_inside_its_deadline():
    """The interaction that has to work, end to end: 100 seconds of wall
    clock, 80 of them waiting for a human, against a 60-second budget."""
    state = _state(run_started_at=1000.0, paused_seconds=80.0)
    settings = _settings(run_deadline_seconds=60.0)

    assert run_budget_exhausted(state, settings, now=1100.0) is None
