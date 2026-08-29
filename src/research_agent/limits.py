"""
limits.py — the run-level budget predicate (D-132).

Purpose:
    Answer one question, in one place: has THIS run spent its wall-clock
    deadline or its token budget? Read by the two nodes that can act on
    the answer (progress_checker_node, compiler_node) and by nothing else.

WHY A RUN NEEDS A BUDGET AT ALL, given four termination bounds already
exist. `max_depth`, `max_revisions`, `max_escalations` and LangGraph's
`recursion_limit` bound the run in STEPS. None of them bounds it in TIME
or in SPEND, and those are the two units an operator is actually held to.
Live (p205.267-check): a run inside every one of those four bounds --
depth 3 of 3, three critique passes, two escalations -- took 237 seconds
and made 9 provider calls. Nothing was misbehaving. There was simply no
number anywhere that said "this run has taken long enough".

WHY A SOFT STOP AND NOT A CANCELLATION. Nothing here raises, cancels a
node mid-flight, or ends the graph. When a budget is spent this sets a
FLAG; the routing functions in orchestration/graph.py read it and send
the run to the compiler instead of round another gather lap or another
revision. Every path still reaches telemetry and still produces a
report, which is this codebase's standing rule for every other stop
condition it has (D-1's empty backlog, D-21's zero goals, D-22's
exhausted revisions) -- a run cut short still owes the caller an answer.

WHY WALL CLOCK IS STORED AS AN EPOCH, not a monotonic reading. A run can
PAUSE at an interrupt() and resume in a different process minutes later
(D-8/D-20/D-23) -- time.monotonic() is meaningless across that boundary,
and ResearchState is serialised to Postgres between the two halves. The
cost is that a system clock adjustment mid-run perturbs the measurement;
the alternative is a measurement that does not survive the resume this
project's whole HITL story depends on. cli.py keeps using monotonic for
the wall-time line it prints, where nothing is checkpointed.

That epoch's RESOLUTION is platform-dependent, which is worth knowing
before reading a small number here: time.time() is
GetSystemTimeAsFileTime on Windows, resolution 15.625 ms, against 1 ns
on Linux (`time.get_clock_info("time").resolution` reports both). A
pause or an elapsed span shorter than one tick therefore reads as
exactly 0.0 on Windows. Immaterial to a budget denominated in seconds --
and the reason a `paused_seconds > 0` assertion that passed on Linux
failed on Windows against an instantaneous test pause. That was a
property of the clock, not of the credit; the test now pauses long
enough to clear a tick (D-135).

HUMAN REVIEW TIME IS NOT RESEARCH TIME. `paused_seconds` accumulates
every interrupt() pause (agents/escalation.py) and is subtracted here.
Without it, a reviewer who takes four minutes to type "approve" spends
the run's research budget on their own reading -- live, p205.267-check
paused 68 seconds at one E4 and 7 at another. A budget that punishes a
run for being reviewed would make HITL and deadlines mutually exclusive,
which is not a trade anyone should have to make.
"""

import time
from typing import Optional

from research_agent.config import Settings
from research_agent.state import ResearchState


def tokens_used(state: ResearchState) -> int:
    """Total provider tokens this run has been billed for, so far.

    The SAME two counters telemetry_node reports as `llm_total_tokens`
    (D-86), read from the same place -- not a second tally that could
    disagree with the one in the run record.
    """
    counters = state.counters or {}
    return int(counters.get("llm_prompt_tokens", 0)
               + counters.get("llm_completion_tokens", 0))


def elapsed_seconds(state: ResearchState, now: Optional[float] = None) -> float:
    """Seconds of RESEARCH time so far -- pauses for human review excluded.

    0.0 before classify_node has stamped `run_started_at`, which is also
    what every offline test that constructs a bare ResearchState sees, so
    an unstamped run can never be judged over budget.

    `now` is injectable for tests; production callers pass nothing.
    """
    if not state.run_started_at:
        return 0.0
    return max(0.0, (now if now is not None else time.time())
               - state.run_started_at - state.paused_seconds)


def run_budget_exhausted(state: ResearchState, settings: Settings,
                         now: Optional[float] = None) -> Optional[str]:
    """Return "deadline", "tokens", or None.

    CALLED BY   agents/gathering.py::progress_checker_node (once per
                gather lap) and agents/compilation.py::compiler_node
                (once per compile). Both are natural boundaries: work has
                just finished and a routing decision is about to be made.
                Deliberately NOT called inside search_worker -- a check
                that fires mid-fan-out would abandon retrieval already
                paid for, and the lap ends microseconds later anyway.

    Both budgets are OFF by default (0 = disabled), which makes this
    function return None for every run that has not opted in, and leaves
    the graph byte-identical to before D-132 existed. That is the same
    posture HITL_ENABLED, MCP_ENABLED, CONTRADICTION_DETECTION_ENABLED
    and CLAIM_VERIFICATION_ENABLED all ship with, and it matters more
    here than for any of them: this is the first setting in this codebase
    that can END a run early.

    The deadline is tested FIRST and the order is not arbitrary: a run
    past its wall clock is late whatever it spent, and "late" is the
    answer an operator waiting on it needs.
    """
    if settings.run_deadline_seconds > 0:
        if elapsed_seconds(state, now) > settings.run_deadline_seconds:
            return "deadline"
    if settings.run_token_budget > 0:
        if tokens_used(state) > settings.run_token_budget:
            return "tokens"
    return None
