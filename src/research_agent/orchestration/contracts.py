"""
orchestration/contracts.py — Runtime enforcement of the worker return contract.

Purpose:
    Make design decision D-15 executable: a fanned-out worker may return
    ONLY reducer-backed state keys. This is enforced at runtime, not by
    code review, because the failure it prevents is invisible to review:
    a non-reducer key in a worker return passes every single-task test and
    raises InvalidUpdateError only when two workers land in the same
    superstep — i.e. non-deterministically, under parallel load.

Responsibilities:
    - WORKER_WRITABLE_KEYS: the whitelist, defined HERE, adjacent to its
      enforcement, so admitting a new reducer-backed field is a one-file,
      reviewable change.
    - validated_worker: decorator that checks each worker return against
      the whitelist. Raises WorkerContractViolation so violations surface
      the FIRST time a worker runs, deterministically — in any environment.

Design decision (always raise, no log-and-filter mode):
    The full design (v3.1 D-15) specifies raise-in-dev / filter-in-prod.
    This reference build is explicitly not for production, so the softer
    prod mode would only add a config axis nobody exercises. Deferred:
    an ENV switch selecting filter behavior — a 5-line change here.
"""

import functools
import logging
from typing import Any, Callable, Dict

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

# The ONLY ResearchState fields a parallel worker may write (all reducer-backed;
# see state.py). Keep this list and state.py's Annotated fields in lockstep.
WORKER_WRITABLE_KEYS = frozenset(
    {"evidence", "completed_task_keys", "failed_task_keys", "counters"}
)


class WorkerContractViolation(RuntimeError):
    """A worker returned a state key outside the reducer-backed whitelist."""


def validated_worker(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap a worker node function with return-key validation.

    Parameters:
        fn: a worker node taking a WorkerPayload and returning a state-update
            dict (LangGraph convention).

    Returns:
        The wrapped function. Off-whitelist keys raise WorkerContractViolation
        immediately, naming the offending keys — turning a rare concurrent
        failure into a deterministic unit-test failure.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        update = fn(*args, **kwargs)
        illegal = set(update) - WORKER_WRITABLE_KEYS
        if illegal:
            log_event(logger, "worker.contract_violation",
                      level=logging.ERROR, worker=fn.__name__, keys=sorted(illegal))
            raise WorkerContractViolation(
                f"{fn.__name__} wrote non-reducer keys {sorted(illegal)}; "
                f"allowed: {sorted(WORKER_WRITABLE_KEYS)}")
        return update

    return wrapper
