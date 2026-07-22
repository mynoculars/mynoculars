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

Python mechanics used in this file, if any of this is new to you:
    functools.wraps(fn)
        Used inside a decorator (see below) to copy fn's metadata — its
        __name__, its docstring, etc. — onto the wrapper function that
        replaces it. Without this, every function wrapped by
        @validated_worker would report its name as "wrapper" in error
        messages, stack traces, and tools like help(), instead of its real
        name (e.g. "search_worker") — @functools.wraps(fn) fixes that.
    def validated_worker(fn): ... return wrapper
        This whole function IS the decorator that agents/gathering.py
        applies with "@validated_worker" above search_worker's definition.
        Concretely, writing:
            @validated_worker
            def search_worker(payload): ...
        is EXACTLY equivalent to writing:
            def search_worker(payload): ...
            search_worker = validated_worker(search_worker)
        i.e. "take the function I just wrote, pass it into
        validated_worker, and use whatever comes back instead." What comes
        back is `wrapper` — a brand new function that calls the ORIGINAL
        search_worker internally, checks its return value, and only then
        hands that return value back to whoever called wrapper(). The
        original search_worker code is completely unaware any of this is
        happening around it.
    *args: Any, **kwargs: Any   (in wrapper's signature)
        *args collects any number of positional arguments into a tuple;
        **kwargs collects any number of keyword arguments into a dict (see
        logging_setup.py's docstring for the **kwargs half of this). Using
        both together, as here, means "accept whatever arguments the
        original function accepted, no matter what they are, and pass them
        straight through" — necessary because this decorator has no way of
        knowing in advance exactly what arguments search_worker (or any
        other function it might someday wrap) will be called with.
    frozenset({...})
        Like a regular Python set, but IMMUTABLE — once created, it cannot
        be modified (no .add(), no .remove()). Used here because
        WORKER_WRITABLE_KEYS is meant to be a fixed, read-only constant;
        frozenset communicates and enforces that intent, whereas a plain
        set would allow (accidental) mutation elsewhere in the codebase.
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
    """A worker returned a state key outside the reducer-backed whitelist.

    Inheriting from RuntimeError (a built-in Python exception type) means
    this behaves like any other exception: it can be raised, caught with
    `except WorkerContractViolation:` or the more general `except Exception:`,
    and carries whatever message string is passed to it when raised — see
    the `raise WorkerContractViolation(...)` call below for the exact
    message this one carries.
    """


def validated_worker(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap a worker node function with return-key validation.

    See the module docstring above for exactly how "@validated_worker"
    above a function definition connects to this function.

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
        # Run the REAL worker function (e.g. search_worker) exactly as it
        # would have run without this decorator, and capture whatever dict
        # it returns.
        update = fn(*args, **kwargs)
        # set(update) turns the dict's KEYS into a set (values are ignored).
        # "-" between two sets is SET DIFFERENCE: everything in the left set
        # that is NOT also in the right set. So `illegal` ends up containing
        # any key the worker returned that is NOT one of the four allowed
        # names in WORKER_WRITABLE_KEYS.
        illegal = set(update) - WORKER_WRITABLE_KEYS
        if illegal:
            log_event(logger, "worker.contract_violation",
                      level=logging.ERROR, worker=fn.__name__, keys=sorted(illegal))
            raise WorkerContractViolation(
                f"{fn.__name__} wrote non-reducer keys {sorted(illegal)}; "
                f"allowed: {sorted(WORKER_WRITABLE_KEYS)}")
        return update

    return wrapper
