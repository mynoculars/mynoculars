"""
research_agent/langfuse/helpers.py — small, reusable utilities that don't
belong on Observer itself.

Two things live here:
  1. `thread_id_from_config` -- pulls the run identity out of a LangGraph
     RunnableConfig the same way it's already threaded everywhere else in
     this codebase (Postgres checkpointing, the run_id ContextVar in
     logging_setup.py).
  2. `traced_node` -- wraps ANY existing node callable with span
     instrumentation, with NO change to the node's own code. This is what
     makes "instrument every graph node" a one-line change per node at
     `build_graph` call time, rather than touching all thirteen
     `agents/*.py` node functions individually -- see graph.py for the
     one place this is actually used.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Dict, Optional


def thread_id_from_config(config: Optional[dict]) -> str:
    """Extract thread_id from a LangGraph RunnableConfig, e.g.
    {"configurable": {"thread_id": "run-abc123"}, "recursion_limit": 60}
    (see cli.py::_run, which builds exactly this shape). Falls back to
    "unknown" rather than raising -- a missing thread_id should degrade
    observability, never break the node it's wrapping."""
    if not config:
        return "unknown"
    return str(config.get("configurable", {}).get("thread_id", "unknown"))


def _fn_accepts_config(fn: Callable) -> bool:
    """True if `fn` itself declares a `config` parameter (positional-or-
    keyword or keyword-only), OR accepts **kwargs that would swallow one.

    WHY THIS EXISTS -- fixes a real bug from the first cut of this file:
    the wrapper below used to unconditionally call `fn(state)`, silently
    dropping any `config` argument on the floor. No node in this
    codebase currently declares `config` (every `build_*_node` factory
    returns a plain `def X_node(state) -> Dict`), so nothing broke in
    practice -- but the wrapper's own docstring claimed a wrapped node
    COULD receive its thread_id via `config`, which was simply false: it
    never reached the node at all. The first node ever written to accept
    `config` would have hit `fn(state)` with a missing required argument
    the moment it was wrapped. Inspecting the ACTUAL signature, once,
    here, and forwarding conditionally, is what makes the docstring's
    claim true instead of aspirational.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # A small number of callables (some C-implemented ones) refuse
        # signature introspection entirely -- degrade to "config not
        # accepted" rather than raising out of the wrapper itself.
        return False
    for p in sig.parameters.values():
        if p.name == "config" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def traced_node(observer_getter: Callable[[], Any], node_name: str,
                 fn: Callable[[Any], Dict[str, Any]]) -> Callable:
    """Wrap a LangGraph node function with a Langfuse span, changing
    nothing about what the node itself does or returns.

    `observer_getter` is a zero-arg callable returning the current
    Observer (see langfuse/__init__.py::get_observer) rather than the
    Observer itself, so this wrapper always uses whichever Observer is
    active at CALL time -- important for tests, which construct a fresh
    Observer per test rather than relying on the module-level singleton.

    The OUTER wrapper always accepts an optional second `config`
    parameter, so LangGraph (which inspects the wrapper's own signature,
    not the wrapped function's) always hands it one. Whether that
    `config` is then forwarded to the INNER `fn` depends on whether `fn`
    itself declared one -- checked once, at wrap time, via
    `_fn_accepts_config` above, not on every call.
    """
    forward_config = _fn_accepts_config(fn)

    @functools.wraps(fn)
    def wrapper(state, config=None):
        thread_id = thread_id_from_config(config)
        observer = observer_getter()
        start = time.time()
        error: Optional[str] = None
        result = None
        try:
            result = fn(state, config=config) if forward_config else fn(state)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            raise
        finally:
            # Phase 3 review (#7/#9): this used to carry NO content at
            # all -- just start/end/error. Passing input/output here,
            # generically, for every node in one place, is consistent
            # with how every OTHER span/generation call in this codebase
            # already works (e.g. retrieval/hybrid.py, llm/client.py) --
            # this is the one node-agnostic fix that makes a search_
            # worker's actual query and a memory node's actual counts
            # visible without hand-writing per-node instrumentation,
            # which would have meant re-introducing the thirteen-files
            # problem this wrapper exists specifically to avoid. The
            # tradeoff, stated plainly: `state`/`result` grow larger as
            # a run progresses (evidence accumulates), so later nodes'
            # spans carry more payload than earlier ones -- accepted
            # here as consistent with the rest of the module rather than
            # solved with per-node truncation logic.
            observer.span(
                thread_id, f"node:{node_name}",
                input=state, output=result,
                metadata={"error": error} if error else None,
                start_time=start, end_time=time.time(),
                level="ERROR" if error else "DEFAULT",
            )
    return wrapper
