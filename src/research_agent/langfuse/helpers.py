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


def traced_node(observer_getter: Callable[[], Any], node_name: str,
                 fn: Callable[[Any], Dict[str, Any]]) -> Callable:
    """Wrap a LangGraph node function with a Langfuse span, changing
    nothing about what the node itself does or returns.

    `observer_getter` is a zero-arg callable returning the current
    Observer (see langfuse/__init__.py::get_observer) rather than the
    Observer itself, so this wrapper always uses whichever Observer is
    active at CALL time -- important for tests, which construct a fresh
    Observer per test rather than relying on the module-level singleton.

    The wrapped function accepts an optional second `config` parameter.
    LangGraph inspects a node's signature and passes `config` only to
    functions that declare it (see LangGraph's own node-calling
    convention) -- accepting it here, once, is what lets every wrapped
    node receive its thread_id without any of the underlying
    `agents/*.py` functions changing their own signature at all.
    """
    @functools.wraps(fn)
    def wrapper(state, config=None):
        thread_id = thread_id_from_config(config)
        observer = observer_getter()
        start = time.time()
        error: Optional[str] = None
        try:
            result = fn(state)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            raise
        finally:
            observer.span(
                thread_id, f"node:{node_name}",
                metadata={"error": error} if error else None,
                start_time=start, end_time=time.time(),
                level="ERROR" if error else "DEFAULT",
            )
    return wrapper
