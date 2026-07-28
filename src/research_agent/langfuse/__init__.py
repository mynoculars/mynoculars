"""
research_agent/langfuse/__init__.py — the ONLY import surface business
modules are allowed to use for observability.

    from research_agent.langfuse import start_trace, span, generation, \\
        event, score, end_trace

Every business file (cli.py, orchestration/graph.py, llm/router.py,
retrieval/hybrid.py, memory/semantic_memory.py, agents/escalation.py,
storage/postgres.py, evaluation/quality.py, agents/gathering.py,
agents/compilation.py) imports ONLY these thin functions. None of them
ever see a Langfuse SDK object, a trace handle, or a span handle -- that
is the whole point of the module boundary Phase 3 asked for. All of the
actual SDK usage, trace-lifecycle bookkeeping, and fail-open behaviour
lives in observer.py and client.py.

LIFECYCLE: `init_from_settings(settings)` is called exactly once, from
`cli.py::build_app_and_settings` (the project's single existing wiring
point -- see that function's own docstring), before anything else in
this module is used. Every function below silently no-ops if
`init_from_settings` was never called (get_observer() lazily builds a
disabled Observer the first time it's needed) -- so an existing test or
script that never touches Langfuse at all keeps working exactly as
before, with zero behaviour change.
"""

from __future__ import annotations

from typing import Any, Optional

from research_agent.langfuse.client import build_client
from research_agent.langfuse.helpers import thread_id_from_config, traced_node
from research_agent.langfuse.observer import Observer

__all__ = [
    "init_from_settings", "get_observer", "is_enabled",
    "start_trace", "end_trace", "span", "generation", "event", "score",
    "flush", "shutdown",
    "thread_id_from_config", "traced_node",
]

_observer: Optional[Observer] = None
_init_settings: Optional[Any] = None


def init_from_settings(settings) -> Observer:
    """Build (or rebuild) the process-wide Observer from a Settings
    object. Idempotent-ish: calling it again with a different settings
    object replaces the observer (used by tests); calling it with the
    SAME settings instance repeatedly is harmless and cheap once past
    the first call, since build_client() itself is what's expensive
    (network) and only runs here."""
    global _observer, _init_settings
    client = build_client(settings)
    _observer = Observer(client, settings)
    _init_settings = settings
    return _observer


def get_observer() -> Observer:
    """Return the active Observer, lazily constructing a disabled one
    (client=None) if init_from_settings was never called. This is what
    guarantees every thin function below is always safe to call from
    anywhere, including tests and scripts that never wire up Langfuse at
    all -- there is never a "did you forget to init me" crash, only a
    silent no-op."""
    global _observer
    if _observer is None:
        _observer = Observer(client=None, settings=_init_settings)
    return _observer


def is_enabled() -> bool:
    """True only when a real, working Langfuse client is active. Useful
    for a call site that wants to skip building an expensive metadata
    payload for what would be a no-op anyway (see llm/router.py)."""
    return get_observer().enabled


def start_trace(thread_id: str, name: str, *, input: Any = None,
                metadata: Optional[dict] = None) -> None:
    get_observer().start_trace(thread_id, name, input=input, metadata=metadata)


def end_trace(thread_id: str, *, output: Any = None,
              metadata: Optional[dict] = None) -> None:
    get_observer().end_trace(thread_id, output=output, metadata=metadata)


def span(thread_id: str, name: str, *, input: Any = None, output: Any = None,
         metadata: Optional[dict] = None, start_time: Optional[float] = None,
         end_time: Optional[float] = None, level: str = "DEFAULT") -> None:
    get_observer().span(thread_id, name, input=input, output=output,
                        metadata=metadata, start_time=start_time,
                        end_time=end_time, level=level)


def generation(thread_id: str, name: str, *, provider: str, model: str,
               input: Any = None, output: Any = None,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               start_time: Optional[float] = None,
               end_time: Optional[float] = None,
               metadata: Optional[dict] = None) -> None:
    get_observer().generation(
        thread_id, name, provider=provider, model=model, input=input,
        output=output, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, start_time=start_time,
        end_time=end_time, metadata=metadata)


def event(thread_id: str, name: str, *, input: Any = None,
          metadata: Optional[dict] = None, level: str = "DEFAULT") -> None:
    get_observer().event(thread_id, name, input=input, metadata=metadata, level=level)


def score(thread_id: str, name: str, value: Any, *,
          comment: Optional[str] = None) -> None:
    get_observer().score(thread_id, name, value, comment=comment)


def flush() -> None:
    get_observer().flush()


def shutdown() -> None:
    """Flush and release the client. Called from cli.py's finally block
    alongside close_checkpointer()/router.close() -- the same pattern
    every other closeable resource in this codebase already follows."""
    get_observer().shutdown()
