"""
research_agent/langfuse/observer.py — trace/span/generation/event/score
lifecycle, cost calculation, graceful degradation.

WHY THIS FILE EXISTS: this is the ONE class every other file in the
codebase actually talks to (indirectly, through the thin functions in
__init__.py). It owns everything Phase 3 asked to be encapsulated:
"SDK initialization, trace lifecycle, spans, generations, events, scores,
cost calculation, metadata, graceful shutdown, flush." Nothing here ever
raises out to a caller -- every public method is wrapped so a Langfuse
outage, a bad host, a version mismatch, or the SDK simply not being
installed degrades to "this call did nothing" rather than "this call
took down the research run."

SDK VERSION NOTE, worth knowing before touching this file: the installed
`langfuse` package is v4.x, which is entirely OpenTelemetry-based -- it
has NO `.trace()`/stateful-trace-object API (the shape most Langfuse
tutorials and the v2/v3 docs still show). Every observation is created
against a `TraceContext(trace_id=...)` instead of a live trace handle.
This module derives that trace_id DETERMINISTICALLY from the thread_id
via `client.create_trace_id(seed=thread_id)` -- the same thread_id
always maps to the same trace_id, so every span/generation/event/score
for one run lands on the same trace with NO in-memory registry required
(unlike an earlier draft of this file, which tried to cache live trace
objects per thread_id -- unnecessary complexity once the seed-based id
is available, and it would have leaked memory across long-lived
processes like the API server).

ONE HONEST LIMITATION: v4's Python client exposes no dedicated
"set trace session_id" call in its public API (confirmed by inspecting
`dir(Langfuse)` on the installed version -- no `update_current_trace`).
Session/thread_id is therefore attached as `metadata={"session_id":
thread_id, ...}` on the root span only, which is the documented
convention for grouping in the Langfuse UI but is NOT independently
verified here against a live Langfuse project (no live credentials in
this environment). If your Langfuse project doesn't group runs by
thread_id the way you expect, check this convention against your
Langfuse version's current docs before assuming this module is wrong.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from research_agent.langfuse.pricing import TokenUsage, calculate_cost

logger = logging.getLogger("research_agent.langfuse")


class Observer:
    """Wraps a (possibly None) Langfuse v4 client. Every method is safe
    to call regardless of whether observability is enabled, installed,
    reachable, or currently degraded -- see the module docstring."""

    def __init__(self, client: Optional[Any], settings: Any):
        self._client = client
        self._settings = settings
        # Only tracks OPEN root spans (start_trace -> end_trace pairs),
        # so end_trace can attach final output/duration. Nothing else
        # needs state: every other call derives its trace_id fresh from
        # the deterministic seed, see _trace_context().
        self._roots: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _safe(self, op_name: str, fn) -> None:
        """Run `fn`, swallowing and logging any exception. The single
        chokepoint every public method below routes through, so the
        fail-open guarantee lives in exactly one place."""
        if self._client is None:
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- observability fails open
            logger.warning(
                "langfuse.call_failed",
                extra={"op": op_name, "reason": type(exc).__name__,
                       "error": str(exc)[:300]},
            )

    def _trace_id(self, thread_id: str) -> str:
        """Deterministic trace_id from thread_id -- same thread_id always
        produces the same trace_id, so every observation for one run
        lands on one trace with no registry needed."""
        return self._client.create_trace_id(seed=thread_id)

    def _trace_context(self, thread_id: str):
        from langfuse.types import TraceContext
        return TraceContext(trace_id=self._trace_id(thread_id))

    @property
    def enabled(self) -> bool:
        """True only when a real, working client is behind this Observer.
        Every method below is safe to call unconditionally even when
        this is False -- it's an optimization hook, not a precondition."""
        return self._client is not None

    # ------------------------------------------------------------------
    # trace lifecycle
    # ------------------------------------------------------------------

    def start_trace(self, thread_id: str, name: str, *,
                     input: Any = None, metadata: Optional[dict] = None) -> None:
        """One root trace per user query (Phase 3's own requirement),
        keyed by thread_id/run_id -- the same identity already used for
        Postgres checkpointing and structured-log correlation. Modeled
        as a root SPAN (v4 has no separate "trace" object to open) that
        end_trace() closes."""
        def _do():
            root = self._client.start_observation(
                name=name, as_type="span",
                trace_context=self._trace_context(thread_id),
                input=input,
                metadata={**(metadata or {}), "session_id": thread_id,
                          "environment": self._settings.langfuse_environment,
                          "project": self._settings.langfuse_project or None},
            )
            self._roots[thread_id] = root
        self._safe("start_trace", _do)

    def end_trace(self, thread_id: str, *, output: Any = None,
                  metadata: Optional[dict] = None) -> None:
        def _do():
            root = self._roots.pop(thread_id, None)
            if root is None:
                logger.debug("langfuse.no_active_root_trace", extra={"thread_id": thread_id})
                return
            if output is not None or metadata is not None:
                root.update(output=output, metadata=metadata)
            root.end()
        self._safe("end_trace", _do)

    # ------------------------------------------------------------------
    # spans (non-LLM units of work: retrieval, memory, checkpointing, ...)
    # ------------------------------------------------------------------

    def span(self, thread_id: str, name: str, *,
             input: Any = None, output: Any = None,
             metadata: Optional[dict] = None,
             start_time: Optional[float] = None,
             end_time: Optional[float] = None,
             level: str = "DEFAULT") -> None:
        """Record one already-finished unit of work. `start_time`/
        `end_time` are `time.time()` floats (seconds) used only to
        compute a `duration_ms` metadata field for display -- v4's own
        `.end()` call takes no useful wall-clock override in this
        module's tested version, so wall-clock duration is carried as
        plain metadata instead of fighting the SDK's internal OTel
        timing."""
        def _do():
            duration_ms = None
            if start_time is not None and end_time is not None:
                duration_ms = round((end_time - start_time) * 1000, 2)
            obs = self._client.start_observation(
                name=name, as_type="span",
                trace_context=self._trace_context(thread_id),
                input=input, output=output,
                metadata={**(metadata or {}), "duration_ms": duration_ms},
                level=level,
            )
            obs.end()
        self._safe("span", _do)

    # ------------------------------------------------------------------
    # generations (LLM calls specifically -- carries usage + cost)
    # ------------------------------------------------------------------

    def generation(self, thread_id: str, name: str, *,
                   provider: str, model: str,
                   input: Any = None, output: Any = None,
                   prompt_tokens: int = 0, completion_tokens: int = 0,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   metadata: Optional[dict] = None) -> None:
        """One LLM call. Cost is computed here, once, from
        Settings-configured per-provider rates (see pricing.py) -- no
        call site needs to know how to price a token, only how many it
        used. `provider` should be one of FallbackRouter's own provider
        names ("primary", "mistral", "gemini") so pricing.py's lookup
        resolves; an unrecognized name simply reports no cost."""
        def _do():
            usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            cost = calculate_cost(self._settings, provider, usage)
            duration_ms = None
            if start_time is not None and end_time is not None:
                duration_ms = round((end_time - start_time) * 1000, 2)
            usage_details = {
                "input": int(prompt_tokens),
                "output": int(completion_tokens),
                "total": int(prompt_tokens + completion_tokens),
            }
            cost_details = None
            if cost is not None:
                cost_details = {
                    "input": cost.input_cost_usd,
                    "output": cost.output_cost_usd,
                    "total": cost.total_usd,
                }
            obs = self._client.start_observation(
                name=name, as_type="generation", model=model,
                trace_context=self._trace_context(thread_id),
                input=input, output=output,
                metadata={**(metadata or {}), "provider": provider, "duration_ms": duration_ms},
                usage_details=usage_details,
                cost_details=cost_details,
            )
            obs.end()
        self._safe("generation", _do)

    # ------------------------------------------------------------------
    # events (instantaneous facts: fallback hop, HITL trigger, error, ...)
    # ------------------------------------------------------------------

    def event(self, thread_id: str, name: str, *,
              input: Any = None, metadata: Optional[dict] = None,
              level: str = "DEFAULT") -> None:
        def _do():
            self._client.create_event(
                trace_context=self._trace_context(thread_id),
                name=name, input=input, metadata=metadata, level=level,
            )
        self._safe("event", _do)

    # ------------------------------------------------------------------
    # scores (recall, coverage, critique score, groundedness, ...)
    # ------------------------------------------------------------------

    def score(self, thread_id: str, name: str, value: Any, *,
              comment: Optional[str] = None) -> None:
        """`value` may be a float (0..1 scores like recall/coverage) or a
        string for categorical scores -- passed straight through; v4's
        create_score infers NUMERIC vs CATEGORICAL from the Python type."""
        def _do():
            self._client.create_score(
                name=name, value=value, trace_id=self._trace_id(thread_id),
                comment=comment,
            )
        self._safe("score", _do)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        self._safe("flush", lambda: self._client.flush())

    def shutdown(self) -> None:
        """Flush any buffered events, then release the client. Safe to
        call even if start_trace/end_trace were never called, and safe
        to call twice."""
        def _do():
            self._client.shutdown()
        self._safe("shutdown", _do)
        self._roots.clear()
