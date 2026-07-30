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
for the trace_id itself.

SESSION_ID / ENVIRONMENT GROUPING -- fixed and VERIFIED, not assumed
(this was an open gap in an earlier revision of this file). v4 has no
`session_id=` kwarg on `start_observation`/`create_event` directly; the
documented mechanism is the top-level `propagate_attributes
(session_id=..., environment=...)` context manager (OTel context/baggage
under the hood), entered in `start_trace()` and exited in `end_trace()`.
`create_score` LOOKS like an exception -- a score is not an OTel span
(it leaves through the separate score-ingestion path, which reads nothing
from OTel context) and the SDK's `create_score` signature does accept
`session_id=`. It must NOT be used anyway: the ingestion API accepts
exactly ONE score target and rejects traceId+sessionId together with HTTP
400. Scores pass traceId only and inherit the session from the trace they
attach to. See `score()` below.
Two things had to hold for this to be safe to rely on in THIS codebase,
and both were checked against the real, installed SDK and LangGraph with
an in-memory OTel span exporter, not inferred from docs alone:
  1. Does a span created well OUTSIDE the `with propagate_attributes(...)`
     block's own lexical scope -- which is how every call in this module
     works, since start_trace() opens the context and dozens of separate
     function calls across other files create spans long after that --
     still pick up the propagated session_id? CONFIRMED: entered the
     context, called `__enter__()` directly (not via `with`), created a
     span from a completely separate function afterward, and the
     exported span carried `session.id` correctly.
  2. Does it survive LangGraph's OWN parallel-fan-out thread dispatch
     (search_worker's `Send` fan-out runs on a background thread pool)?
     CONFIRMED by reproducing LangGraph's exact dispatch pattern --
     `langgraph/pregel/_executor.py::BackgroundExecutor.submit` calls
     `contextvars.copy_context()` then runs the task via `ctx.run(fn,
     ...)` on a `ThreadPoolExecutor` -- against the real SDK with an
     in-memory exporter: a span created inside that exact pattern still
     carried the correct `session.id`.
Both checks are reproduced as offline regression tests in
tests/unit/test_langfuse.py using a fake client (a live in-memory-
exporter check isn't reproducible in the normal offline suite, since it
depends on which exact SDK version happens to be installed) -- the fakes
assert the CALL SHAPE (propagate_attributes entered/exited with the
right kwargs), which is what this module actually controls; the SDK's
own behavior given that call shape was verified separately, above, by
hand, against the real package.

TRACE NESTING -- verified against the installed SDK, not assumed. Passing
`TraceContext(trace_id=...)` alone does NOT make an observation a child of
the run's root span; the SDK synthesizes a remote parent from the trace_id
and every observation comes out a sibling at the top of the trace, which is
what made traces read as a flat list. `TraceContext` also accepts an
OPTIONAL `parent_span_id`, and supplying the root span's own `.id` there
produces real OTel parentage. Confirmed by exporting spans through an
in-memory OTel exporter and reading `span.parent.span_id` back: with
`parent_span_id` the child's parent is the root's span id exactly, without
it the parent is an unrelated synthesized id.

The root span handle is already tracked in `self._roots` for end_trace's
sake, and `LangfuseSpan.id` is the OTel span id, so nesting needs NO new
state and NO call-site changes -- `_trace_context()` looks the open root up
on the way past. Two consequences worth knowing: (1) nesting is only
available while a root is open, so an observation emitted outside a
start_trace/end_trace pair still lands flat on the trace rather than being
dropped, and (2) `start_trace` itself must ask for a NON-nesting context
(`nest=False`), or a retry would parent the new root under the stale one it
is about to close.

THREAD SAFETY: `self._roots` and `self._session_contexts` are guarded by
one `threading.Lock`. Not because today's specific access pattern is
provably unsafe under CPython's GIL (it likely isn't, for the reasons
discussed when this was reviewed) -- but because this module's own
docstrings claim support for "long-lived processes like the API server,"
where genuinely concurrent requests (different thread_ids) DO write to
these dicts concurrently, and relying on GIL implementation details
instead of an explicit lock is the kind of thing that's cheap to just
not do.
"""

from __future__ import annotations

import logging
import threading
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
        # Guards both dicts below -- see the module docstring's Thread
        # Safety section.
        self._lock = threading.Lock()
        # Tracks OPEN root spans (start_trace -> end_trace pairs). Two
        # jobs: end_trace attaches final output/duration to the handle,
        # and _trace_context() reads `.id` off it to parent every other
        # observation under it (see the module docstring's Trace Nesting
        # section). The trace_id itself still needs no registry -- every
        # call derives it fresh from the deterministic seed.
        self._roots: Dict[str, Any] = {}
        # The open propagate_attributes(...) context per thread_id,
        # entered in start_trace and exited in end_trace -- this is what
        # makes session_id/environment grouping (see module docstring)
        # actually reach every span/generation/event for the run,
        # including ones created on LangGraph's parallel-fan-out worker
        # threads.
        self._session_contexts: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _safe(self, op_name: str, fn, thread_id: Optional[str] = None) -> None:
        """Run `fn`, swallowing and logging any exception. The single
        chokepoint every public method below routes through, so the
        fail-open guarantee lives in exactly one place.

        `thread_id`, when supplied, is validated here for the same reason:
        a blank run identity has no trace to attach anything to. Every
        non-CLI call site reads its thread_id from `run_id_var`
        (logging_setup.py), whose default is the EMPTY STRING -- and
        Langfuse's own `create_trace_id(seed="")` treats an empty seed as
        "no seed at all" and returns a fresh RANDOM id. So an observation
        emitted outside a run (a script importing SemanticMemory, say)
        would silently mint a brand-new orphan trace holding exactly one
        observation, every single call. Dropping it is the honest outcome,
        and a debug log says so."""
        if self._client is None:
            return
        if thread_id is not None and not str(thread_id).strip():
            logger.debug("langfuse.skipped_blank_thread_id",
                         extra={"op": op_name})
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

    def _trace_context(self, thread_id: str, *, nest: bool = True):
        """The TraceContext every observation for `thread_id` is created
        against. When `nest` is true (the default) and a root span is
        currently open for this thread_id, the root's span id is added as
        `parent_span_id`, which is what actually produces a nested trace
        rather than a flat list of siblings -- see the module docstring's
        Trace Nesting section.

        `nest=False` exists for exactly one caller: start_trace(), whose
        own root must never be parented under a previous, still-open root
        it is in the middle of replacing."""
        from langfuse.types import TraceContext
        ctx = TraceContext(trace_id=self._trace_id(thread_id))
        if not nest:
            return ctx
        with self._lock:
            root = self._roots.get(thread_id)
        # getattr, and an explicit str check, deliberately: a root handle
        # is whatever the SDK returned, and this must degrade to "flat,
        # as before" rather than raise if that object has no usable `.id`.
        parent_span_id = getattr(root, "id", None)
        if isinstance(parent_span_id, str) and parent_span_id:
            ctx["parent_span_id"] = parent_span_id
        return ctx

    def _close_root(self, thread_id: str, root: Any) -> None:
        """End one root span, best-effort. Shared by end_trace's replace
        path, start_trace's stale path, and shutdown's drain loop, so all
        three fail the same way and log the same event name."""
        if root is None:
            return
        try:
            root.end()
        except Exception as exc:  # noqa: BLE001 -- best-effort
            logger.warning("langfuse.root_end_failed",
                           extra={"thread_id": thread_id,
                                  "reason": type(exc).__name__})

    def _close_context(self, thread_id: str, ctx: Any) -> None:
        """Exit one propagate_attributes context, best-effort. Same
        sharing rationale as _close_root above."""
        if ctx is None:
            return
        try:
            ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 -- best-effort
            logger.warning("langfuse.context_exit_failed",
                           extra={"thread_id": thread_id,
                                  "reason": type(exc).__name__})

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
            # Enter propagate_attributes FIRST, so the root span itself
            # (created next) is already inside the propagation context --
            # the SDK's own docs warn pre-existing spans are never
            # retroactively updated, so this ordering is load-bearing,
            # not stylistic. See module docstring for what this was
            # verified to do (and not do) against the real SDK.
            from langfuse import propagate_attributes
            ctx = propagate_attributes(
                session_id=thread_id, trace_name=name,
                environment=self._settings.langfuse_environment or None,
            )
            ctx.__enter__()
            try:
                root = self._client.start_observation(
                    name=name, as_type="span",
                    trace_context=self._trace_context(thread_id, nest=False),
                    input=input,
                    metadata={**(metadata or {}),
                              "project": self._settings.langfuse_project or None},
                )
            except Exception:
                # start_observation failed AFTER the context was already
                # entered -- exit it here rather than leaking it, since
                # nothing else will ever reach the paired __exit__ in
                # end_trace() for a root that was never recorded.
                ctx.__exit__(None, None, None)
                raise
            with self._lock:
                stale_root = self._roots.pop(thread_id, None)
                stale_ctx = self._session_contexts.pop(thread_id, None)
                self._roots[thread_id] = root
                self._session_contexts[thread_id] = ctx
            # A second start_trace() for a thread_id that still has one open
            # (a retry, or any caller that never reached end_trace) used to
            # just overwrite both dicts. That dropped the previous root span
            # without ever calling .end() -- which in v4's OTel model means
            # it is never exported at all, not merely left incomplete -- and
            # dropped the previous propagate_attributes context without
            # __exit__, leaking its session_id onto this thread for good.
            # Close both, outside the lock, before moving on.
            self._close_root(thread_id, stale_root)
            self._close_context(thread_id, stale_ctx)
        self._safe("start_trace", _do, thread_id)

    def end_trace(self, thread_id: str, *, output: Any = None,
                  metadata: Optional[dict] = None) -> None:
        def _do():
            with self._lock:
                root = self._roots.pop(thread_id, None)
                ctx = self._session_contexts.pop(thread_id, None)
            try:
                if root is None:
                    logger.debug("langfuse.no_active_root_trace",
                                 extra={"thread_id": thread_id})
                else:
                    # try/finally, not a plain sequence: root.update() can
                    # raise on its own (an output payload the SDK cannot
                    # serialize, say), and _safe would then swallow that
                    # exception with root.end() never reached -- losing the
                    # WHOLE trace rather than just its final output, since
                    # an un-.end()ed span is never exported.
                    try:
                        if output is not None or metadata is not None:
                            root.update(output=output, metadata=metadata)
                    finally:
                        root.end()
            finally:
                # Exit the propagation context regardless of whether a root
                # span was found, and regardless of whether ending it
                # raised -- an orphaned open context left on this thread is
                # worse than a slightly-redundant close, since it would leak
                # session_id into whatever OTel activity happens next on
                # this thread/process.
                self._close_context(thread_id, ctx)
        self._safe("end_trace", _do, thread_id)

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
            meta = dict(metadata or {})
            # Only carry duration_ms when it was actually measured. Writing
            # the key with a None value made every span whose caller passed
            # no timings show an empty "duration_ms" field in the Langfuse
            # UI, which reads as "measured, and it was nothing" rather than
            # "not measured".
            if start_time is not None and end_time is not None:
                meta["duration_ms"] = round((end_time - start_time) * 1000, 2)
            obs = self._client.start_observation(
                name=name, as_type="span",
                trace_context=self._trace_context(thread_id),
                input=input, output=output,
                metadata=meta,
                level=level,
            )
            obs.end()
        self._safe("span", _do, thread_id)

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
            meta = dict(metadata or {})
            meta["provider"] = provider
            # Same reasoning as span() above: an unmeasured duration is
            # absent, not present-and-null.
            if start_time is not None and end_time is not None:
                meta["duration_ms"] = round((end_time - start_time) * 1000, 2)
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
                metadata=meta,
                usage_details=usage_details,
                cost_details=cost_details,
            )
            obs.end()
        self._safe("generation", _do, thread_id)

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
        self._safe("event", _do, thread_id)

    # ------------------------------------------------------------------
    # scores (recall, coverage, critique score, groundedness, ...)
    # ------------------------------------------------------------------

    def score(self, thread_id: str, name: str, value: Any, *,
              comment: Optional[str] = None) -> None:
        """`value` may be a float (0..1 scores like recall/coverage) or a
        string for categorical scores -- passed straight through; v4's
        create_score infers NUMERIC vs CATEGORICAL from the Python type.

        `trace_id` is the ONLY target passed, and that is deliberate. The
        SDK's create_score signature also accepts `session_id=`, and
        passing both looks like the way to get session grouping onto a
        score -- but the ingestion API requires exactly one of traceId /
        sessionId / datasetRunId and returns HTTP 400 ("Provide exactly
        one of the following: traceId (with optional observationId),
        sessionId or datasetRunId") for the combination, which silently
        dropped EVERY score in every run. Confirmed against the live API:
        traceId alone returns 201. Do not re-add it. Session grouping
        still works transitively -- the score attaches to the trace, and
        the trace carries the session from start_trace()'s
        propagate_attributes context."""
        def _do():
            self._client.create_score(
                name=name, value=value, trace_id=self._trace_id(thread_id),
                comment=comment,
            )
        self._safe("score", _do, thread_id)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        self._safe("flush", lambda: self._client.flush())

    def shutdown(self) -> None:
        """Flush any buffered events, then release the client. Safe to
        call even if start_trace/end_trace were never called, and safe
        to call twice.

        End every still-OPEN root span first. In v4's OTel model, a span
        that never had `.end()` called is never handed to the exporter
        at all -- not "possibly stuck," genuinely never sent. Before this
        fix, `self._roots.clear()` discarded the Python-side handle
        without ever calling `.end()`, so ANY run that reached shutdown()
        without going through end_trace() (a genuine crash, not just
        GraphRecursionError -- see cli.py's own try/finally, which is the
        primary fix) silently lost its entire trace, including exactly
        the crashed runs you'd most want visibility into. This is the
        backstop for whatever cli.py's try/finally doesn't catch, not a
        replacement for it -- see cli.py::_run's docstring.

        Also closes any still-open propagate_attributes context for the
        same reason: an unclosed one would otherwise leak session_id
        into whatever OTel activity happens on that thread next."""
        def _do():
            with self._lock:
                roots = list(self._roots.items())
                contexts = list(self._session_contexts.items())
                self._roots.clear()
                self._session_contexts.clear()
            for thread_id, root in roots:
                self._close_root(thread_id, root)
            for thread_id, ctx in contexts:
                self._close_context(thread_id, ctx)
            self._client.shutdown()
        self._safe("shutdown", _do)
        with self._lock:
            self._roots.clear()
            self._session_contexts.clear()
        # Actually release the client, which is half of what this method's
        # own name and docstring promise. Keeping the reference meant
        # `enabled` stayed True after shutdown and every later call went to
        # a client the SDK had already torn down -- each one failing into a
        # langfuse.call_failed warning rather than the silent no-op the
        # module's fail-open contract describes. Dropping it here also makes
        # a second shutdown() genuinely free instead of merely harmless.
        self._client = None
