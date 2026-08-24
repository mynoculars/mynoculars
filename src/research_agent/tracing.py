"""
tracing.py — the on/off switch for narrative logging, not a second recorder.

Purpose:
    When enabled (--debug on the CLI, or DEBUG_TRACE=true in .env), turn on
    the human-readable narrative view of this run's events and, at the end
    of the run, flush that run's buffered narrative to a single file:
    logs/run-<run_id>.txt.

    Before this design, Tracer recorded events itself — record_llm() and
    record_retrieval() built their own banner-formatted strings and
    appended them to a private buffer, SEPARATELY from (and in addition
    to) the log_event() call already describing the same LLM/retrieval
    call at each site (llm/client.py::OpenAICompatibleClient.complete,
    storage/{qdrant_store,opensearch_store}.py::*.search). That was two
    independent instrumentation paths recording one event. There is now
    exactly one: log_event(). Every one of those call sites now attaches
    its full prompt/response (or full retrieval hits) directly onto the
    SAME log_event("llm.call", ...) / log_event("retrieval.raw", ...) call
    it already had to make for the plain JSON line, instead of making a
    second call to this module. Tracer's only remaining job is to turn on
    the handler that renders those same events as prose
    (logging_setup.py::NarrativeBufferHandler) and, later, flush this
    run's share of it to disk.

Responsibilities:
    - Tracer: enables narrative capture for the process (idempotent — see
      logging_setup.py::enable_narrative_logging) and flushes ONE run's
      buffered narrative to logs/run-<run_id>.txt.
    - NullTracer: the disabled case. Enables nothing, flushes nothing —
      the non-debug path pays no formatting or I/O cost, exactly as before.

Design decision (thin switch, not a second sink):
    The tracer is still injected the same way every other dependency is
    (see cli.py), so it is testable and has no import-time side effects.
    What changed is WHAT it does when called — it no longer owns any
    formatting logic or file-writing logic itself; that all lives in
    logging_setup.py now, shared with the JSON view, so there is exactly
    one place that turns a LogRecord into text, not two.

Python mechanics used in this file, if any of this is new to you:
    class NullTracer(Tracer):
        NullTracer INHERITS from Tracer (the parenthesized name after the
        class name is its "base class" / "parent class"). This means
        NullTracer automatically has every method Tracer defines UNLESS it
        explicitly overrides that method with its own version — which it
        does for every method below (__init__, flush), replacing each one
        with a version that does nothing / returns immediately. The
        benefit: any code elsewhere that expects "an object with a
        .flush() method" (a Tracer) works identically whether it was
        actually handed a real Tracer or a NullTracer — the caller never
        needs an `if tracing_is_on` check anywhere.
    @property
        A decorator (see agents/gathering.py's docstring for what a
        decorator is) that lets you call a method WITHOUT parentheses, as
        if it were a plain attribute: `tracer.enabled` instead of
        `tracer.enabled()`. It's used here so callers can write
        `if tracer.enabled:` and get real, computed behaviour (in
        NullTracer's case, always False) rather than exposing a raw
        attribute that some code path could accidentally overwrite.
"""

from __future__ import annotations

from typing import Optional

from research_agent.reporting.narrative import enable_narrative_logging, flush_narrative


class Tracer:
    """Turns on narrative logging for the process; flushes one run's
    buffered narrative to a file.

    A single Tracer instance is created per invoke (keyed by run_id) and
    passed to every component that touches an external boundary — those
    components no longer call anything ON this object (record_llm/
    record_retrieval are gone); they only ever check `tracer.enabled` to
    decide whether to attach the heavy prompt/response/hits fields to the
    log_event() call they were already making. See llm/client.py and
    storage/{qdrant_store,opensearch_store}.py.
    """

    def __init__(self, run_id: str, log_dir: str = "logs"):
        """
        CALLED BY   cli.py::main (once per CLI invocation, only when
                    --debug or DEBUG_TRACE is set — otherwise a NullTracer
                    is built instead and this __init__ never runs).
        """
        self.run_id = run_id
        self._log_dir = log_dir
        # Idempotent — see logging_setup.py::enable_narrative_logging. Safe
        # to call from every Tracer instance in a long-lived API process;
        # only the first call actually attaches the handler.
        enable_narrative_logging()

    @property
    def enabled(self) -> bool:
        return True

    def flush(self) -> Optional[str]:
        """Write this run's buffered narrative lines to
        logs/run-<run_id>.txt. Returns the path written, or None if
        nothing was buffered for this run_id.

        CALLED BY   cli.py::main, exactly once, right after the graph
                    invoke loop finishes (whether the run completed
                    normally or went through a human-escalation pause/
                    resume cycle).
        """
        return flush_narrative(self.run_id, self._log_dir)


class NullTracer(Tracer):
    """No-op tracer used when tracing is disabled. Enables nothing, writes
    nothing — the non-debug path pays no formatting or I/O cost.

    See the module docstring above for what it means that this class
    INHERITS from Tracer and overrides its methods.
    """

    def __init__(self) -> None:  # noqa: D107 — intentionally skips base init
        # Deliberately does NOT call Tracer.__init__(self, ...) (there is no
        # super().__init__() here) — NullTracer never needs self.run_id or
        # self._log_dir, and must NOT call enable_narrative_logging(),
        # which is the entire point of the disabled path costing nothing.
        pass

    @property
    def enabled(self) -> bool:
        return False

    def flush(self) -> Optional[str]:
        return None
