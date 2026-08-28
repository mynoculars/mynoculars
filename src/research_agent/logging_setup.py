"""
logging_setup.py -- the ONE instrumentation path every module calls.

Purpose:
    log_event() is the only function any module in this codebase calls to
    record anything; JsonLineFormatter is the machine-parseable rendering
    of that same stream, always on. The human-readable second rendering
    (one execution narrative per run, logs/run-<run_id>.txt, written only
    when --debug / DEBUG_TRACE=true) lives in reporting/narrative.py (S-2)
    -- a presentation-layer report generator, not a second instrumentation
    path, and split out so it is no longer imported by every module that
    only ever wants log_event/run_id_var.

Responsibilities:
    - Configure the root logger's JSON handler exactly once
      (configure_logging).
    - Provide log_event(), the single helper every module uses, so log
      structure stays uniform without pulling in a logging framework.
    - Own run_id_var, the correlation id every log line carries.

Design decision (stdlib over structlog):
    structlog is nicer, but a reference implementation should minimize
    dependencies a learner must understand. Plain logging.Formatter/Handler
    subclasses over stdlib logging cover everything this project needs.

Python mechanics used in this file, if any of this is new to you:
    ContextVar (from contextvars, stdlib)
        A special kind of variable designed for concurrent/async code: each
        separate "context" of execution (roughly, each request or each
        invoke() call in this codebase) can set its OWN value for the same
        ContextVar without stepping on another context's value -- unlike a
        normal global variable, which would be shared and overwritten by
        whichever code ran most recently. Here it lets multiple runs happen
        "at once" (e.g. the API serving two /research requests concurrently)
        while each run's log lines still carry only ITS OWN run_id.
    logging.Formatter subclass
        Python's stdlib logging module lets you plug in a custom
        "Formatter" -- an object responsible for turning one LogRecord (the
        internal object logging.info(...)/logging.warning(...) etc. build)
        into the final string that actually gets written out. Subclassing
        Formatter and overriding its format() method is the standard way to
        change what a log line looks like; here it becomes one JSON object
        instead of the plain-text default.
    **fields: Any   (in log_event's signature)
        The ** prefix on a parameter means "collect any number of EXTRA
        keyword arguments the caller passes, and bundle them into a dict
        named `fields`". So a call like
            log_event(logger, "llm.call", provider="mistral", latency_s=0.4)
        results in fields = {"provider": "mistral", "latency_s": 0.4}
        inside log_event, with no need to declare "provider" or "latency_s"
        as named parameters up front -- different call sites pass completely
        different sets of extra fields, and this is what makes that legal.
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# Correlation ID for the current run (review item: end-to-end traceability).
# Set once per invoke (cli/api); every log line then carries run_id, so a
# grep on one id reconstructs one run even with interleaved parallel logs.
#
# ContextVar("run_id", default="") declares a context-local variable named
# "run_id" (the name is just for debugging/repr purposes) whose value is ""
# until something calls run_id_var.set(...) -- see cli.py::main and
# api/server.py, which each do this once per run/request.
run_id_var: ContextVar[str] = ContextVar("run_id", default="")


class JsonLineFormatter(logging.Formatter):
    """Render each record as one JSON object per line.

    Every call to logging.info(...), logging.warning(...), etc. anywhere in
    the codebase eventually reaches this format() method, which is what
    turns it into the actual JSON text written to stderr.
    """

    # Fields that exist ONLY for NarrativeFormatter's benefit (the full
    # prompt transcript / raw response / raw retrieval hits — see
    # llm/client.py::OpenAICompatibleClient.complete and
    # storage/{qdrant_store,opensearch_store}.py::*.search, which attach
    # these to the SAME log_event("llm.call", ...) / log_event(
    # "retrieval.raw", ...) call the JSON line below already needed,
    # rather than making a second call). Dropping them here keeps the JSON
    # line the same size it was before narrative capture existed — a
    # 2000-token compiler prompt has no business bloating the machine log
    # every debug run; NarrativeFormatter reads them straight off
    # record.event_fields, bypassing this formatter entirely.
    _NARRATIVE_ONLY_KEYS = frozenset({"prompt_messages", "response", "hits"})

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        # record is the stdlib LogRecord — it carries the log level, the
        # message, the logger name, and (if we attached one — see log_event
        # below) our custom "event_fields" dict of extra structured data.
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            # Provenance (P2-15 follow-up): funcName/lineno are already on
            # every stdlib LogRecord — see log_event's stacklevel=2 below
            # for why these correctly name the CALLER of log_event(), not
            # log_event() itself.
            "func": record.funcName,
            "line": record.lineno,
        }
        run_id = run_id_var.get()
        if run_id:
            payload["run_id"] = run_id
        # getattr(record, "event_fields", None) reads an attribute that may
        # or may not exist on this particular record — log_event() below
        # attaches "event_fields" via the `extra=` argument to logger.log(),
        # but a log line written by some OTHER library (e.g. httpx) won't
        # have it, so this needs to tolerate its absence rather than assume
        # it's always there. That's what the `None` default is for.
        extra = getattr(record, "event_fields", None)
        if extra:
            payload.update({k: v for k, v in extra.items()
                            if k not in self._NARRATIVE_ONLY_KEYS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # json.dumps(..., default=str) turns the dict into a JSON string.
        # default=str is a fallback: if any value in the dict is something
        # JSON doesn't natively know how to serialize (e.g. a Volatility
        # enum member, or an arbitrary object), call str(...) on it instead
        # of raising an error.
        return json.dumps(payload, ensure_ascii=False, default=str)


class ProblemCollector(logging.Handler):
    """Keeps every WARNING-and-above record of this process, for D-118.

    WHY THIS EXISTS: the narrative file (D-117) shows an operator what
    went wrong, but it is written only under --debug/DEBUG_TRACE. A
    normal run's warnings go past in a JSON stream -- run
    p205.265-check's 403, the one saying the provider account had no
    credits, was one line among six hundred. An administrator watching a
    scheduled run sees stdout and an exit code, and that is all.

    Deliberately NOT a second logging path: this stores the records the
    existing log_event calls already produce, changing nothing about what
    is logged or how. cli.py drains it once at the end of a run and
    prints a summary.

    BOUNDED at _MAX_PROBLEMS. A run that somehow warns in a loop must not
    turn a diagnostic aid into a memory leak; past the cap the overflow
    is counted, and the count is reported, so the summary never claims to
    be complete when it is not.
    """

    _MAX_PROBLEMS = 200

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list = []
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) >= self._MAX_PROBLEMS:
            self.dropped += 1
            return
        self.records.append(
            (record.levelname, record.getMessage(),
             dict(getattr(record, "event_fields", None) or {})))


_problem_collector: "ProblemCollector | None" = None


def drain_problems() -> tuple:
    """Return (records, dropped) collected so far, and reset (D-118).

    CALLED BY   cli.py, once, at the end of a run -- both the normal path
                and the failure path. Draining rather than peeking keeps
                a long-lived process (the API server) from reporting one
                request's warnings against the next.
    RETURNS     ([(level, event_name, fields), ...], overflow_count), or
                ([], 0) when logging was never configured.
    """
    if _problem_collector is None:
        return [], 0
    records, dropped = _problem_collector.records, _problem_collector.dropped
    _problem_collector.records, _problem_collector.dropped = [], 0
    return records, dropped


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger. Safe to call twice.

    CALLED BY   cli.py::build_app_and_settings, api/server.py at import
                time — both call this before any node runs, so every log
                line the graph ever produces goes through JsonLineFormatter.
    """
    # logging.getLogger() with NO name argument returns the ROOT logger —
    # the ancestor of every other logger in the process (every module in
    # this codebase does `logger = logging.getLogger(__name__)`, and those
    # loggers all funnel up to this root one unless told otherwise).
    root = logging.getLogger()
    # getattr(root, "_agent_configured", False): the root logger object is
    # just a regular Python object, so we can stash our own custom flag on
    # it (root._agent_configured = True, a few lines down) to remember
    # "we already did this setup" — guards against configure_logging()
    # being called twice (e.g. once by cli.py, once by a test fixture) and
    # ending up with duplicate handlers, which would print every log line
    # twice.
    if getattr(root, "_agent_configured", False):
        return
    # JsonLineFormatter emits ensure_ascii=False, so log lines can contain
    # non-ASCII (a query in any non-English script, an em dash in a node
    # message, a corpus snippet). On a Windows console still defaulting to
    # a legacy code page such as cp1252, writing those raises
    # UnicodeEncodeError from inside the logging handler -- which surfaces
    # as a "--- Logging error ---" traceback on stderr and silently drops
    # the log line, i.e. the run loses exactly the diagnostic output it was
    # trying to produce. reconfigure() exists on Python 3.7+ TextIOWrapper;
    # guarded because sys.stderr may be a plain object under pytest capture
    # or a redirected pipe.
    reconfigure = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):  # already-detached or non-seekable stream
            pass
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    # root.handlers[:] = [handler, *narrative] REPLACES the entire list of
    # handlers attached to the root logger with a list containing just the
    # JSON handler PLUS any NarrativeBufferHandler already attached — cli.py
    # constructs a Tracer (which calls enable_narrative_logging()) BEFORE
    # calling build_app_and_settings (which calls this function), so without
    # this the narrative handler would be silently dropped the instant a
    # --debug run's configure_logging call fired. The [:] means "replace the
    # contents of this existing list object in place" rather than creating a
    # brand new list and rebinding the name, which matters if anything else
    # is holding a reference to the old list.
    from research_agent.reporting.narrative import NarrativeBufferHandler
    narrative = [h for h in root.handlers if isinstance(h, NarrativeBufferHandler)]
    # D-118: built into the replacement list, NOT addHandler'd before it.
    # That assignment REPLACES the whole handler list -- which is exactly
    # why `narrative` is re-collected and passed through above, and the
    # collector needs the same treatment or it is silently discarded one
    # line after being attached. (It was, on the first attempt: the
    # collector existed, held no records, and the console summary printed
    # nothing at all.)
    global _problem_collector
    if _problem_collector is None:
        _problem_collector = ProblemCollector()
    root.handlers[:] = [handler, *narrative, _problem_collector]
    root.setLevel(level.upper())
    root._agent_configured = True  # type: ignore[attr-defined]

    # Mutes the third-party stack traces while keeping your clean degradation log lines. Level 1 run
    logging.getLogger("opensearch").setLevel(logging.ERROR)
    logging.getLogger("qdrant_client").setLevel(logging.ERROR)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event — the ONE function every module in this
    codebase calls instead of logger.info()/logger.warning() directly, so
    every log line ends up with the same JSON shape.

    Parameters:
        logger: the module logger (logging.getLogger(__name__)).
        msg:    short human-readable event name, e.g. "llm.fallback".
        level:  stdlib level constant.
        fields: arbitrary structured payload (node, duration_ms, counts...).
            See the **fields explanation in the module docstring above for
            what this "catch everything else" parameter syntax means.

    logger.log(level, msg, extra={"event_fields": fields}) is the stdlib
    call underneath: `extra=` is stdlib logging's own mechanism for
    attaching arbitrary data to a LogRecord, which is exactly what
    JsonLineFormatter.format() above reads back out via
    getattr(record, "event_fields", None). stacklevel=2 tells stdlib
    logging to attribute funcName/lineno to log_event's OWN caller — the
    line that actually decided to log something — rather than to this
    line inside log_event itself, which every call site would otherwise
    show identically and uselessly.
    """
    logger.log(level, msg, extra={"event_fields": fields}, stacklevel=2)
