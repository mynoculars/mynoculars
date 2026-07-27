"""
logging_setup.py — Structured (JSON-lines) logging for the whole agent.

Purpose:
    One-call logging configuration producing machine-parseable JSON lines so
    a reader can reconstruct an entire run: node order, tool calls, LLM
    requests, fallback decisions, timings, errors.

Responsibilities:
    - Configure the root logger exactly once.
    - Provide log_event(), the single helper every module uses, so log
      structure stays uniform without pulling in a logging framework.

Design decision (stdlib over structlog):
    structlog is nicer, but a reference implementation should minimize
    dependencies a learner must understand. A 30-line JSON formatter over
    stdlib logging covers everything this project needs. Deferred: log
    correlation IDs per run — easy later addition (put thread_id into every
    log_event call site).

Python mechanics used in this file, if any of this is new to you:
    ContextVar (from contextvars, stdlib)
        A special kind of variable designed for concurrent/async code: each
        separate "context" of execution (roughly, each request or each
        invoke() call in this codebase) can set its OWN value for the same
        ContextVar without stepping on another context's value — unlike a
        normal global variable, which would be shared and overwritten by
        whichever code ran most recently. Here it lets multiple runs happen
        "at once" (e.g. the API serving two /research requests concurrently)
        while each run's log lines still carry only ITS OWN run_id.
    logging.Formatter subclass
        Python's stdlib logging module lets you plug in a custom
        "Formatter" — an object responsible for turning one LogRecord (the
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
        as named parameters up front — different call sites pass completely
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
# until something calls run_id_var.set(...) — see cli.py::main and
# api/server.py, which each do this once per run/request.
run_id_var: ContextVar[str] = ContextVar("run_id", default="")


class JsonLineFormatter(logging.Formatter):
    """Render each record as one JSON object per line.

    Every call to logging.info(...), logging.warning(...), etc. anywhere in
    the codebase eventually reaches this format() method, which is what
    turns it into the actual JSON text written to stderr.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        # record is the stdlib LogRecord — it carries the log level, the
        # message, the logger name, and (if we attached one — see log_event
        # below) our custom "event_fields" dict of extra structured data.
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
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
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # json.dumps(..., default=str) turns the dict into a JSON string.
        # default=str is a fallback: if any value in the dict is something
        # JSON doesn't natively know how to serialize (e.g. a Volatility
        # enum member, or an arbitrary object), call str(...) on it instead
        # of raising an error.
        return json.dumps(payload, ensure_ascii=False, default=str)


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
    # root.handlers[:] = [handler] REPLACES the entire list of handlers
    # attached to the root logger with a list containing just this one —
    # the [:] means "replace the contents of this existing list object in
    # place" rather than creating a brand new list and rebinding the name,
    # which matters if anything else is holding a reference to the old list.
    root.handlers[:] = [handler]
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
    getattr(record, "event_fields", None).
    """
    logger.log(level, msg, extra={"event_fields": fields})
