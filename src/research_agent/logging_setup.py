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
run_id_var: ContextVar[str] = ContextVar("run_id", default="")


class JsonLineFormatter(logging.Formatter):
    """Render each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        run_id = run_id_var.get()
        if run_id:
            payload["run_id"] = run_id
        extra = getattr(record, "event_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger. Safe to call twice."""
    root = logging.getLogger()
    if getattr(root, "_agent_configured", False):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root._agent_configured = True  # type: ignore[attr-defined]


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event.

    Parameters:
        logger: the module logger (logging.getLogger(__name__)).
        msg:    short human-readable event name, e.g. "llm.fallback".
        level:  stdlib level constant.
        fields: arbitrary structured payload (node, duration_ms, counts...).
    """
    logger.log(level, msg, extra={"event_fields": fields})
