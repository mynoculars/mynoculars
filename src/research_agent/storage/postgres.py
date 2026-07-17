"""
storage/postgres.py — Durable checkpointing and run history.

Purpose:
    Provide the LangGraph checkpointer (design decision D-8: every run is
    resumable and inspectable under its thread_id) and a small run-history
    table for after-the-fact inspection.

Responsibilities:
    - get_checkpointer(): PostgresSaver when Postgres is reachable,
      in-memory MemorySaver otherwise — same graceful-degradation policy as
      the other storage modules, so the agent runs with zero infrastructure.
    - record_run(): one row per completed run (query, recall, telemetry).

Design decision (why degradation instead of hard dependency):
    A learner's first run should succeed on a bare laptop. Durability is an
    upgrade you turn on by starting docker-compose, not a prerequisite. The
    log line at startup makes the active mode unambiguous.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    query TEXT NOT NULL,
    recall REAL,
    telemetry JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


def get_checkpointer(dsn: str) -> Tuple[Any, bool]:
    """Return (checkpointer, durable_flag).

    Tries PostgresSaver first; on any failure returns MemorySaver with
    durable_flag=False so callers can surface the mode to the user.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg
        conn = psycopg.connect(dsn, autocommit=True)
        saver = PostgresSaver(conn)
        saver.setup()
        log_event(logger, "checkpointer.postgres_active")
        return saver, True
    except Exception as exc:  # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver
        log_event(logger, "checkpointer.memory_fallback",
                  level=logging.WARNING, reason=type(exc).__name__)
        return MemorySaver(), False


def record_run(dsn: str, thread_id: str, query: str,
               recall: float, telemetry: Dict[str, Any]) -> Optional[int]:
    """Insert one run-history row; returns row id or None when degraded."""
    try:
        import psycopg
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(_RUNS_DDL)
            row = conn.execute(
                "INSERT INTO agent_runs (thread_id, query, recall, telemetry) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (thread_id, query, recall, json.dumps(telemetry)),
            ).fetchone()
            return int(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "run_history.skipped", level=logging.WARNING,
                  reason=type(exc).__name__)
        return None
