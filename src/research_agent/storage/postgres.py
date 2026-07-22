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

This file has TWO INDEPENDENT WRITERS to the same Postgres database, and
it's worth being explicit about which is which:
    - The LangGraph library itself owns and writes the checkpointer tables
      (checkpoints, checkpoint_blobs, checkpoint_writes,
      checkpoint_migrations) — get_checkpointer below only CONNECTS and
      calls saver.setup() (which creates those tables if missing); nothing
      in this codebase ever writes to them directly afterward. LangGraph's
      own internals handle that every time the graph runs a superstep.
    - This file's OWN code owns the "agent_runs" table (the _RUNS_DDL below
      and record_run's INSERT) — that one is this project's, written by
      and read only by this project's code.

Python mechanics used in this file, if any of this is new to you:
    Tuple[Any, bool]  (the return type of get_checkpointer)
        A TUPLE is an ordered, fixed-size collection of values, like a list
        but immutable. Tuple[Any, bool] means "exactly two values: some
        object of any type, followed by a bool" — here, (checkpointer,
        durable_flag). The caller unpacks it with
        `checkpointer, durable = get_checkpointer(dsn)`, binding the first
        tuple element to `checkpointer` and the second to `durable` in one
        line.
    "with psycopg.connect(dsn, autocommit=True) as conn:"
        A CONTEXT MANAGER, same mechanism explained in tracing.py's
        flush() method — it opens a database connection, runs the indented
        block with `conn` bound to that open connection, and guarantees the
        connection gets closed afterward even if something inside the block
        raises an exception.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

# The SQL that creates this project's OWN table (as opposed to LangGraph's
# checkpoint tables, which LangGraph creates itself — see the module
# docstring). "CREATE TABLE IF NOT EXISTS" makes this safe to run on EVERY
# call to record_run below — if the table already exists, this statement is
# simply a no-op rather than an error.
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

    CALLED BY   cli.py::build_app_and_settings — once per run/process, the
                only call site. Its return value is handed straight into
                orchestration/graph.py::build_graph as the `checkpointer`
                argument, which is what LangGraph uses to persist state
                after every superstep (this is what makes --thread-id
                resume and human-escalation pause/resume possible at all).
    RETURNS     (a real PostgresSaver, True) if Postgres is reachable and
                its setup() call succeeds; otherwise (an in-memory
                MemorySaver, False). The boolean is a signal callers COULD
                use to warn the user that runs won't survive a process
                restart — though note that today build_app_and_settings
                receives this flag and does not currently do anything with
                it (a known gap, not a design decision).

    Tries PostgresSaver first; on any failure returns MemorySaver with
    durable_flag=False so callers can surface the mode to the user.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg
        # autocommit=True means each SQL statement takes effect immediately
        # rather than needing an explicit conn.commit() call — appropriate
        # here since PostgresSaver manages its own transactions internally
        # once it has a connection to use.
        conn = psycopg.connect(dsn, autocommit=True)
        saver = PostgresSaver(conn)
        # setup() creates the four checkpoint* tables (see the module
        # docstring) if they don't already exist — this is LangGraph's own
        # method, not custom code in this file.
        saver.setup()
        log_event(logger, "checkpointer.postgres_active")
        return saver, True
    except Exception as exc:  # noqa: BLE001
        # Any failure connecting to Postgres, or PostgresSaver.setup()
        # itself failing, lands here — the whole try block is treated as
        # one unit: either Postgres-backed durability fully works, or we
        # fall all the way back to the in-memory alternative below.
        from langgraph.checkpoint.memory import MemorySaver
        log_event(logger, "checkpointer.memory_fallback",
                  level=logging.WARNING, reason=type(exc).__name__)
        return MemorySaver(), False


def record_run(dsn: str, thread_id: str, query: str,
               recall: float, telemetry: Dict[str, Any]) -> Optional[int]:
    """Insert one run-history row; returns row id or None when degraded.

    CALLED BY   cli.py::main, exactly once, right after a run finishes and
                its report/telemetry have already been printed. NEVER
                called from api/server.py — API-driven runs get no history
                row in agent_runs, a known asymmetry between the two
                interfaces.
    WRITES      one row into the agent_runs table described by _RUNS_DDL
                above — nothing else in this codebase reads that table back
                afterward; it exists purely for manual inspection (e.g.
                with a SQL client).
    """
    try:
        import psycopg
        with psycopg.connect(dsn, autocommit=True) as conn:
            # Run the CREATE TABLE IF NOT EXISTS on every single call —
            # cheap and idempotent, and means this function never depends
            # on some separate migration step having been run beforehand.
            conn.execute(_RUNS_DDL)
            # %s placeholders are psycopg's parameterized-query syntax —
            # values are passed SEPARATELY from the SQL string (as the
            # tuple on the next line) rather than being string-formatted
            # directly into it. This is the standard, safe way to include
            # user-controlled or dynamic values (like the query text) in
            # SQL — it prevents SQL-injection-style bugs that string
            # formatting (e.g. an f-string) would risk.
            row = conn.execute(
                "INSERT INTO agent_runs (thread_id, query, recall, telemetry) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (thread_id, query, recall, json.dumps(telemetry)),
            ).fetchone()
            # fetchone() returns the single result row of the RETURNING id
            # clause above, as a tuple like (42,); row[0] pulls out just
            # the integer id. `if row else None` guards against fetchone()
            # somehow returning nothing at all (shouldn't normally happen
            # with RETURNING, but costs nothing to check).
            return int(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001
        # Any database problem here (unreachable, permissions, etc.) simply
        # means this run's history row is skipped — it does NOT affect the
        # report or telemetry the user already saw printed to stdout before
        # this function was even called.
        log_event(logger, "run_history.skipped", level=logging.WARNING,
                  reason=type(exc).__name__)
        return None
