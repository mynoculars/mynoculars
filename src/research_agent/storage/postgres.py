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
    - record_run(): one row per COMPLETED run (query, recall, telemetry).
    - record_failed_run(): one row per FAILED run (D-103) -- recall NULL,
      telemetry {"run_outcome": "failed", "failure": {...}}. Both go
      through the same private _insert_run.
    Read back by scripts/analyze_runs.py (D-92). BOTH interfaces call
    record_run: cli.py unconditionally after printing the report, and
    api/server.py::_respond on its "done" branch (P2-08). Only
    record_failed_run is CLI-only -- see D-121.

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
from research_agent.state import Evidence, Goal, SearchTask, Volatility

logger = logging.getLogger(__name__)

# Every custom type that can appear ANYWHERE inside ResearchState and
# therefore ever needs to be checkpointed to Postgres (or held by the
# in-memory fallback). Passing the classes themselves here — rather than
# string tuples like ("research_agent.state", "Goal") — is deliberate:
# LangGraph's allowlist accepts either form, and passing the actual class
# means this list can never silently drift out of sync with a future
# rename of the module or the class itself (see build_checkpoint_serde
# below for exactly how this gets used).
_CHECKPOINTABLE_STATE_TYPES = (Goal, Volatility, Evidence, SearchTask)


def _build_checkpoint_serde():
    """Build the (de)serializer used by BOTH the Postgres saver and the
    in-memory fallback, with this project's own state types explicitly
    allowlisted.

    CALLED BY   get_checkpointer, below — both branches (Postgres success
                and the in-memory fallback) use the SAME serde, so a run
                that starts durable and one that degrades to in-memory
                behave identically with respect to this warning.
    WHY THIS EXISTS: without an explicit allowlist, LangGraph's default
    JsonPlusSerializer still round-trips our Goal/Volatility/Evidence/
    SearchTask objects correctly (nothing was ever actually broken) — but
    it does so via a "warn but allow" path, logging one WARNING per type
    the first time each is deserialized in a process:
        Deserializing unregistered type research_agent.state.Goal from
        checkpoint. This will be blocked in a future version. Set
        LANGGRAPH_STRICT_MSGPACK=true to block now, or add to
        allowed_msgpack_modules to allow explicitly: [...]
    That message is not cosmetic noise to silence — it is LangGraph telling
    you plainly that this leniency is being phased out, and that a future
    version (or LANGGRAPH_STRICT_MSGPACK=true set today) would BLOCK
    deserialization entirely, which would break --thread-id resume and
    HITL pause/resume outright: the checkpointer would no longer be able
    to reconstruct this project's own state. Explicitly allowlisting our
    four types here is the fix LangGraph's own warning is asking for.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return JsonPlusSerializer(allowed_msgpack_modules=list(_CHECKPOINTABLE_STATE_TYPES))

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


def close_checkpointer(checkpointer: Any) -> None:
    """Close the underlying Postgres connection a checkpointer holds, if any
    (P2-08).

    CALLED BY   cli.py::main, in a `finally` block wrapping the whole run,
                and api/server.py's FastAPI shutdown handler — the two
                processes that ever construct a checkpointer via
                get_checkpointer, below.
    WHY THIS EXISTS: get_checkpointer's PostgresSaver branch opens a
    psycopg connection that nothing in this codebase previously ever
    closed — harmless for a single short CLI run that exits right after
    (the OS reclaims it), but a real gap for a long-lived FastAPI process
    that could rebuild the graph (and therefore a fresh connection) more
    than once in its lifetime, and in general just bad practice for a
    resource this codebase itself opened. A MemorySaver (the degraded
    fallback) holds no connection at all, so this is a safe no-op for it.
    getattr(checkpointer, "conn", None) is used rather than a type check
    because PostgresSaver's connection attribute is an implementation
    detail of the langgraph library, not part of its public contract —
    this degrades to "do nothing" rather than raising if that attribute
    name ever changes upstream.
    """
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        try:
            # A ConnectionPool exposes close() too, so this one call covers
            # both branches of get_checkpointer below (pooled and single
            # connection) with no type check.
            conn.close()
            log_event(logger, "checkpointer.closed")
        except Exception as exc:  # noqa: BLE001 — closing is best-effort
            log_event(logger, "checkpointer.close_failed", level=logging.WARNING,
                      reason=type(exc).__name__)


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
        # A POOL when psycopg_pool is installed, a single connection
        # otherwise. LangGraph dispatches every fanned-out search_worker in
        # one superstep and checkpoints after it, so all of those writes
        # previously queued behind ONE connection — psycopg3 is thread-safe
        # but serialises on an internal lock, so concurrency at the graph
        # level was silently serialised at the storage level. A pool sized
        # for the fan-out removes that ceiling.
        #
        # Falls back rather than hard-requiring the extra: psycopg_pool
        # ships as `psycopg[pool]`, and an existing install that only has
        # `psycopg[binary]` must keep working exactly as before.
        conn = None
        try:
            from psycopg_pool import ConnectionPool
            conn = ConnectionPool(dsn, min_size=1, max_size=10, open=True,
                                  kwargs={"autocommit": True})
            log_event(logger, "checkpointer.pool_active", max_size=10)
        except ImportError:
            # autocommit=True means each SQL statement takes effect
            # immediately rather than needing an explicit conn.commit()
            # call — appropriate here since PostgresSaver manages its own
            # transactions internally once it has a connection to use.
            conn = psycopg.connect(dsn, autocommit=True)
            log_event(logger, "checkpointer.single_connection",
                      level=logging.WARNING,
                      reason="psycopg_pool not installed; install psycopg[pool]")
        # serde=_build_checkpoint_serde() is the fix for the LangGraph
        # "Deserializing unregistered type" warning — see that function's
        # docstring above for exactly why this matters, not just how.
        saver = PostgresSaver(conn, serde=_build_checkpoint_serde())
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
        # Same serde as the Postgres branch above — a degraded, in-memory
        # run should be just as free of this warning as a durable one.
        return MemorySaver(serde=_build_checkpoint_serde()), False


def record_run(dsn: str, thread_id: str, query: str,
               recall: Optional[float],
               telemetry: Dict[str, Any]) -> Optional[int]:
    """Insert one COMPLETED run's history row; id, or None when degraded.

    CALLED BY   cli.py::_run, exactly once, right after a run finishes and
                its report/telemetry have already been printed, AND
                api/server.py::_respond on its "done" branch (P2-08).

                D-121: this docstring said "NEVER called from
                api/server.py — API-driven runs get no history row", which
                was false when it was written and had been for some time:
                `_respond` imports and calls this function, and says so in
                its own CALLS section. The claim was then copied into
                README, OPERATIONS and LEARNING_GUIDE by D-109 in the
                belief it was the correction. §14.3's rule caught its own
                author: a comment asserting a property the code does not
                have is the thing a reviewer trusts instead of checking.

                The asymmetry that IS real: only the CLI records FAILED
                runs. api/server.py never calls record_failed_run, so an
                API run that raises leaves no row at all.
    WRITES      one row into the agent_runs table described by _RUNS_DDL
                above. Read back by scripts/analyze_runs.py (D-92).

    D-103: `recall` is now Optional and the caller passes
    telemetry.get("recall") rather than telemetry.get("recall", 0.0). A run
    that reached this line with no recall in its telemetry recorded a
    literal 0.0 — a number nothing measured, indistinguishable in the
    column from a run that genuinely retrieved nothing. NULL is the honest
    value and the column has always allowed it.

    A row written here carries NO `run_outcome` key, and that absence is
    the contract: see record_failed_run below.
    """
    return _insert_run(dsn, thread_id, query, recall, telemetry)


def record_failed_run(dsn: str, thread_id: str, query: str,
                      failure: Dict[str, Any]) -> Optional[int]:
    """Insert one FAILED run's history row; id, or None when degraded.

    CALLED BY   cli.py::_run's `except Exception` block — the single point
                every failing run passes through exactly once, whether or
                not main() goes on to recognise the exception type.
    WRITES      one row with recall NULL and a telemetry payload of
                {"run_outcome": "failed", "failure": {...}} — `failure` is
                built by the caller (cli.py::_failure_record), because
                knowing that a ProviderChainExhausted has a chain is
                cli.py's business, not this module's.

    D-103. Before this, `record_run` sat on the happy path only, so a run
    lost to provider exhaustion left NO row at all: "how often do we lose a
    run this way" was unanswerable by analyze_runs.py (D-92), the tool
    built to answer exactly that class of question — and a failed run is
    the one you most want in the history.

    WHY THIS DOES NOT BREAK telemetry_node's "never invent a number" rule,
    which is why the D-92 review deferred this item: no telemetry figure is
    written here, invented or otherwise. The row says the run failed and
    why. That is a fact about the process, recorded by the process, not an
    estimate of anything the graph would have measured.

    WHY NO SCHEMA CHANGE: `_RUNS_DDL` already declares `recall REAL` and
    `telemetry JSONB`, both NULLABLE. A migration would have been the
    expensive part of this item and it is not needed.

    WHY `run_outcome` IS ABSENT FROM SUCCESS ROWS: every row written before
    this change lacks it, so "absent means completed" is already true of
    the whole history. Stamping it onto new success rows only would create
    two shapes of "completed" and make the older one look unclassified.
    """
    return _insert_run(dsn, thread_id, query, None,
                       {"run_outcome": "failed", "failure": failure})


def _insert_run(dsn: str, thread_id: str, query: str,
                recall: Optional[float],
                telemetry: Dict[str, Any]) -> Optional[int]:
    """The shared INSERT behind both recorders above (D-103).

    Factored out rather than duplicated so the two paths cannot drift on
    the DDL, the parameterisation, or the degraded-database posture — the
    failure path in particular must not be the one that learns to raise.
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
