"""
tests/unit/test_storage_postgres.py — storage/postgres.py's
close_checkpointer (P2-08).

Covers: closing the underlying connection when present, a safe no-op for
MemorySaver (which has no .conn at all), and surviving a connection that
itself raises on close (logged, not propagated).
"""

import logging

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.storage.postgres import close_checkpointer


class _FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeCheckpointerWithConn:
    def __init__(self, conn):
        self.conn = conn


def test_close_checkpointer_closes_underlying_connection():
    conn = _FakeConn()
    close_checkpointer(_FakeCheckpointerWithConn(conn))
    assert conn.closed is True


def test_close_checkpointer_is_a_noop_for_memory_saver():
    # MemorySaver has no .conn attribute at all — must not raise.
    close_checkpointer(MemorySaver())


def test_close_checkpointer_survives_a_conn_that_errors_on_close(caplog):
    class _AngryConn:
        def close(self):
            raise RuntimeError("already gone")

    with caplog.at_level(logging.WARNING):
        close_checkpointer(_FakeCheckpointerWithConn(_AngryConn()))
    assert any("checkpointer.close_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# D-149 -- every connection attempt is bounded
#
# psycopg's default is no timeout: it waits for whatever the OS TCP stack
# decides, which on an unreachable host is minutes. Measured on a real
# Windows run, five POST /research calls that reached record_run against
# 127.0.0.1:1 cost 130 SECONDS EACH -- 650.61s of a 662.25s suite, 98% of
# the total, against ~12s for everything else combined. On Linux the same
# connect refuses instantly, which is why it was invisible in review.
#
# It is not only a test problem: record_run runs at the END of every CLI
# run, so an operator whose Postgres went away gets a two-minute hang with
# the answer already on screen.
# ---------------------------------------------------------------------------


def test_the_timeout_matches_the_rest_of_the_codebase():
    """QdrantStore and OpenSearchStore both pass timeout=5, and
    scripts/check_services.py has always used connect_timeout=5. Postgres
    was the one store that bounded nothing."""
    from research_agent.storage.postgres import CONNECT_TIMEOUT_SECONDS

    assert CONNECT_TIMEOUT_SECONDS == 5


def test_record_run_passes_a_connect_timeout(monkeypatch):
    from research_agent.storage import postgres

    seen = {}

    class _Conn:
        def __enter__(self): raise RuntimeError("stop here; the connect is the assertion")
        def __exit__(self, *a): return False

    def _connect(dsn, **kwargs):
        seen.update(kwargs)
        return _Conn()

    monkeypatch.setattr(postgres, "psycopg",
                        type("m", (), {"connect": staticmethod(_connect)}),
                        raising=False)
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "psycopg",
                        type("m", (), {"connect": staticmethod(_connect)}))

    postgres.record_run("postgresql://x:x@127.0.0.1:1/x", "t", "q", None, {})

    assert seen.get("connect_timeout") == postgres.CONNECT_TIMEOUT_SECONDS


def test_an_unreachable_database_still_degrades_rather_than_raising(caplog):
    """The contract this timeout must not break: a database problem skips
    the history row and never touches the report the user already saw."""
    import logging as _logging

    from research_agent.storage.postgres import record_run

    with caplog.at_level(_logging.WARNING):
        result = record_run("postgresql://x:x@192.0.2.1:5432/x", "t", "q",
                            None, {})

    assert result is None
    assert [r for r in caplog.records if "run_history.skipped" in r.message]


def test_the_checkpointer_pool_is_bounded_too(monkeypatch):
    """A pool against an unreachable Postgres blocks its caller
    indefinitely without `timeout`, which would defeat get_checkpointer's
    whole degrade-don't-die contract.

    Two ordering details, both found by this test failing for the wrong
    reason rather than by reading the code:

      - patch the ATTRIBUTE on the real psycopg_pool, not the module in
        sys.modules. langgraph.checkpoint.postgres._internal evaluates
        `ConnectionPool[Connection[DictRow]]` at import time, and a
        stand-in class is not subscriptable;
      - force that langgraph import BEFORE patching. get_checkpointer
        imports it inside the same try block, so on the first call in a
        process the import happens after the patch and raises TypeError --
        and get_checkpointer swallows it, falls back to MemorySaver, and
        the pool line is never reached at all.
    """
    psycopg_pool = pytest.importorskip("psycopg_pool")
    pytest.importorskip("langgraph.checkpoint.postgres")

    from research_agent.storage import postgres

    seen = {}

    class _Pool:
        def __init__(self, dsn, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here; the construction is the assertion")

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", _Pool)

    postgres.get_checkpointer("postgresql://x:x@127.0.0.1:1/x")

    assert seen.get("timeout") == postgres.CONNECT_TIMEOUT_SECONDS
    assert seen.get("kwargs", {}).get("connect_timeout") == \
        postgres.CONNECT_TIMEOUT_SECONDS

