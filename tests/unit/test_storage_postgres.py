"""
tests/unit/test_storage_postgres.py — storage/postgres.py's
close_checkpointer (P2-08).

Covers: closing the underlying connection when present, a safe no-op for
MemorySaver (which has no .conn at all), and surviving a connection that
itself raises on close (logged, not propagated).
"""

import logging

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
