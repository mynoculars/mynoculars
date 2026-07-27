"""
tests/unit/test_api_server.py — api/server.py.

WHY THIS FILE EXISTS: api/server.py previously had ZERO tests, and shipped
with `_graph, _settings, _durable, _checkpointer = _bundle` — a four-name
tuple-unpack of an AppBundle that had grown a fifth field (mcp_bridge) in
P2-13. That raises ValueError at IMPORT time, so every endpoint, /health,
and the P2-08 record_run parity were unreachable in any build after P2-13.
Nothing caught it because nothing ever imported this module under test.

The tests below are deliberately cheap and structural: import the module
with a fake AppBundle injected, and confirm the wiring holds. They do not
exercise the graph (tests/integration/ already does) — they exist so that
"the API process can start at all" is a thing the suite asserts.
"""

import importlib
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from research_agent.cli import AppBundle  # noqa: E402


class _FakeGraph:
    """Stands in for the compiled LangGraph app; never invoked here."""

    def invoke(self, *a, **k):  # pragma: no cover - not exercised
        raise AssertionError("these tests must not run the graph")


class _FakeSettings:
    llm_mode = "stub"
    recursion_limit = 60
    postgres_dsn = "postgresql://x:x@127.0.0.1:1/x"


class _FakeBridge:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _import_server(bundle):
    """Import api/server.py fresh with `bundle` injected.

    build_app_and_settings runs at that module's IMPORT time, so it must be
    patched before the import and the module must be evicted from
    sys.modules first — otherwise a previously-imported copy is reused and
    the patch has no effect.
    """
    sys.modules.pop("research_agent.api.server", None)
    with patch("research_agent.cli.build_app_and_settings", return_value=bundle):
        return importlib.import_module("research_agent.api.server")


def _bundle(**overrides):
    base = dict(app=_FakeGraph(), settings=_FakeSettings(), durable=True,
                checkpointer=object(), mcp_bridge=None, router=None)
    base.update(overrides)
    return AppBundle(**base)


def test_server_module_imports_with_a_full_appbundle():
    """The regression guard. A tuple-unpack of the wrong arity raises
    ValueError right here, before any endpoint is even reachable."""
    server = _import_server(_bundle())
    assert server.app is not None


def test_server_consumes_the_bundle_by_name_not_by_position():
    """Field ORDER must not matter: every consumer reads named attributes.
    Constructing the bundle purely by keyword and asserting each value
    landed on the right module global proves no positional assumption
    survives."""
    settings = _FakeSettings()
    checkpointer = object()
    bridge = _FakeBridge()
    server = _import_server(_bundle(settings=settings, durable=False,
                                    checkpointer=checkpointer,
                                    mcp_bridge=bridge))
    assert server._settings is settings
    assert server._durable is False
    assert server._checkpointer is checkpointer
    assert server._mcp_bridge is bridge


def test_health_reports_llm_mode_and_durability():
    server = _import_server(_bundle(durable=False))
    with TestClient(server.app) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "llm_mode": "stub", "durable": False}


def test_shutdown_closes_the_mcp_bridge_as_well_as_the_checkpointer():
    """cli.py has always closed both; api/server.py closed only the
    checkpointer, leaving an MCP subprocess and its background thread
    running past shutdown."""
    bridge = _FakeBridge()
    server = _import_server(_bundle(mcp_bridge=bridge))
    with patch("research_agent.api.server.close_checkpointer") as closer:
        with TestClient(server.app):
            pass
    assert bridge.closed is True
    assert closer.call_count == 1


def test_respond_tolerates_a_run_that_never_reached_telemetry():
    """A run that ends without telemetry_node (recursion limit, abandoned
    resume) must not KeyError its way into a 500."""
    server = _import_server(_bundle())
    with patch("research_agent.api.server.record_run", return_value=None):
        out = server._respond("t-1", {"raw_query": "q"})
    assert out["status"] == "done"
    assert out["telemetry"] == {}
    assert out["report"] == ""


def test_respond_returns_the_review_payload_when_interrupted():
    server = _import_server(_bundle())

    class _Interrupt:
        value = {"trigger": "E3", "actions": ["approve", "redirect", "abort"]}

    out = server._respond("t-2", {"__interrupt__": [_Interrupt()]})
    assert out == {"thread_id": "t-2", "status": "interrupted",
                   "review": _Interrupt.value}
