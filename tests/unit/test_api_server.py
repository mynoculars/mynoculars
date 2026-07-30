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


class _ScriptedGraph:
    """A graph whose invoke() returns a caller-supplied result (or raises
    a caller-supplied exception), so the endpoints can be driven end to
    end without running anything real."""

    def __init__(self, result=None, raises=None):
        self.result, self.raises = result, raises
        self.invocations = 0

    def invoke(self, *a, **k):
        self.invocations += 1
        if self.raises is not None:
            raise self.raises
        return self.result


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


class _RecordingObserver:
    """Records the trace lifecycle calls the API makes, in order, so a
    test can assert PAIRING rather than just presence -- an unbalanced
    start_trace is the specific bug this file's new tests exist to
    catch, and a call-count assertion would not see it."""

    def __init__(self):
        self.calls = []

    def start_trace(self, thread_id, name, **kwargs):
        self.calls.append(("start", thread_id, name))

    def end_trace(self, thread_id, **kwargs):
        self.calls.append(("end", thread_id))

    def score(self, thread_id, name, value, **kwargs):
        self.calls.append(("score", thread_id, name, value))

    def shutdown(self):
        self.calls.append(("shutdown",))

    # Everything else the facade exposes, as no-ops -- these tests are
    # about the trace lifecycle, not about span/generation/event.
    def __getattr__(self, _name):
        return lambda *a, **k: None


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


# ---------------------------------------------------------------------------
# Langfuse trace lifecycle on the API path (Item 7)
# ---------------------------------------------------------------------------

def _server_with_observer(graph, observer):
    """Import server.py with a scripted graph AND a recording Observer
    swapped in behind the langfuse facade."""
    server = _import_server(_bundle(app=graph))
    return server


def test_research_opens_and_closes_exactly_one_trace():
    """The pairing is the point: start_trace without a matching end_trace
    leaks an attached OTel context onto a REUSED threadpool worker (see
    _traced_request's docstring), which is how one caller's session_id
    ends up stamped on another caller's run."""
    obs = _RecordingObserver()
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {},
                                   "raw_query": "q"})
    server = _server_with_observer(graph, obs)
    with patch.object(server.lf, "get_observer", return_value=obs), \
         patch.object(server, "record_run", return_value=None), \
         patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace), \
         patch.object(server.lf, "score", obs.score):
        client = TestClient(server.app)
        resp = client.post("/research", json={"query": "q"})
    assert resp.status_code == 200
    kinds = [c[0] for c in obs.calls]
    assert kinds.count("start") == 1
    assert kinds.count("end") == 1
    assert kinds.index("start") < kinds.index("end")


def test_research_closes_the_trace_even_when_the_graph_raises():
    """An un-.end()ed span is never exported at all, so a request that
    blew up would otherwise produce NO trace -- exactly the request you
    most want to inspect."""
    obs = _RecordingObserver()
    graph = _ScriptedGraph(raises=RuntimeError("boom"))
    server = _server_with_observer(graph, obs)
    with patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace):
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.post("/research", json={"query": "q"})
    assert resp.status_code == 500
    kinds = [c[0] for c in obs.calls]
    assert kinds == ["start", "end"], f"expected balanced pair, got {kinds}"


def test_interrupted_research_still_closes_its_trace():
    """A HITL pause must NOT leave the propagation context open across
    the gap between /research and /resume -- that gap can be minutes,
    and the worker thread serves other requests in the meantime."""
    obs = _RecordingObserver()

    class _Interrupt:
        value = {"trigger": "E3"}

    graph = _ScriptedGraph(result={"__interrupt__": [_Interrupt()]})
    server = _server_with_observer(graph, obs)
    with patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace):
        client = TestClient(server.app)
        resp = client.post("/research", json={"query": "q"})
    assert resp.json()["status"] == "interrupted"
    kinds = [c[0] for c in obs.calls]
    assert kinds == ["start", "end"]


def test_resume_opens_its_own_trace_on_the_same_thread_id():
    """Two HTTP requests produce two root spans -- but both land on the
    same Langfuse trace, because trace_id is derived deterministically
    from thread_id. That is the version that cannot leak."""
    obs = _RecordingObserver()
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {},
                                   "raw_query": "q"})
    server = _server_with_observer(graph, obs)
    with patch.object(server, "record_run", return_value=None), \
         patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace), \
         patch.object(server.lf, "score", obs.score):
        client = TestClient(server.app)
        client.post("/resume", json={"thread_id": "api-abc",
                                     "action": "approve"})
    starts = [c for c in obs.calls if c[0] == "start"]
    ends = [c for c in obs.calls if c[0] == "end"]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0][1] == "api-abc" and ends[0][1] == "api-abc"
    assert starts[0][2] == "research_resume"


def test_concurrent_requests_never_interleave_their_traces():
    """The failure mode this whole design exists to prevent: FastAPI runs
    these `def` endpoints in a REUSED threadpool, so an unbalanced
    context would bleed one request's session onto the next. Assert every
    start is followed by its OWN end before another start begins, under
    genuine concurrency."""
    import concurrent.futures

    obs = _RecordingObserver()
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {},
                                   "raw_query": "q"})
    server = _server_with_observer(graph, obs)

    with patch.object(server, "record_run", return_value=None), \
         patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace), \
         patch.object(server.lf, "score", obs.score):
        client = TestClient(server.app)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(client.post, "/research",
                                   json={"query": f"q{i}"})
                       for i in range(8)]
            results = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in results)
    thread_ids = {r.json()["thread_id"] for r in results}
    assert len(thread_ids) == 8, "each request must get its own thread_id"

    # Every thread_id must have exactly one start and one end.
    for tid in thread_ids:
        starts = [c for c in obs.calls if c[0] == "start" and c[1] == tid]
        ends = [c for c in obs.calls if c[0] == "end" and c[1] == tid]
        assert len(starts) == 1, f"{tid}: {len(starts)} starts"
        assert len(ends) == 1, f"{tid}: {len(ends)} ends"


def test_scores_are_recorded_only_for_a_finished_run():
    """A still-interrupted response has no telemetry yet -- scoring it
    would write zeros that look like real measurements."""
    obs = _RecordingObserver()

    class _Interrupt:
        value = {"trigger": "E3"}

    graph = _ScriptedGraph(result={"__interrupt__": [_Interrupt()]})
    server = _server_with_observer(graph, obs)
    with patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace), \
         patch.object(server.lf, "score", obs.score):
        client = TestClient(server.app)
        client.post("/research", json={"query": "q"})
    assert not [c for c in obs.calls if c[0] == "score"]


def test_finished_run_records_the_same_scores_the_cli_does():
    obs = _RecordingObserver()
    telemetry = {"recall": 0.75, "critique_passed": True,
                 "evidence_items": 20, "goals": 4,
                 "search_calls": 10, "memory_hits": 5}
    graph = _ScriptedGraph(result={"final_report": "r", "raw_query": "q",
                                   "telemetry": telemetry})
    server = _server_with_observer(graph, obs)
    with patch.object(server, "record_run", return_value=None), \
         patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace), \
         patch.object(server.lf, "score", obs.score):
        client = TestClient(server.app)
        client.post("/research", json={"query": "q"})
    scored = {c[2]: c[3] for c in obs.calls if c[0] == "score"}
    assert scored["recall"] == 0.75
    assert scored["critique_passed"] is True
    assert scored["evidence_per_goal"] == 5.0
    assert scored["memory_hit_rate"] == 0.5
