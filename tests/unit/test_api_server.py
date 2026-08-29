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

import asyncio
import importlib
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from research_agent.assembly import AppBundle  # noqa: E402


class _FakeGraph:
    """Stands in for the compiled LangGraph app; never invoked here."""

    def invoke(self, *a, **k):  # pragma: no cover - not exercised
        raise AssertionError("these tests must not run the graph")


class _StateSnapshot:
    """Minimal stand-in for LangGraph's StateSnapshot -- only `.values` is
    read by assembly.reject_if_thread_in_use (M-2)."""

    def __init__(self, values=None):
        self.values = values or {}


class _ScriptedGraph:
    """A graph whose invoke() returns a caller-supplied result (or raises
    a caller-supplied exception), so the endpoints can be driven end to
    end without running anything real.

    get_state() defaults to an empty snapshot (no prior raw_query) so the
    M-2 thread-reuse guard in api/server.py::research passes through
    unless a test explicitly sets `prior_state` to simulate a thread
    already in use.
    """

    def __init__(self, result=None, raises=None, prior_state=None):
        self.result, self.raises = result, raises
        self.prior_state = prior_state or {}
        self.invocations = 0

    def invoke(self, *a, **k):
        self.invocations += 1
        if self.raises is not None:
            raise self.raises
        return self.result

    def get_state(self, *a, **k):
        return _StateSnapshot(self.prior_state)


class _FakeSettings:
    llm_mode = "stub"
    recursion_limit = 60
    postgres_dsn = "postgresql://x:x@127.0.0.1:1/x"
    # D-133: "" is the shipped default -- every test above this line
    # exercises the UNGATED posture, which is the one that must stay
    # byte-identical.
    api_key = ""


class _FakeBridge:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _import_server(bundle, monkeypatch):
    """Import api/server.py fresh with `bundle` injected, then drive its
    startup so every test sees a fully-built module immediately.

    D-78: build_app_and_settings() now runs LAZILY, inside _lifespan's
    startup phase -- not at import time, so patching it only around the
    import call (this function's previous shape, a `with patch(...):`
    that exited before returning) no longer has any effect on the
    module's globals; nothing calls the patched function until something
    later triggers lifespan. monkeypatch.setattr, not a context-manager
    patch, is what keeps the fake bundle active for the rest of the
    test -- it reverts automatically at teardown, same as every other
    monkeypatch fixture use in this suite.

    Also drives _lifespan's startup half here (via a throwaway event
    loop), rather than requiring every test to open a `with
    TestClient(...)` first: several tests below inspect module globals
    or call server._respond()/server._payload... directly, with no HTTP
    request at all, and need _settings/_checkpointer/etc. populated
    regardless. Tests that DO go on to use TestClient trigger a SECOND,
    independent lifespan pass of their own -- harmless, since it re-reads
    the same monkeypatched bundle and simply re-sets the same globals.
    """
    sys.modules.pop("research_agent.api.server", None)
    server = importlib.import_module("research_agent.api.server")
    monkeypatch.setattr(server, "build_app_and_settings", lambda: bundle)

    async def _advance():
        cm = server._lifespan(server.app)
        await cm.__aenter__()

    asyncio.run(_advance())
    return server


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


def test_server_module_imports_with_a_full_appbundle(monkeypatch):
    """The regression guard. A tuple-unpack of the wrong arity raises
    ValueError right here, before any endpoint is even reachable."""
    server = _import_server(_bundle(), monkeypatch)
    assert server.app is not None


def test_server_consumes_the_bundle_by_name_not_by_position(monkeypatch):
    """Field ORDER must not matter: every consumer reads named attributes.
    Constructing the bundle purely by keyword and asserting each value
    landed on the right module global proves no positional assumption
    survives."""
    settings = _FakeSettings()
    checkpointer = object()
    bridge = _FakeBridge()
    server = _import_server(_bundle(settings=settings, durable=False,
                                    checkpointer=checkpointer,
                                    mcp_bridge=bridge), monkeypatch)
    assert server._settings is settings
    assert server._durable is False
    assert server._checkpointer is checkpointer
    assert server._mcp_bridge is bridge


def test_health_reports_llm_mode_and_durability(monkeypatch):
    server = _import_server(_bundle(durable=False), monkeypatch)
    with TestClient(server.app) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "llm_mode": "stub", "durable": False}


# ---------------------------------------------------------------------------
# Guardrail G5 (P205 Phase 2): input length validation on ResearchRequest
# ---------------------------------------------------------------------------


def test_research_rejects_an_empty_query(monkeypatch):
    """min_length=1 -- an empty string previously reached classify_node
    and produced a meaningless LLM call; now it never leaves the API
    layer. Asserted as a 422 (FastAPI's own validation-error status),
    not a graph invocation -- the fake graph in this test file raises
    if invoked at all, so a passing test here already proves the graph
    was never reached."""
    server = _import_server(_bundle(), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": ""})
    assert resp.status_code == 422


def test_research_rejects_a_query_over_the_length_cap(monkeypatch):
    server = _import_server(_bundle(), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "x" * 2001})
    assert resp.status_code == 422


def test_research_accepts_a_query_at_the_length_cap(monkeypatch):
    """Boundary check: exactly max_length must still be accepted -- the
    cap guards against UNBOUNDED input, not against ordinary long
    queries."""
    server = _import_server(_bundle(app=_ScriptedGraph(result={
        "final_report": "ok", "telemetry": {}})), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "x" * 2000})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# M-2: the API shares cli.py's D-20 thread-reuse guard
# ---------------------------------------------------------------------------


def test_research_rejects_a_thread_id_already_in_use(monkeypatch):
    """A caller-supplied thread_id whose checkpoint already holds a
    raw_query must be refused (409), not silently blended (D-20) -- see
    assembly.reject_if_thread_in_use. Previously api/server.py had no
    get_state() call anywhere, so an HTTP client reusing a thread_id hit
    this exact defect the CLI already guards against."""
    graph = _ScriptedGraph(prior_state={"raw_query": "earlier query"})
    server = _import_server(_bundle(app=graph), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "new query",
                                               "thread_id": "reused-thread"})
    assert resp.status_code == 409
    assert "earlier query" in resp.json()["detail"]
    assert graph.invocations == 0  # refused before the graph ever ran


def test_research_with_a_fresh_thread_id_is_not_rejected(monkeypatch):
    """The common case -- no thread_id, or one that has never been used --
    must still reach the graph exactly as before M-2."""
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {}})
    server = _import_server(_bundle(app=graph), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "q"})
    assert resp.status_code == 200
    assert graph.invocations == 1


def test_shutdown_closes_the_mcp_bridge_as_well_as_the_checkpointer(monkeypatch):
    """cli.py has always closed both; api/server.py closed only the
    checkpointer, leaving an MCP subprocess and its background thread
    running past shutdown."""
    bridge = _FakeBridge()
    server = _import_server(_bundle(mcp_bridge=bridge), monkeypatch)
    with patch("research_agent.api.server.close_checkpointer") as closer:
        with TestClient(server.app):
            pass
    assert bridge.closed is True
    assert closer.call_count == 1


def test_respond_tolerates_a_run_that_never_reached_telemetry(monkeypatch):
    """A run that ends without telemetry_node (recursion limit, abandoned
    resume) must not KeyError its way into a 500."""
    server = _import_server(_bundle(), monkeypatch)
    with patch("research_agent.api.server.record_run", return_value=None):
        out = server._respond("t-1", {"raw_query": "q"})
    assert out["status"] == "done"
    assert out["telemetry"] == {}
    assert out["report"] == ""


def test_respond_returns_the_review_payload_when_interrupted(monkeypatch):
    server = _import_server(_bundle(), monkeypatch)

    class _Interrupt:
        value = {"trigger": "E3", "actions": ["approve", "redirect", "abort"]}

    out = server._respond("t-2", {"__interrupt__": [_Interrupt()]})
    assert out == {"thread_id": "t-2", "status": "interrupted",
                   "review": _Interrupt.value}


# ---------------------------------------------------------------------------
# Langfuse trace lifecycle on the API path (Item 7)
# ---------------------------------------------------------------------------

def _server_with_observer(graph, observer, monkeypatch):
    """Import server.py with a scripted graph AND a recording Observer
    swapped in behind the langfuse facade."""
    server = _import_server(_bundle(app=graph), monkeypatch)
    return server


def test_research_opens_and_closes_exactly_one_trace(monkeypatch):
    """The pairing is the point: start_trace without a matching end_trace
    leaks an attached OTel context onto a REUSED threadpool worker (see
    _traced_request's docstring), which is how one caller's session_id
    ends up stamped on another caller's run."""
    obs = _RecordingObserver()
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {},
                                   "raw_query": "q"})
    server = _server_with_observer(graph, obs, monkeypatch)
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


def test_research_closes_the_trace_even_when_the_graph_raises(monkeypatch):
    """An un-.end()ed span is never exported at all, so a request that
    blew up would otherwise produce NO trace -- exactly the request you
    most want to inspect."""
    obs = _RecordingObserver()
    graph = _ScriptedGraph(raises=RuntimeError("boom"))
    server = _server_with_observer(graph, obs, monkeypatch)
    with patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace):
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.post("/research", json={"query": "q"})
    assert resp.status_code == 500
    kinds = [c[0] for c in obs.calls]
    assert kinds == ["start", "end"], f"expected balanced pair, got {kinds}"


def test_interrupted_research_still_closes_its_trace(monkeypatch):
    """A HITL pause must NOT leave the propagation context open across
    the gap between /research and /resume -- that gap can be minutes,
    and the worker thread serves other requests in the meantime."""
    obs = _RecordingObserver()

    class _Interrupt:
        value = {"trigger": "E3"}

    graph = _ScriptedGraph(result={"__interrupt__": [_Interrupt()]})
    server = _server_with_observer(graph, obs, monkeypatch)
    with patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace):
        client = TestClient(server.app)
        resp = client.post("/research", json={"query": "q"})
    assert resp.json()["status"] == "interrupted"
    kinds = [c[0] for c in obs.calls]
    assert kinds == ["start", "end"]


def test_resume_opens_its_own_trace_on_the_same_thread_id(monkeypatch):
    """Two HTTP requests produce two root spans -- but both land on the
    same Langfuse trace, because trace_id is derived deterministically
    from thread_id. That is the version that cannot leak."""
    obs = _RecordingObserver()
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {},
                                   "raw_query": "q"})
    server = _server_with_observer(graph, obs, monkeypatch)
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


def test_concurrent_requests_never_interleave_their_traces(monkeypatch):
    """The failure mode this whole design exists to prevent: FastAPI runs
    these `def` endpoints in a REUSED threadpool, so an unbalanced
    context would bleed one request's session onto the next. Assert every
    start is followed by its OWN end before another start begins, under
    genuine concurrency."""
    import concurrent.futures

    obs = _RecordingObserver()
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {},
                                   "raw_query": "q"})
    server = _server_with_observer(graph, obs, monkeypatch)

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


def test_scores_are_recorded_only_for_a_finished_run(monkeypatch):
    """A still-interrupted response has no telemetry yet -- scoring it
    would write zeros that look like real measurements."""
    obs = _RecordingObserver()

    class _Interrupt:
        value = {"trigger": "E3"}

    graph = _ScriptedGraph(result={"__interrupt__": [_Interrupt()]})
    server = _server_with_observer(graph, obs, monkeypatch)
    with patch.object(server.lf, "start_trace", obs.start_trace), \
         patch.object(server.lf, "end_trace", obs.end_trace), \
         patch.object(server.lf, "score", obs.score):
        client = TestClient(server.app)
        client.post("/research", json={"query": "q"})
    assert not [c for c in obs.calls if c[0] == "score"]


def test_finished_run_records_the_same_scores_the_cli_does(monkeypatch):
    obs = _RecordingObserver()
    telemetry = {"recall": 0.75, "critique_passed": True,
                 "evidence_items": 20, "goals": 4,
                 "search_calls": 10, "memory_hits": 5}
    graph = _ScriptedGraph(result={"final_report": "r", "raw_query": "q",
                                   "telemetry": telemetry})
    server = _server_with_observer(graph, obs, monkeypatch)
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


# ---------------------------------------------------------------------------
# D-78 -- a bad config must degrade the server, never take the whole
# process down before it can even bind a port
# ---------------------------------------------------------------------------


def _import_server_with_failing_build(exc, monkeypatch):
    """Import api/server.py fresh, but with build_app_and_settings()
    RAISING instead of returning a bundle -- and drive startup, same as
    _import_server above, so _build_error is populated by the time this
    returns (never left to a caller's own TestClient entry, since some
    tests below check module globals directly)."""
    sys.modules.pop("research_agent.api.server", None)
    server = importlib.import_module("research_agent.api.server")

    def _raise():
        raise exc

    monkeypatch.setattr(server, "build_app_and_settings", _raise)

    async def _advance():
        cm = server._lifespan(server.app)
        await cm.__aenter__()

    asyncio.run(_advance())
    return server


def test_a_failed_build_does_not_prevent_the_module_from_importing(monkeypatch):
    """The regression this whole rework exists to fix: a bad MCP config
    (or any other build_app_and_settings failure) used to raise at
    IMPORT time, which took the entire uvicorn worker down before it
    could bind its port -- confirmed live, not hypothetical (see
    DECISIONS.md D-78's own account). Importing must always succeed."""
    server = _import_server_with_failing_build(
        ValueError("MCPBridge requires a url"), monkeypatch)
    assert server.app is not None


def test_health_reports_the_failure_instead_of_being_unreachable(monkeypatch):
    """/health must stay a normal 200 even when the build failed --
    reporting failure IN the body, not by refusing to answer at all."""
    server = _import_server_with_failing_build(
        ValueError("MCPBridge requires a url"), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert "MCPBridge requires a url" in body["detail"]


def test_research_returns_503_after_a_failed_build(monkeypatch):
    """A caller hitting /research after a bad startup config gets one
    clear, actionable error -- not an AttributeError on None.invoke(...)
    (a 500 pointing at the wrong problem entirely)."""
    server = _import_server_with_failing_build(
        ValueError("MCPBridge requires a url"), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "q"})
    assert resp.status_code == 503
    assert "MCPBridge requires a url" in resp.json()["detail"]


def test_resume_returns_503_after_a_failed_build(monkeypatch):
    server = _import_server_with_failing_build(
        ValueError("MCPBridge requires a url"), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.post("/resume", json={"thread_id": "t-1",
                                             "action": "approve"})
    assert resp.status_code == 503


def test_shutdown_after_a_failed_build_does_not_raise(monkeypatch):
    """close_checkpointer/bridge.close() must never be called against
    None -- a failed build left every one of those as None, and shutdown
    must tolerate that rather than raising ITS OWN exception on top of
    the original build failure."""
    server = _import_server_with_failing_build(
        ValueError("boom"), monkeypatch)
    with TestClient(server.app):
        pass  # entry already ran startup (failed); exit runs shutdown
    # No exception escaping the `with` block is the assertion.


def test_a_successful_build_still_reports_ok(monkeypatch):
    """The success path is unaffected by any of the above -- D-78 only
    changes WHEN the build happens and how a FAILURE is reported."""
    server = _import_server(_bundle(durable=True), monkeypatch)
    with TestClient(server.app) as client:
        resp = client.get("/health")
    assert resp.json() == {"status": "ok", "llm_mode": "stub", "durable": True}


# ---------------------------------------------------------------------------
# D-94: GET /state/{thread_id}
# ---------------------------------------------------------------------------


class _StateGraph:
    """A graph whose get_state returns a canned snapshot."""

    def __init__(self, values, nxt=()):
        self._values, self._next = values, nxt

    def get_state(self, config):
        class _Snap:
            values = self._values
            next = self._next
        return _Snap()


def _state_client(monkeypatch, values, nxt=()):
    import research_agent.api.server as srv
    from fastapi.testclient import TestClient

    monkeypatch.setattr(srv, "_graph", _StateGraph(values, nxt))
    monkeypatch.setattr(srv, "_settings", _FakeSettings())
    monkeypatch.setattr(srv, "_build_error", None)
    return TestClient(srv.app)


def test_state_returns_progress_for_a_live_thread(monkeypatch):
    from research_agent.state import Evidence, Goal

    client = _state_client(monkeypatch, {
        "raw_query": "Compare Armies of China and India",
        "goals": [Goal(goal_id="g1", description="PLA size", covered=True)],
        "evidence": [Evidence(task_key="t", goal_id="g1", source="web",
                              content="x", score=0.7)],
        "iteration_depth": 2, "recall_score": 1.0, "grounded_score": 0.0,
        "final_report": "# R\n",
    })

    body = client.get("/state/demo").json()

    assert body["thread_id"] == "demo"
    assert body["status"] == "idle"
    assert body["iteration_depth"] == 2
    assert body["goals"][0]["goal_id"] == "g1"
    assert body["evidence_items"] == 1
    assert body["evidence_by_source"] == {"web": 1}


def test_state_never_returns_evidence_content(monkeypatch):
    """Load-bearing, not tidiness. Evidence is unbounded verbatim corpus
    and third-party web text, and this interface has no auth -- returning
    it would make the endpoint a full-text export of the operator's
    ingested corpus. Counts answer the question; content would be an
    exfiltration route."""
    from research_agent.state import Evidence

    secret = "PROPRIETARY-CORPUS-SENTENCE"
    client = _state_client(monkeypatch, {
        "raw_query": "q",
        "evidence": [Evidence(task_key="t", goal_id="g1", source="corpus",
                              content=secret, score=0.9)],
    })

    assert secret not in client.get("/state/demo").text


def test_state_reports_a_paused_run_as_interrupted(monkeypatch):
    client = _state_client(monkeypatch, {"raw_query": "q"},
                           nxt=("human_escalation",))

    body = client.get("/state/demo").json()

    assert body["status"] == "interrupted"
    assert body["next"] == ["human_escalation"]


def test_state_404s_for_a_thread_that_holds_no_run(monkeypatch):
    client = _state_client(monkeypatch, {})

    resp = client.get("/state/never-used")

    assert resp.status_code == 404
    assert "holds no run" in resp.json()["detail"]


def test_state_503s_when_the_bundle_failed_to_build(monkeypatch):
    """D-78 parity with /research and /resume."""
    import research_agent.api.server as srv
    from fastapi.testclient import TestClient

    monkeypatch.setattr(srv, "_build_error", "ValueError: bad config")
    assert TestClient(srv.app).get("/state/demo").status_code == 503


# ---------------------------------------------------------------------------
# D-133 (P6-5) -- the optional API key
# ---------------------------------------------------------------------------


class _KeyedSettings(_FakeSettings):
    api_key = "s3cret-key"


def _keyed_bundle(**overrides):
    return _bundle(settings=_KeyedSettings(), **overrides)


def test_with_no_key_configured_every_endpoint_stays_open(monkeypatch):
    """THE property that makes this safe to ship: the default is
    unchanged, and it is the posture this repo has always documented."""
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {}})
    server = _import_server(_bundle(app=graph), monkeypatch)

    with TestClient(server.app) as client:
        assert client.post("/research", json={"query": "q"}).status_code == 200
    assert graph.invocations == 1


def test_a_configured_key_rejects_a_caller_that_sends_none(monkeypatch):
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {}})
    server = _import_server(_keyed_bundle(app=graph), monkeypatch)

    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "q"})

    assert resp.status_code == 401
    assert "X-API-Key" in resp.json()["detail"]
    assert graph.invocations == 0, "the request must not reach the graph"


def test_a_wrong_key_is_rejected(monkeypatch):
    graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {}})
    server = _import_server(_keyed_bundle(app=graph), monkeypatch)

    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "q"},
                           headers={"X-API-Key": "not-the-key"})

    assert resp.status_code == 401
    assert graph.invocations == 0


def test_the_right_key_is_accepted_in_either_header(monkeypatch):
    """Two accepted shapes, one secret -- refusing the Bearer form
    produces a 401 that looks like a wrong key rather than a wrong
    header."""
    for headers in ({"X-API-Key": "s3cret-key"},
                    {"Authorization": "Bearer s3cret-key"}):
        graph = _ScriptedGraph(result={"final_report": "r", "telemetry": {}})
        server = _import_server(_keyed_bundle(app=graph), monkeypatch)
        with TestClient(server.app) as client:
            resp = client.post("/research", json={"query": "q"},
                               headers=headers)
        assert resp.status_code == 200, headers
        assert graph.invocations == 1


def test_a_malformed_authorization_header_is_not_parsed_further(monkeypatch):
    """A Basic credential is not a bearer token and must not be treated
    as one -- it yields "" and is rejected by the comparison."""
    server = _import_server(_keyed_bundle(), monkeypatch)

    with TestClient(server.app) as client:
        resp = client.post("/research", json={"query": "q"},
                           headers={"Authorization": "Basic czNjcmV0"})

    assert resp.status_code == 401


def test_resume_and_state_are_guarded_too(monkeypatch):
    """All three endpoints that touch a run, not just the one that
    starts it -- /state/{thread_id} is a live feed of what a caller is
    researching (its own docstring says so)."""
    # _ScriptedGraph, not _FakeGraph: the authenticated call below is
    # meant to REACH the handler, which reads get_state().
    server = _import_server(_keyed_bundle(app=_ScriptedGraph()), monkeypatch)

    with TestClient(server.app) as client:
        assert client.post("/resume", json={"thread_id": "t",
                                            "action": "approve"}).status_code == 401
        assert client.get("/state/t").status_code == 401
        # 404: authenticated, reached the handler, and that thread holds
        # no run -- which is the endpoint working, not the guard.
        assert client.get("/state/t",
                          headers={"X-API-Key": "s3cret-key"}).status_code == 404


def test_health_stays_open_when_a_key_is_configured(monkeypatch):
    """A liveness probe that needs credentials is a liveness probe that
    fails for the wrong reason."""
    server = _import_server(_keyed_bundle(durable=False), monkeypatch)

    with TestClient(server.app) as client:
        body = client.get("/health").json()

    assert body == {"status": "ok", "llm_mode": "stub", "durable": False}


def test_a_failed_build_withholds_its_detail_from_a_stranger(monkeypatch):
    """The detail is f"{type}: {exc}", and the exceptions that reach it
    name MCP URLs, DSN fragments and file paths."""
    server = _import_server(_keyed_bundle(), monkeypatch)
    monkeypatch.setattr(server, "_build_error",
                        "ValueError: MCPBridge requires a url http://internal:8765/mcp")

    with TestClient(server.app) as client:
        anonymous = client.get("/health").json()
        authenticated = client.get(
            "/health", headers={"X-API-Key": "s3cret-key"}).json()

    assert anonymous == {"status": "error"}
    assert "internal:8765" in authenticated["detail"]


def test_a_failed_build_still_reports_its_detail_when_no_key_is_set(monkeypatch):
    """D-78's diagnosability is unchanged for a deployment that chose the
    open posture -- gating it there would take away something this
    project deliberately built, in exchange for nothing."""
    server = _import_server(_bundle(), monkeypatch)
    monkeypatch.setattr(server, "_build_error", "ValueError: boom")

    with TestClient(server.app) as client:
        assert client.get("/health").json() == {"status": "error",
                                                "detail": "ValueError: boom"}


def test_auth_is_checked_before_the_failed_build_503(monkeypatch):
    """An unauthenticated caller must not learn whether this deployment
    built, what its configuration is, or that it is degraded at all."""
    server = _import_server(_keyed_bundle(), monkeypatch)
    monkeypatch.setattr(server, "_build_error", "ValueError: boom")

    with TestClient(server.app) as client:
        assert client.post("/research", json={"query": "q"}).status_code == 401
        assert client.post("/research", json={"query": "q"},
                           headers={"X-API-Key": "s3cret-key"}).status_code == 503


def test_startup_says_which_posture_the_process_started_in(monkeypatch, caplog):
    """Empty is not silent. Logged at API startup and never in
    get_settings(), because a CLI run has no HTTP surface to protect."""
    import logging

    with caplog.at_level(logging.WARNING):
        _import_server(_bundle(), monkeypatch)
    assert [r for r in caplog.records if "api.unauthenticated" in r.message]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        _import_server(_keyed_bundle(), monkeypatch)
    assert [r for r in caplog.records if "api.authenticated" in r.message]
    assert not [r for r in caplog.records if "api.unauthenticated" in r.message]


def test_the_key_comparison_is_constant_time(monkeypatch):
    """hmac.compare_digest, never `==`: a shared secret sent on every
    call is exactly the shape a timing oracle is built from. Asserted
    structurally -- timing cannot be tested reliably, but the call can."""
    import inspect

    server = _import_server(_keyed_bundle(), monkeypatch)
    source = inspect.getsource(server._key_is_valid)

    assert "compare_digest" in source
    assert server._key_is_valid("s3cret-key")
    assert not server._key_is_valid("s3cret-ke")     # prefix
    assert not server._key_is_valid("s3cret-keys")   # extension
    assert not server._key_is_valid("")
