"""
tests/unit/test_tools_mcp_client.py — tools/mcp_client.py (P2-13, D-76).

Covers: make_mcp_tool's Evidence-construction/parsing logic against a fake
bridge, and MCPBridge's real connection/lifecycle behavior over a real
Streamable HTTP connection (unreachable-server error, timeout error
messages, concurrent first-call safety).

D-76: MCPBridge only ever connects to a standalone, already-running MCP
server now -- this process spawns nothing, ever. (Earlier revisions of
this file also covered stdio: a subprocess-env allowlist helper and
several tests against a real spawned subprocess. D-76 removed the stdio
transport from MCPBridge entirely, so that coverage moved or was retired
along with the code it tested -- see DECISIONS.md D-76.)

Unlike every other file in this suite (see conftest.py's module
docstring: "every test runs fully offline"), THREE tests below
(test_mcp_tool_round_trips_through_a_real_streamable_http_server,
test_mcp_bridge_survives_many_concurrent_first_calls,
test_mcp_bridge_timeout_error_is_actually_informative) genuinely spawn a
real server subprocess and talk real MCP protocol over a real loopback
HTTP connection. This is still fully self-contained and offline in the
sense that matters (no EXTERNAL network call, no external service, no
non-deterministic dependency) — the "servers" are
tests/fixtures/mcp_echo_http_server.py and mcp_slow_http_server.py, each
a ~30-line fixture shipped in this repo, launched and torn down entirely
within each test. This is deliberately NOT mocked: an earlier, mock-only
version of this work shipped an MCPBridge.close() that raised
`RuntimeError: Attempted to exit cancel scope in a different task than
it was entered in` under real use — a genuine anyio/asyncio structured-
concurrency constraint that no amount of mocking the SDK's objects would
have caught. Real subprocess, real network connection, real protocol
round trip is what caught it, and is kept here specifically so a future
change to MCPBridge's lifecycle gets the same check.
"""

import pathlib
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# See tests/unit/test_mcp_corpus_server.py for why this is a skip.
pytest.importorskip("mcp")


def _free_port() -> int:
    """Ask the OS for an ephemeral port, then release it immediately --
    good enough for a test fixture server started microseconds later;
    avoids hardcoding a port that a concurrent test run could collide on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout_s: float = 10.0) -> None:
    """Poll until something is listening on 127.0.0.1:port, or raise.

    Spawning the fixture server and immediately trying to talk MCP to it
    is racy -- the process needs a moment to import FastMCP/uvicorn and
    bind its socket. A plain TCP connect poll is the cheapest reliable
    way to know the HTTP listener is actually up, distinct from (and
    prior to) speaking any MCP protocol to it.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"nothing listening on 127.0.0.1:{port} after {timeout_s}s")


# ---------------------------------------------------------------------------
# make_mcp_tool — Evidence-construction/parsing logic, fully faked
# ---------------------------------------------------------------------------


class _FakeContentBlock:
    def __init__(self, text=None):
        self.text = text


class _FakeCallToolResult:
    def __init__(self, content=None, isError=False):
        self.content = content or []
        self.isError = isError


class _FakeBridgeForToolParsing:
    """A fake satisfying exactly what make_mcp_tool's closure calls:
    .call_tool(name, arguments, timeout_seconds=...) -> an object with
    .content / .isError. Never touches a real MCPBridge or real asyncio
    machinery -- this tests ONLY the Evidence-construction/parsing logic
    make_mcp_tool's closure wraps around whatever a bridge returns."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def call_tool(self, name, arguments, timeout_seconds=30.0):
        self.calls.append((name, arguments, timeout_seconds))
        return self._result


def _task(query="q", key="t1", goal_id="g1"):
    from research_agent.state import SearchTask

    return SearchTask(key=key, goal_id=goal_id, query=query, depth=0)


def test_make_mcp_tool_converts_text_content_to_evidence():
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[_FakeContentBlock(text="fact one")])
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search", query_arg_name="query")

    evidence = tool(_task(query="redis vs cassandra", key="t1", goal_id="g1"))

    assert len(evidence) == 1
    assert evidence[0].content == "fact one"
    assert evidence[0].source == "mcp"
    assert evidence[0].task_key == "t1"
    assert evidence[0].goal_id == "g1"
    assert bridge.calls == [("search", {"query": "redis vs cassandra"}, 30.0)]


def test_make_mcp_tool_produces_one_evidence_item_per_text_block():
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[
        _FakeContentBlock(text="fact A"), _FakeContentBlock(text="fact B")])
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search")

    evidence = tool(_task())
    assert [e.content for e in evidence] == ["fact A", "fact B"]


def test_make_mcp_tool_skips_non_text_content_blocks():
    """A content block with no .text (e.g. an image) is skipped, not an
    error -- Evidence in this build is text-only."""
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[
        _FakeContentBlock(text=None), _FakeContentBlock(text="only this one")])
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search")

    evidence = tool(_task())
    assert len(evidence) == 1
    assert evidence[0].content == "only this one"


def test_make_mcp_tool_returns_empty_list_on_tool_reported_error():
    """isError=True is a TOOL-level failure (the server ran fine, the
    tool itself reported nothing useful) -- treated as "no results",
    not raised as an exception."""
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[_FakeContentBlock(text="ignored")], isError=True)
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search")

    assert tool(_task()) == []


def test_make_mcp_tool_uses_the_configured_query_arg_name():
    from research_agent.tools.mcp_client import make_mcp_tool

    bridge = _FakeBridgeForToolParsing(_FakeCallToolResult())
    tool = make_mcp_tool(bridge, "search", query_arg_name="search_text")

    tool(_task(query="hello"))
    assert bridge.calls[0][1] == {"search_text": "hello"}


def test_make_mcp_tool_content_is_capped_at_800_chars():
    """Same slicing cap tools/corpus_search.py's corpus_search uses --
    one enormous content block shouldn't dominate the compile prompt."""
    from research_agent.tools.mcp_client import make_mcp_tool

    long_text = "x" * 2000
    bridge = _FakeBridgeForToolParsing(_FakeCallToolResult(content=[_FakeContentBlock(text=long_text)]))
    tool = make_mcp_tool(bridge, "search")

    evidence = tool(_task())
    assert len(evidence[0].content) == 800


# ---------------------------------------------------------------------------
# MCPBridge — real subprocess spawning and lifecycle
# ---------------------------------------------------------------------------


def test_mcp_bridge_surfaces_a_clear_error_for_an_unreachable_server():
    """Nothing listening at the configured URL must fail fast and
    clearly, not hang -- proven against the REAL connection path, not a
    mock (a mock could never demonstrate a real connection failure).

    BaseException, not Exception: an unreachable server surfaces as
    asyncio.CancelledError (a BaseException subclass since Python 3.8,
    not an Exception subclass) bubbling up from the underlying
    anyio/HTTP machinery -- confirmed empirically, not assumed. MCPBridge
    itself catches BaseException in _serve() for exactly this reason
    (see that method's own except clause); this test matches it."""
    from research_agent.tools.mcp_client import MCPBridge

    port = _free_port()  # freed immediately -- guaranteed nothing is listening
    bridge = MCPBridge(url=f"http://127.0.0.1:{port}/mcp",
                       startup_timeout_seconds=5.0)
    try:
        # B017 is deliberate here, unlike the four others this ruleset
        # found: what an unreachable server raises comes from the MCP SDK's
        # own transport, and pinning a type would make this a test about
        # the SDK's internals rather than about the bridge failing loudly
        # instead of hanging.
        with pytest.raises(BaseException):  # noqa: B017
            bridge.call_tool("search", {"query": "x"}, timeout_seconds=5.0)
    finally:
        bridge.close()  # must not itself raise, even after a failed start


def test_mcp_tool_round_trips_through_a_real_streamable_http_server():
    """The genuine end-to-end proof: a real subprocess, real MCP
    protocol, real loopback HTTP connection, using
    tests/fixtures/mcp_echo_http_server.py (a ~30-line FastMCP server
    shipped in this repo, deterministic, no external dependencies of its
    own). This is what caught the anyio cancel-scope bug in
    MCPBridge.close() that no amount of mocking would have -- see this
    module's docstring for the full story. Also specifically asserts
    that closing this bridge does NOT kill the server process -- the
    entire point of D-76's standalone-only design."""
    from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool

    server_path = str(pathlib.Path(__file__).parent.parent / "fixtures"
                      / "mcp_echo_http_server.py")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, server_path, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port)
        bridge = MCPBridge(url=f"http://127.0.0.1:{port}/mcp",
                           startup_timeout_seconds=10.0)
        tool = make_mcp_tool(bridge, "search", query_arg_name="query")
        try:
            evidence = tool(_task(query="redis vs cassandra", key="t1", goal_id="g1"))
            assert len(evidence) == 1
            assert evidence[0].content == "canned result for: redis vs cassandra"
            assert evidence[0].source == "mcp"

            # A second call on the SAME bridge proves the connection is
            # actually PERSISTENT (reused), not re-established per call --
            # the whole point of the background-loop design over
            # asyncio.run() per call (see MCPBridge's own docstring).
            evidence2 = tool(_task(query="second query", key="t2", goal_id="g1"))
            assert evidence2[0].content == "canned result for: second query"
        finally:
            bridge.close()  # must succeed cleanly -- this is the regression check
        # The server process must still be alive after bridge.close() --
        # closing an HTTP client connection must never stop an
        # independent server it does not own (D-76's entire point).
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_mcp_bridge_survives_many_concurrent_first_calls():
    """Regression test for a REAL bug a live run caught: multiple threads
    calling call_tool() at nearly the same moment, before the bridge has
    finished connecting, used to make every thread EXCEPT the one that
    created the background thread skip the readiness wait entirely and
    crash with AttributeError ('NoneType' object has no attribute
    'call_tool') -- exactly reproducing LangGraph fanning several
    search_worker instances out concurrently for one gather-cycle
    superstep. Uses the REAL fixture server (tests/fixtures/
    mcp_echo_http_server.py) and REAL threads -- a mock could not have
    caught this, since the bug was a genuine race between real OS
    threads."""
    from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool

    server_path = str(pathlib.Path(__file__).parent.parent / "fixtures"
                      / "mcp_echo_http_server.py")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, server_path, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port)
        bridge = MCPBridge(url=f"http://127.0.0.1:{port}/mcp",
                           startup_timeout_seconds=10.0)
        tool = make_mcp_tool(bridge, "search", query_arg_name="query")

        def run_one(i):
            return tool(_task(query=f"concurrent query {i}", key=f"t{i}", goal_id="g1"))

        try:
            # 8 threads all calling the SAME bridge for the first time at
            # once -- exactly the shape that crashed before the fix, at a
            # concurrency level at least as high as this codebase's own
            # MAX_FANOUT default ever produces in one gather cycle.
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(run_one, range(8)))

            for i, evidence in enumerate(results):
                assert len(evidence) == 1
                assert evidence[0].content == f"canned result for: concurrent query {i}"
        finally:
            bridge.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_mcp_bridge_timeout_error_is_actually_informative():
    """Regression test for a fair, direct complaint: a real failure showed
    up in this codebase's own D-16 failure log as bare "reason=
    TimeoutError" with zero further detail -- concurrent.futures.
    TimeoutError's own message is EMPTY (confirmed: str(TimeoutError())
    == ""), so there was genuinely nothing else to show. This uses a REAL
    server (tests/fixtures/mcp_slow_http_server.py, which sleeps 5s) and a
    deliberately short 1s timeout to trigger the real timeout path fast
    and deterministically, then checks the raised exception's message
    actually names the tool, the arguments, and how long it waited --
    not just the exception's class name."""
    from research_agent.tools.mcp_client import MCPBridge

    server_path = str(pathlib.Path(__file__).parent.parent / "fixtures"
                      / "mcp_slow_http_server.py")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, server_path, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port)
        bridge = MCPBridge(url=f"http://127.0.0.1:{port}/mcp",
                           startup_timeout_seconds=10.0)
        try:
            try:
                bridge.call_tool("search", {"query": "redis vs cassandra"}, timeout_seconds=1.0)
                raise AssertionError("expected a TimeoutError")
            except TimeoutError as exc:
                message = str(exc)
                assert message, "the whole point of this fix: the message must NOT be empty"
                assert "search" in message
                assert "redis vs cassandra" in message
                assert "1.0" in message  # the configured timeout, visible in the message
        finally:
            bridge.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# make_web_search_tool (Phase 4 / D-57) -- parsing only, fully faked
# ---------------------------------------------------------------------------


class _FakeWebResult:
    """A CallToolResult stand-in carrying structuredContent, text blocks, or
    both -- so each test can pin down which channel the parser actually
    used."""

    def __init__(self, structuredContent=None, content=None, isError=False):
        self.structuredContent = structuredContent
        self.content = content or []
        self.isError = isError


def _payload(rank, score, url=None, title="T", snippet="s", domain=None):
    return {"title": f"{title}{rank}", "url": url or f"https://s{rank}.com/p",
            "snippet": f"{snippet}{rank}", "rank": rank, "engine": "ddg_text",
            "domain": domain or f"s{rank}.com", "score": score}


def _web_tool(result, **kwargs):
    from research_agent.tools.mcp_client import make_web_search_tool

    bridge = _FakeBridgeForToolParsing(result)
    return make_web_search_tool(bridge, "web_search", **kwargs), bridge


def test_web_tool_reads_structured_content_and_tags_source_web():
    """source="web", NOT "mcp", even though this arrived over MCP. "mcp" is
    tested for set-membership in progress_checker_node and telemetry_node as
    a proxy for "a real DOCUMENT backed this"; tagging snippets "mcp" would
    make every one of them inflate grounded_score and corpus_recall."""
    result = _FakeWebResult(structuredContent={
        "result": [_payload(1, 0.75), _payload(2, 0.60)]})
    tool, bridge = _web_tool(result)

    evidence = tool(_task(query="redis vs memcached", key="t1", goal_id="g1"))

    assert [e.source for e in evidence] == ["web", "web"]
    assert [e.score for e in evidence] == [0.75, 0.60]
    assert all(e.task_key == "t1" and e.goal_id == "g1" for e in evidence)
    assert bridge.calls == [("web_search", {"query": "redis vs memcached"}, 45.0)]


def test_web_tool_carries_url_and_domain_onto_evidence():
    result = _FakeWebResult(structuredContent={"result": [
        _payload(1, 0.75, url="https://arxiv.org/abs/1", domain="arxiv.org")]})
    tool, _ = _web_tool(result)

    e = tool(_task())[0]
    assert e.url == "https://arxiv.org/abs/1"
    assert e.domain == "arxiv.org"


def test_web_tool_marks_evidence_volatile():
    """A live page today is a stale answer next month with nothing in the
    text saying so. This also feeds D-51's hedging pass."""
    from research_agent.state import Volatility

    result = _FakeWebResult(structuredContent={"result": [_payload(1, 0.75)]})
    tool, _ = _web_tool(result)
    assert tool(_task())[0].volatility is Volatility.VOLATILE


def test_web_tool_joins_title_and_snippet():
    """Each alone loses something: the snippet is the substance, the title is
    often the only place the subject is actually named."""
    result = _FakeWebResult(structuredContent={"result": [{
        "title": "PLA modernization", "url": "https://a.com/1",
        "snippet": "The report finds...", "rank": 1, "engine": "e",
        "domain": "a.com", "score": 0.75}]})
    tool, _ = _web_tool(result)
    assert tool(_task())[0].content == "PLA modernization — The report finds..."


def test_web_tool_drops_an_item_with_no_usable_score(caplog):
    """No default is safe. Too high silently defeats the D-17 coverage gate
    (the hardcoded 1.0 this very module shipped once and had to fix); too low
    makes a genuinely retrieved result unable to cover a goal while still
    consuming a compile-prompt slot. Dropping is the smaller and, crucially,
    the VISIBLE failure -- it is counted and logged."""
    import logging

    result = _FakeWebResult(structuredContent={"result": [
        {"title": "no score", "url": "https://a.com/1", "snippet": "s",
         "rank": 1, "engine": "e", "domain": "a.com"},
        _payload(2, 0.70)]})
    tool, _ = _web_tool(result)

    with caplog.at_level(logging.WARNING):
        evidence = tool(_task())

    assert len(evidence) == 1 and evidence[0].score == 0.70
    assert [r for r in caplog.records
            if "web_search.dropped_unscored_items" in r.message]


def test_web_tool_clamps_a_score_that_overshoots_one():
    """Evidence.score is bounded [0,1] by its own Field constraint; a server
    returning 1.0000001 through float round-tripping must not raise a
    ValidationError inside a worker."""
    result = _FakeWebResult(structuredContent={
        "result": [_payload(1, 1.0000001)]})
    tool, _ = _web_tool(result)
    assert tool(_task())[0].score == 1.0


def test_web_tool_skips_an_item_with_neither_title_nor_snippet():
    result = _FakeWebResult(structuredContent={"result": [
        {"title": "", "url": "https://a.com/1", "snippet": "", "rank": 1,
         "engine": "e", "domain": "a.com", "score": 0.75},
        _payload(2, 0.70)]})
    tool, _ = _web_tool(result)
    assert len(tool(_task())) == 1


def test_web_tool_falls_back_to_json_text_blocks():
    """Reached when an SDK version stops populating structuredContent, or a
    differently-built server returns text only."""
    import json

    result = _FakeWebResult(content=[
        _FakeContentBlock(text=json.dumps(_payload(1, 0.75))),
        _FakeContentBlock(text=json.dumps(_payload(2, 0.60)))])
    tool, _ = _web_tool(result)

    evidence = tool(_task())
    assert [e.score for e in evidence] == [0.75, 0.60]
    assert evidence[0].url == "https://s1.com/p"


def test_web_tool_prefers_structured_content_over_text_blocks():
    """structuredContent is the only channel where score survives as a
    NUMBER rather than as text to be re-parsed."""
    import json

    result = _FakeWebResult(
        structuredContent={"result": [_payload(1, 0.75, title="STRUCTURED")]},
        content=[_FakeContentBlock(text=json.dumps(_payload(9, 0.61, title="TEXT")))])
    tool, _ = _web_tool(result)

    evidence = tool(_task())
    assert len(evidence) == 1
    assert evidence[0].content.startswith("STRUCTURED")


def test_web_tool_tolerates_a_non_json_text_block():
    """A server that also emits a human-readable preamble block must not
    break the whole call."""
    import json

    result = _FakeWebResult(content=[
        _FakeContentBlock(text="Found 1 result:"),
        _FakeContentBlock(text=json.dumps(_payload(1, 0.75)))])
    tool, _ = _web_tool(result)
    assert len(tool(_task())) == 1


def test_web_tool_returns_empty_on_a_tool_reported_error():
    """A TOOL-level error is data arriving cleanly to say "this did not
    work", not a protocol failure -- so the ladder escalates rather than the
    task failing under D-16."""
    tool, _ = _web_tool(_FakeWebResult(isError=True))
    assert tool(_task()) == []


def test_web_tool_returns_empty_on_an_unrecognized_shape():
    """A server sending neither channel yields [] -- which reads as "found
    nothing" and lets the ladder escalate, rather than raising and burning
    the task over a shape problem."""
    tool, _ = _web_tool(_FakeWebResult(structuredContent={"unexpected": 1}))
    assert tool(_task()) == []


def test_web_tool_caps_content_at_800_chars():
    """Same cap corpus_search.py and make_mcp_tool use, so no one tier can
    crowd the compile prompt."""
    result = _FakeWebResult(structuredContent={"result": [{
        "title": "T", "url": "https://a.com/1", "snippet": "x" * 5000,
        "rank": 1, "engine": "e", "domain": "a.com", "score": 0.75}]})
    tool, _ = _web_tool(result)
    assert len(tool(_task())[0].content) == 800


def test_web_tool_uses_the_configured_query_arg_name_and_timeout():
    result = _FakeWebResult(structuredContent={"result": []})
    tool, bridge = _web_tool(result, query_arg_name="q",
                            call_timeout_seconds=12.5)
    tool(_task(query="hello"))
    assert bridge.calls == [("web_search", {"q": "hello"}, 12.5)]


def test_make_mcp_tool_is_unchanged_by_phase_4():
    """Regression lock. make_web_search_tool is a SEPARATE factory precisely
    so the proven Phase 1-3 corpus path cannot regress from this work. If
    someone later merges the two, this is where it shows up."""
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[_FakeContentBlock(text="fact one")])
    bridge = _FakeBridgeForToolParsing(result)
    evidence = make_mcp_tool(bridge, "search")(_task())

    assert len(evidence) == 1
    assert evidence[0].source == "mcp"
    assert evidence[0].url is None and evidence[0].domain is None


# ---------------------------------------------------------------------------
# MCPBridge construction validation
# ---------------------------------------------------------------------------


def test_mcp_bridge_rejects_construction_with_no_url():
    """Validated eagerly at construction, not deferred to the first
    call_tool() -- a config with no URL should fail loudly at startup,
    not three tool calls into a run (D-76: url is now MCPBridge's only
    required argument)."""
    from research_agent.tools.mcp_client import MCPBridge

    with pytest.raises(ValueError, match="requires a url"):
        MCPBridge(url="")
