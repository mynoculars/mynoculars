"""
tests/unit/test_tools_mcp_client.py — tools/mcp_client.py (P2-13).

Covers: config.py's split_csv helper, the subprocess-env allowlist
(_build_subprocess_env never forwards os.environ wholesale — D-30),
make_mcp_tool's Evidence-construction/parsing logic against a fake
bridge, and MCPBridge's real subprocess-spawning/lifecycle behavior
(bad command, timeout error messages, concurrent first-call safety).

Unlike every other file in this suite (see conftest.py's module
docstring: "every test runs fully offline"), THREE tests below
(test_mcp_tool_round_trips_through_a_real_stdio_server,
test_mcp_bridge_survives_many_concurrent_first_calls,
test_mcp_bridge_timeout_error_is_actually_informative) genuinely spawn a
real subprocess and talk real MCP protocol over real stdio pipes. This
is still fully self-contained and offline in the sense that matters (no
network call, no external service, no non-deterministic dependency) —
the "servers" are tests/fixtures/mcp_echo_server.py and
mcp_slow_server.py, ~20-line fixtures shipped in this repo, launched and
torn down entirely within each test. This is deliberately NOT mocked: an
earlier, mock-only version of this work shipped an MCPBridge.close()
that raised `RuntimeError: Attempted to exit cancel scope in a different
task than it was entered in` under real use — a genuine anyio/asyncio
structured-concurrency constraint that no amount of mocking the SDK's
objects would have caught. Real subprocess, real pipes, real protocol
round trip is what caught it, and is kept here specifically so a future
change to MCPBridge's lifecycle gets the same check.
"""

import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

# See tests/unit/test_mcp_corpus_server.py for why this is a skip.
pytest.importorskip("mcp")


def test_split_csv_strips_and_drops_empty_entries():
    from research_agent.config import split_csv

    assert split_csv("a, b ,,c") == ["a", "b", "c"]
    assert split_csv("") == []
    assert split_csv("   ") == []
    assert split_csv("single") == ["single"]


def test_build_subprocess_env_only_includes_allowlisted_names(monkeypatch):
    from research_agent.tools.mcp_client import _build_subprocess_env

    monkeypatch.setenv("MCP_TEST_ALLOWED_VAR", "yes")
    monkeypatch.setenv("MCP_TEST_FORBIDDEN_VAR", "should-not-leak")

    env = _build_subprocess_env(["MCP_TEST_ALLOWED_VAR", "MCP_TEST_NOT_SET_VAR"])

    assert env == {"MCP_TEST_ALLOWED_VAR": "yes"}
    assert "MCP_TEST_FORBIDDEN_VAR" not in env
    assert "MCP_TEST_NOT_SET_VAR" not in env  # allowlisted but never set -> absent, not an error


def test_build_subprocess_env_returns_empty_dict_for_empty_allowlist(monkeypatch):
    from research_agent.tools.mcp_client import _build_subprocess_env

    monkeypatch.setenv("MCP_TEST_SOME_VAR", "x")
    assert _build_subprocess_env([]) == {}


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


def test_mcp_bridge_surfaces_a_clear_error_for_a_nonexistent_command():
    """A bad command must fail fast and clearly (FileNotFoundError, in
    practice), not hang -- proven against the REAL subprocess-spawning
    path, not a mock (a mock could never demonstrate a real spawn
    failure)."""
    from research_agent.tools.mcp_client import MCPBridge

    bridge = MCPBridge(command="this-command-does-not-exist-anywhere",
                       args=[], env_allowlist=[], startup_timeout_seconds=5.0)
    try:
        with pytest.raises(Exception):
            bridge.call_tool("search", {"query": "x"}, timeout_seconds=5.0)
    finally:
        bridge.close()  # must not itself raise, even after a failed start


def test_mcp_tool_round_trips_through_a_real_stdio_server():
    """The genuine end-to-end proof: a real subprocess, real MCP protocol,
    real stdio pipes, using tests/fixtures/mcp_echo_server.py (a ~20-line
    FastMCP server shipped in this repo, deterministic, no external
    dependencies of its own). This is what caught the anyio cancel-scope
    bug in MCPBridge.close() that no amount of mocking would have --
    see this module's docstring for the full story."""
    from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool

    server_path = str(pathlib.Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")
    bridge = MCPBridge(command=sys.executable, args=[server_path], env_allowlist=[])
    tool = make_mcp_tool(bridge, "search", query_arg_name="query")
    try:
        evidence = tool(_task(query="redis vs cassandra", key="t1", goal_id="g1"))
        assert len(evidence) == 1
        assert evidence[0].content == "canned result for: redis vs cassandra"
        assert evidence[0].source == "mcp"

        # A second call on the SAME bridge proves the connection is
        # actually PERSISTENT (reused), not re-spawned per call -- the
        # whole point of the background-loop design over asyncio.run()
        # per call (see MCPBridge's own docstring).
        evidence2 = tool(_task(query="second query", key="t2", goal_id="g1"))
        assert evidence2[0].content == "canned result for: second query"
    finally:
        bridge.close()  # must succeed cleanly -- this is the regression check


def test_mcp_bridge_survives_many_concurrent_first_calls():
    """Regression test for a REAL bug a live run caught: multiple threads
    calling call_tool() at nearly the same moment, before the bridge has
    finished connecting, used to make every thread EXCEPT the one that
    created the background thread skip the readiness wait entirely and
    crash with AttributeError ('NoneType' object has no attribute
    'call_tool') -- exactly reproducing LangGraph fanning several
    search_worker instances out concurrently for one gather-cycle
    superstep. Uses the REAL fixture server (tests/fixtures/
    mcp_echo_server.py) and REAL threads -- a mock could not have caught
    this, since the bug was a genuine race between real OS threads."""
    from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool

    server_path = str(pathlib.Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")
    bridge = MCPBridge(command=sys.executable, args=[server_path], env_allowlist=[])
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


def test_mcp_bridge_timeout_error_is_actually_informative():
    """Regression test for a fair, direct complaint: a real failure showed
    up in this codebase's own D-16 failure log as bare "reason=
    TimeoutError" with zero further detail -- concurrent.futures.
    TimeoutError's own message is EMPTY (confirmed: str(TimeoutError())
    == ""), so there was genuinely nothing else to show. This uses a REAL
    server (tests/fixtures/mcp_slow_server.py, which sleeps 5s) and a
    deliberately short 1s timeout to trigger the real timeout path fast
    and deterministically, then checks the raised exception's message
    actually names the tool, the arguments, and how long it waited --
    not just the exception's class name."""
    from research_agent.tools.mcp_client import MCPBridge

    server_path = str(pathlib.Path(__file__).parent.parent / "fixtures" / "mcp_slow_server.py")
    # sys.executable, not a hardcoded "python3" -- FOUND BY A REAL FAILURE:
    # on Windows there's typically no "python3" on PATH at all (the
    # official installer only provides "python.exe"), so this fell back to
    # whichever OTHER Python happened to resolve from PATH -- a completely
    # different interpreter than the venv running pytest, missing the mcp
    # package entirely, which crashed the subprocess immediately (surfacing
    # as a confusing "McpError: Connection closed" rather than the
    # ModuleNotFoundError that was the real cause, visible only in the
    # subprocess's own captured stderr). Every other MCP test in this file
    # already used sys.executable correctly; this one test didn't.
    bridge = MCPBridge(command=sys.executable, args=[server_path], env_allowlist=[])
    try:
        try:
            bridge.call_tool("search", {"query": "redis vs cassandra"}, timeout_seconds=1.0)
            assert False, "expected a TimeoutError"
        except TimeoutError as exc:
            message = str(exc)
            assert message, "the whole point of this fix: the message must NOT be empty"
            assert "search" in message
            assert "redis vs cassandra" in message
            assert "1.0" in message  # the configured timeout, visible in the message
    finally:
        bridge.close()
