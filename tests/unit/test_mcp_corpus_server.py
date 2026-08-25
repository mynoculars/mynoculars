"""
tests/unit/test_mcp_corpus_server.py — scripts/mcp_corpus_server.py's
OWN wrapping logic (hits_for_query, search, _get_corpus_tool).

A real MCP server wrapping the EXISTING tools/corpus_search.py tool --
built because a fair question ("the MCP server just has to call the
corpus tools, right?") pointed out that tests/fixtures/mcp_echo_http_server.py
proves the wiring but returns nothing genuinely useful. This tests
scripts/mcp_corpus_server.py's own wrapping logic via a fake corpus tool
substituted directly into the module's _corpus_tool global -- deliberately
NOT importing it in a way that would trigger the real lazy QdrantStore/
OpenSearchStore construction (see that file's own module docstring for
why importing it eagerly builds nothing slow, and for the separate
"First-import gotcha" this suite's threshold below accounts for).
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# mcp is an OPTIONAL extra (only reached when MCP_ENABLED=true). Skip
# rather than fail on a minimal install -- see tests/unit/test_gc_memory.py.
# qdrant_client/opensearchpy are named too because scripts/mcp_corpus_server.py
# imports BOTH eagerly at module load (deliberately -- see that file's
# "First-import gotcha"), so importing it here needs them present.
pytest.importorskip("mcp")
pytest.importorskip("qdrant_client")
pytest.importorskip("opensearchpy")


def _load_mcp_corpus_server():
    import importlib.util
    import pathlib

    script_path = (pathlib.Path(__file__).parent.parent.parent
                  / "scripts" / "mcp_corpus_server.py")
    spec = importlib.util.spec_from_file_location("mcp_corpus_server", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_corpus_tool_returning(*contents):
    """A stand-in for tools/corpus_search.py's own returned closure --
    same ToolFn shape (task in, Evidence list out), fixed canned content
    regardless of the task's actual query, purely for testing
    mcp_corpus_server.py's OWN wrapping logic in isolation."""
    from research_agent.state import Evidence, Volatility

    def tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="corpus",
                         content=c, score=0.9, volatility=Volatility.SEMI_STABLE)
                for c in contents]
    return tool


def test_mcp_corpus_server_imports_instantly_without_a_live_backend():
    """Regression guard for the real bug a manual test caught: importing
    this module must NOT eagerly build a real QdrantStore/OpenSearchStore
    CONNECTION (that used to take ~10s of retry/backoff even though it
    degrades gracefully) -- _corpus_tool must start as None.

    The wall-clock threshold below is deliberately loose (30s, not the
    original 2s). A later fix (see mcp_corpus_server.py's module
    docstring, "First-import gotcha") made this module eagerly IMPORT
    qdrant_client/opensearchpy -- not connect to them -- at module load
    time, on purpose, to avoid a real ~120s stall that happened when
    their first import instead happened lazily on a worker thread during
    a live tool call. That eager import alone genuinely costs several
    seconds (confirmed: ~9s in one environment) even with no live
    backend reachable, which is an accepted, intentional trade-off: a few
    seconds of slower importability in exchange for never silently
    hanging on a real request. This test still fails fast (30s, not
    unbounded) if that cost balloons far beyond what a plain package
    import should cost, and still asserts _corpus_tool stays None -- the
    thing this test actually guards against (an eager CONNECTION attempt)
    is unchanged.
    """
    t0 = time.time()
    mod = _load_mcp_corpus_server()
    elapsed = time.time() - t0

    assert mod._corpus_tool is None
    assert elapsed < 30.0, f"import took {elapsed}s -- far more than a plain package import should cost"


def test_hits_for_query_wraps_the_corpus_tool_correctly():
    mod = _load_mcp_corpus_server()
    mod._corpus_tool = _fake_corpus_tool_returning("hit one", "hit two")

    result = mod.hits_for_query("redis vs cassandra")

    assert result == ["hit one", "hit two"]


def test_hits_for_query_constructs_a_valid_search_task():
    """The corpus tool receives a real SearchTask, not a bare string --
    confirms the wrapping actually goes through this repo's normal
    SearchTask/Evidence contract, not some shortcut."""
    mod = _load_mcp_corpus_server()
    seen_tasks = []

    def capturing_tool(task):
        seen_tasks.append(task)
        return []

    mod._corpus_tool = capturing_tool
    mod.hits_for_query("my query")

    assert len(seen_tasks) == 1
    assert seen_tasks[0].query == "my query"
    assert seen_tasks[0].key  # non-empty
    assert seen_tasks[0].goal_id  # non-empty


def test_mcp_corpus_server_search_function_matches_hits_for_query():
    """search() (the actual @mcp.tool()-decorated function FastMCP
    exposes) is `async def` (P2-13 Tier 3 concurrency fix: the blocking
    call is offloaded to a thread pool so FastMCP's single event loop
    isn't held up -- see README.md Limitations #6) but must still be a
    thin wrapper -- same result as calling hits_for_query directly, just
    awaited."""
    mod = _load_mcp_corpus_server()
    mod._corpus_tool = _fake_corpus_tool_returning("a", "b", "c")

    assert asyncio.run(mod.search("q")) == mod.hits_for_query("q") == ["a", "b", "c"]


def test_get_corpus_tool_only_builds_once():
    """The lazy-singleton pattern: _get_corpus_tool must not rebuild on
    every call once _corpus_tool is already set."""
    mod = _load_mcp_corpus_server()
    sentinel = _fake_corpus_tool_returning("x")
    mod._corpus_tool = sentinel

    first = mod._get_corpus_tool()
    second = mod._get_corpus_tool()

    assert first is sentinel
    assert second is sentinel


def test_get_corpus_tool_builds_exactly_once_under_real_concurrent_load():
    """Regression test for a REAL bug a live run caught: the original
    _get_corpus_tool had no lock around its "if _corpus_tool is None:
    build it" check. FastMCP dispatches concurrent tool calls to worker
    threads (this server's search() does blocking Qdrant/OpenSearch
    calls), so six search_worker tasks firing at once -- this codebase's
    normal gather-cycle fan-out -- meant six threads could all see
    _corpus_tool is None SIMULTANEOUSLY and all six would build their OWN
    separate QdrantStore/OpenSearchStore AT THE SAME TIME, turning one
    measured ~13s cold start into six concurrently-competing ones that
    blew past a 30s client-side timeout. Uses REAL threads and a slow
    fake build function (not the real Qdrant/OpenSearch, which isn't
    available in this test environment) to prove only ONE build ever
    happens no matter how many threads race in at once -- the actual
    mechanism under test is the lock, not the real retrieval backend."""
    mod = _load_mcp_corpus_server()
    build_count = []
    build_lock = threading.Lock()  # only protects the counter itself, not _get_corpus_tool

    def slow_fake_build():
        with build_lock:
            build_count.append(1)
        time.sleep(0.2)  # long enough that concurrent callers would
                          # overlap if the real lock weren't doing its job
        return _fake_corpus_tool_returning("built")

    mod._build_corpus_tool = slow_fake_build

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: mod._get_corpus_tool(), range(8)))

    assert len(build_count) == 1, (
        f"expected exactly 1 build, got {len(build_count)} -- "
        "the thundering-herd race is back")
    assert all(r is results[0] for r in results), "every caller must get the SAME instance"
