"""
scripts/mcp_corpus_server.py -- A real MCP server exposing the SAME corpus
search this repo already has, over the MCP protocol.

Purpose:
    tools/mcp_client.py (P2-13) proves the CLIENT-side MCP plumbing works,
    but has nothing genuinely useful to talk to without a real MCP server
    -- tests/fixtures/mcp_echo_http_server.py is a deliberately trivial
    fixture for testing that plumbing, not something you'd want the agent
    actually citing evidence from. This script closes that gap: it is a
    real MCP server whose "search" tool wraps THIS SAME REPO'S existing
    tools/corpus_search.py::make_corpus_tool -- the exact HybridRetriever
    (dense Qdrant + keyword OpenSearch, fused) that cli.py already builds
    for the non-MCP path. Running the agent with MCP_ENABLED=true and this
    server running returns REAL evidence from your real ingested corpus,
    round-tripped through real MCP protocol -- not a canned string, and
    not a reimplementation of retrieval logic (this file contains no
    retrieval logic of its own at all; it only calls the existing tool
    and reformats its output).

D-76: standalone Streamable HTTP only -- this process is never spawned
by the agent. Start it yourself, in its own terminal, and leave it
running; stop it whenever you want, independent of any research_agent
run's own lifetime.

Usage:
    # In its own terminal, left running:
    python scripts/mcp_corpus_server.py --port 8765
    # Stop it whenever you want: Ctrl+C, or kill the process. Every
    # research_agent run simply fails to connect if the server isn't
    # up -- it does not, and cannot, start or stop this process for you.

    # In .env, pointing at that already-running server:
    MCP_ENABLED=true
    MCP_SERVER_URL=http://127.0.0.1:8765/mcp
    MCP_TOOL_NAME=search
    MCP_QUERY_ARG_NAME=query

    This file lives at the REPO ROOT's scripts/ directory (like
    ingest_sample_data.py) and puts its own repo-relative "src" on
    sys.path (resolved from __file__, not the CWD), so it needs no
    PYTHONPATH set in ITS OWN terminal's environment for the script
    itself to import -- but the shell you run it from still needs the
    venv active (see OPERATIONS.md's "Running the MCP servers
    standalone" for the T7/T8 terminal setup).

Scope (deliberately minimal):
    ONE tool, "search", taking ONE argument, "query" -- matching
    tools/corpus_search.py's own single-argument SearchTask.query shape.
    Returns a plain list[str] (FastMCP turns each list item into its own
    TextContent block -- confirmed against the actual installed FastMCP
    behavior, not assumed) so tools/mcp_client.py's existing "one Evidence
    per text content block" parsing needs no changes to consume this
    server's responses.

Concurrency fix (P2-13 Tier 3, confirmed live: 6 concurrent calls, wall
time 13.5s vs 79.2s summed -- ratio 1.02, i.e. genuinely concurrent):
    search() below is `async def` and offloads the blocking corpus lookup
    to a dedicated ThreadPoolExecutor, instead of a plain `def search`
    that FastMCP would call inline on its own single event loop (which
    would serialize every concurrent request behind whichever one is
    mid-flight). See that function's docstring for the full account.

First-import gotcha (found the hard way -- see git history/PR discussion
for the full investigation): qdrant_client and opensearchpy are imported
EAGERLY below, at module load time, on the main thread, before mcp.run()
starts. This looks redundant (QdrantStore/OpenSearchStore already import
them lazily inside their own __init__), but it is NOT dead code -- do
not remove it. On at least one real deployment machine, the FIRST import
of qdrant_client, when it happened lazily on a _search_executor worker
thread during a live tool call (i.e. while this process's asyncio
Proactor loop was already running real overlapped I/O on the stdio
pipes), stalled for ~120 seconds before any network call even started --
consistent with antivirus/EDR real-time scanning of native-extension
DLLs the first time this process touches them in this unusual execution
context (no window, piped stdio, spawned by another process). Forcing
that one-time cost to happen here, at startup, on the main thread, before
the event loop is doing any real I/O, avoided the stall entirely in
testing. If you ever see a mysterious ~120s stall on the very first
search() call again, check whether these two imports are still here.

Honesty note: this was NOT verified against a live Qdrant/OpenSearch in
the environment this was written in (none was reachable there -- same
limitation P2-10 flagged). The wrapping logic itself (constructing a
SearchTask, calling the existing make_corpus_tool closure, reformatting
Evidence into list[str]) is covered by a unit test using a fake tool
function; the REAL retrieval round trip has since been confirmed live,
including under real concurrency (see the fix note above).
"""
import asyncio
import atexit
import argparse
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Resolve "src" RELATIVE TO THIS FILE, never relative to the current
# working directory. `sys.path.insert(0, "src")` only resolved when the
# process happened to be launched from the repo root -- not guaranteed
# for a script launched as an MCP_SERVER_COMMAND subprocess, from a
# Windows shortcut or scheduled task, or from any other directory --
# and failed with an opaque ModuleNotFoundError when it did not.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# D-76: this server ALWAYS runs Streamable HTTP -- standalone-only, no
# stdio mode. host/port are parsed BEFORE FastMCP() is constructed, not
# inside `if __name__ == "__main__":` at the bottom -- they are FastMCP
# CONSTRUCTOR arguments, not mcp.run() arguments, so they must be known
# before the `mcp = FastMCP(...)` line below runs.
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    # parse_known_args, not parse_args: this module is also imported
    # directly by tests/unit/test_mcp_corpus_server.py (to exercise its
    # wrapping logic against a fake tool, with no real server ever run)
    # -- at IMPORT time, under pytest's own argv (-q, file paths, ...),
    # which are not this script's flags and must be silently ignored
    # rather than raising SystemExit before the test module's own code
    # ever runs.
    args, _unknown = parser.parse_known_args()
    return args


_args = _parse_args()
from research_agent.config import get_settings  # noqa: E402
from research_agent.retrieval.hybrid import HybridRetriever  # noqa: E402
from research_agent.state import SearchTask  # noqa: E402
from research_agent.storage.opensearch_store import OpenSearchStore  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore  # noqa: E402
from research_agent.tools.corpus_search import make_corpus_tool  # noqa: E402

# See "First-import gotcha" in the module docstring above before touching
# these two lines -- they look like dead/redundant imports but are not.
import qdrant_client  # noqa: F401,E402
import opensearchpy  # noqa: F401,E402

mcp = FastMCP("corpus-search-server", host=_args.host, port=_args.port)


def _build_corpus_tool():
    """Construct the SAME corpus tool cli.py builds for the non-MCP path.

    CALLED BY   module level, below, ONCE at import time -- this server
                process is short-lived (one per MCPBridge, torn down with
                it), so there is no benefit to lazy construction the way
                storage/qdrant_store.py's own lazy-connect pattern has for
                a long-running agent process.
    Deliberately mirrors cli.py::build_app_and_settings's own
    dense/keyword/HybridRetriever/make_corpus_tool construction line for
    line -- see that function if this ever needs to change, since the two
    should stay in sync (both are "the corpus tool", just reached two
    different ways).
    """
    settings = get_settings()
    dense = QdrantStore(settings.qdrant_url, settings.corpus_index)
    keyword = OpenSearchStore(
        settings.opensearch_url, settings.corpus_index,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs)
    return make_corpus_tool(HybridRetriever(dense, keyword, min_similarity=settings.min_similarity))


# Deliberately NOT built at import time (an earlier version was, and a
# real test run showed importing this module for testing alone triggers
# a real ~10s Qdrant/OpenSearch connection-retry sequence even though
# both degrade gracefully -- correct behavior, just far too slow and
# noisy for something a test should be able to import instantly). Built
# lazily instead, exactly once, on first real use -- see
# _get_corpus_tool() below. A test that wants to exercise hits_for_query
# without any live backend just assigns _corpus_tool directly, bypassing
# _get_corpus_tool()'s lazy-build path entirely.
_corpus_tool = None
# _corpus_tool_lock: protects lazy singleton construction of the corpus
# tool. The first call triggers _build_corpus_tool() (Qdrant/OpenSearch
# connections + fastembed model load), which is slow (~10-13s); every
# later call reuses the same instance. The lock ensures only ONE thread
# ever builds it -- concurrent arrivals block on the lock, then return
# the already-built instance. This matters because search() below runs
# the blocking corpus work via a ThreadPoolExecutor, so multiple
# search() calls can execute concurrently on the thread pool; without
# the lock, every concurrent arrival would redundantly build its own
# QdrantStore/OpenSearchStore/embedding model.
# Six search_worker tasks firing at once (this codebase's normal gather-cycle
# fan-out) meant six threads could all see _corpus_tool is None
# SIMULTANEOUSLY and all six would build their OWN separate QdrantStore/
# OpenSearchStore/fastembed-model-load AT THE SAME TIME -- turning one
# ~13s cold start (confirmed by direct measurement, bypassing MCP/stdio
# entirely) into six REDUNDANT, CONCURRENTLY-COMPETING cold starts, easily
# pushing every single one of them past a 30s client-side timeout even
# though none of them was individually broken.
_corpus_tool_lock = threading.Lock()


def _get_corpus_tool():
    """Return the module-level corpus tool, building it on first call.

    CALLED BY   hits_for_query, below -- every real call, but only the
                FIRST one actually builds anything; every later call
                reuses the same closure, same lazy-singleton shape
                storage/qdrant_store.py's own _embedder field already
                uses for its embedding model.

    THE FIX: the lock ensures only ONE thread ever runs
    _build_corpus_tool() -- any other thread arriving while that's in
    progress blocks on the lock (not on a redundant build of its own),
    then sees _corpus_tool already set once it gets the lock, and returns
    the SAME instance immediately. Costs nothing once built: every later
    call is an near-instant lock acquire + release + return, same as
    MCPBridge.start()'s already-ready Event.wait() case.
    """
    global _corpus_tool
    with _corpus_tool_lock:
        if _corpus_tool is None:
            _corpus_tool = _build_corpus_tool()
        return _corpus_tool


def hits_for_query(query: str) -> list:
    """Run `query` through the existing corpus tool, return plain strings.

    CALLED BY   search(), below -- factored out as its own function so a
                test can call this DIRECTLY with a fake `_corpus_tool`
                substituted in, without needing FastMCP, stdio, or a real
                Qdrant/OpenSearch at all (see
                tests/test_tier3.py::test_hits_for_query_wraps_the_corpus_tool_correctly).
    CALLS       _corpus_tool(task) -- the EXACT closure
                tools/corpus_search.py::make_corpus_tool returns; this
                function contains no retrieval logic of its own.
    RETURNS     one string per Evidence item the corpus tool produced,
                content only (dropping score/volatility/etc -- this
                server's MCP tool schema is intentionally minimal, just
                text in, text list out; task_key/goal_id are synthesized
                per call below since an MCP tool call has no concept of
                this repo's own SearchTask identity).
    """
    # Diagnostic timing, to stderr ONLY -- NEVER stdout in an MCP stdio
    # server: stdout IS the JSON-RPC message channel itself, and printing
    # anything else there would corrupt the protocol stream. thread_name
    # lets you tell whether FastMCP is actually running concurrent
    # requests on separate threads (several different thread names
    # interleaved in the log) or fully serializing them (the same thread
    # name for every request, one full start/end pair at a time).
    thread_name = threading.current_thread().name
    t0 = time.time()
    print(f"[mcp_corpus_server] START thread={thread_name} query={query!r}",
         file=sys.stderr, flush=True)
    task = SearchTask(key=f"mcp::{query}", goal_id="mcp", query=query, depth=0)
    evidence = _get_corpus_tool()(task)
    elapsed = time.time() - t0
    print(f"[mcp_corpus_server] DONE  thread={thread_name} query={query!r} "
         f"took={elapsed:.1f}s items={len(evidence)}", file=sys.stderr, flush=True)
    return [e.content for e in evidence]


_search_executor = ThreadPoolExecutor(max_workers=get_settings().mcp_max_workers)
# Shut the pool down on interpreter exit. Without this a server process torn
# down by MCPBridge.close() sits in atexit's default join waiting on
# in-flight worker threads with no bound and no diagnostic -- and on Windows
# an un-joined pool still holding a live Qdrant/OpenSearch client is exactly
# the shape of hang that is hardest to attribute after the fact.
atexit.register(_search_executor.shutdown, wait=False)


@mcp.tool()
async def search(query: str) -> list:
    """Search the ingested corpus (the same one cli.py's non-MCP path
    searches) and return matching passages.

    async def + a dedicated thread-pool offload (P2-13 Tier 3 fix): a
    plain `def search` would have FastMCP invoke it directly, inline, on
    the server's single event loop -- confirmed by reading
    func_metadata.py::call_fn_with_arg_validation, which calls a
    synchronous tool handler with no thread offload of its own. That
    serializes every concurrent request behind whichever one is
    currently blocked on a real Qdrant/OpenSearch round trip. Making this
    `async def` and running the actual blocking work (hits_for_query) on
    `_search_executor` instead lets FastMCP's event loop keep servicing
    other requests while this one's blocking call is in flight.

    Confirmed live (after fixing the separate first-import stall
    documented in the module docstring): 6 concurrent calls completed in
    13.5s wall time vs 79.2s summed -- ratio 1.02, i.e. genuinely
    concurrent, not serialized.

    hits_for_query (above) is still the entire implementation; this is a
    thin async wrapper around it, not a reimplementation.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_search_executor, hits_for_query, query)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")