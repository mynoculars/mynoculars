"""
scripts/mcp_corpus_server.py -- A real MCP server exposing the SAME corpus
search this repo already has, over the MCP protocol.

Purpose:
    tools/mcp_client.py (P2-13) proves the CLIENT-side MCP plumbing works,
    but has nothing genuinely useful to talk to without a real MCP server
    -- tests/fixtures/mcp_echo_server.py is a deliberately trivial fixture
    for testing that plumbing, not something you'd want the agent
    actually citing evidence from. This script closes that gap: it is a
    real MCP server, launchable as MCP_SERVER_COMMAND/MCP_SERVER_ARGS,
    whose "search" tool wraps THIS SAME REPO'S existing
    tools/corpus_search.py::make_corpus_tool -- the exact HybridRetriever
    (dense Qdrant + keyword OpenSearch, fused) that cli.py already builds
    for the non-MCP path. Running the agent with MCP_ENABLED=true and this
    script configured returns REAL evidence from your real ingested
    corpus, round-tripped through real MCP protocol -- not a canned
    string, and not a reimplementation of retrieval logic (this file
    contains no retrieval logic of its own at all; it only calls the
    existing tool and reformats its output).

Usage (as an MCP_SERVER_COMMAND, not run directly by a person):
    MCP_ENABLED=true
    MCP_SERVER_COMMAND=<path to your venv's python>
    MCP_SERVER_ARGS=scripts/mcp_corpus_server.py
    MCP_SERVER_ENV_ALLOWLIST=
    MCP_TOOL_NAME=search
    MCP_QUERY_ARG_NAME=query

    This file lives at the REPO ROOT's scripts/ directory (like
    ingest_sample_data.py) and does its own sys.path.insert(0, "src"), so
    it needs no PYTHONPATH set in the MCP subprocess's environment --
    matching every other script in this repo, and matching P2-13's own
    env-allowlist design (MCP_SERVER_ENV_ALLOWLIST can stay empty; this
    server needs no inherited environment variables to build its own
    Settings, since get_settings() reads directly from THIS process's
    .env/environment at startup, same as any other entry point).

Scope (deliberately minimal):
    ONE tool, "search", taking ONE argument, "query" -- matching
    tools/corpus_search.py's own single-argument SearchTask.query shape.
    Returns a plain list[str] (FastMCP turns each list item into its own
    TextContent block -- confirmed against the actual installed FastMCP
    behavior, not assumed) so tools/mcp_client.py's existing "one Evidence
    per text content block" parsing needs no changes to consume this
    server's responses.

Honesty note: this was NOT verified against a live Qdrant/OpenSearch in
the environment this was written in (none was reachable there -- same
limitation P2-10 flagged). The wrapping logic itself (constructing a
SearchTask, calling the existing make_corpus_tool closure, reformatting
Evidence into list[str]) is covered by a unit test using a fake tool
function; the REAL retrieval round trip needs to be confirmed against
your actual running Qdrant/OpenSearch.
"""

import sys
import threading
import time

sys.path.insert(0, "src")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from research_agent.config import get_settings  # noqa: E402
from research_agent.retrieval.hybrid import HybridRetriever  # noqa: E402
from research_agent.state import SearchTask  # noqa: E402
from research_agent.storage.opensearch_store import OpenSearchStore  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore  # noqa: E402
from research_agent.tools.corpus_search import make_corpus_tool  # noqa: E402

mcp = FastMCP("corpus-search-server")


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
# FIXED BUG (found via a real concurrent run, not a test -- the same
# class of race already caught once in tools/mcp_client.py::MCPBridge.
# start(), this time here): the original _get_corpus_tool had no lock
# around its "if _corpus_tool is None: build it" check. FastMCP dispatches
# concurrent tool calls to worker threads (this server's own search()
# function is synchronous, doing blocking Qdrant/OpenSearch HTTP calls --
# a typical MCP server framework offloads a sync tool handler to a thread
# pool so it doesn't block the server's single event loop). Six
# search_worker tasks firing at once (this codebase's normal gather-cycle
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
    # anything else there would corrupt the protocol stream. This exists
    # because a real concurrency investigation had no visibility at all
    # into what this SERVER process was actually doing while the CLIENT
    # side sat timing out -- "reason=TimeoutError" alone gave no way to
    # tell whether this server ever started the request, was still
    # working on it, or had already finished and something else swallowed
    # the response. thread_name lets you tell whether FastMCP is actually
    # running concurrent requests on separate threads (several different
    # thread names interleaved in the log) or fully serializing them (the
    # same thread name for every request, one full start/end pair at a
    # time).
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


@mcp.tool()
def search(query: str) -> list:
    """Search the ingested corpus (the same one cli.py's non-MCP path
    searches) and return matching passages.

    This is the ONLY tool this server exposes, and hits_for_query (above)
    is its entire implementation -- see that function's docstring for
    what it actually does and why it's factored out separately.
    """
    return hits_for_query(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
