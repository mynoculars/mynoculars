"""
scripts/mcp_web_search_server.py -- A real MCP server exposing web search
over the MCP protocol, as a sibling to scripts/mcp_corpus_server.py.

Purpose:
    Phase 4 (D-57). The agent's retrieval ladder (D-38) ends in the model's
    own recollection, because corpus_search, the corpus MCP server and
    semantic memory all resolve to the SAME ingested documents -- so "the
    corpus does not contain it" had nowhere left to escalate. This server is
    that missing rung: a real search engine, reachable over the transport
    the agent already speaks.

Usage -- standalone Streamable HTTP server (D-76: this is the only mode):
    # In its own terminal, left running:
    python scripts/mcp_web_search_server.py --port 8766
    # Stop it whenever you want: Ctrl+C, or kill the process.

    # In .env, pointing at that already-running server:
    WEB_SEARCH_ENABLED=true
    WEB_MCP_SERVER_URL=http://127.0.0.1:8766/mcp
    WEB_MCP_TOOL_NAME=web_search
    WEB_MCP_QUERY_ARG_NAME=query

    Like scripts/mcp_corpus_server.py, this file lives in the repo root's
    scripts/ directory and puts its own repo-relative "src" on sys.path
    resolved from __file__ (never from the CWD) -- but the shell you run
    it from still needs the venv active (see OPERATIONS.md's "Running
    the MCP servers standalone" for the T7/T8 terminal setup).

    THIS PROCESS MAKES OUTBOUND INTERNET REQUESTS. Behind a corporate
    proxy, set HTTPS_PROXY/HTTP_PROXY/NO_PROXY on THIS terminal's own
    environment before launching it (D-76: there is no longer an
    agent-side env allowlist to configure this through -- this is now an
    independent process with its own environment, set however you
    normally would for any long-running server).

Scope (deliberately minimal, mirroring the corpus server's own posture):
    ONE tool, "web_search", taking a query and an optional result cap.
    Returns list[dict] -- one dict per result, in the shape
    research_agent.websearch.provider.as_payload defines, which is the
    contract tools/mcp_client.py parses on the agent side.

    WIRE SHAPE, verified against the installed FastMCP rather than assumed
    (the same standard scripts/mcp_corpus_server.py held itself to for its
    own list[str] return): a `-> list[dict]` annotation makes FastMCP emit
    BOTH

        result.structuredContent == {"result": [ {...}, {...} ]}
        result.content           == one TextContent block per item, each
                                    carrying that item as pretty-printed JSON

    structuredContent is the channel the agent side reads, because it is the
    only one carrying per-item `score` as a real number rather than as text
    to be re-parsed. The text blocks are a usable fallback and are why an
    older, unmodified mcp_client would degrade to "one Evidence per result
    with JSON as its content" rather than failing outright.

Why this server does the SCORING (websearch/scoring.py), rather than
returning bare ranks for the agent to score:
    Ranking policy belongs with the search implementation. If the agent
    scored ranks itself, swapping DDGS for a provider whose ranking means
    something different would silently change what a score means on the
    agent side, with no code change anywhere near the swap. Scoring here
    means a provider owns its own calibration.

Concurrency (the same fix, for the same reason, as the corpus server):
    web_search() below is `async def` and offloads the blocking provider
    call to a dedicated ThreadPoolExecutor. A plain `def` handler would be
    invoked inline on FastMCP's single event loop, serializing every
    concurrent request behind whichever one is mid-flight -- and this
    codebase's normal gather-cycle fan-out is six tasks at once.

First-import gotcha (see scripts/mcp_corpus_server.py's module docstring
for the full investigation): ddgs is imported EAGERLY below, at module load,
on the main thread, before mcp.run() starts. That looks redundant --
websearch/ddgs_provider.py imports it lazily inside DDGSProvider.__init__ --
and it is NOT dead code. On a real deployment machine, the first-ever import
of a package pulling native extensions, happening lazily on a worker thread
while this process's Proactor loop was already running overlapped I/O on the
stdio pipes, stalled ~120s before any network call started. Forcing that
one-time cost to happen here, at startup, avoids it.

    The try/except around that import is deliberate and does not weaken it:
    when ddgs IS installed the eager import still happens, exactly as
    required. It only allows this module to be IMPORTED without ddgs
    present, so tests/unit/test_mcp_web_search_server.py can exercise this
    file's own wrapping logic on a minimal install (the same posture
    tests/unit/test_gc_memory.py and test_mcp_corpus_server.py already take
    toward their optional dependencies). A real call with ddgs missing still
    fails, with DDGSProvider.__init__'s explicit "pip install ddgs" message
    rather than a bare NameError.

Honesty note: the wrapping logic in this file (building the provider,
filtering, scoring, reformatting to the payload shape) is covered by unit
tests against a fake provider, and the MCP wire shape is covered by a real
stdio round trip against tests/fixtures/mcp_web_search_echo_server.py. What
is NOT covered by any test, by design, is a live DDGS query -- this repo's
test suite is entirely offline (see tests/conftest.py) and a test that hits
the real internet would be non-deterministic, rate-limitable, and would
break in CI for reasons having nothing to do with this code. Verify the live
path with scripts/check_services.py, which is where D-33 puts exactly this
kind of check.
"""
import asyncio
import atexit
import argparse
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Resolve "src" RELATIVE TO THIS FILE, never relative to the current working
# directory -- a script launched as an MCP server subprocess has no
# guaranteed CWD. Identical to scripts/mcp_corpus_server.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# D-76: this server ALWAYS runs Streamable HTTP -- standalone-only, same
# reasoning as scripts/mcp_corpus_server.py's identical block -- host/port
# parsed BEFORE FastMCP() below, since they are constructor arguments,
# not mcp.run() arguments.
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    # parse_known_args -- see mcp_corpus_server.py's identical block for
    # why (this module is also imported directly by
    # tests/unit/test_mcp_web_search_server.py, under pytest's own argv).
    args, _unknown = parser.parse_known_args()
    return args


_args = _parse_args()
from research_agent.config import get_settings  # noqa: E402
from research_agent.websearch.filtering import cap_by_domain, dedupe_by_url  # noqa: E402
from research_agent.websearch.provider import as_payload  # noqa: E402
from research_agent.websearch.scoring import rank_to_score  # noqa: E402

# See "First-import gotcha" in the module docstring above before touching
# this -- it looks like a redundant import and is not.
try:  # pragma: no cover - install-shape branch
    import ddgs  # noqa: F401,E402
except ImportError:  # pragma: no cover - install-shape branch
    ddgs = None

mcp = FastMCP("web-search-server", host=_args.host, port=_args.port)


def _build_provider():
    """Construct the configured SearchProvider from Settings.

    CALLED BY   _get_provider(), below, at most once per process.
    Mirrors scripts/mcp_corpus_server.py::_build_corpus_tool: read
    get_settings() here, in the server process, rather than expecting the
    agent to forward configuration across the MCP boundary. The server is an
    entry point in its own right and reads its own .env, which is exactly
    why WEB_MCP_SERVER_ENV_ALLOWLIST can stay minimal.
    """
    settings = get_settings()
    from research_agent.websearch.ddgs_provider import build_provider
    return build_provider(
        settings.web_search_provider,
        region=settings.web_search_region,
        safesearch=settings.web_search_safesearch,
        timeout_seconds=settings.web_search_provider_timeout_seconds)


# Deliberately NOT built at import time, for the same reason the corpus
# server's _corpus_tool is not: importing this module (for a test, or for
# --help) must not construct an HTTP client or resolve a provider. A test
# that wants to exercise hits_for_query without ddgs installed assigns
# _provider directly, bypassing _get_provider()'s lazy-build path entirely.
_provider = None
# _provider_lock: protects lazy singleton construction. Six search_worker
# tasks firing at once (this codebase's normal gather-cycle fan-out) means
# six threads can see `_provider is None` SIMULTANEOUSLY and all six build
# their own provider -- six HTTP clients where one was wanted, and six
# separate sessions arriving at the same endpoint at the same instant, which
# is the shape most likely to be throttled. The corpus server learned this
# the expensive way (see its _corpus_tool_lock comment: one ~13s cold start
# became six competing ones); it is cheaper to inherit the lesson than to
# rediscover it.
_provider_lock = threading.Lock()


def _get_provider():
    """Return the module-level provider, building it on first call.

    Costs nothing once built: every later call is a near-instant lock
    acquire/release/return, the same shape as MCPBridge.start()'s
    already-ready Event.wait().
    """
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = _build_provider()
        return _provider


def hits_for_query(query: str, max_results: int = None) -> list:
    """Run `query` through the configured provider, return payload dicts.

    CALLED BY   web_search(), below -- factored out as its own plain
                synchronous function so a test can call it DIRECTLY with a
                fake `_provider` substituted in, needing no FastMCP, no
                stdio, no ddgs and no network. Exactly the seam
                scripts/mcp_corpus_server.py::hits_for_query provides.

    PIPELINE, in this order, and the order matters:

        provider.search()   engine's own ranking, best first
        dedupe_by_url()     drop exact repeats
        cap_by_domain()     at most N per registrable domain
        rank_to_score()     ordinal rank -> score, over the SURVIVING count
        as_payload()        flatten to the MCP wire shape

    Scoring runs LAST, after both filters, so the band is interpolated
    across the results actually being returned. Scoring first and filtering
    after would leave gaps -- the worst survivor would not carry the floor,
    and the spread would silently compress by an amount depending on how
    many duplicates the engine happened to return.

    RAISES whatever the provider raises. Deliberately not caught: the MCP
    layer turns an exception into a tool-level error the agent can see,
    whereas returning [] here would report a broken engine as "found
    nothing" -- and the retrieval ladder would then quietly escalate to the
    model tier as if the web genuinely had no answer. Making those two
    states indistinguishable is the defect the old hardcoded MCP score=1.0
    and MIN_EVIDENCE_SCORE=0.0 both were.
    """
    settings = get_settings()
    if max_results is None:
        max_results = settings.web_search_max_results
    # Clamp rather than reject: an agent-side caller asking for 500 results
    # is a caller bug, but failing the whole search over it is a worse
    # outcome than quietly serving the configured maximum. The le=25 bound
    # on the Settings field is the policy; this enforces it against a value
    # that arrived over the wire instead of from .env.
    max_results = max(1, min(int(max_results), settings.web_search_max_results))

    # Diagnostic timing, to stderr ONLY -- NEVER stdout in an MCP stdio
    # server: stdout IS the JSON-RPC message channel, and anything else
    # printed there corrupts the protocol stream. thread_name lets you see
    # whether FastMCP is genuinely running requests concurrently (several
    # interleaved thread names) or serializing them (one name, one
    # start/end pair at a time).
    thread_name = threading.current_thread().name
    t0 = time.time()
    print(f"[mcp_web_search_server] START thread={thread_name} "
          f"query={query!r} max_results={max_results}",
          file=sys.stderr, flush=True)

    results = _get_provider().search(query, max_results=max_results)
    results = dedupe_by_url(results)
    results = cap_by_domain(results, settings.web_search_max_per_domain)

    total = len(results)
    payload = [
        as_payload(r, round(rank_to_score(
            r.rank, total,
            settings.web_search_min_score,
            settings.web_search_max_score), 4))
        for r in results
    ]

    elapsed = time.time() - t0
    print(f"[mcp_web_search_server] DONE  thread={thread_name} query={query!r} "
          f"took={elapsed:.1f}s items={len(payload)} "
          f"domains={len({p['domain'] for p in payload})}",
          file=sys.stderr, flush=True)
    return payload


_search_executor = ThreadPoolExecutor(
    max_workers=get_settings().web_search_max_workers)
# Shut the pool down on interpreter exit -- same reasoning as the corpus
# server's: without this, a process torn down by MCPBridge.close() sits in
# atexit's default join waiting on in-flight worker threads with no bound and
# no diagnostic, and on Windows an un-joined pool still holding a live HTTP
# session is the shape of hang that is hardest to attribute after the fact.
atexit.register(_search_executor.shutdown, wait=False)


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return ranked, scored results.

    Each result carries: title, url, snippet, rank (1-based, best first),
    engine, domain, and score (already mapped onto the configured band, so
    the caller does not need to know this server's ranking policy).

    Results are deduplicated by URL and capped per domain, so five hits from
    one site cannot masquerade as five independent sources.

    `max_results` is a request, not a guarantee: it is clamped to the
    server's configured WEB_SEARCH_MAX_RESULTS, and filtering may return
    fewer.

    async def + thread-pool offload: see the module docstring. hits_for_query
    (above) is the entire implementation; this is a thin async wrapper, not a
    reimplementation.

    The default of 5 is written literally rather than read from Settings
    because FastMCP captures a tool's defaults into the SCHEMA it advertises
    over the protocol, at import time -- a Settings lookup here would bake
    one process's configuration into a published schema. Passing max_results
    explicitly, or omitting it, both end up honouring the configured value
    inside hits_for_query, which is where the setting genuinely belongs.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _search_executor, hits_for_query, query, max_results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
