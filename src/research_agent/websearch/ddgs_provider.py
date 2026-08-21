"""
websearch/ddgs_provider.py — the ONLY module in this repo that imports ddgs.

Purpose:
    A concrete SearchProvider backed by DDGS (the DuckDuckGo metasearch
    client, distributed as the `ddgs` package -- renamed from
    `duckduckgo-search`; both names appear in older write-ups, `ddgs` is the
    current one and the only one this file uses).

Why DDGS for the first implementation:
    No API key, no account, no per-call cost, works from a Windows-native
    checkout with nothing else running. That makes it the right DEFAULT for
    a reference implementation someone clones and runs.

    What it is NOT: a production search dependency. DDGS is an unofficial
    client against an endpoint that does not promise it anything -- it can
    be throttled or CAPTCHA'd under load, and it has broken across its own
    releases before. That is precisely why provider.SearchProvider exists
    and why this file is the only one that names ddgs: replacing it with a
    keyed API (or a self-hosted SearXNG) is a new sibling module plus one
    setting, and touches nothing in src/research_agent/agents/,
    orchestration/, or tools/.

Import policy (two DIFFERENT rules, both deliberate -- do not "simplify"
one into the other):

    HERE, the `ddgs` import is LAZY, inside __init__. Importing this module
    must not require ddgs to be installed, so that tests/unit/
    test_websearch_provider.py can import the package and exercise the
    provider-independent logic on a minimal install, exactly as
    tests/unit/test_gc_memory.py and test_mcp_corpus_server.py already
    importorskip their optional dependencies.

    IN scripts/mcp_web_search_server.py, the same import is EAGER, at module
    top level, on the main thread, before mcp.run(). That is not
    redundancy -- it is the fix for the first-import stall documented in
    scripts/mcp_corpus_server.py's "First-import gotcha": a first-ever
    import of a package pulling native extensions, happening lazily on a
    thread-pool worker while the Proactor loop is already doing real
    overlapped I/O on the stdio pipes, stalled ~120s on a real machine.
    Forcing that one-time cost to happen at startup avoids it.
"""

import logging
from typing import List, Optional

from research_agent.logging_setup import log_event
from research_agent.websearch.provider import WebResult, coerce_results

logger = logging.getLogger(__name__)

# The DDGS backend name stamped onto every WebResult this provider makes.
# A constant rather than a literal at the construction site so a future
# news/images backend is an obvious sibling, not a magic string edit.
ENGINE_TEXT = "ddg_text"


class DDGSProvider:
    """SearchProvider backed by DDGS text search.

    LIFECYCLE   The underlying DDGS client is constructed ONCE, here, and
                reused for every search. DDGS holds an HTTP session
                internally; building a fresh one per query would discard
                connection reuse and, worse, look to the far end like a
                burst of unrelated clients -- the shape most likely to be
                throttled.
    THREADING   scripts/mcp_web_search_server.py serves concurrent tool
                calls on a ThreadPoolExecutor, so search() below can run on
                several threads at once against this one instance. That is
                the same concurrency shape scripts/mcp_corpus_server.py
                already runs, and the same reason its _corpus_tool_lock
                exists -- see _get_provider() there for the singleton
                construction guard on this object.

    Deliberately does NOT subclass provider.SearchProvider: it is a
    typing.Protocol (see that module), so structural compatibility is the
    whole contract and inheritance would add nothing.
    """

    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate",
                 timeout_seconds: float = 20.0):
        """Build the provider and its one long-lived DDGS client.

        Parameters:
            region: DDGS region code ("wt-wt" is its no-region default;
                "in-en" biases toward India/English). Configurable rather
                than hardcoded because a research agent run from Bengaluru
                and one run from Frankfurt should not be forced to the same
                result set -- and because hardcoding a region is exactly
                the class of runtime setting this project keeps in .env.
            safesearch: DDGS's own filter level ("on"/"moderate"/"off").
            timeout_seconds: per-request HTTP timeout inside DDGS. Bounded
                here as well as at the MCP call boundary
                (WEB_MCP_CALL_TIMEOUT_SECONDS) because the two protect
                different things: this one stops a single hung HTTP request
                from occupying a thread-pool slot indefinitely; that one
                stops the agent waiting on a wedged server subprocess.

        Raises ImportError, with a message naming the install, if ddgs is
        absent -- rather than failing later with a bare NameError on first
        use. See the module docstring for why the import is lazy at all.
        """
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover - install-shape branch
            raise ImportError(
                "The 'ddgs' package is required for WEB_SEARCH_PROVIDER=ddgs. "
                "Install it with:  pip install ddgs   (it is also pulled in by "
                "the 'websearch' extra:  pip install research-agent[websearch])"
            ) from exc

        self._region = region
        self._safesearch = safesearch
        self._timeout = timeout_seconds
        self._client = DDGS(timeout=timeout_seconds)

    def search(self, query: str, max_results: int) -> List[WebResult]:
        """Run one text search. Best first, `rank` assigned, junk dropped.

        Implements provider.SearchProvider.search -- see that docstring for
        the full contract this honours, in particular: [] means "ran, found
        nothing"; an exception means "could not run", and is NOT caught
        here. The caller (scripts/mcp_web_search_server.py) owns turning a
        raised exception into an MCP tool-level error, the same division of
        responsibility agents/gathering.py::search_worker has with
        tools/corpus_search.py.

        `max_results` is requested from DDGS AND enforced again in
        coerce_results. Belt and braces on purpose: DDGS's max_results is a
        hint its backend does not always honour exactly, and a provider
        returning more rows than asked for would silently widen the set
        scoring.rank_to_score interpolates across.
        """
        if not query or not query.strip():
            # An empty query is a caller bug, not a search that found
            # nothing -- but raising here would turn a harmless upstream
            # slip into a D-16 task failure. Log it and return empty: the
            # ladder then escalates to the next tier, which is the correct
            # outcome either way.
            log_event(logger, "websearch.empty_query", level=logging.WARNING)
            return []

        raw = list(self._client.text(
            query,
            region=self._region,
            safesearch=self._safesearch,
            max_results=max_results,
        ))
        results = coerce_results(raw, engine=ENGINE_TEXT, max_results=max_results)
        log_event(logger, "websearch.provider_returned",
                  engine=ENGINE_TEXT, raw_rows=len(raw), kept=len(results))
        return results


def build_provider(name: str, region: str = "wt-wt",
                   safesearch: str = "moderate",
                   timeout_seconds: float = 20.0) -> "DDGSProvider":
    """Construct the configured provider by name.

    CALLED BY   scripts/mcp_web_search_server.py, once, lazily, on the
                first real tool call.

    The single dispatch point a second backend gets added to -- so the
    server script never grows an if/elif chain over provider names, and a
    new provider is one entry here plus one sibling module.

    An unknown name raises ValueError with the known names listed, rather
    than silently falling back to the default. A silent fallback would mean
    WEB_SEARCH_PROVIDER=tavily on a build without Tavily support runs
    DuckDuckGo and reports nothing unusual -- the same silent-misconfig
    failure config.py::warn_on_likely_env_typos exists to eliminate.
    """
    known = {"ddgs"}
    key = (name or "").strip().lower()
    if key not in known:
        raise ValueError(
            f"Unknown WEB_SEARCH_PROVIDER={name!r}. "
            f"Known providers: {', '.join(sorted(known))}.")
    return DDGSProvider(region=region, safesearch=safesearch,
                        timeout_seconds=timeout_seconds)
