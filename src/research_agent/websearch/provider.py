"""
websearch/provider.py — the search-engine seam: one result type, one protocol.

Purpose:
    Define WHAT a web search returns, without saying WHO performs it. Every
    other module in this package, and scripts/mcp_web_search_server.py, is
    written against these two names only -- never against a concrete engine.

Responsibilities:
    - WebResult: one search hit, normalized. The single shape every provider
      must produce and every consumer may rely on.
    - SearchProvider: the callable contract a provider implements.
    - registrable_domain(): URL -> the domain used for attribution and for
      the per-domain diversity cap (filtering.py).

Design decision -- why a Protocol and not a base class:
    Same reasoning agents/gathering.py::ToolFn already uses for the retrieval
    tool seam: the consumer needs a SHAPE, not an inheritance relationship. A
    test fake is then just a small class (or even a function object) with a
    search() method -- no import of this module required, no base class to
    subclass, nothing to keep in sync. typing.Protocol makes that duck typing
    explicit and checkable rather than merely conventional.

Design decision -- why this package is never imported by the agent:
    tools/mcp_client.py talks to a web-search MCP SERVER over stdio; the
    server subprocess is what imports this package (see
    scripts/mcp_web_search_server.py). The agent process therefore never
    imports ddgs, never opens an outbound HTTP connection of its own, and
    never grows a dependency on whichever engine is configured. Swapping
    DDGS for a keyed API is a change to ddgs_provider.py plus one setting --
    zero lines in src/research_agent/agents/, orchestration/ or tools/.
    This is the same "the seam is the point" argument tools/corpus_search.py
    made for MCP, applied one level out.

Python mechanics used here, if any of this is new:
    typing.Protocol
        A class you never inherit from. Declaring `class SearchProvider
        (Protocol)` with a `search(...)` method means "anything with a
        compatible search() method counts as a SearchProvider" -- checked by
        a type checker (mypy/pyright), not enforced at runtime. Contrast with
        a normal base class, which requires the implementer to import and
        subclass it.
    @runtime_checkable
        Opts the Protocol into `isinstance(x, SearchProvider)` working at
        runtime. It only checks that the METHOD NAMES exist, never their
        signatures -- so it is useful for a clear error message, never as a
        substitute for a real test.
"""

from typing import List, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field


class WebResult(BaseModel):
    """One normalized web search hit.

    Deliberately NOT research_agent.state.Evidence. Evidence is the agent's
    domain model and carries agent concerns (task_key, goal_id, volatility,
    hedge_specific) that a search engine knows nothing about. This package
    runs in a SEPARATE PROCESS from the agent and must not need those; the
    translation WebResult -> Evidence happens on the agent side, in
    tools/mcp_client.py, from the MCP payload. Keeping the two types apart
    is what lets the server process avoid importing the agent's state model
    at all.

    model_config = ConfigDict(extra="forbid") for the same reason every
    model in state.py has it (D-29): a typo'd field name fails LOUDLY at
    construction rather than silently creating an object with a missing
    field and a useless extra attribute.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str
    # 1-based, as the engine ranked it. 1-based rather than 0-based because
    # this number is reported to a human in logs and telemetry ("result 3 of
    # 5"), and because scoring.rank_to_score's formula reads more clearly
    # against the ordinal a person would say out loud.
    rank: int = Field(ge=1)
    # Which backend produced this hit ("ddg_text", "ddg_news", ...). Carried
    # so a mixed-provider future, or a debug trace today, can attribute a
    # result to the engine that found it without inferring it from the URL.
    engine: str

    @property
    def domain(self) -> str:
        """The registrable domain, for attribution and the diversity cap.

        A property rather than a stored field: it is derived entirely from
        `url` and storing it would create a second source of truth that can
        drift (the exact defect the DDD reviews in other projects call out).
        Consumers that need it in a serialized payload compute it at the
        boundary -- see scripts/mcp_web_search_server.py.
        """
        return registrable_domain(self.url)


@runtime_checkable
class SearchProvider(Protocol):
    """What every concrete search backend must offer.

    IMPLEMENTED BY  websearch/ddgs_provider.py::DDGSProvider (the only
                    implementation shipped today), and by whatever fake a
                    test constructs.
    CALLED BY       scripts/mcp_web_search_server.py, once per MCP
                    web_search tool call.
    """

    def search(self, query: str, max_results: int) -> List[WebResult]:
        """Run one query, return at most `max_results` hits, best first.

        CONTRACT, and every clause of it is load-bearing:

        - Returns a list ordered BEST FIRST, with `rank` set to each item's
          1-based position in that order. scoring.rank_to_score depends on
          this ordering being real, not incidental.
        - Returns [] for "the engine ran and found nothing". That is a
          normal, non-exceptional outcome -- the same way an empty corpus
          result is normal in tools/corpus_search.py.
        - RAISES for "the engine could not run" (network failure, throttle,
          malformed response). Deliberately does not catch: the caller
          (the MCP server) owns turning a failure into a tool-level error,
          exactly as agents/gathering.py::search_worker owns turning a tool
          exception into a D-16 failure record rather than the tool doing
          it. A provider that swallows its own errors and returns [] makes
          "no results" and "broken" indistinguishable, which is the failure
          mode min_evidence_score=0.0 and the old hardcoded MCP score=1.0
          both were.
        """
        ...


def registrable_domain(url: str) -> str:
    """Return the display domain for a URL, or "" if it has none.

    CALLED BY   WebResult.domain, and websearch/filtering.py's per-domain
                diversity cap.

    Deliberately simple: lower-cased netloc with any leading "www." and any
    ":port" removed. This is NOT a Public Suffix List implementation -- it
    will report "co.uk" style multi-label suffixes as, e.g.,
    "bbc.co.uk" (correct here, since the netloc already carries the full
    host) but will not collapse "news.bbc.co.uk" and "www.bbc.co.uk" to one
    registrable domain. That distinction matters for a real PSL consumer
    (cookie scoping, certificate policy); it does not matter for either use
    here -- an attribution label a reader sees, and a "don't take five hits
    from one place" heuristic. Adding a PSL dependency (tldextract, which
    downloads and caches a suffix list at runtime) to get the last few
    percent of correctness on a heuristic is not a trade worth making.

    Returns "" rather than raising on an unparseable or non-HTTP URL: a
    weird URL is a data problem in a third-party response, not a reason to
    fail a whole search.
    """
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        # urlparse raises ValueError on a handful of genuinely malformed
        # inputs (e.g. an IPv6 literal with an unmatched bracket).
        return ""
    host = netloc.split("@")[-1]        # strip any user:pass@ prefix
    host = host.split(":")[0]           # strip any :port suffix
    host = host.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def as_payload(result: WebResult, score: float) -> dict:
    """Flatten one WebResult + its computed score into the MCP wire shape.

    CALLED BY   scripts/mcp_web_search_server.py, immediately before
                returning results over the MCP protocol.
    WHY IT LIVES HERE, not in the server script: this dict IS the contract
    between the server process and tools/mcp_client.py in the agent
    process. Defining it beside WebResult keeps the two definitions
    adjacent, so a field added to one is visibly a change to the other --
    the same reasoning scripts/mcp_corpus_server.py's docstring gives for
    keeping its list[str] output shape pinned to what mcp_client.py parses.

    `domain` is materialized here (not left as a property) precisely
    because this crosses a process boundary: the agent side receives plain
    JSON and has no WebResult object to ask.
    """
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "rank": result.rank,
        "engine": result.engine,
        "domain": result.domain,
        "score": score,
    }


def coerce_results(raw: List[dict], engine: str,
                   max_results: Optional[int] = None) -> List[WebResult]:
    """Turn a provider's raw dicts into ranked WebResults, dropping junk.

    CALLED BY   websearch/ddgs_provider.py, and available to any future
                provider -- the rank assignment and the "is this row even
                usable" test are provider-independent and should not be
                re-implemented per backend.

    Rules, each closing a real class of third-party response defect:

    1. A row with no URL, or no title AND no snippet, is DROPPED. There is
       nothing to cite and nothing to read; carrying it would produce an
       Evidence item with empty content that still occupies a slot in the
       compile prompt.
    2. `rank` is assigned from the position in the SURVIVING list, not the
       raw list. A dropped row must not leave a gap in the ranking, or
       scoring.rank_to_score's linear interpolation is computed against a
       total that does not match the items it is scoring.
    3. Whitespace is collapsed in title/snippet. Scraped snippets routinely
       carry newlines and runs of spaces, which read badly inside the
       <evidence> block the compiler sees.
    """
    out: List[WebResult] = []
    for row in raw:
        url = (row.get("href") or row.get("url") or "").strip()
        title = " ".join((row.get("title") or "").split())
        snippet = " ".join((row.get("body") or row.get("snippet") or "").split())
        if not url or not (title or snippet):
            continue
        out.append(WebResult(title=title, url=url, snippet=snippet,
                             rank=len(out) + 1, engine=engine))
        if max_results is not None and len(out) >= max_results:
            break
    return out
