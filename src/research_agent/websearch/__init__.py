"""
research_agent.websearch — the web-search implementation, kept OUT of the
agent's own dependency graph.

WHAT THIS PACKAGE IS FOR (Phase 4, D-57):
    Everything needed to actually perform a web search and normalize its
    results. It is imported by scripts/mcp_web_search_server.py -- which
    runs in its OWN subprocess -- and by nothing in
    src/research_agent/agents/, orchestration/ or tools/.

    That separation is the whole design. The agent reaches web search the
    same way it reaches the corpus MCP server: over stdio, through
    tools/mcp_client.py, against a tool schema. So the agent process never
    imports ddgs, never makes an outbound search request itself, and never
    acquires a dependency on which engine is configured. Swapping DDGS for
    a keyed API is a new module in this package plus one .env setting.

    This is the same "the seam is the point" argument tools/corpus_search.py
    made about MCP, applied one level further out -- and it is why the
    Phase 4 requirement "do not couple the rest of the codebase to the
    chosen search implementation" is satisfied structurally, by the process
    boundary, rather than by convention.

MODULE MAP:
    provider.py        WebResult, SearchProvider (Protocol), registrable_domain,
                       coerce_results, as_payload -- the types and the wire shape.
    scoring.py         rank_to_score: ordinal rank -> a score the D-17
                       coverage gate can consume. Pure arithmetic.
    filtering.py       dedupe_by_url, cap_by_domain -- post-engine hygiene.
    ddgs_provider.py   DDGSProvider, build_provider. The ONLY module that
                       imports ddgs.

LAZY RE-EXPORT of DDGSProvider/build_provider (module-level __getattr__,
PEP 562): importing this package must NOT require ddgs to be installed.
Everything except ddgs_provider.py is pure-stdlib-plus-pydantic and is
imported eagerly below; the two ddgs-backed names are resolved only when
somebody actually asks for them. Without this, `from research_agent.
websearch import WebResult` in a test on a minimal install would fail on an
import the test never needed -- the same optional-dependency posture
langfuse/client.py already takes for the Langfuse SDK (D-35) and
tools/mcp_client.py takes for mcp (no module-level `import mcp` anywhere).
"""

from research_agent.websearch.filtering import cap_by_domain, dedupe_by_url
from research_agent.websearch.provider import (
    SearchProvider,
    WebResult,
    as_payload,
    coerce_results,
    registrable_domain,
)
from research_agent.websearch.scoring import rank_to_score

__all__ = [
    "SearchProvider",
    "WebResult",
    "as_payload",
    "build_provider",
    "cap_by_domain",
    "coerce_results",
    "dedupe_by_url",
    "rank_to_score",
    "registrable_domain",
    "DDGSProvider",
]


def __getattr__(name: str):
    """Resolve the two ddgs-backed names on first access (PEP 562).

    Python calls this function when an attribute is NOT found by normal
    lookup on the module -- so `websearch.WebResult` (imported eagerly
    above) never reaches here, while `websearch.DDGSProvider` does, and
    only then is ddgs_provider.py (and, inside DDGSProvider.__init__, ddgs
    itself) touched at all.

    The explicit AttributeError at the end preserves normal Python
    semantics for a genuine typo -- without it, `websearch.WebReslt` would
    return None instead of raising.
    """
    if name in ("DDGSProvider", "build_provider"):
        from research_agent.websearch import ddgs_provider
        return getattr(ddgs_provider, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
