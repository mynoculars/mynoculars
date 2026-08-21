"""
websearch/filtering.py — post-engine result hygiene: dedupe, domain cap.

Purpose:
    A metasearch endpoint will happily return five hits from one site. To
    the compiler, five items saying the same thing under five different
    titles reads as CORROBORATION -- five independent sources agreeing --
    when it is one source repeated. That is a correctness problem, not a
    tidiness one, and it is invisible in telemetry because
    `web_search_results: 5` looks identical either way.

Responsibilities:
    - dedupe_by_url(): drop exact repeats of the same URL.
    - cap_by_domain(): allow at most N hits per registrable domain.

Both preserve INPUT ORDER and RE-RANK the survivors, for the same reason
provider.coerce_results does (see rule 2 there): a gap in the ranking
breaks scoring.rank_to_score's interpolation, which is computed against the
count of the items actually being scored.

Deliberately NOT here: near-duplicate detection by snippet similarity. That
needs a threshold nobody in this project has measured, and a wrong
threshold silently discards a genuinely distinct source -- strictly worse
than keeping a near-duplicate, which merely wastes a slot. Same
measure-before-enforcing posture as D-54.
"""

from typing import List

from research_agent.websearch.provider import WebResult


def _rerank(results: List[WebResult]) -> List[WebResult]:
    """Renumber `rank` to 1..len(results), preserving order.

    model_copy(update={...}) rather than mutating in place: WebResult is a
    Pydantic model used as a value object, and the caller's list may be
    shared. Same idiom tools/retrieval_chain.py already uses when it re-tags
    reformulated-retry evidence onto the original task key.
    """
    return [r.model_copy(update={"rank": i + 1}) for i, r in enumerate(results)]


def dedupe_by_url(results: List[WebResult]) -> List[WebResult]:
    """Keep the first occurrence of each URL; drop later exact repeats.

    Exact URL match only, deliberately -- no normalization of trailing
    slashes, query strings or fragments. Two URLs differing only by a
    tracking parameter really are the same page, but two differing by a
    query string often are NOT (a search results page vs. an article), and
    guessing wrong drops real evidence. The cheap, certain win is taken;
    the expensive, uncertain one is left.
    """
    seen = set()
    out: List[WebResult] = []
    for r in results:
        if r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
    return _rerank(out)


def cap_by_domain(results: List[WebResult], max_per_domain: int) -> List[WebResult]:
    """Allow at most `max_per_domain` results from any one domain.

    CALLED BY   scripts/mcp_web_search_server.py, after dedupe_by_url and
                before scoring.

    Because input order is best-first, the survivors from a capped domain
    are always its HIGHEST-ranked ones -- the cap trims the tail of a
    dominant site, never its best hit.

    max_per_domain <= 0 disables the cap entirely and returns the input
    unchanged (still re-ranked, so callers get a consistent shape either
    way). That is the documented way to reproduce uncapped behaviour
    deliberately, matching how min_similarity=0.0 is the documented way to
    reproduce pre-P2-01 retrieval.

    A result whose URL yields no parseable domain (provider.
    registrable_domain returns "") is NEVER capped -- it is passed through.
    Grouping every unparseable URL together under one empty-string key
    would let one malformed row suppress an unrelated malformed row, which
    is a filter doing damage on data it does not understand.
    """
    if max_per_domain <= 0:
        return _rerank(list(results))
    counts: dict = {}
    out: List[WebResult] = []
    for r in results:
        domain = r.domain
        if not domain:
            out.append(r)
            continue
        if counts.get(domain, 0) >= max_per_domain:
            continue
        counts[domain] = counts.get(domain, 0) + 1
        out.append(r)
    return _rerank(out)
