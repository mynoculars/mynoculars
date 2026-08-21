"""
websearch/scoring.py — turning an engine's ORDINAL ranking into a score the
coverage gate can consume.

Purpose:
    A web search returns an ORDER, not a similarity. The agent's coverage
    rule (D-17: `e.score > settings.min_evidence_score`) and the retrieval
    ladder's quality floor (D-38, same predicate, passed in so the two can
    never drift) both need a NUMBER. This module is the entire conversion,
    and it is the only place that conversion happens.

Why not a flat constant (the obvious alternative, and what the Phase 4
advisory proposed):
    Stamping every hit with one fixed score (e.g. 0.6) tells every
    downstream consumer that result #1 and result #5 are equally good
    evidence. They are not -- ordering is the ONLY relevance signal a
    metasearch endpoint gives us, and discarding it at the boundary throws
    away the single most informative thing in the response. The compiler
    sees the evidence block sorted and scored; a flat score makes that
    block unsorted in fact while looking sorted.

Why the band is bounded at BOTH ends, and where the bounds come from:

    FLOOR (settings.web_search_min_score, default 0.60)
        Must be strictly greater than min_evidence_score (0.5), or the tier
        cannot mark a goal covered at all and the whole feature is inert --
        the exact failure mode min_evidence_score=0.0 was
        (config.py::warn_on_inert_coverage_gate) and that
        make_mcp_tool's `unscored_score` parameter exists to prevent in the
        other direction. Even the WORST result the engine returned is still
        a result the engine chose to return; it should clear the gate.

    CEILING (settings.web_search_max_score, default 0.75)
        Must stay well below the ~1.0 a document both retrieval legs agreed
        on reaches after tools/corpus_search.py's RRF_SQUASH. D-38's
        ordering invariant is that a real document always outranks weaker
        provenance; a web snippet that could score 0.95 would sit above
        genuinely fused corpus evidence in the compiler's context and
        invert that. 0.75 is deliberately below the corpus ceiling and
        above model_knowledge_score (0.60) -- a live retrieved snippet is
        better provenance than recollection, worse than a curated document.

Nothing here calls an LLM, touches the network, or reads Settings. It is
pure arithmetic on numbers passed in, so it is trivially unit-testable and
runs inside the search server's thread pool at no cost.
"""


def rank_to_score(rank: int, total: int, floor: float, ceiling: float) -> float:
    """Map a 1-based rank within `total` results onto [floor, ceiling].

    CALLED BY   scripts/mcp_web_search_server.py, once per result, before
                the payload crosses the MCP boundary. Scoring happens
                SERVER-side deliberately: ranking policy belongs with the
                search implementation, not smeared into the agent, so
                swapping providers cannot silently change what a score
                means on the agent side.

    Parameters:
        rank: 1-based position, best first. Values outside [1, total] are
            CLAMPED rather than rejected -- a provider that miscounts is a
            data problem, and refusing to score an otherwise usable result
            over an off-by-one would be a worse outcome than scoring it at
            an end of the band.
        total: how many results are being scored together. This is the
            SURVIVING count (see provider.coerce_results rule 2), not the
            raw response length.
        floor: score for the LAST result. Pass settings.web_search_min_score.
        ceiling: score for the FIRST result. Pass settings.web_search_max_score.

    Returns:
        float in [min(floor, ceiling), max(floor, ceiling)].

    Linear, not exponential or reciprocal-rank. Deliberate: over a window of
    5-10 results there is no evidence in this project for any particular
    curve, and an invented curve would be a precision claim the data does
    not support. Linear is the honest default, it is obvious at a glance
    what any given rank scores, and the band is narrow enough (0.60-0.75)
    that the curve's shape changes little. Revisit only with measurements,
    the same measure-before-enforcing discipline D-54 applied to the call
    budget.
    """
    # Guard an inverted band rather than silently producing a descending
    # scale that runs the wrong way. Callers get the band they meant, and
    # config.py's own validation warns about the misconfiguration
    # separately -- this function still has to behave sanely if it is
    # called before anyone reads that warning.
    lo, hi = (floor, ceiling) if floor <= ceiling else (ceiling, floor)

    if total <= 1:
        # A single result has no ordering information at all. Awarding it
        # the ceiling is the only defensible choice: it IS the engine's
        # best (and only) answer. Averaging the band instead would penalize
        # a query that had exactly one good hit.
        return hi

    clamped = min(max(int(rank), 1), int(total))
    # (clamped - 1) / (total - 1): 0.0 for rank 1, 1.0 for rank == total.
    fraction = (clamped - 1) / (total - 1)
    return hi - fraction * (hi - lo)
