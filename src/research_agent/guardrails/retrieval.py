"""
guardrails/retrieval.py — the two-stage relevance floor (P2-01).

CALLED BY   retrieval/hybrid.py::HybridRetriever.search (the pre-fusion
            floor) and agents/gathering.py::progress_checker_node (the
            post-fusion coverage gate).

Both checks existed before this move — this only relocates them from two
separate inline comparisons into one place, so the two DIFFERENT operators
below (>= for one, strict > for the other) stay documented next to each
other instead of silently drifting apart if someone edits one site without
noticing the other exists. See each function's own docstring for why the
operators differ; they are not a typo of each other.
"""

from typing import List

from research_agent.retrieval.terms import distinctive_terms
from research_agent.state import Evidence

# S-7: moved from prompts/templates.py, where it lived despite being a
# retrieval-scoring constant, not a prompt. Not approximate -- exactly
# 0.5, for any query, regardless of how relevant the document actually
# was, because RRF scores rank position rather than similarity. A score
# at or below this ceiling means no document that both retrieval legs
# agreed on, which is the strongest cheap signal available that the
# corpus may not cover the goal. tests/unit/test_prompts.py still
# cross-checks this against RRF_K/RRF_SQUASH, so a change to either
# fails the suite rather than silently rotting this threshold.
SINGLE_LEG_SCORE_CEILING = 0.5


def passes_similarity_floor(similarity: float, floor: float) -> bool:
    """The DENSE-leg floor, applied BEFORE fusion (retrieval/hybrid.py).

    A dense index always returns its k nearest neighbours, however far
    away they actually are — "nearest" is not "relevant." Without this
    floor, an out-of-domain query could never produce zero evidence; it
    would always get back its k closest (but possibly meaningless) hits.

    Inclusive (>=): a hit exactly AT the floor passes. This matches the
    ORIGINAL comparison in retrieval/hybrid.py unchanged by this move —
    the pre-fusion floor was never the site of the exact-boundary bug
    passes_evidence_gate below exists to close; only the POST-fusion gate
    was.
    """
    return similarity >= floor


def passes_evidence_gate(score: float, floor: float) -> bool:
    """The POST-fusion coverage gate (agents/gathering.py::progress_checker_node).

    Strict greater-than, not >=. A score landing EXACTLY on the floor is,
    under single-leg RRF fusion, indistinguishable from "ranked first
    among whatever came back" — it carries no information about actual
    relevance. Requiring the score to EXCEED the floor, not merely meet
    it, closes that specific loophole: a rank-0 hit from a single
    surviving retrieval leg squashes to exactly 0.5 under this
    codebase's RRF_SQUASH constant, and a >= comparison let that exact
    value through every time (see progress_checker_node's own docstring
    for the live trace that found this).
    """
    return score > floor


def has_grounded_evidence(goal_id: str, goal_terms: set,
                          evidence: List[Evidence], min_score: float) -> bool:
    """True if a REAL DOCUMENT, actually about this goal, covers it (G2/D-47).

    Three conjuncts, none of them redundant:
      - source in ("corpus", "mcp") -- a document, not recollection and not
        a web snippet (D-57: web COVERS but never GROUNDS).
      - score above the coverage floor -- passes_evidence_gate above, so
        this and the post-fusion coverage gate can never disagree.
      - shares distinctive vocabulary with the goal's own description --
        the D-39 topical gate. Without it, an off-topic corpus hit that
        cleared the floor by cross-leg agreement counted as grounding for
        a goal it had nothing to do with (observed live, run p205.132).

    Single source of truth for this predicate (M-1): previously
    duplicated between agents/gathering.py and an inline block in
    agents/compilation.py::telemetry_node with a hardcoded `> 0.5`
    instead of `min_score` -- the two metrics silently disagreed
    whenever MIN_EVIDENCE_SCORE was changed from its default. Moved here
    (D-59 originally extracted it to module level in gathering.py; this
    relocates it next to passes_evidence_gate, the gate it reuses) so
    every caller shares one comparison and one floor.
    """
    return any(
        e.goal_id == goal_id and e.source in ("corpus", "mcp")
        and passes_evidence_gate(e.score, min_score)
        and (not goal_terms or goal_terms & distinctive_terms(e.content))
        for e in evidence)
