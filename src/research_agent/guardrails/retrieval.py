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
