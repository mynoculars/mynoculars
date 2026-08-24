"""
retrieval/terms.py -- the shared "what counts as a distinctive term"
predicate (S-7), moved out of tools/retrieval_chain.py.

Purpose:
    distinctive_terms() is the single topical-overlap primitive this
    codebase uses to decide "is X actually about the same subject as Y" --
    reused by the retrieval ladder's own sufficiency gate, the D-39/D-47
    grounding predicate (guardrails/retrieval.py), and the D-59/D-64
    Sources-section topical gate (guardrails/sources.py). One
    implementation means "on topic" cannot come to mean three different
    things in three files.

Why moved here:
    distinctive_terms was previously underscore-prefixed
    (`_distinctive_terms`) and defined inside tools/retrieval_chain.py, a
    single retrieval TIER implementation -- but imported by three other
    modules as if it were public API. The underscore said "do not import
    this"; three modules did. This gives it a home matching what it
    actually is: a small, general retrieval primitive, not tier-specific
    logic. tools/retrieval_chain.py imports it back from here like every
    other caller.
"""

# Query scaffolding that carries no topical signal. Shared by every
# caller of distinctive_terms() below -- "comparison" appearing in both a
# query about armies and a document about Redis is not evidence of
# anything.
FILLER = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with",
    "to", "vs", "versus", "between", "comparison", "compare",
    "comparative", "analysis", "analyze", "evaluate", "evaluation",
    "assess", "assessment", "examine", "including", "their", "both",
    "its", "recent", "current", "overview", "study", "report",
    "data", "rate", "rates", "scale", "system", "systems", "features",
}


def distinctive_terms(text: str) -> set:
    """Lower-cased content words carrying topical signal, minus scaffolding.

    Pure and cheap: this runs inside every parallel worker, on every
    tier, so it must not allocate much or call anything.

    Three rules, each closing a live-observed hole in the gate that
    consumes this (retrieval_chain's `_sufficient`, and `corpus_recall`/
    `has_grounded_evidence` and the Sources topical gate, which
    deliberately reuse this same function so none of them can disagree):

    1. Words longer than three characters are kept, as before.
    2. SHORT ALL-CAPS TOKENS ARE KEPT TOO. The old length-only rule threw
       away exactly the tokens carrying the most topical signal in this
       project's real traffic: "GDP growth India US 2020-2023" retained
       neither GDP nor US, and "Indian Army ... Chinese PLA" dropped PLA
       -- the single most distinctive word in the query. An acronym is
       the opposite of filler, and length is the wrong proxy for it.
    3. BARE NUMBERS ARE DROPPED. A standalone number is a weak topical
       signal that travels in pairs: "2020-2023" contributed TWO terms,
       so any off-topic document mentioning the same two years cleared a
       two-term overlap bar on years alone. Mixed alphanumerics (pm10,
       t90, 155mm) are real terms and are kept -- only all-digit tokens
       go.
    """
    out = set()
    for word in "".join(
            c if c.isalnum() else " " for c in text).split():
        low = word.lower()
        if low in FILLER or low.isdigit():
            continue
        if len(low) > 3 or (len(low) >= 2 and word.isupper()):
            out.add(low)
    return out
