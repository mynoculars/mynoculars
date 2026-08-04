"""
guardrails/ — deterministic correctness checks pulled out of the nodes and
call sites that use them, so each check has exactly one implementation
regardless of how many places need it.

Scope, deliberately narrow:
    Only checks that were ALREADY implemented somewhere in this codebase,
    duplicated or scattered, get moved here. This package is a refactor of
    existing behaviour, not a place to add new guardrail categories —
    doing that would need its own design discussion, the same way any new
    D-xx decision does.

What lives here today:
    retrieval.py   the two-stage relevance thresholds (P2-01) —
                   previously duplicated as inline comparisons in
                   retrieval/hybrid.py and agents/gathering.py, with
                   DIFFERENT operators (>= for the pre-fusion floor, >
                   for the post-fusion gate) that a future edit could
                   easily blur together by accident if left inline twice.
    citations.py   the deterministic citation repair pass (D-40/D-43/D-45),
                   moved from agents/compilation.py unchanged — it was
                   already a single, self-contained function; this only
                   changes where it lives, not what it does.
    fencing.py     the prompt-injection fence for retrieved evidence text
                   (D-18's mitigation), moved from prompts/templates.py.

What deliberately does NOT live here, and why:
    State reducers (state.py) — LangGraph requires these as Annotated[...]
    type hints directly on the state model's own field declarations; there
    is no way to define a field's merge function anywhere else.

    The escalation-check-first ordering inside each routing function
    (orchestration/graph.py) — this only works because it's evaluated
    against LIVE state at the exact moment routing decides, as the first
    statement in that function. Pulling it out would either duplicate the
    state read or introduce a timing gap between when a guardrail checks
    and when routing acts on it.

    Per-store degradation handling (storage/qdrant_store.py,
    storage/opensearch_store.py, retrieval/hybrid.py::_safe_leg) — each
    try/except is tied to that specific client's own failure semantics,
    which genuinely differ per store. The PATTERN is shared; the code
    correctly is not.
"""
