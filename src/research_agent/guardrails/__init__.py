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
    hedging.py     Guardrail G3's enforcement half (P205.135 follow-up).
                   UNLIKE every other file above, this is NOT a relocation
                   of existing code -- it is new. Recorded here anyway,
                   deliberately, rather than left undocumented: it is the
                   exact same SHAPE of check as citations.py (deterministic
                   post-processing of the compiled report against
                   state.evidence, called from the same compiler_node call
                   site, right next to clean_citations), so it belongs in
                   this package on architectural grounds even though the
                   package docstring's "only moved code" rule doesn't
                   technically cover it. See hedging.py's own docstring
                   for why it exists and what it does.
    dedup.py       Collapses byte-identical evidence per goal before it
                   enters a prompt (FIX-5). Also new rather than moved,
                   and recorded here for the same reason hedging.py is:
                   it is deterministic post-retrieval shaping applied at
                   a fixed point, from the same compiler_node call site,
                   and it is the corpus/MCP counterpart to a guard this
                   codebase already accepted for web results
                   (websearch/filtering.py::cap_by_domain -- repeated
                   hits read to the compiler as independent sources
                   agreeing). Touches only the PROMPT's copy of the
                   evidence, never ResearchState.evidence, so no
                   telemetry figure moves.

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
