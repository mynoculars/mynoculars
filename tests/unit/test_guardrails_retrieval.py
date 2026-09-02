"""
tests/unit/test_guardrails_retrieval.py -- guardrails/retrieval.py, the
two-stage relevance floor (P2-01) and the M-1 grounding predicate.

Covers: passes_similarity_floor's inclusive boundary, passes_evidence_gate's
strict boundary (the exact-0.5 loophole that motivated the >, not >=),
has_grounded_evidence's three conjuncts independently (source, score,
topical overlap), and SINGLE_LEG_SCORE_CEILING's cross-check against the
RRF constants it is derived from. Previously untested at the unit level --
covered only indirectly through gathering/telemetry integration paths.
"""

from research_agent.guardrails.retrieval import (SINGLE_LEG_SCORE_CEILING,
                                                  has_grounded_evidence,
                                                  passes_evidence_gate,
                                                  passes_similarity_floor)
from research_agent.state import Evidence


def _ev(goal_id, content, score, source="corpus"):
    return Evidence(task_key=f"k-{goal_id}-{content[:8]}", goal_id=goal_id,
                    source=source, content=content, score=score)


# ---------------------------------------------------------------------------
# passes_similarity_floor -- inclusive (>=), the pre-fusion dense-leg floor
# ---------------------------------------------------------------------------


def test_similarity_floor_passes_a_hit_above_the_floor():
    assert passes_similarity_floor(0.61, 0.60) is True


def test_similarity_floor_passes_a_hit_exactly_on_the_floor():
    """Inclusive by design -- unlike passes_evidence_gate below, this is
    NOT the site of the exact-boundary bug; the original comparison in
    retrieval/hybrid.py was always >=."""
    assert passes_similarity_floor(0.60, 0.60) is True


def test_similarity_floor_rejects_a_hit_below_the_floor():
    assert passes_similarity_floor(0.59, 0.60) is False


# ---------------------------------------------------------------------------
# passes_evidence_gate -- strict (>), the post-fusion coverage gate
# ---------------------------------------------------------------------------


def test_evidence_gate_passes_a_score_above_the_floor():
    assert passes_evidence_gate(0.51, 0.5) is True


def test_evidence_gate_rejects_a_score_exactly_on_the_floor():
    """The exact loophole this function exists to close: a rank-0 hit
    from a single surviving retrieval leg squashes to exactly
    SINGLE_LEG_SCORE_CEILING (0.5) under this codebase's RRF math, and
    that value must NOT count as coverage on its own."""
    assert passes_evidence_gate(0.5, 0.5) is False


def test_evidence_gate_rejects_a_score_below_the_floor():
    assert passes_evidence_gate(0.3, 0.5) is False


# ---------------------------------------------------------------------------
# has_grounded_evidence -- the M-1 predicate, three independent conjuncts
# ---------------------------------------------------------------------------


def test_grounded_evidence_true_for_a_scoring_on_topic_document():
    evidence = [_ev("g1", "Redis throughput benchmarks", 0.9, source="corpus")]
    assert has_grounded_evidence("g1", {"redis", "throughput"}, evidence, 0.5) is True


def test_grounded_evidence_false_when_score_does_not_clear_the_floor():
    """Uses the SAME strict-> comparison as passes_evidence_gate -- an
    item scoring exactly at min_score must not count, for the identical
    reason that function's own test above pins."""
    evidence = [_ev("g1", "Redis throughput benchmarks", 0.5, source="corpus")]
    assert has_grounded_evidence("g1", {"redis", "throughput"}, evidence, 0.5) is False


def test_grounded_evidence_false_for_web_source_even_with_a_good_score():
    """D-57: web COVERS (counts toward recall) but never GROUNDS (does
    not count toward corpus_recall/grounded_score) -- web is deliberately
    excluded from the source conjunct regardless of score or topicality."""
    evidence = [_ev("g1", "Redis throughput benchmarks", 0.95, source="web")]
    assert has_grounded_evidence("g1", {"redis", "throughput"}, evidence, 0.5) is False


def test_grounded_evidence_false_for_model_source():
    """source="model" (the model's own recollection) is not a document
    either -- same exclusion as web, for the same reason."""
    evidence = [_ev("g1", "Redis throughput benchmarks", 0.95, source="model")]
    assert has_grounded_evidence("g1", {"redis", "throughput"}, evidence, 0.5) is False


def test_grounded_evidence_false_for_an_off_topic_document_despite_a_good_score():
    """D-39: the topical gate. A corpus hit that cleared the score floor
    by cross-leg agreement but shares no distinctive vocabulary with the
    goal must not count as grounding for that goal (observed live, run
    p205.132 -- Redis-monitoring corpus hits counted toward an
    India-vs-US comparison goal)."""
    evidence = [_ev("g1", "Redis cluster monitoring dashboards", 0.9, source="corpus")]
    assert has_grounded_evidence("g1", {"india", "military"}, evidence, 0.5) is False


def test_grounded_evidence_empty_goal_terms_skips_the_topical_gate():
    """goal_terms can be empty (a goal description with nothing but
    filler words) -- the topical check is `not goal_terms or ...`, so an
    empty set does not reject every candidate outright."""
    evidence = [_ev("g1", "some content here", 0.9, source="corpus")]
    assert has_grounded_evidence("g1", set(), evidence, 0.5) is True


def test_grounded_evidence_ignores_evidence_for_a_different_goal():
    evidence = [_ev("g2", "Redis throughput benchmarks", 0.9, source="corpus")]
    assert has_grounded_evidence("g1", {"redis"}, evidence, 0.5) is False


def test_grounded_evidence_false_on_empty_evidence_list():
    assert has_grounded_evidence("g1", {"redis"}, [], 0.5) is False


def test_grounded_evidence_true_if_any_one_item_qualifies():
    """Only one qualifying item is needed even when other items for the
    same goal fail every conjunct."""
    evidence = [
        _ev("g1", "Redis cluster monitoring", 0.9, source="corpus"),  # off-topic
        _ev("g1", "Redis throughput benchmarks", 0.3, source="corpus"),  # low score
        _ev("g1", "Redis throughput benchmarks", 0.9, source="web"),  # wrong source
        _ev("g1", "Redis throughput benchmarks", 0.9, source="corpus"),  # qualifies
    ]
    assert has_grounded_evidence("g1", {"redis", "throughput"}, evidence, 0.5) is True


# ---------------------------------------------------------------------------
# SINGLE_LEG_SCORE_CEILING -- drift guard
# ---------------------------------------------------------------------------


def test_single_leg_score_ceiling_matches_the_rrf_derivation():
    """SINGLE_LEG_SCORE_CEILING must equal the actual score a rank-0 hit
    from a single retrieval leg gets after RRF_SQUASH -- a change to
    either RRF_K or RRF_SQUASH without updating this constant would
    silently reopen the exact-0.5 loophole passes_evidence_gate closes."""
    from research_agent.retrieval.hybrid import RRF_K
    from research_agent.tools.corpus_search import RRF_SQUASH

    assert SINGLE_LEG_SCORE_CEILING == min(1.0, (1 / RRF_K) * RRF_SQUASH)


# ---------------------------------------------------------------------------
# D-164 -- MCP corroborates, it does not ground
# ---------------------------------------------------------------------------


def test_mcp_evidence_never_counts_as_grounding():
    """The "mcp" arm of this predicate was dead code and said otherwise.

    The corpus MCP server's tool schema returns text only -- it discards
    the score its own HybridRetriever computed -- so assembly.py stamps
    every item with `unscored_score=min_evidence_score`, exactly the
    floor, and passes_evidence_gate is a strict `>`. The conjunction
    could never be true. Live: run p205.290-check carried 21 mcp evidence
    items and `tier_answers {"web": 7}`.

    Until the server sends the scores it already has, the honest reading
    is that tier 3 corroborates in the prompt and grounds nothing."""
    from research_agent.guardrails.retrieval import (GROUNDING_SOURCES,
                                                     has_grounded_evidence)
    from research_agent.state import Evidence, Volatility

    assert "mcp" not in GROUNDING_SOURCES

    terms = {"redis", "memcached", "session"}
    content = "Redis and Memcached both serve session caching workloads."
    for score in (0.5, 0.75, 0.99):
        item = Evidence(task_key="k", goal_id="g1", source="mcp",
                        content=content, score=score,
                        volatility=Volatility.SEMI_STABLE)
        assert has_grounded_evidence("g1", terms, [item], 0.5) is False, score


def test_a_corpus_item_with_the_same_shape_still_grounds():
    """The removal must be about PROVENANCE, not about having broken the
    predicate for everyone."""
    from research_agent.guardrails.retrieval import has_grounded_evidence
    from research_agent.state import Evidence, Volatility

    item = Evidence(task_key="k", goal_id="g1", source="corpus",
                    content="Redis and Memcached both serve session caching.",
                    score=0.75, volatility=Volatility.SEMI_STABLE)

    assert has_grounded_evidence("g1", {"redis", "memcached"}, [item], 0.5) is True
