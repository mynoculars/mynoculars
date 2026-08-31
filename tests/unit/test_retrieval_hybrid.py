"""
tests/unit/test_retrieval_hybrid.py — retrieval/hybrid.py's
HybridRetriever.

Covers ONLY the P2-07 follow-up: per-call retrieval-boundary telemetry
(retrieval_dense_calls/retrieval_keyword_calls/retrieval_leg_unavailable),
implemented via threading.local() so concurrent search_worker dispatch
under LangGraph can't race on shared mutable state — proven here with a
direct multi-thread test, not just asserted. Does NOT cover the actual
dense+BM25 fusion/RRF logic itself or the min_similarity floor (see
tests/integration/test_hitl_escalation.py's
test_min_similarity_floor_drops_low_relevance_dense_hits for that one,
kept alongside the HITL scenario it was written to explain).
"""

import concurrent.futures

from research_agent.retrieval.hybrid import HybridRetriever, rrf_fuse


class _FakeLeg:
    """Minimal fake store: only .search() and .available — enough to drive
    HybridRetriever without touching real Qdrant/OpenSearch."""

    def __init__(self, hits=None, available=True):
        self._hits = hits or []
        self.available = available

    def search(self, query, top_k):
        return list(self._hits)


def test_hybrid_retriever_drain_counts_tracks_calls_and_resets():
    retriever = HybridRetriever(_FakeLeg(), _FakeLeg(), min_similarity=0.0)
    retriever.search("q", top_k=3)
    retriever.search("q2", top_k=3)
    counts = retriever.drain_counts()
    assert counts["retrieval_dense_calls"] == 2
    assert counts["retrieval_keyword_calls"] == 2
    assert counts["retrieval_leg_unavailable"] == 0
    # Draining resets — nothing left for a second call on the same thread.
    assert retriever.drain_counts() == {}


def test_hybrid_retriever_counts_unavailable_legs_not_empty_results():
    # dense unavailable, keyword available but legitimately returns nothing
    # — these are two DIFFERENT situations and must be counted differently.
    retriever = HybridRetriever(_FakeLeg(available=False),
                                _FakeLeg(hits=[], available=True),
                                min_similarity=0.0)
    retriever.search("obscure query", top_k=3)
    counts = retriever.drain_counts()
    assert counts["retrieval_leg_unavailable"] == 1  # only dense, not keyword


# ---------------------------------------------------------------------------
# Guardrail G1: run-level relevance-floor drop-ratio telemetry
# ---------------------------------------------------------------------------


def test_g1_counts_dense_candidates_and_floor_drops():
    """Regression target: run p205.131-check, where min_similarity=0.55
    dropped EVERY dense hit, every query, all run -- invisible outside
    raw debug logs because nothing aggregated it. drain_counts() must
    expose both the raw drop count and the total candidate count, so a
    caller can compute the ratio rather than just a bare (uninformative
    on its own) number."""
    dense_hits = [{"id": "d1", "content": "x", "similarity": 0.4},
                  {"id": "d2", "content": "y", "similarity": 0.3}]
    retriever = HybridRetriever(_FakeLeg(hits=dense_hits),
                                _FakeLeg(hits=[]), min_similarity=0.55)
    retriever.search("q", top_k=3)
    counts = retriever.drain_counts()
    assert counts["retrieval_dense_candidates"] == 2
    assert counts["retrieval_dropped_by_floor"] == 2  # both below 0.55


def test_g1_reports_zero_drops_when_hits_clear_the_floor():
    dense_hits = [{"id": "d1", "content": "x", "similarity": 0.9}]
    retriever = HybridRetriever(_FakeLeg(hits=dense_hits),
                                _FakeLeg(hits=[]), min_similarity=0.55)
    retriever.search("q", top_k=3)
    counts = retriever.drain_counts()
    assert counts["retrieval_dense_candidates"] == 1
    assert counts.get("retrieval_dropped_by_floor", 0) == 0


def test_g1_contributes_no_floor_counters_when_floor_disabled():
    """min_similarity=0.0 keeps the pre-P2-01 no-floor behaviour and,
    with it, must not even bump these counters -- a caller aggregating
    retrieval_dense_candidates across many runs should never see a
    misleading 0/0 masquerading as a real ratio."""
    retriever = HybridRetriever(_FakeLeg(hits=[{"id": "d1", "content": "x",
                                                "similarity": 0.1}]),
                                _FakeLeg(hits=[]), min_similarity=0.0)
    retriever.search("q", top_k=3)
    counts = retriever.drain_counts()
    assert "retrieval_dense_candidates" not in counts
    assert "retrieval_dropped_by_floor" not in counts


def test_hybrid_retriever_counts_are_thread_local():
    """Direct verification of the thread-safety claim: N threads, each
    doing exactly one search()+drain() pair, must each see ONLY their own
    call — never another thread's count leaking in, and never losing their
    own to a race."""
    retriever = HybridRetriever(_FakeLeg(), _FakeLeg(), min_similarity=0.0)

    def one_call_and_drain():
        retriever.search("q", top_k=3)
        return retriever.drain_counts()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: one_call_and_drain(), range(8)))

    for counts in results:
        assert counts["retrieval_dense_calls"] == 1
        assert counts["retrieval_keyword_calls"] == 1


# ---------------------------------------------------------------------------
# P205 regression: same-leg duplicate credit (run p205.70-check)
# ---------------------------------------------------------------------------


def test_rrf_never_credits_the_same_doc_twice_within_one_ranking():
    """A corpus with duplicate rows returns the same title at two ranks of
    ONE leg. Summing both (1/60 + 1/62 = 0.0328) makes a single-leg hit
    score like a document BOTH legs ranked first -- 0.98 after
    corpus_search's RRF_SQUASH, which is how an irrelevant Memcached
    document marked a goal about India and the US 'covered' at recall 1.0.
    Cross-leg agreement must still count; same-leg repetition must not.
    """
    dup = rrf_fuse([["docA", "docB", "docA"]])
    solo = rrf_fuse([["docA", "docB"]])
    assert dup["docA"] == solo["docA"], (
        "a doc repeated inside one ranking must score exactly as it would "
        "appear once -- otherwise duplicate corpus rows manufacture coverage")
    # The real invariant the rest of the codebase depends on: a single-leg
    # rank-0 hit stays at the SINGLE_LEG_SCORE_CEILING after squashing.
    assert dup["docA"] * 30.0 == 0.5

    # Cross-leg agreement is still rewarded -- that is the point of fusion.
    both = rrf_fuse([["docA"], ["docA"]])
    assert both["docA"] > dup["docA"]


# ---------------------------------------------------------------------------
# D-150 -- the floor's verdict binds BOTH legs
#
# min_similarity only ever gated the dense leg. OpenSearch hits went
# straight into fusion, so an off-topic corpus document could still become
# Evidence for a query the corpus does not cover. Live (p205.282-check,
# "Compare the Armies of China and India" against a Redis/Memcached
# corpus):
#
#     OPENSEARCH (BM25)
#     query: "organizational structures command hierarchies Chinese People's"
#     [hit 1]  bm25=0.92  topic=redis
#
# 42 corpus + 36 mcp items entered a military run through that door.
# ---------------------------------------------------------------------------


class _Leg:
    """One retrieval leg, with the `available` flag the real stores carry."""

    def __init__(self, hits, available=True):
        self._hits = hits
        self.available = available
        self.calls = 0

    def search(self, query, top_k=5):
        self.calls += 1
        return list(self._hits)


def _dense(sim, content="doc"):
    return {"content": content, "title": content, "similarity": sim}


def _kw(content="doc", bm25=0.92):
    return {"content": content, "title": content, "bm25": bm25}


def test_the_p205_282_shape_no_longer_reaches_fusion():
    """Every dense candidate below the floor means the corpus does not
    cover this query. The keyword leg must not overrule that."""
    from research_agent.retrieval.hybrid import HybridRetriever

    dense = _Leg([_dense(0.41, "Redis data structures"),
                  _dense(0.38, "Throughput characteristics")])
    keyword = _Leg([_kw("Redis data structures")])
    retriever = HybridRetriever(dense, keyword, min_similarity=0.55)

    results = retriever.search("organizational structures command hierarchies")

    assert results == []
    assert retriever.drain_counts()["retrieval_keyword_dropped_off_topic"] == 1


def test_a_query_the_corpus_does_cover_keeps_its_keyword_hits():
    """The half that matters more. Embeddings are what handle the
    vocabulary mismatch BM25 cannot, so a query whose dense leg clears the
    floor keeps everything -- this is what a >=2 term-overlap gate would
    have broken (measured: it drops `in-corpus-operational`, a query two
    documents genuinely answer using different words)."""
    from research_agent.retrieval.hybrid import HybridRetriever

    dense = _Leg([_dense(0.81, "Redis persistence"), _dense(0.40, "noise")])
    keyword = _Leg([_kw("Redis failure modes")])
    retriever = HybridRetriever(dense, keyword, min_similarity=0.55)

    results = retriever.search("failure modes and operational tooling")

    assert len(results) == 2
    assert "retrieval_keyword_dropped_off_topic" not in retriever.drain_counts()


def test_a_degraded_dense_leg_never_silences_the_keyword_leg():
    """Single-leg degradation must keep working exactly as documented. An
    unavailable Qdrant returns no hits, which must not be read as the
    corpus rejecting the query."""
    from research_agent.retrieval.hybrid import HybridRetriever

    dense = _Leg([], available=False)
    keyword = _Leg([_kw("Redis persistence")])
    retriever = HybridRetriever(dense, keyword, min_similarity=0.55)

    results = retriever.search("redis persistence")

    assert len(results) == 1
    assert "retrieval_keyword_dropped_off_topic" not in retriever.drain_counts()


def test_a_dense_leg_that_returned_nothing_at_all_is_not_a_verdict():
    """No candidates is not the same as candidates that all failed. Only
    the latter says the corpus does not cover the query."""
    from research_agent.retrieval.hybrid import HybridRetriever

    dense = _Leg([])
    keyword = _Leg([_kw("Redis persistence")])
    retriever = HybridRetriever(dense, keyword, min_similarity=0.55)

    results = retriever.search("redis persistence")

    assert len(results) == 1
    assert "retrieval_keyword_dropped_off_topic" not in retriever.drain_counts()


def test_the_floor_switched_off_leaves_both_legs_exactly_as_before():
    """MIN_SIMILARITY=0.0 is the documented escape hatch and must restore
    pre-D-150 behaviour too, not just pre-P2-01."""
    from research_agent.retrieval.hybrid import HybridRetriever

    dense = _Leg([_dense(0.01, "irrelevant")])
    keyword = _Leg([_kw("also irrelevant")])
    retriever = HybridRetriever(dense, keyword, min_similarity=0.0)

    assert len(retriever.search("anything")) == 2

