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
