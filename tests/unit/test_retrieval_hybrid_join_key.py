"""
tests/unit/test_retrieval_hybrid_join_key.py — regression cover for D-61(b),
the content-identity join key in retrieval/hybrid.py::_join_key.

Lives apart from test_retrieval_hybrid.py for the reason given at the top of
test_llm_router_quality_gate.py: every test-file edit in this change set is
delivered as a NEW file, because an in-place append could not be landed by
`git apply` on the target checkout and a new file has no context to match.
Fold all three back into their sibling modules once the checkouts are
reconciled -- D-34's one-file-per-source-module rule is the intended end
state, not this.

Closes the README's long-standing Limitation 2.
"""

import pytest

from research_agent.retrieval.hybrid import HybridRetriever, _join_key


class _FakeLeg:
    """Minimal fake store: only .search() and .available -- enough to drive
    HybridRetriever without touching real Qdrant/OpenSearch. Duplicated from
    test_retrieval_hybrid.py rather than imported, so this file stands alone
    and cannot be broken by an edit to that one."""

    def __init__(self, hits=None, available=True):
        self._hits = hits or []
        self.available = available

    def search(self, query, top_k):
        return list(self._hits)

# ---------------------------------------------------------------------------
# FIX-4 — RRF joins on content identity, not on `title`
#
# Closes the README's long-standing Limitation 2 ("RRF joins the two legs on
# title, not on any store id — silently wrong for a corpus with duplicate or
# missing titles"). Three distinct failures the old
# `title or content[:60]` key produced, one test each.
# ---------------------------------------------------------------------------


def test_two_documents_sharing_a_title_no_longer_fuse_into_one():
    # The failure the README named. Under the old key both rows collapsed to
    # the single id "Release notes" and one document vanished silently.
    dense = [{"title": "Release notes", "content": "Redis 7 adds functions.",
              "similarity": 0.9},
             {"title": "Release notes", "content": "Memcached 1.6 adds TLS.",
              "similarity": 0.8}]
    retriever = HybridRetriever(_FakeLeg(hits=dense), _FakeLeg(), min_similarity=0.0)
    out = retriever.search("x", top_k=5)
    assert {h["content"] for h in out} == {"Redis 7 adds functions.",
                                           "Memcached 1.6 adds TLS."}


def test_two_documents_sharing_a_60_char_prefix_no_longer_fuse():
    # The fallback half of the old key was just as collision-prone: shared
    # boilerplate leads, templated headers, a common opening sentence.
    lead = "This document describes the operational characteristics of the "
    dense = [{"content": lead + "Redis deployment.", "similarity": 0.9},
             {"content": lead + "Memcached deployment.", "similarity": 0.8}]
    retriever = HybridRetriever(_FakeLeg(hits=dense), _FakeLeg(), min_similarity=0.0)
    assert len(retriever.search("x", top_k=5)) == 2


def test_same_document_titled_in_one_leg_only_now_fuses():
    # The opposite error, and the one with real scoring consequences: a
    # genuine two-leg agreement used to be counted as two separate
    # single-leg hits whenever the two stores disagreed about the title,
    # and a rank-0 SINGLE-leg hit squashes to exactly min_evidence_score
    # (the P2-01 follow-up boundary collision).
    body = "Redis supports primary-replica replication."
    dense = [{"title": "Replication", "content": body, "similarity": 0.9}]
    kw = [{"content": body, "score": 4.2}]
    retriever = HybridRetriever(_FakeLeg(hits=dense), _FakeLeg(hits=kw),
                                min_similarity=0.0)
    out = retriever.search("replication", top_k=5)
    assert len(out) == 1
    # Both legs contributed, so the fused score is the two-leg sum --
    # exactly double what one leg alone at the same rank contributes.
    # Asserted as a ratio rather than a literal so this test does not
    # silently encode RRF_K or the rank offset.
    single = HybridRetriever(_FakeLeg(hits=dense), _FakeLeg(),
                             min_similarity=0.0).search("replication", top_k=5)
    assert out[0]["fused_score"] == pytest.approx(2 * single[0]["fused_score"])


def test_join_key_tolerates_a_hit_with_no_content_field():
    # Previously `h.get("content", "")[:60]` made these all collapse to the
    # empty-string key rather than raising. Same tolerance, same collapse.
    dense = [{"title": "a", "similarity": 0.9}, {"title": "b", "similarity": 0.8}]
    retriever = HybridRetriever(_FakeLeg(hits=dense), _FakeLeg(), min_similarity=0.0)
    assert len(retriever.search("x", top_k=5)) == 1


def test_join_key_is_stable_across_processes():
    # `content_id` is uuid5, not Python's per-process-randomised hash() —
    # the same defect Limitation 3 called out for Evidence.task_key. A key
    # that changes between runs would make the two legs fail to fuse
    # non-deterministically.
    assert _join_key({"content": "abc"}) == _join_key({"content": "abc"})
    assert _join_key({"content": "abc"}) != _join_key({"content": "abd"})
