"""
retrieval/hybrid.py — Hybrid corpus retrieval: dense + BM25 fused with RRF.

Purpose:
    The retrieval engine the search workers call. Runs a dense similarity
    query (Qdrant) and a keyword query (OpenSearch) over the ingested
    corpus, then fuses the two rankings with Reciprocal Rank Fusion.

Responsibilities:
    - rrf_fuse(): the fusion math, kept as a pure standalone function so it
      is unit-testable and readable — this is the educational heart of the
      module.
    - HybridRetriever.search(): orchestrate both legs, fuse, return unified
      results. Degrades to whichever leg is available (or [] if neither).

Design decision (Python-side fusion in the core build):
    The full design (D-27) pushes fusion + decay server-side into Qdrant's
    FormulaQuery for one-round-trip retrieval. Here fusion is deliberately
    in Python: a learner can read the RRF formula instead of a query DSL.
    Tradeoff: two round trips and client-side merge cost — irrelevant at
    sample-corpus scale, documented as the upgrade path in the README.
"""

import logging
from typing import Any, Dict, List

from research_agent.logging_setup import log_event
from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

RRF_K = 60  # standard smoothing constant; larger = flatter rank influence


def rrf_fuse(rankings: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked ID lists.

    score(d) = sum over rankings of 1 / (k + rank_of_d)   (rank is 0-based)

    Why RRF: it needs no score normalization across systems whose score
    scales are incomparable (cosine similarity vs BM25) — only ranks.

    CALLED BY   HybridRetriever.search, below — the one and only call site.
                This function is deliberately kept STANDALONE (not a method
                on any class) and has no dependency on Qdrant, OpenSearch,
                or anything else in this codebase, which is exactly what
                makes it trivial to unit-test with plain lists of strings.

    Parameters:
        rankings: e.g. [["docA","docB"], ["docB","docC"]].
        k: smoothing constant.

    Returns:
        {doc_id: fused_score}, higher is better.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        # enumerate(ranking) gives us both the position (rank, starting at
        # 0 for the first/best-ranked item) and the doc_id at that
        # position, for every ranked list passed in.
        for rank, doc_id in enumerate(ranking):
            # scores.get(doc_id, 0.0) reads the running total for this
            # doc_id, defaulting to 0.0 the first time it's ever seen (it
            # might not have appeared in an earlier ranking at all).
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Dense + keyword retrieval over the ingested corpus.

    CALLED BY   tools/corpus_search.py::corpus_search, which is the actual
                "tool" every parallel search_worker invokes.
    """

    def __init__(self, dense: QdrantStore, keyword: OpenSearchStore,
                min_similarity: float = 0.0):
        """Both stores may be degraded; search() adapts per leg.

        min_similarity (P2-01): a floor applied to the DENSE leg's raw
        cosine similarity score, BEFORE a hit ever enters RRF fusion or
        becomes Evidence. Default 0.0 preserves the old behaviour (every
        dense hit passes) for any caller that doesn't explicitly opt in.
        Only the dense leg gets this floor — BM25 scores are corpus-
        dependent and unbounded, so there's no principled fixed cutoff to
        apply there the way there is for a 0..1 cosine similarity.
        """
        self.dense = dense
        self.keyword = keyword
        self.min_similarity = min_similarity

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return fused results: dicts with content/title + 'fused_score'.

        READS       nothing from ResearchState — receives only the plain
                    query string its caller (tools/corpus_search.py) built
                    from a SearchTask.
        CALLS       self.dense.search(...) (Qdrant, meaning-based nearest
                    neighbours) and self.keyword.search(...) (OpenSearch,
                    exact-word BM25 matching) — both unconditionally; each
                    one independently returns [] if its underlying store is
                    unreachable, rather than raising, so this method never
                    needs to check availability itself.
        RETURNS     a list of dicts, each the original hit's fields plus a
                    new "fused_score" key, sorted best-first and capped at
                    top_k.

        Degradation behavior (deliberate, logged): both legs up -> true
        hybrid; one leg up -> that leg's ranking passes through RRF alone
        (order-preserving); none -> [].
        """
        dense_hits = self.dense.search(query, top_k)
        kw_hits = self.keyword.search(query, top_k)

        # P2-01: drop dense hits below the similarity floor BEFORE they can
        # enter fusion or become Evidence at all. Without this, a dense
        # index always returns its k nearest neighbours no matter how
        # irrelevant the query — an out-of-domain question could never
        # produce zero evidence. This is a NEW filter step, not a change
        # to what dense.search() itself returns.
        if self.min_similarity > 0.0:
            dropped = sum(1 for h in dense_hits if h.get("similarity", 0.0) < self.min_similarity)
            dense_hits = [h for h in dense_hits if h.get("similarity", 0.0) >= self.min_similarity]
            if dropped:
                log_event(logger, "retrieval.below_floor", query=query,
                          dropped=dropped, floor=self.min_similarity)

        # by_id maps a computed "document identity" string to the FULL hit
        # dict (with every field the store returned) so we can look the
        # whole document back up later, after RRF has decided the fused
        # ranking using ONLY the id strings.
        by_id: Dict[str, Dict[str, Any]] = {}
        dense_rank: List[str] = []
        for h in dense_hits:
            # h.get("title") or h["content"][:60] — use the document's
            # title as its identity if it has one; otherwise fall back to
            # the first 60 characters of its content (h["content"][:60] is
            # a slice, same mechanism used elsewhere in this codebase).
            # This is the JOIN KEY between the two legs — see the module's
            # design decision above and this project's README for the
            # known limitation this creates with duplicate/missing titles.
            doc_id = h.get("title") or h["content"][:60]
            by_id[doc_id] = h
            dense_rank.append(doc_id)
        kw_rank: List[str] = []
        for h in kw_hits:
            doc_id = h.get("title") or h["content"][:60]
            # by_id.setdefault(doc_id, h): if doc_id is ALREADY a key in
            # by_id (e.g. this same document also appeared in the dense
            # leg above), leave the existing value untouched; only insert
            # `h` if doc_id is genuinely new. This means the dense leg's
            # version of a document "wins" if both legs found the exact
            # same doc_id — a minor, deliberate asymmetry.
            by_id.setdefault(doc_id, h)
            kw_rank.append(doc_id)

        # [r for r in (dense_rank, kw_rank) if r] is a LIST COMPREHENSION
        # (see orchestration/graph.py's docstring for the general idea)
        # that keeps only the NON-EMPTY rankings — if, say, OpenSearch was
        # unreachable, kw_rank would be [] (falsy), and this line quietly
        # drops it from consideration rather than passing an empty ranking
        # into rrf_fuse for no benefit.
        rankings = [r for r in (dense_rank, kw_rank) if r]
        if not rankings:
            log_event(logger, "retrieval.no_backends", level=logging.WARNING)
            return []

        fused = rrf_fuse(rankings)
        # sorted(fused, key=fused.get, reverse=True) sorts the DICTIONARY'S
        # KEYS (doc_ids) by looking up each one's fused score via
        # fused.get, highest score first, then [:top_k] keeps only the
        # first `top_k` of them (see task_utils.py for the same slicing
        # idiom).
        ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = []
        for doc_id in ordered:
            # dict(by_id[doc_id]) makes a SHALLOW COPY of the stored hit
            # dict before modifying it — this avoids mutating the original
            # dict still referenced inside by_id, in case anything else
            # were to read from by_id again later.
            doc = dict(by_id[doc_id])
            doc["fused_score"] = fused[doc_id]
            results.append(doc)
        log_event(logger, "retrieval.hybrid", query=query,
                  dense=len(dense_rank), keyword=len(kw_rank), fused=len(results))
        return results
