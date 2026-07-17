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

    Parameters:
        rankings: e.g. [["docA","docB"], ["docB","docC"]].
        k: smoothing constant.

    Returns:
        {doc_id: fused_score}, higher is better.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Dense + keyword retrieval over the ingested corpus."""

    def __init__(self, dense: QdrantStore, keyword: OpenSearchStore):
        """Both stores may be degraded; search() adapts per leg."""
        self.dense = dense
        self.keyword = keyword

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return fused results: dicts with content/title + 'fused_score'.

        Degradation behavior (deliberate, logged): both legs up -> true
        hybrid; one leg up -> that leg's ranking passes through RRF alone
        (order-preserving); none -> [].
        """
        dense_hits = self.dense.search(query, top_k)
        kw_hits = self.keyword.search(query, top_k)

        by_id: Dict[str, Dict[str, Any]] = {}
        dense_rank: List[str] = []
        for h in dense_hits:
            doc_id = h.get("title") or h["content"][:60]
            by_id[doc_id] = h
            dense_rank.append(doc_id)
        kw_rank: List[str] = []
        for h in kw_hits:
            doc_id = h.get("title") or h["content"][:60]
            by_id.setdefault(doc_id, h)
            kw_rank.append(doc_id)

        rankings = [r for r in (dense_rank, kw_rank) if r]
        if not rankings:
            log_event(logger, "retrieval.no_backends", level=logging.WARNING)
            return []

        fused = rrf_fuse(rankings)
        ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = []
        for doc_id in ordered:
            doc = dict(by_id[doc_id])
            doc["fused_score"] = fused[doc_id]
            results.append(doc)
        log_event(logger, "retrieval.hybrid", query=query,
                  dense=len(dense_rank), keyword=len(kw_rank), fused=len(results))
        return results
