"""
tools/corpus_search.py — The retrieval tool exposed to search workers.

Purpose:
    The single "tool" the core build's workers can invoke: hybrid search
    over the ingested sample corpus, returning Evidence entities.

Responsibilities:
    - Translate retrieval results into the Evidence domain model (scores
      normalized, provenance stamped) so nodes never see raw store dicts.

Design decision (function-registry now, MCP later):
    The full design (D-26) mediates every tool through MCP so workers
    discover tools from servers. This core build uses a plain callable —
    deliberately: the graph-level tool-calling pattern (worker receives a
    task, invokes a tool, returns evidence) is identical, so upgrading the
    plumbing to MCP later touches only this module. That seam is the point.
"""

from typing import List

from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import Evidence, SearchTask, Volatility


def make_corpus_tool(retriever: HybridRetriever, top_k: int = 3):
    """Build the corpus-search tool bound to a retriever.

    Parameters:
        retriever: the hybrid retriever over the ingested corpus.
        top_k: evidence items to return per task.

    Returns:
        callable(task: SearchTask) -> List[Evidence]. May raise — the
        worker owns failure recording (D-16), not the tool.
    """

    def corpus_search(task: SearchTask) -> List[Evidence]:
        hits = retriever.search(task.query, top_k=top_k)
        evidence: List[Evidence] = []
        for h in hits:
            evidence.append(Evidence(
                task_key=task.key,
                goal_id=task.goal_id,
                source="corpus",
                content=h.get("content", "")[:800],
                # fused_score is unbounded-ish; squash into 0..1 for the
                # coverage rule (D-17). Simple monotone squash is enough here.
                score=min(1.0, h.get("fused_score", 0.0) * RRF_SQUASH),
                volatility=Volatility.SEMI_STABLE,
            ))
        return evidence

    return corpus_search


# One dense+one keyword first-rank hit ≈ 2/60 ≈ 0.033 fused. Scale so that a
# top hit in both legs lands near 1.0 — keeps MIN_EVIDENCE_SCORE tunable on
# an intuitive 0..1 axis.
RRF_SQUASH = 30.0
