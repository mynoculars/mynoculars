"""
memory/semantic_memory.py — Long-term memory: retrieval with decay, write-back.

Purpose:
    Give the agent cross-run memory (design decision D-24): evidence from
    past runs is stored in Qdrant and retrieved at plan time, reranked by a
    volatility-aware recency decay.

Responsibilities:
    - decay_factor(): the staleness math, pure and unit-testable.
    - SemanticMemory.retrieve(): similarity search + decay rerank; returns
      Evidence tagged source="memory" so downstream nodes can treat memory
      as just another evidence party (contradiction machinery included).
    - SemanticMemory.store_run(): persist a run's FRESH evidence after the
      critique passes (memory-sourced items are never re-written).

Design decisions:
    - Why decay is a RERANK, never a filter: a stable fact from a year ago
      must remain retrievable; only volatile facts should fade fast. One
      TTL for both is wrong at both ends — hence per-volatility half-lives.
    - Deferred (documented): supersession links, server-side decay via
      Qdrant FormulaQuery (D-27), per-item volatility classification —
      items currently inherit SEMI_STABLE unless the tool says otherwise.
"""

import logging
import math
from typing import List

from research_agent.logging_setup import log_event
from research_agent.state import Evidence, Volatility
from research_agent.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def decay_factor(age_days: float, volatility: Volatility,
                 half_life_semi: float, half_life_volatile: float) -> float:
    """Return the 0..1 freshness multiplier for a memory item.

    stable      -> 1.0 always (near-flat by design)
    semi_stable -> exponential with configured half-life (default 90d)
    volatile    -> exponential with configured half-life (default 14d)

    Exponential over linear: freshness value drops fastest when new — the
    natural shape for "is this still true".
    """
    if volatility == Volatility.STABLE:
        return 1.0
    half_life = half_life_semi if volatility == Volatility.SEMI_STABLE else half_life_volatile
    return math.exp(-math.log(2.0) * max(age_days, 0.0) / half_life)


class SemanticMemory:
    """Cross-run memory over a dedicated Qdrant collection."""

    def __init__(self, store: QdrantStore, top_k: int,
                 half_life_semi: float, half_life_volatile: float):
        """store may be degraded — retrieve() then returns [] and
        store_run() no-ops, i.e. the agent silently runs memory-off."""
        self.store = store
        self.top_k = top_k
        self.half_life_semi = half_life_semi
        self.half_life_volatile = half_life_volatile

    def retrieve(self, query: str) -> List[Evidence]:
        """Similarity search reranked by similarity x decay.

        Returns Evidence with source='memory'; score already decay-adjusted
        so the coverage rule (D-17) needs no special-casing for memory.
        """
        hits = self.store.search(query, top_k=self.top_k * 2)  # over-fetch, rerank, cut
        scored = []
        for h in hits:
            vol = Volatility(h.get("volatility", Volatility.SEMI_STABLE.value))
            d = decay_factor(h["age_days"], vol, self.half_life_semi, self.half_life_volatile)
            scored.append((h["similarity"] * d, h, vol))
        scored.sort(key=lambda t: t[0], reverse=True)

        out: List[Evidence] = []
        for final, h, vol in scored[: self.top_k]:
            out.append(Evidence(
                task_key=f"memory-{abs(hash(h.get('content',''))) % 10_000}",
                goal_id=h.get("goal_id", "memory"),
                source="memory",
                content=h.get("content", ""),
                score=min(1.0, final),
                volatility=vol,
            ))
        if out:
            log_event(logger, "memory.retrieved", count=len(out))
        return out

    def store_run(self, query: str, evidence: List[Evidence]) -> int:
        """Persist fresh evidence from a passed run. Returns items written."""
        fresh = [e for e in evidence if e.source != "memory"]
        items = [{
            "content": e.content,
            "goal_id": e.goal_id,
            "volatility": e.volatility.value,
            "source_query": query,
        } for e in fresh]
        written = self.store.upsert_texts(items)
        log_event(logger, "memory.stored", count=written)
        return written
