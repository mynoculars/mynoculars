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

Python mechanics used in this file, if any of this is new to you:
    math.exp(-math.log(2.0) * age / half_life)
        This is the standard "exponential decay with a half-life" formula.
        math.log(2.0) is the natural logarithm of 2 (≈0.693); the whole
        expression computes 0.5 raised to the power (age / half_life) —
        i.e. the value is exactly 0.5 when age == half_life, 0.25 when
        age == 2×half_life, and so on. It's written using exp/log instead
        of a direct "0.5 ** (age/half_life)" purely as a common
        mathematical convention, not for any performance reason.
    sorted(scored, key=lambda t: t[0], reverse=True)
        Sorts a list of TUPLES by looking at each tuple's FIRST element
        (t[0]) — here, each tuple is (final_score, hit_dict, volatility),
        built a few lines above, and this sorts them from highest
        final_score to lowest.
    scored[: self.top_k]
        A slice taking the first self.top_k elements of the now-sorted
        list — i.e. "keep only the best `top_k` results after reranking."
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

    CALLED BY   SemanticMemory.retrieve, below — once per candidate memory
                hit, to compute how much to discount its raw similarity
                score based on how old it is and how quickly facts of its
                kind go stale.

    stable      -> 1.0 always (near-flat by design)
    semi_stable -> exponential with configured half-life (default 90d)
    volatile    -> exponential with configured half-life (default 14d)

    Exponential over linear: freshness value drops fastest when new — the
    natural shape for "is this still true".
    """
    if volatility == Volatility.STABLE:
        return 1.0
    # A conditional expression (see prompts/templates.py for the same
    # construct): pick whichever configured half-life matches this item's
    # volatility class.
    half_life = half_life_semi if volatility == Volatility.SEMI_STABLE else half_life_volatile
    # max(age_days, 0.0) guards against a negative age (which shouldn't
    # normally happen, but could arise from clock skew between when a point
    # was written and when it's read back) — never let "freshness" exceed
    # 1.0 because of a negative age making the exponent negative.
    return math.exp(-math.log(2.0) * max(age_days, 0.0) / half_life)


class SemanticMemory:
    """Cross-run memory over a dedicated Qdrant collection.

    Every method on this class treats a degraded (unreachable) underlying
    store the same way QdrantStore itself does: retrieve() quietly returns
    [] and store_run() quietly writes nothing, rather than raising — so the
    rest of the graph can call these methods unconditionally without ever
    checking availability itself.
    """

    def __init__(self, store: QdrantStore, top_k: int,
                 half_life_semi: float, half_life_volatile: float):
        """store may be degraded — retrieve() then returns [] and
        store_run() no-ops, i.e. the agent silently runs memory-off.

        CALLED BY   cli.py::build_app_and_settings — constructed once per
                    run, wrapping a QdrantStore already pointed at the
                    memory collection (a DIFFERENT collection name than the
                    corpus one — see storage/qdrant_store.py's docstring).
        """
        self.store = store
        self.top_k = top_k
        self.half_life_semi = half_life_semi
        self.half_life_volatile = half_life_volatile

    def retrieve(self, query: str) -> List[Evidence]:
        """Similarity search reranked by similarity x decay.

        CALLED BY   agents/planning.py::memory_retrieve_node — the second
                    node of every run, right after classify, and BEFORE any
                    goal has been composed (see that node's docstring for
                    why the ordering matters).
        CALLS       self.store.search(...) — Qdrant similarity search,
                    over-fetching 2x self.top_k candidates so there is room
                    for the decay rerank below to actually change which
                    items make the final cut, not just their order.
        RETURNS     up to self.top_k Evidence objects, tagged
                    source="memory", already decay-adjusted so the coverage
                    rule in agents/gathering.py needs no special case for
                    memory-sourced evidence.

        Returns Evidence with source='memory'; score already decay-adjusted
        so the coverage rule (D-17) needs no special-casing for memory.
        """
        hits = self.store.search(query, top_k=self.top_k * 2)  # over-fetch, rerank, cut
        scored = []
        for h in hits:
            # Volatility(h.get(...)) CONSTRUCTS an Enum member from its
            # string value — e.g. Volatility("semi_stable") gives back
            # Volatility.SEMI_STABLE. The .get(..., default) call supplies
            # "semi_stable" as a fallback if this particular stored point
            # somehow has no "volatility" key in its payload at all.
            vol = Volatility(h.get("volatility", Volatility.SEMI_STABLE.value))
            d = decay_factor(h["age_days"], vol, self.half_life_semi, self.half_life_volatile)
            # Build a tuple of (combined_score, original_hit_dict,
            # volatility) for each hit — a common Python pattern for
            # "attach a computed sort key to each item before sorting",
            # since Python's sort needs something to compare, and the raw
            # hit dicts alone don't have an obvious ordering.
            scored.append((h["similarity"] * d, h, vol))
        # See the module docstring for exactly what this sort call does:
        # order by the first tuple element (the combined score), best
        # first.
        scored.sort(key=lambda t: t[0], reverse=True)

        out: List[Evidence] = []
        # scored[: self.top_k] — see the module docstring's slice
        # explanation. This loop unpacks each surviving (final, h, vol)
        # tuple back into three separate names.
        for final, h, vol in scored[: self.top_k]:
            out.append(Evidence(
                # A synthetic task_key is invented here purely so this
                # Evidence object has SOME unique-ish identifier, since
                # memory items were never dispatched as an actual
                # SearchTask (unlike fresh corpus evidence — see
                # tools/corpus_search.py). abs(hash(...)) % 10_000 turns
                # arbitrary content text into a short numeric suffix.
                task_key=f"memory-{abs(hash(h.get('content',''))) % 10_000}",
                # P2-02: NAMESPACED, not the raw stored goal_id. Before this
                # fix, a memory item's goal_id was whichever earlier run's
                # goal it happened to be filed under — and since every
                # run's goals are always named g1, g2, g3... an old,
                # unrelated run's "g3" fact could silently satisfy THIS
                # run's unrelated "g3" goal in agents/gathering.py's
                # coverage check (e.goal_id == g.goal_id), just by string
                # collision. Prefixing with "memory::" makes that equality
                # impossible to ever accidentally satisfy — real goal ids
                # are always bare "g1".."g5", never "memory::anything". The
                # original goal_id is kept, not discarded, purely as a
                # readable label (shown in the compiled report's evidence
                # listing) — it just can no longer impersonate a CURRENT
                # goal.
                goal_id=f"memory::{h.get('goal_id', 'unknown')}",
                source="memory",
                content=h.get("content", ""),
                score=min(1.0, final),
                volatility=vol,
            ))
        if out:
            log_event(logger, "memory.retrieved", count=len(out))
        return out

    def store_run(self, query: str, evidence: List[Evidence]) -> int:
        """Persist fresh evidence from a passed run. Returns items written.

        CALLED BY   agents/compilation.py::memory_writer_node — reachable
                    ONLY when that run's critique passed (see
                    orchestration/graph.py::route_after_critique) — a
                    report that failed its own quality bar never reaches
                    this method at all.
        WRITES      new points into the Qdrant memory collection, via
                    self.store.upsert_texts (storage/qdrant_store.py) — one
                    per fresh evidence item.

        `fresh = [e for e in evidence if e.source != "memory"]` is a list
        comprehension filtering OUT anything that was itself recalled from
        memory earlier in THIS run — so a fact this run already knew
        (because a past run remembered it) is never re-written back into
        memory as if it were new, which would otherwise let the exact same
        fact accumulate duplicate points every single run it gets recalled
        in.
        """
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
