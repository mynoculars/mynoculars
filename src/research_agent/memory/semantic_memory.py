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
    - Server-side decay via Qdrant FormulaQuery (D-27) is now IMPLEMENTED
      (P2-10, storage/qdrant_store.py::search_with_decay) -- gated by
      settings.memory_server_side_decay, off by default; the Python path
      below remains the permanent parity oracle either way.
    - Content-identity dedup for memory writes (P2-15) is now IMPLEMENTED
      too -- see store_run's docstring for exactly what "supersession"
      ended up meaning once an EXACT-content-hash identity scheme was
      chosen (deliberately, not fuzzy/semantic matching -- see
      storage/qdrant_store.py::content_id's docstring for why).
    - Still deferred (documented): per-item volatility classification —
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
from typing import Any, List

from research_agent import langfuse as lf
from research_agent.logging_setup import log_event, run_id_var
from research_agent.state import Evidence, Volatility
from research_agent.storage.qdrant_store import QdrantStore, content_id

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


def coerce_volatility(raw: Any) -> Volatility:
    """Turn a stored payload's raw "volatility" value into a Volatility.

    CALLED BY   SemanticMemory.retrieve, below, on BOTH the Python and the
                server-side-decay paths.
    WHY THIS EXISTS: Volatility(<unrecognised string>) raises ValueError,
    and memory_retrieve_node (agents/planning.py) wraps this in no
    try/except -- so a single point whose payload carries an unexpected
    value (hand-edited, a schema change, a half-migrated collection of the
    exact "mixed old/new points" kind ensure_payload_indexes's docstring
    warns about) killed the entire run at the second node. That directly
    contradicts this class's own contract, which promises retrieve()
    "quietly returns []" rather than raising. Unknown values now fall back
    to SEMI_STABLE -- the same default the .get() call already used for a
    MISSING key -- and are logged once so the bad data is still visible.
    """
    try:
        return Volatility(raw)
    except ValueError:
        log_event(logger, "memory.unknown_volatility", level=logging.WARNING,
                  value=str(raw)[:60])
        return Volatility.SEMI_STABLE


class SemanticMemory:
    """Cross-run memory over a dedicated Qdrant collection.

    Every method on this class treats a degraded (unreachable) underlying
    store the same way QdrantStore itself does: retrieve() quietly returns
    [] and store_run() quietly writes nothing, rather than raising — so the
    rest of the graph can call these methods unconditionally without ever
    checking availability itself.
    """

    def __init__(self, store: QdrantStore, top_k: int,
                 half_life_semi: float, half_life_volatile: float,
                 server_side_decay: bool = False):
        """store may be degraded — retrieve() then returns [] and
        store_run() no-ops, i.e. the agent silently runs memory-off.

        CALLED BY   cli.py::build_app_and_settings — constructed once per
                    run, wrapping a QdrantStore already pointed at the
                    memory collection (a DIFFERENT collection name than the
                    corpus one — see storage/qdrant_store.py's docstring).

        server_side_decay (P2-10, default False): when True, retrieve()
        below asks the store to do fusion/decay server-side
        (QdrantStore.search_with_decay) instead of over-fetching and
        reranking in Python. The Python path (decay_factor(), just below
        this class) is kept as the PARITY ORACLE either way — it is never
        removed, and stays the default for anyone who hasn't opted in.
        """
        self.store = store
        self.top_k = top_k
        self.half_life_semi = half_life_semi
        self.half_life_volatile = half_life_volatile
        self.server_side_decay = server_side_decay

    def retrieve(self, query: str) -> List[Evidence]:
        """Similarity search reranked by similarity x decay — either in
        Python (default) or server-side in Qdrant (P2-10, opt-in).

        CALLED BY   agents/planning.py::memory_retrieve_node — the second
                    node of every run, right after classify, and BEFORE any
                    goal has been composed (see that node's docstring for
                    why the ordering matters).
        CALLS       EITHER self.store.search_with_decay(...) (P2-10, server-
                    side formula — only when self.server_side_decay is
                    True) OR self.store.search(...) (the original Python
                    path, over-fetching 2x self.top_k candidates so there
                    is room for the decay rerank below to actually change
                    which items make the final cut, not just their order).
        RETURNS     up to self.top_k Evidence objects, tagged
                    source="memory", already decay-adjusted so the coverage
                    rule in agents/gathering.py needs no special case for
                    memory-sourced evidence — identical contract whichever
                    path computed the score.

        Returns Evidence with source='memory'; score already decay-adjusted
        so the coverage rule (D-17) needs no special-casing for memory.

        P2-10: when self.server_side_decay is True, `scored` below is built
        directly from search_with_decay's results — its "similarity" is
        ALREADY the decay-adjusted final score (Qdrant computed vector
        similarity x decay itself), so decay_factor() is deliberately NOT
        called again here; doing so would double-apply decay. When False
        (the default), this is byte-for-byte the original Python path:
        decay_factor() (below this class, kept as the permanent PARITY
        ORACLE — see its own docstring and the module header) computes the
        multiplier explicitly, over 2x-over-fetched candidates.
        """
        if self.server_side_decay:
            hits = self.store.search_with_decay(
                query, top_k=self.top_k, decay_field="volatility",
                half_lives={Volatility.STABLE.value: None,
                           Volatility.SEMI_STABLE.value: self.half_life_semi,
                           Volatility.VOLATILE.value: self.half_life_volatile})
            scored = [
                (h.get("similarity", 0.0), h,
                 coerce_volatility(h.get("volatility", Volatility.SEMI_STABLE.value)))
                for h in hits
            ]
            # Already ranked best-first by Qdrant and already cut to
            # top_k — no Python-side sort/slice needed on this path.
        else:
            hits = self.store.search(query, top_k=self.top_k * 2)  # over-fetch, rerank, cut
            scored = []
            for h in hits:
                # Volatility(h.get(...)) CONSTRUCTS an Enum member from its
                # string value — e.g. Volatility("semi_stable") gives back
                # Volatility.SEMI_STABLE. The .get(..., default) call supplies
                # "semi_stable" as a fallback if this particular stored point
                # somehow has no "volatility" key in its payload at all.
                vol = coerce_volatility(h.get("volatility", Volatility.SEMI_STABLE.value))
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
            scored = scored[: self.top_k]

        out: List[Evidence] = []
        # scored is already at most self.top_k items on EITHER path above
        # (server-side: Qdrant's `limit=top_k`; Python: the slice just
        # above) — this loop unpacks each surviving (final, h, vol) tuple
        # back into three separate names, same as before P2-10.
        for final, h, vol in scored:
            out.append(Evidence(
                # A synthetic task_key is invented here purely so this
                # Evidence object has SOME unique-ish identifier, since
                # memory items were never dispatched as an actual
                # SearchTask (unlike fresh corpus evidence — see
                # tools/corpus_search.py). abs(hash(...)) % 10_000 turns
                # arbitrary content text into a short numeric suffix.
                # Built from content_id (the same uuid5 content identity
                # storage/qdrant_store.py already uses for point ids) rather
                # than the previous abs(hash(content)) % 10_000: Python
                # randomises str hashes per process unless PYTHONHASHSEED is
                # pinned, so the OLD key for the same remembered fact changed
                # on every run and collided freely inside a 10k space. This
                # one is stable across runs and machines, for free --
                # content_id is already imported here for store_run.
                task_key=f"memory-{content_id(h.get('content', ''))}",
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
        lf.event(run_id_var.get(), "memory.retrieved",
                input={"query": query}, metadata={"count": len(out)})
        return out

    def store_run(self, query: str, evidence: List[Evidence],
                  min_score: float = 0.0) -> int:
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

        P2-15: id_fn=lambda item: content_id(item["content"]) makes this
        write IDEMPOTENT on exact-duplicate content, the same way P2-03
        already made corpus ingest idempotent. Before this, EVERY passed
        run wrote a fresh, randomly-id'd point for each of its fresh
        evidence items -- so a fact independently re-discovered from the
        CORPUS (source="corpus", never filtered by the `fresh` check
        above, since that check only catches facts recalled FROM memory,
        not facts re-found fresh from the corpus every time) accumulated
        one duplicate point per run it kept getting rediscovered in,
        forever, with no cap. Passing id_fn here means the SAME content
        string now overwrites its own prior point in place instead --
        which, as a side effect, also REFRESHES that point's created_at/
        created_at_iso to the current instant (upsert-by-id fully replaces
        the payload, it doesn't merge), so a fact that keeps getting
        reaffirmed run after run naturally reads as "recently confirmed"
        rather than aging out under decay -- this IS what "supersession"
        ends up meaning once identity is exact-content-hash: the old
        point isn't archived-and-linked, it simply becomes the new one
        (same id, replaced payload). A genuinely DIFFERENT wording of the
        same underlying fact (e.g. paraphrased by a different provider on
        a different run) gets a different id and remains a SEPARATE
        point -- catching that case would need semantic/fuzzy matching,
        deliberately out of scope here (see content_id's own docstring).
        Garbage-collecting points that have decayed near zero regardless
        of exact-duplicate status is a separate concern -- see
        scripts/gc_memory.py.

        P2-15 follow-up: the "memory.stored" log line used to report only
        a single `count` -- how many items were WRITTEN, which after the
        id_fn change above conflates two very different outcomes: a
        brand-new fact landing in memory for the first time, versus an
        already-known fact simply refreshing its timestamp. Both count as
        "written" from upsert_texts's point of view (same code path
        either way), but they mean different things operationally -- a
        run where every single item was "new" is discovering a lot of
        fresh material; a run where everything was "overwritten" is
        mostly just reaffirming what memory already knew. This computes
        that split BEFORE calling upsert_texts (existing_point_ids checks
        which of this batch's computed ids Qdrant already has), and logs
        both numbers instead of one.
        """
        # D-42: model recollection is NOT a research finding and must not
        # enter durable memory. Live (runs p205.96/.97-check): the army
        # run stored 28 items of which 24 were source="model"; the very
        # next, unrelated run ("Compare India and US") recalled five of
        # them and goal_manager -- which is handed memory as "relevant
        # facts from earlier research" -- composed an entirely MILITARY
        # goal set for an open question, inheriting PLA doctrine prose
        # verbatim into an India-vs-US report.
        #
        # Two independent reasons this must not happen:
        #  1. It gains nothing. The model can regenerate its own
        #     recollection on demand -- storing it caches a lookup that
        #     was never expensive.
        #  2. It launders guesses into facts. An unverified claim written
        #     here comes back on a later run tagged source="memory",
        #     indistinguishable from something a DOCUMENT supported, and
        #     steers that run before any retrieval happens. That is a
        #     self-reinforcing loop, and the one place a single
        #     fabrication can become permanent.
        # Memory is for what RETRIEVAL found (D-24). Recollection is not.
        # "web" added by Phase 4 (D-57), for D-42's reason plus one more.
        # D-42's reason: what enters durable memory comes back on a LATER
        # run as source="memory" at raw cosine similarity, indistinguishable
        # from something a document supported. The extra reason for web: a
        # snippet is volatile by construction (make_web_search_tool stamps
        # Volatility.VOLATILE on every one), so a stored copy of today's
        # search result is simply a wrong answer next month, with nothing in
        # the text marking it stale. Memory is for what RETRIEVAL of the
        # CORPUS found; a live lookup should be repeated, not cached.
        fresh = [e for e in evidence
                 if e.source not in ("memory", "model", "web")]
        # D-24 quality gate. A passed critique says the REPORT is
        # acceptable; it says nothing about whether the evidence behind
        # it ever cleared the coverage bar. Live (run p205.71-check):
        # recall 0.0, every goal reported "no reliable data", and 16
        # sub-threshold items were still written to long-term memory.
        # Those return next run as source="memory" at raw cosine
        # similarity (~0.75), which OUTRANKS anything fresh retrieval
        # produces under RRF -- so junk that failed the bar once comes
        # back permanently promoted. Default 0.0 preserves the old
        # behaviour for any caller that does not opt in.
        if min_score > 0.0:
            kept = [e for e in fresh if e.score > min_score]
            if len(kept) != len(fresh):
                log_event(logger, "memory.below_quality_floor",
                          dropped=len(fresh) - len(kept), floor=min_score)
            fresh = kept
        items = [{
            "content": e.content,
            "goal_id": e.goal_id,
            "volatility": e.volatility.value,
            "source_query": query,
        } for e in fresh]
        ids = [content_id(item["content"]) for item in items]
        # existing_point_ids fails open (empty set) if it can't run for any
        # reason -- worst case, every item is counted as "new" below, which
        # is never WRONG about what got written, only imprecise about the
        # new/overwritten split. The actual upsert_texts call just below is
        # entirely unaffected either way.
        already_existed = self.store.existing_point_ids(ids)
        overwritten = sum(1 for i in ids if i in already_existed)
        new = len(items) - overwritten

        written = self.store.upsert_texts(
            items, id_fn=lambda item: content_id(item["content"]))
        log_event(logger, "memory.stored", count=written, new=new, overwritten=overwritten)
        lf.event(run_id_var.get(), "memory.stored",
                metadata={"count": written, "new": new, "overwritten": overwritten})
        return written
