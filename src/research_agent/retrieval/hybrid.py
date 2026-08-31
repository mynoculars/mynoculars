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
import threading
import time
from typing import Any, Dict, List

from research_agent import langfuse as lf
from research_agent.guardrails.retrieval import passes_similarity_floor
from research_agent.logging_setup import log_event, run_id_var
from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.qdrant_store import QdrantStore, content_id

logger = logging.getLogger(__name__)

RRF_K = 60  # standard smoothing constant; larger = flatter rank influence


def _join_key(hit: Dict[str, Any]) -> str:
    """The identity a document is fused on across the dense and keyword legs.

    CALLED BY   HybridRetriever.search, once per hit from each leg.

    FIX-4 — was `hit.get("title") or hit.get("content", "")[:60]`, i.e. the
    document TITLE, falling back to a 60-character content prefix. Both
    halves were wrong in the same direction, and the README carried this as
    a known limitation ("silently wrong for a corpus with duplicate or
    missing titles") for several revisions:

      - two genuinely different documents sharing a title FUSE INTO ONE,
        so one of them silently disappears from the results and the other
        inherits a rank it did not earn;
      - two different documents sharing a 60-char prefix (boilerplate
        headers, a common lead sentence, templated release notes) do the
        same;
      - and the same document indexed with a title in one store but not
        the other NEVER fuses, so a real two-leg agreement is scored as
        two separate single-leg hits — which matters directly, because a
        rank-0 single-leg hit squashes to exactly the min_evidence_score
        boundary (the P2-01 follow-up collision).

    `content_id` (uuid5 of the content) is the identity function this
    codebase already uses for Qdrant point ids and memory dedup, so this
    makes the join key agree with how the documents were STORED rather
    than inventing a third notion of document identity. Neither store
    chunks, so one JSONL line is one document in both — the contents are
    byte-identical by construction and hash to the same id.

    The `or ""` guard keeps the previous tolerance for a hit missing
    `content` entirely: those all collapse to one key, exactly as an empty
    prefix slice did before, rather than raising.
    """
    return content_id(hit.get("content") or "")


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
        # A doc_id may legitimately appear in SEVERAL rankings (that is the
        # whole point of fusion — agreement across legs is signal). It must
        # never be credited twice WITHIN one ranking: that is not agreement,
        # it is the same document counted twice.
        #
        # Found live (run p205.66-check): the corpus had accumulated
        # duplicate rows, so one BM25 leg returned the same title at ranks 0
        # and 2 — "keyword": 3, "fused": 2 in the retrieval.hybrid log line.
        # Summed, that is 1/60 + 1/62 = 0.0328, which corpus_search's
        # RRF_SQUASH turns into 0.98 — indistinguishable from a document
        # BOTH legs ranked first. An off-topic Memcached document therefore
        # scored above min_evidence_score and marked a goal about the Indian
        # and Chinese armies "covered". Deduping per ranking caps a
        # single-leg hit back at the SINGLE_LEG_SCORE_CEILING the rest of
        # this codebase is built around (see prompts/templates.py).
        seen: set = set()
        ranking = [d for d in ranking if not (d in seen or seen.add(d))]
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

        P2-07 follow-up (retrieval-side boundary telemetry): self._counts
        is a threading.local(), not a plain dict — a single HybridRetriever
        instance is shared across every parallel search_worker invocation
        this run (see cli.py::build_app_and_settings, which constructs one
        HybridRetriever and wraps it once in make_corpus_tool), and
        LangGraph dispatches those workers as N simultaneous invocations
        (design doc, orchestration/graph.py's module docstring). A plain
        shared dict here would race under real concurrency: two workers'
        counts could clobber each other between one worker's bump and its
        own drain. threading.local() gives each worker's OS thread its own
        private counts, so drain_counts() (below) only ever reads back what
        THIS call's own thread wrote, no lock required.
        """
        self.dense = dense
        self.keyword = keyword
        self.min_similarity = min_similarity
        self._counts = threading.local()

    def _bump_retrieval_counts(self) -> None:
        """Record one retrieval attempt, BEFORE calling either leg (P2-07
        follow-up). Bumping first — not after a successful return — means
        an attempt that raises partway through (e.g. the QdrantStore
        NotFoundError seen in live testing when a collection doesn't exist
        yet) is still counted as an attempted call, the same "attempts,
        win or lose" philosophy llm/router.py's llm_provider_calls uses.

        CALLED BY   search(), below, as its very first statement.
        WRITES      this thread's private counts dict (see __init__'s
                    threading.local() note) — never state.counters
                    directly; this class has no knowledge of the graph.
        """
        # (0 if available else 1) for each leg: retrieval_leg_unavailable
        # counts DEGRADED legs, not empty result sets — a leg can be fully
        # available and legitimately return zero hits for an obscure query;
        # that is not degradation and must not be conflated with it.
        # getattr(..., "available", True): QdrantStore/OpenSearchStore both
        # always set self.available in __init__, but minimal test fakes
        # (e.g. test_hitl.py's FakeDense/FakeKeyword, which only implement
        # .search()) predate this attribute and don't set it — defaulting
        # to True (assume available) rather than raising AttributeError
        # keeps this a purely additive change, not a rewrite of those
        # existing Phase 1 test fixtures.
        unavailable = ((0 if getattr(self.dense, "available", True) else 1)
                      + (0 if getattr(self.keyword, "available", True) else 1))
        counts: Dict[str, float] = getattr(self._counts, "data", {})
        counts["retrieval_dense_calls"] = counts.get("retrieval_dense_calls", 0) + 1
        counts["retrieval_keyword_calls"] = counts.get("retrieval_keyword_calls", 0) + 1
        counts["retrieval_leg_unavailable"] = (
            counts.get("retrieval_leg_unavailable", 0) + unavailable)
        self._counts.data = counts

    def _bump(self, key: str) -> None:
        """Add one to a single thread-local retrieval counter."""
        counts: Dict[str, float] = getattr(self._counts, "data", {})
        counts[key] = counts.get(key, 0) + 1
        self._counts.data = counts

    def _bump_by(self, key: str, n: int) -> None:
        """Add an arbitrary amount to a single thread-local retrieval
        counter (Guardrail G1). Same threading.local() storage as _bump
        and _bump_retrieval_counts -- a no-op for n=0 so callers don't
        need to guard the zero case themselves.
        """
        if n <= 0:
            return
        counts: Dict[str, float] = getattr(self._counts, "data", {})
        counts[key] = counts.get(key, 0) + n
        self._counts.data = counts

    def _safe_leg(self, leg: Any, name: str, query: str,
                  top_k: int) -> List[Dict[str, Any]]:
        """Call one retrieval leg, converting a MID-RUN failure into [].

        WHY THIS EXISTS: both stores only decide availability ONCE, from a
        liveness probe in their own __init__. self.available therefore says
        nothing about whether the store is still reachable NOW. A store that
        dies AFTER startup (restart, network blip, expired credential, a
        collection dropped underneath us) raises straight out of .search(),
        through this class, through tools/corpus_search.py, and lands in
        agents/gathering.py::search_worker's except -- which records the
        whole task as a D-16 failure and discards the OTHER, healthy leg's
        hits along with it. With MAX_FANOUT workers all hitting the same
        dead store at once, every task in the cycle fails together and the
        run reports a research failure for what was a transient store
        outage.

        Catching per LEG (here) rather than per store keeps the stores
        policy-free -- they stay thin wrappers that raise -- while making
        the degradation this class's docstring has always PROMISED
        ("each one independently returns [] if unreachable") actually true.
        retrieval_leg_unavailable is bumped so a mid-run failure is visible
        in telemetry, not just in the log; before this it only ever counted
        boot-time unavailability.
        """
        try:
            return leg.search(query, top_k)
        except Exception as exc:  # noqa: BLE001 -- degrade this leg, not the task
            log_event(logger, "retrieval.leg_failed", level=logging.WARNING,
                      leg=name, query=query, reason=type(exc).__name__,
                      error=str(exc)[:300])
            self._bump("retrieval_leg_unavailable")
            return []

    def drain_counts(self) -> Dict[str, float]:
        """Return this thread's accumulated retrieval counts, and reset them.

        CALLED BY   tools/corpus_search.py::make_corpus_tool's returned
                    corpus_search function, which exposes this as
                    corpus_search.drain_retrieval_counts — a bound method
                    reference, so calling it from a DIFFERENT thread than
                    the one that just called search() still correctly
                    reads that OTHER thread's own counts, since
                    threading.local() keys storage by the calling thread,
                    not by which thread originally constructed the object.
                    In practice search_worker (agents/gathering.py) always
                    calls tool(task) and drains on the SAME thread, back to
                    back, so this is a non-issue here — noted for
                    completeness.
        Same drain-not-peek reasoning as FallbackRouter.drain_counters
        (llm/router.py): resets so a later call on the same thread never
        double-reports an earlier call's counts.
        """
        data = getattr(self._counts, "data", {})
        self._counts.data = {}
        return data

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
        self._bump_retrieval_counts()
        _span_start = time.time()
        # Each leg is independently failure-isolated -- see _safe_leg. One
        # dead backend degrades to single-leg fusion; it never costs the
        # healthy leg's hits or burns the task as a D-16 failure.
        dense_hits = self._safe_leg(self.dense, "dense", query, top_k)
        kw_hits = self._safe_leg(self.keyword, "keyword", query, top_k)

        # P2-01: drop dense hits below the similarity floor BEFORE they can
        # enter fusion or become Evidence at all. Without this, a dense
        # index always returns its k nearest neighbours no matter how
        # irrelevant the query — an out-of-domain question could never
        # produce zero evidence. This is a NEW filter step, not a change
        # to what dense.search() itself returns.
        if self.min_similarity > 0.0:
            dropped = sum(1 for h in dense_hits
                         if not passes_similarity_floor(h.get("similarity", 0.0), self.min_similarity))
            # Guardrail G1: count candidates BEFORE filtering (not just
            # what's dropped) so drain_counts() can report a per-run
            # DROP RATIO, not just a raw count -- a raw count alone can't
            # distinguish "3 dropped out of 3" (floor is starving this
            # run) from "3 dropped out of 300" (floor is doing its job).
            self._bump_by("retrieval_dense_candidates", len(dense_hits))
            self._bump_by("retrieval_dropped_by_floor", dropped)
            dense_hits = [h for h in dense_hits
                          if passes_similarity_floor(h.get("similarity", 0.0), self.min_similarity)]
            if dropped:
                log_event(logger, "retrieval.below_floor", query=query,
                          dropped=dropped, floor=self.min_similarity)
            # D-150: the floor's verdict now binds BOTH legs.
            #
            # It only ever gated the dense leg. OpenSearch hits went
            # straight into fusion, so an off-topic corpus document could
            # still become Evidence for a query the corpus does not cover
            # -- exactly what the floor exists to prevent, reached by the
            # other door. Live (run p205.282-check, "Compare the Armies of
            # China and India" against a Redis/Memcached corpus):
            #
            #     OPENSEARCH (BM25)
            #     query: "organizational structures command hierarchies
            #             Chinese People's"
            #     [hit 1]  bm25=0.92  topic=redis
            #
            # BM25 matched on "command" and "structures" -- Redis
            # vocabulary -- and 42 corpus + 36 mcp items entered a military
            # run. corpus_recall stayed 0.0 because D-47's topical gate
            # correctly refused to call them grounded, but they had already
            # consumed the prompt budget (32 items dropped for space), and
            # since D-142 ranks corpus above web they would LEAD the prompt
            # if they survived dedup.
            #
            # WHY THIS RULE AND NOT A BM25 THRESHOLD. BM25 scores are
            # unbounded and corpus-relative; there is no number to
            # calibrate that transfers. WHY NOT A TERM-OVERLAP GATE,
            # which was the obvious candidate: measured against this
            # repo's own golden set, a >=2 distinctive-term bar drops the
            # Redis hits correctly (they share exactly ONE term) but also
            # drops `in-corpus-operational`, a query two documents
            # genuinely answer using different words. Too high a floor
            # silently discarding real evidence is D-42's stated failure
            # mode, so that idea was measured and rejected rather than
            # shipped.
            #
            # What IS already calibrated is min_similarity itself, derived
            # per corpus by OPERATIONS.md's procedure. If every dense
            # candidate the store returned fell below it, the corpus does
            # not cover this query -- and that verdict should bind the
            # keyword leg too, instead of being ignored by it. Embeddings
            # are also what handle the vocabulary mismatch BM25 cannot, so
            # `in-corpus-operational` keeps its hits: its dense leg clears
            # the floor.
            #
            # Guarded so it can only fire on a real verdict: the dense
            # store must have been AVAILABLE and must have RETURNED
            # candidates. A degraded Qdrant leaves the keyword leg
            # untouched, which keeps single-leg degradation working
            # exactly as documented above.
            # `dropped and not dense_hits` together mean exactly: the
            # store returned candidates, and every one of them fell below
            # the floor.
            if (dropped and not dense_hits and kw_hits
                    and getattr(self.dense, "available", False)):
                self._bump_by("retrieval_keyword_dropped_off_topic",
                              len(kw_hits))
                log_event(logger, "retrieval.keyword_off_topic",
                          query=query, dropped=len(kw_hits),
                          floor=self.min_similarity,
                          reason="every dense candidate fell below the "
                                 "floor, so the corpus does not cover this "
                                 "query")
                kw_hits = []

        # by_id maps a computed "document identity" string to the FULL hit
        # dict (with every field the store returned) so we can look the
        # whole document back up later, after RRF has decided the fused
        # ranking using ONLY the id strings.
        by_id: Dict[str, Dict[str, Any]] = {}
        dense_rank: List[str] = []
        for h in dense_hits:
            doc_id = _join_key(h)
            by_id[doc_id] = h
            dense_rank.append(doc_id)
        kw_rank: List[str] = []
        for h in kw_hits:
            doc_id = _join_key(h)
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
            # NOT necessarily "the backends are down". Both legs returning
            # nothing is the ordinary, correct answer for a query the corpus
            # does not cover — and with a min_similarity floor set high
            # enough (0.6 in the live run that exposed this) the dense leg is
            # emptied HERE, by this class, after the store answered
            # perfectly well. Logging that as a WARNING named
            # "no_backends" sent a real off-corpus diagnosis chasing an
            # infrastructure fault that did not exist. Report what actually
            # happened: zero results, and whether a leg was genuinely
            # unavailable is already carried by retrieval_leg_unavailable.
            log_event(logger, "retrieval.no_results", query=query,
                      dense_available=getattr(self.dense, "available", True),
                      keyword_available=getattr(self.keyword, "available", True))
            lf.span(run_id_var.get(), "retrieval.hybrid_search",
                   input={"query": query}, output={"fused": 0},
                   metadata={"dense": 0, "keyword": 0, "degraded": True,
                             "reason": "no_backends_available"},
                   start_time=_span_start, end_time=time.time(), level="WARNING")
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
        # Phase 3: one span per hybrid search call -- dense/keyword/RRF
        # hit counts, plus whether this call ran degraded (one leg down),
        # which is exactly what retrieval_leg_unavailable already tracks
        # in telemetry (see agents/gathering.py) -- this just makes the
        # SAME fact visible per-call in Langfuse, not a new signal.
        lf.span(run_id_var.get(), "retrieval.hybrid_search",
               input={"query": query}, output={"fused_doc_ids": ordered},
               metadata={"dense_hits": len(dense_rank), "keyword_hits": len(kw_rank),
                         "fused_results": len(results),
                         "degraded": not (dense_rank and kw_rank)},
               start_time=_span_start, end_time=time.time())
        return results
