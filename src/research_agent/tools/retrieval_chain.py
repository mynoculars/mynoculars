"""
tools/retrieval_chain.py — the retrieval escalation ladder (D-38).

Purpose:
    Try every available way to answer one SearchTask before concluding
    that nothing can be found. Exposed as a single ToolFn, so the graph,
    the worker contract (D-6/D-15) and the Send-fanout are all unchanged.

The ladder, in order, stopping at the first tier that yields evidence
above the quality floor:

    1. corpus         hybrid dense + BM25 over the ingested documents
    2. corpus, again  ONE reformulation of the query (a retrieval miss is
                      very often a phrasing miss -- see below)
    3. mcp            the external specialist tool, when one is wired
    4. model          the answering model's own knowledge

Why an ordered chain rather than a router decision:
    Cheapest and most authoritative first. A real document always beats a
    recollection when a real document exists, and the model tier is only
    ever paid for on tasks that genuinely exhausted everything else.

Why the reformulation retry:
    Live evidence (runs p205.66-.81): the task producer writes long,
    highly-specific queries -- "Chinese People's Liberation Army PLA
    equipment technology comparison with Indian Army" -- which then miss
    BM25 entirely and drag the dense leg below its similarity floor. The
    SAME goal expressed in three or four core terms often retrieves fine.
    One retry is cheap and closes a large class of self-inflicted misses.

Quality floor:
    A tier "answered" only if it returned at least one item scoring above
    min_evidence_score -- the same predicate progress_checker uses to mark
    a goal covered (D-17). Anything weaker is kept (it is still context
    for the compiler) but does NOT stop the ladder, because evidence that
    cannot cover a goal cannot end the search for one.
"""

import logging
from typing import Any, Callable, List, Optional

from research_agent.logging_setup import log_event
from research_agent.state import Evidence, SearchTask

logger = logging.getLogger(__name__)

# Query scaffolding that carries no topical signal. Shared by the
# reformulator (which strips it) and the relevance gate (which must not
# count it as an on-topic match -- "comparison" appearing in both a query
# about armies and a document about Redis is not evidence of anything).
_FILLER = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "with",
    "to", "vs", "versus", "between", "comparison", "compare",
    "comparative", "analysis", "analyze", "evaluate", "evaluation",
    "assess", "assessment", "examine", "including", "their", "both",
    "its", "recent", "current", "overview", "study", "report",
    "data", "rate", "rates", "scale", "system", "systems", "features",
}

ToolFn = Callable[[SearchTask], List[Evidence]]


def _reformulate(query: str) -> str:
    """Collapse a long, over-specified query to its distinctive terms.

    Pure string work, no LLM call: this runs inside a parallel worker and
    must stay cheap and deterministic. Drops connective filler and the
    comparison scaffolding that task producers habitually emit, keeps
    proper nouns and content words, and caps length.
    """
    filler = _FILLER
    words = [w for w in query.replace(":", " ").replace(",", " ").split()
             if w.strip()]
    kept = [w for w in words if w.lower().strip(".") not in filler]
    # Keep proper nouns and the first few content words -- enough to stay
    # on topic, short enough to actually match.
    if not kept:
        kept = words
    return " ".join(kept[:6])


def _distinctive_terms(text: str) -> set:
    """Lower-cased content words of 4+ characters, minus scaffolding.

    Pure and cheap: this runs inside every parallel worker, on every tier,
    so it must not allocate much or call anything.
    """
    return {w for w in (
        "".join(c if c.isalnum() else " " for c in text.lower()).split())
        if len(w) > 3 and w not in _FILLER}


def make_retrieval_chain(corpus: ToolFn, min_evidence_score: float,
                         mcp: Optional[ToolFn] = None,
                         model: Optional[ToolFn] = None,
                         reformulate: bool = True) -> ToolFn:
    """Build the escalating ToolFn.

    CALLED BY   assembly.py::build_app_and_settings -- it replaces the
                bare corpus tool as the DEFAULT tool handed to
                build_graph, so every search_worker gets the full ladder
                without any graph change.

    Parameters:
        corpus: the tier-1 tool (tools/corpus_search.py).
        min_evidence_score: the quality floor a tier must clear to stop
            the ladder -- pass settings.min_evidence_score so this can
            never drift from progress_checker's coverage rule (D-17).
        mcp: optional tier-3 specialist (tools/mcp_client.py), or None.
        model: optional tier-4 parametric tier
            (tools/model_knowledge.py), or None to disable it.
        reformulate: whether to spend tier 2 on a rephrased corpus retry.

    A tier that RAISES is treated as a miss and the ladder continues --
    one dead backend must not abort a task that a later tier could still
    answer. The exception is logged, never swallowed silently.
    """

    def _try(tier: str, fn: ToolFn, task: SearchTask) -> List[Evidence]:
        try:
            return fn(task) or []
        except Exception as exc:  # noqa: BLE001 -- a dead tier is not a dead task
            log_event(logger, "chain.tier_failed", level=logging.WARNING,
                      tier=tier, task=task.key, reason=type(exc).__name__,
                      error=str(exc)[:300])
            return []

    def _sufficient(evidence: List[Evidence], query: str) -> bool:
        """A tier answered only if it returned something that clears the
        coverage floor AND is lexically ON TOPIC.

        The score test alone is a RANKING signal, not a relevance one.
        Live (run p205.90-check): asked for "GDP per capita India US
        comparison 2023" against a corpus of ten Redis documents, both
        retrieval legs returned the same three irrelevant documents,
        cross-leg agreement pushed them to ~1.0, this function said
        "answered", and the ladder never escalated -- so the report said
        the evidence did not cover GDP per capita, which is precisely the
        give-up this ladder exists to remove. A fixed-k search over a small
        corpus ALWAYS returns k documents; without a topical check there is
        no empty state and no tier can ever fail.

        One shared distinctive term is a deliberately low bar -- this is a
        floor against wholly off-topic matches, not a relevance ranker.
        """
        on_floor = [e for e in evidence if e.score > min_evidence_score]
        if not on_floor:
            return False
        terms = _distinctive_terms(query)
        if not terms:
            return True  # nothing to test against; trust the score
        # ONE shared term was too weak a bar. Live (run p205.101-check):
        # "Comparative analysis Redis Cassandra DynamoDB petabyte scale
        # performance scalability cost operational complexity" has NINE
        # distinctive terms; a Redis session-caching document shares
        # exactly one of them ("redis"), which passed the gate, stopped the
        # ladder at tier 1, and left model_sourced_items at 0 -- so the
        # report said "no retrieved evidence quantifies..." for all five
        # goals. That is the give-up this ladder exists to remove, re-
        # entering through a too-permissive gate. A long, specific query
        # matching on its ONE broad subject word is not a topical match:
        # scale the requirement with how specific the query was.
        need = min(3, max(1, len(terms) // 4))
        return any(len(terms & _distinctive_terms(e.content)) >= need
                   for e in on_floor)

    def retrieval_chain(task: SearchTask) -> List[Evidence]:
        collected: List[Evidence] = []

        found = _try("corpus", corpus, task)
        collected.extend(found)
        if _sufficient(found, task.query):
            log_event(logger, "chain.answered", tier="corpus", task=task.key,
                      items=len(found))
            return collected

        if reformulate:
            short = _reformulate(task.query)
            if short and short.lower() != task.query.lower():
                retry_task = task.model_copy(update={"query": short})
                found = _try("corpus_reformulated", corpus, retry_task)
                # Re-tag onto the ORIGINAL task key so dedup, coverage and
                # the D-16 failure ledger all still see one task.
                found = [e.model_copy(update={"task_key": task.key})
                         for e in found]
                collected.extend(found)
                if _sufficient(found, short):
                    log_event(logger, "chain.answered", tier="corpus_reformulated",
                              task=task.key, items=len(found), query=short)
                    return collected

        if mcp is not None:
            found = _try("mcp", mcp, task)
            collected.extend(found)
            if _sufficient(found, task.query):
                log_event(logger, "chain.answered", tier="mcp", task=task.key,
                          items=len(found))
                return collected

        if model is not None:
            found = _try("model", model, task)
            collected.extend(found)
            log_event(logger, "chain.answered", tier="model", task=task.key,
                      items=len(found),
                      sufficient=_sufficient(found, task.query))
            return collected

        log_event(logger, "chain.exhausted", level=logging.WARNING,
                  task=task.key, items=len(collected))
        return collected

    # Preserve the duck-typed telemetry seam the search worker drains
    # (agents/gathering.py) -- the corpus tier is the only tier that has
    # retrieval-leg counters to report.
    drain = getattr(corpus, "drain_retrieval_counts", None)
    if drain is not None:
        retrieval_chain.drain_retrieval_counts = drain  # type: ignore[attr-defined]
    return retrieval_chain
