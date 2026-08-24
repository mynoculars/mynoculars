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
    4. web            a real search engine, when one is wired (D-57)
    5. model          the answering model's own knowledge

Why web sits AFTER mcp and not in its place:
    In every deployment of this repo so far, the MCP server wired at tier 3
    is scripts/mcp_corpus_server.py -- i.e. the corpus, reached a second
    way. Tiers 1-3 therefore all resolve to the SAME ingested documents,
    which is precisely why "the corpus does not contain it" used to fall
    straight through to recollection. Web is the first tier that can return
    something the corpus never held, so it belongs after every corpus route
    is exhausted and before the model is asked to remember.

    A run wiring a genuinely external tier-3 specialist keeps that ordering
    too: a curated specialist source outranks an open web search.

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
from research_agent.retrieval.terms import FILLER, distinctive_terms
from research_agent.state import Evidence, SearchTask

logger = logging.getLogger(__name__)

ToolFn = Callable[[SearchTask], List[Evidence]]


def _reformulate(query: str) -> str:
    """Collapse a long, over-specified query to its distinctive terms.

    Pure string work, no LLM call: this runs inside a parallel worker and
    must stay cheap and deterministic. Drops connective filler and the
    comparison scaffolding that task producers habitually emit, keeps
    proper nouns and content words, and caps length.
    """
    words = [w for w in query.replace(":", " ").replace(",", " ").split()
             if w.strip()]
    kept = [w for w in words if w.lower().strip(".") not in FILLER]
    # Keep proper nouns and the first few content words -- enough to stay
    # on topic, short enough to actually match.
    if not kept:
        kept = words
    return " ".join(kept[:6])


def make_retrieval_chain(corpus: ToolFn, min_evidence_score: float,
                         mcp: Optional[ToolFn] = None,
                         web: Optional[ToolFn] = None,
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
        web: optional tier-4 web search
            (tools/mcp_client.py::make_web_search_tool), or None to
            disable it. None whenever settings.web_search_enabled is
            false, which is the default -- so the ladder is byte-identical
            to every pre-Phase-4 run unless it is deliberately turned on.
        model: optional tier-5 parametric tier
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

        The score test alone is a RANKING signal, not a relevance one. A
        fixed-k search over a small corpus ALWAYS returns k documents;
        without a topical check there is no empty state, so no tier can
        ever fail and the ladder can never escalate.

        `need` scales with how specific the query was, with a floor of 2
        and a cap at the query's own term count. Three live runs shaped
        those bounds -- p205.90 (no topical test at all), p205.101 (one
        broad subject word satisfying a nine-term query) and p205.141 (one
        ACCIDENTAL word satisfying a short reformulated query, which is
        every retry by construction, since _reformulate caps at 6 words).
        The cap exists because a single-term query cannot share two terms
        with anything, however relevant. Full accounts in DECISIONS.md
        D-39, D-44 and D-55; the tests in
        tests/unit/test_tools_retrieval_chain.py name the run each one
        locks down.

        One term is deliberately NOT enough, at any query length. This is
        a floor against off-topic matches, not a relevance ranker.
        """
        on_floor = [e for e in evidence if e.score > min_evidence_score]
        if not on_floor:
            return False
        terms = distinctive_terms(query)
        if not terms:
            return True  # nothing to test against; trust the score
        need = min(3, len(terms), max(2, len(terms) // 4))
        return any(len(terms & distinctive_terms(e.content)) >= need
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
                # Logged BEFORE the attempt, unconditionally — not only on
                # success (see chain.answered below). The retrieval itself
                # always happens here regardless of whether it later turns
                # out sufficient; without this line, an insufficient
                # reformulated attempt's retrieval.raw/retrieval.hybrid
                # events (query=short) have no event anywhere logging that
                # exact query string, so the narrative log's per-task
                # grouping (logging_setup.py::NarrativeFormatter.
                # _render_fanout) can't correlate them back to this task.
                log_event(logger, "chain.attempt", tier="corpus_reformulated",
                          task=task.key, query=short)
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

        # Tiers 3-4: mcp, web -- same shape (_try -> extend -> _sufficient
        # -> log chain.answered -> return), expressed once as data instead
        # of two near-identical blocks (S-6). Web's extra failure-path log
        # (chain.web_gate_rejected) is the one thing that still differs
        # per tier, so it stays a small special case inside the loop
        # rather than forcing mcp to carry an on-miss hook it never uses.
        for name, fn in (("mcp", mcp), ("web", web)):
            if fn is None:
                continue
            found = _try(name, fn, task)
            collected.extend(found)
            if _sufficient(found, task.query):
                log_event(logger, "chain.answered", tier=name, task=task.key,
                          items=len(found))
                return collected
            if name == "web" and found:
                # _sufficient's topical gate applies here exactly as it
                # does to every other tier, deliberately, even though a
                # search engine has ALREADY done its own relevance
                # judgement. Two reasons: consistency (one rule for what
                # "a tier answered" means), and the fact that an engine's
                # relevance is judged against the query string, while this
                # gate is judged against the query's distinctive terms --
                # which is the check that catches a plausible-looking
                # result set about the wrong subject.
                #
                # The risk this creates is real and worth watching: web
                # snippets are short (~150-250 chars), so a genuinely
                # relevant result can fail a 2-distinctive-term overlap
                # and drop the ladder to the model tier unnecessarily.
                # That is why the miss is logged rather than silent --
                # measure before tuning, the same posture D-54 took
                # toward the call budget.
                log_event(logger, "chain.web_gate_rejected",
                          task=task.key, items=len(found),
                          reason="web returned results that missed the "
                                 "score or topical gate")

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
