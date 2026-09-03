"""
reporting/report_metrics.py -- the telemetry figures derived from the
SHIPPED REPORT and the run's goals, and the last-line-of-sight WARNINGs
that read them.

WHY THIS FILE EXISTS (S-13). D-146 split the counter-only half of a
531-line telemetry_node into reporting/telemetry.py and stopped there.
What it left behind was still 173 lines of real code at cyclomatic
complexity 41 -- the third most complex block in the codebase -- and it
was not homogeneous: half of it aggregates `state.counters`, and half
of it reads `state.final_report` and `state.goals` and fires five
WARNINGs about what the reader actually received. This is that second
half, extracted on the same seam and for the same reason.

THE DISTINCTION THAT MAKES IT A SEAM. reporting/telemetry.py's three
functions take one argument -- the counter dict -- and are pure. Every
figure here needs `state` and `settings`, and five of them LOG. Those
are different dependency shapes and different testability stories, so
they are different modules rather than three more functions in the
counter file.

D-12 IS UNCHANGED AND IS THE POINT: nothing here judges anything. Every
figure is counted from state, and every WARNING reports a count against
a configured threshold.

WHY THESE READ THE REPORT AND NEVER A COUNTER (D-59, D-88). compiler_node
runs ONCE PER REVISION and `state.counters` merges additively, so a
counter taken there sums every compile ATTEMPT. A two-rewrite run once
reported 44 listed / 25 suppressed sources against 35 web items, for a
report containing 34 entries -- every figure arithmetically correct,
none describing the artifact the reader received. Everything in
shipped_report_metrics below is derived from `state.final_report`, which
is report-scoped by construction.
"""

import logging
from typing import Any, Dict

from research_agent.config import Settings
from research_agent.guardrails.citations import (residual_glue_sites,
                                                 residual_paste_sites)
from research_agent.guardrails.claims import (audit_cited_figures,
                                              cited_goal_ids_in_prose)
from research_agent.guardrails.grounding import report_carries_grounding_notice
from research_agent.guardrails.retrieval import has_grounded_evidence
from research_agent.guardrails.sources import count_listed_sources
from research_agent.logging_setup import log_event
from research_agent.retrieval.terms import distinctive_terms
from research_agent.state import ResearchState

logger = logging.getLogger(__name__)


def goal_coverage(state: ResearchState, settings: Settings) -> Dict[str, Any]:
    """The three goal-scoped figures, and nothing else. Pure -- no logging.

    RETURNS goal_ids, corpus_recall, goals_without_evidence, grounding_ratio.

    corpus_recall must apply the SAME topical gate the retrieval ladder
    uses (D-39), not just the score floor. Live (runs p205.99/.100-check)
    it reported corpus_recall 1.0 against a ten-document Redis corpus for
    queries about armies and about India vs the US: off-topic hits still
    cleared score > 0.5 via cross-leg agreement, so the one metric added
    specifically as the honesty counterpart to recall was fooled exactly
    the way the chain's sufficiency test used to be. A goal counts here
    only if a DOCUMENT both scored above the floor and shares vocabulary
    with that goal's own description -- has_grounded_evidence (M-1) is the
    single implementation of that predicate, shared with
    progress_checker_node and gap_generator_node in gathering.py, so this
    floor and theirs can never drift apart again the way a second,
    hardcoded `> 0.5` copy previously let them.

    grounding_ratio is a RATIO, not just the list, so it is trendable
    across runs and comparable across queries with different goal counts.
    Guarded against a run that produced no goals at all (a planning
    failure), where 0/0 is undefined rather than perfect.
    """
    goal_ids = [g.goal_id for g in state.goals]
    goal_terms = {g.goal_id: distinctive_terms(g.description)
                  for g in state.goals}
    doc_covered = {
        g.goal_id for g in state.goals
        if has_grounded_evidence(g.goal_id, goal_terms[g.goal_id],
                                 state.evidence, settings.min_evidence_score)}
    corpus_recall = (round(len([g for g in goal_ids if g in doc_covered])
                           / len(goal_ids), 3) if goal_ids else 0.0)
    evidenced = {e.goal_id for e in state.evidence}
    goals_without_evidence = [g for g in goal_ids if g not in evidenced]
    grounding_ratio = (
        round((len(goal_ids) - len(goals_without_evidence)) / len(goal_ids), 3)
        if goal_ids else 0.0)
    return {"goal_ids": goal_ids,
            "corpus_recall": corpus_recall,
            "goals_without_evidence": goals_without_evidence,
            "grounding_ratio": grounding_ratio}


def _warn_uncited(state: ResearchState, settings: Settings) -> None:
    """D-66 backstop: did a report with evidence ship citing nothing?

    The deterministic critic gate is meant to stop a zero-citation report
    before it ships, but a run with HITL disabled and its revision budget
    exhausted still ships whatever the gate last rejected (raise_or_log's
    stub path -- "log loudly and ship the report unreviewed" is this
    codebase's existing posture for an exhausted budget). This is the
    last-line-of-sight check for that case: purely observational, WARNs
    rather than blocks.

    Gated off in stub mode for the same reason the critic gate is:
    StubClient's fixed placeholder report never carries [gN] markers by
    design. Every check in this module carries that same gate.
    """
    if (settings.llm_mode != "stub" and state.evidence
            and not cited_goal_ids_in_prose(state.final_report)):
        log_event(logger, "report.shipped_with_no_citations",
                  level=logging.WARNING,
                  evidence_items=len(state.evidence))


def _audit_figures(state: ResearchState, settings: Settings) -> tuple:
    """D-91: the cited-figure audit. For every sentence that states a
    figure AND cites a goal, does any evidence under that goal actually
    contain the figure?

    RETURNS (findings, counters).

    Run at telemetry time rather than in compiler_node for D-59's reason:
    this is a property of the artifact, and compiler_node runs once per
    revision with additively-merged counters, so a count taken there would
    describe the compile attempts (exactly the D-88 problem). Taken
    against state.final_report it is report-scoped by construction.
    """
    findings: list = []
    counters: Dict[str, float] = {}
    if settings.llm_mode != "stub":
        findings, counters = audit_cited_figures(
            state.final_report, state.goals, state.evidence)
    # D-174: the audit examined nothing, and there WERE figures it could
    # not reach. Distinct from a clean report, and the two were until
    # now indistinguishable in telemetry -- both reported
    # cited_figures_checked: 0. WARNING rather than a finding: nothing
    # here says the report is wrong, only that this check did not get
    # to judge it, which an operator reading a confident-looking run
    # record needs to know. CAP_UNSUPPORTED_FIGURES in
    # reporting/confidence.py cannot fire on a run in this state.
    outside = int(counters.get("figures_outside_citation_scope", 0))
    if outside and not counters.get("cited_figures_checked"):
        log_event(logger, "report.figure_audit_saw_nothing",
                  level=logging.WARNING,
                  figures_outside_citation_scope=outside,
                  effect="no cited figure could be checked; the "
                         "unsupported-figure confidence cap cannot "
                         "apply to this run")
    # D-179: reported separately because the remedies are opposite.
    # An operator reading "unsupported" reaches for the report; an
    # operator reading "misattributed" reaches for the citation, and
    # the figure itself is fine. Merging them cost a true figure
    # live (p205.308-check).
    misattributed = [f for f in findings
                     if f.get("kind") == "misattributed"]
    if misattributed:
        log_event(logger, "report.misattributed_cited_figures",
                  level=logging.WARNING,
                  misattributed=int(counters.get(
                      "cited_figures_misattributed", 0)),
                  checked=int(counters.get("cited_figures_checked", 0)),
                  effect="the figures are supported by evidence this "
                         "run retrieved, under a goal the sentence "
                         "does not cite; the citation is wrong, not "
                         "the figure",
                  examples=misattributed[:5])
    findings = [f for f in findings if f.get("kind") != "misattributed"]
    if findings:
        log_event(logger, "report.unsupported_cited_figures",
                  level=logging.WARNING,
                  unsupported=len(findings),
                  checked=int(counters.get("cited_figures_checked", 0)),
                  # Capped: a report can state many figures, and one log
                  # line should stay readable. The full count is the field
                  # above; telemetry carries the same capped sample for the
                  # run record.
                  examples=findings[:5])
    return findings, counters


def _residual_repairs(state: ResearchState, settings: Settings) -> tuple:
    """What the two citation repairs LEFT, not what they removed.

    RETURNS (residual_pastes, residual_glue).

    D-99: a run reporting 21 removals and a run reporting 21 removals with
    four pastes still standing are indistinguishable in telemetry
    otherwise -- and live (run p205.253-check) the second one is what
    shipped.

    D-137: the same question asked of the OTHER signature. The paste
    counter read 0 on two shipped reports carrying 9 and 22 welded
    sentence joins, because a paste was the only thing it could see. This
    reads the artifact with the signature that matched them, so a repair
    that stops working cannot present as a clean report.
    """
    pastes = glue = 0
    if settings.llm_mode != "stub":
        pastes = residual_paste_sites(state.final_report, state.evidence)
        glue = residual_glue_sites(state.final_report)
    if pastes:
        log_event(logger, "report.residual_pasted_evidence",
                  level=logging.WARNING, sites=pastes,
                  removed=int(state.last_compile_guardrails.get(
                      "citations_pasted_evidence_removed", 0)))
    if glue:
        log_event(logger, "report.residual_glued_sentences",
                  level=logging.WARNING, sites=glue,
                  repaired=int(state.last_compile_guardrails.get(
                      "citations_glued_sentences_repaired", 0)))
    return pastes, glue


def shipped_report_metrics(state: ResearchState, settings: Settings,
                           evidence_by_source: Dict[str, int],
                           ) -> Dict[str, Any]:
    """Everything telemetry_node needs that is derived from the SHIPPED
    report or from the run's goals, plus the five WARNINGs that read them.

    CALLED BY   agents/compilation.py::telemetry_node, once, near the top.
                It unpacks the result into locals rather than threading a
                dict through the telemetry literal below it -- deliberately,
                so that extraction changed no line of the telemetry contract
                itself and a key-set comparison could prove it.
    RETURNS     goal_ids, corpus_recall, goals_without_evidence,
                grounding_ratio, web_sources_listed, web_sourced_items,
                grounding_notice_shipped, figure_findings, figure_counters,
                residual_pastes, residual_glue.
    LOGS        report.shipped_with_no_citations (D-66),
                report.unsupported_cited_figures (D-91),
                report.residual_pasted_evidence (D-99),
                report.residual_glued_sentences (D-137),
                report.shipped_ungrounded (D-85) -- in that order, which is
                the order they fired in before this was extracted.
    """
    metrics = goal_coverage(state, settings)

    # D-59: derived from the shipped report, not from the additive counter
    # dict -- see the module docstring for the two-rewrite run that made
    # the difference visible.
    metrics["web_sources_listed"] = count_listed_sources(state.final_report)
    metrics["web_sourced_items"] = sum(1 for e in state.evidence
                                       if e.source == "web")

    _warn_uncited(state, settings)

    # D-85, same last-line-of-sight shape: did a run whose corpus
    # contributed too little actually ship saying so? Read from the SHIPPED
    # report (D-59's rule), never from a counter.
    metrics["grounding_notice_shipped"] = report_carries_grounding_notice(
        state.final_report)

    findings, counters = _audit_figures(state, settings)
    metrics["figure_findings"] = findings
    metrics["figure_counters"] = counters

    pastes, glue = _residual_repairs(state, settings)
    metrics["residual_pastes"] = pastes
    metrics["residual_glue"] = glue

    if (settings.llm_mode != "stub" and state.evidence and state.goals
            and metrics["corpus_recall"] < settings.grounded_recall_target):
        log_event(logger, "report.shipped_ungrounded",
                  level=logging.WARNING,
                  corpus_recall=metrics["corpus_recall"],
                  grounded_recall_target=settings.grounded_recall_target,
                  notice_shipped=metrics["grounding_notice_shipped"],
                  web_sourced_items=metrics["web_sourced_items"],
                  model_sourced_items=int(evidence_by_source.get("model", 0)))
    return metrics
