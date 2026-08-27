"""
guardrails/grounding.py -- the deterministic provenance notice (D-85).

Purpose:
    When a run finishes with too few of its goals backed by a real
    ingested DOCUMENT, say so IN THE REPORT -- deterministically, in the
    artifact the reader actually receives, rather than only in telemetry
    the reader never sees.

The gap this closes, stated precisely:
    This codebase measures grounding unusually well. `grounded_score`
    (D-47), `corpus_recall` (D-43), `web_sourced_items` and
    `model_sourced_items` (D-38/D-57) all exist specifically so a run
    answered from recollection or from the open web reads as such rather
    than as a corpus-backed result. Every one of those numbers was
    correct on run p205.246-check: `grounded_score 0.0`, the whole answer
    carried by the web tier, `retrieval.floor_starvation` WARNing that
    100% of dense candidates were dropped.

    And the run still finished `Final status: SUCCESS`, critique passed
    on the first pass, 15,154 characters shipped -- with nothing in the
    report itself saying the corpus had contributed nothing. The
    measurement was honest and the DELIVERABLE was not. Telemetry is read
    by whoever runs the agent; the report is read by whoever asked the
    question, and those are not always the same person.

Why a notice and NOT a critique failure:
    The obvious alternative -- reuse D-66's shape and fail the critique
    deterministically when grounding is short -- was considered and
    rejected, for two independent reasons.

    1. A rewrite cannot fix it. D-66's zero-citation gate works because
       missing [gN] markers ARE a writing defect: recompiling the same
       evidence with a clearer instruction genuinely can produce a cited
       report. Grounding is not a property of the writing at all -- it is
       a property of what RETRIEVAL found. Failing the critique would
       spend a full extra compile cycle (a real LLM call against the
       largest prompt in the run) on a finding no rewrite can remedy.
       That is D-44's lesson verbatim: a demand for new evidence cannot
       be served by recompiling the same evidence block.

    2. An ungrounded answer is not, by this codebase's own rules, a
       failed one. D-38 deliberately permits the model tier, and D-57
       deliberately permits the web tier, with attribution -- reporting a
       corpus miss as an absence of knowledge is the single worst failure
       mode this project has already fixed once. Failing the critique
       here would re-punish the exact path those decisions built.

    So the harm being prevented is narrower than "the report is wrong".
    It is "the report is INDISTINGUISHABLE from a corpus-backed one".
    That is an honesty problem, and this codebase already has an
    established answer for honesty problems the prompt cannot be trusted
    to solve: enforce it deterministically in the artifact afterwards.
    D-51 did it for hedging (`(unverified figure)` markers) and D-57 did
    it for attribution (the `## Sources` section) -- both after a prompt
    instruction was measured being followed unreliably. This is the third
    application of the same pattern, and it costs no LLM call.

CALLED BY   agents/compilation.py::compiler_node, LAST -- after
            clean_citations, enforce_hedging AND append_web_sources.
            Order matters for the same reason sources.py documents for
            itself: the first two search the report for literal spans of
            evidence content, so generated text they were never meant to
            inspect belongs out of their reach. Running last also means
            the notice cannot perturb the Sources block that
            count_listed_sources (D-59) parses back out.

Two deliberate constraints on the notice's own text, both load-bearing:
    NO `[gN]` MARKERS. `cited_goal_ids` (guardrails/sources.py) is read
    by compiler_node's `evidence_cited` count AND by critic_node's D-66
    zero-citation gate. A citation-shaped string here would inflate the
    first and, worse, could let a report that cites nothing at all slip
    past the second on the strength of a marker the model never wrote.

    NO `##`-`######` HEADING. `count_sections` (reporting/metrics.py) is
    read by both the `node.compiled` log line and cli.py's terminal
    RESULT block -- S-10 exists because those two once disagreed. A
    blockquote renders prominently in Markdown and leaves that count
    describing the model's own structure, which is what it is for.
"""

from typing import Dict, List, Tuple

from research_agent.guardrails.retrieval import has_grounded_evidence
from research_agent.retrieval.terms import distinctive_terms
from research_agent.state import Evidence, Goal

# The literal opening of the notice, used for two things: making the
# insertion idempotent (below) and letting telemetry detect, from the
# SHIPPED text, whether the notice actually made it into the report --
# see report_carries_grounding_notice.
#
# A constant rather than a repeated literal for D-59's reason: the
# producer and the detector must share one definition, or they drift and
# telemetry starts describing an artifact nobody received.
NOTICE_MARKER = "**Provenance notice"


def grounded_goal_count(goals: List[Goal], evidence: List[Evidence],
                        min_evidence_score: float) -> int:
    """How many goals a real, on-topic DOCUMENT actually backs.

    Delegates the per-goal verdict to has_grounded_evidence
    (guardrails/retrieval.py) -- the M-1 single implementation already
    shared by progress_checker_node's `grounded_score` and
    telemetry_node's `corpus_recall`. Reusing it is what makes the notice
    the report-side rendering of `corpus_recall` rather than a fourth,
    subtly different grounding number that can disagree with the other
    three.
    """
    return sum(
        1 for g in goals
        if has_grounded_evidence(g.goal_id, distinctive_terms(g.description),
                                 evidence, min_evidence_score))


def report_carries_grounding_notice(report: str) -> bool:
    """Whether the SHIPPED report text contains the notice.

    Derived from the report, never from a counter -- D-59's rule, and for
    its exact reason: compiler_node runs once per REVISION and its
    counters merge additively (state.py::merge_counters), so a counter
    would describe the compile attempts rather than the artifact. Live
    evidence for why that distinction is not theoretical is in
    count_listed_sources' own docstring.
    """
    return NOTICE_MARKER in report


def annotate_ungrounded_report(
        report: str,
        goals: List[Goal],
        evidence: List[Evidence],
        min_evidence_score: float,
        grounded_recall_target: float,
) -> Tuple[str, Dict[str, float]]:
    """Prepend a provenance notice when too few goals are document-backed.

    Returns (report, counters) -- the same shape every other guardrail
    pass in this package returns, so compiler_node can fold the counters
    in exactly as it already does for citations, hedging and sources.

    RETURNS THE REPORT BYTE-IDENTICAL, with zeroed counters, whenever the
    notice does not apply. That no-op path is the common one -- every run
    whose corpus genuinely answers the question takes it -- so it must be
    exactly unchanged, not merely similar. The conditions are:

      - no goals: there is nothing to be grounded, and a run that
        produced none already failed earlier and more loudly (D-21).
      - no evidence at all: a different, already-visible failure. D-66's
        gate draws the same line for the same reason -- a report that
        cites nothing because NOTHING was retrieved is not the failure
        mode being guarded against, and the report's own prose already
        says so.
      - grounding at or above target: the corpus did its job.
      - the notice is already present: idempotent, so a rewrite pass
        cannot stack two of them.

    `grounded_recall_target` is reused deliberately rather than adding a
    fourth grounding threshold to config.py. The project's own
    calibration caveat (README, and OPERATIONS.md Step 3) is that a
    threshold nobody measured is worse than no threshold -- inventing one
    here would ship exactly that.
    """
    counters: Dict[str, float] = {}
    if not goals or not evidence:
        return report, counters
    if report_carries_grounding_notice(report):
        return report, counters

    total = len(goals)
    grounded = grounded_goal_count(goals, evidence, min_evidence_score)
    if total and (grounded / total) >= grounded_recall_target:
        return report, counters

    # Phrased from counted facts only -- D-12's "aggregate, never invent"
    # applied to prose. The notice states WHAT IS MISSING and where the
    # rest of the answer came from; it never characterises the answer as
    # wrong, because by D-38/D-57 it is not.
    if grounded == 0:
        supported = f"None of this report's {total} research goal(s) are"
    else:
        supported = (f"Only {grounded} of this report's {total} research "
                     f"goal(s) are")
    # The second sentence is deliberately phrased to read correctly for
    # BOTH openings above -- "the rest" would be odd after "None of ...",
    # and naming a remainder count would repeat what the first sentence
    # already said.
    notice = (
        f"> {NOTICE_MARKER} — inserted automatically, not written by the "
        f"model.**\n"
        f"> {supported} supported by a document from the ingested corpus. "
        f"Everything not backed by a document rests on web search results "
        f"and/or the model's own general knowledge. Both are legitimate "
        f"sources here and are attributed in the text, but neither is "
        f"curated material — treat specific figures, dates and named "
        f"entities below as unverified unless a listed source confirms "
        f"them.\n\n")

    counters["grounding_notice_inserted"] = 1.0
    counters["grounding_notice_goals_ungrounded"] = float(total - grounded)
    return notice + report, counters
