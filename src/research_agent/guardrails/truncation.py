"""
guardrails/truncation.py -- the deterministic "stopped early" notice (D-132).

Purpose:
    When a run is cut short by its wall-clock deadline or its token
    budget, say so IN THE REPORT -- in the artifact the reader receives,
    not only in telemetry the reader never sees.

The gap this closes is D-85's, in a second place. That decision exists
because run p205.246-check shipped 15,154 characters with `grounded_score
0.0` and nothing in the deliverable admitting the corpus had contributed
nothing: the measurement was honest and the ARTIFACT was not. A run
stopped at depth 1 of 3 by a deadline has exactly the same shape --
thinner coverage than the question deserves, and a report that reads
like a complete answer. Telemetry is read by whoever runs the agent; the
report is read by whoever asked the question.

Why a notice and NOT a critique failure, again:
    A rewrite cannot add the gather laps a deadline stopped. This is
    D-85's own argument verbatim (and D-44's before it): a finding no
    rewrite can remedy must not spend a revision cycle -- least of all
    here, where the run is being stopped precisely because it has no
    budget left to spend.

Two deliberate constraints on the notice's text, both inherited from
guardrails/grounding.py because the same downstream readers apply:
    NO `[gN]` MARKERS -- `cited_goal_ids` feeds compiler_node's
    `evidence_cited` count and critic_node's D-66 zero-citation gate; a
    citation-shaped string here would inflate the first and could slip a
    report that cites nothing past the second.
    NO `##`-`######` HEADING -- `count_sections` feeds the `node.compiled`
    log line and cli.py's RESULT block (S-10 exists because those two
    once disagreed). A blockquote renders prominently and leaves that
    count describing the model's own structure.

ORDER, in compiler_node: this runs LAST, after D-85's provenance notice,
which means its blockquote ends up ABOVE that one. Deliberate -- "this
run was stopped early" changes how a reader should weigh everything
below it, including the provenance notice itself.
"""

from typing import Dict, Optional, Tuple

# Same marker convention as guardrails/grounding.py: a fixed prefix the
# idempotency check below matches on, so a revision pass cannot stack two
# notices, and telemetry can report from the SHIPPED text (D-59) rather
# than from a counter that would sum every compile attempt.
NOTICE_MARKER = "**Run stopped early"

_REASON_TEXT = {
    "deadline": ("its configured wall-clock deadline"),
    "tokens": ("its configured token budget"),
}


def report_carries_truncation_notice(report: str) -> bool:
    """Whether the SHIPPED report text carries the notice (D-59's rule)."""
    return NOTICE_MARKER in report


def annotate_truncated_run(report: str, reason: Optional[str]
                           ) -> Tuple[str, Dict[str, float]]:
    """Prepend a "stopped early" notice when a run budget was spent.

    CALLED BY   agents/compilation.py::compiler_node, last of the report
                passes.
    RETURNS     (report, counters) -- the shape every guardrail pass in
                this package returns, so compiler_node folds the counters
                in exactly as it already does for citations, hedging,
                sources and grounding.

    RETURNS THE REPORT BYTE-IDENTICAL, with zeroed counters, whenever the
    notice does not apply -- and that no-op is the path EVERY run takes
    while both budgets are at their default 0. The conditions:

      - no reason: no budget was spent (or none is configured);
      - an unrecognised reason: this function will not invent prose for a
        stop condition it does not know about, and a silent no-op is
        safer than a notice that describes the wrong thing;
      - the notice is already present: idempotent, so the rewrite path
        (compiler runs once per revision) cannot stack two.

    The text states only what was measured -- which budget, and that the
    report was compiled from what had been gathered by then. It does not
    characterise the answer as wrong: a run stopped early can still have
    answered the question, and telemetry carries the numbers that say how
    far it got.
    """
    counters: Dict[str, float] = {}
    if not reason or reason not in _REASON_TEXT:
        return report, counters
    if report_carries_truncation_notice(report):
        return report, counters

    notice = (
        f"> {NOTICE_MARKER} — inserted automatically, not written by the "
        f"model.**\n"
        f"> This run reached {_REASON_TEXT[reason]} before its research "
        f"loop finished, and was compiled from the evidence gathered up "
        f"to that point. Coverage may therefore be thinner than the "
        f"question deserves — the run's telemetry records how far it got "
        f"(iterations, recall, corpus_recall) and what stopped it "
        f"(run_budget_exhausted).\n\n")

    counters["truncation_notice_inserted"] = 1.0
    return notice + report, counters
