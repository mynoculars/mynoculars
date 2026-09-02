"""
guardrails/critique.py -- resolving a critic verdict the evidence
contradicts (D-155).

THE FAILURE THIS EXISTS FOR.

`critic_node` took the LLM's verdict unconditionally:

    passed = bool(result.get("passed", False))

That is the ONLY place in this codebase where an LLM judgement has no
deterministic counterweight. D-66's zero-citation gate, D-51's hedging
enforcement and D-91's figure audit all exist because the same rule keeps
proving itself -- check deterministically where possible, ask an LLM only
where a mechanical check genuinely cannot judge -- and the pass/fail
decision itself never got that treatment.

Live (run p205.287-check), the critic emitted seven notes. THREE were
supportive:

    "Goal g1: The report's figures for active personnel (China ~2-2.1 M;
     India ~1.4 M) are supported by evidence [g1]."
    "Goal g2: ... supported by multiple evidence items [g2]."
    "Goal g3: ... supported [g3]."

and then it failed the report on four of this shape:

    "Unfaithful: The report claims the PLA 'fields the world's largest
     standing military with approximately 2 million personnel'. The
     evidence states 'approximately 2 million to 2.1 million', so the
     report's 'approximately 2 million' is a paraphrase that omits the
     upper bound and is therefore not faithful."

That is a correct summary marked wrong for rounding. Checked against the
run's own evidence, ALL FOUR objections were false on their own stated
test -- 2.1, 1.23, 1948 and 2015 each appear verbatim in an evidence item,
including the 1948 the note asserts "no evidence item supports".

Meanwhile `guardrails/claims.py::audit_cited_figures` -- a deterministic
check answering the SAME question about the SAME report -- reported
`cited_figures_checked: 3, cited_figures_unsupported: 0`.

So two checks disagreed and the LLM won unconditionally. Two of the three
most recent runs ended E4 -> human approval on notes of this kind, which
makes a system whose headline claim is "converges, or halts honestly"
halt on almost every run.

WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT.

It does NOT overrule the critic. It resolves a disagreement between two
checks that answer the same question, in favour of the deterministic one
-- and only when the critic's own premise is falsifiable and false.

A note is DISMISSED only when every figure it names appears verbatim in
some evidence item. That directly falsifies the prompt's own bar ("a
FIGURE ... that appears in no evidence item of any source"): if the figure
IS there, the note is not reporting the thing it was asked to report.

A note this module cannot adjudicate SURVIVES, and a single survivor stops
the verdict being changed at all:

  - a note naming no figure at all -- "the report never addresses goal g3",
    or "the report says India's army is larger, which the evidence
    contradicts". Those are coverage and semantic findings, which are
    exactly what an LLM critic is FOR and what no deterministic check can
    make. They must stand.
  - a note naming a figure genuinely absent from the evidence. That is the
    critic doing its job correctly.

One consequence is worth stating rather than discovering: a note that
quotes BOTH figures -- "the report says 1.4 million where the evidence
says 1.23 million" -- names 1.4, which is absent from the evidence
precisely because the report rounded it. That note survives. Nothing here
can tell which of the two figures is being disputed, and guessing would
mean dismissing a note on a figure that was never checked, which is this
module's own failure mode in reverse. Surviving is the correct answer.

The counterweight is therefore narrow by construction: it can only ever
dismiss an objection whose factual premise the evidence itself refutes.

NEVER SILENT. `critique_notes_dismissed` reaches telemetry and a WARNING
names every dismissed note, so a run where this fired is always
distinguishable from one the critic passed on its own.
"""

import logging
import re
from typing import Dict, List, Sequence, Tuple

from research_agent.logging_setup import log_event
from research_agent.state import Evidence

logger = logging.getLogger(__name__)

# Numeric tokens with at least three characters: 1948, 2015, 2.1, 1.23.
#
# WHY THREE, and why this is not guardrails/claims.py::figures_in. That
# function answers a different question -- which figures in REPORT PROSE
# are substantive enough to audit -- and deliberately drops bare years,
# because "2015" in a heading is formatting. Here the input is a critic's
# NOTE and a year is precisely the kind of claim being disputed, so years
# must count.
#
# The floor is three characters rather than one because a bare "2" or "31"
# appears in almost any evidence block by accident, and a rule that
# dismisses a note on an accidental match is worse than no rule. Three is
# the shortest token that carries enough signal to be worth matching:
# every figure in the four live notes that motivated this module (2.1,
# 1.23, 1948, 2015) clears it.
_DISPUTED_FIGURE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
MIN_FIGURE_CHARS = 3

# D-162: a number that POINTS AT PART OF THE REPORT is not a figure the
# critic is disputing, and counting it as one turned this module into the
# opposite of what it was built to be.
#
# Live shape, at shipped defaults:
#
#   note:     "Goal g3 (defence budgets) is never addressed; section 2.1
#              of the report is missing entirely."
#   evidence: "China fields approximately 2.1 million active personnel."
#
# "2.1" is a SECTION NUMBER. It clears MIN_FIGURE_CHARS, it appears
# verbatim in the evidence blob, so `falsified_by_evidence` reported the
# note refuted -- and being the only note, it took the whole failing
# verdict with it. The report shipped as a clean pass and routed to
# memory_writer, which D-24 exists to prevent. This module's own docstring
# promises the opposite: "a note naming no figure at all is a coverage
# finding an LLM critic is FOR, it survives".
#
# A coverage note names no figure. It names a LOCATION, and a location is
# introduced by one of a small closed set of words. Stripping those
# references before extracting figures leaves the note with an empty
# figure set, which is exactly the "unadjudicatable, therefore safe"
# state disputed_figures already documents.
#
# Deliberately narrow: only a number IMMEDIATELY following one of these
# words is dropped. "the 2.1 million figure in section 3" keeps 2.1
# million -- that note really is disputing a figure, and it should still
# be adjudicable.
_LOCATION_REFERENCE_RE = re.compile(
    r"\b(?:section|sections|part|parts|step|steps|figure|figures|fig\.?|"
    r"table|tables|heading|headings|item|items|point|points|paragraph|"
    r"paragraphs|line|lines|page|pages|bullet|bullets|goal|goals|"
    r"appendix|chapter|note|notes)\s+#?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE)


def disputed_figures(note: str) -> set:
    """Numeric tokens a critic note names, normalised for comparison.

    Returns an EMPTY set for a note that names none -- which is what makes
    a coverage or semantic finding unadjudicatable here, and therefore
    safe from dismissal.
    """
    found = set()
    # D-162: drop "section 2.1"-shaped references first -- see
    # _LOCATION_REFERENCE_RE for the live note that made this necessary.
    scrubbed = _LOCATION_REFERENCE_RE.sub(" ", note or "")
    for raw in _DISPUTED_FIGURE_RE.findall(scrubbed):
        cleaned = raw.replace(",", "").rstrip(".")
        if len(cleaned) >= MIN_FIGURE_CHARS:
            found.add(cleaned)
    return found


def _evidence_text(evidence: Sequence[Evidence]) -> str:
    """One searchable blob, with separators stripped so "1,234" in the
    evidence matches "1234" from a note."""
    joined = " ".join(e.content or "" for e in evidence)
    return joined.replace(",", "")


def falsified_by_evidence(note: str, evidence_text: str) -> bool:
    """Does the evidence refute this note's own premise?

    True only when the note names at least one figure AND every figure it
    names is present in the evidence. A note naming no figure is never
    falsified here -- this function cannot judge it, and says so by
    returning False.
    """
    figures = disputed_figures(note)
    if not figures:
        return False
    return all(f in evidence_text for f in figures)


def resolve_verdict(passed: bool, notes: List[str],
                    evidence: Sequence[Evidence],
                    unsupported_figures: int,
                    ) -> Tuple[bool, List[str], Dict[str, float]]:
    """Let a failing verdict stand, unless the evidence refutes every note.

    -> (passed, notes, counters)

    RETURNS the inputs UNCHANGED on every path that decides not to act --
    the critic passed, there are no notes, the deterministic audit found a
    genuinely unsupported figure, or any note survived. That no-op path is
    the healthy one and must stay exact.

    `unsupported_figures` is D-91's own count for this report. A nonzero
    value means the deterministic check AGREES something is unsupported,
    and the verdict stands however the notes read -- the two checks are
    only in conflict when one of them is clean.
    """
    if passed or not notes or unsupported_figures:
        return passed, notes, {}

    text = _evidence_text(evidence)
    surviving = [n for n in notes if not falsified_by_evidence(n, text)]
    dismissed = len(notes) - len(surviving)
    if surviving or not dismissed:
        # Either the critic said something this cannot adjudicate, or it
        # said something the evidence supports. Both stand.
        return passed, notes, {}

    log_event(logger, "critic.failure_not_corroborated",
              level=logging.WARNING,
              dismissed=dismissed,
              notes=notes,
              effect="every note disputes a figure that IS present in the "
                     "evidence, and the D-91 audit found no unsupported "
                     "cited figure, so the failure had no corroboration; "
                     "the report passes and this line is the record")
    return True, notes, {"critique_notes_dismissed": float(dismissed)}
