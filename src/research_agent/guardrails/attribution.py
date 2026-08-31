"""
guardrails/attribution.py -- deterministic citation attachment (D-144).

Purpose:
    When the compiler ships a report with NO `[gN]` markers at all, work
    out which goal each prose sentence rests on and attach the marker the
    model failed to write -- but only where the evidence makes that
    unambiguous, and only ever as an ADDITION.

WHY THIS EXISTS, and why the three previous attempts were not enough.

D-40 asks the model for `[gN]` markers. Compliance has been the single
most-repaired behaviour in this codebase, and every fix so far has been
either a prompt instruction or a form-normaliser:

  - D-40      asked for the form in the prompt.
  - D-43/D-45 (clean_citations) repaired markers that were WRONG.
  - D-73      repeated the instruction next to the revision block, after
              runs p205.239/240 came back with evidence_cited 0 on the
              rewrite twice each.
  - D-99      (normalise_citation_form) converted `(g1)`, `[g1, g4]` and
              `[g1 | corpus | 0.5]` into the canonical form, after run
              p205.253 wrote its goal ids in parentheses.

None of them help when the model emits no goal id in any form. That has
now shipped four times -- p205.276, p205.277 and p205.280 each shipped a
report with zero markers against 35-100 evidence items -- and the damage
is never confined to the missing markers, because THREE separate
guardrails read attribution through sources.py::cited_goal_ids and all
three fail together and silently:

    p205.280-check, 100 evidence items:
      evidence_cited              0
      web_sources_listed          0     of 58 items, 33 distinct domains
      web_sources_suppressed     58
      cited_figures_checked       0     (D-91 audited nothing)
      report.shipped_with_no_citations  WARNING
      -> E4, and a human approved it because the prose read fine

The report still carried D-85's provenance notice telling the reader to
treat figures as unverified "unless a listed source confirms them", above
a page with no listed sources.

So this is the deterministic half that D-51's argument has always implied:
where a prompt instruction is unreliable and a mechanical check can decide,
check mechanically. Same shape as guardrails/hedging.py, which exists
because the prompt's hedging instruction was followed unreliably enough to
reach hedge_specific_items 29 with zero visible hedging.

SCOPE, DELIBERATELY NARROW. Four limits, each closing a way this could
assert something untrue:

  1. ALL-OR-NOTHING. Runs only when the report cites NOTHING. A partially
     cited report is a model making choices about which sentences rest on
     which evidence, and second-guessing those choices needs to read
     meaning. A report with zero markers is a formatting failure, and
     formatting failures are what deterministic passes are for.

  2. UNAMBIGUOUS ONLY. A sentence is attributed only when exactly ONE goal
     achieves the best term overlap. A tie attributes nothing -- see
     _best_goal. "Probably g1" is not a citation.

  3. A FLOOR, NOT A BEST GUESS. The winning overlap must reach
     MIN_TERM_OVERLAP distinctive terms. Without it, one shared word would
     be enough, and D-59's own motivating failure (nine Redis-monitoring
     URLs listed under [g1] in an India-vs-US report) is exactly what one
     shared word produces.

  4. IT ONLY ADDS. Nothing is deleted, reworded or moved. Every marker it
     writes then passes through clean_citations like any the model wrote,
     so an attachment to a goal that turns out to have no evidence is
     dropped by the existing guard rather than by a special case here.

It reuses distinctive_terms for "on topic" for the reason terms.py's own
docstring gives: one implementation means the answer cannot come to mean
four different things in four files.

COUNTED, NOT SILENT. `citations_attached` reaches telemetry and the D-88
per-report block, so a reader can always tell a report the model cited
from one this module rescued. That distinction matters and must never be
invisible.

CALLED BY   agents/compilation.py's report pipeline, between
            normalise_citation_form (which must settle the FORM first --
            a `(g1)` is not a citation to cited_goal_ids, so running
            before it would see a report that already had attribution and
            do nothing) and clean_citations (which then validates what
            this attached).
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from research_agent.guardrails.claims import (cited_goal_ids_in_prose,
                                              report_body)
from research_agent.logging_setup import log_event
from research_agent.retrieval.terms import distinctive_terms
from research_agent.state import Evidence, Goal

logger = logging.getLogger(__name__)

# How many distinctive terms a sentence must share with a goal's evidence
# before that goal may be cited for it.
#
# Two, not one. One shared term is the bar that produced D-59's motivating
# failure -- nine Redis-monitoring URLs listed under [g1] in an India-vs-US
# report -- and distinctive_terms already strips the scaffolding words that
# would otherwise make a single match meaningless. Three was measured
# against p205.280-check's shipped report and attributed noticeably less
# without attributing anything more accurately, so two is where this sits
# until a run argues otherwise.
MIN_TERM_OVERLAP = 2

# Sentences below this length are not claims. A four-word fragment
# ("Naval Forces", "In addition") can clear a two-term overlap on its own
# subject heading and gains nothing from a citation.
MIN_SENTENCE_CHARS = 40

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
# Sentence-final punctuation followed by whitespace. Deliberately the same
# shape guardrails/claims.py splits on, so the two passes agree about where
# a sentence ends.
_SENTENCE_END_RE = re.compile(r"([.!?])(\s|$)")


def _goal_term_index(goals: List[Goal],
                     evidence: List[Evidence]) -> Dict[str, Set[str]]:
    """goal_id -> the distinctive terms its OWN evidence actually contains.

    Built from evidence content, not from the goal description. The
    description is what the run set out to find; the evidence is what it
    found, and a citation is a claim about the latter. A goal whose
    retrieval came back empty gets no entry at all and can never be
    attached -- which is the same verdict clean_citations reaches
    afterwards, arrived at one pass earlier.

    Memory-sourced evidence is excluded. P2-02 namespaces it as
    "memory::gN", which is deliberately never a current goal id, so it
    contributes to no goal's index -- recalled text must not be able to
    make a sentence look supported by THIS run's retrieval.
    """
    index: Dict[str, Set[str]] = {}
    evidenced = {e.goal_id for e in evidence}
    for goal in goals:
        if goal.goal_id not in evidenced:
            continue
        terms: Set[str] = set()
        for item in evidence:
            if item.goal_id == goal.goal_id:
                terms |= distinctive_terms(item.content)
        if terms:
            index[goal.goal_id] = terms
    return index


def _best_goal(sentence: str,
               index: Dict[str, Set[str]]) -> Optional[str]:
    """The one goal this sentence unambiguously rests on, or None.

    None on all three of: nothing clears MIN_TERM_OVERLAP, two goals tie
    for best, or the index is empty. A tie is genuinely undecidable from
    term overlap alone and belongs to the critic, not here.
    """
    terms = distinctive_terms(sentence)
    if not terms:
        return None
    scored = sorted(((len(terms & goal_terms), goal_id)
                     for goal_id, goal_terms in index.items()),
                    reverse=True)
    if not scored or scored[0][0] < MIN_TERM_OVERLAP:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None  # tie: attribute nothing
    return scored[0][1]


def _attach_to_line(line: str, index: Dict[str, Set[str]]) -> Tuple[str, int]:
    """Attach markers to the qualifying sentences in one prose line."""
    out, attached, pos = [], 0, 0
    for match in _SENTENCE_END_RE.finditer(line):
        end = match.end(1)
        sentence = line[pos:end]
        if len(sentence.strip()) >= MIN_SENTENCE_CHARS:
            goal_id = _best_goal(sentence, index)
            if goal_id:
                out.append(f"{sentence} [{goal_id}]")
                attached += 1
                pos = end
                continue
        out.append(sentence)
        pos = end
    out.append(line[pos:])
    return "".join(out), attached


def attach_missing_citations(report: str, goals: List[Goal],
                             evidence: List[Evidence]
                             ) -> Tuple[str, Dict[str, float]]:
    """Attach `[gN]` markers to an uncited report. -> (report, counters).

    Returns the report BYTE-IDENTICAL, with empty counters, on every path
    that decides not to act -- the report already cites something, there
    are no goals with evidence, or no sentence cleared the bar. That
    no-op path is the common one on a healthy run and must stay exact, in
    the same spirit as append_web_sources' own no-op contract.
    """
    if cited_goal_ids_in_prose(report):
        return report, {}  # the model did its job; leave it alone

    index = _goal_term_index(goals, evidence)
    if not index:
        return report, {}

    # Only the model's own prose is eligible. report_body strips the
    # Sources block and the D-85 provenance notice for exactly the reason
    # D-139 gave: text this codebase generated is not text to annotate.
    body = report_body(report)
    if not body:
        return report, {}

    rewritten, attached = [], 0
    for line in body.splitlines(keepends=True):
        if _HEADING_RE.match(line) or not line.strip():
            # Headings carry ordinals, not claims (D-97), and
            # claims.py::iter_cited_sentences already refuses to treat
            # them as sentences. Citing one here would create a SCOPE
            # that silently attributes every sentence beneath it.
            rewritten.append(line)
            continue
        new_line, n = _attach_to_line(line, index)
        rewritten.append(new_line)
        attached += n

    if not attached:
        log_event(logger, "citations.attachment_found_nothing",
                  goals_with_evidence=len(index))
        return report, {}

    log_event(logger, "citations.attached", attached=attached,
              goals_with_evidence=len(index),
              min_term_overlap=MIN_TERM_OVERLAP)
    return report.replace(body, "".join(rewritten), 1), {
        "citations_attached": float(attached)}
