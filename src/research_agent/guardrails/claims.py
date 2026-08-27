"""
guardrails/claims.py -- the cited-figure audit (D-91).

Purpose:
    For every sentence that states a FIGURE and cites a goal, check
    whether any evidence filed under that goal actually contains the
    figure. Report what does not match. Deterministic, no LLM call.

The gap this closes, and the gap it does NOT:
    README's Limitations has said for several revisions that "none of
    this is a programmatic, claim-by-claim relevance check". Everything
    aimed at the problem so far judges either the WHOLE report (the
    critic, D-43/D-46), or the EVIDENCE SET behind it (grounded_score
    D-47, corpus_recall D-43, the provenance notice D-85) -- nothing
    checks an individual claim against the specific evidence it cites.

    This does, for one narrow class of claim, chosen because it is the
    class where fabrication actually costs the reader something and the
    only one a machine can settle without judgement: a number. D-41's
    anti-fabrication limits and D-51's hedging both already single out
    figures for the same reason.

    It does NOT verify that cited evidence SUPPORTS a claim in any
    semantic sense. A sentence can carry a figure that appears verbatim
    in its cited evidence and still misread it. That check needs meaning,
    which is the critic's job (D-43/D-46) and is explicitly not
    reintroduced here.

Why this is now feasible, when telemetry_node once rejected it:
    telemetry_node carries a long, evidence-backed refusal of report
    PARSING (see its own comment): across four live runs the model cited
    goal ids four different ways -- `[g1 | corpus | score=0.50]`,
    `[g1, g4]`, `(g1)` in headings -- and one run used no bracket
    citations at all, so a regex would have been silent on the least
    grounded report of the set. That refusal was correct then.

    What changed since is that `[gN]` is now ENFORCED rather than
    requested: D-40 fixed the form, D-43/D-45's clean_citations repairs
    it deterministically, and D-66 fails a report that cites nothing at
    all. Citation shape is no longer a thing to hope for. This module
    still depends on that enforcement rather than re-deriving it -- it
    calls guardrails/sources.py::cited_goal_ids, the same single
    definition of "cited" compiler_node's evidence_cited count and
    critic_node's D-66 gate already share.

WARN-ONLY, deliberately, and this is a considered position rather than
timidity:
    Every guardrail added since Phase 1 that could not PROVE its finding
    ships observational -- G1 (retrieval floor), G4 (quality judge), G7
    (call budget) all say so in their own comments, citing D-54: do not
    build enforcement against a failure mode nothing has measured yet.
    This check has a real false-positive surface (see `_substantive`
    below for the shapes deliberately excluded, and the ones that will
    still slip through), and nothing has yet measured how often it fires
    on a good report.

    Making it fail a critique would also burn revision budget on a
    finding a rewrite frequently CANNOT fix -- the same objection that
    made D-85 a notice rather than a gate. Measure first. If the numbers
    justify enforcement later, the check itself is already written and
    the move is to call it from compiler_node instead.

CALLED BY   agents/compilation.py::telemetry_node, against the SHIPPED
            report -- joining report.shipped_with_no_citations (D-66) and
            report.shipped_ungrounded (D-85) as a last-line-of-sight
            check on the artifact the reader received.
            Running it there rather than in compiler_node is not
            incidental: compiler_node runs once per REVISION and its
            counters merge additively, so a count taken there describes
            the compile attempts (the D-88 problem). Taken here, against
            state.final_report, it is report-scoped by construction --
            D-59's rule.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from research_agent.guardrails.grounding import NOTICE_MARKER
from research_agent.guardrails.sources import SOURCES_HEADING, cited_goal_ids
from research_agent.state import Evidence, Goal

# Sentence boundaries, deliberately crude: a period/question/exclamation
# followed by whitespace. Markdown reports are not prose novels -- there
# are no "Dr." or "e.g." minefields in a generated research report often
# enough to justify a real segmenter dependency, and a mis-split only
# ever changes WHICH sentence a figure is attributed to, never whether
# the figure is supported (the evidence lookup is per cited goal, and
# citations travel with their sentence).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# A number, optionally with thousands separators and a decimal part,
# optionally followed by a percent marker. The trailing group is what
# rescues short percentages ("45%") from the length rule in
# _substantive below -- a percentage is a claim at any magnitude.
_FIGURE_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(%|percent|per cent)?",
                        re.IGNORECASE)


def _substantive(raw: str, percent_marker: Optional[str]) -> bool:
    """Is this number a CLAIM, or incidental formatting?

    Kept narrow on purpose -- a noisy guardrail is worse than no
    guardrail, because it trains people to ignore the output (the same
    reasoning check_services.py gives for reporting a disabled MCP as
    SKIPPED rather than FAIL).

    Counted as a claim:
      - anything with a decimal point   (8.9, 1.2)
      - anything with a percent marker  (45%, 45 percent)
      - any run of 3+ digits            (230, 2023, 1400000)
      - anything written with thousands separators (2,300)

    Deliberately NOT counted:
      - bare one- and two-digit integers. These are overwhelmingly
        markdown list numbering, heading numbers, "the 3 goals below",
        and counts the report states about itself. Their claim density
        is low and their formatting density is high, so including them
        would bury the real findings.

    Known to still slip through, stated rather than hidden: a year
    inside a proper noun ("Fortune 500"), a version number, and a port
    number all read as three-digit figures. None of them is common in a
    research report's prose, and every one of them shows up in the
    finding text where a reader can see what it was -- which is the
    posture G1/G4/G7 already take toward their own imprecision.
    """
    normalised = raw.replace(",", "")
    if percent_marker:
        return True
    if "." in normalised:
        return True
    if "," in raw:
        return True
    return len(normalised) >= 3


def figures_in(text: str) -> Set[str]:
    """Every substantive figure in `text`, normalised for comparison.

    Normalisation strips thousands separators, so "2,300" and "2300"
    compare equal -- the report and its evidence routinely format the
    same number differently, and a mismatch on punctuation would be a
    false positive every time.

    Percent markers are dropped from the KEY rather than kept, so "8.9%"
    matches an evidence item saying "8.9 percent". The figure is what is
    being checked; the unit is prose.
    """
    found: Set[str] = set()
    for match in _FIGURE_RE.finditer(text):
        raw, percent_marker = match.group(1), match.group(2)
        if _substantive(raw, percent_marker):
            found.add(raw.replace(",", ""))
    return found


def report_body(report: str) -> str:
    """The report's own prose, with deterministically-appended blocks removed.

    Two things this module must not audit, because neither is the
    model's writing and both are full of numbers:

      - the `## Sources` block (D-57), whose entries carry numbering and
        URLs that routinely contain digits.
      - the provenance notice (D-85), which states counts ABOUT the
        report ("None of this report's 2 research goal(s)...").

    Auditing either would produce findings against text this codebase
    itself generated, which is the fastest possible way to make a
    guardrail untrustworthy.
    """
    body = report
    head, marker, _block = body.rpartition(f"\n{SOURCES_HEADING}\n")
    if marker:
        body = head
    if NOTICE_MARKER in body:
        # The notice is a run of blockquote lines at the very top (see
        # guardrails/grounding.py). Drop leading "> " lines only -- a
        # blockquote the MODEL wrote later in the report is its own
        # prose and stays in scope.
        lines = body.splitlines(keepends=True)
        while lines and lines[0].startswith(">"):
            lines.pop(0)
        body = "".join(lines)
    return body


def audit_cited_figures(report: str, goals: List[Goal],
                        evidence: List[Evidence]
                        ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Find cited sentences stating a figure no cited evidence contains.

    RETURNS (findings, counters).

      findings  one dict per unsupported figure:
                {"figure": "2300000", "goals": ["g1"], "sentence": "..."}
                Capped preview text, so a finding is loggable without
                dumping a paragraph into a JSON log line.
      counters  {"cited_figures_checked": n, "cited_figures_unsupported": m}

    ONLY sentences that carry a citation are examined. An UNcited
    sentence is a different failure with its own existing checks
    (D-40's attribution rule, D-66's zero-citation gate), and treating
    the two as one would make this number impossible to act on.

    A sentence citing a goal with NO evidence at all is skipped rather
    than reported: D-45's clean_citations already strips exactly those
    markers, and telemetry's goals_without_evidence already counts the
    condition. Reporting it again here would be a third count of one
    fact.
    """
    counters = {"cited_figures_checked": 0.0, "cited_figures_unsupported": 0.0}
    if not report or not evidence:
        return [], counters

    known_goal_ids = {g.goal_id for g in goals}
    figures_by_goal: Dict[str, Set[str]] = {}
    for item in evidence:
        if item.goal_id not in figures_by_goal:
            figures_by_goal[item.goal_id] = set()
        figures_by_goal[item.goal_id] |= figures_in(item.content)

    findings: List[Dict[str, object]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(report_body(report)):
        cited = cited_goal_ids(sentence)
        if not cited:
            continue
        # Only goals that exist AND retrieved something can settle a
        # figure either way -- see the docstring on why the other cases
        # are somebody else's count.
        #
        # Membership (`g in figures_by_goal`), NOT truthiness: a goal
        # whose evidence contains no figures at all still HAS evidence,
        # and a figure cited to it is unsupported by definition. Testing
        # `figures_by_goal.get(g)` conflated the two and silently skipped
        # the single most important case this module exists to catch --
        # caught by test_findings_name_the_figure_the_goals_and_a_bounded_sentence,
        # whose evidence ("The PLA is large.") deliberately states no
        # figure at all.
        checkable = {g for g in cited
                     if g in known_goal_ids and g in figures_by_goal}
        if not checkable:
            continue
        supported: Set[str] = set()
        for goal_id in checkable:
            supported |= figures_by_goal[goal_id]
        # Citations carry no figures of their own under D-40's [gN]-only
        # rule, but a report that slipped a `[g1 | corpus | score=0.90]`
        # form past clean_citations would otherwise contribute "0.90"
        # as a claim. Strip the markers before reading figures.
        prose = re.sub(r"\[g\d+[^\]]*\]", " ", sentence)
        stated = figures_in(prose)
        if not stated:
            continue
        counters["cited_figures_checked"] += float(len(stated))
        for figure in sorted(stated - supported):
            findings.append({
                "figure": figure,
                "goals": sorted(checkable),
                "sentence": " ".join(sentence.split())[:200],
            })
    counters["cited_figures_unsupported"] = float(len(findings))
    return findings, counters