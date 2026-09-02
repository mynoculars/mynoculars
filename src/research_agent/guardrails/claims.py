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
    requested. That sentence was written before it was true, and the
    correction is worth keeping: this docstring used to credit
    D-43/D-45's clean_citations with "repairing the form
    deterministically", which it never did -- it drops markers for
    unevidenced goals and nothing more. Run p205.253-check proved the
    gap: the compiler wrote `(g1)` through `(g4)`, this module audited
    zero figures, and the Sources block listed zero of 78 web items.
    D-99 is the repair that was missing, and it now runs first in
    compiler_node's guardrail chain. This module still depends on that
    enforcement rather than re-deriving it -- it calls
    guardrails/sources.py::cited_goal_ids, the same single definition of
    "cited" compiler_node's evidence_cited count and critic_node's D-66
    gate already share.

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

# A markdown heading line: up to three spaces of indent, one to six
# hashes, whitespace. Headings are handled separately from prose (see
# iter_cited_sentences) for two independent reasons, both of them live
# findings from run p205.251-check:
#
#   1. A heading's ORDINAL is not a claim. "### 1.1 Active Personnel"
#      contributed the figure "1.1" -- it has a decimal point, so
#      _substantive waved it through -- and four of that run's five
#      findings were section numbers. The docstring on _substantive
#      already excluded heading numbers, but only the bare one- and
#      two-digit kind; "N.1" walked straight past the decimal rule.
#   2. A heading's CITATION governs the prose beneath it. That report
#      cited goals only in its headings ("## 1. Military Size and
#      Composition [g1]") and never inline, so with headings merely
#      stripped this module would have audited nothing at all. Scoping
#      the citation down to the section's sentences is what makes the
#      check work on a report that attributes by section -- and it
#      raises coverage sharply, because every sentence under a cited
#      heading is now in scope rather than only the one sentence that
#      happened to swallow the heading during splitting.
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s")

# A number, optionally with thousands separators and a decimal part,
# optionally followed by a percent marker. The trailing group is what
# rescues short percentages ("45%") from the length rule in
# _substantive below -- a percentage is a claim at any magnitude.
#
# A separator must be followed by exactly three digits (D-97). The
# looser `\d[\d,]*` swallowed ordinary sentence punctuation: live (run
# p205.251-check) "reforms since 2015, emphasizing" matched as the
# figure "2015,", whose comma then satisfied _substantive's
# thousands-separator rule and smuggled a year past the year exclusion
# below.
_FIGURE_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(%|percent|per cent)?", re.IGNORECASE)


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
      - bare four-digit integers that read as a YEAR (D-97). Live (run
        p205.251-check) "including the 1962 Sino-Indian War and periodic
        standoffs in 2017 and 2020" produced three findings from one
        sentence of uncontested history. A date is not the class of
        claim this module was built for -- D-41 and D-51 single out
        figures because a fabricated MAGNITUDE misleads a reader who
        cannot check it, and a wrong year is both rarer and cheaper.
        Dates are also the densest numeric feature of a research report,
        so leaving them in buries everything else. A year written with a
        separator or a decimal is not a year, and still counts.

    Known to still slip through, stated rather than hidden: a version
    number and a port number read as three-digit figures, and a figure
    the evidence states at a different SCALE ("2.5 million" against a
    source saying "2,535,000") reads as unsupported. None of the first
    two is common in a research report's prose, the third is a real
    limitation, and every one of them shows up in the finding text where
    a reader can see what it was -- which is the posture G1/G4/G7
    already take toward their own imprecision.
    """
    normalised = raw.replace(",", "")
    if percent_marker:
        return True
    if "." in normalised:
        return True
    if "," in raw:
        return True
    if len(normalised) == 4 and 1500 <= int(normalised) <= 2200:
        return False
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


# Magnitude words a report writes instead of digits (D-98).
#
# Single-letter forms ("2.1 m", "$78 B") are deliberately absent: "m"
# collides with metres and the rescue below is worth having only while
# it stays conservative.
#
# `lakh` and `crore` are absent for a stronger reason, and it is a
# limitation rather than an oversight. This corpus's subject matter uses
# them constantly, but two things about how they are written defeat the
# grammar here: Indian digit grouping ("2,10,000" = 210,000) does not
# fit the three-digit-group pattern _FIGURE_RE and _SCALED_RE share, and
# they COMPOUND -- "7.85 lakh crore" is 7.85e12, not 7.85e5. A one-word
# lookup gets both wrong, and a wrong scale is worse here than no scale:
# it could confirm a figure against evidence that says something else
# entirely. Left out, those figures simply carry no scale, which costs
# nothing -- a figure with no magnitude word gets an interval a fraction
# of a unit wide, so only its own exact value can confirm it, and the
# exact path already did that. Supporting them properly means parsing
# the grouping and the compound together, which is its own change.
_SCALE_WORDS = {"thousand": 1e3, "million": 1e6,
                "billion": 1e9, "trillion": 1e12}

# The same figure grammar as _FIGURE_RE, plus the percent marker and an
# optional trailing magnitude word.
_SCALED_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(%|percent|per cent)?"
    r"\s*(thousands?|millions?|billions?|trillions?)?",
    re.IGNORECASE)

# What may sit between the two halves of a range. "2.0-2.1 million"
# writes the magnitude word ONCE, at the end, so the first number has to
# inherit it or the whole rescue misses exactly the shape that motivated
# it -- live (run p205.252-check) all three findings were the "2.0" of
# "roughly 2.0-2.1 million personnel".
_RANGE_JOIN_RE = re.compile(r"^\s*(?:[-\u2013\u2014]|to|and|through)\s*$",
                            re.IGNORECASE)


def _scaled_matches(text: str):
    """[(raw, scale, percent_marker)] for every substantive figure.

    `scale` is the multiplier the figure's magnitude word implies, or
    None when it has none. Range members inherit the scale written after
    the range (see _RANGE_JOIN_RE).
    """
    hits = []
    for match in _SCALED_RE.finditer(text):
        raw, percent_marker, word = match.group(1, 2, 3)
        if not _substantive(raw, percent_marker):
            continue
        scale = None
        if word:
            scale = _SCALE_WORDS.get(word.lower().rstrip("s"))
        hits.append([match.start(), match.end(), raw, scale, percent_marker])
    for i in range(len(hits) - 2, -1, -1):
        if hits[i][3] is None and hits[i + 1][3] is not None:
            if _RANGE_JOIN_RE.match(text[hits[i][1]:hits[i + 1][0]]):
                hits[i][3] = hits[i + 1][3]
    return [(raw, scale, percent_marker)
            for _start, _end, raw, scale, percent_marker in hits]


def scaled_values(text: str) -> Set[float]:
    """Every substantive figure in `text` as an absolute VALUE.

    "2,035,000" and "2.035 million" both come back as 2035000.0, which is
    the whole point: the evidence and the report state the same quantity
    at whatever scale each source happened to use.

    Percentages are excluded. A percentage has no magnitude to rescale --
    "20%" against "20.33%" is a rounding question about the SAME scale,
    which the exact path already answers correctly by disagreeing.
    """
    values: Set[float] = set()
    for raw, scale, percent_marker in _scaled_matches(text):
        if percent_marker:
            continue
        # Rounded because a scale multiply is not exact in binary --
        # 2.035 * 1e6 is 2035000.0000000002 -- and these values are
        # compared against interval BOUNDARIES below. Six decimal places
        # is far finer than any quantity a report states and removes the
        # representation noise entirely.
        values.add(round(float(raw.replace(",", "")) * (scale or 1.0), 6))
    return values


def scaled_claims(text: str) -> Dict[str, Tuple[float, float]]:
    """{normalised figure: (low, high)} -- the interval each stated
    figure's OWN PRECISION claims.

    This is the rule that makes the rescue below safe, and it is worth
    stating carefully because a flat percentage tolerance would not be.
    A figure written to a given number of decimal places asserts a
    rounding interval and nothing tighter: "2.0 million" says the true
    value lies in [1.95m, 2.05m], so evidence reading 2,035,000 CONFIRMS
    it. "2.1 million" says [2.05m, 2.15m], and the same evidence refutes
    it. The report's own notation decides the width -- nothing is
    tuned, and a more precise claim is held to a proportionally
    stricter standard, which is the behaviour you want.

    Bare integers are self-limiting: only three digits or more are
    substantive (see _substantive), and at that length the interval is
    +/- 0.5, i.e. exact. The wide intervals belong to figures that
    openly declared themselves imprecise.
    """
    spans: Dict[str, Tuple[float, float]] = {}
    for raw, scale, percent_marker in _scaled_matches(text):
        if percent_marker:
            continue
        digits = raw.replace(",", "")
        multiplier = scale or 1.0
        value = round(float(digits) * multiplier, 6)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        half = round(0.5 * (10.0 ** -decimals) * multiplier, 6)
        low, high = round(value - half, 6), round(value + half, 6)
        if digits in spans:
            # The same figure written twice in one sentence at different
            # scales: keep the union, so the rescue is never made
            # stricter by a second mention.
            low = min(low, spans[digits][0])
            high = max(high, spans[digits][1])
        spans[digits] = (low, high)
    return spans


def _confirmed_by_scale(figure: str,
                        spans: Dict[str, Tuple[float, float]],
                        values: Set[float]) -> bool:
    """Does any evidence VALUE fall inside what this figure claims?"""
    span = spans.get(figure)
    if span is None:
        return False
    low, high = span
    return any(low <= value <= high for value in values)


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


def cited_goal_ids_in_prose(report: str) -> set:
    """Which goals the report's OWN PROSE cites (D-144).

    CALLED BY   agents/compilation.py -- the evidence_cited count, the D-66
                zero-citation gate and telemetry's shipped-with-no-citations
                backstop -- and guardrails/attribution.py's all-or-nothing
                gate.

    WHY THIS IS NOT sources.py::cited_goal_ids. That function matches
    `[gN]` anywhere in the string it is given, which is correct for the
    single line or single sentence claims.py hands it. Given a WHOLE
    report it also matches the Sources block, whose every entry begins
    "1. [g1] " by construction (D-57) -- so a report whose prose cites
    nothing at all, but which carries a Sources block, reads back as
    fully cited.

    That is a real defect and it predates D-144: evidence_cited would
    report 4 for a report with no markers in its prose, the D-66 gate
    would skip a report it exists to catch, and telemetry's backstop
    would agree with both. It was unreachable only because the block was
    itself gated on prose citations -- the exact coupling D-144 removes.
    Fixing the readers is what makes removing that coupling safe.

    report_body already strips both deterministically-appended blocks (the
    Sources section and D-85's provenance notice) for the same reason, so
    this is that boundary reused, not a second one.
    """
    return cited_goal_ids(report_body(report))


def iter_cited_sentences(body: str):
    """Yield (sentence, cited_goal_ids) over a report body's PROSE.

    Headings are never yielded as sentences -- their ordinals are
    formatting, not claims (D-97) -- but a heading that carries `[gN]`
    markers opens a SCOPE: every prose sentence beneath it is treated as
    citing those goals, until the scope closes.

    A scope closes at the next heading that either carries citations of
    its own (which replaces it) or sits at the same or a shallower depth
    (which ends it). A DEEPER uncited heading inherits, so "### 1.1
    Active Personnel" keeps the `[g1]` from "## 1. Military Size and
    Composition [g1]" above it, while "## 5. Historical Conflicts",
    which cites nothing and is no deeper than the section before it,
    correctly falls out of scope and is not audited.

    An inline citation on the sentence itself always wins over the
    scope. Sentences carrying neither are yielded with an empty set and
    skipped by the caller, exactly as before.
    """
    scope: Set[str] = set()
    scope_depth = 0
    for line in body.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            depth = len(heading.group(1))
            cited = cited_goal_ids(line)
            if cited:
                scope, scope_depth = cited, depth
            elif depth <= scope_depth:
                scope, scope_depth = set(), 0
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            if not sentence.strip():
                continue
            inline = cited_goal_ids(sentence)
            yield sentence, (inline or scope)


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
    values_by_goal: Dict[str, Set[float]] = {}
    for item in evidence:
        if item.goal_id not in figures_by_goal:
            figures_by_goal[item.goal_id] = set()
            values_by_goal[item.goal_id] = set()
        figures_by_goal[item.goal_id] |= figures_in(item.content)
        values_by_goal[item.goal_id] |= scaled_values(item.content)

    findings: List[Dict[str, object]] = []
    for sentence, cited in iter_cited_sentences(report_body(report)):
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
        supported_values: Set[float] = set()
        for goal_id in checkable:
            supported |= figures_by_goal[goal_id]
            supported_values |= values_by_goal[goal_id]
        # Citations carry no figures of their own under D-40's [gN]-only
        # rule, but a report that slipped a `[g1 | corpus | score=0.90]`
        # form past clean_citations would otherwise contribute "0.90"
        # as a claim. Strip the markers before reading figures.
        prose = re.sub(r"\[g\d+[^\]]*\]", " ", sentence)
        stated = figures_in(prose)
        if not stated:
            continue
        counters["cited_figures_checked"] += float(len(stated))
        # Exact string match first -- fast, and precise by construction.
        # Only what it could not settle goes to the scale comparison,
        # so a report and its evidence that already agree literally are
        # never subjected to interval arithmetic.
        unsupported = stated - supported
        if unsupported:
            spans = scaled_claims(prose)
            unsupported = {
                figure for figure in unsupported
                if not _confirmed_by_scale(figure, spans, supported_values)}
        for figure in sorted(unsupported):
            findings.append({
                "figure": figure,
                "goals": sorted(checkable),
                "sentence": " ".join(sentence.split())[:200],
            })
    counters["cited_figures_unsupported"] = float(len(findings))
    return findings, counters
