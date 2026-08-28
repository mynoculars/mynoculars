"""
tests/unit/test_guardrails_claims.py -- guardrails/claims.py (D-91).

WHY THIS FILE EXISTS: README's Limitations has said for several revisions
that "none of this is a programmatic, claim-by-claim relevance check".
D-91 adds one, for the single class of claim a machine can settle without
judgement -- a figure. These tests pin both halves of that: it must catch
a cited number that appears in no cited evidence, and it must NOT fire on
the many shapes of number that are formatting rather than claims. The
second half matters more: a guardrail that cries wolf is one people learn
to ignore, which is worse than not having it.
"""

from research_agent.guardrails.claims import (audit_cited_figures,
                                              figures_in, report_body,
                                              scaled_claims, scaled_values)
from research_agent.guardrails.grounding import annotate_ungrounded_report
from research_agent.state import Evidence, Goal


def _goal(goal_id="g1", description="PLA active personnel strength"):
    return Goal(goal_id=goal_id, description=description)


def _ev(content, goal_id="g1", source="corpus", score=0.9):
    return Evidence(task_key="t", goal_id=goal_id, source=source,
                    content=content, score=score)


# ---------------------------------------------------------------------------
# figures_in -- what counts as a claim, and what is formatting
# ---------------------------------------------------------------------------


def test_decimals_percentages_and_large_numbers_are_claims():
    assert figures_in("growth was 8.9 percent") == {"8.9"}
    assert figures_in("inflation hit 45%") == {"45"}
    assert figures_in("about 2,300 units") == {"2300"}
    assert figures_in("roughly 230 aircraft") == {"230"}


def test_a_bare_year_is_a_date_not_a_claim():
    """P205 regression (run p205.251-check, "Compare Armies of China and
    India"). "including the 1962 Sino-Indian War and periodic standoffs
    in 2017 and 2020" produced three findings out of one sentence of
    uncontested history, and dates are the densest numeric feature of a
    research report -- leaving them in buries the findings that matter.
    D-41 and D-51 single out figures because a fabricated MAGNITUDE
    misleads a reader who cannot check it; a wrong year is both rarer
    and cheaper. This was previously asserted the other way round."""
    assert figures_in("in 2023 the figure rose") == set()
    assert figures_in("the 1962 war and the 2020 standoff") == set()


def test_a_year_carrying_sentence_punctuation_is_still_a_year():
    r"""P205 regression (run p205.251-check). "reforms since 2015,
    emphasizing" matched as the figure "2015," under the old
    `\d[\d,]*` pattern -- prose punctuation read as a thousands
    separator, whose presence then satisfied _substantive and smuggled
    the year straight past the exclusion above. A separator now has to
    be followed by exactly three digits."""
    assert figures_in("reforms since 2015, emphasizing jointness") == set()
    assert figures_in("2,535,000 troops, deployed") == {"2535000"}


def test_a_year_sized_number_that_is_really_a_quantity_still_counts():
    """The exclusion is deliberately narrow: it only forgives a BARE
    four-digit integer. Written with a separator or a decimal, the same
    magnitude is a quantity again and stays in scope."""
    assert figures_in("2,020 aircraft") == {"2020"}
    assert figures_in("2020.5 tonnes") == {"2020.5"}


def test_short_bare_integers_are_formatting_not_claims():
    """Markdown list numbering, heading numbers and "the 3 goals below"
    dominate one- and two-digit integers in a generated report. Counting
    them would bury every real finding."""
    assert figures_in("1. First point") == set()
    assert figures_in("## 2. Findings") == set()
    assert figures_in("across all 4 goals") == set()


def test_thousands_separators_normalise_so_formatting_never_mismatches():
    """The report and its evidence routinely format the same number
    differently. A mismatch on punctuation would be a false positive
    every single time."""
    assert figures_in("2,300,000 troops") == figures_in("2300000 troops")


def test_a_percentage_matches_the_same_figure_written_as_prose():
    """8.9% in the report against "8.9 percent" in the evidence is one
    claim, not two -- the figure is what is checked, the unit is prose."""
    assert figures_in("8.9%") == figures_in("8.9 percent")


# ---------------------------------------------------------------------------
# report_body -- never audit text this codebase generated itself
# ---------------------------------------------------------------------------


def test_the_sources_block_is_out_of_scope():
    """D-57's Sources entries carry numbering and URLs full of digits.
    Auditing them would produce findings against our own output."""
    report = ("# R\n\nBody text [g1].\n\n## Sources\n\n"
              "1. [g1] Some Title (example.org) — https://example.org/a/1234\n")
    assert "example.org" not in report_body(report)
    assert "Body text" in report_body(report)


def test_the_provenance_notice_is_out_of_scope():
    """D-85's notice states counts ABOUT the report ("None of this
    report's 2 research goal(s)..."). Those are our numbers, not the
    model's claims."""
    annotated, _ = annotate_ungrounded_report(
        "# R\n\nBody text [g1].\n", [_goal(), _goal("g2", "Indian Army size")],
        [_ev("The PLA is large.", source="web", score=0.7)], 0.5, 0.5)

    body = report_body(annotated)

    assert "Provenance notice" not in body
    assert "Body text" in body


def test_a_blockquote_the_model_wrote_stays_in_scope():
    """Only LEADING blockquote lines are the notice. A blockquote further
    down is the model's own prose and must still be audited."""
    report = "# R\n\nIntro.\n\n> The PLA fields 2,300,000 troops [g1].\n"
    assert "2,300,000" in report_body(report)


# ---------------------------------------------------------------------------
# audit_cited_figures -- the check itself
# ---------------------------------------------------------------------------


def test_a_cited_figure_absent_from_its_evidence_is_reported():
    report = "# R\n\nThe PLA fields 2,300,000 active personnel [g1].\n"
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("The PLA fields about 2,000,000 personnel.")])

    assert counters["cited_figures_unsupported"] == 1.0
    assert findings[0]["figure"] == "2300000"
    assert findings[0]["goals"] == ["g1"]
    assert "2,300,000" in findings[0]["sentence"]


def test_a_cited_figure_present_in_its_evidence_is_not_reported():
    report = "# R\n\nThe PLA fields 2,300,000 active personnel [g1].\n"
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("The PLA fields about 2,300,000 personnel.")])

    assert findings == []
    assert counters["cited_figures_checked"] == 1.0
    assert counters["cited_figures_unsupported"] == 0.0


def test_a_figure_supported_by_any_one_of_several_cited_goals_passes():
    """A sentence may cite more than one goal. The figure needs to appear
    under one of them, not all."""
    report = "# R\n\nBoth forces exceed 1,200,000 personnel [g1] [g2].\n"
    findings, _ = audit_cited_figures(
        report, [_goal(), _goal("g2", "Indian Army size")],
        [_ev("The PLA is large."),
         _ev("The Indian Army fields 1,200,000 personnel.", goal_id="g2")])

    assert findings == []


def test_an_uncited_sentence_is_not_audited():
    """An uncited claim is a DIFFERENT failure with its own checks (D-40's
    attribution rule, D-66's zero-citation gate). Folding the two together
    would make this number impossible to act on."""
    report = "# R\n\nThe PLA fields 2,300,000 personnel.\n\nCited prose [g1].\n"
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("The PLA is large.")])

    assert findings == []
    assert counters["cited_figures_checked"] == 0.0


def test_a_sentence_citing_a_goal_with_no_evidence_is_skipped():
    """D-45's clean_citations already strips those markers and
    telemetry's goals_without_evidence already counts the condition.
    Reporting it here too would be a third count of one fact."""
    report = "# R\n\nThe PLA fields 2,300,000 personnel [g2].\n"
    findings, counters = audit_cited_figures(
        report, [_goal(), _goal("g2", "Indian Army size")],
        [_ev("The PLA is large.")])  # nothing under g2 at all

    assert findings == []
    assert counters["cited_figures_checked"] == 0.0


def test_a_citation_carrying_its_own_score_contributes_no_figure():
    """A `[g1 | corpus | score=0.90]` form that slipped past
    clean_citations would otherwise be read as the report claiming
    "0.90"."""
    report = "# R\n\nThe PLA is large [g1 | corpus | score=0.90].\n"
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("The PLA fields 2,300,000 personnel.")])

    assert findings == []
    assert counters["cited_figures_checked"] == 0.0


def test_no_evidence_at_all_is_a_clean_no_op():
    """Nothing to check against -- and a report with no evidence behind it
    is already reported by grounding_ratio and goals_without_evidence."""
    findings, counters = audit_cited_figures(
        "# R\n\nThe PLA fields 2,300,000 personnel [g1].\n", [_goal()], [])

    assert findings == []
    assert counters == {"cited_figures_checked": 0.0,
                        "cited_figures_unsupported": 0.0}


def test_findings_name_the_figure_the_goals_and_a_bounded_sentence():
    """A finding has to be actionable without re-running anything, and
    loggable without dumping a paragraph into a JSON line."""
    long_tail = " padding" * 100
    report = f"# R\n\nThe PLA fields 2,300,000 personnel [g1].{long_tail}\n"
    findings, _ = audit_cited_figures(
        report, [_goal()], [_ev("The PLA is large.")])

    assert set(findings[0]) == {"figure", "goals", "sentence"}
    assert len(findings[0]["sentence"]) <= 200


def test_evidence_stating_no_figure_at_all_still_makes_a_cited_figure_unsupported():
    """The single most important case, and the one an early version of
    this module silently skipped: the cited goal HAS evidence, that
    evidence contains no figures, and the report states one anyway. A
    membership test rather than a truthiness test on the per-goal figure
    set is what separates this from "the goal retrieved nothing", which
    is deliberately somebody else's count."""
    report = "# R\n\nThe PLA fields 2,300,000 personnel [g1].\n"
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("The PLA is large and well equipped.")])

    assert counters["cited_figures_unsupported"] == 1.0
    assert findings[0]["figure"] == "2300000"


# ---------------------------------------------------------------------------
# D-97: markdown headings -- their ordinals are not claims, their
# citations scope the prose beneath them
# ---------------------------------------------------------------------------


def test_heading_ordinals_are_not_audited_as_figures():
    """P205 regression (run p205.251-check). FOUR of that run's five
    findings were section numbers -- "1.1", "2.1", "3.1", "4.1" -- lifted
    out of "### 1.1 Active Personnel and Force Structure". The heading
    number has a decimal point, so _substantive's decimal rule waved it
    through even though the docstring already meant to exclude heading
    numbers; and the crude sentence split had glued the heading onto the
    paragraph below it, so the ordinal was attributed to a real
    sentence. Precision on that run was zero."""
    report = ("## 1. Military Size [g1]\n\n"
              "### 1.1 Active Personnel\n\n"
              "The PLA fields 2,535,000 active troops.\n")
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("China fields 2,535,000 active troops")])

    assert findings == []
    assert counters["cited_figures_unsupported"] == 0.0


def test_a_heading_citation_scopes_every_sentence_beneath_it():
    """The other half of the same finding, and the reason headings are
    scoped rather than simply deleted. That report cited its goals ONLY
    in headings and never inline, so stripping headings outright would
    have left this module auditing nothing at all."""
    report = ("## 1. Military Size [g1]\n\n"
              "The PLA fields 2,535,000 active troops. "
              "India fields 9,999,999 active troops.\n")
    findings, _ = audit_cited_figures(
        report, [_goal()], [_ev("China fields 2,535,000 active troops")])

    assert [f["figure"] for f in findings] == ["9999999"]
    assert findings[0]["goals"] == ["g1"]


def test_a_deeper_uncited_heading_inherits_the_scope():
    """"### 1.1 Active Personnel" carries no citation of its own and sits
    UNDER "## 1. ... [g1]", so the prose below it is still g1's."""
    report = ("## 1. Military Size [g1]\n\n"
              "### 1.1 Active Personnel\n\n"
              "India fields 9,999,999 active troops.\n")
    findings, _ = audit_cited_figures(
        report, [_goal()], [_ev("China fields 2,535,000 active troops")])

    assert [f["figure"] for f in findings] == ["9999999"]


def test_a_heading_at_the_same_depth_closes_the_scope():
    """Live (run p205.251-check) "## 5. Historical Conflicts and Lessons"
    cited nothing. Its prose must fall OUT of scope rather than inherit
    the previous section's [g1] -- inheriting would attribute claims to
    a goal the report never associated them with, which is the exact
    dishonesty D-45 drops unevidenced markers to prevent."""
    report = ("## 1. Military Size [g1]\n\n"
              "The PLA fields 2,535,000 active troops.\n\n"
              "## 5. Historical Conflicts\n\n"
              "India fields 9,999,999 active troops.\n")
    findings, _ = audit_cited_figures(
        report, [_goal()], [_ev("China fields 2,535,000 active troops")])

    assert findings == [], "the uncited section is nobody's claim"


def test_an_inline_citation_overrides_the_heading_scope():
    report = ("## 1. Military Size [g1]\n\n"
              "India fields 9,999,999 active troops [g2].\n")
    findings, _ = audit_cited_figures(
        report, [_goal(), _goal("g2", "India")],
        [_ev("China fields 2,535,000", goal_id="g1"),
         _ev("India fields 1,400,000", goal_id="g2")])

    assert [f["goals"] for f in findings] == [["g2"]]


# ---------------------------------------------------------------------------
# D-98: the same quantity written at a different scale
# ---------------------------------------------------------------------------


def test_a_figure_is_read_at_the_scale_its_magnitude_word_gives_it():
    assert scaled_values("about 2,035,000 troops") == {2035000.0}
    assert scaled_values("about 2.035 million troops") == {2035000.0}


def test_indian_scales_are_left_unscaled_rather_than_parsed_wrongly():
    """A stated limitation, not an oversight. Indian digit grouping
    ("2,10,000" = 210,000) does not fit the three-digit-group pattern
    this module's figure grammar uses, and lakh/crore COMPOUND ("7.85
    lakh crore" is 7.85e12). A one-word lookup gets both wrong, and a
    wrong scale could confirm a figure against evidence that says
    something else. Carrying no scale costs nothing instead: the
    interval is then a fraction of a unit wide, so only the figure's own
    exact value confirms it -- which the exact path already handles."""
    assert 2.1e12 not in scaled_values("worth Rs 2,10,000 crore")
    assert 785000.0 not in scaled_values("Rs 7.85 lakh crore to defence")


def test_a_percentage_carries_no_magnitude_to_rescale():
    """"20%" against "20.33%" is a rounding question about the SAME
    scale, and the exact path already answers it by disagreeing. Letting
    a percentage into the interval arithmetic would only widen it."""
    assert scaled_values("a 20% increase") == set()


def test_a_stated_figure_claims_only_the_precision_it_was_written_with():
    """The rule the whole rescue rests on. A figure written to one
    decimal place asserts a rounding interval and nothing tighter --
    nothing here is tuned, and a more precise claim is held to a
    proportionally stricter standard."""
    spans = scaled_claims("roughly 2.0 million personnel")
    assert spans["2.0"] == (1_950_000.0, 2_050_000.0)

    spans = scaled_claims("roughly 2.05 million personnel")
    assert spans["2.05"] == (2_045_000.0, 2_055_000.0)


def test_a_range_inherits_the_magnitude_word_written_after_it():
    """P205 regression (run p205.252-check). All three findings on that
    run were the "2.0" of "roughly 2.0-2.1 million personnel". The
    magnitude word is written ONCE, at the end, so without inheritance
    the first half of every range is read as a bare 2.0 and misses the
    rescue entirely -- which is exactly the shape that motivated it."""
    spans = scaled_claims("roughly 2.0-2.1 million personnel")

    assert spans["2.0"] == (1_950_000.0, 2_050_000.0)
    assert spans["2.1"] == (2_050_000.0, 2_150_000.0)


def test_the_same_quantity_at_a_different_scale_is_supported():
    """P205 regression (run p205.252-check): the report said "roughly
    2.0-2.1 million personnel" and the evidence said "2,035,000". Same
    number, different notation, reported as an unsupported figure three
    times over."""
    report = "## 1. Manpower [g1]\n\nChina fields roughly 2.0-2.1 million personnel.\n"
    findings, counters = audit_cited_figures(
        report, [_goal()],
        [_ev("China's active force stands at 2,035,000 personnel"),
         _ev("China fields about 2.1 million active troops")])

    assert findings == []
    assert counters["cited_figures_unsupported"] == 0.0


def test_p205_251_the_other_live_rescale_miss():
    """The same defect on the previous run: "2.1 to 2.5 million active
    personnel" against evidence reading "2,535,000 active troops"."""
    report = ("## 1. Manpower [g1]\n\n"
              "The PLA maintains 2.1 to 2.5 million active personnel.\n")
    findings, _ = audit_cited_figures(
        report, [_goal()],
        [_ev("China fields 2,535,000 active troops"),
         _ev("China's armed forces have over 2.1 million active personnel")])

    assert findings == []


def test_a_figure_outside_its_own_interval_is_still_unsupported():
    """The rescue must not become a way for any figure to match any
    evidence. 9.9 million is not 2,535,000 at any scale."""
    report = "## 1. Manpower [g1]\n\nThe PLA fields 9.9 million personnel.\n"
    findings, _ = audit_cited_figures(
        report, [_goal()], [_ev("China fields 2,535,000 active troops")])

    assert [f["figure"] for f in findings] == ["9.9"]


def test_a_more_precise_claim_is_held_to_a_stricter_standard():
    """"2.5 million" is confirmed by 2,535,000; "2.53 million" is
    confirmed by it too; "2.1 million" is refuted by it. The report's own
    notation decides, which is what keeps this from being a tolerance
    knob somebody has to tune."""
    ev = [_ev("China fields 2,535,000 active troops")]
    head = "## 1. Manpower [g1]\n\n"

    for stated, expected in (("2.5", []), ("2.53", []), ("2.1", ["2.1"])):
        findings, _ = audit_cited_figures(
            f"{head}The PLA fields {stated} million personnel.\n",
            [_goal()], ev)
        assert [f["figure"] for f in findings] == expected, stated


def test_an_exact_match_never_reaches_the_scale_comparison():
    """Cheap path first: a report and its evidence that already agree
    literally are never subjected to interval arithmetic."""
    report = "## 1. Manpower [g1]\n\nThe PLA fields 2,535,000 personnel.\n"
    findings, counters = audit_cited_figures(
        report, [_goal()], [_ev("China fields 2,535,000 active troops")])

    assert findings == []
    assert counters["cited_figures_checked"] == 1.0