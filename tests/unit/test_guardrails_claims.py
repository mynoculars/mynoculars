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
                                              figures_in, report_body)
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
    assert figures_in("in 2023 the figure rose") == {"2023"}


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
    