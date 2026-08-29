"""
tests/unit/test_guardrails_truncation.py — the "stopped early" notice
(D-132, P6-4).

Same test shape as D-85's provenance notice, deliberately: the two
notices share every downstream reader (cited_goal_ids, count_sections,
the revision path), so they need the same guarantees proved the same way.
"""

from research_agent.guardrails.sources import cited_goal_ids
from research_agent.guardrails.truncation import (
    annotate_truncated_run, report_carries_truncation_notice)
from research_agent.reporting.metrics import count_sections

_REPORT = ("# Findings\n\n## Scale\n\nThe two forces differ in size [g1].\n\n"
           "## Doctrine\n\nDoctrine differs as well [g2].\n")


def test_a_run_that_finished_on_its_own_terms_is_byte_identical():
    """The path EVERY run takes while both budgets sit at their default
    0. It has to be exactly unchanged, not merely similar."""
    assert annotate_truncated_run(_REPORT, None) == (_REPORT, {})


def test_a_deadline_stop_says_so_in_the_report():
    out, counters = annotate_truncated_run(_REPORT, "deadline")

    assert out.startswith("> **Run stopped early")
    assert "wall-clock deadline" in out
    assert counters == {"truncation_notice_inserted": 1.0}
    assert _REPORT in out, "the report itself is untouched below the notice"


def test_a_token_stop_names_the_other_budget():
    out, _ = annotate_truncated_run(_REPORT, "tokens")
    assert "token budget" in out


def test_an_unrecognised_reason_is_a_no_op():
    """This function will not invent prose for a stop condition it does
    not know about -- a silent no-op beats a notice describing the wrong
    thing."""
    assert annotate_truncated_run(_REPORT, "something_else") == (_REPORT, {})


def test_the_notice_is_idempotent_across_revision_passes():
    """compiler_node runs once per revision. Two notices stacked on one
    report is the defect this guards."""
    once, _ = annotate_truncated_run(_REPORT, "deadline")
    twice, counters = annotate_truncated_run(once, "deadline")

    assert twice == once
    assert counters == {}


def test_the_notice_adds_no_citation_marker():
    """cited_goal_ids feeds compiler_node's evidence_cited count AND
    critic_node's D-66 zero-citation gate -- a citation-shaped string
    here could slip a report that cites nothing past that gate."""
    out, _ = annotate_truncated_run("A report with no citations at all.",
                                    "deadline")
    assert cited_goal_ids(out) == set()


def test_the_notice_adds_no_heading():
    """count_sections feeds the node.compiled log line and cli.py's
    RESULT block; S-10 exists because those two once disagreed."""
    out, _ = annotate_truncated_run(_REPORT, "deadline")
    assert count_sections(out) == count_sections(_REPORT)


def test_the_shipped_report_can_be_asked_whether_it_carries_the_notice():
    """D-59's rule: telemetry reads the ARTIFACT, never a counter that
    would sum every compile attempt."""
    out, _ = annotate_truncated_run(_REPORT, "tokens")

    assert report_carries_truncation_notice(out)
    assert not report_carries_truncation_notice(_REPORT)
