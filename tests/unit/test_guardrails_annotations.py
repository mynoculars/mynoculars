"""
tests/unit/test_guardrails_annotations.py — guardrails/annotations.py (D-139).

WHY THIS FILE EXISTS: live (run p205.276-check) three of the critic's six
notes demanded the removal of the D-85 provenance notice — text the
compiler never wrote and cannot remove, because annotate_ungrounded_report
re-adds it after every compile. One revision was spent on an instruction no
rewrite can satisfy, and the compile that followed dropped its citations
entirely. These tests cover the separation that fixes it.
"""

from research_agent.guardrails.annotations import strip_machine_annotations
from research_agent.guardrails.grounding import annotate_ungrounded_report
from research_agent.guardrails.sources import SOURCES_HEADING
from research_agent.guardrails.truncation import annotate_truncated_run
from research_agent.state import Evidence, Goal

_BODY = "# Report\n\nChina fields more personnel [g1].\n"


def test_a_report_with_no_annotations_is_returned_unchanged():
    """The common path — a grounded run ships no notice and, with web
    search off, no Sources block. It must be exactly unchanged."""
    assert strip_machine_annotations(_BODY) == _BODY


def test_the_provenance_notice_is_removed():
    annotated, _ = annotate_ungrounded_report(
        _BODY, [Goal(goal_id="g1", description="army size")],
        [Evidence(task_key="t1", goal_id="g1", content="web text",
                 source="web", score=0.7)],
        0.5, 0.5)
    assert "Provenance notice" in annotated

    assert strip_machine_annotations(annotated) == _BODY


def test_the_stopped_early_notice_is_removed():
    annotated, _ = annotate_truncated_run(_BODY, "deadline")
    assert "Run stopped early" in annotated

    assert strip_machine_annotations(annotated) == _BODY


def test_both_notices_at_once_are_removed():
    """D-132 stacks its notice ABOVE D-85's. Neither may survive, and the
    gap between them must not either."""
    annotated, _ = annotate_ungrounded_report(
        _BODY, [Goal(goal_id="g1", description="army size")],
        [Evidence(task_key="t1", goal_id="g1", content="web text",
                 source="web", score=0.7)],
        0.5, 0.5)
    annotated, _ = annotate_truncated_run(annotated, "tokens")

    assert strip_machine_annotations(annotated) == _BODY


def test_the_sources_block_is_removed():
    report = (_BODY + "\n" + SOURCES_HEADING
              + "\n\n1. [g1] A Title (example.com) — https://example.com/a\n"
                "2. [g1] B Title (other.com) — https://other.com/b\n")

    assert strip_machine_annotations(report) == _BODY


def test_a_sources_heading_the_model_wrote_mid_report_is_still_removed():
    """Honest limitation, asserted rather than left to be discovered: the
    cut runs from the heading to the end of the document, so a model that
    writes its own '## Sources' section loses whatever follows it. That is
    acceptable because append_web_sources always appends LAST — anything
    under that heading in a finished report is this system's own text —
    and a false positive costs the critic some context, never the reader
    anything."""
    report = _BODY + "\n" + SOURCES_HEADING + "\n\nmodel prose\n"

    assert strip_machine_annotations(report) == _BODY


def test_report_body_quotes_are_kept():
    """Only a blockquote carrying a machine marker goes. A blockquote the
    model wrote is part of the answer and the critic must still see it."""
    report = "# Report\n\n> A quotation the model chose to include.\n\nProse.\n"

    assert strip_machine_annotations(report) == report


def test_empty_report_is_handled():
    assert strip_machine_annotations("") == ""
