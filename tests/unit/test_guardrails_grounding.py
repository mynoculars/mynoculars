"""
tests/unit/test_guardrails_grounding.py -- guardrails/grounding.py (D-85).

WHY THIS FILE EXISTS: run p205.246-check shipped a 15,154-character
report with `grounded_score 0.0`, every goal carried by the web tier, the
retrieval floor dropping 36 of 36 dense candidates -- and finished
`Final status: SUCCESS` with nothing in the REPORT saying the corpus had
contributed nothing. Telemetry was honest; the deliverable was silent.
These tests cover the notice that closes that gap, and -- just as
importantly -- the no-op path, which every corpus-answered run takes and
which must leave the report byte-identical.
"""

from research_agent.guardrails.citations import clean_citations
from research_agent.guardrails.grounding import (NOTICE_MARKER,
                                                 annotate_ungrounded_report,
                                                 grounded_goal_count,
                                                 report_carries_grounding_notice)
from research_agent.guardrails.sources import cited_goal_ids
from research_agent.reporting.metrics import count_sections
from research_agent.state import Evidence, Goal

REPORT = ("# Research Report\n\n## Findings\n\nThe PLA fields roughly two "
          "million active personnel [g1].\n\n## Outlook\n\nMore prose [g2].\n")

# The two thresholds compiler_node passes through from Settings.
MIN_EVIDENCE_SCORE = 0.5
GROUNDED_TARGET = 0.5


def _goal(goal_id, description):
    return Goal(goal_id=goal_id, description=description)


def _ev(goal_id, source, content, score=0.9):
    return Evidence(task_key="t", goal_id=goal_id, source=source,
                    content=content, score=score)


def _annotate(report, goals, evidence):
    return annotate_ungrounded_report(report, goals, evidence,
                                      MIN_EVIDENCE_SCORE, GROUNDED_TARGET)


# ---------------------------------------------------------------------------
# The no-op paths -- byte-identical, because every healthy run takes one
# ---------------------------------------------------------------------------


def test_a_well_grounded_report_is_returned_byte_identical():
    """The common path for any corpus that genuinely answers the question.
    Asserted as exact equality, not "contains", for the reason
    append_web_sources' own no-op test uses: a guardrail that fires when
    it should not is a guardrail nobody can trust."""
    goals = [_goal("g1", "PLA active personnel strength")]
    evidence = [_ev("g1", "corpus",
                    "The PLA maintains roughly two million active personnel.")]

    report, counters = _annotate(REPORT, goals, evidence)

    assert report == REPORT
    assert counters == {}


def test_no_goals_is_a_noop():
    """A run that produced no goals already failed earlier and louder
    (D-21); there is nothing for a provenance notice to describe."""
    assert _annotate(REPORT, [], [_ev("g1", "web", "x")]) == (REPORT, {})


def test_no_evidence_at_all_is_a_noop():
    """Deliberately scoped out, on D-66's precedent and for its reason: a
    report that is empty-handed because NOTHING was retrieved is a
    different, already-visible failure, and its own prose says so."""
    goals = [_goal("g1", "PLA active personnel strength")]
    assert _annotate(REPORT, goals, []) == (REPORT, {})


def test_grounding_exactly_at_target_is_a_noop():
    """The comparison is `>= target`, so a run sitting exactly on the
    floor passes. Pinned because an off-by-one here would fire the notice
    on runs that met their configured bar."""
    goals = [_goal("g1", "Redis eviction policies"),
             _goal("g2", "Memcached slab allocation")]
    evidence = [_ev("g1", "corpus",
                    "Redis eviction policies include allkeys-lru.")]

    report, counters = _annotate(REPORT, goals, evidence)  # 1/2 == 0.5 target

    assert report == REPORT
    assert counters == {}


# ---------------------------------------------------------------------------
# The firing paths
# ---------------------------------------------------------------------------


def test_a_wholly_web_answered_run_is_annotated():
    """The p205.246-check shape exactly: every goal covered, none of it by
    a document. D-57 makes web evidence COVER a goal but never GROUND
    one, so grounded_goal_count must see zero here."""
    goals = [_goal("g1", "PLA active personnel strength"),
             _goal("g2", "Indian Army active personnel strength")]
    evidence = [_ev("g1", "web", "The PLA fields about two million troops."),
                _ev("g2", "web", "The Indian Army fields about 1.2 million.")]

    report, counters = _annotate(REPORT, goals, evidence)

    assert report.startswith("> ")
    assert NOTICE_MARKER in report
    assert "None of this report's 2 research goal(s)" in report
    assert counters["grounding_notice_inserted"] == 1.0
    assert counters["grounding_notice_goals_ungrounded"] == 2.0
    # The original report survives underneath, unmodified.
    assert report.endswith(REPORT)


def test_a_partially_grounded_run_reports_the_real_counts():
    """D-12 applied to prose: the notice states counted facts, never a
    judgement. One of three grounded must read as one of three."""
    goals = [_goal("g1", "Redis eviction policies"),
             _goal("g2", "PLA active personnel strength"),
             _goal("g3", "Indian Army modernization budget")]
    evidence = [_ev("g1", "corpus",
                    "Redis eviction policies include allkeys-lru."),
                _ev("g2", "web", "The PLA fields about two million troops."),
                _ev("g3", "model", "India's defence budget rose in 2023.")]

    report, counters = _annotate(REPORT, goals, evidence)

    assert "Only 1 of this report's 3 research goal(s)" in report
    assert counters["grounding_notice_goals_ungrounded"] == 2.0


def test_model_tier_evidence_never_grounds_a_goal():
    """D-42's rule, enforced through the shared has_grounded_evidence
    predicate rather than re-implemented here."""
    goals = [_goal("g1", "PLA active personnel strength")]
    evidence = [_ev("g1", "model",
                    "The PLA maintains roughly two million active personnel.")]

    assert grounded_goal_count(goals, evidence, MIN_EVIDENCE_SCORE) == 0
    assert NOTICE_MARKER in _annotate(REPORT, goals, evidence)[0]


def test_an_off_topic_corpus_hit_does_not_ground_a_goal():
    """The D-39 topical gate, inherited for free by delegating to
    has_grounded_evidence (M-1). A Redis document scoring well against an
    armies goal is exactly the shape that fooled corpus_recall before
    D-39, and it must not ground the goal here either."""
    goals = [_goal("g1", "PLA active personnel strength versus India")]
    evidence = [_ev("g1", "corpus",
                    "Redis and Memcached both support session caching "
                    "with different eviction policies.")]

    assert grounded_goal_count(goals, evidence, MIN_EVIDENCE_SCORE) == 0
    assert NOTICE_MARKER in _annotate(REPORT, goals, evidence)[0]


def test_evidence_below_the_score_floor_does_not_ground_a_goal():
    goals = [_goal("g1", "Redis eviction policies")]
    evidence = [_ev("g1", "corpus",
                    "Redis eviction policies include allkeys-lru.",
                    score=MIN_EVIDENCE_SCORE)]  # strict `>`, so this fails

    assert grounded_goal_count(goals, evidence, MIN_EVIDENCE_SCORE) == 0


def test_the_notice_is_idempotent():
    """compiler_node runs once per REVISION. Two passes over the same text
    must not stack two notices."""
    goals = [_goal("g1", "PLA active personnel strength")]
    evidence = [_ev("g1", "web", "The PLA fields about two million troops.")]

    once, _ = _annotate(REPORT, goals, evidence)
    twice, counters = _annotate(once, goals, evidence)

    assert twice == once
    assert counters == {}
    assert once.count(NOTICE_MARKER) == 1


# ---------------------------------------------------------------------------
# The two invariants the notice's own TEXT has to respect
# ---------------------------------------------------------------------------


def test_the_notice_adds_no_citation_markers():
    """Load-bearing, not cosmetic. cited_goal_ids is read by
    compiler_node's evidence_cited count AND by critic_node's D-66
    zero-citation gate -- a citation-shaped string here could let a report
    that cites nothing at all slip past that gate on a marker the model
    never wrote."""
    goals = [_goal("g1", "PLA active personnel strength"),
             _goal("g2", "Indian Army strength")]
    evidence = [_ev("g1", "web", "The PLA fields about two million troops.")]

    annotated, _ = _annotate(REPORT, goals, evidence)

    assert cited_goal_ids(annotated) == cited_goal_ids(REPORT)


def test_the_notice_adds_no_markdown_section_heading():
    """count_sections (S-10) is read by both the node.compiled log line
    and cli.py's terminal RESULT block. The notice is a blockquote so that
    count keeps describing the model's own structure."""
    goals = [_goal("g1", "PLA active personnel strength")]
    evidence = [_ev("g1", "web", "The PLA fields about two million troops.")]

    annotated, _ = _annotate(REPORT, goals, evidence)

    assert count_sections(annotated) == count_sections(REPORT)


def test_the_notice_survives_a_citation_repair_pass_unharmed():
    """Belt and braces on the ordering rule. The notice is inserted AFTER
    clean_citations in compiler_node precisely so this can never come up
    -- but if a future change reorders them, the notice must not be
    mistaken for pasted evidence text and stripped."""
    goals = [_goal("g1", "PLA active personnel strength")]
    evidence = [_ev("g1", "web", "The PLA fields about two million troops.")]

    annotated, _ = _annotate(REPORT, goals, evidence)
    repaired, _ = clean_citations(annotated, goals, evidence)

    assert NOTICE_MARKER in repaired


# ---------------------------------------------------------------------------
# report_carries_grounding_notice -- telemetry's read-back (D-59's rule)
# ---------------------------------------------------------------------------


def test_report_carries_grounding_notice_detects_what_annotate_wrote():
    goals = [_goal("g1", "PLA active personnel strength")]
    evidence = [_ev("g1", "web", "The PLA fields about two million troops.")]

    assert not report_carries_grounding_notice(REPORT)
    assert report_carries_grounding_notice(_annotate(REPORT, goals, evidence)[0])
