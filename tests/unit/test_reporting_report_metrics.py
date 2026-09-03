"""
tests/unit/test_reporting_report_metrics.py -- S-13's extraction, tested
against the module directly rather than only through telemetry_node.

WHY A DEDICATED FILE. D-146 extracted reporting/telemetry.py and left it
covered transitively by test_agents_compilation.py; four of the six
reporting/ modules have their own file, and this one is the half of
telemetry_node that reads the SHIPPED REPORT and fires five WARNINGs --
the half where a silent regression is least likely to fail a graph test
and most likely to matter, because every one of these figures is an
honesty signal a reader acts on.

The stub gate is the recurring subject here: four of the five checks are
skipped when llm_mode == "stub", because StubClient's fixed placeholder
report carries no [gN] markers by design and auditing it would report
findings about the harness rather than about a run.
"""

import pytest

from research_agent.config import Settings
from research_agent.reporting.report_metrics import (goal_coverage,
                                                     shipped_report_metrics)
from research_agent.state import Evidence, Goal, ResearchState


def _ev(goal_id, source="corpus", content="Redis persistence uses AOF.",
        score=0.9, **kw):
    return Evidence(task_key=f"{goal_id}::t", goal_id=goal_id, source=source,
                    content=content, score=score, **kw)


def _state(goals=(), evidence=(), report=""):
    return ResearchState(raw_query="q", goals=list(goals),
                         evidence=list(evidence), final_report=report)


def _live_settings(**kw):
    """A non-stub Settings, which is what un-gates the four report checks."""
    return Settings(_env_file=None, llm_mode="live", **kw)


# --- goal_coverage: pure, no logging ---------------------------------------

def test_no_goals_gives_zero_rather_than_dividing_by_zero(settings):
    """A planning failure produces no goals, and 0/0 is undefined rather
    than perfect -- the guard that keeps a crashed run from reporting 1.0."""
    facts = goal_coverage(_state(), settings)
    assert facts["corpus_recall"] == 0.0
    assert facts["grounding_ratio"] == 0.0
    assert facts["goals_without_evidence"] == []


def test_grounding_ratio_counts_presence_and_corpus_recall_counts_topic(settings):
    """The two numbers answer different questions and must be able to
    disagree: grounding_ratio asks "did this goal get ANY evidence",
    corpus_recall asks "did a DOCUMENT on this topic cover it". An
    off-topic corpus hit satisfies the first and not the second."""
    goals = [Goal(goal_id="g1", description="Redis persistence durability")]
    off_topic = _ev("g1", content="Elephants migrate across Botswana yearly.")
    facts = goal_coverage(_state(goals, [off_topic]), settings)
    assert facts["grounding_ratio"] == 1.0
    assert facts["corpus_recall"] == 0.0
    assert facts["goals_without_evidence"] == []


def test_an_on_topic_document_grounds_the_goal(settings):
    goals = [Goal(goal_id="g1", description="Redis persistence durability")]
    facts = goal_coverage(_state(goals, [_ev("g1")]), settings)
    assert facts["corpus_recall"] == 1.0


def test_a_goal_with_no_evidence_is_named_not_just_counted(settings):
    """The LIST is the actionable half -- a ratio says how bad, the ids say
    which."""
    goals = [Goal(goal_id="g1", description="Redis persistence"),
             Goal(goal_id="g2", description="Memcached slab allocation")]
    facts = goal_coverage(_state(goals, [_ev("g1")]), settings)
    assert facts["goals_without_evidence"] == ["g2"]
    assert facts["grounding_ratio"] == 0.5


def test_web_evidence_never_grounds_a_goal(settings):
    """D-57's invariant, asserted here as well as in the web tests: a
    snippet COVERS but never GROUNDS, so corpus_recall stays 0.0 however
    on-topic the snippet is."""
    goals = [Goal(goal_id="g1", description="Redis persistence durability")]
    web = _ev("g1", source="web", url="https://x.test/a", domain="x.test")
    facts = goal_coverage(_state(goals, [web]), settings)
    assert facts["grounding_ratio"] == 1.0
    assert facts["corpus_recall"] == 0.0


# --- shipped_report_metrics: the report-derived half ------------------------

def test_stub_mode_skips_every_report_check(settings):
    """StubClient's placeholder report carries no [gN] markers by design,
    so auditing it would report findings about the harness. All four
    stub-gated figures come back at their empty values."""
    goals = [Goal(goal_id="g1", description="Redis persistence")]
    state = _state(goals, [_ev("g1")], report="A report with no citations.")
    m = shipped_report_metrics(state, settings, {"corpus": 1})
    assert m["figure_findings"] == []
    assert m["figure_counters"] == {}
    assert m["residual_pastes"] == 0
    assert m["residual_glue"] == 0


def test_it_returns_every_key_telemetry_node_unpacks():
    """The contract this extraction has to keep. telemetry_node unpacks
    these by name; a missing key is an AttributeError-shaped failure in the
    final node of every run, which is the worst place to find one."""
    m = shipped_report_metrics(_state(), _live_settings(), {})
    assert set(m) == {
        "goal_ids", "corpus_recall", "goals_without_evidence",
        "grounding_ratio", "web_sources_listed", "web_sourced_items",
        "grounding_notice_shipped", "figure_findings", "figure_counters",
        "residual_pastes", "residual_glue"}


def test_web_sourced_items_counts_evidence_and_listed_reads_the_report():
    """D-59: the two numbers come from different places on purpose --
    what was RETRIEVED is counted from state.evidence, what was ATTRIBUTED
    is parsed out of the shipped report, never from an additive counter
    that would sum every compile attempt."""
    web = [_ev("g1", source="web", url="https://a.test/1", domain="a.test"),
           _ev("g1", source="web", url="https://b.test/2", domain="b.test")]
    state = _state([Goal(goal_id="g1", description="Redis")], web,
                   report="Body [g1]\n\n## Sources\n\n1. [g1] A (a.test) — https://a.test/1\n")
    m = shipped_report_metrics(state, _live_settings(), {})
    assert m["web_sourced_items"] == 2
    assert m["web_sources_listed"] == 1


def test_an_ungrounded_run_warns_and_a_grounded_one_does_not(caplog):
    """D-85's last-line-of-sight check. The WARNING is the whole reason
    this code runs at telemetry time rather than in the compiler."""
    goals = [Goal(goal_id="g1", description="Redis persistence durability")]
    settings = _live_settings(grounded_recall_target=0.5)

    with caplog.at_level("WARNING"):
        shipped_report_metrics(
            _state(goals, [_ev("g1", source="web", url="https://a.test/1")],
                   report="Body."), settings, {})
    assert any(r.msg == "report.shipped_ungrounded" for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING"):
        shipped_report_metrics(_state(goals, [_ev("g1")], report="Body [g1]."),
                               settings, {})
    assert not any(r.msg == "report.shipped_ungrounded" for r in caplog.records)


@pytest.mark.parametrize("report,expected", [
    ("A plain report with no notice.", False),
])
def test_grounding_notice_is_read_from_the_shipped_report(report, expected):
    """Derived from state.final_report, never from a counter -- D-59's rule,
    because compiler_node runs once per revision and counters merge
    additively."""
    m = shipped_report_metrics(_state(report=report), _live_settings(), {})
    assert m["grounding_notice_shipped"] is expected
