"""
tests/unit/test_guardrails_sources_guidance.py — regression cover for D-64,
the reviewer-guidance term union in
guardrails/sources.py::append_web_sources.

The reported symptom: after a human redirect the report came back with no
"## Sources" section at all, and nothing in the prose or the log to explain
why. Cause: D-59's topical gate tests web evidence against the goal
descriptions composed BEFORE the human intervened, so pages fetched because
a reviewer explicitly asked for them were judged off-topic against goals
that predate the request.

Own file rather than appended to a sibling, for the delivery reason recorded
in DECISIONS.md D-62/D-63.
"""

from research_agent.guardrails.sources import append_web_sources
from research_agent.state import Evidence, Goal, Volatility

# The live shape from run p205.103-check: goals about governance, a reviewer
# who asked for press-freedom and democracy-index sources.
_GOALS = [Goal(goal_id="g1",
               description="Political systems and governance structures in India and US")]
_REPORT = "# Report\n\n## Governance\nThe two differ substantially [g1].\n"
_GUIDANCE = "UN reports of press freedom, democracy index, human rights abuses"


def _ev(content, goal_id="g1", url="https://ohchr.org/x", score=0.8):
    return Evidence(task_key="t", goal_id=goal_id, source="web", content=content,
                    score=score, volatility=Volatility.SEMI_STABLE,
                    url=url, domain="ohchr.org")


_ON_GUIDANCE = _ev("UN press freedom index — watchdog ranking of democratic backsliding")
_ON_GOAL = _ev("Governance structures in India — a comparison of political systems")
_DRIFTED = _ev("Redis monitoring guide — SLOWLOG and MEMORY USAGE for cache operators")


def test_guidance_relevant_evidence_survives_the_gate():
    # The bug, directly. Without guidance this item is dropped and the
    # report ships with no Sources section.
    report, counters = append_web_sources(_REPORT, [_ON_GUIDANCE], _GOALS, _GUIDANCE)
    assert "## Sources" in report
    assert counters["web_sources_listed"] == 1.0
    assert counters["web_sources_suppressed"] == 0.0


def test_the_same_evidence_is_still_dropped_without_guidance():
    # Pins the actual defect: identical inputs, guidance omitted, no Sources.
    # If this ever starts passing, the gate has been widened too far.
    report, counters = append_web_sources(_REPORT, [_ON_GUIDANCE], _GOALS)
    assert "## Sources" not in report
    assert counters["web_sources_suppressed"] == 1.0


def test_d59_still_drops_genuinely_drifted_evidence():
    # D-59's motivating failure: nine Redis URLs listed under [g1] in an
    # India-vs-US report. Matches neither the goal terms nor the guidance,
    # so it stays dropped. The union widens the gate by exactly one thing.
    report, counters = append_web_sources(_REPORT, [_DRIFTED], _GOALS, _GUIDANCE)
    assert "## Sources" not in report
    assert counters["web_sources_suppressed"] == 1.0


def test_evidence_matching_the_goal_still_passes_when_guidance_is_present():
    # Guidance must ADD to the goal terms, never replace them — a redirect
    # asking for one new angle should not suppress the original coverage.
    report, counters = append_web_sources(_REPORT, [_ON_GOAL], _GOALS, _GUIDANCE)
    assert "## Sources" in report
    assert counters["web_sources_listed"] == 1.0


def test_goal_and_guidance_matches_are_listed_together():
    report, counters = append_web_sources(
        _REPORT, [_ON_GOAL, _ev("UN democracy index — press freedom scores by country",
                                url="https://ohchr.org/y")],
        _GOALS, _GUIDANCE)
    assert counters["web_sources_listed"] == 2.0
    assert counters["web_sources_suppressed"] == 0.0


def test_empty_guidance_is_byte_identical_to_the_pre_d64_behaviour():
    # No redirect is the common path; it must not change at all.
    with_arg, c1 = append_web_sources(_REPORT, [_ON_GOAL, _DRIFTED], _GOALS, "")
    without, c2 = append_web_sources(_REPORT, [_ON_GOAL, _DRIFTED], _GOALS)
    assert with_arg == without
    assert c1 == c2


def test_guidance_is_ignored_when_goals_are_not_supplied():
    # The pre-D-59 signature skips the gate entirely; adding guidance must
    # not accidentally re-enable it.
    report, counters = append_web_sources(_REPORT, [_DRIFTED], None, _GUIDANCE)
    assert "## Sources" in report
    assert counters["web_sources_listed"] == 1.0


def test_guidance_cannot_rescue_evidence_for_an_uncited_goal():
    # The cited-goal filter runs BEFORE the topical gate and is untouched:
    # a Sources list is a claim about what backed THIS report, and guidance
    # does not make an uncited goal cited.
    item = _ev("UN press freedom index — watchdog ranking", goal_id="g9")
    report, counters = append_web_sources(_REPORT, [item], _GOALS, _GUIDANCE)
    assert "## Sources" not in report


def test_whitespace_only_guidance_behaves_as_no_guidance():
    report, _ = append_web_sources(_REPORT, [_ON_GUIDANCE], _GOALS, "   \n  ")
    assert "## Sources" not in report
