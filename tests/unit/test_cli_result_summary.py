"""
tests/unit/test_cli_result_summary.py — cli.py::_fmt_result_summary.

Covers the verdict block printed between the report and the raw telemetry
dump. The point of these tests is not formatting for its own sake: runs
p205.211 and p205.212 differed by critique outcome, revision count and a
652-vs-9,603-character report, and none of that was visible without
reading 45 lines of JSON. Each test below pins one of the figures that
would have made the difference obvious on sight.

Its own file rather than appended to a sibling, for the delivery reason
recorded in DECISIONS.md D-62/D-63.
"""

from research_agent.cli import _fmt_result_summary


# Telemetry as actually emitted by run p205.212, trimmed to the keys the
# summary reads. Using the real shape rather than an invented one means a
# key rename in the telemetry node breaks these tests, which is the point.
_P205_212 = {
    "goals": 5,
    "evidence_items": 128,
    "evidence_by_source": {"memory": 5, "corpus": 40, "mcp": 33,
                           "web": 30, "model": 20},
    "goals_without_evidence": [],
    "grounding_ratio": 1.0,
    "recall": 1.0,
    "corpus_recall": 0.0,
    "grounded_score": 0.0,
    "llm_provider_calls": 16,
    "llm_fallback_hops": 6,
    "llm_quality_calls": 2,
    "llm_quality_calls_failed": 1,
    "search_calls": 11,
    "search_failures": 0,
    "revision_cycles": 2,
    "critique_passed": False,
    "planning_error": None,
    "escalations": [{"trigger": "E4", "action": "approve"}],
}

_REPORT = "# Title\n\n## One\nbody\n\n## Two\nbody\n\n## Three\nbody\n"


def test_a_failed_critique_says_so_in_words():
    # The single most important line. p205.212 reported
    # "critique_passed": false 40 lines deep in JSON; nobody reads that.
    out = _fmt_result_summary(_P205_212, _REPORT)
    assert "FAILED" in out
    assert "2 revision cycle(s)" in out


def test_a_passed_critique_says_so_too():
    out = _fmt_result_summary({**_P205_212, "critique_passed": True,
                               "revision_cycles": 0}, _REPORT)
    assert "PASSED" in out
    assert "revision cycle" not in out  # no cycles -> no clause


def test_escalations_render_as_trigger_and_action():
    assert "E4 -> approve" in _fmt_result_summary(_P205_212, _REPORT)


def test_no_escalations_renders_none_not_an_empty_line():
    out = _fmt_result_summary({**_P205_212, "escalations": []}, _REPORT)
    assert "Escalations  : none" in out


def test_report_shape_is_measured_from_the_report_itself():
    # Section count and length are NOT in telemetry, and they are exactly
    # what exposed p205.211: 3 sections / 2,737 chars, then 0 sections /
    # 652 chars on the revision. Counted here from the text.
    out = _fmt_result_summary(_P205_212, _REPORT)
    assert "3 section(s)" in out
    assert f"{len(_REPORT):,} chars" in out


def test_the_p205_211_failure_shape_is_visible_at_a_glance():
    # A stub report with no headings at all -- the exact artifact the old
    # console output printed without comment.
    stub = "India and the US differ in several respects, including growth."
    out = _fmt_result_summary({**_P205_212, "critique_passed": False}, stub)
    assert "0 section(s)" in out
    assert "FAILED" in out


def test_evidence_sources_are_listed_not_just_totalled():
    out = _fmt_result_summary(_P205_212, _REPORT)
    assert "128 item(s)" in out
    assert "corpus 40" in out and "web 30" in out and "model 20" in out


def test_grounding_figures_appear_beside_recall():
    # recall 1.0 next to grounded 0.0 is the honest, and startling, pair:
    # every goal has evidence attached and none of the report rests on the
    # corpus. Separating them across 40 lines of JSON hides that.
    out = _fmt_result_summary(_P205_212, _REPORT)
    assert "Recall       : 1.0" in out
    assert "grounded 0.0" in out
    assert "corpus_recall 0.0" in out


def test_quality_check_failures_are_shown_as_a_ratio():
    assert "1/2 quality check(s) failed" in _fmt_result_summary(_P205_212, _REPORT)


def test_planning_error_line_appears_only_when_there_is_one():
    assert "Planning" not in _fmt_result_summary(_P205_212, _REPORT)
    out = _fmt_result_summary({**_P205_212, "planning_error": "zero goals"},
                              _REPORT)
    assert "Planning     : zero goals" in out


def test_empty_telemetry_renders_instead_of_raising():
    # A run that never reached the telemetry node arrives here with {}.
    # The summary is MOST useful exactly then, so it must not KeyError.
    out = _fmt_result_summary({}, "(no report was produced)")
    assert "=== RESULT ===" in out
    assert "n/a" in out
    assert "did not reach the critic" in out


def test_summary_invents_no_numbers_of_its_own():
    # D-12's rule: aggregate, never invent. Every figure shown must trace
    # to a telemetry key or to the report text. Guard against a future
    # edit computing a derived score here.
    out = _fmt_result_summary(_P205_212, _REPORT)
    shown = {tok.strip(" ,()") for line in out.splitlines()
             for tok in line.split() if tok.strip(" ,()").replace(".", "").isdigit()}
    allowed = {str(v) for v in _P205_212.values() if isinstance(v, (int, float))}
    allowed |= {str(v) for v in _P205_212["evidence_by_source"].values()}
    allowed |= {"3", str(len(_REPORT)), f"{len(_REPORT):,}", "0"}
    assert shown <= allowed, f"unexplained figures in summary: {shown - allowed}"
