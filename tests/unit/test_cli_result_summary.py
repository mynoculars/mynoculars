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

from research_agent.cli import (_fmt_judge_line,
                                 _fmt_result_summary)


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
    # D-176: two CRITIC PASSES over one report is one revision.
    assert "1 revision(s)" in out


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


# ---------------------------------------------------------------------------
# D-108 -- what the judge DECIDED reaches the block a human reads
#
# D-106 recorded the mean, the rejections and the distribution and routed
# them to agent_runs and the cross-run report. The RESULT block printed
# after EVERY run still showed only "0/2 quality check(s) failed" -- the
# failure ratio, and nothing about the judgement. Recording a signal and
# then not showing it is the defect PHASE5-HONESTY 14.6 and 16.5 both
# record; this is the third instance, closed.
# ---------------------------------------------------------------------------


def test_the_judge_line_reports_mean_rejections_and_distribution():
    line = _fmt_judge_line({
        "llm_quality_scores_judged": 3, "llm_quality_score_mean": 0.5,
        "llm_quality_rejections": 2,
        "llm_quality_bands": {"very_low": 1, "mid": 1, "very_high": 1},
        "llm_quality_calls": 3, "llm_quality_calls_failed": 0})

    assert "3 judgement(s)" in line
    assert "mean 0.5" in line
    assert "2 below threshold" in line
    assert "very_low 1" in line and "very_high 1" in line


def test_a_run_whose_judge_failed_every_time_says_the_gate_was_inert():
    """The state D-53's WARNING exists for, said in the summary rather
    than only in a log line: a run with no working judge had NO quality
    gate, and every other number on that screen looks exactly as it would
    have if the gate had passed everything."""
    line = _fmt_judge_line({"llm_quality_calls": 2, "llm_quality_calls_failed": 2})

    assert "no judgement" in line
    assert "inert" in line
    assert "mean" not in line, "there is no mean to report"


def test_a_run_that_never_needed_a_second_opinion_says_so():
    """The common case -- a single-provider chain, or a first answer that
    cleared the gate with no hop. It must not read as a failure."""
    line = _fmt_judge_line({})

    assert "not asked" in line
    assert "failed" not in line


def test_partial_judge_failure_is_reported_alongside_the_scores():
    """Some scored, some failed open. Both facts matter and neither
    replaces the other."""
    line = _fmt_judge_line({
        "llm_quality_scores_judged": 1, "llm_quality_score_mean": 0.9,
        "llm_quality_rejections": 0, "llm_quality_bands": {"very_high": 1},
        "llm_quality_calls": 3, "llm_quality_calls_failed": 2})

    assert "1 judgement(s)" in line
    assert "2 scoring call(s) failed open" in line


def test_the_result_summary_carries_the_judge_line():
    summary = _fmt_result_summary(
        {"llm_quality_scores_judged": 1, "llm_quality_score_mean": 0.4,
         "llm_quality_rejections": 1, "llm_quality_bands": {"low": 1},
         "llm_quality_calls": 1, "llm_quality_calls_failed": 0},
        "# A report")

    assert "Quality judge:" in summary
    # The pre-existing attempt/failure line must still be there -- the two
    # answer different questions and neither replaces the other.
    assert "quality check(s) failed" in summary


def test_a_telemetry_dict_with_no_quality_keys_at_all_still_formats():
    """Every pre-D-106 row, and every stub run."""
    assert "Quality judge:" in _fmt_result_summary({}, "")


# ---------------------------------------------------------------------------
# D-145 / D-144 -- the three lines the RESULT block never had
#
# p205.280-check printed six raw signals and left the reader to integrate
# them. Finding "0 sources listed against 58 web items" meant reading 45
# lines of JSON.
# ---------------------------------------------------------------------------


def _p205_280_telemetry():
    from research_agent.reporting.confidence import score_report

    t = {
        "goals": 4, "evidence_items": 100, "evidence_cited": 0,
        "citations_attached": 0, "grounding_ratio": 1.0, "recall": 1.0,
        "corpus_recall": 0.0, "grounded_score": 0.0,
        "retrieval_floor_drop_ratio": 1.0, "llm_quality_score_mean": 0.067,
        "llm_quality_scores_judged": 3, "llm_quality_calls": 3,
        "llm_quality_calls_failed": 0, "llm_quality_rejections": 3,
        "llm_quality_bands": {"very_low": 2, "low": 1},
        "critique_passed": False, "revision_cycles": 2,
        "cited_figures_unsupported": 0, "goals_without_evidence": [],
        "escalations": [{"trigger": "E4", "action": "approve"}],
        "evidence_by_source": {"memory": 5, "web": 58, "corpus": 19, "mcp": 18},
        "web_sources_listed": 0, "web_sourced_items": 58,
        "web_source_domains": 33, "llm_provider_calls": 8,
        "llm_fallback_hops": 2, "search_calls": 12, "search_failures": 0,
    }
    t["confidence"] = score_report(t)
    return t


def test_the_verdict_is_the_first_line_after_the_banner():
    out = _fmt_result_summary(_p205_280_telemetry(), "x" * 4341)

    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines[0] == "=== RESULT ==="
    assert lines[1].startswith("Confidence   : UNRELIABLE")


def test_the_attribution_failure_is_visible_without_reading_json():
    out = _fmt_result_summary(_p205_280_telemetry(), "x" * 4341)

    assert "Citations    : 0 goal(s) cited in the prose" in out
    assert "100 evidence item(s) available" in out
    assert "Sources      : 0 listed / 58 web item(s) across 33 domain(s)" in out


def test_a_deterministic_rescue_says_so_on_the_citations_line():
    """A reader must never be unable to tell a report the model cited from
    one this codebase repaired."""
    from research_agent.reporting.confidence import score_report

    t = _p205_280_telemetry()
    t.update({"evidence_cited": 4, "citations_attached": 5,
              "web_sources_listed": 12})
    t["confidence"] = score_report(t)

    out = _fmt_result_summary(t, "x" * 4341)

    assert "5 attached deterministically" in out


def test_a_run_with_no_web_evidence_says_so_rather_than_printing_zeroes():
    """WEB_SEARCH_ENABLED defaults false, so this is the common shape."""
    from research_agent.reporting.confidence import score_report

    t = _p205_280_telemetry()
    t.update({"web_sourced_items": 0, "web_source_domains": 0,
              "web_sources_listed": 0})
    t["confidence"] = score_report(t)

    out = _fmt_result_summary(t, "x" * 100)

    assert "Sources      : 0 listed (no web evidence retrieved)" in out


def test_an_empty_telemetry_dict_still_renders():
    """A degraded or interrupted run reaches this line, and the whole point
    of the summary is to stay readable exactly then."""
    out = _fmt_result_summary({}, "")

    assert "=== RESULT ===" in out
    assert "Confidence   : n/a" in out
def test_a_single_critic_pass_is_not_reported_as_a_revision():
    """D-176, the p205.302-check defect.

    `revision_cycles` is bumped once per critic INVOCATION, including the
    first, so a run whose execution plan is a single compiler -> critic
    pass arrives here with 1. It reported "PASSED after 1 revision
    cycle(s)" -- a revision that never happened, on the one line of the
    run summary a showcase audience actually reads.
    """
    out = _fmt_result_summary({**_P205_212, "critique_passed": True,
                               "revision_cycles": 1}, _REPORT)

    assert "PASSED" in out
    assert "revision" not in out


def test_three_critic_passes_report_two_revisions():
    out = _fmt_result_summary({**_P205_212, "critique_passed": True,
                               "revision_cycles": 3}, _REPORT)

    assert "2 revision(s)" in out
