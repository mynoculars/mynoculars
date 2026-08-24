"""
tests/unit/test_cli_hitl_wall_time.py -- cli.py::_fmt_hitl_wall_time_line.

Covers the "HITL triggered | Total wall time: ..." / "HITL Not triggered |
..." line printed once a run finishes, alongside the RESULT summary. Its
own file rather than appended to test_cli_result_summary.py, for the same
delivery reason recorded in DECISIONS.md D-62/D-63.
"""

from research_agent.cli import _fmt_hitl_wall_time_line


def test_reports_hitl_triggered_when_a_human_review_fired():
    line = _fmt_hitl_wall_time_line(True, 10.0)
    assert line.startswith("HITL triggered |")


def test_reports_hitl_not_triggered_when_no_human_review_fired():
    line = _fmt_hitl_wall_time_line(False, 10.0)
    assert line.startswith("HITL Not triggered |")


def test_formats_minutes_and_seconds():
    """2 minutes, 5.40 seconds -- must not collapse into raw seconds or
    drop the fractional part."""
    line = _fmt_hitl_wall_time_line(True, 125.4)
    assert "2min 5.40secs" in line


def test_formats_under_a_minute_as_zero_minutes():
    line = _fmt_hitl_wall_time_line(False, 5.2)
    assert "0min 5.20secs" in line


def test_formats_over_an_hour_by_rolling_minutes_not_hours():
    """No hours component -- this codebase's runs are measured in
    minutes; 65 minutes prints as 65min, not 1h 5min."""
    line = _fmt_hitl_wall_time_line(True, 3900.0)
    assert "65min 0.00secs" in line


def test_zero_elapsed_time_does_not_error():
    line = _fmt_hitl_wall_time_line(False, 0.0)
    assert "0min 0.00secs" in line
