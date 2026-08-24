"""
tests/unit/test_reporting_metrics.py -- reporting/metrics.py::count_sections
(S-10).

WHY THIS FILE EXISTS: compiler_node's narrative log and cli.py's terminal
RESULT block used to count "sections" with two different regexes and
disagreed on the same report in the same run (25 vs 8). These tests pin
the one shared definition both call sites now use.
"""

from research_agent.reporting.metrics import count_sections


def test_counts_level_two_headings():
    report = "# Title\n\n## First\n\nbody\n\n## Second\n\nbody"
    assert count_sections(report) == 2


def test_does_not_count_the_level_one_title():
    """The report's own H1 title is not a section of itself -- this is
    the exact discrepancy S-10 fixes (compiler_node's old regex counted
    it, cli.py's old check did not)."""
    report = "# Comparative Analysis\n\n## Overview\n\nbody"
    assert count_sections(report) == 1


def test_counts_headings_at_every_level_two_through_six():
    report = "# Title\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5\n\n###### H6"
    assert count_sections(report) == 5


def test_a_report_with_only_a_title_has_zero_sections():
    assert count_sections("# Comparative Analysis\n\nno sections here") == 0


def test_empty_report_has_zero_sections():
    assert count_sections("") == 0


def test_a_hash_inside_a_url_or_prose_is_not_a_heading():
    """The regex is anchored to the start of a line -- a stray '#' mid
    text (a URL fragment, a hashtag mentioned in prose) must not count."""
    report = "# Title\n\nSee https://example.org/page#section for details."
    assert count_sections(report) == 0
