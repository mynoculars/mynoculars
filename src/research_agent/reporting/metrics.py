"""
reporting/metrics.py -- shared, cheap metrics computed FROM a rendered
report string (S-10), rather than duplicated at each call site that
wants one.

Purpose:
    count_sections() is read by both agents/compilation.py::compiler_node
    (the "node.compiled" log line) and cli.py::_fmt_result_summary (the
    terminal RESULT block) -- previously two different regexes counting
    two different things and calling both "sections". Live (same run,
    same report): the narrative log said `sections=25`, the terminal
    RESULT block said `8 section(s)`. Both are user-facing; a reader
    comparing them saw a contradiction with no way to resolve it. One
    function, one definition, used by both.
"""

import re

# Level-2-through-6 headings only. A level-1 `# ` heading is the report's
# TITLE (compile_report's own template emits exactly one, at the top),
# not a section of it -- counting it inflated every report's section
# count by exactly one, which is what compiler_node's prior `^#{1,6} `
# regex did and cli.py's prior `## `-prefix check did not, and is the
# single biggest reason the two numbers disagreed.
_SECTION_HEADING_RE = re.compile(r"(?m)^#{2,6} ")


def count_sections(report: str) -> int:
    """How many level-2-to-6 Markdown headings `report` contains.

    Pure and cheap -- safe to call from both a log line and a terminal
    summary without worrying about cost or side effects.
    """
    return len(_SECTION_HEADING_RE.findall(report))
