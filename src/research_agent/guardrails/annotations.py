"""
guardrails/annotations.py — separating what the MODEL wrote from what this
system inserted (D-139).

CALLED BY   agents/compilation.py::critic_node, on the report it is about
            to judge.

WHY THIS EXISTS. compiler_node returns `final_report` only after four
passes have added text the model never wrote: D-85's provenance notice and
D-132's stopped-early notice are PREPENDED, D-57's `## Sources` block is
APPENDED. critic_node then judged `state.final_report` — all of it.

Live (run p205.276-check) the critic's verdict came back:

    "The report's 'Provenance notice' section contains several claims that
     are not supported by any evidence item ... should be removed."
    "... the provenance notice and meta-text inside the report body must be
     removed to restore faithfulness."

Three of that critique's six notes were about the notice. They are not
wrong on their own terms — the notice genuinely is untagged, unsupported
meta-text — but they are UNACTIONABLE: annotate_ungrounded_report re-adds
it after every compile, so no rewrite can satisfy them, and the compiler
spent a revision trying. The very next compile dropped its citations
entirely and the run escalated.

WHAT THIS IS NOT. The notice is not being hidden from the READER — it is
the one part of the report guaranteed to be true (D-85), and it ships
exactly as before. This changes only what the CRITIC is asked to be
faithful to: the model can only be held to the text the model produced.
"""

import re

from .grounding import NOTICE_MARKER as GROUNDING_NOTICE_MARKER
from .sources import SOURCES_HEADING
from .truncation import NOTICE_MARKER as TRUNCATION_NOTICE_MARKER

# Each notice is a markdown blockquote: consecutive lines opening with ">".
# Matching the BLOCK rather than a fixed number of lines means a reworded
# notice cannot leave half of itself behind — the markers are imported from
# the modules that write them, so there is one spelling of each, not two.
_NOTICE_MARKERS = (GROUNDING_NOTICE_MARKER, TRUNCATION_NOTICE_MARKER)

_SOURCES_RE = re.compile(
    r"\n*" + re.escape(SOURCES_HEADING) + r"\s*\n.*\Z", re.DOTALL)


def _strip_notice_blocks(report: str) -> str:
    """Drop every blockquote block that carries a machine notice marker."""
    lines = report.splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(">") and any(
                marker in line for marker in _NOTICE_MARKERS):
            # Consume the whole quote, plus the blank lines it is separated
            # from the report by, so removing it leaves no gap the critic
            # could read as a formatting fault of the model's.
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def strip_machine_annotations(report: str) -> str:
    """The report as the MODEL wrote it: notices and Sources block removed.

    Returns the report unchanged when it carries no annotations, which is
    the common path — every run whose corpus grounds the answer ships no
    notice, and every run with WEB_SEARCH_ENABLED false ships no Sources
    block.

    Deliberately NOT applied to the zero-citation gate (D-66), which reads
    `state.final_report` and is correct as it stands: `append_web_sources`
    only lists sources for goals the PROSE already cites, so a report whose
    prose cites nothing has no Sources block for the gate to miscount. Read
    against the code rather than assumed — the gate needs no change and is
    left alone.
    """
    if not report:
        return report
    return _SOURCES_RE.sub("", _strip_notice_blocks(report)).rstrip() + "\n"
