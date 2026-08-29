"""
guardrails/fencing.py — the prompt-injection mitigation for retrieved
evidence text (D-18's defence-in-depth measure).

CALLED BY   prompts/templates.py — every builder that inlines retrieved
            Evidence text into a prompt (compile_report, generate_gaps,
            detect_contradictions).

Moved from prompts/templates.py, unchanged. The <evidence>...</evidence>
tags themselves are still opened by each caller in prompts/templates.py,
right next to the prompt-specific instruction text — only the delimiter-
neutralising helper moved here, since it's a generic string operation with
no dependency on which prompt is calling it.
"""

import re

# D-129 (P6-1): the delimiter itself, defined ONCE.
#
# fence_untrusted below neutralises these two strings INSIDE untrusted
# content; EVIDENCE_SPAN_RE is how a caller finds a whole fenced span
# again afterwards -- evaluation/quality.py removes it from the
# transcript the quality judge is sent, so a scoring call no longer
# resends the entire evidence block it was never asked to check. Both
# jobs are the same fact about the same delimiter, so they read it from
# one place instead of two independent literals that can drift apart
# (M-1's rule, applied to a string rather than to a predicate).
#
# NON-GREEDY MATCHING IS SAFE HERE, and only because of fence_untrusted:
# content inside a span can never contain a literal "</evidence>", since
# that function already replaced it with "(/evidence)" on the way in. So
# the first closing tag after an opening one is always that span's real
# end, and `.*?` cannot swallow a second span.
#
# prompts/templates.py still writes the tags as literals inside its
# f-strings, deliberately unchanged: substituting a constant into three
# multi-line f-strings costs readability at every call site to remove a
# duplication that has never drifted -- and those call sites OPEN the
# span around content this module has already neutralised, which is the
# direction that actually matters.
EVIDENCE_OPEN = "<evidence>"
EVIDENCE_CLOSE = "</evidence>"
EVIDENCE_SPAN_RE = re.compile(
    re.escape(EVIDENCE_OPEN) + r".*?" + re.escape(EVIDENCE_CLOSE),
    re.DOTALL)


def fence_untrusted(text: str) -> str:
    """Neutralise the evidence delimiter inside retrieved content.

    WHY THIS EXISTS: Evidence.content comes from an ingested corpus or a
    third-party MCP server, i.e. from OUTSIDE this system's trust
    boundary, and was previously interpolated into prompts verbatim with
    no delimiter and no instruction to treat it as data. A single
    poisoned document could therefore address the model directly — most
    consequentially the contradiction detector, where "return an empty
    contested_goal_ids" defeats D-18 outright and leaves a genuinely
    contested goal looking covered.

    The <evidence> tags (opened by each caller, with this function
    stripping any the content itself contains) plus the shared system
    prompt (prompts/templates.py::_SYSTEM) are the standard RAG
    mitigation: mark the untrusted span, and state that spans so marked
    are never instructions. This is defence in depth, not a guarantee —
    no prompt-level measure is — but it closes the trivially exploitable
    version.
    """
    return (text.replace(EVIDENCE_OPEN, "(evidence)")
                .replace(EVIDENCE_CLOSE, "(/evidence)"))
