"""
guardrails/hedging.py — Guardrail G3's enforcement half (P205.135
follow-up).

CALLED BY   agents/compilation.py::compiler_node, immediately after
            clean_citations -- same call site, same shape of check
            (deterministic post-processing of the compiled report
            against state.evidence), just a different failure mode.

WHY THIS EXISTS: tools/model_knowledge.py flags a model-tier claim as
overspecific (Evidence.hedge_specific=True) when its own text pairs a
precise year with a precise quantity -- see that module's docstring.
prompts/templates.py::compile_report's ATTRIBUTION RULE then INSTRUCTS
the compiler to round or hedge any evidence item tagged
UNVERIFIED-SPECIFIC rather than restating it with the same false
precision. Live evidence (run p205.135-check) shows that instruction is
not reliably followed: hedge_specific_items reached 29 that run, and
the shipped report still stated "target to install 500 GW... by 2030",
"5 million metric tonnes... by 2030", "60.5 million metric tonnes... by
2025-26" as flat, uncited fact -- the exact quantities their source
evidence items were flagged for, carried through unhedged. The critic
caught this (critique_passed: false, five unsupported-claim notes) but
a human then approved anyway -- HITL working as designed (a human has
final say), but the mechanism by which a soft, ignorable LLM
instruction reached a shipped report regardless.

WHAT THIS DOES: for every hedge_specific evidence item, find its
flagged quantity span (tools/model_knowledge.py::overspecific_span) and
check whether that EXACT text survived into the compiled report. If it
did, and the surrounding text doesn't already carry an obvious hedge
word, append a visible "(unverified figure)" marker right after it --
deterministic, cannot be silently skipped the way a prompt instruction
can. This is detection made durable, not a rewrite: the number itself
is never changed or removed (that would risk fabricating a DIFFERENT
wrong number), only flagged in place, the same "annotate, don't guess
at the author's intent" posture citations.py already takes with pasted
evidence and unevidenced-goal markers.

WHAT THIS DELIBERATELY DOES NOT DO: it does not judge whether an
already-hedged claim ("approximately 500 GW") is ALSO unsupported --
that's still the critic's job, same division of labor D-40 already
draws between this file's deterministic half and the LLM's semantic
half. It also does not touch claims from corpus/mcp evidence, or
model-tier claims that were never flagged as overspecific in the first
place -- only the exact spans this codebase's own detector already
flagged, and only where the compiler ignored the request to hedge them.
"""

from typing import Dict, List, Tuple

from research_agent.state import Evidence
from research_agent.tools.model_knowledge import overspecific_span

# Any of these appearing shortly before a flagged figure means the
# compiler DID hedge it, just not with a bracketed marker -- "roughly
# 500 GW" is honest prose and must not be double-flagged. Checked as a
# plain substring test over a short preceding window, not a full NLP
# pass: this codebase's guardrails are deliberately string-level checks,
# not judgment calls (see guardrails/__init__.py).
_HEDGE_WORDS = ("approximately", "roughly", "about", "around", "estimated",
                "nearly", "on the order of", "some")
_HEDGE_LOOKBACK = 30
_MARKER = " (unverified figure)"


def _already_hedged(report_lower: str, start: int) -> bool:
    window = report_lower[max(0, start - _HEDGE_LOOKBACK):start]
    return any(w in window for w in _HEDGE_WORDS)


def enforce_hedging(report: str, evidence: List[Evidence]) -> Tuple[str, Dict[str, float]]:
    """Deterministically flag any UNVERIFIED-SPECIFIC figure that
    survived into `report` unhedged.

    Returns (annotated_report, {counter_name: count}) -- same shape as
    guardrails/citations.py::clean_citations, so compiler_node can merge
    both counter dicts into state.counters identically.
    """
    counters: Dict[str, float] = {}
    tagged = 0
    seen_spans: set = set()

    for e in evidence:
        if not e.hedge_specific:
            continue
        span = overspecific_span(e.content)
        if not span:
            # Content changed shape since the flag was set, or the flag
            # came from elsewhere -- nothing concrete to search for.
            continue
        key = span.lower()
        if key in seen_spans:
            continue  # already processed this exact quantity once
        seen_spans.add(key)

        lower_report = report.lower()
        start = lower_report.find(key)
        while start != -1:
            end = start + len(span)
            already_marked = report[end:end + len(_MARKER)] == _MARKER
            if not already_marked and not _already_hedged(lower_report, start):
                report = report[:end] + _MARKER + report[end:]
                lower_report = report.lower()
                tagged += 1
                start = lower_report.find(key, end + len(_MARKER))
            else:
                start = lower_report.find(key, end)

    if tagged:
        counters["hedge_markers_inserted"] = float(tagged)
    return report, counters
