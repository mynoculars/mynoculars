"""
reporting/pipeline.py -- the report post-processing passes, as an ordered
list you can read (D-146).

WHAT THIS REPLACES, and why the replacement is the point.

compiler_node ran twelve post-processing steps as straight-line code with
a 10-20 line comment between each one explaining why it sat where it sat.
Every constraint in those comments is real and was learned from a live
failure. None of them was visible: nothing in the code said "this is a
pipeline with ordering constraints", so the only way to know that
append_web_sources must follow clean_citations was to read the paragraph
above it -- and the only thing keeping it there was that nobody moved it.

Measured, before this change: agents/compilation.py was 1,149 lines, of
which 707 were comment or docstring against 419 of executable code -- a
1.69:1 prose-to-code ratio in the module that produces the deliverable.
That is the shape of the complaint this addresses: not that the comments
are wrong, but that they were the only place the design lived.

So the ordering rationale moves into data. Each pass declares the passes it
must follow, in `after`, and test_reporting_pipeline.py verifies that
REPORT_PASSES is a valid topological order of those declarations. A comment
asking politely that the order be preserved is now an assertion that fails
if it is not -- and the full argument for each constraint lives in
DECISIONS.md and in each guardrail module's own docstring, where it is
read once rather than re-read on every pass through the compiler.

WHAT DID NOT CHANGE. The order itself, the functions, their arguments and
their counters are all byte-for-byte what compiler_node did before. This
is a restructuring, not a behaviour change, and the existing
test_agents_compilation.py suite is what says so.

CALLED BY   agents/compilation.py::compiler_node, once per compile.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from research_agent.guardrails.attribution import attach_missing_citations
from research_agent.guardrails.citations import (clean_citations,
                                                 normalise_citation_form,
                                                 repair_glued_sentences)
from research_agent.guardrails.grounding import annotate_ungrounded_report
from research_agent.guardrails.hedging import enforce_hedging
from research_agent.guardrails.sources import append_web_sources
from research_agent.guardrails.truncation import annotate_truncated_run
from research_agent.llm.client import strip_code_fence

logger = logging.getLogger(__name__)

# A pass returns (report, counters). Several of the underlying guardrails
# already have exactly that shape; the two that do not are adapted below
# rather than changed, so nothing else that calls them has to move.
PassResult = Tuple[str, Dict[str, float]]


@dataclass(frozen=True)
class PassContext:
    """Everything the passes read, gathered once.

    Frozen because a pass must never be able to change what a later pass
    sees -- the report is the only thing that flows down the chain.
    """

    goals: List[Any]
    evidence: List[Any]
    guidance: str
    budget_exhausted: Optional[str]
    llm_mode: str
    min_evidence_score: float
    grounded_recall_target: float


@dataclass(frozen=True)
class ReportPass:
    """One post-processing step, with its ordering constraint attached.

    `after` names the passes this one must follow. It is the machine-
    readable half of what used to be a comment; test_reporting_pipeline.py
    reads it and fails if REPORT_PASSES is not a valid topological order.

    `why` is one line, for a reader of this file. The full argument lives
    in the guardrail's own module docstring and in DECISIONS.md -- this is
    a pointer, not a second copy that can drift.

    `enabled` gates a pass on the context. Only the D-85 provenance notice
    uses it today (off in stub mode).
    """

    name: str
    fn: Callable[[str, PassContext], PassResult]
    after: Tuple[str, ...]
    why: str
    enabled: Callable[[PassContext], bool] = lambda ctx: True


def _strip_fence(report: str, ctx: PassContext) -> PassResult:
    return strip_code_fence(report), {}


def _normalise_form(report: str, ctx: PassContext) -> PassResult:
    return normalise_citation_form(report, ctx.goals)


def _attach_citations(report: str, ctx: PassContext) -> PassResult:
    return attach_missing_citations(report, ctx.goals, ctx.evidence)


def _clean_citations(report: str, ctx: PassContext) -> PassResult:
    return clean_citations(report, ctx.goals, ctx.evidence)


def _repair_glue(report: str, ctx: PassContext) -> PassResult:
    return repair_glued_sentences(report)


def _enforce_hedging(report: str, ctx: PassContext) -> PassResult:
    return enforce_hedging(report, ctx.evidence)


def _web_sources(report: str, ctx: PassContext) -> PassResult:
    return append_web_sources(report, ctx.evidence, ctx.goals, ctx.guidance,
                              list_when_uncited=True)


def _grounding_notice(report: str, ctx: PassContext) -> PassResult:
    return annotate_ungrounded_report(report, ctx.goals, ctx.evidence,
                                      ctx.min_evidence_score,
                                      ctx.grounded_recall_target)


def _truncation_notice(report: str, ctx: PassContext) -> PassResult:
    return annotate_truncated_run(report, ctx.budget_exhausted)


# The pipeline. Order here is the order that runs; `after` is what makes
# that order checkable rather than merely conventional.
REPORT_PASSES: Tuple[ReportPass, ...] = (
    ReportPass(
        name="strip_fence", fn=_strip_fence, after=(),
        why="a fallback provider can still fence its Markdown answer; "
            "everything downstream reads the report as Markdown"),
    ReportPass(
        name="normalise_form", fn=_normalise_form, after=("strip_fence",),
        why="D-99: settle the citation FORM first -- `(g1)` is not a "
            "malformed citation to the readers below, it is no citation "
            "at all, and they all fail silently together"),
    ReportPass(
        name="attach_citations", fn=_attach_citations,
        after=("normalise_form",),
        why="D-144: only after the form is settled can 'does this cite "
            "anything' be asked; before clean_citations so what this "
            "attaches is validated by the same guard as the model's own"),
    ReportPass(
        name="clean_citations", fn=_clean_citations,
        after=("attach_citations",),
        why="D-43/D-45: drop markers for goals that retrieved nothing, and "
            "remove pasted source text"),
    ReportPass(
        name="repair_glue", fn=_repair_glue, after=("clean_citations",),
        why="D-137: after clean_citations, so a verbatim paste is DELETED "
            "rather than merely punctuated -- the stronger verdict gets "
            "first refusal at every site"),
    ReportPass(
        name="enforce_hedging", fn=_enforce_hedging, after=("clean_citations",),
        why="Guardrail G3: the compiler's instruction to hedge "
            "UNVERIFIED-SPECIFIC claims is not reliably followed alone"),
    ReportPass(
        name="web_sources", fn=_web_sources,
        after=("clean_citations", "repair_glue", "enforce_hedging"),
        why="D-57: LAST of the text passes. Both of those search the "
            "report for literal spans of evidence content, and a block "
            "full of titles and URLs is exactly what could be mistaken "
            "for a paste; repair_glue would also punctuate a URL"),
    ReportPass(
        name="grounding_notice", fn=_grounding_notice, after=("web_sources",),
        why="D-85: after web_sources, which keeps the notice clear of the "
            "Sources block that count_listed_sources parses back out",
        # Gated off in stub mode exactly like D-66's zero-citation gate:
        # StubClient's fixed placeholder report exists to prove the graph
        # executes offline and models nothing about where evidence came
        # from. Annotating it would be noise in the one mode that is
        # deliberately not a real answer.
        enabled=lambda ctx: ctx.llm_mode != "stub"),
    ReportPass(
        name="truncation_notice", fn=_truncation_notice,
        after=("grounding_notice",),
        why="D-132: LAST of all, which puts this notice ABOVE the "
            "provenance one in the shipped text -- 'this run was stopped "
            "early' changes how a reader should weigh everything below "
            "it, that notice included"),
)


def run_report_passes(report: str, ctx: PassContext) -> PassResult:
    """Run every enabled pass in order. -> (report, merged counters).

    Counters merge by later-wins, which is what the dict-splat chain in
    compiler_node did before -- no two passes emit the same key, and a
    test asserts that stays true.
    """
    counters: Dict[str, float] = {}
    for step in REPORT_PASSES:
        if not step.enabled(ctx):
            continue
        report, produced = step.fn(report, ctx)
        counters.update(produced)
    return report, counters
