"""
tests/unit/test_reporting_pipeline.py — reporting/pipeline.py (D-146).

The ordering constraints in this pipeline were each learned from a live
failure and were, until D-146, recorded only in comments beside the code.
These tests are what turns that record into an invariant: REPORT_PASSES
must be a valid topological order of every pass's declared `after`, so
reordering the list without also reasoning about the constraint fails here
rather than in a shipped report six weeks later.
"""

import pytest

from research_agent.reporting.pipeline import (REPORT_PASSES, PassContext,
                                               ReportPass, run_report_passes)


def _ctx(**overrides):
    base = dict(goals=[], evidence=[], guidance="", budget_exhausted=None,
                llm_mode="live", min_evidence_score=0.5,
                grounded_recall_target=0.5)
    base.update(overrides)
    return PassContext(**base)


# ---------------------------------------------------------------------------
# The invariant that replaces twelve comments
# ---------------------------------------------------------------------------


def test_the_declared_order_is_a_valid_topological_order():
    """Every pass must appear AFTER every pass it declares it follows."""
    position = {step.name: i for i, step in enumerate(REPORT_PASSES)}

    for step in REPORT_PASSES:
        for required in step.after:
            assert required in position, (
                f"{step.name} declares after={required!r}, which is not a pass")
            assert position[required] < position[step.name], (
                f"{step.name} must run after {required}, but runs before it")


def test_every_pass_name_is_unique():
    names = [step.name for step in REPORT_PASSES]

    assert len(names) == len(set(names))


def test_no_pass_declares_itself():
    for step in REPORT_PASSES:
        assert step.name not in step.after


def test_every_pass_explains_itself_in_one_line():
    """`why` is a pointer, not a second copy of the rationale -- but it must
    exist, or this file becomes the comment-free version of the problem it
    was written to fix."""
    for step in REPORT_PASSES:
        assert step.why and len(step.why) > 20, step.name


# ---------------------------------------------------------------------------
# The specific constraints, named, so a reader sees WHICH ones are load-bearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("earlier,later,decision", [
    ("normalise_form", "attach_citations", "D-99/D-144"),
    ("attach_citations", "clean_citations", "D-144"),
    ("clean_citations", "repair_glue", "D-137"),
    ("clean_citations", "web_sources", "D-57"),
    ("repair_glue", "web_sources", "D-57"),
    ("enforce_hedging", "web_sources", "D-57"),
    ("web_sources", "grounding_notice", "D-85"),
    ("grounding_notice", "truncation_notice", "D-132"),
])
def test_the_load_bearing_orderings_hold(earlier, later, decision):
    position = {step.name: i for i, step in enumerate(REPORT_PASSES)}

    assert position[earlier] < position[later], decision


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_no_two_passes_emit_the_same_counter_key():
    """run_report_passes merges by later-wins, exactly as compiler_node's
    dict-splat chain did. That is only safe while the keys are disjoint."""
    seen = {}
    report = ("## Findings\n\nA sentence that says something specific about "
              "the subject at hand.\n")
    for step in REPORT_PASSES:
        _, produced = step.fn(report, _ctx())
        for key in produced:
            assert key not in seen, (
                f"{step.name} and {seen[key]} both emit {key!r}")
            seen[key] = step.name


def test_a_disabled_pass_is_skipped_entirely():
    """Stub mode gates the D-85 provenance notice off; it must not merely
    no-op, it must not run."""
    calls = []

    def _spy(report, ctx):
        calls.append(report)
        return report, {}

    passes = (ReportPass(name="spy", fn=_spy, after=(), why="x" * 30,
                         enabled=lambda ctx: False),)
    import research_agent.reporting.pipeline as pipeline

    original = pipeline.REPORT_PASSES
    try:
        pipeline.REPORT_PASSES = passes
        out, counters = pipeline.run_report_passes("body", _ctx())
    finally:
        pipeline.REPORT_PASSES = original

    assert calls == []
    assert out == "body" and counters == {}


def test_the_stub_gate_is_what_disables_the_provenance_notice():
    notice = next(s for s in REPORT_PASSES if s.name == "grounding_notice")

    assert notice.enabled(_ctx(llm_mode="live")) is True
    assert notice.enabled(_ctx(llm_mode="stub")) is False


def test_an_empty_report_survives_every_pass():
    """An abort or planning-error report reaches the compiler with nothing
    to post-process; every pass must no-op rather than raise.

    append_web_sources' zeroed counters are its documented contract even on
    the no-op path, so they are expected here -- what must not appear is
    any counter reporting WORK done."""
    out, counters = run_report_passes("", _ctx())

    assert out == ""
    assert not any(value for value in counters.values()), counters


def test_the_context_cannot_be_mutated_by_a_pass():
    """Frozen on purpose: the report is the only thing that flows down the
    chain, and a pass changing what a later pass sees would make the
    ordering constraints above meaningless."""
    ctx = _ctx()

    with pytest.raises(Exception):
        ctx.goals = ["something else"]
