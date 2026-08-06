"""
tests/unit/test_guardrails_hedging.py — guardrails/hedging.py's
enforce_hedging (Guardrail G3's enforcement half, P205.135 follow-up).

Regression target: run p205.135-check, where hedge_specific_items
reached 29 but the shipped report still stated several of those exact
flagged quantities as flat fact -- "target to install 500 GW... by
2030", "5 million metric tonnes... by 2030" -- because the compiler
prompt's instruction to hedge them is not reliably followed on its own.
"""

from research_agent.guardrails.hedging import enforce_hedging
from research_agent.state import Evidence


def _model_evidence(content: str, hedge_specific: bool = True) -> Evidence:
    return Evidence(task_key="t1", goal_id="g5", source="model",
                    content=content, score=0.6, hedge_specific=hedge_specific)


def test_tags_an_unhedged_flagged_figure_that_survived_into_the_report():
    evidence = [_model_evidence(
        "India's National Solar Mission targets 500 GW of non-fossil "
        "capacity by 2030.")]
    report = ("## Environmental Policy\nIndia's target to install 500 GW "
              "of renewable energy capacity by 2030 is ambitious.")
    cleaned, counters = enforce_hedging(report, evidence)
    assert "500 GW (unverified figure)" in cleaned
    assert counters["hedge_markers_inserted"] == 1.0


def test_does_not_double_tag_a_figure_the_compiler_already_hedged():
    evidence = [_model_evidence(
        "India's National Solar Mission targets 500 GW of non-fossil "
        "capacity by 2030.")]
    report = "India aims for roughly 500 GW of renewable capacity by 2030."
    cleaned, counters = enforce_hedging(report, evidence)
    assert cleaned == report
    assert counters == {}


def test_does_not_tag_a_figure_that_never_survived_into_the_report():
    evidence = [_model_evidence(
        "India's National Solar Mission targets 500 GW of non-fossil "
        "capacity by 2030.")]
    report = "This report says nothing about solar capacity."
    cleaned, counters = enforce_hedging(report, evidence)
    assert cleaned == report
    assert counters == {}


def test_ignores_evidence_never_flagged_as_hedge_specific():
    evidence = [_model_evidence(
        "India has a large and youthful population.", hedge_specific=False)]
    report = "India has a large and youthful population."
    cleaned, counters = enforce_hedging(report, evidence)
    assert cleaned == report
    assert counters == {}


def test_tags_every_occurrence_of_a_repeated_unhedged_figure():
    evidence = [_model_evidence(
        "The scheme targets 60.5 million metric tonnes of oil "
        "equivalent by 2025-26.")]
    report = ("The PAT scheme aims to save 60.5 million metric tonnes "
              "of oil equivalent by 2025-26. Elsewhere the report "
              "repeats: 60.5 million metric tonnes of oil equivalent "
              "by 2025-26 is the stated target.")
    cleaned, counters = enforce_hedging(report, evidence)
    assert cleaned.count("(unverified figure)") == 2
    assert counters["hedge_markers_inserted"] == 2.0


def test_never_touches_corpus_or_mcp_sourced_evidence():
    """hedge_specific is only ever set by tools/model_knowledge.py
    (source="model") -- but this function trusts the flag on the
    Evidence object, not the source string, so this is really testing
    that a corpus item with hedge_specific left at its default (False)
    is correctly left alone."""
    evidence = [Evidence(task_key="t1", goal_id="g1", source="corpus",
                         content="India's GDP grew 6.7% in 2022.",
                         score=0.9, hedge_specific=False)]
    report = "India's GDP grew 6.7% in 2022, per the retrieved corpus data."
    cleaned, counters = enforce_hedging(report, evidence)
    assert cleaned == report
    assert counters == {}


def test_two_different_unhedged_figures_each_get_tagged_once():
    evidence = [
        _model_evidence("India targets 500 GW of capacity by 2030."),
        _model_evidence("The US cut emissions 14% between 2010 and 2020."),
    ]
    report = ("India targets 500 GW of capacity by 2030. The US cut "
              "emissions 14% between 2010 and 2020.")
    cleaned, counters = enforce_hedging(report, evidence)
    assert cleaned.count("(unverified figure)") == 2
    assert counters["hedge_markers_inserted"] == 2.0


def test_a_hedge_word_on_one_figure_does_not_suppress_tagging_the_other():
    """Mixed case: one flagged figure the compiler already hedged
    ("about 14%"), one it didn't (bare "500 GW"). Only the unhedged one
    should be tagged -- an already-honest hedge must not be touched."""
    evidence = [
        _model_evidence("India targets 500 GW of capacity by 2030."),
        _model_evidence("The US cut emissions by about 14% between "
                        "2010 and 2020."),
    ]
    report = ("India targets 500 GW of capacity by 2030. The US cut "
              "emissions by about 14% between 2010 and 2020.")
    cleaned, counters = enforce_hedging(report, evidence)
    assert "500 GW (unverified figure)" in cleaned
    assert "14% (unverified figure)" not in cleaned
    assert counters["hedge_markers_inserted"] == 1.0


def test_a_flagged_span_inside_a_larger_number_is_not_tagged():
    """Code-review finding. The flagged span "5.2%" matched INSIDE
    "15.2%" via plain substring search, putting an (unverified figure)
    marker on a different, never-flagged number. A marker on a figure the
    detector never flagged is worse than a missing marker: it asserts
    something false about a number that may be perfectly well-sourced.
    The genuine occurrence must still be tagged."""
    evidence = [_model_evidence("Growth was 5.2% in 2021.")]
    report = "Growth was 15.2% in 2021, not 5.2%."
    cleaned, counters = enforce_hedging(report, evidence)
    assert "15.2% (unverified figure)" not in cleaned
    assert "not 5.2% (unverified figure)" in cleaned
    assert counters["hedge_markers_inserted"] == 1.0
