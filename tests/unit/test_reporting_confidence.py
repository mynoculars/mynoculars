"""
tests/unit/test_reporting_confidence.py — reporting/confidence.py (D-145).

Covers the cap model (why a weighted mean was rejected), each individual
cap, the graded contributions, the partial-telemetry contract, and the
calibration claim: sample_data/golden_queries.jsonl's in-corpus cases must
land HIGH/MODERATE and its off-corpus cases must not.
"""

import pytest

from research_agent.reporting.confidence import (BANDS, format_line,
                                                 score_report)


def _healthy(**overrides):
    """A clean in-corpus run: every signal good."""
    base = {
        "evidence_items": 30, "evidence_cited": 4, "citations_attached": 0,
        "grounding_ratio": 1.0, "recall": 1.0, "corpus_recall": 1.0,
        "grounded_score": 1.0, "retrieval_floor_drop_ratio": 0.1,
        "llm_quality_score_mean": 0.9, "critique_passed": True,
        "cited_figures_unsupported": 0, "goals_without_evidence": [],
        "escalations": [], "web_sources_listed": 3,
    }
    base.update(overrides)
    return base


# The exact telemetry p205.280-check emitted, transcribed from
# tmp/console-output.txt. This dict is the whole reason this module exists.
P205_280 = {
    "evidence_items": 100, "evidence_cited": 0, "citations_attached": 0,
    "grounding_ratio": 1.0, "recall": 1.0, "corpus_recall": 0.0,
    "grounded_score": 0.0, "retrieval_floor_drop_ratio": 1.0,
    "llm_quality_score_mean": 0.067, "critique_passed": False,
    "cited_figures_unsupported": 0, "goals_without_evidence": [],
    "web_sources_listed": 0, "web_sourced_items": 58,
    "escalations": [{"trigger": "E4", "action": "approve"}],
}


# ---------------------------------------------------------------------------
# The run that motivated this
# ---------------------------------------------------------------------------


def test_p205_280_is_unreliable():
    """Every signal in that run was bad and the report shipped anyway,
    because six of them printed as separate raw numbers and a human
    approved the escalation."""
    result = score_report(P205_280)

    assert result["band"] == "UNRELIABLE"
    assert result["score"] <= 15


def test_a_weighted_mean_would_have_passed_it_which_is_why_caps_exist():
    """recall and grounding_ratio were BOTH 1.0 in that run. Averaged
    against the rest they carry it to roughly the middle of the range --
    which is exactly the arithmetic this module refuses to do."""
    naive = (P205_280["grounding_ratio"] + P205_280["recall"]
             + P205_280["corpus_recall"] + P205_280["grounded_score"]) / 4 * 100

    assert naive >= 45, "the naive average really is that generous"
    assert score_report(P205_280)["score"] < 25


def test_the_reasons_lead_with_the_worst_cap():
    result = score_report(P205_280)

    assert "cites no evidence" in result["reasons"][0]


# ---------------------------------------------------------------------------
# Each cap, individually
# ---------------------------------------------------------------------------


def test_zero_citations_against_real_evidence_caps_at_unreliable():
    assert score_report(_healthy(evidence_cited=0))["band"] == "UNRELIABLE"


def test_no_evidence_at_all_is_not_a_citation_failure():
    """A run that retrieved nothing has nothing to cite. That is a
    retrieval problem, and the no-citations cap must not double-count it."""
    result = score_report(_healthy(evidence_items=0, evidence_cited=0))

    assert "cites no evidence" not in " ".join(result["reasons"])


def test_a_failed_critique_caps_below_moderate():
    assert score_report(_healthy(critique_passed=False))["score"] <= 45


def test_floor_starvation_caps_the_score():
    result = score_report(_healthy(retrieval_floor_drop_ratio=1.0))

    assert result["score"] <= 45
    # D-152 split the WORDING of this cap in two -- an off-corpus query
    # and a genuinely over-tight floor produce the same ratio and need
    # different remedies. The cap itself is unchanged, which is what this
    # test is about; the two wordings have their own tests below.
    assert any("floor dropped at least 80%" in r or
               "no material on this subject" in r
               for r in result["reasons"]), result["reasons"]


def test_unsupported_cited_figures_cap_the_score():
    result = score_report(_healthy(cited_figures_unsupported=3))

    assert result["score"] <= 40
    assert any("3 cited figure(s)" in r for r in result["reasons"])


def test_synthesised_attribution_cannot_reach_high():
    """D-144's pass only runs when the model cited nothing, so a nonzero
    count means every marker is this codebase's term-overlap judgement
    standing in for the writer's own."""
    result = score_report(_healthy(citations_attached=5))

    assert result["band"] != "HIGH"
    assert result["score"] <= 60


def test_an_uncited_sources_block_caps_the_score():
    result = score_report(_healthy(web_sources_listed_uncited=12))

    assert result["score"] <= 45


def test_a_planning_error_caps_near_zero():
    result = score_report(_healthy(planning_error="Goal composition produced 0"))

    assert result["band"] == "UNRELIABLE"


def test_an_aborted_run_scores_zero():
    assert score_report(_healthy(abort_reason="operator aborted"))["score"] == 0


def test_the_lowest_cap_wins_when_several_apply():
    result = score_report(_healthy(evidence_cited=0, critique_passed=False,
                                   retrieval_floor_drop_ratio=1.0))

    assert result["score"] <= 15
    assert len(result["caps"]) >= 3


# ---------------------------------------------------------------------------
# The graded half
# ---------------------------------------------------------------------------


def test_a_clean_in_corpus_run_scores_high():
    assert score_report(_healthy())["band"] == "HIGH"


def test_corpus_recall_moves_the_score():
    strong = score_report(_healthy())["score"]
    weak = score_report(_healthy(corpus_recall=0.0))["score"]

    assert weak < strong


def test_an_unasked_judge_is_not_a_bad_judge():
    """The common case with a single provider, or a chain that never needed
    a second opinion. Absence of a judgement must not read as a failed one."""
    unasked = score_report(_healthy(llm_quality_score_mean=None))
    rejected = score_report(_healthy(llm_quality_score_mean=0.05))

    assert unasked["score"] > rejected["score"]


def test_a_human_approval_is_annotated_without_changing_the_number():
    plain = score_report(_healthy())
    approved = score_report(_healthy(
        escalations=[{"trigger": "E4", "action": "approve"}]))

    assert approved["score"] == plain["score"]
    assert any("human approval" in r for r in approved["reasons"])


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_partial_telemetry_never_raises():
    """An interrupted or degraded run reaches this with a partial dict, and
    staying readable exactly then is the point."""
    for partial in ({}, {"evidence_items": None}, {"corpus_recall": "n/a"},
                    {"escalations": None}, {"goals_without_evidence": None}):
        result = score_report(partial)
        assert result["band"] in {name for _, name in BANDS}


def test_the_score_is_always_a_percentage():
    for telemetry in (P205_280, _healthy(), {}, _healthy(corpus_recall=99.0)):
        assert 0 <= score_report(telemetry)["score"] <= 100


def test_it_makes_no_llm_call_and_reads_nothing_it_was_not_given():
    """D-12: aggregate, never invent. A pure function over one dict."""
    import inspect

    from research_agent.reporting import confidence

    source = inspect.getsource(confidence.score_report)
    assert "router" not in source and "complete(" not in source


def test_format_line_truncates_and_says_how_many_it_hid():
    line = format_line(score_report(P205_280), max_reasons=1)

    assert line.startswith("UNRELIABLE (")
    assert "more)" in line


def test_format_line_on_a_clean_run_is_just_the_verdict():
    assert format_line(score_report(_healthy())) == "HIGH (98%)"


# ---------------------------------------------------------------------------
# Calibration against the golden set (D-136)
#
# The bands are not fitted to a labelled corpus -- none exists -- so the
# claim they make has to be checkable somewhere. These are the telemetry
# shapes sample_data/golden_queries.jsonl's own `expect` blocks describe.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case,telemetry,allowed", [
    ("in-corpus-comparison",
     _healthy(corpus_recall=0.75, recall=0.9), {"HIGH", "MODERATE"}),
    ("in-corpus-narrow-fact",
     _healthy(corpus_recall=0.5, recall=0.6), {"HIGH", "MODERATE"}),
    ("off-corpus-must-say-so",
     _healthy(corpus_recall=0.0, grounded_score=0.0), {"MODERATE", "LOW"}),
    ("off-corpus-numeric-bait",
     _healthy(corpus_recall=0.0, cited_figures_unsupported=2),
     {"LOW", "UNRELIABLE"}),
])
def test_the_golden_cases_land_in_the_bands_they_describe(case, telemetry,
                                                          allowed):
    assert score_report(telemetry)["band"] in allowed, case


def test_an_in_corpus_case_always_outscores_the_off_corpus_one():
    """The ordering claim, which matters more than either absolute number."""
    in_corpus = score_report(_healthy(corpus_recall=0.75))["score"]
    off_corpus = score_report(_healthy(corpus_recall=0.0))["score"]

    assert in_corpus > off_corpus


# ---------------------------------------------------------------------------
# D-152 -- the same ratio, two different meanings
#
# Live (p205.282-check) a China-vs-India question against a Redis corpus
# reported "retrieval was starved", which reads as a misconfiguration to
# fix and was not one: the corpus genuinely has no material on the PLA.
# ---------------------------------------------------------------------------


def test_an_off_corpus_query_is_not_reported_as_a_starved_floor():
    result = score_report(_healthy(retrieval_floor_drop_ratio=0.986,
                                   corpus_recall=0.0,
                                   tier_answers={"web": 12}))

    reasons = " ".join(result["reasons"])
    assert "no material on this subject" in reasons
    assert "starved" not in reasons


def test_a_genuinely_starved_floor_still_says_so():
    """The corpus DID answer, and the floor still dropped 80%+ -- that is
    D-42's failure mode and a real defect to fix."""
    result = score_report(_healthy(retrieval_floor_drop_ratio=0.9,
                                   tier_answers={"corpus": 4, "web": 2}))

    reasons = " ".join(result["reasons"])
    assert "real evidence may be being discarded" in reasons
    assert "no material on this subject" not in reasons


def test_the_mcp_tier_counts_as_the_corpus_answering():
    """Tiers 1-3 of the D-38 ladder all resolve to the operator's own
    ingested documents; mcp reaching them is not "the corpus has nothing"."""
    result = score_report(_healthy(retrieval_floor_drop_ratio=0.9,
                                   tier_answers={"mcp": 6}))

    assert "no material on this subject" not in " ".join(result["reasons"])


def test_the_cap_is_the_same_either_way():
    """Only the remedy differs, not the verdict: an answer with no corpus
    behind it is LOW whichever the cause."""
    off = score_report(_healthy(retrieval_floor_drop_ratio=0.986,
                                tier_answers={"web": 12}))
    starved = score_report(_healthy(retrieval_floor_drop_ratio=0.986,
                                    tier_answers={"corpus": 1}))

    assert off["score"] == starved["score"]


def test_a_run_with_no_tier_data_keeps_the_original_wording():
    """Rows written before D-87 carry no tier_answers, and a cross-run
    report must not crash or invent a diagnosis for them."""
    result = score_report(_healthy(retrieval_floor_drop_ratio=0.9,
                                   tier_answers={}))

    assert "real evidence may be being discarded" in " ".join(result["reasons"])

