"""
tests/unit/test_websearch_scoring.py — websearch/scoring.py::rank_to_score.

Pure arithmetic, no fixtures, no I/O. The assertions that matter most are
not the arithmetic ones but the two INVARIANT ones at the bottom, which
lock the band against the two settings it has to sit between:

    min_evidence_score (0.5) < web_search_min_score .. web_search_max_score
                                                     < a fused corpus hit (~1.0)

Break the lower invariant and the whole web tier is inert -- it can never
mark a goal covered, the exact failure MIN_EVIDENCE_SCORE=0.0 was.
Break the upper one and a web snippet outranks a document both retrieval
legs agreed on, inverting D-38's ordering guarantee.
"""

import pytest

from research_agent.websearch import rank_to_score

FLOOR = 0.60
CEILING = 0.75


def test_first_result_gets_the_ceiling():
    assert rank_to_score(1, 5, FLOOR, CEILING) == pytest.approx(CEILING)


def test_last_result_gets_the_floor():
    assert rank_to_score(5, 5, FLOOR, CEILING) == pytest.approx(FLOOR)


def test_scores_decrease_monotonically_with_rank():
    scores = [rank_to_score(r, 6, FLOOR, CEILING) for r in range(1, 7)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 6, "every rank must be distinguishable"


def test_interpolation_is_linear():
    """Midpoint of a 5-result set sits at the midpoint of the band."""
    assert rank_to_score(3, 5, 0.6, 0.8) == pytest.approx(0.7)


def test_a_single_result_gets_the_ceiling_not_the_midpoint():
    """One result carries no ordering information at all -- it IS the
    engine's best answer. Averaging the band would penalize a query that
    happened to have exactly one good hit."""
    assert rank_to_score(1, 1, FLOOR, CEILING) == pytest.approx(CEILING)


def test_zero_total_is_treated_as_a_single_result():
    """Defensive: total<=1 takes the same branch. A caller that computed a
    total of 0 while still having an item to score is confused, but the
    scorer must not raise on it."""
    assert rank_to_score(1, 0, FLOOR, CEILING) == pytest.approx(CEILING)


@pytest.mark.parametrize("rank", [0, -3, 99])
def test_out_of_range_ranks_are_clamped_not_rejected(rank):
    """A provider that miscounts is a data problem; refusing to score an
    otherwise usable result over an off-by-one is a worse outcome."""
    score = rank_to_score(rank, 5, FLOOR, CEILING)
    assert FLOOR <= score <= CEILING


def test_an_inverted_band_is_normalized_rather_than_running_backwards():
    """Passing floor > ceiling by mistake must not silently produce an
    ASCENDING scale where rank 1 is worst. config.py warns about the
    misconfiguration separately; this function still has to behave."""
    good = rank_to_score(1, 4, CEILING, FLOOR)
    bad = rank_to_score(4, 4, CEILING, FLOOR)
    assert good > bad
    assert good == pytest.approx(CEILING) and bad == pytest.approx(FLOOR)


# ---------------------------------------------------------------------------
# The two band invariants
# ---------------------------------------------------------------------------


def test_every_score_clears_the_default_coverage_gate():
    """D-17's predicate is a STRICT `>` (agents/gathering.py::
    progress_checker_node). Even the worst-ranked web result must clear it,
    or a retrieved web hit cannot cover a goal and the tier is inert."""
    min_evidence_score = 0.5
    for total in (1, 3, 5, 10):
        for rank in range(1, total + 1):
            assert rank_to_score(rank, total, FLOOR, CEILING) > min_evidence_score


def test_no_score_reaches_a_fused_corpus_hit_or_beats_it():
    """tools/corpus_search.py's RRF_SQUASH puts a document both legs ranked
    first near 1.0. D-38's invariant is that a real document always wins;
    the band's ceiling must stay clearly below that."""
    for total in (1, 5, 10):
        for rank in range(1, total + 1):
            assert rank_to_score(rank, total, FLOOR, CEILING) <= CEILING < 0.95


def test_the_band_sits_above_model_knowledge_score():
    """model_knowledge_score defaults to 0.60. A live retrieved snippet is
    better provenance than the model's own recollection, so the web band's
    ceiling must exceed it -- otherwise the compiler sees recollection
    ranked at least as highly as a real, current source."""
    assert CEILING > 0.60
