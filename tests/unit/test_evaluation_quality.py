"""
tests/unit/test_evaluation_quality.py — evaluation/quality.py's
score_answer (P2-11).

Covers: reading the score from whatever judge is passed in, clamping
out-of-range scores, fail-open behavior when the judge itself errors,
and the on_score_failed callback firing only on that error path, never
on a genuine (even low) score. Does NOT cover the ROUTER-level behavior
this enables (always judging with the NEXT provider in the chain, never
the one being judged, and the llm_quality_calls_failed counter) — see
test_llm_router.py for that.
"""

from research_agent.evaluation.quality import score_answer


class _FixedScoreJudge:
    """A ChatClient stub that always reports a fixed quality score,
    regardless of what answer it's asked to judge — enough to prove
    score_answer reads the score from WHATEVER `judge` it's given, not
    from some other, hidden "self" the answer came from (there is no
    such hidden self any more — see evaluation/quality.py's P2-11 note)."""

    def __init__(self, score):
        self._score = score

    def complete_json(self, messages, temperature=0.0):
        return {"score": self._score}


def test_score_answer_uses_the_judge_passed_in():
    judge = _FixedScoreJudge(0.3)
    score = score_answer(judge, [{"role": "user", "content": "q"}], "some answer")
    assert score == 0.3


def test_score_answer_clamps_out_of_range_scores():
    assert score_answer(_FixedScoreJudge(1.7), [], "x") == 1.0
    assert score_answer(_FixedScoreJudge(-4.0), [], "x") == 0.0


def test_score_answer_fails_open_when_judge_errors():
    class _BrokenJudge:
        def complete_json(self, messages, temperature=0.0):
            raise RuntimeError("judge is down")

    # Fail-open by design (see module docstring): a broken judge must never
    # take down a working answer path.
    assert score_answer(_BrokenJudge(), [], "x") == 1.0


def test_score_answer_invokes_on_score_failed_only_when_judge_errors():
    """P2-11 follow-up: the callback exists so a caller can count
    "couldn't be scored" separately from "scored low" — confirm it fires
    on the error path and stays silent on a genuine (even low) score."""
    class _BrokenJudge:
        def complete_json(self, messages, temperature=0.0):
            raise RuntimeError("judge is down")

    calls = []
    score = score_answer(_BrokenJudge(), [], "x", on_score_failed=lambda: calls.append(1))
    assert score == 1.0
    assert calls == [1]


def test_score_answer_does_not_invoke_on_score_failed_on_a_genuine_low_score():
    calls = []
    score = score_answer(_FixedScoreJudge(0.1), [], "x", on_score_failed=lambda: calls.append(1))
    assert score == 0.1
    assert calls == []
