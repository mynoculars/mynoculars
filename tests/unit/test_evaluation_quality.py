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


# ---------------------------------------------------------------------------
# D-106 -- the judge's reason, and what counts as a real judgement
# ---------------------------------------------------------------------------


class _Judge:
    """A judge returning whatever JSON the test hands it."""

    name = "judge"

    def __init__(self, payload=None, error=False):
        self._payload = payload
        self._error = error

    def complete_json(self, messages, temperature=0.0):
        if self._error:
            raise RuntimeError("judge down")
        return self._payload


def test_the_reason_reaches_the_caller_alongside_the_score():
    """The whole point of D-106: a score you cannot interrogate is a
    number you can only tune blindly."""
    seen = []
    score = score_answer(_Judge({"score": 0.4, "reason": "no evidence cited"}),
                         [{"role": "user", "content": "q"}], "an answer",
                         on_scored=lambda s, why: seen.append((s, why)))

    assert score == 0.4
    assert seen == [(0.4, "no evidence cited")]


def test_a_judge_that_omits_the_reason_still_scores_normally():
    """`score` stays the only required key -- StubClient's canned
    {"score": 0.9} and any judge that ignores the new field must parse
    exactly as before."""
    seen = []
    score = score_answer(_Judge({"score": 0.9}),
                         [{"role": "user", "content": "q"}], "an answer",
                         on_scored=lambda s, why: seen.append((s, why)))

    assert score == 0.9
    assert seen == [(0.9, "")]


def test_the_callback_receives_the_CLAMPED_score_not_the_raw_one():
    """A distribution built from unclamped values would have bands the
    router cannot represent."""
    seen = []
    score_answer(_Judge({"score": 1.7}), [{"role": "user", "content": "q"}],
                 "an answer", on_scored=lambda s, why: seen.append(s))

    assert seen == [1.0]


def test_a_fail_open_never_reports_a_judgement():
    """THE property this callback exists for. score_answer returns a
    fabricated 1.0 when scoring breaks; folding that into a score
    distribution would make a dead judge look like a generous one --
    exactly the confusion P2-11 added llm_quality_calls_failed to end."""
    scored, failed = [], []
    score = score_answer(_Judge(error=True), [{"role": "user", "content": "q"}],
                         "an answer",
                         on_score_failed=lambda: failed.append(1),
                         on_scored=lambda s, why: scored.append(s))

    assert score == 1.0          # fail-open contract unchanged
    assert failed == [1]
    assert scored == [], "a fabricated 1.0 is not a judgement"


def test_a_non_string_reason_does_not_cost_us_the_score():
    """A judge answering `"reason": 3` must not raise inside the try and
    be misreported as a scoring FAILURE -- that would turn a usable score
    into a fail-open 1.0 over a cosmetic field."""
    seen = []
    score = score_answer(_Judge({"score": 0.3, "reason": 3}),
                         [{"role": "user", "content": "q"}], "an answer",
                         on_scored=lambda s, why: seen.append((s, why)))

    assert score == 0.3
    assert seen == [(0.3, "3")]


def test_a_reason_is_truncated_before_it_reaches_a_log_line():
    """A judge that ignores "one short sentence" must not be able to turn
    one JSON log record into a page."""
    seen = []
    score_answer(_Judge({"score": 0.5, "reason": "x" * 5000}),
                 [{"role": "user", "content": "q"}], "an answer",
                 on_scored=lambda s, why: seen.append(why))

    assert len(seen[0]) == 240


def test_omitting_the_callback_entirely_changes_nothing():
    """Every existing caller passes neither callback."""
    assert score_answer(_Judge({"score": 0.7}),
                        [{"role": "user", "content": "q"}], "a") == 0.7


# ---------------------------------------------------------------------------
# D-110 -- a failing judge says WHICH failure
# ---------------------------------------------------------------------------


def test_a_failing_judge_records_the_http_status(caplog):
    """Five runs of `quality.score_failed reason: HTTPStatusError` never
    once said whether the judge was misconfigured, unauthorised or out of
    quota."""
    import logging

    class _Resp:
        status_code = 404

    class _Boom:
        name = "gemini"

        def complete_json(self, messages, temperature=0.0):
            exc = RuntimeError("nope")
            exc.response = _Resp()
            raise exc

    with caplog.at_level(logging.INFO):
        assert score_answer(_Boom(), [{"role": "user", "content": "q"}], "a") == 1.0

    rec = [r for r in caplog.records if "quality.score_failed" in r.message]
    assert rec
    assert rec[0].event_fields["status"] == 404
    assert rec[0].event_fields["judge"] == "gemini"
    assert rec[0].event_fields["reason"] == "RuntimeError"


def test_a_judge_failure_with_no_response_object_still_logs(caplog):
    """A timeout or a transport error carries no status. It must degrade
    to None rather than raising inside the handler for a failure."""
    import logging

    class _Boom:
        name = "mistral"

        def complete_json(self, messages, temperature=0.0):
            raise TimeoutError("slow")

    with caplog.at_level(logging.INFO):
        assert score_answer(_Boom(), [{"role": "user", "content": "q"}], "a") == 1.0

    rec = [r for r in caplog.records if "quality.score_failed" in r.message]
    assert rec and rec[0].event_fields["status"] is None
