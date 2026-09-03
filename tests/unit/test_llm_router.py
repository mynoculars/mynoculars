"""
tests/unit/test_llm_router.py — llm/router.py's FallbackRouter.

Covers: the N-provider fallback chain itself (error -> next provider,
first good answer wins), judge-model quality gating (P2-11: the judge is
always the NEXT provider in the chain, never the one being judged), and
the router-boundary telemetry counters (llm_provider_calls,
llm_fallback_hops, llm_quality_calls, llm_quality_calls_failed — P2-07 /
P2-11 follow-up). Does NOT cover the real ChatClient/StubClient
implementations themselves (see test_llm_client.py) — every provider
here is a minimal local fake satisfying only what FallbackRouter calls
on it (.name, .complete, .complete_json, and for the quality-gate tests,
nothing else).
"""

import json
import logging as _logging

import pytest

from research_agent.llm.client import TruncatedGenerationError
from research_agent.llm.router import FallbackRouter, ProviderChainExhausted

# ---------------------------------------------------------------------------
# Basic fallback: error -> next provider, first success wins
# ---------------------------------------------------------------------------


class _Boom:
    name = "boom"

    def complete(self, messages, temperature=0.2):
        raise RuntimeError("primary down")

    def complete_json(self, messages, temperature=0.0):
        raise RuntimeError("primary down")


class _Fine:
    name = "fine"

    def complete(self, messages, temperature=0.2):
        return '{"ok": true}'

    def complete_json(self, messages, temperature=0.0):
        return {"ok": True}


def test_router_falls_back_on_primary_error():
    router = FallbackRouter([_Boom(), _Fine()], quality_threshold=0.6)
    assert router.complete_json([{"role": "user", "content": "x"}]) == {"ok": True}


def test_router_raises_when_no_fallback():
    router = FallbackRouter([_Boom()], quality_threshold=0.6)
    with pytest.raises(RuntimeError):
        router.complete_json([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# Fallback CHAIN (primary -> Mistral -> Gemini), N-provider router, plus
# judge-model quality gating (P2-11)
# ---------------------------------------------------------------------------


class _Named:
    """Stub provider: errors, low-quality, or good answer. `behavior` in
    {"error","low","answer"}. On a TASK=quality scoring call it reports the
    fixed score baked into ITS OWN `behavior` — regardless of whose answer
    it's actually asked to judge (P2-11: the router always passes the NEXT
    provider in the chain as judge, never the answering provider itself, so
    tests wire up whichever _Named instance should play judge with
    behavior="low"/"answer" for that purpose)."""

    def __init__(self, name, behavior):
        self.name = name
        self.behavior = behavior

    def complete(self, messages, temperature=0.2):
        if messages and "TASK=quality" in messages[-1]["content"]:
            return json.dumps({"score": 0.2 if self.behavior == "low" else 0.9})
        if self.behavior == "error":
            raise RuntimeError(f"{self.name} down")
        return f"answer from {self.name}"

    def complete_json(self, messages, temperature=0.0):
        return json.loads(self.complete(messages, temperature))


def test_chain_steps_primary_to_mistral_to_gemini_on_error():
    chain = FallbackRouter(
        [_Named("primary", "error"), _Named("mistral", "error"),
         _Named("gemini", "answer")], quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "answer from gemini"


def test_chain_stops_at_first_good_provider():
    chain = FallbackRouter(
        [_Named("primary", "answer"), _Named("mistral", "answer"),
         _Named("gemini", "answer")], quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "answer from primary"


def test_chain_steps_on_low_quality_then_serves_next():
    # P2-11: the JUDGE is now the next provider in the chain (mistral), never
    # the provider being judged (primary) — so it's mistral's `behavior`
    # that must be "low" to reject primary's answer here, not primary's own.
    # primary's own `behavior` ("answer") only governs the text IT returns,
    # never its own quality score any more — confirming self-scoring is
    # genuinely gone, not just relabeled.
    chain = FallbackRouter(
        [_Named("primary", "answer"), _Named("mistral", "low")],
        quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "answer from mistral"


def test_chain_ignores_providers_own_self_report_as_judge():
    # P2-11 regression guard: if quality scoring were still self-scoring,
    # primary reporting itself as "low" would cause a fallback hop even
    # though the judge (mistral) would score it fine. Confirm primary's
    # OWN low self-report is irrelevant now — only the judge's opinion
    # (mistral, "answer" -> scores 0.9) decides, so primary's answer is kept.
    chain = FallbackRouter(
        [_Named("primary", "low"), _Named("mistral", "answer")],
        quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "answer from primary"


def test_chain_json_cascades_on_error():
    class _Json:
        name = "j"
        def complete(self, m, temperature=0.2): return json.dumps({"ok": True})
        def complete_json(self, m, temperature=0.0): return {"ok": True}

    chain = FallbackRouter([_Named("primary", "error"), _Json()],
                           quality_threshold=0.6)
    assert chain.complete_json([{"role": "user", "content": "x"}]) == {"ok": True}


# ---------------------------------------------------------------------------
# Router-boundary telemetry (P2-07 / P2-11 follow-up)
# ---------------------------------------------------------------------------


def test_drain_counters_counts_provider_attempts_and_resets():
    router = FallbackRouter([_Named("primary", "error"), _Named("mistral", "answer")],
                           quality_threshold=0.6)
    router.complete([{"role": "user", "content": "x"}])
    drained = router.drain_counters()
    assert drained["llm_provider_calls"] == 2   # primary attempt + mistral attempt
    assert drained["llm_fallback_hops"] == 1    # exactly one hop, primary -> mistral

    # Draining resets — a second call with no further activity yields nothing.
    assert router.drain_counters() == {}


def test_drain_counters_counts_quality_scoring_calls():
    # Two providers means the first one's answer is quality-scored before
    # being accepted (there's a fallback to check against); the last
    # provider in a chain is never scored (see router.py's has_next logic).
    router = FallbackRouter([_Named("primary", "answer"), _Named("mistral", "answer")],
                           quality_threshold=0.6)
    router.complete([{"role": "user", "content": "x"}])
    drained = router.drain_counters()
    assert drained["llm_quality_calls"] == 1
    assert drained.get("llm_fallback_hops", 0) == 0  # quality passed, no hop needed


class _SimpleAnswerer:
    """Minimal ChatClient: always answers the same fixed text, never
    errors. Used as the ANSWERING provider (position 0) in the
    llm_quality_calls_failed tests below — its own quality is never
    self-scored (P2-11), so its complete_json is never even exercised
    as a judge here."""

    name = "primary"

    def complete(self, messages, temperature=0.2):
        return "primary answer"

    def complete_json(self, messages, temperature=0.0):
        return {}


class _AlwaysErroringJudge:
    """A ChatClient whose complete_json (the method score_answer calls)
    always raises — simulating exactly what the real Gemini 429 did:
    the JUDGE, not the answering provider, is the one that's down."""

    name = "judge"

    def complete(self, messages, temperature=0.2):
        return "judge answer"  # only used if this ever became the answerer

    def complete_json(self, messages, temperature=0.0):
        raise RuntimeError("judge is down")


class _GoodJudge:
    """A ChatClient whose complete_json always scores well."""

    name = "judge"

    def complete(self, messages, temperature=0.2):
        return "judge answer"

    def complete_json(self, messages, temperature=0.0):
        return {"score": 0.9}


def test_router_bumps_llm_quality_calls_failed_when_judge_errors():
    """End-to-end version of the exact live-run shape this follow-up was
    written for: the answering provider succeeds, the NEXT provider in
    the chain (the judge) is unreachable. The answer must still be kept
    (fail-open), and the failure must now be visible in telemetry as
    llm_quality_calls_failed — not just as a "quality.score_failed" log
    line with no counter behind it."""
    router = FallbackRouter([_SimpleAnswerer(), _AlwaysErroringJudge()],
                           quality_threshold=0.6)
    answer = router.complete([{"role": "user", "content": "x"}])
    assert answer == "primary answer"

    drained = router.drain_counters()
    assert drained["llm_quality_calls"] == 1        # the attempt was made
    assert drained["llm_quality_calls_failed"] == 1  # and it couldn't be scored
    assert drained.get("llm_fallback_hops", 0) == 0  # fail-open kept the answer


def test_router_never_bumps_llm_quality_calls_failed_on_a_genuine_score():
    """Regression guard: a working judge that scores normally must never
    touch the new counter, whatever the score — it's reserved for
    "couldn't be scored", not "scored something"."""
    router = FallbackRouter([_SimpleAnswerer(), _GoodJudge()], quality_threshold=0.6)
    router.complete([{"role": "user", "content": "x"}])

    drained = router.drain_counters()
    assert drained["llm_quality_calls"] == 1
    assert drained.get("llm_quality_calls_failed", 0) == 0


# ---------------------------------------------------------------------------
# Guardrail G6 (P205 Phase 2): max_tokens threaded through from_settings
# ---------------------------------------------------------------------------


def test_from_settings_passes_llm_max_tokens_to_every_provider():
    """Same cap on every provider in the chain, unlike the two DIFFERENT
    timeouts primary/fallback get -- see from_settings' own docstring
    for why max_tokens doesn't need that same split."""
    from research_agent.config import Settings

    settings = Settings(_env_file=None, llm_mode="live",
                        llm_max_tokens=777,
                        llm_mistral_api_key="fake-key")
    router = FallbackRouter.from_settings(settings)
    assert [p._max_tokens for p in router.providers] == [777, 777]


def test_from_settings_uses_the_configured_default_when_unset():
    from research_agent.config import Settings

    settings = Settings(_env_file=None, llm_mode="live")
    router = FallbackRouter.from_settings(settings)
    assert router.providers[0]._max_tokens == settings.llm_max_tokens



# ---------------------------------------------------------------------------
# FIX-3 — "best answer wins", not "last answer wins"
#
# Regression coverage for the defect diagnosed from runs p205.211 (bad) and
# p205.212 (good). Same code, same query; the ONLY difference was that the
# last provider in the chain errored in one run and succeeded in the other.
# The run where it succeeded shipped a 732-char fragment over a complete
# 10,103-char report, because a rejected answer was discarded permanently and
# the last provider was never judged. These tests pin both halves shut.
# ---------------------------------------------------------------------------


class _Fixed:
    """Provider returning a fixed answer, and a fixed score when asked to judge.

    Deliberately separate from _Named above: these tests need the answer text
    and the judging score to vary INDEPENDENTLY (the whole defect is about a
    provider that answers badly but is never scored), which _Named's single
    `behavior` field cannot express.
    """

    def __init__(self, name, answer=None, judge_score=0.9, error=False):
        self.name = name
        self._answer = answer if answer is not None else f"answer from {name}"
        self._judge_score = judge_score
        self._error = error
        self.judged = 0

    def complete(self, messages, temperature=0.2):
        if messages and "TASK=quality" in messages[-1]["content"]:
            self.judged += 1
            return json.dumps({"score": self._judge_score})
        if self._error:
            raise RuntimeError(f"{self.name} down")
        return self._answer

    def complete_json(self, messages, temperature=0.0):
        return json.loads(self.complete(messages, temperature))


def test_last_provider_answer_cannot_silently_replace_a_better_rejected_one():
    # The p205.211 shape exactly. `judge_score` is the score a provider HANDS
    # OUT when acting as judge, so: gemini scores mistral's report 0.45
    # (rejected, below 0.6), and mistral then scores gemini's fragment 0.1.
    # Under the old "last answer wins" policy gemini's fragment was never
    # scored at all and shipped regardless. Now the 0.45 answer beats the 0.1
    # one and the complete report survives.
    mistral = _Fixed("mistral", answer="THE GOOD REPORT", judge_score=0.1)
    gemini = _Fixed("gemini", answer="fragment", judge_score=0.45)
    chain = FallbackRouter([mistral, gemini], quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "THE GOOD REPORT"
    # And the last provider's answer WAS actually judged — the old code
    # never asked, which is precisely why the fragment could win.
    assert mistral.judged >= 1


def test_last_provider_answer_still_wins_when_it_genuinely_scores_higher():
    # The mirror case, so the fix is "keep the best", not "always keep the
    # earlier one". Here gemini rejects mistral at 0.45, and mistral scores
    # gemini's answer 0.9 — gemini's answer is genuinely better and wins.
    mistral = _Fixed("mistral", answer="thin draft", judge_score=0.9)
    gemini = _Fixed("gemini", answer="BETTER REPORT", judge_score=0.45)
    chain = FallbackRouter([mistral, gemini], quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "BETTER REPORT"


def test_last_provider_is_still_unjudged_when_nothing_was_rejected():
    # The common path must cost no extra scoring call. Only provider is last,
    # nothing was rejected before it, so its answer is accepted as-is.
    solo = _Fixed("solo", answer="only answer")
    chain = FallbackRouter([solo], quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "only answer"
    assert solo.judged == 0


def test_chain_exhausted_returns_the_best_scoring_answer_not_the_last():
    # Three providers, every one rejected, scores 0.5 / 0.1 / (last, judged
    # by the one before it). The best-scoring answer must come back.
    a = _Fixed("a", answer="A", judge_score=0.1)   # judges b -> 0.1
    b = _Fixed("b", answer="B", judge_score=0.5)   # judges a -> 0.5
    c = _Fixed("c", answer="C", judge_score=0.2)   # judges b -> 0.2
    chain = FallbackRouter([a, b, c], quality_threshold=0.6)
    # a scored 0.5 by b (rejected), b scored 0.2 by c (rejected),
    # c scored 0.5 by b (last provider, now judged, ties rather than beats).
    # Best is a at 0.5 -- the FIRST answer, which the old code could never
    # return once it had been rejected.
    assert chain.complete([{"role": "user", "content": "x"}]) == "A"


def test_all_providers_erroring_still_raises():
    chain = FallbackRouter([_Fixed("a", error=True), _Fixed("b", error=True)],
                           quality_threshold=0.6)
    with pytest.raises(RuntimeError):
        chain.complete([{"role": "user", "content": "x"}])


def test_error_on_last_provider_falls_back_to_the_rejected_answer():
    # This is what accidentally SAVED run p205.212. It must keep working —
    # but now as the designed path, not as luck.
    mistral = _Fixed("mistral", answer="THE GOOD REPORT", judge_score=0.9)
    gemini = _Fixed("gemini", judge_score=0.35, error=True)
    chain = FallbackRouter([mistral, gemini], quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "THE GOOD REPORT"


# ---------------------------------------------------------------------------
# D-86: run-level token accounting
# ---------------------------------------------------------------------------


class _UsageClient:
    """A provider that reports token usage the way OpenAICompatibleClient
    does -- via a drain_usage() that returns once and then clears."""

    def __init__(self, name, reply="answer", usage=(100, 20), raises=None):
        self.name = name
        self._reply = reply
        self._usage = usage
        self._raises = raises
        self.drained = 0

    def complete(self, messages, temperature=0.2):
        if self._raises:
            raise self._raises
        return self._reply

    def complete_json(self, messages, temperature=0.0):
        if self._raises:
            raise self._raises
        return {"ok": True}

    def drain_usage(self):
        self.drained += 1
        usage, self._usage = self._usage, (0, 0)
        return usage

    def set_trace_node(self, node):
        pass


def test_json_calls_accumulate_prompt_and_completion_tokens():
    router = FallbackRouter([_UsageClient("primary", usage=(300, 40))], 0.6)

    router.complete_json([{"role": "user", "content": "q"}])
    counters = router.drain_counters()

    assert counters["llm_prompt_tokens"] == 300.0
    assert counters["llm_completion_tokens"] == 40.0


def test_a_failed_provider_contributes_no_tokens_but_the_next_one_does():
    """Tokens follow real completed calls, not attempts. A provider that
    raised produced no usage to report; llm_provider_calls is the field
    that counts the attempt."""
    router = FallbackRouter(
        [_UsageClient("primary", raises=RuntimeError("boom")),
         _UsageClient("mistral", usage=(120, 15))], 0.6)

    router.complete_json([{"role": "user", "content": "q"}])
    counters = router.drain_counters()

    assert counters["llm_provider_calls"] == 2.0
    assert counters["llm_prompt_tokens"] == 120.0
    assert counters["llm_completion_tokens"] == 15.0


def test_a_provider_without_drain_usage_is_skipped_not_an_error():
    """drain_usage is an OPTIONAL, duck-typed capability -- StubClient and
    every hand-written test fake lack it and must keep working."""
    class _NoUsage:
        name = "plain"

        def complete_json(self, messages, temperature=0.0):
            return {"ok": True}

        def set_trace_node(self, node):
            pass

    router = FallbackRouter([_NoUsage()], 0.6)

    router.complete_json([{"role": "user", "content": "q"}])
    counters = router.drain_counters()

    assert "llm_prompt_tokens" not in counters
    assert counters["llm_provider_calls"] == 1.0


def test_usage_is_drained_once_per_call_so_it_cannot_be_counted_twice():
    """drain, not peek -- the same reasoning drain_counters itself gives.
    Two calls against a provider that reports usage only on the first must
    total that usage once, not twice."""
    provider = _UsageClient("primary", usage=(50, 5))
    router = FallbackRouter([provider], 0.6)

    router.complete_json([{"role": "user", "content": "q"}])
    router.complete_json([{"role": "user", "content": "q"}])
    counters = router.drain_counters()

    assert provider.drained == 2
    assert counters["llm_prompt_tokens"] == 50.0


# ---------------------------------------------------------------------------
# D-93: skipping a hop whose context window cannot take the prompt
# ---------------------------------------------------------------------------


class _CtxClient(_UsageClient):
    def __init__(self, name, context_tokens=0, **kw):
        super().__init__(name, **kw)
        self.context_tokens = context_tokens
        self.calls = 0

    def complete_json(self, messages, temperature=0.0):
        self.calls += 1
        return {"served_by": self.name}

    def complete(self, messages, temperature=0.2):
        self.calls += 1
        return f"answer from {self.name}"


_BIG = [{"role": "user", "content": "x" * 40000}]    # ~10k estimated tokens
_SMALL = [{"role": "user", "content": "x" * 400}]    # ~100 estimated tokens


def test_a_prompt_far_over_the_window_skips_that_provider():
    """The live shape: a 1536-token window and a 7,198-token prompt. The
    primary rejected it in 29ms every run -- deterministic, not flaky."""
    primary = _CtxClient("primary", context_tokens=1536)
    fallback = _CtxClient("mistral")
    router = FallbackRouter([primary, fallback], 0.6)

    result = router.complete_json(_BIG)
    counters = router.drain_counters()

    assert result == {"served_by": "mistral"}
    assert primary.calls == 0, "the doomed call must not be made at all"
    assert counters["llm_context_skips"] == 1.0
    assert counters["llm_provider_calls"] == 1.0, (
        "a hop never attempted must not be counted as an attempt")


def test_a_prompt_that_fits_still_goes_to_the_primary():
    primary = _CtxClient("primary", context_tokens=1536)
    router = FallbackRouter([primary, _CtxClient("mistral")], 0.6)

    assert router.complete_json(_SMALL) == {"served_by": "primary"}
    assert primary.calls == 1
    assert "llm_context_skips" not in router.drain_counters()


def test_an_unconfigured_provider_is_never_skipped():
    """context_tokens defaults to 0 everywhere. With no configuration the
    routing decision must be byte-identical to before D-93 existed."""
    primary = _CtxClient("primary")  # no window configured
    router = FallbackRouter([primary, _CtxClient("mistral")], 0.6)

    assert router.complete_json(_BIG) == {"served_by": "primary"}
    assert primary.calls == 1


def test_the_last_provider_is_never_skipped():
    """Skipping is an optimisation that only makes sense when there is
    somewhere to fall through TO. Skipping the sole provider would leave
    complete() with no candidate AND no exception, tripping its own
    `assert last_exc is not None` -- a crash instead of a run."""
    solo = _CtxClient("primary", context_tokens=1536)
    router = FallbackRouter([solo], 0.6)

    assert router.complete_json(_BIG) == {"served_by": "primary"}
    assert solo.calls == 1
    assert "llm_context_skips" not in router.drain_counters()


def test_a_prompt_near_the_boundary_is_still_attempted():
    """estimate_prompt_tokens is ~4 chars/token and says so. The 1.1x
    margin means a mis-estimate near the limit costs one recovered failed
    call, never a silently discarded working provider -- a false skip is
    invisible and permanent, a false attempt is one log line."""
    primary = _CtxClient("primary", context_tokens=1536)
    router = FallbackRouter([primary, _CtxClient("mistral")], 0.6)
    near = [{"role": "user", "content": "x" * (1536 * 4)}]  # ~1536 tokens

    router.complete_json(near)

    assert primary.calls == 1


def test_the_free_text_path_skips_the_same_way():
    primary = _CtxClient("primary", context_tokens=1536)
    router = FallbackRouter([primary, _CtxClient("mistral")], 0.6)

    assert router.complete(_BIG) == "answer from mistral"
    assert primary.calls == 0



# ---------------------------------------------------------------------------
# D-101 -- ProviderChainExhausted
#
# Diagnosed from run p205.254-check's fifth compile: primary
# HTTPStatusError, mistral ReadTimeout, gemini TruncatedGenerationError,
# nothing to ship. The bare `raise last_exc` handed cli.py only the LAST
# provider's exception, which says nothing about the other two.
# ---------------------------------------------------------------------------


class _Typed:
    """Raises a NAMED exception type, so a test can assert the chain
    summary reports each provider's OWN failure and not just the last."""

    def __init__(self, name, exc):
        self.name = name
        self._exc = exc

    def complete(self, messages, temperature=0.2):
        raise self._exc

    def complete_json(self, messages, temperature=0.0):
        raise self._exc


def _dead_chain():
    return FallbackRouter(
        [_Typed("primary", RuntimeError("400")),
         _Typed("mistral", TimeoutError("read timeout")),
         _Typed("gemini", TruncatedGenerationError("cut off"))], 0.6)


def test_exhaustion_names_every_provider_and_how_each_one_failed():
    chain = _dead_chain()
    chain.set_node("compiler")

    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete([{"role": "user", "content": "x"}])

    assert exc.value.attempts == [
        ("primary", "RuntimeError"),
        ("mistral", "TimeoutError"),
        ("gemini", "TruncatedGenerationError")]
    assert exc.value.node == "compiler"
    assert exc.value.mode == "text"


def test_exhaustion_keeps_the_last_real_failure_as_the_cause():
    """`raise ... from last_exc` -- cli.py prints __cause__ for the
    detail the chain summary cannot carry (which ceiling truncated it),
    and a traceback must still show the real failure underneath."""
    chain = _dead_chain()

    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete([{"role": "user", "content": "x"}])

    assert isinstance(exc.value.__cause__, TruncatedGenerationError)


def test_exhaustion_is_still_a_runtime_error():
    """The containment property this change depends on: every existing
    caller catches Exception broadly and the two pre-existing exhaustion
    tests assert RuntimeError. Subclassing keeps both true."""
    chain = _dead_chain()
    with pytest.raises(RuntimeError):
        chain.complete_json([{"role": "user", "content": "x"}])


def test_the_json_path_reports_its_own_mode():
    chain = _dead_chain()
    chain.set_node("critic")
    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete_json([{"role": "user", "content": "x"}])
    assert exc.value.mode == "json"
    assert exc.value.node == "critic"


def test_a_context_skipped_hop_is_reported_as_skipped_not_as_a_failure():
    """D-93 skips a hop it never attempted. Calling that a failure would
    be a lie; omitting it would make the chain look shorter than it is."""
    chain = FallbackRouter(
        [_CtxClient("primary", context_tokens=1536),
         _Typed("mistral", TimeoutError("read timeout")),
         _Typed("gemini", RuntimeError("500"))], 0.6)

    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete(_BIG)

    assert exc.value.attempts == [
        ("primary", "skipped_for_context"),
        ("mistral", "TimeoutError"),
        ("gemini", "RuntimeError")]


def test_a_chain_that_produced_any_answer_never_raises():
    """The boundary this exception must NOT cross. p205.254-check's
    FOURTH compile had two providers fail and one return a 0.1-scored
    report; _best() shipped it and the run continued. Only a chain with
    nothing at all to ship is exhausted."""
    chain = FallbackRouter(
        [_Typed("primary", RuntimeError("400")),
         _Fixed("mistral", answer="A BAD BUT REAL REPORT", judge_score=0.1),
         _Typed("gemini", TruncatedGenerationError("cut off"))], 0.6)

    assert chain.complete([{"role": "user", "content": "x"}]) == \
        "A BAD BUT REAL REPORT"


def test_the_node_name_is_absent_rather_than_guessed_when_never_set():
    """set_node is called by every real node and by almost no test
    router. An unset node reports None; the message simply omits the
    'at the ... node' clause rather than inventing one."""
    chain = _dead_chain()
    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete([{"role": "user", "content": "x"}])
    assert exc.value.node is None
    assert "at the" not in str(exc.value)



# ---------------------------------------------------------------------------
# D-106 -- the score distribution the threshold can finally be judged against
# ---------------------------------------------------------------------------


def _judged_once(judge_score):
    """Run one free-text call producing EXACTLY ONE judgement, and return
    the drained counters.

    The second provider judges normally but cannot answer (_Fixed checks
    for TASK=quality before its error flag). That matters: in a plain
    two-provider chain a REJECTED first answer falls through, and the
    second provider's answer is then judged in turn by the first -- two
    judgements, and a sum that is the total of both. Making the second
    provider unable to answer collapses it to one for every score, high
    or low, which is what lets these assertions be exact.

    It is also p205.254-check's own shape: one provider answered, the
    next died, and _best() shipped the scored-but-rejected answer.
    """
    chain = FallbackRouter(
        [_Fixed("primary", answer="A"),
         _Fixed("mistral", judge_score=judge_score, error=True)], 0.6)
    chain.set_node("compiler")
    chain.complete([{"role": "user", "content": "x"}])
    return chain.drain_counters()


def test_a_judgement_lands_in_exactly_one_band():
    only = _judged_once(0.1)
    assert only["llm_quality_scores_judged"] == 1
    assert [k for k in only if k.startswith("llm_quality_band_")] == \
        ["llm_quality_band_very_low"]
    assert _judged_once(0.1)["llm_quality_band_very_low"] == 1
    assert _judged_once(0.3)["llm_quality_band_low"] == 1
    assert _judged_once(0.5)["llm_quality_band_mid"] == 1
    assert _judged_once(0.7)["llm_quality_band_high"] == 1
    assert _judged_once(0.9)["llm_quality_band_very_high"] == 1


def test_a_perfect_score_is_not_lost_off_the_top_band():
    """The last band's bound is 1.01, not 1.0, precisely so a judge that
    answers 1.0 is counted rather than silently dropped."""
    counters = _judged_once(1.0)

    assert counters["llm_quality_band_very_high"] == 1
    assert counters["llm_quality_scores_judged"] == 1


def test_the_sum_and_count_support_a_mean():
    counters = _judged_once(0.4)

    assert counters["llm_quality_score_sum"] == 0.4
    assert counters["llm_quality_scores_judged"] == 1


def test_only_scores_below_the_threshold_count_as_rejections():
    assert _judged_once(0.5).get("llm_quality_rejections") == 1
    # 0.6 is the threshold itself, and the router accepts on >=.
    assert "llm_quality_rejections" not in _judged_once(0.6)


def test_an_accepted_score_is_still_recorded():
    """An accepted 0.61 says as much about where the threshold belongs as
    a rejected 0.59 does, and only one of the two was ever recorded."""
    counters = _judged_once(0.9)

    assert counters["llm_quality_scores_judged"] == 1
    assert counters["llm_quality_score_sum"] == 0.9


def test_a_judge_that_fails_open_contributes_nothing_to_the_distribution():
    """llm_quality_calls counts ATTEMPTS and would include this one;
    llm_quality_scores_judged must not, or the mean has a fabricated 1.0
    in it and a dead judge reads as a generous one."""
    chain = FallbackRouter(
        [_Fixed("primary", answer="A"),
         _Typed("mistral", RuntimeError("judge down"))], 0.6)

    chain.complete([{"role": "user", "content": "x"}])
    counters = chain.drain_counters()

    assert counters["llm_quality_calls"] == 1
    assert counters["llm_quality_calls_failed"] == 1
    assert "llm_quality_scores_judged" not in counters
    assert "llm_quality_score_sum" not in counters
    assert not any(k.startswith("llm_quality_band_") for k in counters)


def test_the_bands_do_not_move_with_the_threshold():
    """Fixed bands are what make 'is 0.6 in the right place' answerable.
    A band set that tracked the threshold could never answer it."""
    chain = FallbackRouter(
        [_Fixed("primary", answer="A"),
         _Fixed("mistral", answer="B", judge_score=0.5)],
        quality_threshold=0.3)          # 0.5 now PASSES
    chain.complete([{"role": "user", "content": "x"}])
    counters = chain.drain_counters()

    assert counters["llm_quality_band_mid"] == 1, "still the 0.4-0.6 band"
    assert "llm_quality_rejections" not in counters, "but no longer rejected"


def test_the_scored_event_carries_the_reason_and_the_verdict(caplog):
    chain = FallbackRouter(
        [_Fixed("primary", answer="A"),
         _Fixed("mistral", answer="B", judge_score=0.9)], 0.6)

    with caplog.at_level(_logging.INFO):
        chain.complete([{"role": "user", "content": "x"}])

    assert any("llm.quality_scored" in r.message for r in caplog.records)



# ---------------------------------------------------------------------------
# D-114 -- the cloud fallback slot is named, not hardwired
#
# Its base URL, key and model were always generic; only the NAME was
# hardwired, so pointing the three at another OpenAI-compatible provider
# produced a working chain whose every log line, counter, health-check row
# and D-110 error said "gemini" while calling something else.
# ---------------------------------------------------------------------------


def _live(**over):
    from research_agent.config import Settings
    base = {"_env_file": None, "llm_mode": "live",
            "llm_mistral_api_key": "k", "llm_fallback_api_key": "k"}
    base.update(over)
    return Settings(**base)


def test_the_slot_defaults_to_gemini_unchanged():
    """Every existing .env leaves LLM_FALLBACK_NAME unset. That must build
    exactly the chain it built before this setting existed."""
    chain = FallbackRouter.from_settings(_live())

    assert [p.name for p in chain.providers] == ["primary", "mistral", "gemini"]


def test_switching_the_name_renames_the_provider_everywhere():
    chain = FallbackRouter.from_settings(
        _live(llm_fallback_name="grok",
              llm_fallback_base_url="https://api.x.ai/v1",
              llm_fallback_model="grok-x"))
    slot = chain.providers[2]

    assert slot.name == "grok"
    assert "XAI GROK" in slot._label and "grok-x" in slot._label
    assert "api.x.ai" in str(slot._http.base_url)


def test_an_unknown_vendor_name_keeps_its_own_name_in_the_label():
    """An honest uppercase name beats a guessed vendor -- the same rule
    narrative.py::_PROSE applies to unlisted event names."""
    chain = FallbackRouter.from_settings(_live(llm_fallback_name="something-new"))

    assert "SOMETHING-NEW" in chain.providers[2]._label


def test_a_keyless_fallback_is_still_omitted_from_the_chain():
    """Naming the slot must not change from_settings' own rule: no key,
    not in the chain."""
    chain = FallbackRouter.from_settings(
        _live(llm_fallback_name="grok", llm_fallback_api_key=""))

    assert [p.name for p in chain.providers] == ["primary", "mistral"]


def test_the_name_reaches_a_chain_exhaustion_report():
    """D-101's chain summary is one of the places that used to say
    "gemini" regardless of what was actually called."""
    chain = FallbackRouter(
        [_Typed("primary", RuntimeError("x")), _Typed("grok", RuntimeError("y"))], 0.6)

    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete([{"role": "user", "content": "x"}])

    assert [n for n, _ in exc.value.attempts] == ["primary", "grok"]


# ---------------------------------------------------------------------------
# D-130 (P6-3) -- a provider already known to be dead is not called again
# ---------------------------------------------------------------------------


class _DeadClient(_CtxClient):
    """A client that has already recorded a non-transient failure -- the
    state llm/client.py sets on a 401/403/404. Constructed directly here
    rather than driven through a real HTTP failure: this file tests the
    ROUTER's reaction to the verdict; test_llm_client.py owns the verdict
    itself."""

    def __init__(self, name, reason="403 permission_denied", **kw):
        super().__init__(name, **kw)
        self.disabled_reason = reason


def test_a_disabled_provider_is_skipped_and_never_counted_as_an_attempt():
    dead = _DeadClient("primary")
    live = _CtxClient("mistral")
    router = FallbackRouter([dead, live], 0.6)

    result = router.complete_json(_SMALL)
    counters = router.drain_counters()

    assert result == {"served_by": "mistral"}
    assert dead.calls == 0, "a provider known to be dead must not be called"
    assert counters["llm_disabled_skips"] == 1.0
    assert counters["llm_provider_calls"] == 1.0
    assert "llm_fallback_hops" not in counters, (
        "a hop never attempted did not fall back from anything")


def test_the_free_text_path_skips_a_disabled_provider_too():
    dead = _DeadClient("primary")
    live = _CtxClient("mistral")
    router = FallbackRouter([dead, live], 0.6)

    assert router.complete(_SMALL) == "answer from mistral"
    assert dead.calls == 0
    assert router.drain_counters()["llm_disabled_skips"] == 1.0


def test_the_last_provider_is_never_skipped_even_when_disabled():
    """Same rule D-93 already applies: skipping is only meaningful when
    there is somewhere to fall through TO. Attempting a probably-doomed
    call beats leaving complete() with no candidate and no exception."""
    dead = _DeadClient("primary")
    router = FallbackRouter([dead], 0.6)

    assert router.complete_json(_SMALL) == {"served_by": "primary"}
    assert dead.calls == 1
    assert "llm_disabled_skips" not in router.drain_counters()


def test_a_disabled_hop_is_reported_as_skipped_not_as_a_failure():
    """D-101's chain summary must stay honest: a hop nobody attempted is
    not a failure, and omitting it would make the chain look shorter than
    it is."""
    chain = FallbackRouter(
        [_DeadClient("primary"),
         _Typed("mistral", TimeoutError("read timeout")),
         _Typed("grok", RuntimeError("500"))], 0.6)

    with pytest.raises(ProviderChainExhausted) as exc:
        chain.complete(_SMALL)

    assert exc.value.attempts == [
        ("primary", "skipped_disabled"),
        ("mistral", "TimeoutError"),
        ("grok", "RuntimeError")]


def test_disabled_is_checked_before_context():
    """A dead provider reported as "skipped_for_context" would send an
    operator to LLM_PRIMARY_CONTEXT_TOKENS to fix a billing problem. Both
    conditions are true of this hop; only one of them is the reason."""
    dead = _DeadClient("primary", context_tokens=1536)      # dead AND too small
    router = FallbackRouter([dead, _CtxClient("mistral")], 0.6)

    assert router.complete_json(_BIG) == {"served_by": "mistral"}
    counters = router.drain_counters()

    assert counters["llm_disabled_skips"] == 1.0
    assert "llm_context_skips" not in counters, (
        "the hop was skipped once, for the reason an operator must act on")


def test_a_judge_that_dies_takes_the_provider_out_of_the_answering_chain():
    """The reason the verdict lives on the CLIENT and not on the router.
    evaluation/quality.py swallows the judge's exception (fail-open), so
    the router never sees it -- but the client that suffered it is the
    same object the router later asks to ANSWER. Live shape: grok 403'd
    three times as the judge and three times as an answerer in one run."""
    import httpx as _httpx
    from research_agent.llm.client import OpenAICompatibleClient

    def _dead_handler(request):
        return _httpx.Response(403, text='{"error":"no credits"}')

    judge = OpenAICompatibleClient("grok", "http://x", "k", "grok-4.6")
    judge._http = _httpx.Client(transport=_httpx.MockTransport(_dead_handler),
                                base_url="http://x")
    first = _CtxClient("primary")
    last = _CtxClient("mistral")
    router = FallbackRouter([first, judge, last], 0.6)

    # 1. A free-text call: `first` answers, `judge` is asked to score it
    #    and 403s. Fail-open keeps the answer (P2-11's contract).
    assert router.complete(_SMALL) == "answer from primary"
    assert judge.disabled_reason == "403 permission_denied"
    assert router.drain_counters()["llm_quality_calls_failed"] == 1.0

    # 2. The next call has `first` fail. The chain must now go straight
    #    past the dead judge to `mistral` instead of spending a request
    #    on a provider whose account has already refused three times.
    router = FallbackRouter([_Typed("primary", RuntimeError("boom")),
                             judge, last], 0.6)
    assert router.complete_json(_SMALL) == {"served_by": "mistral"}
    counters = router.drain_counters()
    assert counters["llm_disabled_skips"] == 1.0
    assert counters["llm_provider_calls"] == 2.0, (
        "primary attempted, grok skipped, mistral attempted")


# ---------------------------------------------------------------------------
# D-153 -- every provider carries its own context window
#
# The router already READ this per-provider (_skips_for_context's getattr)
# and D-151 already LEARNS it per client. Only from_settings was
# primary-only.
# ---------------------------------------------------------------------------


def _chain_settings(**overrides):
    from research_agent.config import Settings

    base = {"_env_file": None, "llm_mode": "live",
            "llm_mistral_api_key": "k", "llm_fallback_api_key": "k"}
    base.update(overrides)
    return Settings(**base)


def test_each_slot_gets_its_own_window():
    from research_agent.llm.router import FallbackRouter

    router = FallbackRouter.from_settings(_chain_settings(
        llm_primary_context_tokens=1536,
        llm_mistral_context_tokens=32768,
        llm_fallback_context_tokens=1048576))

    assert {p.name: p.context_tokens for p in router.providers} == {
        "primary": 1536, "mistral": 32768, "gemini": 1048576}


def test_unset_slots_stay_zero_which_is_the_old_behaviour():
    """0 means "unknown" and can never be skipped, so an existing .env
    routes byte-identically to before D-153."""
    from research_agent.llm.router import FallbackRouter

    router = FallbackRouter.from_settings(_chain_settings(
        llm_primary_context_tokens=1536))

    assert [p.context_tokens for p in router.providers] == [1536, 0, 0]


def test_the_window_follows_the_slot_when_the_slot_is_renamed():
    """D-114 renames the third provider by configuration. Its window must
    follow the SLOT, not a vendor name."""
    from research_agent.llm.router import FallbackRouter

    router = FallbackRouter.from_settings(_chain_settings(
        llm_fallback_name="grok", llm_fallback_context_tokens=131072))

    grok = [p for p in router.providers if p.name == "grok"]
    assert grok and grok[0].context_tokens == 131072


def test_a_middle_provider_can_now_be_skipped():
    """The behaviour the feature exists for: a second LOCAL server in the
    mistral slot, too small for a compile prompt."""
    from research_agent.llm.router import FallbackRouter

    router = FallbackRouter.from_settings(_chain_settings(
        llm_mistral_context_tokens=2048))
    mistral = [p for p in router.providers if p.name == "mistral"][0]
    messages = [{"role": "user", "content": "x" * 40000}]  # ~10k tokens

    assert router._skips_for_context(mistral, messages) is True


class _TinyWindowClient:
    """A provider with a window far too small for anything, that answers
    if it is ever actually called."""

    def __init__(self, name):
        self.name = name
        self.context_tokens = 512
        self.calls = 0
        self.disabled_reason = ""

    def complete(self, messages, temperature=0.2):
        self.calls += 1
        return f"answered by {self.name}"

    def complete_json(self, messages, temperature=0.0):
        self.calls += 1
        return {"provider": self.name}

    def drain_usage(self):
        return (0, 0)  # (prompt_tokens, completion_tokens), per D-86


def test_the_last_provider_is_still_never_skipped():
    """The guard that stops the chain skipping its way to zero attempts.

    It was incidental while only the primary could be skipped -- there was
    always an unskippable provider behind it. With a window on EVERY slot
    it becomes the only thing preventing a chain that skips everything,
    returns no candidate and no exception, and trips complete()'s own
    `assert last_exc is not None`.
    """
    from research_agent.llm.router import FallbackRouter

    chain = [_TinyWindowClient(n) for n in ("primary", "mistral", "gemini")]
    router = FallbackRouter(chain, quality_threshold=0.0)
    huge = [{"role": "user", "content": "x" * 40000}]  # ~10k tokens

    answer = router.complete(huge)

    assert answer == "answered by gemini"
    assert [c.calls for c in chain] == [0, 0, 1], "only the last was attempted"
    assert router.drain_counters()["llm_context_skips"] == 2


def test_a_skip_records_which_provider_it_was():
    """One integer was unambiguous with one configurable window. With three
    it is not: "3 skips" says nothing about whether the chain lost its
    local hop or its cloud one."""
    from research_agent.llm.router import FallbackRouter

    router = FallbackRouter.from_settings(_chain_settings(
        llm_primary_context_tokens=512, llm_mistral_context_tokens=512))
    huge = [{"role": "user", "content": "x" * 40000}]

    for provider in router.providers[:2]:
        router._skips_for_context(provider, huge)
    counters = router.drain_counters()

    assert counters["llm_context_skips"] == 2
    assert counters["llm_context_skipped_primary"] == 1
    assert counters["llm_context_skipped_mistral"] == 1

# --------------------------------------------------------------------------
# S-26: the skip decision and the failed-hop record, extracted from the two
# copies that complete() and complete_json() each carried. These test the
# shared units directly; the cascade tests above still cover them in situ.
# --------------------------------------------------------------------------


class _Skippable:
    """Duck-typed provider exposing only what _skip_reason reads."""

    def __init__(self, name, context_tokens=0, disabled_reason=None):
        self.name = name
        self.context_tokens = context_tokens
        self.disabled_reason = disabled_reason

    def complete(self, messages, temperature=0.2):
        return "answer from " + self.name

    def complete_json(self, messages, temperature=0.0):
        return {"ok": self.name}


_BIG = [{"role": "user", "content": "x " * 4000}]


def test_skip_reason_never_skips_the_last_provider():
    """No next hop means a skip would be a silent failure, not a fallback.

    Both flags are set on p1 and it is still returned as callable, because
    it is last: this is the invariant _skips_for_context's docstring calls
    out (skipping it would leave complete() with no candidate and no
    exception, tripping its own `assert last_exc is not None`).
    """
    router = FallbackRouter(
        [_Skippable("p0"), _Skippable("p1", context_tokens=4,
                                      disabled_reason="http_400")],
        quality_threshold=0.6,
    )

    assert router._skip_reason(1, router.providers[1], _BIG) is None


def test_skip_reason_reports_disabled_before_context():
    """D-130 wins over D-93 when a provider trips both.

    A provider already known to be dead is not "skipped for context", and
    reporting it that way sends an operator to the wrong setting.
    """
    router = FallbackRouter(
        [_Skippable("p0", context_tokens=4, disabled_reason="http_400"),
         _Skippable("p1")],
        quality_threshold=0.6,
    )

    assert router._skip_reason(0, router.providers[0], _BIG) == "skipped_disabled"


def test_skip_reason_returns_none_for_a_callable_provider():
    router = FallbackRouter([_Skippable("p0"), _Skippable("p1")],
                            quality_threshold=0.6)

    assert router._skip_reason(0, router.providers[0], _BIG) is None


def test_record_hop_bumps_the_counter_only_when_a_next_hop_exists():
    """P2-07 counts real hops; the last provider failing is a dead end.

    Same router, same call, two positions -- the only difference is whether
    there is anywhere left to fall through to.
    """
    router = FallbackRouter([_Skippable("p0"), _Skippable("p1")],
                            quality_threshold=0.6)
    outcomes = []

    router._record_hop(0, router.providers[0], RuntimeError("x"), "text", outcomes)
    assert router.drain_counters()["llm_fallback_hops"] == 1

    router._record_hop(1, router.providers[1], RuntimeError("x"), "text", outcomes)
    assert router.drain_counters().get("llm_fallback_hops", 0) == 0


def test_record_hop_appends_the_exception_type_in_chain_order():
    """D-101: outcomes is built alongside last_exc, not reconstructed."""
    router = FallbackRouter([_Skippable("p0"), _Skippable("p1")],
                            quality_threshold=0.6)
    outcomes = []

    router._record_hop(0, router.providers[0], ValueError("a"), "json", outcomes)
    router._record_hop(1, router.providers[1], KeyError("b"), "json", outcomes)

    assert outcomes == [("p0", "ValueError"), ("p1", "KeyError")]
