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

import pytest

from research_agent.llm.router import FallbackRouter

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
