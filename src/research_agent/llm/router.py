"""
llm/router.py — Chained provider fallback (primary → … → last).

Purpose:
    Every LLM call in the agent goes through FallbackRouter, which tries an
    ORDERED CHAIN of providers and steps to the next one whenever a provider
    is unavailable/errors OR returns below-threshold quality. Default chain:
    local Qwen Cogito → Mistral → Gemini Flash.

Responsibilities:
    - Route complete()/complete_json() down the chain.
    - Decide WHEN to step to the next provider (the policy lives here, nowhere
      else), applying the SAME rule at every hop:
        1. transport/HTTP error from the current provider,
        2. unparseable JSON when JSON was required,
        3. self-evaluated quality score below llm_quality_threshold
           (free-text answers only; see evaluation/quality.py).
    - Log every hop so runs are auditable (which provider served, why we moved).

Design decisions:
    - Generalized from a fixed primary/fallback pair to an N-provider list so
      adding a fourth provider is a config change, not a code change. The chain
      is just an ordered list of ChatClients; routing logic is identical at
      every position.
    - Same trigger (error OR low quality) at every hop, per requirement.
      Quality is scored by the SAME provider that produced the answer — a weak
      but cheap signal that catches broken output. A provider that errors on the
      quality-scoring call is treated as "quality unknown → keep its answer"
      rather than cascading further, so a flaky scorer can't burn the whole
      chain (see _passes_quality).
    - Stub mode builds a single-element chain with no downstream providers, so
      deterministic tests never silently route elsewhere.
"""

import logging
from typing import Any, Dict, List, Optional

from research_agent.config import Settings
from research_agent.evaluation.quality import score_answer
from research_agent.llm.client import ChatClient, Message, OpenAICompatibleClient, StubClient
from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


class FallbackRouter:
    """Ordered-chain orchestration over one or more ChatClients."""

    def __init__(self, providers: List[ChatClient], quality_threshold: float):
        """`providers` is the fallback ORDER: index 0 is tried first, and each
        subsequent provider is a fallback for the ones before it. Must be
        non-empty. A single-element chain simply never falls back."""
        if not providers:
            raise ValueError("FallbackRouter needs at least one provider")
        self.providers = providers
        self.quality_threshold = quality_threshold

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_settings(cls, s: Settings) -> "FallbackRouter":
        """Build the router the way cli/api do.

        Stub mode -> a single stub provider, no downstream (deterministic tests
        must never silently route elsewhere).

        Live mode -> the chain [primary, *fallbacks], skipping any fallback that
        has no API key configured, so an unconfigured provider is simply absent
        from the chain rather than a guaranteed error mid-run.
        """
        if s.llm_mode == "stub":
            return cls([StubClient()], s.llm_quality_threshold)

        chain: List[ChatClient] = [OpenAICompatibleClient(
            "primary", s.llm_primary_base_url, s.llm_primary_api_key,
            s.llm_primary_model, s.llm_timeout_seconds)]

        # Ordered fallbacks. Each is included only if it has an API key -- a
        # provider you haven't configured is skipped, not a landmine.
        for name, base, key, model in (
            ("mistral", s.llm_mistral_base_url, s.llm_mistral_api_key, s.llm_mistral_model),
            ("gemini", s.llm_fallback_base_url, s.llm_fallback_api_key, s.llm_fallback_model),
        ):
            if key:
                chain.append(OpenAICompatibleClient(
                    name, base, key, model, s.llm_timeout_seconds))

        log_event(logger, "llm.chain_built", providers=[p.name for p in chain])
        return cls(chain, s.llm_quality_threshold)

    # -- internals ----------------------------------------------------------

    def _passes_quality(self, provider: ChatClient, messages: List[Message],
                        answer: str) -> bool:
        """True if `answer` clears the quality gate (or can't be scored).

        A scorer that itself errors returns 1.0 (see evaluation.quality), so a
        flaky quality check keeps the answer rather than cascading -- the gate
        exists to catch bad ANSWERS, not to punish a bad scoring call.
        """
        score = score_answer(provider, messages, answer)
        if score < self.quality_threshold:
            log_event(logger, "llm.quality_reject", provider=provider.name,
                      score=score, threshold=self.quality_threshold)
            return False
        return True

    # -- routed calls -------------------------------------------------------

    def complete_json(self, messages: List[Message]) -> Dict[str, Any]:
        """Structured call. Step down the chain on error/unparseable JSON.

        No quality gate here: a successfully parsed JSON object either satisfies
        the caller's schema or it doesn't, and the nodes validate their own
        required keys. Returns the first provider's parsed object; raises the
        LAST provider's error only if every provider in the chain fails.
        """
        last_exc: Optional[Exception] = None
        for i, provider in enumerate(self.providers):
            try:
                result = provider.complete_json(messages)
                if i > 0:
                    log_event(logger, "llm.served_by_fallback",
                              provider=provider.name, position=i, mode="json")
                return result
            except Exception as exc:  # noqa: BLE001 -- any failure steps to next
                last_exc = exc
                nxt = (self.providers[i + 1].name
                       if i + 1 < len(self.providers) else None)
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=nxt, reason=type(exc).__name__, mode="json")
        assert last_exc is not None
        raise last_exc

    def complete(self, messages: List[Message]) -> str:
        """Free-text call. Step down the chain on error OR low quality.

        Same trigger at every hop. Returns the first answer that both succeeds
        and clears the quality gate; if none does, returns the LAST provider's
        answer when it produced one (better a low-scored answer than nothing),
        or raises if the entire chain errored.
        """
        last_exc: Optional[Exception] = None
        last_answer: Optional[str] = None
        last_name: str = ""

        for i, provider in enumerate(self.providers):
            # 1. availability / error
            try:
                answer = provider.complete(messages)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                nxt = (self.providers[i + 1].name
                       if i + 1 < len(self.providers) else None)
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=nxt, reason=type(exc).__name__, mode="text")
                continue

            last_answer, last_name = answer, provider.name

            # 2. quality gate -- only worth checking if a fallback remains
            has_next = i + 1 < len(self.providers)
            if has_next and not self._passes_quality(provider, messages, answer):
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=self.providers[i + 1].name,
                          reason="low_quality", mode="text")
                continue

            if i > 0:
                log_event(logger, "llm.served_by_fallback",
                          provider=provider.name, position=i, mode="text")
            return answer

        # Chain exhausted. Prefer the last answer we got over raising.
        if last_answer is not None:
            log_event(logger, "llm.chain_exhausted_low_quality", provider=last_name)
            return last_answer
        assert last_exc is not None
        raise last_exc
