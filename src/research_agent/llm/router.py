"""
llm/router.py — Primary-then-fallback model routing.

Purpose:
    Every LLM call in the agent goes through FallbackRouter, which tries the
    primary model (Qwen Cogito) and falls back to Gemini Flash when the call
    fails or its quality is below threshold.

Responsibilities:
    - Route complete()/complete_json() with automatic fallback.
    - Decide WHEN to fall back (the policy lives here, nowhere else):
        1. transport/HTTP error from the primary,
        2. unparseable JSON when JSON was required,
        3. self-evaluated quality score below llm_quality_threshold
           (free-text answers only; see evaluation/quality.py).
    - Log every fallback decision so runs are auditable.

Design decision (self-evaluated quality):
    A model scoring its own answer is a weak but cheap signal — good enough
    to catch obviously broken output from a small local model, which is the
    actual failure mode this guards. Alternatives considered: a second-model
    judge on every call (doubles cost/latency) or perplexity heuristics
    (needs logprobs the compat endpoint may not return). Limitation stated
    plainly: self-scores are optimistic; tune the threshold empirically.
"""

import logging
from typing import Any, Dict, List, Optional

from research_agent.config import Settings
from research_agent.evaluation.quality import score_answer
from research_agent.llm.client import ChatClient, Message, OpenAICompatibleClient, StubClient
from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


class FallbackRouter:
    """Primary/fallback orchestration over two ChatClients."""

    def __init__(self, primary: ChatClient, fallback: Optional[ChatClient],
                 quality_threshold: float):
        """fallback may be None (e.g. stub mode) — then errors just raise."""
        self.primary = primary
        self.fallback = fallback
        self.quality_threshold = quality_threshold

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_settings(cls, s: Settings) -> "FallbackRouter":
        """Build the router the way cli/api do. Stub mode gets no fallback —
        deterministic tests must never silently route elsewhere."""
        if s.llm_mode == "stub":
            return cls(StubClient(), None, s.llm_quality_threshold)
        primary = OpenAICompatibleClient(
            "primary", s.llm_primary_base_url, s.llm_primary_api_key,
            s.llm_primary_model, s.llm_timeout_seconds)
        fallback = OpenAICompatibleClient(
            "fallback", s.llm_fallback_base_url, s.llm_fallback_api_key,
            s.llm_fallback_model, s.llm_timeout_seconds)
        return cls(primary, fallback, s.llm_quality_threshold)

    # -- routed calls -------------------------------------------------------

    def complete_json(self, messages: List[Message]) -> Dict[str, Any]:
        """Structured call. Fallback on error or unparseable JSON.

        Returns the parsed object; raises only if BOTH providers fail —
        callers treat that as a node failure handled by graph-level policy.
        """
        try:
            return self.primary.complete_json(messages)
        except Exception as exc:  # noqa: BLE001 — any primary failure routes over
            if self.fallback is None:
                raise
            log_event(logger, "llm.fallback", reason=type(exc).__name__, mode="json")
            return self.fallback.complete_json(messages)

    def complete(self, messages: List[Message]) -> str:
        """Free-text call. Fallback on error OR low self-evaluated quality."""
        try:
            answer = self.primary.complete(messages)
        except Exception as exc:  # noqa: BLE001
            if self.fallback is None:
                raise
            log_event(logger, "llm.fallback", reason=type(exc).__name__, mode="text")
            return self.fallback.complete(messages)

        if self.fallback is not None:
            quality = score_answer(self.primary, messages, answer)
            if quality < self.quality_threshold:
                log_event(logger, "llm.fallback", reason="low_quality",
                          score=quality, threshold=self.quality_threshold)
                return self.fallback.complete(messages)
        return answer
