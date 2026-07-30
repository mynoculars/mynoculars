"""
research_agent/langfuse/pricing.py — cost calculation for LLM generations.

WHY THIS FILE EXISTS: Phase 3's requirement is explicit -- "Pricing must
NOT be hardcoded. Configure via Settings / environment." Every $/1M-token
rate lives in config.py as a Settings field (LANGFUSE_PRICE_<PROVIDER>_
<IN|OUT>_PER_1M), never as a literal in this file. This module only knows
HOW to turn (provider, prompt_tokens, completion_tokens) into a dollar
figure given whatever rates Settings handed it -- it never decides what
those rates ARE.

A provider with a rate of 0.0 (the default for every provider, including
the local primary) costs exactly $0.0 -- correct for a genuinely free
local model, and an honest "not configured" for a cloud provider whose
real rate you haven't set, rather than silently inventing a plausible-
looking number that happens to be wrong.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

logger = logging.getLogger("research_agent.langfuse")


class TokenUsage(NamedTuple):
    """Plain token counts for one LLM call. Mirrors what llm/client.py
    already extracts from a provider response, so callers never need to
    reshape anything to hand this module data."""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CostBreakdown(NamedTuple):
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


# Maps the same provider names FallbackRouter already uses internally
# ("primary", "mistral", "gemini" -- see llm/router.py's provider list)
# to the pair of Settings fields that carry that provider's rate. Adding
# a fourth provider later means adding one line here and two fields in
# config.py -- never a new code path in the SDK-facing modules.
_PROVIDER_RATE_FIELDS = {
    "primary": ("langfuse_price_primary_in_per_1m", "langfuse_price_primary_out_per_1m"),
    "mistral": ("langfuse_price_mistral_in_per_1m", "langfuse_price_mistral_out_per_1m"),
    "gemini": ("langfuse_price_gemini_in_per_1m", "langfuse_price_gemini_out_per_1m"),
}


COST_MODE_EXPLICIT = "explicit"
COST_MODE_INFER = "infer"
_VALID_COST_MODES = (COST_MODE_EXPLICIT, COST_MODE_INFER)


def resolve_cost_mode(raw) -> str:
    """Normalize `settings.langfuse_cost_mode`.

    WHY A MODE EXISTS AT ALL: a computed cost of exactly $0 is ambiguous.
    It means "this model is free" for the local primary, and "nobody ever
    set LANGFUSE_PRICE_<provider>_*_PER_1M" for a cloud provider -- and
    the two produce byte-identical inputs here. Reading the difference off
    Settings is not possible either: pydantic-settings passes every
    resolved value into the model, so `model_fields_set` reports even
    untouched defaults as set (verified, not assumed). So the distinction
    has to be declared, not detected.

    Unrecognized values resolve to COST_MODE_EXPLICIT -- the behavior that
    predates this setting, so a typo changes nothing rather than silently
    switching how every generation is priced.
    """
    mode = str(raw or "").strip().lower()
    if mode in _VALID_COST_MODES:
        return mode
    logger.warning(
        "langfuse.cost_mode_unrecognized",
        extra={"configured": str(raw)[:60], "using": COST_MODE_EXPLICIT,
               "valid": ",".join(_VALID_COST_MODES)},
    )
    return COST_MODE_EXPLICIT


def calculate_cost(settings, provider: str, usage: TokenUsage) -> Optional[CostBreakdown]:
    """Return the USD cost of one generation, or None if the provider name
    isn't recognized at all (never guessed at, never defaulted to some
    other provider's rate).

    `settings` is the same Settings object every other module already
    receives (see config.py) -- this function does not read os.environ
    and does not cache; get_settings() is already an lru_cache singleton
    upstream, so there's nothing to gain and a testability cost to lose
    by caching again here.
    """
    fields = _PROVIDER_RATE_FIELDS.get(provider)
    if fields is None:
        return None
    in_field, out_field = fields
    # max(0.0, ...) guards a misconfigured negative LANGFUSE_PRICE_* env
    # var from producing a negative cost figure -- a typo (e.g. a stray
    # "-" pasted into the value) should degrade to "treat it as free",
    # not silently report negative dollars.
    rate_in = max(0.0, getattr(settings, in_field, 0.0) or 0.0)
    rate_out = max(0.0, getattr(settings, out_field, 0.0) or 0.0)
    input_cost = (usage.prompt_tokens / 1_000_000.0) * rate_in
    output_cost = (usage.completion_tokens / 1_000_000.0) * rate_out
    return CostBreakdown(input_cost_usd=input_cost, output_cost_usd=output_cost)
