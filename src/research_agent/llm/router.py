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

Python mechanics used in this file, if any of this is new to you:
    @classmethod
        A decorator marking a method that receives the CLASS itself (here
        named `cls`) as its first argument, instead of an instance (`self`).
        FallbackRouter.from_settings(...) below is called directly on the
        class — FallbackRouter.from_settings(settings) — without ever
        having created a FallbackRouter instance first; inside the method,
        `cls(...)` at the very end constructs and returns one. This is the
        standard Python pattern for an "alternative constructor" — a second
        way to build an object besides calling FallbackRouter(...) directly,
        used here because building the provider chain from Settings needs
        several steps (checking which API keys are set, etc.) that don't
        belong inside __init__ itself.
    for name, base, key, model, label in (("mistral", ...), ("gemini", ...)):
        This loops over a TUPLE OF TUPLES, and on each iteration UNPACKS the
        inner tuple's five elements into five separate variable names in one
        line — equivalent to writing, for the first iteration:
            name, base, key, model, label = ("mistral", s.llm_mistral_base_url, ...)
        This is just a compact way to loop over several related config
        values (one row per optional fallback provider) without repeating
        near-identical code for Mistral and then again for Gemini.
    enumerate(self.providers)
        Loops over a list while also giving you each item's index — so
        `for i, provider in enumerate(self.providers):` gives you both the
        position (0, 1, 2, ...) AND the provider object on each pass,
        instead of having to track the index by hand with a separate
        counter variable.
"""

import logging
from typing import Any, Dict, List, Optional

from research_agent.config import Settings
from research_agent.tracing import NullTracer, Tracer
from research_agent.evaluation.quality import score_answer
from research_agent.llm.client import ChatClient, Message, OpenAICompatibleClient, StubClient
from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


class FallbackRouter:
    """Ordered-chain orchestration over one or more ChatClients.

    Every node in agents/*.py that needs an LLM calls router.complete(...)
    or router.complete_json(...) on ONE shared instance of this class (see
    cli.py::build_app_and_settings, which constructs it once per run/process
    via from_settings() below) — no node ever talks to an OpenAICompatibleClient
    or StubClient directly.
    """

    def __init__(self, providers: List[ChatClient], quality_threshold: float,
                 tracer: Optional[Tracer] = None):
        """`providers` is the fallback ORDER: index 0 is tried first, and each
        subsequent provider is a fallback for the ones before it. Must be
        non-empty. A single-element chain simply never falls back. `tracer`
        (optional) is forwarded to the clients for debug tracing.
        """
        if not providers:
            raise ValueError("FallbackRouter needs at least one provider")
        self.providers = providers
        self.quality_threshold = quality_threshold
        self.tracer = tracer or NullTracer()
        # P2-07: boundary-scoped telemetry. This router is the ONE place
        # that actually knows how many real provider requests, fallback
        # hops, and quality-scoring calls happened underneath a single
        # node's complete()/complete_json() call — node-level counters
        # (agents/*.py's "llm_node_calls") only ever counted NODE
        # executions, invisible to fallback hops and self-scoring calls
        # made entirely inside this class. Accumulated here, then drained
        # by each calling node into its own returned counters dict (see
        # drain_counters below) — never written to ResearchState directly,
        # since this class has no knowledge of the graph at all.
        self._counters: Dict[str, float] = {}

    def _bump(self, key: str, amount: float = 1.0) -> None:
        """Internal: add `amount` to one accumulated counter."""
        self._counters[key] = self._counters.get(key, 0.0) + amount

    def drain_counters(self) -> Dict[str, float]:
        """Return everything accumulated since the last drain, and reset.

        CALLED BY   every agents/*.py node right after its router.complete()
                    or router.complete_json() call, to fold these
                    provider-level counts into the SAME counters dict the
                    node already returns (state.counters is reducer-backed
                    via merge_counters, so adding these keys needs no state
                    change — see state.py).
        WHY DRAIN, NOT PEEK: each node call should only ever report the
        provider activity ITS OWN call caused, never a stale total left
        over from an earlier node in the same run — draining (read + reset
        in one step) makes that structurally guaranteed rather than
        something every call site has to remember to do correctly.
        """
        drained = self._counters
        self._counters = {}
        return drained

    def set_node(self, node: Optional[str]) -> None:
        """Tag subsequent calls with the current graph node, so the trace and
        the llm.call log line show WHICH node issued each call.

        CALLED BY   every node function in agents/*.py that makes an LLM
                    call, right before calling router.complete_json(...) or
                    router.complete(...) — e.g. agents/planning.py's
                    classify_node does `router.set_node("classify")` first.
        """
        for p in self.providers:
            # getattr(p, "set_trace_node", None) looks up an ATTRIBUTE (here,
            # a method) on object p BY NAME, returning None instead of
            # raising an error if it doesn't exist. This is defensive:
            # every ChatClient this codebase actually uses DOES define
            # set_trace_node, but writing it this way means a future,
            # simpler ChatClient implementation that skips tracing entirely
            # wouldn't crash this loop.
            setter = getattr(p, "set_trace_node", None)
            if setter:
                setter(node)

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_settings(cls, s: Settings, tracer: Optional[Tracer] = None) -> "FallbackRouter":
        """Build the router the way cli/api do.

        CALLED BY   cli.py::build_app_and_settings — the only call site.
        Stub mode -> a single stub provider, no downstream (deterministic tests
        must never silently route elsewhere).

        Live mode -> the chain [primary, *fallbacks], skipping any fallback that
        has no API key configured, so an unconfigured provider is simply absent
        from the chain rather than a guaranteed error mid-run.

        Two DIFFERENT timeouts are used here, not one shared value: the
        primary (local Cogito) gets settings.llm_primary_timeout_seconds,
        every cloud fallback gets settings.llm_timeout_seconds. See
        config.py's comment on those two fields for why they're split —
        in short, a local model can need much longer than a cloud API
        before a slow response is actually a problem worth failing over.
        """
        if s.llm_mode == "stub":
            return cls([StubClient(tracer=tracer)], s.llm_quality_threshold, tracer)

        chain: List[ChatClient] = [OpenAICompatibleClient(
            "primary", s.llm_primary_base_url, s.llm_primary_api_key,
            s.llm_primary_model, s.llm_primary_timeout_seconds, tracer,
            display_label=f"LOCAL PRIMARY ({s.llm_primary_model})")]

        # See the module docstring for exactly what this tuple-of-tuples
        # loop with unpacking is doing. Each row here is one OPTIONAL
        # fallback provider; the loop body only actually adds it to `chain`
        # if its API key is non-empty.
        for name, base, key, model, label in (
            ("mistral", s.llm_mistral_base_url, s.llm_mistral_api_key,
             s.llm_mistral_model, f"MISTRAL ({s.llm_mistral_model})"),
            ("gemini", s.llm_fallback_base_url, s.llm_fallback_api_key,
             s.llm_fallback_model, f"GOOGLE GEMINI ({s.llm_fallback_model})"),
        ):
            if key:
                chain.append(OpenAICompatibleClient(
                    name, base, key, model, s.llm_timeout_seconds, tracer,
                    display_label=label))

        log_event(logger, "llm.chain_built", providers=[p.name for p in chain])
        return cls(chain, s.llm_quality_threshold, tracer)

    # -- internals ----------------------------------------------------------

    def _passes_quality(self, provider: ChatClient, messages: List[Message],
                        answer: str) -> bool:
        """True if `answer` clears the quality gate (or can't be scored).

        CALLED BY   self.complete() below, ONLY when a further fallback
                    hop is available — see complete()'s docstring for why
                    it isn't worth checking on the last provider in the
                    chain.
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

        CALLED BY   classify_node, goal_manager_node, task_expander_node,
                    gap_generator_node, critic_node (agents/planning.py,
                    agents/gathering.py, agents/compilation.py) — every
                    node that needs a structured (JSON) answer from a model.
        READS       nothing from ResearchState directly — receives only the
                    prompt `messages` its caller built (see
                    prompts/templates.py).
        RETURNS     the first provider's successfully parsed dict. Raises
                    the LAST provider's error only if EVERY provider in the
                    chain failed.

        No quality gate here: a successfully parsed JSON object either satisfies
        the caller's schema or it doesn't, and the nodes validate their own
        required keys. Returns the first provider's parsed object; raises the
        LAST provider's error only if every provider in the chain fails.
        """
        last_exc: Optional[Exception] = None
        # enumerate(self.providers) — see the module docstring — gives us
        # both the position `i` (0, 1, 2...) and the `provider` object on
        # each pass, in the fixed order the chain was built in.
        for i, provider in enumerate(self.providers):
            self._bump("llm_provider_calls")  # P2-07: one real attempt, win or lose
            try:
                result = provider.complete_json(messages)
                if i > 0:
                    log_event(logger, "llm.served_by_fallback",
                              provider=provider.name, position=i, mode="json")
                return result
            except Exception as exc:  # noqa: BLE001 -- any failure steps to next
                # Catching the broad `Exception` here is DELIBERATE (the
                # noqa comment tells a linter "yes, I meant to do this, stop
                # warning me about it") — any kind of failure from this
                # provider, whatever its exact type, should trigger the same
                # fallback behaviour: try the next one.
                last_exc = exc
                nxt = (self.providers[i + 1].name
                       if i + 1 < len(self.providers) else None)
                if nxt is not None:
                    self._bump("llm_fallback_hops")  # P2-07: a real hop, not the last dead end
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=nxt, reason=type(exc).__name__, mode="json")
        # If we reach this line, every provider in the loop above raised.
        # `assert last_exc is not None` is a sanity check for a human reader
        # (and for tools like mypy) that this line is only reachable when
        # last_exc has actually been set — it would only fail if
        # self.providers were somehow empty, which __init__ already forbids.
        assert last_exc is not None
        raise last_exc

    def complete(self, messages: List[Message]) -> str:
        """Free-text call. Step down the chain on error OR low quality.

        CALLED BY   compiler_node (agents/compilation.py) — the ONLY node
                    in the whole codebase that makes a free-text (as
                    opposed to JSON) LLM call, which is why this is the
                    only call path where the quality gate below ever runs.
        RETURNS     the first answer that both succeeds AND clears the
                    quality gate; if none does, the LAST provider's answer
                    (better a low-scored answer than nothing), or raises if
                    every single provider in the chain errored outright.

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
            self._bump("llm_provider_calls")  # P2-07: one real attempt, win or lose
            try:
                answer = provider.complete(messages)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                nxt = (self.providers[i + 1].name
                       if i + 1 < len(self.providers) else None)
                if nxt is not None:
                    self._bump("llm_fallback_hops")
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=nxt, reason=type(exc).__name__, mode="text")
                continue  # move on to the next provider in the loop

            last_answer, last_name = answer, provider.name

            # 2. quality gate -- only worth checking if a fallback remains.
            # `i + 1 < len(self.providers)` is just "is there at least one
            # more provider after this one in the list?" — if this is
            # already the LAST provider, there is nothing to fall back TO,
            # so scoring its answer would only ever discard it for nothing
            # in return; better to just accept whatever it produced.
            has_next = i + 1 < len(self.providers)
            if has_next:
                self._bump("llm_quality_calls")  # P2-07: a real self-scoring call
            if has_next and not self._passes_quality(provider, messages, answer):
                self._bump("llm_fallback_hops")
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
