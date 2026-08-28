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
        3. quality score (judged by the NEXT provider in the chain, never
           the answering one — P2-11) below llm_quality_threshold
           (free-text answers only; see evaluation/quality.py).
    - Log every hop so runs are auditable (which provider served, why we moved).

Design decisions:
    - Generalized from a fixed primary/fallback pair to an N-provider list so
      adding a fourth provider is a config change, not a code change. The chain
      is just an ordered list of ChatClients; routing logic is identical at
      every position.
    - Same trigger (error OR low quality) at every hop, per requirement.
      Quality is judged by the NEXT provider in the chain (P2-11 — previously
      the SAME provider that produced the answer, which a real run showed
      passing a report with uncited sections). A judge that errors on the
      quality-scoring call is treated as "quality unknown → keep its answer"
      rather than cascading further, so a flaky judge can't burn the whole
      chain (see _score_quality).
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
import threading
from typing import Any, Dict, List, Optional, Tuple

from research_agent.config import Settings
from research_agent.tracing import NullTracer, Tracer
from research_agent.evaluation.quality import score_answer
from research_agent.llm.client import (ChatClient, Message,
                                       OpenAICompatibleClient, StubClient,
                                       estimate_prompt_tokens)
from research_agent import langfuse as lf
from research_agent.logging_setup import log_event, run_id_var

logger = logging.getLogger(__name__)


# D-114: vendor names for the trace banner, for the two providers this
# codebase has actually been run against in the cloud-fallback slot. An
# unlisted name falls back to its own uppercase form -- an honest
# "GROK-NEXT (model)" beats a guessed vendor, the same rule
# reporting/narrative.py::_PROSE applies to unlisted event names.
_FALLBACK_DISPLAY_NAMES = {
    "gemini": "GOOGLE GEMINI",
    "grok": "XAI GROK",
}


def _display_label(name: str) -> str:
    """Human name for the trace banner. See _FALLBACK_DISPLAY_NAMES."""
    return _FALLBACK_DISPLAY_NAMES.get(name, name.upper())


class ProviderChainExhausted(RuntimeError):
    """Every provider in the chain failed; there is no answer to return.

    RAISED BY   FallbackRouter.complete and complete_json below, in place
                of the bare `raise last_exc` they used to end on.
    CAUGHT BY   cli.py::main, which turns it into a diagnosable message
                and a distinct exit code (4) instead of a raw traceback --
                the same treatment GraphRecursionError already gets there,
                for the reason stated in that handler's own comment: an
                operational event is not a crash.

    WHY A TYPE AT ALL (D-101): main() could not identify chain exhaustion
    from the exception it received, because that was whatever the LAST
    provider happened to throw -- three different types across two live
    runs (TruncatedGenerationError, ReadTimeout, HTTPStatusError), none of
    which says on its own that the other two also failed, or how. That
    information exists only here, in the loop that watched all of them.

    Subclasses RuntimeError DELIBERATELY, and this is what keeps the
    change contained: both loops below already catch Exception broadly, no
    node catches a provider exception by type, and the two existing tests
    that assert exhaustion (test_llm_router.py::
    test_router_raises_when_no_fallback and
    ::test_all_providers_erroring_still_raises) assert `pytest.raises(
    RuntimeError)` -- so nothing that used to catch the last provider's
    own exception stops catching this one.

    `raise ... from last_exc` at both call sites keeps the real underlying
    failure as __cause__, so a traceback still shows it in full.

    attempts is the WHOLE chain in order, one (provider_name, outcome)
    pair each, where outcome is an exception type name or the literal
    "skipped_for_context" -- a hop D-93 never attempted is not a failure
    and must not be reported as one, but leaving it out entirely would
    make the chain in the message look shorter than it is.
    """

    def __init__(self, node, attempts, mode):
        self.node = node
        self.attempts = list(attempts)
        self.mode = mode
        chain = " -> ".join(f"{name} {how}" for name, how in self.attempts)
        where = f" at the {node} node" if node else ""
        super().__init__(
            f"provider chain exhausted{where} ({mode} call): {chain}")


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
        # executions, invisible to fallback hops and quality-scoring calls
        # made entirely inside this class. Accumulated here, then drained
        # by each calling node into its own returned counters dict (see
        # drain_counters below) — never written to ResearchState directly,
        # since this class has no knowledge of the graph at all.
        #
        # threading.local(), not a plain dict: ONE router instance is shared
        # process-wide, and _bump/drain_counters is an unlocked
        # read-modify-write. That is correct TODAY only because no two
        # LLM-calling nodes ever share a superstep — an invariant nothing
        # enforced and nothing wrote down. The first Send-parallelised
        # LLM node (parallel gap generation, per-goal critique) would
        # silently lose counts. Per-thread storage makes it structurally
        # safe instead, matching what retrieval/hybrid.py already does for
        # the same reason.
        self._local = threading.local()

    @property
    def _counters(self) -> Dict[str, float]:
        """This thread's own counter dict, created on first touch."""
        counters = getattr(self._local, "counters", None)
        if counters is None:
            counters = {}
            self._local.counters = counters
        return counters

    @property
    def _node(self) -> Optional[str]:
        """This thread's current graph node, or None if set_node was never
        called on this thread (every real node calls it; hand-written test
        routers generally do not)."""
        return getattr(self._local, "node", None)

    def _bump(self, key: str, amount: float = 1.0) -> None:
        """Internal: add `amount` to one accumulated counter."""
        counters = self._counters
        counters[key] = counters.get(key, 0.0) + amount

    # D-106: fixed bands for the score distribution, as (exclusive upper
    # bound, counter suffix) in ascending order. Bands rather than a
    # min/max pair because counters are MERGED BY ADDITION across parallel
    # nodes (state.py::merge_counters) -- a running minimum would be
    # silently wrong the first time two nodes judged in the same superstep,
    # while a histogram sums correctly by construction.
    #
    # The bands are fixed and INDEPENDENT of llm_quality_threshold, which
    # is the point: they show where the threshold sits inside the observed
    # distribution. A band set that moved with the threshold could never
    # answer "is 0.6 in the right place".
    QUALITY_BANDS = ((0.2, "very_low"), (0.4, "low"), (0.6, "mid"),
                     (0.8, "high"), (1.01, "very_high"))

    def _record_quality_score(self, score: float, reason: str,
                              provider: str, judge: str) -> None:
        """Fold one REAL judgement into this thread's counters (D-106).

        CALLED BY   _score_quality's on_scored callback below, and never
                    on the fail-open path -- score_answer only invokes
                    on_scored when the judge actually answered. The
                    fabricated 1.0 of a broken judge is not a judgement,
                    and counting it would make a dead judge look like a
                    generous one, which is the exact confusion P2-11's
                    llm_quality_calls_failed was added to end.

        llm_quality_scores_judged is deliberately NOT llm_quality_calls:
        the latter counts judging ATTEMPTS, including the ones that
        failed open. Only the former is a safe denominator for the mean.

        The reason is logged, not counted -- it is text, and counters are
        float-valued by contract (merge_counters). That means a run's
        justifications live in logs/run-<id>.txt and Langfuse, not in its
        agent_runs row; see D-106's note on what that does and does not
        make answerable.
        """
        self._bump("llm_quality_scores_judged")
        self._bump("llm_quality_score_sum", float(score))
        for upper, name in self.QUALITY_BANDS:
            if score < upper:
                self._bump(f"llm_quality_band_{name}")
                break
        passed = score >= self.quality_threshold
        if not passed:
            self._bump("llm_quality_rejections")
        log_event(logger, "llm.quality_scored", provider=provider, judge=judge,
                  score=score, threshold=self.quality_threshold,
                  passed=passed, reason=reason or None)

    # D-93: slack between the estimate and the limit before skipping a
    # hop. estimate_prompt_tokens is ~4 chars/token -- good enough to tell
    # 7,198 from 216, nowhere near good enough to trust at the boundary.
    # Skipping only past 1.1x the configured window means a mis-estimate
    # near the limit still ATTEMPTS the provider (worst case: today's
    # behaviour, one failed call) rather than silently discarding a hop
    # that would have worked.
    #
    # The asymmetry is deliberate: a false skip is invisible and
    # permanent, a false attempt is one logged, recovered failure.
    CONTEXT_SKIP_MARGIN = 1.1

    def _skips_for_context(self, provider: ChatClient,
                           messages: List[Message]) -> bool:
        """Would this prompt certainly not fit this provider's window?

        NEVER called for the LAST provider in the chain -- both call
        sites guard on `i + 1 < len(self.providers)` first. Skipping is
        an optimisation that only makes sense when there is somewhere
        to fall through TO; skipping the last option would leave
        complete() with no candidate and no exception, tripping its own
        `assert last_exc is not None`. Attempting a call that will
        probably fail is strictly better than crashing the run.

        Returns False -- attempt the call -- unless BOTH a context window
        is configured for this provider AND the estimate exceeds it by
        more than CONTEXT_SKIP_MARGIN. An unconfigured provider
        (context_tokens 0, the default everywhere) can never be skipped,
        so with no configuration this changes nothing at all.

        WHY: live (p205.246-check) the local primary answered 216- and
        444-token prompts and rejected 4,023- and 7,198-token ones with an
        HTTP error in 95ms and 29ms -- an immediate refusal against a
        120-second timeout, i.e. deterministic rather than flaky.
        OPERATIONS.md documents that machine's own invocation as `-c 1536`.
        Every compile and every critique therefore burned a
        guaranteed-failed provider call before falling back, twice per run,
        and the logs read as provider flakiness.
        """
        limit = int(getattr(provider, "context_tokens", 0) or 0)
        if limit <= 0:
            return False
        estimated = estimate_prompt_tokens(messages)
        if estimated <= limit * self.CONTEXT_SKIP_MARGIN:
            return False
        self._bump("llm_context_skips")
        log_event(logger, "llm.skipped_for_context", level=logging.WARNING,
                  provider=provider.name, estimated_prompt_tokens=estimated,
                  context_tokens=limit, margin=self.CONTEXT_SKIP_MARGIN)
        return True

    def _bump_usage(self, provider: ChatClient) -> None:
        """Fold one provider call's token usage into this thread's counters.

        D-86. The run-level token total was the one real-cost figure this
        harness could not report: `llm_provider_calls` counts REQUESTS,
        which says nothing about a run that made three cheap classify
        calls versus one that made three 7,000-token compile calls. The
        numbers were already being parsed and logged per call
        (llm/client.py's `llm.call` line) -- they simply never aggregated
        anywhere, so nothing could answer "what did this run cost".

        CALLED BY   complete_json and complete below, immediately after a
                    provider call SUCCEEDS, and by _score_quality for the
                    judge's own call. A call that raised reported no
                    usage and adds nothing.
        WHY TOKENS AND NOT DOLLARS: langfuse/pricing.py can already turn
        tokens into cost, but every LANGFUSE_PRICE_* setting defaults to
        0.0 (deliberately -- see .env.example: an unconfigured provider
        reporting $0 beats a silently guessed number). A spend figure
        built on those defaults would be structurally zero. Tokens are
        real whether or not anyone has configured a rate, so tokens are
        what this counts.

        getattr, not a direct call: `drain_usage` is an optional,
        duck-typed capability (see its docstring in llm/client.py), so
        StubClient and every hand-written test fake work unchanged.
        """
        drain = getattr(provider, "drain_usage", None)
        if drain is None:
            return
        prompt_tokens, completion_tokens = drain()
        if prompt_tokens:
            self._bump("llm_prompt_tokens", float(prompt_tokens))
        if completion_tokens:
            self._bump("llm_completion_tokens", float(completion_tokens))

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
        self._local.counters = {}
        return drained

    def set_node(self, node: Optional[str]) -> None:
        """Tag subsequent calls with the current graph node, so the trace and
        the llm.call log line show WHICH node issued each call.

        CALLED BY   every node function in agents/*.py that makes an LLM
                    call, right before calling router.complete_json(...) or
                    router.complete(...) — e.g. agents/planning.py's
                    classify_node does `router.set_node("classify")` first.

        D-101 also records the name on THIS object, not only on each
        provider, so ProviderChainExhausted can say which node was being
        served when the whole chain died. Stored on the same
        threading.local() the counters use, for the same reason given
        there: one router instance is shared process-wide, and the
        search_worker fan-out runs nodes on several threads at once.
        """
        self._local.node = node
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

    def close(self) -> None:
        """Release every provider's HTTP resources. Safe to call twice.

        CALLED BY   cli.py::main's finally block, alongside
                    close_checkpointer — see llm/client.py::
                    OpenAICompatibleClient.close for what actually leaks
                    without this.
        """
        for p in self.providers:
            closer = getattr(p, "close", None)
            if closer:
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001 — closing is best-effort
                    log_event(logger, "llm.close_failed", level=logging.WARNING,
                              provider=getattr(p, "name", "?"),
                              reason=type(exc).__name__)

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

        Guardrail G6 (P205 Phase 2): every provider in the chain gets the
        SAME s.llm_max_tokens generation cap, unlike the two timeouts
        above -- there's no equivalent reason to split it per-provider
        (a runaway generation is a property of the MODEL's own behavior
        continuing past its end-of-turn, not of network latency, so the
        primary/cloud distinction that justifies split timeouts doesn't
        apply here).
        """
        if s.llm_mode == "stub":
            return cls([StubClient(tracer=tracer)], s.llm_quality_threshold, tracer)

        chain: List[ChatClient] = [OpenAICompatibleClient(
            "primary", s.llm_primary_base_url, s.llm_primary_api_key,
            s.llm_primary_model, s.llm_primary_timeout_seconds, tracer,
            display_label=f"LOCAL PRIMARY ({s.llm_primary_model})",
            max_tokens=s.llm_max_tokens,
            # D-93: primary only -- see the setting's own comment in
            # config.py for why the cloud hops get no equivalent knob.
            context_tokens=s.llm_primary_context_tokens)]

        # See the module docstring for exactly what this tuple-of-tuples
        # loop with unpacking is doing. Each row here is one OPTIONAL
        # fallback provider; the loop body only actually adds it to `chain`
        # if its API key is non-empty.
        for name, base, key, model, label in (
            ("mistral", s.llm_mistral_base_url, s.llm_mistral_api_key,
             s.llm_mistral_model, f"MISTRAL ({s.llm_mistral_model})"),
            # D-114: named from settings, not hardwired. Everything
            # downstream -- log lines, telemetry, pricing, the D-111
            # health-check row -- keys off this name, so switching the
            # three URL/key/model settings to another OpenAI-compatible
            # provider now renames it everywhere instead of leaving
            # "gemini" on calls to something else.
            (s.llm_fallback_name, s.llm_fallback_base_url,
             s.llm_fallback_api_key, s.llm_fallback_model,
             f"{_display_label(s.llm_fallback_name)} ({s.llm_fallback_model})"),
        ):
            if key:
                chain.append(OpenAICompatibleClient(
                    name, base, key, model, s.llm_timeout_seconds, tracer,
                    display_label=label, max_tokens=s.llm_max_tokens))

        log_event(logger, "llm.chain_built", providers=[p.name for p in chain])
        return cls(chain, s.llm_quality_threshold, tracer)

    # -- internals ----------------------------------------------------------

    def _score_quality(self, provider: ChatClient, messages: List[Message],
                       answer: str, judge: ChatClient) -> float:
        """Score `answer`, log a rejection, and return the score itself.

        Renamed from `_passes_quality` (which returned a bare bool) as part
        of FIX-3: complete() now needs the NUMBER, not just pass/fail, so it
        can keep the best-scoring answer instead of the last one it happened
        to receive. The threshold comparison and the rejection log line are
        unchanged; only the return type is.

        CALLED BY   self.complete() below, ONLY when a further fallback
                    hop is available — see complete()'s docstring for why
                    it isn't worth checking on the last provider in the
                    chain. `judge` is always that NEXT provider (P2-11):
                    the caller passes self.providers[i + 1], never
                    `provider` itself — see evaluation/quality.py's module
                    docstring for why self-scoring was replaced (a real run
                    showed same-model self-scoring pass a report with
                    uncited sections).
        A judge that itself errors returns 1.0 (see evaluation.quality), so a
        flaky judge keeps the answer rather than cascading -- the gate
        exists to catch bad ANSWERS, not to punish a bad judging call.
        `provider` is kept as a parameter purely for the log line below
        (which provider's answer was rejected) — it is never passed to
        score_answer itself any more.

        P2-11 follow-up: a real run showed the judge itself can be
        unreachable (Gemini 429'd right after Mistral answered, one hop
        earlier in the same chain) — fail-open handled it correctly, but
        telemetry couldn't tell "judge said 1.0" from "judge never
        answered, defaulted to 1.0". The on_score_failed callback below
        bumps llm_quality_calls_failed ONLY on that fail-open path — a
        genuinely low score never touches this counter.
        """
        score = score_answer(
            judge, messages, answer,
            on_score_failed=lambda: self._bump("llm_quality_calls_failed"),
            # D-106: every real judgement, not only the rejections. An
            # accepted 0.61 says as much about where the threshold belongs
            # as a rejected 0.59 does, and only one of the two was ever
            # recorded.
            on_scored=lambda s, why: self._record_quality_score(
                s, why, provider.name, judge.name))
        # D-86: a judging call is a real provider request that
        # burns real tokens -- llm_quality_calls already counts
        # that it happened; this counts what it cost. Draining it
        # here also stops the judge's usage lingering on its
        # thread-local and being attributed to that provider's
        # NEXT call, when it is next reached as an answerer.
        self._bump_usage(judge)
        if score < self.quality_threshold:
            log_event(logger, "llm.quality_reject", provider=provider.name,
                      judge=judge.name, score=score, threshold=self.quality_threshold)
        return score

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
        # D-101: (provider_name, outcome) for every hop, in chain order.
        # Built alongside last_exc rather than reconstructed afterwards --
        # by the time the loop ends, only last_exc survives and it cannot
        # say what the earlier providers did.
        outcomes: List[Tuple[str, str]] = []
        # enumerate(self.providers) — see the module docstring — gives us
        # both the position `i` (0, 1, 2...) and the `provider` object on
        # each pass, in the fixed order the chain was built in.
        for i, provider in enumerate(self.providers):
            # D-93: checked BEFORE llm_provider_calls is bumped -- a hop
            # never attempted must not be counted as an attempt. Its own
            # llm_context_skips counter records it instead.
            if (i + 1 < len(self.providers)
                    and self._skips_for_context(provider, messages)):
                outcomes.append((provider.name, "skipped_for_context"))
                continue
            self._bump("llm_provider_calls")  # P2-07: one real attempt, win or lose
            try:
                result = provider.complete_json(messages)
                self._bump_usage(provider)  # D-86
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
                outcomes.append((provider.name, type(exc).__name__))  # D-101
                nxt = (self.providers[i + 1].name
                       if i + 1 < len(self.providers) else None)
                if nxt is not None:
                    self._bump("llm_fallback_hops")  # P2-07: a real hop, not the last dead end
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=nxt, reason=type(exc).__name__, mode="json")
                lf.event(run_id_var.get(), "llm.fallback",
                        metadata={"from_provider": provider.name, "to_provider": nxt,
                                  "reason": type(exc).__name__, "mode": "json"})
        # If we reach this line, every provider in the loop above raised.
        # `assert last_exc is not None` is a sanity check for a human reader
        # (and for tools like mypy) that this line is only reachable when
        # last_exc has actually been set — it would only fail if
        # self.providers were somehow empty, which __init__ already forbids.
        assert last_exc is not None
        # D-101: wrapped, not re-raised bare -- see ProviderChainExhausted
        # above for why the last provider's own exception cannot carry
        # this. `from last_exc` keeps it as __cause__, so the traceback
        # still shows the real failure underneath.
        raise ProviderChainExhausted(self._node, outcomes, "json") from last_exc

    def complete(self, messages: List[Message]) -> str:
        """Free-text call. Step down the chain on error OR low quality.

        CALLED BY   compiler_node (agents/compilation.py) -- the ONLY node
                    in the whole codebase that makes a free-text (as
                    opposed to JSON) LLM call, which is why this is the
                    only call path where the quality gate below ever runs.
        RETURNS     the first answer that both succeeds AND clears the
                    quality gate (an immediate return, mid-loop); failing
                    that, the BEST-scoring candidate any provider produced
                    (S-3's `_best`, below); or raises if the entire chain
                    errored with nothing to fall back to.

        Two phases (S-3, previously one loop with three exit mechanisms --
        continue/break/a bottom return -- and a fabricated score standing
        in for "unscored, accept anyway"):

          1. Walk the chain once, collecting (name, answer, score) for
             every candidate that does NOT win an immediate return. score
             is None in exactly one case: the last provider succeeded with
             no earlier candidate to judge it against, so it is accepted
             unscored, same as before.
          2. `_best(candidates)` picks the winner -- a small, pure,
             unit-testable function, not inline running-max state threaded
             through every branch.

        FIX-3 policy, unchanged: the first answer that clears the quality
        threshold wins immediately; otherwise the BEST-scoring candidate
        wins, not the last one to arrive (run p205.211's defect: a
        732-character fragment shipped over a complete, 10,103-character
        report because the last provider in the chain was exempt from
        scoring). The last provider is judged too, by the PREVIOUS
        provider (P2-11's never-self-judge rule), but ONLY when an
        earlier candidate already exists to compare it against -- so the
        common path (nothing rejected before the last provider) still
        costs no extra scoring call.
        """
        last_exc: Optional[Exception] = None
        outcomes: List[Tuple[str, str]] = []                     # D-101
        # (name, answer, score) for every candidate that did not win an
        # immediate return. Fed to _best() once the chain is exhausted.
        candidates: List[Tuple[str, str, Optional[float]]] = []

        for i, provider in enumerate(self.providers):
            if (i + 1 < len(self.providers)                      # D-93
                    and self._skips_for_context(provider, messages)):
                outcomes.append((provider.name, "skipped_for_context"))
                continue
            self._bump("llm_provider_calls")  # P2-07: one real attempt, win or lose
            try:
                answer = provider.complete(messages)
                # D-86: drained here, BEFORE any quality scoring
                # below, so the judge's own call (also counted, in
                # _score_quality) can never be attributed to the
                # provider that produced the answer.
                self._bump_usage(provider)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                outcomes.append((provider.name, type(exc).__name__))  # D-101
                nxt = (self.providers[i + 1].name
                       if i + 1 < len(self.providers) else None)
                if nxt is not None:
                    self._bump("llm_fallback_hops")
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=nxt, reason=type(exc).__name__, mode="text")
                lf.event(run_id_var.get(), "llm.fallback",
                        metadata={"from_provider": provider.name, "to_provider": nxt,
                                  "reason": type(exc).__name__, "mode": "text"})
                continue

            has_next = i + 1 < len(self.providers)

            if has_next:
                # P2-11: the judge is always the NEXT provider in the
                # chain, never `provider` itself.
                self._bump("llm_quality_calls")  # P2-07: a real judging call
                score = self._score_quality(provider, messages, answer,
                                            judge=self.providers[i + 1])
                if score >= self.quality_threshold:
                    if i > 0:
                        log_event(logger, "llm.served_by_fallback",
                                  provider=provider.name, position=i, mode="text")
                    return answer
                candidates.append((provider.name, answer, score))
                self._bump("llm_fallback_hops")
                log_event(logger, "llm.fallback", from_provider=provider.name,
                          to_provider=self.providers[i + 1].name,
                          reason="low_quality", mode="text")
                lf.event(run_id_var.get(), "llm.fallback",
                        metadata={"from_provider": provider.name,
                                  "to_provider": self.providers[i + 1].name,
                                  "reason": "low_quality", "mode": "text"})
                continue

            # Last provider in the chain -- no further hop to fall back to.
            if candidates:
                # An earlier candidate exists to judge this one against
                # (judged by the PREVIOUS provider, so P2-11's
                # never-self-judge rule still holds).
                self._bump("llm_quality_calls")
                score = self._score_quality(provider, messages, answer,
                                            judge=self.providers[i - 1])
                prior_name, _prior_answer, prior_score = self._best(candidates)
                if score > prior_score:
                    candidates.append((provider.name, answer, score))
                else:
                    log_event(logger, "llm.last_provider_worse",
                              provider=provider.name, score=score,
                              kept_provider=prior_name, kept_score=prior_score)
            else:
                # Nothing to compare against: accept it unscored, exactly
                # as before. Scoring here could only discard the sole
                # answer for nothing in return.
                candidates.append((provider.name, answer, None))
            break

        if candidates:
            best_name, best_answer, best_score = self._best(candidates)
            effective_score = (best_score if best_score is not None
                               else self.quality_threshold)
            if effective_score < self.quality_threshold:
                log_event(logger, "llm.chain_exhausted_low_quality",
                          provider=best_name, score=effective_score)
            elif best_name != self.providers[0].name:
                log_event(logger, "llm.served_by_fallback", provider=best_name,
                          position=[p.name for p in self.providers].index(best_name),
                          mode="text")
            return best_answer
        assert last_exc is not None
        # D-101. Reached ONLY when no provider returned anything at all --
        # a chain where one provider answered, however badly, returns that
        # answer above (run p205.254-check's fourth compile did exactly
        # that with a 0.1-scored report). This line is the fifth compile:
        # primary HTTPStatusError, mistral ReadTimeout, gemini
        # TruncatedGenerationError, nothing to ship.
        raise ProviderChainExhausted(self._node, outcomes, "text") from last_exc

    @staticmethod
    def _best(candidates: List[Tuple[str, str, Optional[float]]]
             ) -> Tuple[str, str, Optional[float]]:
        """Pick the highest-scoring candidate. 6 lines, pure, testable
        without any provider (S-3).

        Ties keep the EARLIER candidate -- a running max only replaces on
        a strict `>`, matching the original inline comparison this
        replaces. None ranks last: it only ever appears as the SOLE
        candidate (the unscored last-provider-with-nothing-to-compare
        case in complete() above), so nothing ever actually has to lose
        to it.
        """
        best = candidates[0]
        for c in candidates[1:]:
            c_score = c[2] if c[2] is not None else -1.0
            best_score = best[2] if best[2] is not None else -1.0
            if c_score > best_score:
                best = c
        return best
