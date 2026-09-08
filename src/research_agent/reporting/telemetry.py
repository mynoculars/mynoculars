"""
reporting/telemetry.py -- the counter-only half of the telemetry record
(D-146).

telemetry_node was 531 lines, of which roughly 350 were one dict literal
interleaved with the comments explaining each field. Adding a field meant
editing that literal, and nothing in it could be tested without running
the whole node.

The three functions below are the part that reads NOTHING but
state.counters -- pure dict-to-dict, independently testable, and the
reason the literal in telemetry_node is now roughly half what it was. The
state-dependent half (evidence, grounding, the report itself) stays in
the node deliberately: those fields are computed alongside the WARNING
side effects that read the same values, and separating them would mean
computing several of them twice. That is a real remaining limitation and
it is stated here rather than discovered later.

D-12 is unchanged by any of this: these add up numbers other nodes
recorded, and invent none.

CALLED BY   agents/compilation.py::telemetry_node, which merges all three
            into the telemetry dict with **.
"""

from typing import Any, Dict

def llm_metrics(c: Dict[str, float]) -> Dict[str, Any]:
    """Provider volume, judge activity and real token cost (P2-07, D-86).

    llm_node_calls counts NODE executions that made an LLM call -- a node
    that fell through two fallback hops is still one. llm_provider_calls is
    the actual request volume. llm_context_skips (D-93) and
    llm_disabled_skips (D-130) are hops NOT made, and are excluded from
    llm_provider_calls, so read them against it.

    Tokens are additive across the run, which is correct here: unlike the
    compile-scoped guardrail counts, every token genuinely was spent.
    llm_provider_calls cannot distinguish three cheap classify calls from
    three 7,000-token compiles; these can.
    """
    prompt = int(c.get("llm_prompt_tokens", 0))
    completion = int(c.get("llm_completion_tokens", 0))
    return {
        "llm_node_calls": int(c.get("llm_node_calls", 0)),
        "llm_provider_calls": int(c.get("llm_provider_calls", 0)),
        "llm_fallback_hops": int(c.get("llm_fallback_hops", 0)),
        "llm_quality_calls": int(c.get("llm_quality_calls", 0)),
        "llm_quality_calls_failed": int(c.get("llm_quality_calls_failed", 0)),
        "llm_quality_rejections": int(c.get("llm_quality_rejections", 0)),
        "llm_prompt_tokens": prompt,
        "llm_completion_tokens": completion,
        "llm_total_tokens": prompt + completion,
        "llm_context_skips": int(c.get("llm_context_skips", 0)),
        # D-153: llm_context_skips, broken down by which provider was
        # skipped. Empty on every run where nothing was skipped, and on
        # every run before this existed.
        "context_skips_by_provider": {
            key[len("llm_context_skipped_"):]: int(value)
            for key, value in sorted(c.items())
            if key.startswith("llm_context_skipped_") and value
        },
        "llm_disabled_skips": int(c.get("llm_disabled_skips", 0)),
        # D-191: the pacing counters, which the router has been
        # accumulating and this function has never emitted.
        #
        # THE THIRD PLACE D-183 WAS MISSING. That decision shipped in
        # three parts -- the router's retry loop, the client's parser,
        # and these fields -- and only the first was ever written. D-185
        # supplied the client half; the counters still stopped at
        # `state.counters`, so a run that DID wait out a 429 reported
        # nothing about it in the telemetry block, the agent_runs row's
        # telemetry JSONB, or `analyze_runs.py`. README's own telemetry
        # table documents all three of the retry_after fields as
        # telemetry, and reading them back returned `null`.
        #
        # It stayed invisible because it only shows once a provider
        # actually 429s or a hop delay is configured -- so a run that
        # never hit either is indistinguishable from a run whose
        # reporting was missing, which is the same shape of silence
        # D-185 itself had.
        #
        # `llm_retry_after_seconds_total` and `llm_hop_delay_seconds_total`
        # are DURATIONS, and state.py::merge_counters' standing rule is
        # "monotonic countables only, never durations". They are legal
        # here for the reason that rule gives: the objection is to
        # merging two PARALLEL nodes' elapsed times into a number that
        # measures nothing. These are seconds this process deliberately
        # SLEPT, which is monotonic and additive -- two threads that each
        # waited 30s did spend 60 seconds of provider cooldown between
        # them. Rounded to 3dp because a float sum of sleeps otherwise
        # reports 35.360000000000004.
        "llm_retry_after_sleeps": int(c.get("llm_retry_after_sleeps", 0)),
        "llm_retry_after_seconds_total": round(
            float(c.get("llm_retry_after_seconds_total", 0.0)), 3),
        "llm_retry_after_skipped_too_long": int(
            c.get("llm_retry_after_skipped_too_long", 0)),
        # D-184's pair, absent for the same reason and fixed in the same
        # pass. A fixed hop delay is opt-in (LLM_HOP_DELAY_SECONDS
        # defaults to 0), so on a default deployment both stay 0 -- which
        # is the point: 0 now means "no pause happened", where before it
        # meant "this run cannot tell you".
        "llm_hop_delay_sleeps": int(c.get("llm_hop_delay_sleeps", 0)),
        "llm_hop_delay_seconds_total": round(
            float(c.get("llm_hop_delay_seconds_total", 0.0)), 3),
    }


def retrieval_metrics(c: Dict[str, float]) -> Dict[str, Any]:
    """Per-leg retrieval attempts and D-38 ladder outcomes (P2-07, D-87).

    retrieval_leg_unavailable counts DEGRADED legs specifically -- a store
    that was unreachable when checked -- not legs that legitimately
    returned zero hits; see retrieval/hybrid.py::_bump_retrieval_counts.

    tier_answers says which tier of the ladder actually answered. Read it
    against corpus_recall: {"corpus": 6} with corpus_recall 1.0 is a
    healthy corpus run, {"web": 6} with corpus_recall 0.0 is the
    p205.246-check shape -- one field instead of three inferred ones.
    """
    return {
        "retrieval_dense_calls": int(c.get("retrieval_dense_calls", 0)),
        "retrieval_keyword_calls": int(c.get("retrieval_keyword_calls", 0)),
        "retrieval_leg_unavailable": int(c.get("retrieval_leg_unavailable", 0)),
        # D-150: keyword hits dropped because the dense leg's calibrated
        # floor had already rejected every candidate for that query -- i.e.
        # off-topic corpus documents that used to reach the compile prompt
        # through the ungated leg. Read against
        # retrieval_dropped_by_floor: both nonzero is the floor working on
        # a query the corpus does not cover.
        "retrieval_keyword_dropped_off_topic": int(
            c.get("retrieval_keyword_dropped_off_topic", 0)),
        "tier_answers": {
            key[len("chain_answered_"):]: int(value)
            for key, value in sorted(c.items())
            if key.startswith("chain_answered_") and value
        },
        "chain_tier_failures": int(c.get("chain_tier_failed", 0)),
        # Tasks that reached the TERMINAL model tier and got something
        # back that cleared neither the coverage floor nor the topical
        # gate. Deliberately not folded into tier_answers above: the
        # ladder returned that evidence (it is real context for the
        # compiler, and evidence_by_source still counts it), but nothing
        # ANSWERED, and rounding the two together is what made the
        # offline L1 demo report tier_answers {"model": 2} while its own
        # confidence verdict read UNRELIABLE.
        #
        # A nonzero value here beside an empty tier_answers is the
        # honest shape of "every tier was tried and none of them
        # sufficed" -- distinguishable, now, from a run that never
        # reached tier 5 at all.
        "chain_model_insufficient": int(c.get("chain_model_insufficient", 0)),
        "chain_exhausted": int(c.get("chain_exhausted", 0)),
    }


def run_metrics(c: Dict[str, float]) -> Dict[str, Any]:
    """Whole-run tallies every node contributed to.

    producer_rejects (P2-06) counts malformed goal/task items the LLM
    returned that were dropped rather than crashing the run.

    critique_notes_dismissed (D-155) counts critic notes the evidence
    itself refuted -- every figure the note disputed was present in the
    evidence the critic was shown. It is reported so that a run whose
    verdict was resolved deterministically is never indistinguishable
    from one the critic passed on its own; a persistently nonzero value
    is a signal to revisit templates.critique, not to widen this check.
    """
    return {
        "search_calls": int(c.get("search_calls", 0)),
        "search_failures": int(c.get("search_failures", 0)),
        "memory_hits": int(c.get("memory_hits", 0)),
        "memory_writes": int(c.get("memory_writes", 0)),
        "revision_cycles": int(c.get("revision_cycles", 0)),
        "producer_rejects": int(c.get("producer_rejects", 0)),
        "critique_notes_dismissed": int(c.get("critique_notes_dismissed", 0)),
        # D-181: nonzero means the critic wrote entries recording that a
        # claim IS supported, which the contract forbids. Meant to sit at
        # zero; a run where it does not is a prompt to fix.
        "critique_affirmations_dropped": int(
            c.get("critique_affirmations_dropped", 0)),
    }
