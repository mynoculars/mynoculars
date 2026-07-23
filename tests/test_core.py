"""
tests/test_core.py — Representative tests per project brief.

Covers: reducers (concurrency math), worker contract enforcement, routing
decisions, LLM fallback policy, memory decay, and one offline end-to-end
run of the whole graph. Not exhaustive by design — each test documents one
load-bearing behavior.
"""

import pytest

from research_agent.config import Settings
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import decay_factor
from research_agent.orchestration.contracts import WorkerContractViolation, validated_worker
from research_agent.orchestration.graph import (dispatch_tasks, route_after_critique,
                                                route_convergence)
from research_agent.state import (Evidence, ResearchState, SearchTask, Volatility,
                                  merge_counters, merge_failed_keys, merge_key_sets)

# ---------------------------------------------------------------------------
# Reducers — the concurrency safety math (D-5/D-16/D-19)
# ---------------------------------------------------------------------------


def test_key_set_union_is_order_independent():
    assert merge_key_sets({"a"}, {"b"}) == merge_key_sets({"b"}, {"a"}) == {"a", "b"}


def test_failed_keys_keep_deepest_failure():
    # D-16: conservative merge — never permit an earlier retry than any
    # worker observed.
    assert merge_failed_keys({"k": 1}, {"k": 3}) == {"k": 3}
    assert merge_failed_keys({"k": 3}, {"k": 1}) == {"k": 3}


def test_counters_merge_additively():
    assert merge_counters({"llm_calls": 2}, {"llm_calls": 1, "x": 1}) == {
        "llm_calls": 3, "x": 1}


# ---------------------------------------------------------------------------
# Worker contract (D-15) — deterministic failure instead of concurrent one
# ---------------------------------------------------------------------------


def test_validated_worker_rejects_illegal_keys():
    @validated_worker
    def bad_worker(payload):
        return {"final_report": "workers must not write this"}

    with pytest.raises(WorkerContractViolation):
        bad_worker(None)


def test_validated_worker_passes_legal_keys():
    @validated_worker
    def good_worker(payload):
        return {"counters": {"search_calls": 1}}

    assert good_worker(None) == {"counters": {"search_calls": 1}}


# ---------------------------------------------------------------------------
# Routing (D-1/D-14/D-22)
# ---------------------------------------------------------------------------

_S = Settings(_env_file=None, llm_mode="stub", max_depth=2, max_revisions=2)


def _state(**kw) -> ResearchState:
    return ResearchState(raw_query="q", **kw)


def test_dispatch_empty_backlog_falls_through_to_compiler():
    # D-1: empty Send list must never silently halt the graph.
    assert dispatch_tasks(_state()) == "compiler"


def test_dispatch_fans_out_one_send_per_task():
    t = SearchTask(key="g1::x", query="x", goal_id="g1")
    sends = dispatch_tasks(_state(pending_tasks=[t]))
    assert isinstance(sends, list) and len(sends) == 1
    assert sends[0].node == "search_worker"


def test_convergence_compiles_on_recall_target():
    assert route_convergence(_state(recall_score=0.9), _S) == "compiler"


def test_convergence_compiles_on_depth_exhaustion():
    assert route_convergence(_state(recall_score=0.1, iteration_depth=2), _S) == "compiler"


def test_convergence_expands_otherwise():
    assert route_convergence(_state(recall_score=0.1, iteration_depth=1), _S) == "gap_generator"


def test_critique_routes_pass_to_memory_writer():
    assert route_after_critique(_state(critique_passed=True), _S) == "memory_writer"


def test_critique_routes_fail_with_budget_to_rewrite():
    assert route_after_critique(_state(critique_passed=False, revision_count=1),
                                _S) == "compiler"


def test_critique_exhausted_skips_memory():
    # E4 stub path: a report that failed its own bar never feeds memory.
    assert route_after_critique(_state(critique_passed=False, revision_count=2),
                                _S) == "telemetry"


# ---------------------------------------------------------------------------
# LLM fallback policy
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
# Memory decay (D-24)
# ---------------------------------------------------------------------------


def test_stable_facts_never_decay():
    assert decay_factor(365, Volatility.STABLE, 90, 14) == 1.0


def test_volatile_decays_faster_than_semi_stable():
    semi = decay_factor(30, Volatility.SEMI_STABLE, 90, 14)
    vol = decay_factor(30, Volatility.VOLATILE, 90, 14)
    assert vol < semi < 1.0


def test_half_life_is_half():
    assert decay_factor(14, Volatility.VOLATILE, 90, 14) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End to end, fully offline
# ---------------------------------------------------------------------------


def test_full_graph_runs_offline(graph, settings):
    """The whole workflow: plan -> gather (fan-out) -> compile -> critique
    -> telemetry, on stub LLM + fake tool, no services, no network."""
    result = graph.invoke(
        ResearchState(raw_query="Compare Redis and Memcached for session caching"),
        config={"configurable": {"thread_id": "test-e2e"},
                "recursion_limit": settings.recursion_limit},
    )
    assert result["final_report"]
    tele = result["telemetry"]
    assert tele["goals"] == 2                # stub composes g1, g2
    assert tele["search_calls"] == 2         # one worker per stub task
    assert tele["recall"] == 1.0             # fake tool covers both goals
    assert tele["critique_passed"] is True
    assert tele["iterations"] >= 1


# ---------------------------------------------------------------------------
# Fallback CHAIN (primary -> Mistral -> Gemini), N-provider router
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
        import json
        if messages and "TASK=quality" in messages[-1]["content"]:
            return json.dumps({"score": 0.2 if self.behavior == "low" else 0.9})
        if self.behavior == "error":
            raise RuntimeError(f"{self.name} down")
        return f"answer from {self.name}"

    def complete_json(self, messages, temperature=0.0):
        import json
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
    import json

    class _Json:
        name = "j"
        def complete(self, m, temperature=0.2): return json.dumps({"ok": True})
        def complete_json(self, m, temperature=0.0): return {"ok": True}

    chain = FallbackRouter([_Named("primary", "error"), _Json()],
                           quality_threshold=0.6)
    assert chain.complete_json([{"role": "user", "content": "x"}]) == {"ok": True}


# ---------------------------------------------------------------------------
# Debug tracer (--debug / DEBUG_TRACE)
# ---------------------------------------------------------------------------


def test_tracer_records_and_flushes(tmp_path):
    from research_agent.tracing import Tracer
    t = Tracer("run-test", log_dir=str(tmp_path))
    t.record_llm("LOCAL PRIMARY (x)", "classify",
                 [{"role": "user", "content": "hello"}], '{"ok":1}', 10, 3, 1.5)
    t.record_retrieval("QDRANT (dense)", "redis vs memcached",
                       [{"title": "doc", "similarity": 0.9}])
    path = t.flush()
    assert path is not None
    text = open(path, encoding="utf-8").read()
    assert "RETRIEVED FROM LOCAL PRIMARY (X)" in text
    assert "node=classify" in text
    assert "RETRIEVED FROM QDRANT (DENSE)" in text
    assert "redis vs memcached" in text


def test_null_tracer_is_noop(tmp_path):
    from research_agent.tracing import NullTracer
    t = NullTracer()
    assert t.enabled is False
    t.record_llm("x", None, [], "", None, None, 0.0)
    t.record_retrieval("x", "q", [])
    assert t.flush() is None
