"""
tests/unit/test_agents_gathering.py — agents/gathering.py's
build_merger_node.

Covers ONLY the P2-12 semantic contradiction gate: off by default
(marker-only behavior unchanged), on calls an LLM-backed detector but
only for goals with 2+ evidence items (cost control), and fails open if
the detector itself errors. Does NOT cover the rest of the gather loop
(dispatch/coverage/gap-generation) — those are exercised through full
graph runs in tests/integration/test_graph_end_to_end.py.
"""

import logging

from research_agent.agents.gathering import build_merger_node
from research_agent.config import Settings
from research_agent.state import Evidence, Goal, ResearchState, Volatility


class _FakeContradictionRouter:
    """Minimal fake satisfying merger_node's actual usage of `router`:
    set_node(name), complete_json(messages), drain_counters() — nothing
    else is called on it. Records whether it was ever invoked, so tests can
    assert the gate genuinely skipped the LLM call when it should have."""

    def __init__(self, contested_goal_ids=None, raise_error=False):
        self._contested = contested_goal_ids or []
        self._raise = raise_error
        self.calls = 0

    def set_node(self, node):
        pass

    def complete_json(self, messages):
        self.calls += 1
        if self._raise:
            raise RuntimeError("detector is down")
        return {"contested_goal_ids": self._contested}

    def drain_counters(self):
        return {"llm_provider_calls": 1.0}


def _settings(contradiction_detection_enabled: bool) -> Settings:
    return Settings(_env_file=None, llm_mode="stub",
                    contradiction_detection_enabled=contradiction_detection_enabled,
                    qdrant_url="http://127.0.0.1:1",
                    postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


def _state_with_two_goals_one_multi_evidence(contradicts_marker=None):
    """g1 has 2 evidence items (the "multi-evidence" case the LLM path
    should actually fire for); g2 has only 1 (should never trigger a call
    on its own)."""
    goals = [Goal(goal_id="g1", description="a"), Goal(goal_id="g2", description="b")]
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="corpus", content="claim A",
                 score=0.9, volatility=Volatility.SEMI_STABLE, contradicts=contradicts_marker),
        Evidence(task_key="t2", goal_id="g1", source="corpus", content="claim B",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
        Evidence(task_key="t3", goal_id="g2", source="corpus", content="claim C",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
    ]
    return ResearchState(raw_query="q", goals=goals, evidence=evidence)


def test_gate_off_preserves_original_marker_only_behavior():
    """Default (off): merger_node must behave EXACTLY as it did before
    P2-12 — only an explicit Evidence.contradicts marker sets `contested`,
    and the LLM is never consulted."""
    settings = _settings(contradiction_detection_enabled=False)
    router = _FakeContradictionRouter(contested_goal_ids=["g1"])  # would fire if consulted
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence(contradicts_marker=None)
    result = node(state)

    assert router.calls == 0, "gate is off — the detector must never be called"
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is False
    assert goals_by_id["g2"].contested is False
    assert result["counters"]["contradictions_flagged"] == 0.0


def test_gate_off_still_honors_explicit_contradicts_marker():
    """The pre-P2-12 marker path is untouched when the gate is off — this
    is what proves the change is additive, not a rewrite."""
    settings = _settings(contradiction_detection_enabled=False)
    router = _FakeContradictionRouter()
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence(contradicts_marker="t2")
    result = node(state)

    assert router.calls == 0
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is True
    assert goals_by_id["g2"].contested is False


def test_gate_on_calls_detector_and_marks_only_contested_goal():
    settings = _settings(contradiction_detection_enabled=True)
    router = _FakeContradictionRouter(contested_goal_ids=["g1"])
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence()
    result = node(state)

    assert router.calls == 1
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is True
    assert goals_by_id["g2"].contested is False  # only 1 evidence item, never even asked about
    assert result["counters"]["contradictions_flagged"] == 1.0
    assert result["counters"]["llm_node_calls"] == 1.0


def test_gate_on_skips_the_llm_call_when_no_goal_has_multiple_evidence_items():
    """Early-exit: a goal with 0 or 1 evidence items can't contradict
    itself, so if NO goal qualifies, the detector must never be invoked —
    this is the cost-control path, not just an optimization detail."""
    settings = _settings(contradiction_detection_enabled=True)
    router = _FakeContradictionRouter(contested_goal_ids=["g1"])
    node = build_merger_node(router, settings)

    goals = [Goal(goal_id="g1", description="a")]
    evidence = [Evidence(task_key="t1", goal_id="g1", source="corpus",
                         content="claim A", score=0.9, volatility=Volatility.SEMI_STABLE)]
    state = ResearchState(raw_query="q", goals=goals, evidence=evidence)
    result = node(state)

    assert router.calls == 0
    assert result["goals"][0].contested is False


def test_gate_on_fails_open_when_detector_errors(caplog):
    """A broken detector must never take the run down — same fail-open
    posture evaluation/quality.py's score_answer already uses."""
    settings = _settings(contradiction_detection_enabled=True)
    router = _FakeContradictionRouter(raise_error=True)
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence()
    with caplog.at_level(logging.WARNING):
        result = node(state)

    assert router.calls == 1
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is False  # fails open -> nothing contested
    assert any("merger.contradiction_detection_failed" in r.message for r in caplog.records)
