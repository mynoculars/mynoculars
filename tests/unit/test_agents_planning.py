"""
tests/unit/test_agents_planning.py -- agents/planning.py, the four
Plan-phase nodes: classify, memory_retrieve, goal_manager, task_expander.

Covers: classify_node's counter/classification write; memory_retrieve_node's
evidence accumulation; goal_manager_node's P2-06 malformed-goal validation,
the D-21 zero-goals path and its D-23 escalation-budget interaction, and
human_guidance consumption; task_expander_node's cap_and_filter wiring and
mcp_enabled tool-hint gating. Previously exercised only indirectly through
full-graph integration runs (tests/integration/test_*_end_to_end.py) --
this file tests each node's own logic directly, with fakes.
"""


from research_agent.agents.planning import (build_classify_node,
                                             build_goal_manager_node,
                                             build_memory_retrieve_node,
                                             build_task_expander_node)
from research_agent.config import Settings
from research_agent.state import Evidence, Goal, ResearchState


class _FakeRouter:
    """Minimal fake satisfying every planning node's actual usage of
    `router`: set_node(name), complete_json(messages), drain_counters()."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    def set_node(self, node):
        pass

    def complete_json(self, messages):
        self.calls += 1
        return self._response

    def drain_counters(self):
        return {"llm_provider_calls": 1.0}


class _FakeMemory:
    def __init__(self, items=None):
        self._items = items or []

    def retrieve(self, query):
        return self._items


def _settings(**overrides):
    return Settings(_env_file=None, llm_mode="stub", **overrides)


# ---------------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------------


def test_classify_node_writes_the_classification_result():
    router = _FakeRouter({"intent": "Comparison", "confidence": 0.9})
    node = build_classify_node(router)
    state = ResearchState(raw_query="Compare Redis and Memcached")
    update = node(state)
    assert update["classification"] == {"intent": "Comparison", "confidence": 0.9}


def test_classify_node_counts_exactly_one_node_call():
    router = _FakeRouter({"intent": "Comparison"})
    node = build_classify_node(router)
    update = node(ResearchState(raw_query="q"))
    assert update["counters"]["llm_node_calls"] == 1


# ---------------------------------------------------------------------------
# memory_retrieve_node
# ---------------------------------------------------------------------------


def test_memory_retrieve_node_returns_recalled_items_as_evidence():
    recalled = [Evidence(task_key="m1", goal_id="g1", source="memory",
                         content="past finding", score=0.8)]
    node = build_memory_retrieve_node(_FakeMemory(recalled))
    update = node(ResearchState(raw_query="q"))
    assert update["evidence"] == recalled
    assert update["counters"]["memory_hits"] == 1


def test_memory_retrieve_node_handles_an_empty_memory_store():
    node = build_memory_retrieve_node(_FakeMemory([]))
    update = node(ResearchState(raw_query="q"))
    assert update["evidence"] == []
    assert update["counters"]["memory_hits"] == 0


# ---------------------------------------------------------------------------
# goal_manager_node
# ---------------------------------------------------------------------------


def test_goal_manager_node_composes_goals_from_a_well_formed_response():
    router = _FakeRouter({"goals": [
        {"goal_id": "g1", "description": "compare throughput"},
        {"goal_id": "g2", "description": "compare latency"},
    ]})
    node = build_goal_manager_node(router, _settings())
    update = node(ResearchState(raw_query="q", classification={"intent": "Comparison"}))
    assert [g.goal_id for g in update["goals"]] == ["g1", "g2"]
    assert update["counters"]["composed_goals"] == 2


def test_goal_manager_node_drops_a_malformed_goal_and_counts_the_reject():
    """P2-06: a goal missing goal_id/description must be dropped and
    counted, not raise a KeyError that aborts the whole run."""
    router = _FakeRouter({"goals": [
        {"goal_id": "g1", "description": "compare throughput"},
        {"goal_id": "", "description": "missing id"},
        {"description": "missing goal_id key entirely"},
    ]})
    node = build_goal_manager_node(router, _settings())
    update = node(ResearchState(raw_query="q", classification={"intent": "Comparison"}))
    assert len(update["goals"]) == 1
    assert update["counters"]["producer_rejects"] == 2


def test_goal_manager_node_zero_goals_sets_planning_error_not_an_exception():
    """D-21: zero goals is a legal, expected output -- never an
    exception."""
    router = _FakeRouter({"goals": []})
    node = build_goal_manager_node(router, _settings(hitl_enabled=False))
    update = node(ResearchState(raw_query="q", classification={"intent": "Comparison"}))
    assert update["goals"] == []
    assert "planning_error" in update
    assert "escalation_trigger" not in update  # HITL off -> stub path, not E1


def test_goal_manager_node_raises_e1_when_zero_goals_and_hitl_enabled():
    router = _FakeRouter({"goals": []})
    node = build_goal_manager_node(router, _settings(hitl_enabled=True, max_escalations=2))
    update = node(ResearchState(raw_query="q", classification={"intent": "Comparison"}))
    assert update["escalation_trigger"] == "E1"


def test_goal_manager_node_suppresses_e1_once_the_escalation_budget_is_spent():
    """D-23 bound: escalation_allowed() folds in the per-run review
    budget -- once spent, no further E1 is raised even with HITL on."""
    router = _FakeRouter({"goals": []})
    node = build_goal_manager_node(router, _settings(hitl_enabled=True, max_escalations=1))
    state = ResearchState(raw_query="q", classification={"intent": "Comparison"},
                          escalation_history=[{"trigger": "E1", "action": "redirect"}])
    update = node(state)
    assert "escalation_trigger" not in update


def test_goal_manager_node_consumes_human_guidance():
    """human_guidance must never leak into a later, unrelated call --
    goal_manager_node always resets it to empty after reading it."""
    router = _FakeRouter({"goals": [{"goal_id": "g1", "description": "d"}]})
    node = build_goal_manager_node(router, _settings())
    state = ResearchState(raw_query="q", classification={"intent": "Comparison"},
                          human_guidance="focus on wars and conflicts")
    update = node(state)
    assert update["human_guidance"] == ""


def test_goal_manager_node_passes_memory_hints_to_the_prompt(monkeypatch):
    """Memory evidence recalled before goal composition must actually
    reach the prompt -- this is the whole point of running
    memory_retrieve BEFORE goal_manager."""
    from research_agent.prompts import templates

    seen = {}
    original = templates.compose_goals

    def spy(query, intent, hints, guidance=""):
        seen["hints"] = hints
        return original(query, intent, hints, guidance=guidance)

    monkeypatch.setattr(templates, "compose_goals", spy)
    router = _FakeRouter({"goals": [{"goal_id": "g1", "description": "d"}]})
    node = build_goal_manager_node(router, _settings())
    state = ResearchState(
        raw_query="q", classification={"intent": "Comparison"},
        evidence=[Evidence(task_key="m1", goal_id="g0", source="memory",
                           content="a past finding worth noting", score=0.8)])
    node(state)
    assert seen["hints"] == ["a past finding worth noting"]


# ---------------------------------------------------------------------------
# task_expander_node
# ---------------------------------------------------------------------------


def test_task_expander_node_produces_tasks_from_a_well_formed_response():
    router = _FakeRouter({"tasks": [
        {"goal_id": "g1", "query": "redis throughput benchmarks", "priority": 1},
    ]})
    node = build_task_expander_node(router, _settings(max_fanout=6))
    state = ResearchState(raw_query="q", goals=[Goal(goal_id="g1", description="d")])
    update = node(state)
    assert len(update["pending_tasks"]) == 1
    assert update["pending_tasks"][0].goal_id == "g1"


def test_task_expander_node_caps_at_max_fanout():
    """D-13: the producer caps at max_fanout, not the dispatcher."""
    router = _FakeRouter({"tasks": [
        {"goal_id": "g1", "query": f"query {i}", "priority": i} for i in range(10)
    ]})
    node = build_task_expander_node(router, _settings(max_fanout=3))
    state = ResearchState(raw_query="q", goals=[Goal(goal_id="g1", description="d")])
    update = node(state)
    assert len(update["pending_tasks"]) == 3


def test_task_expander_node_rejects_a_malformed_task_and_counts_it():
    router = _FakeRouter({"tasks": [
        {"goal_id": "g1", "query": "valid query", "priority": 1},
        {"goal_id": "g1", "query": "", "priority": 1},  # empty query
    ]})
    node = build_task_expander_node(router, _settings(max_fanout=6))
    state = ResearchState(raw_query="q", goals=[Goal(goal_id="g1", description="d")])
    update = node(state)
    assert len(update["pending_tasks"]) == 1
    assert update["counters"]["producer_rejects"] == 1


def test_task_expander_node_does_not_offer_the_mcp_hint_when_mcp_is_disabled():
    """D-25/P2-14: the mcp tool_hint is only ever allowed through when
    settings.mcp_enabled is true -- this is the flag task_expander_node
    reuses directly, with no separate "is mcp available" setting to
    drift out of sync."""
    router = _FakeRouter({"tasks": [
        {"goal_id": "g1", "query": "q1", "priority": 1, "tool_hint": "mcp"},
    ]})
    node = build_task_expander_node(router, _settings(max_fanout=6, mcp_enabled=False))
    state = ResearchState(raw_query="q", goals=[Goal(goal_id="g1", description="d")])
    update = node(state)
    assert update["pending_tasks"][0].tool_hint == ""


def test_task_expander_node_allows_the_mcp_hint_when_mcp_is_enabled():
    router = _FakeRouter({"tasks": [
        {"goal_id": "g1", "query": "q1", "priority": 1, "tool_hint": "mcp"},
    ]})
    node = build_task_expander_node(router, _settings(max_fanout=6, mcp_enabled=True))
    state = ResearchState(raw_query="q", goals=[Goal(goal_id="g1", description="d")])
    update = node(state)
    assert update["pending_tasks"][0].tool_hint == "mcp"
