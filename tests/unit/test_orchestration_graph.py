"""
tests/unit/test_orchestration_graph.py — orchestration/graph.py's
routing functions and static wiring.

Covers: dispatch_tasks (D-1, plus P2-14's tool_hint -> specialist-node
routing, D-25), route_convergence (D-14), route_after_critique (D-22),
and build_graph's static node registration with/without an mcp_tool
wired in (P2-14). Does NOT cover a real graph.invoke() run — those are
integration tests (tests/integration/test_graph_end_to_end.py and
test_mcp_routing_end_to_end.py); everything here calls the routing
functions directly against a hand-built ResearchState, or inspects
build_graph's returned graph object without ever invoking it.
"""

from research_agent.config import Settings
from research_agent.orchestration.graph import (build_graph, dispatch_tasks,
                                                 route_after_critique, route_convergence)
from research_agent.state import ResearchState, SearchTask

_S = Settings(_env_file=None, llm_mode="stub", max_depth=2, max_revisions=2)


def _state(**kw) -> ResearchState:
    return ResearchState(raw_query="q", **kw)


# ---------------------------------------------------------------------------
# dispatch_tasks (D-1/D-25)
# ---------------------------------------------------------------------------


def test_dispatch_empty_backlog_falls_through_to_compiler():
    # D-1: empty Send list must never silently halt the graph.
    # P2-14: dispatch_tasks now takes hint_to_node -- {} here matches
    # every pre-P2-14 call site and behavior exactly (no specialist
    # wired in this graph).
    assert dispatch_tasks(_state(), {}) == "compiler"


def test_dispatch_fans_out_one_send_per_task():
    t = SearchTask(key="g1::x", query="x", goal_id="g1")
    sends = dispatch_tasks(_state(pending_tasks=[t]), {})
    assert isinstance(sends, list) and len(sends) == 1
    assert sends[0].node == "search_worker"


def test_dispatch_routes_a_hinted_task_to_its_specialist_node():
    """P2-14 (D-25): a task whose tool_hint is a KEY in hint_to_node
    routes to that node name instead of the default "search_worker"."""
    t = SearchTask(key="g1::x", query="x", goal_id="g1", tool_hint="mcp")
    sends = dispatch_tasks(_state(pending_tasks=[t]), {"mcp": "mcp_search_worker"})
    assert isinstance(sends, list) and len(sends) == 1
    assert sends[0].node == "mcp_search_worker"


def test_dispatch_falls_back_to_default_for_an_unrecognized_hint():
    """Defense in depth: even if a hint somehow doesn't match anything in
    hint_to_node (shouldn't happen -- cap_and_filter already only sets
    hints present in this same set -- but dispatch_tasks doesn't re-trust
    that), it degrades to the default rather than raising a KeyError."""
    t = SearchTask(key="g1::x", query="x", goal_id="g1", tool_hint="nonexistent")
    sends = dispatch_tasks(_state(pending_tasks=[t]), {"mcp": "mcp_search_worker"})
    assert sends[0].node == "search_worker"


def test_dispatch_mixed_backlog_routes_each_task_independently():
    """One dispatch call, one task with a hint, one without -- each must
    route independently, proving this isn't an all-or-nothing decision
    per call."""
    hinted = SearchTask(key="g1::a", query="a", goal_id="g1", tool_hint="mcp")
    plain = SearchTask(key="g2::b", query="b", goal_id="g2")
    sends = dispatch_tasks(_state(pending_tasks=[hinted, plain]), {"mcp": "mcp_search_worker"})
    nodes = {s.node for s in sends}
    assert nodes == {"mcp_search_worker", "search_worker"}


# ---------------------------------------------------------------------------
# route_convergence (D-14)
# ---------------------------------------------------------------------------


def test_convergence_compiles_on_recall_target():
    assert route_convergence(_state(recall_score=0.9), _S) == "compiler"


def test_convergence_compiles_on_depth_exhaustion():
    assert route_convergence(_state(recall_score=0.1, iteration_depth=2), _S) == "compiler"


def test_convergence_expands_otherwise():
    assert route_convergence(_state(recall_score=0.1, iteration_depth=1), _S) == "gap_generator"


def test_convergence_g2_reloops_when_recall_met_but_ungrounded():
    """Guardrail G2: recall alone must not be enough to declare
    convergence when grounded_score is below settings.grounded_recall_target
    and depth budget remains -- this is the exact combination run
    p205.131-check hit (recall=1.0, corpus_recall=0.0)."""
    state = _state(recall_score=1.0, grounded_score=0.0, iteration_depth=0)
    assert route_convergence(state, _S) == "gap_generator"


def test_convergence_g2_still_compiles_once_depth_is_spent():
    """Even fully ungrounded, once max_depth is reached there is no
    budget left to spend chasing grounding further -- falls through to
    compiler exactly like the ungrounded-recall path always did."""
    state = _state(recall_score=1.0, grounded_score=0.0, iteration_depth=2)
    assert route_convergence(state, _S) == "compiler"


def test_convergence_g2_compiles_when_recall_and_grounding_both_met():
    state = _state(recall_score=1.0, grounded_score=1.0, iteration_depth=0)
    assert route_convergence(state, _S) == "compiler"


# ---------------------------------------------------------------------------
# route_after_critique (D-22)
# ---------------------------------------------------------------------------


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
# build_graph static wiring (P2-14) — node registration only, no invoke()
# ---------------------------------------------------------------------------


def test_build_graph_without_mcp_tool_never_registers_the_specialist_node(graph):
    """graph fixture (conftest.py) never passes mcp_tool -- confirms the
    default shape is completely unchanged from before P2-14."""
    node_names = set(graph.get_graph().nodes.keys())
    assert "mcp_search_worker" not in node_names
    assert "search_worker" in node_names


def test_build_graph_with_mcp_tool_registers_the_specialist_node():
    from langgraph.checkpoint.memory import MemorySaver

    from research_agent.llm.client import StubClient
    from research_agent.llm.router import FallbackRouter
    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.storage.qdrant_store import QdrantStore

    settings = Settings(_env_file=None, llm_mode="stub", hitl_enabled=False,
                       mcp_enabled=True,
                       qdrant_url="http://127.0.0.1:1",
                       postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                       opensearch_url="http://127.0.0.1:1")
    router = FallbackRouter([StubClient()], quality_threshold=0.6)
    memory = SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                            settings.memory_top_k, 90.0, 14.0)

    def fake_tool(task):
        return []

    g = build_graph(router, fake_tool, memory, settings, MemorySaver(), mcp_tool=fake_tool)
    node_names = set(g.get_graph().nodes.keys())
    assert "mcp_search_worker" in node_names
    assert "search_worker" in node_names
