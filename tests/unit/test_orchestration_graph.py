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

from research_agent.agents.gathering import build_progress_checker_node
from research_agent.config import Settings
from research_agent.orchestration.graph import (build_graph, dispatch_tasks,
                                                 route_after_critique, route_convergence)
from research_agent.state import Evidence, Goal, ResearchState, SearchTask

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


def test_convergence_grounding_gets_exactly_one_gap_generator_attempt():
    """S-8: the very first below-target grounded_score measurement
    (grounded_score_prev's -1.0 sentinel, meaning no prior cycle yet)
    must still route to gap_generator -- the stall check only applies
    from the SECOND measurement onward."""
    state = _state(recall_score=1.0, grounded_score=0.0,
                   grounded_score_prev=-1.0, iteration_depth=0)
    assert route_convergence(state, _S) == "gap_generator"


def test_first_ungrounded_cycle_loops_instead_of_reporting_a_stall(caplog):
    """D-80 regression -- and the reason it is composed rather than
    hand-built.

    The test directly above pins the same intended behaviour, but
    constructs `grounded_score_prev=-1.0` by hand: a state the production
    path could not actually produce, because progress_checker_node
    overwrote that field with state.grounded_score (still its unmeasured
    1.0 default) on every cycle INCLUDING the first. So that test passed
    while the real graph did the opposite -- run p205.246-check logged
    `convergence.grounding_stalled grounded=0.0 grounded_prev=1.0 depth=1`
    and compiled at depth 1 with MAX_DEPTH=3 entirely unspent.

    This test therefore builds no intermediate state of its own. It runs
    the REAL progress_checker_node over a fresh ResearchState, feeds its
    actual output into route_convergence, and asserts on the pair -- the
    only arrangement that can catch a producer and a consumer disagreeing
    about one field's contract."""
    import logging

    settings = Settings(_env_file=None, llm_mode="stub", max_depth=3,
                        min_evidence_score=0.5, recall_target=0.85,
                        grounded_recall_target=0.5,
                        model_knowledge_enabled=False)
    fresh = ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[Goal(goal_id="g1",
                    description="PLA size versus Indian Army size")],
        # Covers the goal (recall 1.0) but cannot ground it (D-57) -- the
        # exact shape the live run produced.
        evidence=[Evidence(task_key="t1", goal_id="g1", source="web",
                           content="The PLA fields roughly two million "
                                   "active personnel.", score=0.7)],
    )

    update = build_progress_checker_node(settings, debug=False)(fresh)
    after = fresh.model_copy(update=update)

    assert after.recall_score == 1.0 and after.grounded_score == 0.0, (
        "precondition: the covered-but-ungrounded shape being routed on")

    with caplog.at_level(logging.WARNING):
        destination = route_convergence(after, settings)

    assert destination == "gap_generator", (
        "an ungrounded FIRST cycle must spend a gather cycle chasing "
        "grounding (D-47), never be mistaken for a stall")
    assert not [r for r in caplog.records
                if "convergence.grounding_stalled" in r.message], (
        "nothing has stalled yet -- there is no previous measurement to "
        "compare the first one against")


def test_second_ungrounded_cycle_does_report_a_stall(caplog):
    """The complement of the test above, composed the same way: once a
    REAL previous measurement exists and grounding has not improved on it,
    the stall exit must still fire. D-80 restores the first-cycle
    exemption without weakening what S-8 is actually for -- stopping a run
    from spending its whole depth budget on a condition already shown not
    to move."""
    import logging

    settings = Settings(_env_file=None, llm_mode="stub", max_depth=3,
                        min_evidence_score=0.5, recall_target=0.85,
                        grounded_recall_target=0.5,
                        model_knowledge_enabled=False)
    second_cycle = ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[Goal(goal_id="g1",
                    description="PLA size versus Indian Army size")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="web",
                           content="The PLA fields roughly two million "
                                   "active personnel.", score=0.7)],
        # One cycle already ran and measured 0.0 grounding.
        iteration_depth=1,
        grounded_score=0.0,
    )

    update = build_progress_checker_node(settings, debug=False)(second_cycle)
    after = second_cycle.model_copy(update=update)

    assert after.grounded_score_prev == 0.0, "a real prior measurement"

    with caplog.at_level(logging.WARNING):
        destination = route_convergence(after, settings)

    assert destination == "compiler"
    assert [r for r in caplog.records
            if "convergence.grounding_stalled" in r.message], (
        "expected the S-8 stall WARNING once grounding genuinely fails to "
        "improve between two measured cycles")


def test_convergence_stalled_grounding_compiles_instead_of_looping_again(caplog):
    """S-8: live trace (run p205.131-check) showed grounded_score stuck at
    0.00 for THREE consecutive cycles against an off-topic corpus -- once
    a cycle shows no improvement over the previous one, stop spending
    depth budget on it rather than trying again."""
    import logging
    state = _state(recall_score=1.0, grounded_score=0.0,
                   grounded_score_prev=0.0, iteration_depth=1)
    with caplog.at_level(logging.WARNING):
        result = route_convergence(state, _S)
    assert result == "compiler"
    stalled = [r for r in caplog.records
              if "convergence.grounding_stalled" in r.message]
    assert stalled, "expected a WARNING when grounding stalls"


def test_convergence_does_not_call_it_stalled_when_grounding_improved():
    """A cycle that DID improve grounded_score, even if still below
    target, must keep looping -- only a flat or regressed score counts
    as stalled."""
    state = _state(recall_score=1.0, grounded_score=0.3,
                   grounded_score_prev=0.1, iteration_depth=1)
    assert route_convergence(state, _S) == "gap_generator"


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
