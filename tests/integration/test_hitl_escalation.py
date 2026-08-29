"""
tests/integration/test_hitl_escalation.py — Human-in-the-loop escalation
(D-23/D-28), fully offline.

Covers: interrupts fire only when HITL is enabled; each trigger pauses
with the right payload; approve/redirect/abort resume correctly under
the same thread_id; escalation_history records exactly ONE entry per
escalation (the D-28 idempotency invariant, observed from outside); every
path still terminates with a report.
"""

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.orchestration.graph import build_graph
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import Evidence, ResearchState, Volatility

from tests.conftest import RejectingCriticStub


@pytest.fixture
def hitl_settings() -> Settings:
    """Same tight bounds as the base fixture, HITL switched ON."""
    return Settings(_env_file=None, llm_mode="stub", max_depth=2, max_fanout=4,
                    max_revisions=2, hitl_enabled=True,
                    qdrant_url="http://127.0.0.1:1",
                    postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


def _cfg(settings, thread):
    return {"configurable": {"thread_id": thread},
            "recursion_limit": settings.recursion_limit}


def _resume(action, guidance=""):
    return Command(resume={"action": action, "guidance": guidance})


class ZeroGoalsStub(StubClient):
    """Stub whose goal composition returns an empty goal set (E1 trigger)."""

    def complete(self, messages, temperature=0.2):
        if "TASK=goals" in messages[-1]["content"]:
            return json.dumps({"goals": []})
        return super().complete(messages, temperature)


class BrokenTool:
    """Retrieval tool that always raises — drives recall to 0 (E3 trigger)."""

    def __call__(self, task):
        raise ConnectionError("backend down")


# ---------------------------------------------------------------------------
# E1 — plan anomaly
# ---------------------------------------------------------------------------


def test_e1_interrupts_then_approve_ships_error_report(off_memory, fake_tool,
                                                       hitl_settings):
    router = FallbackRouter([ZeroGoalsStub()], 0.6)
    graph = build_graph(router, fake_tool, off_memory, hitl_settings, MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e1")

    result = graph.invoke(ResearchState(raw_query="q"), config=cfg)
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["trigger"] == "E1"

    result = graph.invoke(_resume("approve"), config=cfg)
    assert "__interrupt__" not in result
    assert "planning failed" in result["final_report"]
    # D-28 observed from outside: exactly one history entry despite the
    # node executing twice (pre- and post-resume).
    assert len(result["escalation_history"]) == 1
    assert result["escalation_history"][0]["action"] == "approve"
    assert result["telemetry"]  # mandatory sink reached


def test_e1_abort_produces_abort_report(off_memory, fake_tool, hitl_settings):
    router = FallbackRouter([ZeroGoalsStub()], 0.6)
    graph = build_graph(router, fake_tool, off_memory, hitl_settings, MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e1-abort")

    graph.invoke(ResearchState(raw_query="q"), config=cfg)
    result = graph.invoke(_resume("abort"), config=cfg)
    assert "aborted by human" in result["final_report"]
    assert result["escalation_history"][0]["action"] == "abort"


# ---------------------------------------------------------------------------
# E3 — non-convergence at depth exhaustion
# ---------------------------------------------------------------------------


def test_e3_interrupts_then_approve_ships_partial(off_memory, stub_router,
                                                  hitl_settings):
    graph = build_graph(stub_router, BrokenTool(), off_memory, hitl_settings,
                        MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e3")

    result = graph.invoke(ResearchState(raw_query="q"), config=cfg)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["trigger"] == "E3"
    assert payload["recall"] == 0.0
    assert payload["uncovered_goals"]  # human sees what's missing

    result = graph.invoke(_resume("approve"), config=cfg)
    assert result["final_report"]
    assert result["telemetry"]["recall"] == 0.0  # shipped partial, honestly


class LowRelevanceTool:
    """Retrieval tool that always returns evidence scored BELOW the
    coverage floor — simulating exactly what corpus_search used to hand
    back for an off-topic query before P2-01: a real hit, not a failure,
    just not relevant enough to actually cover anything. This is the gap
    BrokenTool (above) does NOT test — BrokenTool simulates retrieval
    FAILING outright; this simulates retrieval SUCCEEDING with junk, which
    is the specific pattern that silently produced recall=1.0 while
    MIN_EVIDENCE_SCORE defaulted to 0.0.
    """

    def __call__(self, task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="fake",
                         content=f"barely-related note about {task.query}",
                         score=0.2, volatility=Volatility.SEMI_STABLE)]


# complementary, narrower test — for the retrieval-time floor itself
def test_e3_fires_on_low_relevance_evidence_not_just_tool_failure(
        off_memory, stub_router, hitl_settings):
    """P2-05. Before P2-01, every task here would score 0.2, and with
    MIN_EVIDENCE_SCORE=0.0 the coverage predicate (e.score >= 0.0) was
    TRUE anyway — every goal "covered", recall=1.0, no escalation, ever.
    With the new default (0.5), a 0.2-scored item can't satisfy coverage,
    recall stays below target, and E3 should fire.
    """
    graph = build_graph(stub_router, LowRelevanceTool(), off_memory,
                        hitl_settings, MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e3-low-relevance")

    result = graph.invoke(ResearchState(raw_query="q"), config=cfg)
    assert "__interrupt__" in result, (
        "recall should be below target with only 0.2-scored evidence; "
        "if this fails, min_evidence_score is inert again")
    payload = result["__interrupt__"][0].value
    assert payload["trigger"] == "E3"
    assert payload["recall"] < hitl_settings.recall_target

    result = graph.invoke(_resume("approve"), config=cfg)
    assert result["final_report"]


def test_min_similarity_floor_drops_low_relevance_dense_hits():
    """P2-01's other gate, tested in isolation from the graph. Without this,
    only the coverage-check half of the fix (above) is covered."""
    class FakeDense:
        def search(self, query, top_k):
            return [{"title": "on-topic", "content": "x", "similarity": 0.9},
                   {"title": "off-topic", "content": "y", "similarity": 0.1}]

    class FakeKeyword:
        def search(self, query, top_k):
            return []

    retriever = HybridRetriever(FakeDense(), FakeKeyword(), min_similarity=0.35)
    results = retriever.search("q", top_k=5)
    assert len(results) == 1
    assert results[0]["title"] == "on-topic"


# ---------------------------------------------------------------------------
# E4 — critique exhaustion: approve ships unreviewed; redirect re-arms once
# ---------------------------------------------------------------------------


def test_e4_approve_ships_without_memory(off_memory, fake_tool, hitl_settings):
    router = FallbackRouter([RejectingCriticStub()], 0.6)
    graph = build_graph(router, fake_tool, off_memory, hitl_settings, MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e4")

    result = graph.invoke(ResearchState(raw_query="q"), config=cfg)
    assert result["__interrupt__"][0].value["trigger"] == "E4"

    result = graph.invoke(_resume("approve"), config=cfg)
    assert result["final_report"]
    assert result["telemetry"]["critique_passed"] is False
    assert result["telemetry"]["memory_writes"] == 0  # failed report -> no memory


def test_e4_redirect_rearms_one_cycle_then_reescalates(off_memory, fake_tool,
                                                       hitl_settings):
    router = FallbackRouter([RejectingCriticStub()], 0.6)
    graph = build_graph(router, fake_tool, off_memory, hitl_settings, MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e4-redirect")

    graph.invoke(ResearchState(raw_query="q"), config=cfg)
    # Redirect: human guidance joins the critic notes, compiler rewrites once.
    result = graph.invoke(_resume("redirect", "focus on tradeoffs"), config=cfg)
    # Critic still rejects -> E4 fires again: the loop is now HUMAN-bounded
    # (each cycle costs one explicit decision), per design §6.9.
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["trigger"] == "E4"

    result = graph.invoke(_resume("approve"), config=cfg)
    assert len(result["escalation_history"]) == 2
    assert any("HUMAN REVIEWER: focus on tradeoffs" in n
               for n in result["critique_notes"])


# ---------------------------------------------------------------------------
# HITL disabled — triggers never fire (backwards compatibility)
# ---------------------------------------------------------------------------


def test_disabled_hitl_never_interrupts(graph, settings):
    result = graph.invoke(
        ResearchState(raw_query="q"),
        config=_cfg(settings, "hitl-off"))
    assert "__interrupt__" not in result
    assert result["escalation_history"] == []


def test_e3_stub_logs_when_hitl_disabled(off_memory, stub_router, settings, caplog):
    """P2-09: previously E2/E3 emitted NOTHING when HITL was off, unlike
    E1/E4's existing 'escalation.stub' lines — this proves parity without
    changing routing (the run still reaches telemetry normally, no
    interrupt)."""
    import logging

    graph = build_graph(stub_router, BrokenTool(), off_memory, settings, MemorySaver())
    with caplog.at_level(logging.WARNING):
        result = graph.invoke(
            ResearchState(raw_query="q"),
            config={"configurable": {"thread_id": "test-p209-e3"},
                    "recursion_limit": settings.recursion_limit})
    assert "__interrupt__" not in result  # HITL off: never pauses
    stub_lines = [r for r in caplog.records if "escalation.stub" in r.message]
    assert stub_lines, "expected an escalation.stub WARNING when HITL is off"
    assert stub_lines[0].event_fields["trigger"] in ("E2", "E3")


class _RejectingCriticFreshGaps(RejectingCriticStub):
    """Critic always rejects; gap generation emits genuinely NEW queries.

    The plain stub returns the same canned task list for every producer
    call, so cap_and_filter's dedup (D-2) removes them all as already
    completed and the backlog is empty regardless of routing -- which would
    make this test pass on the broken behaviour too.
    """

    _n = 0

    def complete_json(self, messages, temperature=0.0):
        if "TASK=gaps" in messages[-1]["content"]:
            type(self)._n += 1
            return {"tasks": [{"query": f"fresh gap query {self._n}",
                               "goal_id": "g1", "priority": 1}]}
        return super().complete_json(messages, temperature)


def test_e4_redirect_reaches_retrieval_not_just_the_compiler(off_memory,
                                                             fake_tool,
                                                             hitl_settings):
    """P205 regression (run p205.103-check). The reviewer's guidance was
    "ask for inputs from global watchdogs, UN reports of press freedom,
    human rights abuses, democracy index" -- a request for NEW EVIDENCE.
    E4 redirect routed straight back to the compiler, which can only
    rewrite the same evidence block, so revision 3 failed on exactly the
    same missing support and re-raised E4."""
    calls = []

    def counting_tool(task):
        calls.append(task.key)
        return fake_tool(task)

    router = FallbackRouter([_RejectingCriticFreshGaps()], 0.6)
    graph = build_graph(router, counting_tool, off_memory, hitl_settings,
                        MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-e4-regather")

    graph.invoke(ResearchState(raw_query="q"), config=cfg)
    before = len(calls)
    graph.invoke(_resume("redirect", "consult the UN democracy index"), config=cfg)
    assert len(calls) > before, (
        "a redirect asking for new evidence must trigger new searches")


# ---------------------------------------------------------------------------
# D-132 (P6-4) -- a reviewer's reading time is not the run's research time
# ---------------------------------------------------------------------------


def test_a_pause_is_stamped_while_paused_and_credited_back_on_resume(
        off_memory, fake_tool, hitl_settings):
    """The D-28-safe half of the run budget. human_escalation cannot
    write anything BEFORE interrupt(), so the node that RAISES the
    trigger stamps escalation_started_at, and the resume update -- the
    same place escalation_history is appended, for the same reason --
    turns that stamp into paused_seconds."""
    import time

    router = FallbackRouter([ZeroGoalsStub()], 0.6)
    graph = build_graph(router, fake_tool, off_memory, hitl_settings,
                        MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-budget-pause")

    graph.invoke(ResearchState(raw_query="q"), config=cfg)
    paused_state = graph.get_state(cfg).values
    assert paused_state["escalation_started_at"] > 0, (
        "the raising node must stamp when the pause began")
    assert paused_state["paused_seconds"] == 0.0, (
        "nothing is credited until the human actually answers")

    # A REAL pause, not an instantaneous one -- and the sleep is load-
    # bearing rather than lazy. time.time() is GetSystemTimeAsFileTime on
    # Windows, whose resolution is 15.625 ms
    # (`time.get_clock_info("time").resolution`, 1 ns on Linux); two
    # reads inside one tick return the IDENTICAL float, so an
    # instantaneous pause credits exactly 0.0 there and 30 microseconds
    # on Linux. The first version of this test asserted `> 0` and passed
    # on Linux for that reason alone. 50 ms clears three Windows ticks,
    # which makes a nonzero credit true on every platform rather than on
    # the one it was written on.
    before_resume = time.time()
    time.sleep(0.05)
    result = graph.invoke(_resume("approve"), config=cfg)
    wall_clock_gap = time.time() - before_resume

    assert result["paused_seconds"] > 0, (
        "a measurable pause must be credited back to the run budget")
    assert result["paused_seconds"] <= wall_clock_gap + 1.0, (
        "the credited pause can never exceed the time actually spent "
        "between the two invokes")
    assert result["escalation_started_at"] == 0.0, (
        "the stamp is cleared -- this pause is over")


def test_the_run_clock_survives_a_pause_and_resume(off_memory, stub_router,
                                                   hitl_settings):
    """run_started_at is an EPOCH precisely so it survives being
    checkpointed across an interrupt (limits.py). A monotonic reading
    would be meaningless on the far side of this."""
    graph = build_graph(stub_router, BrokenTool(), off_memory, hitl_settings,
                        MemorySaver())
    cfg = _cfg(hitl_settings, "hitl-budget-clock")

    graph.invoke(ResearchState(raw_query="q"), config=cfg)
    stamped = graph.get_state(cfg).values["run_started_at"]
    assert stamped > 0

    result = graph.invoke(_resume("approve"), config=cfg)
    assert result["run_started_at"] == stamped, (
        "a resumed run continues its original budget, it does not restart it")
