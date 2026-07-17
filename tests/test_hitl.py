"""
tests/test_hitl.py — Human-in-the-loop escalation (D-23/D-28), fully offline.

Covers: interrupts fire only when HITL is enabled; each trigger pauses with
the right payload; approve/redirect/abort resume correctly under the same
thread_id; escalation_history records exactly ONE entry per escalation
(the D-28 idempotency invariant, observed from outside); every path still
terminates with a report.
"""

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.orchestration.graph import build_graph
from research_agent.state import ResearchState

from tests.test_integration_paths import RejectingCriticStub


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
    router = FallbackRouter(ZeroGoalsStub(), None, 0.6)
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
    router = FallbackRouter(ZeroGoalsStub(), None, 0.6)
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


# ---------------------------------------------------------------------------
# E4 — critique exhaustion: approve ships unreviewed; redirect re-arms once
# ---------------------------------------------------------------------------


def test_e4_approve_ships_without_memory(off_memory, fake_tool, hitl_settings):
    router = FallbackRouter(RejectingCriticStub(), None, 0.6)
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
    router = FallbackRouter(RejectingCriticStub(), None, 0.6)
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
