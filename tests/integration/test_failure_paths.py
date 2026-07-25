"""
tests/integration/test_failure_paths.py — full-graph tests for the two
failure paths the base e2e test cannot reach (added per external review
item R4).

1. Critique exhaustion (E4 stub path, HITL off): critic rejects every
   draft -> revision loop must terminate at MAX_REVISIONS, memory must
   NOT be fed, the (unreviewed) report must still ship. (The HITL-ON
   version of this same trigger is tests/integration/
   test_hitl_escalation.py's E4 tests — RejectingCriticStub is shared
   between both files via conftest.py.)
2. Worker failure (D-16 path): the tool raises on every call -> failures
   recorded with depth, run still terminates with a report, nothing
   marked completed.
"""

from langgraph.checkpoint.memory import MemorySaver

import pytest

from research_agent.llm.router import FallbackRouter
from research_agent.orchestration.graph import build_graph
from research_agent.state import ResearchState

from tests.conftest import RejectingCriticStub


@pytest.fixture
def broken_tool():
    """A retrieval tool that always raises (backend down). Local to this
    file -- only test_worker_failures_recorded_and_run_terminates below
    uses it."""

    def tool(task):
        raise ConnectionError("retrieval backend unreachable")

    return tool


def test_critique_exhaustion_terminates_and_skips_memory(
        fake_tool, off_memory, settings):
    router = FallbackRouter([RejectingCriticStub()], quality_threshold=0.6)
    graph = build_graph(router, fake_tool, off_memory, settings, MemorySaver())

    result = graph.invoke(
        ResearchState(raw_query="q"),
        config={"configurable": {"thread_id": "test-e4"},
                "recursion_limit": settings.recursion_limit})

    tele = result["telemetry"]
    # Loop bounded exactly at max_revisions (D-22) — no runaway rewrites.
    assert tele["revision_cycles"] == settings.max_revisions
    assert tele["critique_passed"] is False
    # E4 path: a failed report never feeds long-term memory.
    assert tele["memory_writes"] == 0
    # ...but the run still ships a report — never a silent death.
    assert result["final_report"]
    # Grounded rewrite: critic notes accumulated for the second draft.
    assert result["critique_notes"]


def test_worker_failures_recorded_and_run_terminates(
        broken_tool, off_memory, settings, stub_router):
    graph = build_graph(stub_router, broken_tool, off_memory, settings, MemorySaver())

    result = graph.invoke(
        ResearchState(raw_query="q"),
        config={"configurable": {"thread_id": "test-d16"},
                "recursion_limit": settings.recursion_limit})

    tele = result["telemetry"]
    # Both stub tasks failed; none completed (D-16: failed != completed).
    assert tele["search_failures"] == 2
    assert tele["search_calls"] == 0
    assert len(result["failed_task_keys"]) == 2
    assert not result["completed_task_keys"]
    # Failure depth recorded (initial expansion happens at depth 0).
    assert set(result["failed_task_keys"].values()) == {0}
    # Zero evidence -> recall 0 -> loop runs, gap stub yields nothing,
    # empty-backlog fallthrough (D-1) still lands a report.
    assert tele["recall"] == 0.0
    assert result["final_report"]
