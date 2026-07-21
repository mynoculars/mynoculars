"""
tests/test_integration_paths.py — Full-graph tests for the two failure paths
the base e2e test cannot reach (added per external review item R4).

1. Critique exhaustion (E4 stub path): critic rejects every draft ->
   revision loop must terminate at MAX_REVISIONS, memory must NOT be fed,
   the (unreviewed) report must still ship.
2. Worker failure (D-16 path): the tool raises on every call -> failures
   recorded with depth, run still terminates with a report, nothing marked
   completed.
"""

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.llm.client import StubClient, _extract_json
from research_agent.llm.router import FallbackRouter
from research_agent.orchestration.graph import build_graph
from research_agent.state import ResearchState


class RejectingCriticStub(StubClient):
    """StubClient whose critic ALWAYS fails the report."""

    def complete(self, messages, temperature=0.2):
        last = messages[-1]["content"]
        if "TASK=critique" in last:
            return json.dumps({"passed": False, "score": 0.2,
                               "notes": ["missing tradeoff analysis"]})
        return super().complete(messages, temperature)


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


@pytest.fixture
def broken_tool():
    """A retrieval tool that always raises (backend down)."""

    def tool(task):
        raise ConnectionError("retrieval backend unreachable")

    return tool


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


def test_stub_json_fence_tolerance():
    """Regression guard for _extract_json's fence stripping."""
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
