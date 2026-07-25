"""
tests/integration/test_mcp_routing_end_to_end.py — P2-14 (D-25) typed
specialist workers, full graph run.

The unit-level pieces (cap_and_filter's hint validation, dispatch_tasks'
routing, build_graph's node registration) are covered separately in
tests/unit/test_agents_task_utils.py and test_orchestration_graph.py.
This is the one test that proves the WHOLE chain end to end: one real
graph.invoke(), one task hinted "mcp", one task with no hint, each
landing on the RIGHT tool, with telemetry showing both sources.
"""

import json

from langgraph.checkpoint.memory import MemorySaver

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.state import Evidence, ResearchState, Volatility
from research_agent.storage.qdrant_store import QdrantStore


def _p214_settings(mcp_enabled: bool):
    return Settings(_env_file=None, llm_mode="stub", hitl_enabled=False,
                    mcp_enabled=mcp_enabled,
                    qdrant_url="http://127.0.0.1:1",
                    postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


def _p214_memory(settings):
    return SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                          settings.memory_top_k, 90.0, 14.0)


def test_p2_14_mixed_backlog_routes_to_both_specialists_end_to_end():
    """The real, definitive proof: one full graph run, one task hinted
    "mcp", one task with no hint -- each must land on the RIGHT tool, and
    the resulting evidence must show BOTH sources in telemetry. Uses a
    custom StubClient subclass (only overriding the TASK=expand response;
    every other prompt still gets StubClient's normal canned behavior)
    rather than a full FallbackRouter fake, so this test exercises the
    REAL agents/planning.py::task_expander_node ->
    task_utils.py::cap_and_filter -> orchestration/graph.py::
    dispatch_tasks chain end to end, not a shortcut around it."""

    class _HintingStubClient(StubClient):
        def complete(self, messages, temperature=0.2):
            last = messages[-1]["content"]
            if "TASK=expand" in last:
                return json.dumps({"tasks": [
                    {"query": "corpus query", "goal_id": "g1", "priority": 2},
                    {"query": "mcp query", "goal_id": "g2", "priority": 2, "tool_hint": "mcp"},
                ]})
            return super().complete(messages, temperature)

    settings = _p214_settings(mcp_enabled=True)
    router = FallbackRouter([_HintingStubClient()], quality_threshold=0.6)
    memory = _p214_memory(settings)

    corpus_calls = []
    mcp_calls = []

    def fake_corpus_tool(task):
        corpus_calls.append(task)
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="corpus",
                         content=f"corpus result for {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    def fake_mcp_tool(task):
        mcp_calls.append(task)
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="mcp",
                         content=f"mcp result for {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    g = build_graph(router, fake_corpus_tool, memory, settings, MemorySaver(),
                    mcp_tool=fake_mcp_tool)
    result = g.invoke(ResearchState(raw_query="q"),
                      config={"configurable": {"thread_id": "p214-mixed-backlog"}})

    assert len(corpus_calls) == 1
    assert corpus_calls[0].query == "corpus query"
    assert corpus_calls[0].tool_hint == ""
    assert len(mcp_calls) == 1
    assert mcp_calls[0].query == "mcp query"
    assert mcp_calls[0].tool_hint == "mcp"

    telemetry = result["telemetry"]
    assert telemetry["evidence_by_source"].get("corpus", 0) >= 1
    assert telemetry["evidence_by_source"].get("mcp", 0) >= 1
