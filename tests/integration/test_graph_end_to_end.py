"""
tests/integration/test_graph_end_to_end.py — full graph.invoke() runs,
offline.

Unlike tests/unit/, every test here builds a real graph (via the
`graph` fixture or a hand-assembled build_graph call) and actually
invokes it end to end: plan -> gather (fan-out) -> compile -> critique
-> telemetry, on StubClient + fake tools, no services, no network. Each
test exists to prove something only visible at the WHOLE-graph level —
a single node's unit test can't catch a telemetry field that's wired
correctly in isolation but never reaches the final `result["telemetry"]`
dict, for instance (see test_telemetry_surfaces_llm_quality_calls_failed_end_to_end
below for exactly that shape of bug).
"""

import json
import logging

from langgraph.checkpoint.memory import MemorySaver

from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import Evidence, ResearchState, Volatility
from research_agent.storage.qdrant_store import QdrantStore
from research_agent.tools.corpus_search import make_corpus_tool


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


class MalformedGoalsStub(StubClient):
    """Stub whose goal composition returns one valid goal and one malformed
    one (missing "description") — exercises goal_manager_node's RawGoal
    validation (P2-06) end to end through the graph, not just the helper
    (see tests/unit/test_agents_task_utils.py for the helper-level test)."""

    def complete(self, messages, temperature=0.2):
        if "TASK=goals" in messages[-1]["content"]:
            return json.dumps({"goals": [
                {"goal_id": "g1", "description": "a real goal"},
                {"goal_id": "g2"},  # missing description -> dropped
            ]})
        return super().complete(messages, temperature)


def test_goal_manager_drops_malformed_goal_and_counts_reject(
        off_memory, fake_tool, settings):
    router = FallbackRouter([MalformedGoalsStub()], 0.6)
    graph = build_graph(router, fake_tool, off_memory, settings, MemorySaver())

    result = graph.invoke(
        ResearchState(raw_query="q"),
        config={"configurable": {"thread_id": "test-p206"},
                "recursion_limit": settings.recursion_limit})

    # Only the well-formed goal survives; the run still completes normally
    # (never a KeyError-aborted process) and the reject is counted.
    assert result["telemetry"]["goals"] == 1
    assert result["telemetry"]["producer_rejects"] >= 1


def test_full_graph_telemetry_reports_provider_level_counters(graph, settings):
    """End-to-end: telemetry now distinguishes node-level from provider-level
    LLM activity (P2-07), on top of the pre-existing e2e assertions."""
    result = graph.invoke(
        ResearchState(raw_query="Compare Redis and Memcached for session caching"),
        config={"configurable": {"thread_id": "test-p207"},
                "recursion_limit": settings.recursion_limit},
    )
    tele = result["telemetry"]
    assert "llm_node_calls" in tele
    assert "llm_calls" not in tele  # renamed, not aliased (P2-07 is explicit about this)
    # Single-provider stub chain: one provider attempt per node-level call,
    # no fallbacks, no quality scoring (nothing to fall back to).
    assert tele["llm_provider_calls"] == tele["llm_node_calls"]
    assert tele["llm_fallback_hops"] == 0
    assert tele["llm_quality_calls"] == 0
    assert tele["producer_rejects"] == 0  # stub goals/tasks are always well-formed


class _FakeLeg:
    """Minimal fake store: only .search() and .available — enough to drive
    HybridRetriever without touching real Qdrant/OpenSearch."""

    def __init__(self, hits=None, available=True):
        self._hits = hits or []
        self.available = available

    def search(self, query, top_k):
        return list(self._hits)


def test_full_graph_with_real_corpus_tool_reports_retrieval_counters(
        off_memory, stub_router, settings):
    """End-to-end: wiring the REAL HybridRetriever + make_corpus_tool
    (instead of the fake_tool fixture) through the whole graph, telemetry
    should report actual retrieval_dense_calls/retrieval_keyword_calls —
    the fixture-based e2e tests elsewhere in this suite never exercise
    this path, since fake_tool bypasses HybridRetriever entirely."""
    retriever = HybridRetriever(
        _FakeLeg(hits=[{"content": "fact one", "title": "doc1"}]),
        _FakeLeg(hits=[{"content": "fact one", "title": "doc1"}]),
        min_similarity=0.0)
    tool = make_corpus_tool(retriever, top_k=3)
    graph = build_graph(stub_router, tool, off_memory, settings, MemorySaver())

    result = graph.invoke(
        ResearchState(raw_query="q"),
        config={"configurable": {"thread_id": "test-p207-retrieval"},
                "recursion_limit": settings.recursion_limit})

    tele = result["telemetry"]
    # Stub composes 2 goals -> 2 tasks -> 2 search_worker calls, each one
    # real HybridRetriever.search() call (one dense + one keyword attempt).
    assert tele["retrieval_dense_calls"] == 2
    assert tele["retrieval_keyword_calls"] == 2
    assert tele["retrieval_leg_unavailable"] == 0


class _AlwaysErroringJudge:
    """A ChatClient whose complete_json (the method score_answer calls)
    always raises — simulating exactly what the real Gemini 429 did:
    the JUDGE, not the answering provider, is the one that's down. See
    tests/unit/test_llm_router.py for the router-level (non-graph)
    version of this same fake."""

    name = "judge"

    def complete(self, messages, temperature=0.2):
        return "judge answer"

    def complete_json(self, messages, temperature=0.0):
        raise RuntimeError("judge is down")


def test_telemetry_surfaces_llm_quality_calls_failed_end_to_end(settings):
    """Regression guard for the exact gap a live run found: the counter
    can be correctly bumped in state.counters (proven by the router-level
    tests in tests/unit/test_llm_router.py) and STILL never appear in the
    printed/persisted telemetry, because telemetry_node (agents/
    compilation.py) builds its output dict by explicitly enumerating keys
    rather than passing state.counters through wholesale. This drives a
    real graph run with StubClient as the answering provider (so
    planning/goal composition succeeds normally, same as every other
    full-graph test in this file) and _AlwaysErroringJudge as the NEXT
    provider in the chain — so it's only ever consulted to SCORE the
    compiler's answer, never to produce one — then asserts the key is
    actually present in state.telemetry, not just in the raw counters
    dict some earlier test already checked."""
    router = FallbackRouter([StubClient(), _AlwaysErroringJudge()], quality_threshold=0.6)

    def fake_tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="fake",
                         content=f"fact about {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    memory = SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                            settings.memory_top_k, 90.0, 14.0)

    g = build_graph(router, fake_tool, memory, settings, MemorySaver())
    result = g.invoke(ResearchState(raw_query="q"), config={"configurable": {"thread_id": "t"}})

    assert "llm_quality_calls_failed" in result["telemetry"]
    assert result["telemetry"]["llm_quality_calls_failed"] >= 1


# ---------------------------------------------------------------------------
# P2-13 follow-up: evidence_by_source telemetry + worker.done source log
#
# Direct answer to "is there any indication content was retrieved via
# MCP": before this, there wasn't a deterministic one -- only an
# indirect, LLM-dependent hint. These tests cover the two concrete
# signals added instead.
# ---------------------------------------------------------------------------


def test_telemetry_evidence_by_source_reflects_the_standard_test_fixture(graph):
    """The standard `graph` fixture's own fake_tool tags its evidence
    source="fake" (see conftest.py) -- confirms evidence_by_source counts
    whatever string is ACTUALLY on each Evidence item, not a hardcoded
    list of expected sources like "corpus"/"mcp"/"memory"."""
    result = graph.invoke(ResearchState(raw_query="q"),
                          config={"configurable": {"thread_id": "evidence-by-source-fake"}})

    telemetry = result["telemetry"]
    assert telemetry["evidence_by_source"] == {"fake": telemetry["evidence_items"]}


def test_telemetry_evidence_by_source_distinguishes_mcp_from_corpus(settings):
    """The scenario this was actually built for: a tool tagging its
    evidence source="mcp" (tools/mcp_client.py::make_mcp_tool always does
    this) must show up as such in telemetry, distinctly from any other
    source -- proven here with a minimal stand-in tool rather than a full
    MCPBridge, since only the source-counting behavior is under test."""
    def mcp_shaped_tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="mcp",
                         content=f"mcp fact about {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    router = FallbackRouter([StubClient()], quality_threshold=0.6)
    memory = SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                            settings.memory_top_k, 90.0, 14.0)
    graph = build_graph(router, mcp_shaped_tool, memory, settings, MemorySaver())

    result = graph.invoke(ResearchState(raw_query="q"),
                          config={"configurable": {"thread_id": "evidence-by-source-mcp"}})

    telemetry = result["telemetry"]
    assert telemetry["evidence_by_source"] == {"mcp": telemetry["evidence_items"]}
    assert "corpus" not in telemetry["evidence_by_source"]


def test_worker_done_log_line_reports_the_tools_actual_source(graph, caplog):
    """Per-task, real-time visibility in a --debug trace: which tool
    answered THIS specific task, not just the run-level aggregate."""
    with caplog.at_level(logging.INFO):
        graph.invoke(ResearchState(raw_query="q"),
                    config={"configurable": {"thread_id": "worker-done-source"}})

    done_lines = [r for r in caplog.records if r.message == "worker.done"]
    assert done_lines, "expected at least one worker.done log line"
    for line in done_lines:
        assert line.event_fields["source"] == "fake"  # matches conftest.py's fake_tool
