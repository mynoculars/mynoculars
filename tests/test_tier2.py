"""
tests/test_tier2.py — Tier 2 coverage: P2-06 (producer validation), P2-07
(boundary-scoped telemetry), P2-08 (Postgres lifecycle/API parity), P2-09
(config strictness + escalation-stub logging parity).

Same offline philosophy as the rest of this suite: StubClient/fake tools,
no network, no real Postgres/Qdrant/OpenSearch.
"""

import json
import logging
import pathlib
import sys

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.agents.task_utils import RawTask, cap_and_filter
from research_agent.config import warn_on_likely_env_typos
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.orchestration.graph import build_graph
from research_agent.state import ResearchState, SearchTask
from research_agent.storage.postgres import close_checkpointer

# ---------------------------------------------------------------------------
# P2-06 — producer output validation
# ---------------------------------------------------------------------------


def test_cap_and_filter_drops_malformed_tasks_and_counts_them(settings):
    raw = [
        {"query": "good query", "goal_id": "g1", "priority": 1},
        {"query": "", "goal_id": "g1"},           # empty query -> rejected
        {"goal_id": "g1", "priority": 1},          # missing query -> rejected
        {"query": "another", "goal_id": ""},       # empty goal_id -> rejected
    ]
    state = ResearchState(raw_query="q")
    tasks, rejected = cap_and_filter(raw, state, depth=0, max_fanout=6)
    assert len(tasks) == 1
    assert tasks[0].query == "good query"
    assert rejected == 3


def test_cap_and_filter_never_raises_on_missing_keys():
    # Before P2-06, a dict missing "goal_id"/"query" raised KeyError here
    # and took the whole run down. Now it's just a counted rejection.
    state = ResearchState(raw_query="q")
    tasks, rejected = cap_and_filter([{}], state, depth=0, max_fanout=6)
    assert tasks == []
    assert rejected == 1


class MalformedGoalsStub(StubClient):
    """Stub whose goal composition returns one valid goal and one malformed
    one (missing "description") — exercises goal_manager_node's RawGoal
    validation (P2-06) end to end through the graph, not just the helper."""

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


# ---------------------------------------------------------------------------
# P2-07 — boundary-scoped telemetry
# ---------------------------------------------------------------------------


class _Named:
    """Same fixture shape as test_core.py's _Named — kept local so this
    file's router-boundary tests don't depend on test_core's internals."""

    def __init__(self, name, behavior):
        self.name = name
        self.behavior = behavior

    def complete(self, messages, temperature=0.2):
        if messages and "TASK=quality" in messages[-1]["content"]:
            return json.dumps({"score": 0.2 if self.behavior == "low" else 0.9})
        if self.behavior == "error":
            raise RuntimeError(f"{self.name} down")
        return f"answer from {self.name}"

    def complete_json(self, messages, temperature=0.0):
        return json.loads(self.complete(messages, temperature))


def test_drain_counters_counts_provider_attempts_and_resets():
    router = FallbackRouter([_Named("primary", "error"), _Named("mistral", "answer")],
                           quality_threshold=0.6)
    router.complete([{"role": "user", "content": "x"}])
    drained = router.drain_counters()
    assert drained["llm_provider_calls"] == 2   # primary attempt + mistral attempt
    assert drained["llm_fallback_hops"] == 1    # exactly one hop, primary -> mistral

    # Draining resets — a second call with no further activity yields nothing.
    assert router.drain_counters() == {}


def test_drain_counters_counts_quality_scoring_calls():
    # Two providers means the first one's answer is quality-scored before
    # being accepted (there's a fallback to check against); the last
    # provider in a chain is never scored (see router.py's has_next logic).
    router = FallbackRouter([_Named("primary", "answer"), _Named("mistral", "answer")],
                           quality_threshold=0.6)
    router.complete([{"role": "user", "content": "x"}])
    drained = router.drain_counters()
    assert drained["llm_quality_calls"] == 1
    assert drained.get("llm_fallback_hops", 0) == 0  # quality passed, no hop needed


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


# ---------------------------------------------------------------------------
# P2-08 — Postgres lifecycle + API parity
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeCheckpointerWithConn:
    def __init__(self, conn):
        self.conn = conn


def test_close_checkpointer_closes_underlying_connection():
    conn = _FakeConn()
    close_checkpointer(_FakeCheckpointerWithConn(conn))
    assert conn.closed is True


def test_close_checkpointer_is_a_noop_for_memory_saver():
    # MemorySaver has no .conn attribute at all — must not raise.
    close_checkpointer(MemorySaver())


def test_close_checkpointer_survives_a_conn_that_errors_on_close(caplog):
    class _AngryConn:
        def close(self):
            raise RuntimeError("already gone")

    with caplog.at_level(logging.WARNING):
        close_checkpointer(_FakeCheckpointerWithConn(_AngryConn()))
    assert any("checkpointer.close_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# P2-09 — config strictness + escalation-stub logging parity
# ---------------------------------------------------------------------------


def test_warn_on_likely_env_typos_flags_known_mistakes(monkeypatch, caplog):
    monkeypatch.setenv("HITL", "true")          # should have been HITL_ENABLED
    monkeypatch.delenv("HITL_ENABLED", raising=False)
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert matches
    assert matches[0].event_fields["set_key"] == "HITL"
    assert matches[0].event_fields["probably_meant"] == "HITL_ENABLED"


def test_warn_on_likely_env_typos_silent_when_correct_key_present(monkeypatch, caplog):
    monkeypatch.setenv("HITL", "true")
    monkeypatch.setenv("HITL_ENABLED", "true")  # correct key also set -> no warning
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    assert not [r for r in caplog.records if "config.likely_typo" in r.message]


class BrokenTool:
    """Same shape as test_hitl.py's BrokenTool — retrieval always fails."""

    def __call__(self, task):
        raise ConnectionError("backend down")


def test_e3_stub_logs_when_hitl_disabled(off_memory, stub_router, settings, caplog):
    """P2-09: previously E2/E3 emitted NOTHING when HITL was off, unlike
    E1/E4's existing 'escalation.stub' lines — this proves parity without
    changing routing (the run still reaches telemetry normally, no
    interrupt)."""
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


# ---------------------------------------------------------------------------
# P2-03 follow-up — idempotent corpus ingest (scripts/ingest_sample_data.py
# never actually used QdrantStore.upsert_texts's id_fn parameter, so every
# re-run duplicated the dense-leg corpus even after P2-03 landed)
# ---------------------------------------------------------------------------


def _load_ingest_script():
    """Import scripts/ingest_sample_data.py by file path — it's a script,
    not a package member, so it can't be imported with a normal `import`
    statement from tests/."""
    import importlib.util

    script_path = (pathlib.Path(__file__).parent.parent
                  / "scripts" / "ingest_sample_data.py")
    spec = importlib.util.spec_from_file_location("ingest_sample_data", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_content_id_is_deterministic_for_same_content():
    mod = _load_ingest_script()
    item = {"content": "Redis is an in-memory data store.", "title": "Redis"}
    assert mod.content_id(item) == mod.content_id(dict(item))


def test_content_id_differs_for_different_content():
    mod = _load_ingest_script()
    a = {"content": "Redis is an in-memory data store."}
    b = {"content": "Cassandra is a distributed database."}
    assert mod.content_id(a) != mod.content_id(b)


def test_content_id_is_a_valid_qdrant_point_id_shape():
    # Qdrant point ids must be an unsigned int or a UUID string — a raw
    # hash digest would be rejected outright. uuid.uuid5(...) guarantees
    # this shape; this test would fail loudly if that ever changed to a
    # plain hexdigest by mistake.
    import uuid as uuid_module

    mod = _load_ingest_script()
    result = mod.content_id({"content": "anything"})
    parsed = uuid_module.UUID(result)  # raises ValueError if not a real UUID
    assert str(parsed) == result
