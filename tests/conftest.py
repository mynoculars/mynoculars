"""
tests/conftest.py — Shared fixtures: offline graph with stub LLM + fake tool.

Every test runs fully offline: StubClient for the LLM, an in-process fake
retrieval tool, degraded (off) memory, and an in-memory checkpointer.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.state import Evidence, SearchTask, Volatility
from research_agent.storage.qdrant_store import QdrantStore


@pytest.fixture
def settings() -> Settings:
    """Settings with tight bounds and no env-file surprises.

    hitl_enabled=False is passed EXPLICITLY here, not left to the field's
    own default. Reason (found via a real failure, not theoretical):
    Settings(_env_file=None, ...) only skips reading a .env FILE — it does
    NOTHING to insulate against real OS environment variables, which
    pydantic-settings always checks first regardless of _env_file. A
    developer who ran `$env:HITL_ENABLED = "true"` earlier in the SAME
    shell session (e.g. for manual live testing) and then ran pytest in
    that same window would silently get hitl_enabled=True here too — and
    tests that specifically exercise the HITL-OFF path (this fixture is
    used by tests that expect NO interrupt) would instead pause via
    interrupt() and never reach telemetry_node, producing a confusing
    KeyError on state.telemetry (which stays at its default {} for an
    interrupted run) instead of a clear assertion failure. Passing it
    explicitly here makes that class of failure structurally impossible,
    regardless of what's sitting in whoever's shell.
    """
    return Settings(_env_file=None, llm_mode="stub", max_depth=2,
                    max_fanout=4, max_revisions=2, hitl_enabled=False,
                    qdrant_url="http://127.0.0.1:1", postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


@pytest.fixture
def stub_router() -> FallbackRouter:
    """Router in stub mode: deterministic, no fallback."""
    return FallbackRouter([StubClient()], quality_threshold=0.6)


@pytest.fixture
def fake_tool():
    """Retrieval tool returning one high-score evidence item per task."""

    def tool(task: SearchTask):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="fake",
                         content=f"fact about {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    return tool


@pytest.fixture
def off_memory(settings) -> SemanticMemory:
    """Memory over an unreachable Qdrant — exercises degraded (off) mode."""
    return SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                          settings.memory_top_k, 90.0, 14.0)


@pytest.fixture
def graph(stub_router, fake_tool, off_memory, settings):
    """The full compiled workflow, offline."""
    return build_graph(stub_router, fake_tool, off_memory, settings, MemorySaver())
