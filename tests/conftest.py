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
    """Settings with tight bounds and no env-file surprises."""
    return Settings(_env_file=None, llm_mode="stub", max_depth=2,
                    max_fanout=4, max_revisions=2,
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
