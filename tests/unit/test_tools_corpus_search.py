"""
tests/unit/test_tools_corpus_search.py — tools/corpus_search.py's
make_corpus_tool.

Covers: the tool closure exposes drain_retrieval_counts (P2-07 follow-
up), duck-typed so callers can surface HybridRetriever's per-call
telemetry without widening the ToolFn return type — and confirms a
plain fake tool (no HybridRetriever behind it, like conftest.py's
fake_tool fixture) correctly does NOT have that attribute, since
agents/gathering.py's search_worker must getattr-guard this rather than
assume every ToolFn has it.
"""

from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import SearchTask
from research_agent.tools.corpus_search import make_corpus_tool


class _FakeLeg:
    """Minimal fake store: only .search() and .available."""

    def __init__(self, hits=None, available=True):
        self._hits = hits or []
        self.available = available

    def search(self, query, top_k):
        return list(self._hits)


def test_corpus_search_tool_exposes_drain_retrieval_counts():
    retriever = HybridRetriever(_FakeLeg(hits=[{"content": "x", "title": "t"}]),
                                _FakeLeg(), min_similarity=0.0)
    tool = make_corpus_tool(retriever, top_k=3)
    task = SearchTask(key="g1::x", query="x", goal_id="g1")
    tool(task)
    assert hasattr(tool, "drain_retrieval_counts")
    counts = tool.drain_retrieval_counts()
    assert counts["retrieval_dense_calls"] == 1
    assert counts["retrieval_keyword_calls"] == 1


def test_fake_tool_fixture_has_no_retrieval_counts_attribute(fake_tool):
    # conftest.py's fake_tool is a plain function, not backed by
    # HybridRetriever — search_worker must not assume every ToolFn has
    # drain_retrieval_counts (see agents/gathering.py's getattr guard).
    assert not hasattr(fake_tool, "drain_retrieval_counts")
