"""
tests/integration/test_model_knowledge_fallback.py — D-38 end to end.

The regression these lock down is the system's worst failure mode: a run
whose corpus did not contain the subject reported the SUBJECT as
unanswerable, rather than reporting its own retrieval as insufficient.
Runs p205.66/.68/.71-check all ended that way.
"""

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.orchestration.graph import build_graph
from research_agent.state import Evidence, ResearchState, Volatility
from research_agent.tools.model_knowledge import make_model_knowledge_tool
from research_agent.tools.retrieval_chain import make_retrieval_chain


class KnowledgeableStub(StubClient):
    """Corpus is useless; the MODEL knows the answer -- the exact situation
    a 10-document Redis corpus is in when asked about armies."""

    def complete(self, messages, temperature=0.2):
        if "TASK=recall" in messages[-1]["content"]:
            return json.dumps({"claims": [
                {"text": "The Indian Army fields roughly 1.2 million active "
                         "personnel.", "confidence": 0.9},
                {"text": "The PLA Ground Force fields roughly 1 million "
                         "active personnel.", "confidence": 0.85},
                {"text": "A half-remembered figure.", "confidence": 0.2},
            ]})
        return super().complete(messages, temperature)


def _junk_corpus(task):
    """Real hits, at exactly the single-leg RRF ceiling -- so they can
    never clear the coverage floor. This is the live shape, not a stub
    convenience: every off-corpus run in p205 produced exactly this."""
    return [Evidence(task_key=task.key, goal_id=task.goal_id, source="corpus",
                     content="Redis is an in-memory data store.", score=0.5,
                     volatility=Volatility.SEMI_STABLE)]


@pytest.fixture
def ladder_settings():
    return Settings(_env_file=None, llm_mode="stub", max_depth=3, max_fanout=6,
                    max_revisions=2, hitl_enabled=True, min_evidence_score=0.5,
                    qdrant_url="http://127.0.0.1:1",
                    postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


def _run(settings, off_memory, corpus=_junk_corpus, thread="d38"):
    router = FallbackRouter([KnowledgeableStub()], 0.6)
    model = (make_model_knowledge_tool(router, settings.model_knowledge_score)
             if settings.model_knowledge_enabled else None)
    tool = make_retrieval_chain(corpus, settings.min_evidence_score, model=model)
    graph = build_graph(router, tool, off_memory, settings, MemorySaver())
    return graph.invoke(
        ResearchState(raw_query="Compare Indian and Chinese army on battlefield"),
        config={"configurable": {"thread_id": thread}, "recursion_limit": 60})


def test_off_corpus_query_is_answered_instead_of_escalated(off_memory,
                                                           ladder_settings):
    result = _run(ladder_settings, off_memory)
    assert "__interrupt__" not in result, (
        "an off-corpus query must be ANSWERED from the model tier, not "
        "escalated to a human who has no more corpus to offer")
    assert result["recall_score"] == 1.0
    assert result["final_report"]
    assert result["telemetry"]


def test_telemetry_keeps_corpus_recall_separate_from_recall(off_memory,
                                                            ladder_settings):
    """Answering from recollection is legitimate and attributed -- but it
    must never be invisible. A reader has to be able to tell that the
    corpus contributed nothing."""
    telemetry = _run(ladder_settings, off_memory, thread="d38-tel")["telemetry"]
    assert telemetry["recall"] == 1.0
    assert telemetry["corpus_recall"] == 0.0
    assert telemetry["model_sourced_items"] > 0
    assert telemetry["evidence_by_source"]["model"] > 0


def test_model_evidence_is_never_relabelled_as_corpus(off_memory,
                                                      ladder_settings):
    result = _run(ladder_settings, off_memory, thread="d38-src")
    sources = {e.source for e in result["evidence"]}
    assert "model" in sources
    for e in result["evidence"]:
        if e.source == "model":
            assert "Redis" not in e.content


def test_a_real_document_still_beats_recollection(off_memory, ladder_settings):
    """The ladder must not reach for the model tier when the corpus can
    actually answer -- otherwise D-38 would quietly degrade every run that
    was already working."""
    def good_corpus(task):
        # A genuinely relevant document shares vocabulary with the query --
        # that is what makes it relevant, and what the D-38 relevance gate
        # tests for. Echoing the query is the honest way to stub that.
        return [Evidence(task_key=task.key, goal_id=task.goal_id,
                         source="corpus",
                         content=f"A relevant document about {task.query}",
                         score=0.99, volatility=Volatility.SEMI_STABLE)]

    result = _run(ladder_settings, off_memory, corpus=good_corpus,
                  thread="d38-doc")
    assert {e.source for e in result["evidence"]} == {"corpus"}
    assert result["telemetry"]["corpus_recall"] == 1.0


def test_disabling_the_model_tier_restores_the_old_corpus_only_behaviour(
        off_memory, ladder_settings):
    """The escape hatch has to actually work: MODEL_KNOWLEDGE_ENABLED=false
    must give back exactly the pre-D-38 posture, escalation included."""
    settings = ladder_settings.model_copy(
        update={"model_knowledge_enabled": False})
    result = _run(settings, off_memory, thread="d38-off")
    assert "__interrupt__" in result, (
        "with the ladder disabled an off-corpus run should still escalate")
