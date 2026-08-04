"""
tests/unit/test_tools_model_knowledge.py — Guardrail G3
(tools/model_knowledge.py::_looks_overspecific and its wiring into
Evidence.hedge_specific).

Regression target: run p205.131-check, where the model tier produced a
claim pairing a specific year-range with a specific figure --
"India's population grew from approximately 900 million in 1970 to over
1.4 billion in 2020" -- that the critic later rejected as fabricated;
no evidence item stated that figure. Self-reported `confidence` did not
catch it; this deterministic heuristic is the guardrail that flags it
for the compiler instead.
"""

import json

from research_agent.tools.model_knowledge import (_looks_overspecific,
                                                    make_model_knowledge_tool)
from research_agent.state import SearchTask


class _FakeRouter:
    def __init__(self, payload: dict):
        self._payload = payload

    def complete_json(self, messages):
        return self._payload


def _task() -> SearchTask:
    return SearchTask(key="g1::t1", goal_id="g1", query="population trends")


def test_looks_overspecific_flags_year_plus_figure():
    assert _looks_overspecific(
        "India's population grew from approximately 900 million in 1970 "
        "to over 1.4 billion in 2020.")
    assert _looks_overspecific(
        "By 2023, unemployment had fallen to 3.6 percent.")


def test_looks_overspecific_does_not_flag_year_alone():
    assert not _looks_overspecific(
        "China implemented the One-Child Policy in 1979.")


def test_looks_overspecific_does_not_flag_figure_alone():
    assert not _looks_overspecific(
        "The population density is approximately 153 people per square kilometer.")


def test_model_knowledge_tool_sets_hedge_specific_on_flagged_claims():
    router = _FakeRouter({"claims": [
        {"text": "India's population grew from approximately 900 million "
                 "in 1970 to over 1.4 billion in 2020.", "confidence": 0.9},
        {"text": "India has a large and youthful population.",
         "confidence": 0.9},
    ]})
    tool = make_model_knowledge_tool(router, score=0.6)
    evidence = tool(_task())

    assert len(evidence) == 2
    flagged = {e.content: e.hedge_specific for e in evidence}
    assert flagged["India's population grew from approximately 900 million "
                   "in 1970 to over 1.4 billion in 2020."] is True
    assert flagged["India has a large and youthful population."] is False


def test_model_knowledge_tool_never_flags_a_low_confidence_dropped_claim():
    """Confidence<0.5 is dropped entirely before hedge_specific is even
    evaluated -- the two guardrails are independent, not layered."""
    router = _FakeRouter({"claims": [
        {"text": "India's population grew from approximately 900 million "
                 "in 1970 to over 1.4 billion in 2020.", "confidence": 0.2},
    ]})
    tool = make_model_knowledge_tool(router, score=0.6)
    assert tool(_task()) == []
