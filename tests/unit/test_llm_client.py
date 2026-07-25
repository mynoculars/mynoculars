"""
tests/unit/test_llm_client.py — llm/client.py's StubClient / _extract_json.

Deliberately thin: StubClient itself is exercised implicitly throughout
this whole suite (it IS the default LLM for every offline test), and its
per-TASK canned responses are covered by whatever test actually needs
that behavior (e.g. test_orchestration_graph.py, the integration tests).
This file covers only _extract_json's own parsing robustness, which has
no other natural home.
"""

from research_agent.llm.client import _extract_json


def test_stub_json_fence_tolerance():
    """Regression guard for _extract_json's fence stripping."""
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
