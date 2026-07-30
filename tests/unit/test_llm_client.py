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


def test_prompt_tag_covers_every_node_that_calls_an_llm():
    """Every node this codebase's graph.py actually wires an LLM through
    is expected to be in the table. Written against the KNOWN node names
    from orchestration/graph.py rather than iterating PROMPT_VERSIONS
    itself, so a typo'd or accidentally-removed entry is caught -- testing
    against the registry's own keys would just prove it equals itself."""
    from research_agent.llm.client import _prompt_tag_for_node
    llm_calling_nodes = {"classify", "goal_manager", "task_expander",
                          "gap_generator", "compiler", "critic"}
    for node in llm_calling_nodes:
        tag = _prompt_tag_for_node(node)
        assert "prompt_name" in tag and "prompt_version" in tag, (
            f"{node!r} calls an LLM but has no PROMPT_VERSIONS entry")


def test_prompt_tag_is_empty_not_placeholder_for_an_untagged_node():
    """merger sometimes calls detect_contradictions and sometimes calls
    nothing -- it is deliberately absent from the table (see the
    registry's own docstring) rather than mis-tagged. Absent keys, not a
    placeholder value, so a Langfuse query grouping by prompt_name can
    tell "tagged" from "untagged" apart."""
    from research_agent.llm.client import _prompt_tag_for_node
    assert _prompt_tag_for_node("merger") == {}
    assert _prompt_tag_for_node(None) == {}
    assert _prompt_tag_for_node("some_future_node") == {}


def test_prompt_tag_values_match_the_templates_registry_exactly():
    from research_agent.llm.client import _prompt_tag_for_node
    from research_agent.prompts.templates import PROMPT_VERSIONS
    for node, (name, version) in PROMPT_VERSIONS.items():
        assert _prompt_tag_for_node(node) == {
            "prompt_name": name, "prompt_version": version}


def test_stub_json_fence_tolerance():
    """Regression guard for _extract_json's fence stripping."""
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
