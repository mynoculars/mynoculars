"""
tests/unit/test_prompts.py — prompts/templates.py.

Covers ONLY the P2-14 tool_hint schema gate: when no tool hints are
available (settings.mcp_enabled=False, the default), the task-expansion
prompt itself must carry no "tool_hint" schema at all -- not just that
no task happens to use it. Every other prompt template is exercised
implicitly throughout this suite via StubClient's TASK=<tag> dispatch.
"""

from research_agent.prompts import templates
from research_agent.state import Goal


def test_p2_14_with_mcp_disabled_the_llm_is_never_even_told_about_it():
    """settings.mcp_enabled=False (the default) -- confirms the PROMPT
    itself carries no tool_hint schema at all, not just that no task
    happens to use it. Proven by asserting the actual prompt text sent
    to the router never mentions "tool_hint"."""
    available = frozenset()  # mirrors: frozenset({"mcp"}) if settings.mcp_enabled else frozenset()

    msgs = templates.expand_tasks([Goal(goal_id="g1", description="x")], 5,
                                  available_tool_hints=available)
    assert "tool_hint" not in msgs[1]["content"]
