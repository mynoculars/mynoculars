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


def test_system_prompt_forbids_echoing_the_evidence_tag_literally():
    """Regression: a live trace showed a fallback provider (Mistral)
    reading the <evidence> fencing tags added for prompt-injection
    hardening (M-5) and echoing them back verbatim as a bogus citation
    format, e.g. "[g1 | corpus | score=0.98](<evidence>)" -- the model
    imitated a token it saw in its own context window. This doesn't
    weaken the fencing itself (that instruction stays); it just adds one
    more explicit constraint so a model that fixates on the tag as
    "content" is told plainly that it isn't. Not something a unit test
    can force a live model to obey, but it can confirm the guard is
    actually present in what gets sent."""
    system_content = templates._SYSTEM["content"]
    assert "<evidence>" in system_content  # the tag itself is still named...
    assert "never reproduce" in system_content.lower()  # ...but now with a
    # matching instruction not to echo it back literally.
