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


def test_single_leg_ceiling_still_matches_the_real_rrf_constants():
    """Drift guard. SINGLE_LEG_SCORE_CEILING is hardcoded in templates.py
    (deliberately -- see its comment for why it does not import from the
    retrieval stack), so this test does the cross-module import instead. If
    RRF_K or RRF_SQUASH ever change, the threshold rots silently and every
    WEAK verdict becomes wrong; this fails loudly instead."""
    from research_agent.prompts.templates import SINGLE_LEG_SCORE_CEILING
    from research_agent.retrieval.hybrid import RRF_K
    from research_agent.tools.corpus_search import RRF_SQUASH

    assert SINGLE_LEG_SCORE_CEILING == min(1.0, (1 / RRF_K) * RRF_SQUASH)


def test_compile_report_states_per_goal_evidence_coverage():
    """The observed failure: 41 evidence items all scoring exactly 0.50,
    per-item scores already inlined, and the model wrote a long confident
    report of unretrievable specifics anyway. An explicit per-goal verdict
    is harder to read past than 41 repetitions of score=0.50."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Evidence, Goal

    goals = [Goal(goal_id="g1", description="well covered"),
             Goal(goal_id="g2", description="single-leg only"),
             Goal(goal_id="g3", description="nothing at all")]
    evidence = [
        Evidence(task_key="a", goal_id="g1", source="corpus", content="x", score=0.98),
        Evidence(task_key="b", goal_id="g1", source="corpus", content="y", score=0.50),
        Evidence(task_key="c", goal_id="g2", source="corpus", content="z", score=0.50),
    ]
    body = compile_report("q", goals, evidence, [])[-1]["content"]

    assert "EVIDENCE: 2 item(s), best score 0.98" in body
    assert "WEAK" in body                      # g2: best score sits ON the ceiling
    assert "NO EVIDENCE RETRIEVED" in body     # g3: never retrieved anything
    # g1 is strong and must NOT be flagged -- a warning that fires on
    # healthy goals trains the model to ignore it.
    g1_block = body.split("- g1:")[1].split("- g2:")[0]
    assert "WEAK" not in g1_block


def test_compile_report_forbids_filling_gaps_from_model_knowledge():
    """The instruction has to be specific about WHAT must not be invented.
    'State clearly when evidence is thin' was the previous wording and the
    model satisfied it while still naming equipment, doctrines and dates
    absent from the corpus."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [], [])[-1]["content"]
    assert "GROUNDING RULE" in body
    for forbidden in ("model numbers", "doctrine names", "figures"):
        assert forbidden in body
    assert "Do NOT fill the gap" in body


def test_compile_report_coverage_block_handles_zero_evidence_overall():
    """A run where retrieval returned nothing at all must still render."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [], [])[-1]["content"]
    assert "NO EVIDENCE RETRIEVED" in body
    assert "(no evidence gathered)" in body
