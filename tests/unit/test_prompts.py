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


def test_compile_report_permits_model_evidence_but_demands_attribution():
    """D-38 replaces the old GROUNDING RULE, which forbade the model from
    using its own knowledge at all. That rule was correct when the corpus
    was the only retrieval tier -- and became the direct cause of "the
    retrieved evidence does not cover it" reports once a model tier
    existed to answer those goals. The rule now has to do three jobs at
    once: permit `model` evidence, force it to be attributed, and still
    forbid inventing specifics no evidence item of ANY source supplied."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [], [])[-1]["content"]
    assert "ATTRIBUTION RULE" in body
    # permits
    assert "Use BOTH" in body
    assert "still answered, not skipped" in body.replace("\n", " ")
    # but demands the label
    assert "general knowledge" in body
    assert "Never present a `model` claim as a retrieved finding" in body
    # and still forbids manufacturing specifics
    for forbidden in ("model numbers", "doctrine names", "figures"):
        assert forbidden in body
    assert "must not appear in the report" in body


def test_critique_does_not_fail_a_report_for_attributed_model_knowledge():
    """The critic ran on the old assumption that anything not in the
    corpus was unfaithful. Left unchanged it would reject every report
    the new model tier makes possible."""
    from research_agent.prompts.templates import critique
    from research_agent.state import Goal

    body = critique("q", "report", [Goal(goal_id="g1", description="d")], [])[-1]["content"]
    assert "is FAITHFUL provided the report attributes it" in body
    assert "do not fail a report merely for using it" in body


def test_model_knowledge_prompt_demands_atomic_claims_and_confidence():
    """Claims become individual Evidence items the compiler cites one by
    one, so a paragraph-shaped answer here would be uncitable; and the
    confidence field is load-bearing -- the caller drops anything under
    0.5, because a shaky recollection that still marks a goal covered is
    worse than no item at all."""
    from research_agent.prompts.templates import model_knowledge

    body = model_knowledge("Compare Indian and Chinese army", 4)[-1]["content"]
    assert "TASK=recall" in body
    assert "SEPARATE, self-contained factual" in body
    assert "confidence" in body
    assert "Return [] if you do not reliably know" in body.replace("\n", " ")


def test_compile_report_coverage_block_handles_zero_evidence_overall():
    """A run where retrieval returned nothing at all must still render."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [], [])[-1]["content"]
    assert "NO EVIDENCE RETRIEVED" in body
    assert "(no evidence gathered)" in body


def test_compose_goals_forbids_background_memory_from_reframing_the_query():
    """D-42 (runs p205.96/.97-check). goal_manager is handed recalled memory
    BEFORE any retrieval happens, so a previous run on an unrelated topic can
    silently re-frame an open question. "Compare India and US" became an
    entirely military goal set because the run before it was about armies."""
    from research_agent.prompts.templates import compose_goals

    body = compose_goals("Comparison", "Compare India and US",
                         ["PLA doctrine prioritises forward deployment"])[-1]["content"]
    assert "UNRELATED" in body
    assert "must not narrow or re-frame the question" in body
    assert "ignore it completely" in body
    assert "AS ASKED" in body


def test_critique_demands_a_check_on_unsupported_specifics():
    """D-43's semantic half. Live (run p205.98-check): with
    model_sourced_items 0 and a ten-document Redis corpus, the report still
    asserted "Netflix operates Cassandra clusters exceeding 1 PB ... ~1
    million writes per second [g5]" and the critic passed it. Deterministic
    code cannot judge whether cited evidence supports a sentence; the critic
    has to be asked."""
    from research_agent.prompts.templates import critique
    from research_agent.state import Goal

    body = critique("q", "r", [Goal(goal_id="g1", description="d")], [])[-1]["content"]
    assert "NAMED ENTITY" in body and "FIGURE" in body
    assert "appears in no evidence item of any source" in body
    assert "check that it does, rather than trusting the marker" in body


def test_generate_gaps_demands_coverage_of_every_uncovered_goal():
    """Live (run p205.100-check): all six tasks in one cycle were tagged g1
    and all six in the next were tagged g5, so most uncovered goals got no
    new query and the extra gather cycles were wasted."""
    from research_agent.prompts.templates import generate_gaps
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="a", covered=False),
             Goal(goal_id="g2", description="b", covered=False)]
    body = generate_gaps(goals, [], 1, 6)[-1]["content"]
    assert "SPREAD them across the goals listed above" in body
    assert "at least one query before giving any goal a second" in body


def test_generate_gaps_names_the_research_question(  # D-59
):
    """Live (run p205.203-check): this prompt contained the goal list and an
    evidence tail and NOTHING naming the actual subject under research. The
    tail was dominated by off-topic Redis corpus hits under an India-vs-US
    query, and the gap generator returned six consecutive Redis/Memcached
    queries -- reading the only subject the prompt still showed it. The
    question must appear, and must appear BEFORE the tail."""
    from research_agent.prompts.templates import generate_gaps
    from research_agent.state import Evidence, Goal

    goals = [Goal(goal_id="g1", description="economy", covered=False)]
    tail = [Evidence(goal_id="g1", task_key="t", content="Redis SLOWLOG",
                     source="corpus", score=0.9)]
    body = generate_gaps(goals, tail, 1, 6,
                         query="Compare India and US")[-1]["content"]
    assert "Compare India and US" in body
    assert body.index("Compare India and US") < body.index("<evidence>")
    # The tail must be labelled as retrieval, not as a topic description.
    assert "NOT a description of the topic" in body


def test_generate_gaps_can_target_goals_that_are_covered_but_ungrounded(  # D-59
):
    """D-47's grounded gate routes to gap_generator with recall already at
    target and every goal `covered`. The old wording rendered "Uncovered
    goals: (none)" and still demanded queries for them -- unanswerable, so
    the evidence tail became the only usable signal. target_goals lets the
    caller name the goals the cycle is actually for."""
    from research_agent.prompts.templates import generate_gaps
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="economy", covered=True),
             Goal(goal_id="g2", description="climate", covered=True)]
    body = generate_gaps(goals, [], 2, 6, target_goals=[goals[1]])[-1]["content"]
    assert "g2: climate" in body
    assert "g1: economy" not in body
    # The GOAL list must be populated. ("(none)" still appears further down
    # as the empty evidence tail, which is correct and unrelated.)
    assert "Goals still needing evidence:\n- g2: climate" in body


def test_no_prompt_carries_a_concrete_worked_example():
    """P205 regression (run p205.107-check). The critique prompt contained
    a worked example -- a report citing [g5] for "Netflix operates
    Cassandra clusters exceeding 1 PB ... ~1 million writes per second" --
    and the critic reported that as something it had FOUND, in a Redis vs
    Memcached run. The compiler then wrote a section denying it. Rationale
    for a rule belongs in the code comment beside it, never in the prompt
    text, because anything in the prompt can surface in the output."""
    from research_agent.prompts import templates
    from research_agent.state import Evidence, Goal

    goals = [Goal(goal_id="g1", description="d", covered=False)]
    ev = [Evidence(task_key="t", goal_id="g1", source="corpus",
                   content="x" * 50, score=0.9)]
    rendered = " ".join(
        m["content"] for fn in (
            lambda: templates.compose_goals("Comparison", "q", ["hint"]),
            lambda: templates.generate_gaps(goals, ev, 1, 6),
            lambda: templates.compile_report("q", goals, ev, []),
            lambda: templates.model_knowledge("q", 4),
            lambda: templates.critique("q", "r", goals, ev),
        ) for m in fn())

    for leaked in ("Netflix", "Balakot", "blobRedis", "score=0.60",
                   "Compare India and US", "Cassandra clusters"):
        assert leaked not in rendered, (
            f"{leaked!r} is a worked example and can surface in output")
        
def test_critique_is_shown_the_evidence_it_must_verify_against():
    """P205 regression (runs p205.111/.112-check). The critic is instructed
    to fail any named entity, figure or date that appears in no evidence
    item -- and critique() was never passed the evidence, so the check was
    unanswerable. Both runs failed with "not supported by any evidence
    item" for figures that WERE supplied by model-tier items the critic
    could not see, costing two revisions and an E4 escalation each."""
    from research_agent.prompts.templates import critique
    from research_agent.state import Evidence, Goal


    ev = [Evidence(task_key="t", goal_id="g1", source="model", score=0.6,
                   content="India's median age is about 28 years")]
    body = critique("q", "r", [Goal(goal_id="g1", description="d")], ev)[-1]["content"]
    assert "<evidence>" in body
    assert "median age is about 28" in body
    assert "| model]" in body, "the source tag must be visible to the critic"
    assert "items tagged `model` are evidence too" in body  
    
def test_critique_forbids_the_query_vs_report_standard_for_named_entities():
    """P205 regression (run p205.117-check). With the critic finally shown
    the evidence (D-46), a live run flagged SLOWLOG, AOF, Sentinel,
    Memcached and 20+ other terms as "unsupported named entities... not
    part of the question" -- checking the report against the QUERY's
    wording instead of the EVIDENCE block, exactly the standard D-46 was
    supposed to replace. Every one of those terms was present in the
    evidence; the critic used the wrong document to judge against."""
    from research_agent.prompts.templates import critique
    from research_agent.state import Goal

    body = critique("q", "r", [Goal(goal_id="g1", description="d")], [])[-1]["content"]
    assert "never QUERY vs REPORT" in body
    assert 'not part of the question' in body
    assert "evidence legitimately introduces vocabulary the query itself never used" in body      


# ---------------------------------------------------------------------------
# D-73 -- revision passes lost citations more often than first compiles
# ---------------------------------------------------------------------------


def test_compile_report_reinforces_citations_on_a_revision_pass():
    """D-73: live-evidenced pattern (runs p205.239/240-check) -- a FIRST
    compile cited correctly, but the REWRITE after a critique failure came
    back with zero [gN] citations, twice in a row, in both runs. The
    critique-notes block must carry an explicit reminder that citations
    are still required, immediately next to the "address every note"
    instruction most likely to crowd it out."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [],
                          ["some critique note"])[-1]["content"]
    assert "STILL cite every" in body
    assert "[gN]" in body.split("STILL cite every")[1][:50]


def test_compile_report_first_pass_has_no_revision_reminder():
    """The reminder is specific to revision passes -- a first compile
    (no critique_notes) must not carry it, since there is nothing being
    "addressed" yet for the reminder to reinforce."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [],
                          [])[-1]["content"]
    assert "STILL cite every" not in body
    assert "A reviewer rejected" not in body


def test_compile_report_revision_reminder_follows_the_critique_notes():
    """Ordering matters: the reminder must come immediately AFTER the
    critique notes list, not buried elsewhere, so it is the last thing
    the model reads before "address every note" and the citation
    requirement compete for its attention."""
    from research_agent.prompts.templates import compile_report
    from research_agent.state import Goal

    body = compile_report("q", [Goal(goal_id="g1", description="d")], [],
                          ["fix the budget figure"])[-1]["content"]
    notes_idx = body.index("fix the budget figure")
    reminder_idx = body.index("STILL cite every")
    citation_format_idx = body.index("CITATION FORMAT")
    assert notes_idx < reminder_idx < citation_format_idx
