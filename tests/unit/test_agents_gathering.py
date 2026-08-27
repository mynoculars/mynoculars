"""
tests/unit/test_agents_gathering.py — agents/gathering.py's
build_merger_node.

Covers ONLY the P2-12 semantic contradiction gate: off by default
(marker-only behavior unchanged), on calls an LLM-backed detector but
only for goals with 2+ evidence items (cost control), and fails open if
the detector itself errors. Does NOT cover the rest of the gather loop
(dispatch/coverage/gap-generation) — those are exercised through full
graph runs in tests/integration/test_graph_end_to_end.py.
"""

import logging

from research_agent.agents.gathering import (
    _uncovered_goal_has_strong_evidence,
    build_gap_generator_node,
    build_merger_node,
    build_progress_checker_node,
)
from research_agent.config import Settings
from research_agent.state import Evidence, Goal, ResearchState, Volatility


class _FakeContradictionRouter:
    """Minimal fake satisfying merger_node's actual usage of `router`:
    set_node(name), complete_json(messages), drain_counters() — nothing
    else is called on it. Records whether it was ever invoked, so tests can
    assert the gate genuinely skipped the LLM call when it should have."""

    def __init__(self, contested_goal_ids=None, raise_error=False):
        self._contested = contested_goal_ids or []
        self._raise = raise_error
        self.calls = 0

    def set_node(self, node):
        pass

    def complete_json(self, messages):
        self.calls += 1
        if self._raise:
            raise RuntimeError("detector is down")
        return {"contested_goal_ids": self._contested}

    def drain_counters(self):
        return {"llm_provider_calls": 1.0}


def _settings(contradiction_detection_enabled: bool) -> Settings:
    return Settings(_env_file=None, llm_mode="stub",
                    contradiction_detection_enabled=contradiction_detection_enabled,
                    qdrant_url="http://127.0.0.1:1",
                    postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


def _state_with_two_goals_one_multi_evidence(contradicts_marker=None):
    """g1 has 2 evidence items (the "multi-evidence" case the LLM path
    should actually fire for); g2 has only 1 (should never trigger a call
    on its own)."""
    goals = [Goal(goal_id="g1", description="a"), Goal(goal_id="g2", description="b")]
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="corpus", content="claim A",
                 score=0.9, volatility=Volatility.SEMI_STABLE, contradicts=contradicts_marker),
        Evidence(task_key="t2", goal_id="g1", source="corpus", content="claim B",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
        Evidence(task_key="t3", goal_id="g2", source="corpus", content="claim C",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
    ]
    return ResearchState(raw_query="q", goals=goals, evidence=evidence)


def test_gate_off_preserves_original_marker_only_behavior():
    """Default (off): merger_node must behave EXACTLY as it did before
    P2-12 — only an explicit Evidence.contradicts marker sets `contested`,
    and the LLM is never consulted."""
    settings = _settings(contradiction_detection_enabled=False)
    router = _FakeContradictionRouter(contested_goal_ids=["g1"])  # would fire if consulted
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence(contradicts_marker=None)
    result = node(state)

    assert router.calls == 0, "gate is off — the detector must never be called"
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is False
    assert goals_by_id["g2"].contested is False
    assert result["counters"]["contradictions_flagged"] == 0.0


def test_gate_off_still_honors_explicit_contradicts_marker():
    """The pre-P2-12 marker path is untouched when the gate is off — this
    is what proves the change is additive, not a rewrite."""
    settings = _settings(contradiction_detection_enabled=False)
    router = _FakeContradictionRouter()
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence(contradicts_marker="t2")
    result = node(state)

    assert router.calls == 0
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is True
    assert goals_by_id["g2"].contested is False


def test_gate_on_calls_detector_and_marks_only_contested_goal():
    settings = _settings(contradiction_detection_enabled=True)
    router = _FakeContradictionRouter(contested_goal_ids=["g1"])
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence()
    result = node(state)

    assert router.calls == 1
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is True
    assert goals_by_id["g2"].contested is False  # only 1 evidence item, never even asked about
    assert result["counters"]["contradictions_flagged"] == 1.0
    assert result["counters"]["llm_node_calls"] == 1.0


def test_gate_on_skips_the_llm_call_when_no_goal_has_multiple_evidence_items():
    """Early-exit: a goal with 0 or 1 evidence items can't contradict
    itself, so if NO goal qualifies, the detector must never be invoked —
    this is the cost-control path, not just an optimization detail."""
    settings = _settings(contradiction_detection_enabled=True)
    router = _FakeContradictionRouter(contested_goal_ids=["g1"])
    node = build_merger_node(router, settings)

    goals = [Goal(goal_id="g1", description="a")]
    evidence = [Evidence(task_key="t1", goal_id="g1", source="corpus",
                         content="claim A", score=0.9, volatility=Volatility.SEMI_STABLE)]
    state = ResearchState(raw_query="q", goals=goals, evidence=evidence)
    result = node(state)

    assert router.calls == 0
    assert result["goals"][0].contested is False


def test_gate_on_fails_open_when_detector_errors(caplog):
    """A broken detector must never take the run down — same fail-open
    posture evaluation/quality.py's score_answer already uses."""
    settings = _settings(contradiction_detection_enabled=True)
    router = _FakeContradictionRouter(raise_error=True)
    node = build_merger_node(router, settings)

    state = _state_with_two_goals_one_multi_evidence()
    with caplog.at_level(logging.WARNING):
        result = node(state)

    assert router.calls == 1
    goals_by_id = {g.goal_id: g for g in result["goals"]}
    assert goals_by_id["g1"].contested is False  # fails open -> nothing contested
    assert any("merger.contradiction_detection_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _uncovered_goal_has_strong_evidence / gap_generator's no-strong-evidence skip
# ---------------------------------------------------------------------------
# WHY THIS EXISTS: found live. BM25 has no relevance floor (MIN_SIMILARITY
# only gates the dense leg -- see retrieval/hybrid.py's own docstring), so a
# generic word in an off-topic query can still keyword-match an unrelated
# corpus. Every such hit is single-leg and scores at or below the exact RRF
# rank-0 single-leg value (SINGLE_LEG_SCORE_CEILING, imported from
# prompts/templates.py -- the same constant Item 11's compile_report
# grounding rule uses, not a new threshold). gap_generator used to hand that
# evidence to the model as context regardless of relevance, and the model
# would write new queries themed on whatever subject the tail evidence was
# actually about -- e.g. asked to compare India and the US, it wrote "Redis
# and Memcached licensing models" because that was the only text in the
# tail evidence. Those new queries then matched the (irrelevant) corpus at
# HIGH confidence and the run reported recall: 1.0 / grounding_ratio: 1.0 on
# a topic the corpus never covered.

def _fake_router_that_must_not_be_called():
    class _Router:
        def set_node(self, name):
            pass

        def complete_json(self, messages):
            raise AssertionError(
                "gap_generator must not call the LLM when no uncovered "
                "goal has strong evidence -- that is the whole point of "
                "this guard")

        def drain_counters(self):
            return {}

    return _Router()


def _fake_router_returning(tasks):
    class _Router:
        def set_node(self, name):
            pass

        def complete_json(self, messages):
            return {"tasks": tasks}

        def drain_counters(self):
            return {}

    return _Router()


def test_uncovered_goal_has_strong_evidence_is_a_pure_score_check():
    ev_weak = [Evidence(task_key="a", goal_id="g1", source="corpus",
                        content="x", score=0.5)]
    ev_strong = [Evidence(task_key="b", goal_id="g1", source="corpus",
                          content="y", score=0.501)]
    assert _uncovered_goal_has_strong_evidence("g1", ev_weak) is False
    assert _uncovered_goal_has_strong_evidence("g1", ev_strong) is True
    # A goal with no evidence at all behaves the same as one with only weak
    # evidence -- both mean "nothing the coverage rule can trust."
    assert _uncovered_goal_has_strong_evidence("g1", []) is False
    # Evidence for a DIFFERENT goal must not count.
    assert _uncovered_goal_has_strong_evidence(
        "g2", [Evidence(task_key="c", goal_id="g1", source="corpus",
                        content="z", score=0.9)]) is False


def test_gap_generator_skips_the_llm_call_when_every_uncovered_goal_is_weak():
    """The exact live shape: uncovered goals whose only surviving evidence
    is single-leg (BM25-only) and therefore <= 0.5. The fake router raises
    if complete_json is ever invoked, so this proves the call is skipped,
    not merely that its OUTPUT is discarded afterward."""
    settings = Settings(_env_file=None, hitl_enabled=False,
                        recall_target=0.85, max_fanout=6,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="macro", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated keyword-only hit", score=0.48)],
        recall_score=0.667, iteration_depth=2,
    )
    node = build_gap_generator_node(_fake_router_that_must_not_be_called(),
                                    settings, debug=False)
    result = node(state)
    assert result["pending_tasks"] == []


def test_gap_generator_raises_e3_when_hitl_enabled_and_evidence_is_weak():
    settings = Settings(_env_file=None, hitl_enabled=True,
                        recall_target=0.85, max_fanout=6,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="macro", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated", score=0.48)],
        recall_score=0.667, iteration_depth=2,
    )
    node = build_gap_generator_node(_fake_router_that_must_not_be_called(),
                                    settings, debug=False)
    result = node(state)
    assert result["escalation_trigger"] == "E3"


def test_gap_generator_still_calls_the_llm_when_some_uncovered_goal_is_strong():
    """The guard must not fire just because ONE goal among several is weak
    -- only when EVERY uncovered goal lacks strong evidence. A single
    genuinely-covered-by-both-legs goal is enough reason to keep looping."""
    settings = Settings(_env_file=None, hitl_enabled=False,
                        recall_target=0.85, max_fanout=6,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="q",
        goals=[Goal(goal_id="g1", description="a", covered=False),
               Goal(goal_id="g2", description="b", covered=False)],
        evidence=[
            Evidence(task_key="t1", goal_id="g1", source="corpus",
                     content="weak", score=0.48),
            Evidence(task_key="t2", goal_id="g2", source="corpus",
                     content="strong, both legs agreed", score=0.9),
        ],
        recall_score=0.5, iteration_depth=1,
    )
    node = build_gap_generator_node(
        _fake_router_returning([{"query": "q2", "goal_id": "g1", "priority": 1}]),
        settings, debug=False)
    result = node(state)
    assert len(result["pending_tasks"]) == 1


def test_gap_generator_still_calls_the_llm_when_no_goals_are_uncovered():
    """An empty uncovered list must not trip the guard -- `any([])` is
    False, so the guard's `uncovered and not any(...)` condition correctly
    requires uncovered to be non-empty first."""
    settings = Settings(_env_file=None, hitl_enabled=False,
                        recall_target=0.85, max_fanout=6,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="q",
        goals=[Goal(goal_id="g1", description="a", covered=True)],
        evidence=[], recall_score=1.0, iteration_depth=1,
    )
    node = build_gap_generator_node(
        _fake_router_returning([]), settings, debug=False)
    result = node(state)   # must not raise
    assert result["pending_tasks"] == []


# ---------------------------------------------------------------------------
# The no-strong-evidence guard must defer to a human redirect
# ---------------------------------------------------------------------------
def test_guard_is_bypassed_when_human_guidance_is_set():
    settings = Settings(_env_file=None, hitl_enabled=False,
                        recall_target=0.85, max_fanout=6,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="macro", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated keyword-only hit", score=0.48)],
        recall_score=0.667, iteration_depth=1,
        human_guidance="compare social and political aspects instead",
    )
    node = build_gap_generator_node(
        _fake_router_returning([{"query": "q2", "goal_id": "g1", "priority": 1}]),
        settings, debug=False)
    result = node(state)
    assert len(result["pending_tasks"]) == 1
    assert result["human_guidance"] == ""


def test_guard_still_fires_without_guidance_even_after_a_prior_redirect():
    settings = Settings(_env_file=None, hitl_enabled=True,
                        recall_target=0.85, max_fanout=6,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="macro", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated", score=0.48)],
        recall_score=0.667, iteration_depth=2,
        human_guidance="",
    )
    node = build_gap_generator_node(_fake_router_that_must_not_be_called(),
                                    settings, debug=False)
    result = node(state)
    assert result["escalation_trigger"] == "E3"


# ---------------------------------------------------------------------------
# P205 regression: the guard must not pre-empt the depth budget
# ---------------------------------------------------------------------------


def test_guard_does_not_fire_on_the_first_gather_cycle():
    """Runs p205.66/67/68-check all ended at iterations=1 with MAX_DEPTH
    entirely unused, because this guard ran before the gap generator had
    produced a single new query formulation. D-3/D-14 make depth the loop's
    bound; the guard may confirm non-convergence, never pre-empt it."""
    settings = Settings(_env_file=None, hitl_enabled=True,
                        recall_target=0.85, max_fanout=6, max_depth=3)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="macro", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated keyword-only hit", score=0.48)],
        recall_score=0.667, iteration_depth=1,
    )
    node = build_gap_generator_node(_fake_router_returning([{"query": "fresh q", "goal_id": "g1", "priority": 1}]), settings, debug=False)
    result = node(state)
    assert result.get("escalation_trigger") is None, (
        "the guard must not escalate before the gap generator has had one "
        "real attempt -- MAX_DEPTH is the bound, not cycle 1")
    assert result["pending_tasks"], "cycle 1 must actually produce tasks"


def test_guard_does_not_fire_when_an_uncovered_goal_retrieved_nothing():
    """Run p205.68-check: g2 retrieved zero items, which made the
    strong-evidence test vacuously False and fired the guard on the exact
    case it was never written for. A goal with no evidence has no
    misleading tail to free-associate off -- it is a retry candidate."""
    settings = Settings(_env_file=None, hitl_enabled=True,
                        recall_target=0.85, max_fanout=6, max_depth=3)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="weak", covered=False),
               Goal(goal_id="g2", description="starving", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated", score=0.48)],
        recall_score=0.5, iteration_depth=2,
    )
    node = build_gap_generator_node(_fake_router_returning([{"query": "fresh q", "goal_id": "g1", "priority": 1}]), settings, debug=False)
    result = node(state)
    assert result.get("escalation_trigger") is None
    assert result["pending_tasks"]


def test_escalation_budget_stops_the_guard_re_raising_forever():
    """route_convergence and dispatch_tasks both test escalation_trigger
    BEFORE their terminal exits, so a re-raised E2/E3 re-enters
    human_escalation instead of terminating. Once the per-run review budget
    is spent the node must fall through to the compiler path instead."""
    settings = Settings(_env_file=None, hitl_enabled=True, recall_target=0.85,
                        max_fanout=6, max_depth=3, max_escalations=1,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="weak", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="unrelated", score=0.48)],
        recall_score=0.667, iteration_depth=2,
        escalation_history=[{"trigger": "E3", "action": "redirect"}],
    )
    node = build_gap_generator_node(_fake_router_that_must_not_be_called(),
                                    settings, debug=False)
    result = node(state)
    assert result["pending_tasks"] == []
    assert result.get("escalation_trigger") is None, (
        "budget spent -> empty backlog -> dispatch_tasks routes to compiler, "
        "which writes an honest partial report instead of nagging again")


# ---------------------------------------------------------------------------
# Guardrail G2: progress_checker_node's topical grounding gate
# ---------------------------------------------------------------------------


def test_g2_topical_gate_rejects_on_topic_score_off_topic_content():
    """Regression target: run p205.132-check. gap_generator emitted a task
    tagged g1 (this run's actual goal: GDP/inflation/unemployment) whose
    query drifted onto the sample corpus's real content (Redis vs
    Memcached). That corpus hit scored well above the floor and, before
    this fix, counted as "grounded" for g1 purely on source+score --
    exactly the failure mode telemetry_node's corpus_recall already
    guards against via a topical overlap check. This proves
    progress_checker_node now applies the SAME gate, so the two grounding
    signals cannot disagree the way they did in that run
    (grounded_score moved 0.0 -> 0.2 while corpus_recall correctly held
    at 0.0)."""
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="GDP growth rate comparison "
                                              "between India and US")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="Redis and Memcached both support "
                                   "session caching with different "
                                   "eviction policies.", score=0.9)],
    )
    node = build_progress_checker_node(settings, debug=False)
    result = node(state)
    assert result["recall_score"] == 1.0, "covered -- score alone clears the floor"
    assert result["grounded_score"] == 0.0, (
        "off-topic despite clearing the floor -- must not count as grounded")


def test_first_cycle_writes_the_no_previous_cycle_sentinel():
    """D-80 regression. On the FIRST cycle there is no previous
    grounded_score measurement, so progress_checker_node must write
    grounded_score_prev's -1.0 "no previous cycle yet" sentinel -- NOT
    state.grounded_score, which at that moment is still its 1.0
    construction default (state.py picks 1.0 so a zero-goal run never
    reads as falsely ungrounded) and was never measured by anything.

    Copying that default recorded a phantom "the previous cycle scored
    1.0", which made route_convergence's stall check fire on cycle 1 for
    every ungrounded run. See test_orchestration_graph.py's
    test_first_ungrounded_cycle_loops_instead_of_reporting_a_stall for the
    composed failure this defect actually produced."""
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[Goal(goal_id="g1",
                    description="PLA size versus Indian Army size")],
        # source="web" COVERS a goal but never GROUNDS one (D-57) -- exactly
        # the shape run p205.246-check produced: recall 1.0, grounded 0.0.
        evidence=[Evidence(task_key="t1", goal_id="g1", source="web",
                           content="The PLA fields roughly two million "
                                   "active personnel; the Indian Army "
                                   "around 1.2 million.", score=0.7)],
    )
    result = build_progress_checker_node(settings, debug=False)(state)

    assert state.grounded_score == 1.0, (
        "precondition: the untouched 1.0 default this test exists to stop "
        "being copied into grounded_score_prev")
    assert result["grounded_score"] == 0.0, "web evidence never grounds (D-57)"
    assert result["grounded_score_prev"] == -1.0, (
        "first cycle must report the sentinel, not the unmeasured default")


def test_later_cycles_report_the_previous_cycles_real_measurement():
    """The other half of D-80: from the SECOND cycle onward,
    grounded_score_prev must carry forward what the PREVIOUS cycle
    genuinely measured, so route_convergence's stall comparison has real
    data on both sides. Only the first cycle is special-cased."""
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[Goal(goal_id="g1",
                    description="PLA size versus Indian Army size")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="web",
                           content="The PLA fields roughly two million "
                                   "active personnel.", score=0.7)],
        # A cycle has already completed and measured 0.25.
        iteration_depth=1,
        grounded_score=0.25,
    )
    result = build_progress_checker_node(settings, debug=False)(state)

    assert result["grounded_score_prev"] == 0.25
    assert result["iteration_depth"] == 2


def test_g2_topical_gate_accepts_genuinely_on_topic_evidence():
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="GDP growth rate comparison "
                                              "between India and US")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="India's GDP growth rate rebounded to "
                                   "8.9 percent.", score=0.9)],
    )
    node = build_progress_checker_node(settings, debug=False)
    result = node(state)
    assert result["grounded_score"] == 1.0


def test_g2_model_sourced_evidence_never_counts_as_grounded_even_on_topic():
    """source="model" is excluded from grounding regardless of topical
    overlap -- topicality only narrows what corpus/mcp items count; it
    does not promote a model-tier recollection into a real document."""
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="GDP growth rate comparison "
                                              "between India and US")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="model",
                           content="India's GDP growth rate rebounded to "
                                   "8.9 percent.", score=0.6)],
    )
    node = build_progress_checker_node(settings, debug=False)
    result = node(state)
    assert result["recall_score"] == 1.0
    assert result["grounded_score"] == 0.0


def test_g2_web_sourced_evidence_covers_a_goal_but_never_grounds_it():
    """THE Phase 4 grounding lock (D-57).

    A web snippet is retrieval, not curation. It may COVER a goal -- it is
    real, current, above the floor, and on topic -- but it must never GROUND
    one, because grounded_score (Guardrail G2 / D-47) answers a different
    question: "did a real DOCUMENT back this?"

    The failure this prevents is silent and severe. make_mcp_tool hardcodes
    source="mcp", and progress_checker_node tests `source in ("corpus",
    "mcp")`. Had web search been exposed through the existing MCP client
    unchanged, every snippet would have counted as a grounded document,
    inflating grounded_score and re-creating precisely the recall=1.0 /
    corpus_recall=0.0 blindness D-43 and D-47 exist to expose. The fix is
    that make_web_search_tool tags source="web", and "web" is deliberately
    absent from that tuple.

    A run answered wholly from the web must therefore read recall 1.0 /
    grounded_score 0.0 -- visible rather than flattering.
    """
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        model_knowledge_enabled=False)
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="GDP growth rate comparison "
                                              "between India and US")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="web",
                           content="India's GDP growth rate rebounded to "
                                   "8.9 percent.", score=0.75,
                           url="https://example.org/gdp",
                           domain="example.org")],
    )
    node = build_progress_checker_node(settings, debug=False)
    result = node(state)
    assert result["recall_score"] == 1.0, "a web hit must still cover a goal"
    assert result["grounded_score"] == 0.0, (
        "a web snippet must never count as a grounded document -- if this "
        'fails, check whether "web" was added to the source tuple in '
        "progress_checker_node")


# ---------------------------------------------------------------------------
# D-59 — gap_generator's target selection and query anchoring
#
# Live regression: run p205.203-check. D-47's grounded-convergence gate
# routed here with recall 1.0, every goal `covered`, and grounded_score 0.0.
# `uncovered` was therefore EMPTY, the prompt rendered "Uncovered goals:
# (none)" while still demanding queries for them, and the only remaining
# topical signal was an evidence tail full of off-topic Redis corpus hits
# under an India-vs-US query. The node returned six consecutive
# Redis/Memcached queries. Both halves are covered below: which goals the
# cycle targets, and whether the prompt names the actual research question.
# ---------------------------------------------------------------------------
class _CapturingRouter:
    """Records the prompt it was handed, so a test can assert on what the
    model would actually have seen rather than only on the node's return."""

    def __init__(self, tasks=None):
        self.tasks = tasks or []
        self.messages = None

    def set_node(self, name):
        pass

    def complete_json(self, messages):
        self.messages = messages
        return {"tasks": self.tasks}

    def drain_counters(self):
        return {}


def _settings_for_gaps():
    return Settings(_env_file=None, hitl_enabled=False, recall_target=0.85,
                    max_fanout=6, model_knowledge_enabled=False,
                    min_evidence_score=0.5)


def test_gap_generator_targets_ungrounded_goals_when_none_are_uncovered():
    """The grounded-gate re-entry path. g1 has a real, on-topic corpus
    document; g2 is covered only by a web snippet. Both are `covered`, so
    `uncovered` is empty -- the cycle must target g2 and only g2."""
    settings = _settings_for_gaps()
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="economy growth", covered=True),
               Goal(goal_id="g2", description="climate policy", covered=True)],
        evidence=[
            Evidence(task_key="t1", goal_id="g1", source="corpus",
                     content="economy growth figures", score=0.9),
            Evidence(task_key="t2", goal_id="g2", source="web",
                     content="climate policy snippet", score=0.7),
        ],
        recall_score=1.0, iteration_depth=1,
    )
    router = _CapturingRouter()
    build_gap_generator_node(router, settings, debug=False)(state)
    body = router.messages[-1]["content"]
    assert "g2: climate policy" in body
    assert "g1: economy growth" not in body


def test_gap_generator_prompt_names_the_original_question():
    """Without this the prompt's only topical content is the evidence tail,
    which is exactly how an India-vs-US run produced Redis queries."""
    settings = _settings_for_gaps()
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="economy", covered=False)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="Redis SLOWLOG introspection", score=0.9)],
        recall_score=0.0, iteration_depth=1,
    )
    router = _CapturingRouter()
    build_gap_generator_node(router, settings, debug=False)(state)
    assert "Compare India and US" in router.messages[-1]["content"]


def test_gap_generator_skips_the_llm_when_nothing_is_uncovered_or_ungrounded():
    """A genuinely converged run has no gap to close, so there is no prompt
    worth paying for. D-1's empty backlog routes to the compiler."""
    settings = _settings_for_gaps()
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="economy growth", covered=True)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="economy growth figures", score=0.9)],
        recall_score=1.0, iteration_depth=1,
    )
    router = _CapturingRouter()
    result = build_gap_generator_node(router, settings, debug=False)(state)
    assert router.messages is None, "no uncovered and no ungrounded goal -> no call"
    assert result["pending_tasks"] == []


def test_gap_generator_still_calls_the_llm_for_a_human_redirect():
    """A redirect is new information the evidence never had. Even a fully
    grounded run must honour it rather than take the skip path above."""
    settings = _settings_for_gaps()
    state = ResearchState(
        raw_query="Compare India and US",
        goals=[Goal(goal_id="g1", description="economy growth", covered=True)],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           content="economy growth figures", score=0.9)],
        recall_score=1.0, iteration_depth=1,
        human_guidance="look at defence spending instead",
    )
    router = _CapturingRouter()
    build_gap_generator_node(router, settings, debug=False)(state)
    assert router.messages is not None
