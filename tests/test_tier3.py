"""
tests/test_tier3.py — Tier 3 Batch 1 coverage: P2-11 (judge-model quality
scoring) and P2-12 (semantic contradiction detector).

Same offline philosophy as the rest of this suite: StubClient/fake
routers/tools, no network, no real Qdrant/OpenSearch/Postgres.
"""

import logging

import pytest

from research_agent.agents.gathering import build_merger_node
from research_agent.config import Settings
from research_agent.evaluation.quality import score_answer
from research_agent.llm.router import FallbackRouter
from research_agent.state import Evidence, Goal, ResearchState, Volatility

# ---------------------------------------------------------------------------
# P2-11 — judge-model quality scoring
# ---------------------------------------------------------------------------


class _FixedScoreJudge:
    """A ChatClient stub that always reports a fixed quality score,
    regardless of what answer it's asked to judge — enough to prove
    score_answer reads the score from WHATEVER `judge` it's given, not
    from some other, hidden "self" the answer came from (there is no
    such hidden self any more — see evaluation/quality.py's P2-11 note)."""

    def __init__(self, score):
        self._score = score

    def complete_json(self, messages, temperature=0.0):
        return {"score": self._score}


def test_score_answer_uses_the_judge_passed_in():
    judge = _FixedScoreJudge(0.3)
    score = score_answer(judge, [{"role": "user", "content": "q"}], "some answer")
    assert score == 0.3


def test_score_answer_clamps_out_of_range_scores():
    assert score_answer(_FixedScoreJudge(1.7), [], "x") == 1.0
    assert score_answer(_FixedScoreJudge(-4.0), [], "x") == 0.0


def test_score_answer_fails_open_when_judge_errors():
    class _BrokenJudge:
        def complete_json(self, messages, temperature=0.0):
            raise RuntimeError("judge is down")

    # Fail-open by design (see module docstring): a broken judge must never
    # take down a working answer path.
    assert score_answer(_BrokenJudge(), [], "x") == 1.0


def test_score_answer_invokes_on_score_failed_only_when_judge_errors():
    """P2-11 follow-up: the callback exists so a caller can count
    "couldn't be scored" separately from "scored low" — confirm it fires
    on the error path and stays silent on a genuine (even low) score."""
    class _BrokenJudge:
        def complete_json(self, messages, temperature=0.0):
            raise RuntimeError("judge is down")

    calls = []
    score = score_answer(_BrokenJudge(), [], "x", on_score_failed=lambda: calls.append(1))
    assert score == 1.0
    assert calls == [1]


def test_score_answer_does_not_invoke_on_score_failed_on_a_genuine_low_score():
    calls = []
    score = score_answer(_FixedScoreJudge(0.1), [], "x", on_score_failed=lambda: calls.append(1))
    assert score == 0.1
    assert calls == []


# Regression coverage for the router-level behavior change (P2-11: the judge
# is always the NEXT provider in the chain, never the one being judged) lives
# in tests/test_core.py, alongside the existing FallbackRouter fallback
# tests it directly modifies:
#   - test_chain_steps_on_low_quality_then_serves_next
#   - test_chain_ignores_providers_own_self_report_as_judge


class _SimpleAnswerer:
    """Minimal ChatClient: always answers the same fixed text, never
    errors. Used as the ANSWERING provider (position 0) in the
    llm_quality_calls_failed tests below — its own quality is never
    self-scored (P2-11), so its complete_json is never even exercised
    as a judge here."""

    name = "primary"

    def complete(self, messages, temperature=0.2):
        return "primary answer"

    def complete_json(self, messages, temperature=0.0):
        return {}


class _AlwaysErroringJudge:
    """A ChatClient whose complete_json (the method score_answer calls)
    always raises — simulating exactly what the real Gemini 429 did:
    the JUDGE, not the answering provider, is the one that's down."""

    name = "judge"

    def complete(self, messages, temperature=0.2):
        return "judge answer"  # only used if this ever became the answerer

    def complete_json(self, messages, temperature=0.0):
        raise RuntimeError("judge is down")


class _GoodJudge:
    """A ChatClient whose complete_json always scores well."""

    name = "judge"

    def complete(self, messages, temperature=0.2):
        return "judge answer"

    def complete_json(self, messages, temperature=0.0):
        return {"score": 0.9}


def test_router_bumps_llm_quality_calls_failed_when_judge_errors():
    """End-to-end version of the exact live-run shape this follow-up was
    written for: the answering provider succeeds, the NEXT provider in
    the chain (the judge) is unreachable. The answer must still be kept
    (fail-open), and the failure must now be visible in telemetry as
    llm_quality_calls_failed — not just as a "quality.score_failed" log
    line with no counter behind it."""
    router = FallbackRouter([_SimpleAnswerer(), _AlwaysErroringJudge()],
                           quality_threshold=0.6)
    answer = router.complete([{"role": "user", "content": "x"}])
    assert answer == "primary answer"

    drained = router.drain_counters()
    assert drained["llm_quality_calls"] == 1        # the attempt was made
    assert drained["llm_quality_calls_failed"] == 1  # and it couldn't be scored
    assert drained.get("llm_fallback_hops", 0) == 0  # fail-open kept the answer


def test_router_never_bumps_llm_quality_calls_failed_on_a_genuine_score():
    """Regression guard: a working judge that scores normally must never
    touch the new counter, whatever the score — it's reserved for
    "couldn't be scored", not "scored something"."""
    router = FallbackRouter([_SimpleAnswerer(), _GoodJudge()], quality_threshold=0.6)
    router.complete([{"role": "user", "content": "x"}])

    drained = router.drain_counters()
    assert drained["llm_quality_calls"] == 1
    assert drained.get("llm_quality_calls_failed", 0) == 0


def test_telemetry_surfaces_llm_quality_calls_failed_end_to_end(settings):
    """Regression guard for the exact gap a live run found: the counter
    can be correctly bumped in state.counters (proven by the router-level
    tests above) and STILL never appear in the printed/persisted telemetry,
    because telemetry_node (agents/compilation.py) builds its output dict
    by explicitly enumerating keys rather than passing state.counters
    through wholesale. This drives a real graph run with StubClient as the
    answering provider (so planning/goal composition succeeds normally,
    same as every other full-graph test in this suite) and
    _AlwaysErroringJudge as the NEXT provider in the chain — so it's only
    ever consulted to SCORE the compiler's answer, never to produce one —
    then asserts the key is actually present in state.telemetry, not just
    in the raw counters dict some earlier test already checked."""
    from langgraph.checkpoint.memory import MemorySaver

    from research_agent.llm.client import StubClient
    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.orchestration.graph import build_graph
    from research_agent.state import Evidence, ResearchState, Volatility
    from research_agent.storage.qdrant_store import QdrantStore

    router = FallbackRouter([StubClient(), _AlwaysErroringJudge()], quality_threshold=0.6)

    def fake_tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="fake",
                         content=f"fact about {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    memory = SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                            settings.memory_top_k, 90.0, 14.0)

    g = build_graph(router, fake_tool, memory, settings, MemorySaver())
    result = g.invoke(ResearchState(raw_query="q"), config={"configurable": {"thread_id": "t"}})

    assert "llm_quality_calls_failed" in result["telemetry"]
    assert result["telemetry"]["llm_quality_calls_failed"] >= 1


# ---------------------------------------------------------------------------
# P2-12 — semantic contradiction detector
# ---------------------------------------------------------------------------


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
# P2-10 — Qdrant payload indexes + server-side decay
#
# IMPORTANT LIMITATION, stated plainly: none of these tests run against a
# real Qdrant server — this whole suite is offline by design (see
# conftest.py's module docstring), and no live Qdrant was reachable in the
# environment these were written in either. What IS verified here:
#   1. The FormulaQuery/DecayParamsExpression/etc. objects construct
#      without a Pydantic validation error against the ACTUAL installed
#      qdrant-client version (not guessed, not copied from the design
#      doc's Appendix C, which a prior external review is on record for
#      having invented API symbols in).
#   2. The exact shape of what gets sent (index field names/types, filter
#      values, scale/midpoint numbers) matches what P2-10 was scoped to do.
#   3. A NUMERIC PARITY check: if Qdrant evaluates an exp_decay formula
#      per its documented semantics (output = midpoint ** (|x-target|/scale),
#      i.e. exactly 0.5 at |x-target|==scale when midpoint=0.5), the value
#      it would return is mathematically identical to decay_factor()'s
#      Python output for the same age/half-life. This proves the FORMULA
#      is correct; it does NOT prove Qdrant's server actually executes it
#      this way — that step needs a live run against a real server (see
#      PHASE2_TIER3_IMPLEMENTATION_PLAN.md's P2-10 risk note).
# Before trusting this in production: run search_with_decay against a real
# Qdrant (client 1.18.0 / server 1.17.1 confirmed compatible) and compare
# its output to the Python path on the same corpus.
# ---------------------------------------------------------------------------


from unittest.mock import MagicMock

from research_agent.memory.semantic_memory import decay_factor
from research_agent.storage.qdrant_store import QdrantStore


def _mock_store(collection="test_collection"):
    """A QdrantStore with a real (degraded, unreachable) __init__ pass, then
    forced "available" with a MagicMock in place of the real qdrant-client
    connection — lets us test the Qdrant-API-CALLING code paths (something
    no existing test in this suite does; every prior QdrantStore test only
    ever exercised the degraded no-op path or a standalone helper function)
    without a real server."""
    store = QdrantStore("http://127.0.0.1:1", collection)
    assert store.available is False  # sanity: really did fail to connect
    store.available = True
    store._client = MagicMock()
    store._embedder = MagicMock()
    # A fresh 3-float vector per call — NOT a fixed return_value, which
    # would hand back the SAME (single-use) iterator object on every call
    # and silently break the second of any two _embed() calls in one test.
    store._embedder.embed = MagicMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    # get_collections()... .collections defaults to an empty iterator on a
    # MagicMock, so ensure_collection() always takes the "create it" path
    # — fine for these tests, which only care about what's SENT, not the
    # collection-exists check.
    return store


def test_ensure_payload_indexes_creates_the_two_required_indexes():
    from qdrant_client import models

    store = _mock_store()
    store.ensure_payload_indexes()

    calls = store._client.create_payload_index.call_args_list
    assert len(calls) == 2
    fields = {c.kwargs["field_name"]: c.kwargs["field_schema"] for c in calls}
    assert fields["created_at_iso"] == models.PayloadSchemaType.DATETIME
    assert fields["volatility"] == models.PayloadSchemaType.KEYWORD


def test_ensure_payload_indexes_is_a_noop_when_degraded():
    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    store.ensure_payload_indexes()  # must not raise


def test_ensure_payload_indexes_fails_open_on_client_error(caplog):
    import logging

    store = _mock_store()
    store._client.create_payload_index = MagicMock(side_effect=RuntimeError("qdrant is down"))
    with caplog.at_level(logging.WARNING):
        store.ensure_payload_indexes()  # must not raise
    assert any("qdrant.index_creation_failed" in r.message for r in caplog.records)


def test_ensure_collection_calls_ensure_payload_indexes_every_time():
    """P2-10: unlike collection creation itself (only on first use),
    payload-index creation must run on EVERY ensure_collection() call —
    it's what guarantees the indexes exist even for a collection that
    already existed from before P2-10 shipped."""
    store = _mock_store()
    store.ensure_collection()
    store.ensure_collection()
    assert store._client.create_payload_index.call_count == 4  # 2 indexes x 2 calls


def test_upsert_texts_writes_both_created_at_and_created_at_iso():
    store = _mock_store()
    store.upsert_texts([{"content": "fact one"}])

    upsert_call = store._client.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[-1]
    payload = points[0].payload
    assert "created_at" in payload and isinstance(payload["created_at"], float)
    assert "created_at_iso" in payload
    # Round-trips as a real ISO/RFC3339 instant — this is exactly the shape
    # DatetimeKeyExpression needs; a malformed string here would silently
    # never match server-side, not raise loudly, so this is worth checking.
    import datetime as _dt
    _dt.datetime.fromisoformat(payload["created_at_iso"])


def test_search_with_decay_returns_empty_list_when_degraded():
    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    out = store.search_with_decay("q", top_k=3, decay_field="volatility",
                                  half_lives={"stable": None, "semi_stable": 90.0})
    assert out == []


def test_search_with_decay_builds_a_valid_formula_query_and_correct_scales():
    """The real test of the P2-10 risk this item was flagged for: does the
    FormulaQuery this method builds actually validate against the REAL
    installed qdrant-client Pydantic models (not a hand-rolled dict that
    only LOOKS right)? A ValidationError here would surface immediately —
    this test doesn't even need to inspect the object's shape to prove
    that much, though it does so anyway for the scale-math check below."""
    from qdrant_client import models

    store = _mock_store()
    fake_point = MagicMock()
    fake_point.payload = {"content": "x", "volatility": "semi_stable",
                          "created_at": 1700000000.0,
                          "created_at_iso": "2023-11-14T22:13:20+00:00"}
    fake_point.score = 0.42
    fake_response = MagicMock()
    fake_response.points = [fake_point]
    store._client.query_points = MagicMock(return_value=fake_response)

    out = store.search_with_decay(
        "q", top_k=3, decay_field="volatility",
        half_lives={"stable": None, "semi_stable": 90.0, "volatile": 14.0})

    # Constructed without error (no ValidationError raised above) AND
    # produced a result in the expected shape.
    assert out == [{
        "content": "x", "volatility": "semi_stable",
        "created_at": 1700000000.0, "created_at_iso": "2023-11-14T22:13:20+00:00",
        "similarity": 0.42,
        "age_days": out[0]["age_days"],  # computed from wall-clock "now"; not asserting an exact value
    }]

    call = store._client.query_points.call_args
    formula = call.kwargs["query"]
    assert isinstance(formula, models.FormulaQuery)
    outer = formula.formula
    assert isinstance(outer, models.MultExpression)
    assert outer.mult[0] == "$score"
    decay_sum = outer.mult[1]
    assert isinstance(decay_sum, models.SumExpression)
    assert len(decay_sum.sum) == 3  # one branch per half_lives entry

    # Find the semi_stable branch and check its scale is EXACTLY
    # half_life_days * 86400 seconds — the conversion this method's
    # docstring promises.
    semi_branch = next(
        b for b in decay_sum.sum
        if b.mult[0].must[0].match.value == "semi_stable")
    exp_decay = semi_branch.mult[1]
    assert isinstance(exp_decay, models.ExpDecayExpression)
    assert exp_decay.exp_decay.scale == 90.0 * 86400.0
    assert exp_decay.exp_decay.midpoint == 0.5
    assert exp_decay.exp_decay.x.datetime_key == "created_at_iso"
    assert exp_decay.exp_decay.target.datetime == "now"

    # The "stable" branch (half_life=None) must be a flat 1.0 multiplier,
    # not a decay expression — confirms the None-means-no-decay contract.
    stable_branch = next(
        b for b in decay_sum.sum
        if b.mult[0].must[0].match.value == "stable")
    assert stable_branch.mult[1] == 1.0


def test_search_with_decay_prefetch_overfetches_by_the_given_multiplier():
    from qdrant_client import models

    store = _mock_store()
    fake_response = MagicMock()
    fake_response.points = []
    store._client.query_points = MagicMock(return_value=fake_response)

    store.search_with_decay("q", top_k=5, decay_field="volatility",
                            half_lives={"stable": None}, overfetch=4)

    call = store._client.query_points.call_args
    prefetch = call.kwargs["prefetch"]
    assert isinstance(prefetch, models.Prefetch)
    assert prefetch.limit == 20  # top_k(5) * overfetch(4)
    assert call.kwargs["limit"] == 5  # final cut is still just top_k


# --- Numeric parity: server-side formula math vs. the Python parity oracle ---


def _qdrant_exp_decay_semantics(age_days: float, half_life_days: float) -> float:
    """Reimplements Qdrant's DOCUMENTED exp_decay semantics in pure Python:
    output = midpoint ** (|x - target| / scale), which at midpoint=0.5 and
    scale = half_life_days * 86400 seconds, with |x-target| = age in
    seconds, reduces to exactly decay_factor()'s formula. This function
    exists ONLY to make that equivalence an explicit, checkable claim — it
    is not a substitute for confirming the real server computes it this
    way (see this section's module-level note)."""
    scale_seconds = half_life_days * 86400.0
    age_seconds = age_days * 86400.0
    return 0.5 ** (age_seconds / scale_seconds)


def test_formula_math_matches_python_decay_factor_for_semi_stable():
    from research_agent.state import Volatility

    for age_days in (0.0, 1.0, 45.0, 90.0, 180.0, 365.0):
        server_value = _qdrant_exp_decay_semantics(age_days, half_life_days=90.0)
        python_value = decay_factor(age_days, Volatility.SEMI_STABLE,
                                    half_life_semi=90.0, half_life_volatile=14.0)
        assert abs(server_value - python_value) < 1e-9, (
            f"mismatch at age_days={age_days}: server={server_value} python={python_value}")


def test_formula_math_matches_python_decay_factor_for_volatile():
    from research_agent.state import Volatility

    for age_days in (0.0, 1.0, 7.0, 14.0, 28.0, 60.0):
        server_value = _qdrant_exp_decay_semantics(age_days, half_life_days=14.0)
        python_value = decay_factor(age_days, Volatility.VOLATILE,
                                    half_life_semi=90.0, half_life_volatile=14.0)
        assert abs(server_value - python_value) < 1e-9, (
            f"mismatch at age_days={age_days}: server={server_value} python={python_value}")


def test_formula_math_exp_decay_hits_exactly_midpoint_at_scale():
    """Sanity check on the semantics claim itself: at age == half_life
    (i.e. |x-target| == scale), exp_decay's documented output is EXACTLY
    the midpoint (0.5 here) — same invariant decay_factor()'s own
    docstring states ("exactly 0.5 when age == half_life")."""
    assert abs(_qdrant_exp_decay_semantics(90.0, half_life_days=90.0) - 0.5) < 1e-12


# --- SemanticMemory.retrieve: server-side gate on/off ---


class _FakeDecayStore:
    """Minimal fake satisfying exactly what SemanticMemory.retrieve calls
    on `self.store` — .search() and/or .search_with_decay(), nothing else."""

    def __init__(self, search_hits=None, decay_hits=None):
        self._search_hits = search_hits or []
        self._decay_hits = decay_hits or []
        self.search_calls = 0
        self.search_with_decay_calls = 0

    def search(self, query, top_k):
        self.search_calls += 1
        return self._search_hits

    def search_with_decay(self, query, top_k, decay_field, half_lives):
        self.search_with_decay_calls += 1
        return self._decay_hits


def test_retrieve_gate_off_uses_the_python_path_unchanged():
    from research_agent.memory.semantic_memory import SemanticMemory

    store = _FakeDecayStore(search_hits=[
        {"content": "c1", "goal_id": "g1", "volatility": "semi_stable",
         "similarity": 0.9, "age_days": 10.0},
    ])
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0,
                        server_side_decay=False)
    out = mem.retrieve("q")

    assert store.search_calls == 1
    assert store.search_with_decay_calls == 0
    assert len(out) == 1
    assert out[0].content == "c1"


def test_retrieve_gate_on_uses_search_with_decay_and_skips_double_decay():
    """The critical correctness check: on the server-side path, the hit's
    "similarity" must be used AS THE FINAL SCORE directly — decay_factor()
    must NOT be called again, or the item would be decayed twice."""
    from research_agent.memory.semantic_memory import SemanticMemory

    store = _FakeDecayStore(decay_hits=[
        {"content": "c1", "goal_id": "g1", "volatility": "semi_stable",
         "similarity": 0.42, "age_days": 999.0},  # age_days deliberately
        # implausible for a fresh 0.42 score if decay were re-applied on
        # top — catches an accidental double-decay immediately.
    ])
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0,
                        server_side_decay=True)
    out = mem.retrieve("q")

    assert store.search_with_decay_calls == 1
    assert store.search_calls == 0
    assert len(out) == 1
    assert out[0].score == 0.42  # untouched — exactly what search_with_decay returned


# ---------------------------------------------------------------------------
# P2-15 — content-identity dedup for memory writes + scroll_all/delete_points
#
# Scope note, stated plainly (matches PHASE2_TIER3_IMPLEMENTATION_PLAN.md's
# P2-15 risk callout): identity here is EXACT content-hash only. There is no
# separate Evidence.superseded_by field — "supersession" is achieved simply
# by upsert-by-id overwriting the same point in place (see
# memory/semantic_memory.py::store_run's docstring for the full reasoning).
# Near-duplicate/paraphrased-fact detection remains explicitly out of scope.
# ---------------------------------------------------------------------------


from research_agent.storage.qdrant_store import content_id as _content_id


class _FakeUpsertStore:
    """Minimal fake exposing only what store_run actually calls:
    .upsert_texts(items, id_fn=...) and (P2-15 follow-up)
    .existing_point_ids(ids). Records every upsert_texts call so tests can
    inspect exactly what id_fn was passed and what it produces.
    `preexisting_ids` lets a test simulate "these ids already exist in
    Qdrant" for the new/overwritten split -- empty by default (everything
    looks new), matching a store_run call against a brand-new memory
    collection."""

    def __init__(self, preexisting_ids=None):
        self.calls = []
        self._preexisting_ids = set(preexisting_ids or [])

    def upsert_texts(self, items, id_fn=None):
        self.calls.append((items, id_fn))
        return len(items)

    def existing_point_ids(self, ids):
        return {i for i in ids if i in self._preexisting_ids}


def _one_evidence(content, goal_id="g1"):
    from research_agent.state import Evidence, Volatility

    return [Evidence(task_key="t1", goal_id=goal_id, source="corpus", content=content,
                     score=0.9, volatility=Volatility.SEMI_STABLE)]


def test_store_run_passes_a_content_based_id_fn():
    from research_agent.memory.semantic_memory import SemanticMemory

    store = _FakeUpsertStore()
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    mem.store_run("q", _one_evidence("Redis is fast"))

    assert len(store.calls) == 1
    items, id_fn = store.calls[0]
    assert id_fn is not None
    assert id_fn(items[0]) == _content_id("Redis is fast")


def test_store_run_id_fn_collapses_identical_content_across_two_separate_calls():
    """The actual bug this closes: a fact re-discovered from the corpus on
    two different runs used to get two different random ids (two separate
    points, accumulating forever). Same content -> same id, regardless of
    which query or goal it was filed under either time."""
    from research_agent.memory.semantic_memory import SemanticMemory

    store = _FakeUpsertStore()
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    mem.store_run("first query", _one_evidence("same fact", goal_id="g1"))
    mem.store_run("second query, different run", _one_evidence("same fact", goal_id="g7"))

    (items1, id_fn1), (items2, id_fn2) = store.calls
    assert id_fn1(items1[0]) == id_fn2(items2[0])


def test_store_run_id_fn_differs_for_different_content():
    from research_agent.memory.semantic_memory import SemanticMemory

    store = _FakeUpsertStore()
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    mem.store_run("q", _one_evidence("fact A"))
    mem.store_run("q", _one_evidence("fact B"))

    (items1, id_fn1), (items2, id_fn2) = store.calls
    assert id_fn1(items1[0]) != id_fn2(items2[0])


def test_store_run_logs_all_items_as_new_when_nothing_preexisted(caplog):
    import logging as _logging

    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.storage.qdrant_store import content_id

    store = _FakeUpsertStore()  # empty preexisting_ids -> everything is new
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    with caplog.at_level(_logging.INFO):
        mem.store_run("q", _one_evidence("brand new fact"))

    stored_lines = [r for r in caplog.records if r.message == "memory.stored"]
    assert len(stored_lines) == 1
    assert stored_lines[0].event_fields["new"] == 1
    assert stored_lines[0].event_fields["overwritten"] == 0
    assert stored_lines[0].event_fields["count"] == 1


def test_store_run_logs_overwritten_when_the_id_already_existed(caplog):
    import logging as _logging

    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.storage.qdrant_store import content_id

    already_there = content_id("known fact")
    store = _FakeUpsertStore(preexisting_ids={already_there})
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    with caplog.at_level(_logging.INFO):
        mem.store_run("q", _one_evidence("known fact"))

    stored_lines = [r for r in caplog.records if r.message == "memory.stored"]
    assert stored_lines[0].event_fields["new"] == 0
    assert stored_lines[0].event_fields["overwritten"] == 1


def test_store_run_splits_a_mixed_batch_correctly(caplog):
    import logging as _logging

    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.state import Evidence, Volatility
    from research_agent.storage.qdrant_store import content_id

    already_there = content_id("old fact")
    store = _FakeUpsertStore(preexisting_ids={already_there})
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="corpus", content="old fact",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
        Evidence(task_key="t2", goal_id="g1", source="corpus", content="brand new fact",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
    ]
    with caplog.at_level(_logging.INFO):
        mem.store_run("q", evidence)

    stored_lines = [r for r in caplog.records if r.message == "memory.stored"]
    assert stored_lines[0].event_fields["new"] == 1
    assert stored_lines[0].event_fields["overwritten"] == 1
    assert stored_lines[0].event_fields["count"] == 2


def test_existing_point_ids_returns_only_the_ids_qdrant_actually_has():
    class _Rec:
        def __init__(self, id_):
            self.id = id_

    store = _mock_store()
    store._client.retrieve = MagicMock(return_value=[_Rec("id1"), _Rec("id3")])

    result = store.existing_point_ids(["id1", "id2", "id3"])

    assert result == {"id1", "id3"}
    call = store._client.retrieve.call_args
    assert call.args[0] == "test_collection"
    assert call.kwargs["ids"] == ["id1", "id2", "id3"]
    assert call.kwargs["with_payload"] is False
    assert call.kwargs["with_vectors"] is False


def test_existing_point_ids_returns_empty_set_when_degraded():
    from research_agent.storage.qdrant_store import QdrantStore

    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    assert store.existing_point_ids(["id1"]) == set()


def test_existing_point_ids_fails_open_on_client_error(caplog):
    import logging as _logging

    store = _mock_store()
    store._client.retrieve = MagicMock(side_effect=RuntimeError("qdrant is down"))
    with caplog.at_level(_logging.WARNING):
        result = store.existing_point_ids(["id1"])

    assert result == set()
    assert any("qdrant.existing_point_ids_failed" in r.message for r in caplog.records)


def test_content_id_still_filters_memory_sourced_evidence_before_dedup_even_applies():
    """Regression guard: the PRE-EXISTING `fresh = [... if e.source !=
    "memory"]` filter must still run before id_fn is ever considered —
    P2-15 must not change what counts as "fresh" in the first place, only
    how fresh items get their point id."""
    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.state import Evidence, Volatility

    store = _FakeUpsertStore()
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="corpus", content="fresh fact",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
        Evidence(task_key="t2", goal_id="g1", source="memory", content="recalled fact",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
    ]
    mem.store_run("q", evidence)

    items, _ = store.calls[0]
    assert len(items) == 1
    assert items[0]["content"] == "fresh fact"


# --- scroll_all / delete_points ---


def test_scroll_all_follows_pagination_until_offset_is_none():
    store = _mock_store()
    r1, r2, r3 = MagicMock(), MagicMock(), MagicMock()
    r1.id, r1.payload = "id1", {"content": "a"}
    r2.id, r2.payload = "id2", {"content": "b"}
    r3.id, r3.payload = "id3", {"content": "c"}
    store._client.scroll = MagicMock(side_effect=[([r1, r2], "page2token"), ([r3], None)])

    out = store.scroll_all(batch_size=2)

    assert out == [
        {"id": "id1", "content": "a"},
        {"id": "id2", "content": "b"},
        {"id": "id3", "content": "c"},
    ]
    assert store._client.scroll.call_count == 2


def test_scroll_all_returns_empty_list_when_degraded():
    from research_agent.storage.qdrant_store import QdrantStore

    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    assert store.scroll_all() == []


def test_delete_points_calls_client_delete_with_the_given_ids():
    store = _mock_store()
    n = store.delete_points(["id1", "id2"])

    assert n == 2
    call = store._client.delete.call_args
    assert call.args[0] == "test_collection"
    assert call.kwargs["points_selector"] == ["id1", "id2"]


def test_delete_points_is_a_noop_on_empty_list_or_when_degraded():
    from research_agent.storage.qdrant_store import QdrantStore

    degraded = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert degraded.delete_points(["id1"]) == 0  # degraded -> no client call possible

    store = _mock_store()
    assert store.delete_points([]) == 0
    assert store._client.delete.called is False


def test_delete_points_fails_open_on_client_error(caplog):
    import logging as _logging

    store = _mock_store()
    store._client.delete = MagicMock(side_effect=RuntimeError("qdrant is down"))
    with caplog.at_level(_logging.WARNING):
        n = store.delete_points(["id1"])

    assert n == 0
    assert any("qdrant.delete_points_failed" in r.message for r in caplog.records)


# --- scripts/gc_memory.py::find_gc_candidates ---
#
# gc_memory.py is a standalone operational script (like reset_stores.py,
# which has no pytest coverage of its own at all) -- but unlike
# reset_stores.py, its actual decision logic (find_gc_candidates) is a
# pure function worth testing directly, so it's loaded by file path here,
# the same way tests/test_tier2.py used to load ingest_sample_data.py
# before content_id moved into the regular package.


def _load_gc_script():
    import importlib.util
    import pathlib

    script_path = (pathlib.Path(__file__).parent.parent
                  / "scripts" / "gc_memory.py")
    spec = importlib.util.spec_from_file_location("gc_memory", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_gc_candidates_flags_old_volatile_points():
    import time as _time

    gc_memory = _load_gc_script()
    store = _mock_store()
    now = _time.time()
    old_volatile = MagicMock(id="old_volatile",
                             payload={"content": "stale fact", "volatility": "volatile",
                                     "created_at": now - 200 * 86400.0})
    store._client.scroll = MagicMock(return_value=([old_volatile], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert len(candidates) == 1
    assert candidates[0][0] == "old_volatile"


def test_find_gc_candidates_spares_fresh_points():
    import time as _time

    gc_memory = _load_gc_script()
    store = _mock_store()
    now = _time.time()
    fresh = MagicMock(id="fresh_volatile",
                      payload={"content": "new fact", "volatility": "volatile",
                              "created_at": now - 1 * 86400.0})
    store._client.scroll = MagicMock(return_value=([fresh], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert candidates == []


def test_find_gc_candidates_never_flags_stable_points_regardless_of_age():
    """D-24's own reasoning: stable facts don't fade at all -- decay_factor
    returns exactly 1.0 for Volatility.STABLE regardless of age (see that
    function's docstring), so no threshold ever catches one."""
    import time as _time

    gc_memory = _load_gc_script()
    store = _mock_store()
    now = _time.time()
    ancient_stable = MagicMock(id="old_stable",
                               payload={"content": "old but stable", "volatility": "stable",
                                       "created_at": now - 5000 * 86400.0})
    store._client.scroll = MagicMock(return_value=([ancient_stable], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert candidates == []


def test_find_gc_candidates_defaults_missing_volatility_to_semi_stable():
    """Consistency with SemanticMemory.retrieve's own fallback for the
    same payload gap -- not a second, different guess."""
    import time as _time

    gc_memory = _load_gc_script()
    store = _mock_store()
    now = _time.time()
    # No "volatility" key at all in the payload. 500 days at the default
    # 90-day semi_stable half-life decays to 0.5**(500/90) ~= 0.011, well
    # past the 0.05 threshold (200 days, tried first, only decays to
    # ~0.214 -- still well ABOVE threshold, which is exactly why this
    # needed a real number check rather than an assumed one).
    untagged = MagicMock(id="untagged",
                         payload={"content": "no volatility tag",
                                 "created_at": now - 500 * 86400.0})
    store._client.scroll = MagicMock(return_value=([untagged], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert len(candidates) == 1
    assert candidates[0][0] == "untagged"