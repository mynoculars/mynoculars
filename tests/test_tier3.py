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

# ---------------------------------------------------------------------------
# P2-13 — MCP tool seam
#
# Unlike every other test in this suite (see conftest.py's module
# docstring: "every test runs fully offline"), ONE test below
# (test_mcp_tool_round_trips_through_a_real_stdio_server) genuinely spawns
# a real subprocess and talks real MCP protocol over real stdio pipes to
# it. This is still fully self-contained and offline in the sense that
# matters (no network call, no external service, no non-deterministic
# dependency) -- the "server" is tests/fixtures/mcp_echo_server.py, a
# ~20-line fixture shipped in this repo, launched and torn down entirely
# within the test itself. This is deliberately NOT mocked: an earlier,
# mock-only version of this work shipped an MCPBridge.close() that raised
# `RuntimeError: Attempted to exit cancel scope in a different task than
# it was entered in` under real use -- a genuine anyio/asyncio structured-
# concurrency constraint (cancel scopes must be entered and exited by the
# SAME task, not just the same event loop/thread) that no amount of
# mocking the SDK's objects would have caught, since a mock doesn't
# enforce anyio's actual runtime invariants. Real subprocess, real pipes,
# real protocol round trip is what caught it, and is kept here specifically
# so a future change to MCPBridge's lifecycle gets the same check.
# ---------------------------------------------------------------------------


def test_split_csv_strips_and_drops_empty_entries():
    from research_agent.config import split_csv

    assert split_csv("a, b ,,c") == ["a", "b", "c"]
    assert split_csv("") == []
    assert split_csv("   ") == []
    assert split_csv("single") == ["single"]


def test_build_subprocess_env_only_includes_allowlisted_names(monkeypatch):
    from research_agent.tools.mcp_client import _build_subprocess_env

    monkeypatch.setenv("MCP_TEST_ALLOWED_VAR", "yes")
    monkeypatch.setenv("MCP_TEST_FORBIDDEN_VAR", "should-not-leak")

    env = _build_subprocess_env(["MCP_TEST_ALLOWED_VAR", "MCP_TEST_NOT_SET_VAR"])

    assert env == {"MCP_TEST_ALLOWED_VAR": "yes"}
    assert "MCP_TEST_FORBIDDEN_VAR" not in env
    assert "MCP_TEST_NOT_SET_VAR" not in env  # allowlisted but never set -> absent, not an error


def test_build_subprocess_env_returns_empty_dict_for_empty_allowlist(monkeypatch):
    from research_agent.tools.mcp_client import _build_subprocess_env

    monkeypatch.setenv("MCP_TEST_SOME_VAR", "x")
    assert _build_subprocess_env([]) == {}


class _FakeContentBlock:
    def __init__(self, text=None):
        self.text = text


class _FakeCallToolResult:
    def __init__(self, content=None, isError=False):
        self.content = content or []
        self.isError = isError


class _FakeBridgeForToolParsing:
    """A fake satisfying exactly what make_mcp_tool's closure calls:
    .call_tool(name, arguments, timeout_seconds=...) -> an object with
    .content / .isError. Never touches a real MCPBridge or real asyncio
    machinery -- this tests ONLY the Evidence-construction/parsing logic
    make_mcp_tool's closure wraps around whatever a bridge returns."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def call_tool(self, name, arguments, timeout_seconds=30.0):
        self.calls.append((name, arguments, timeout_seconds))
        return self._result


def _task(query="q", key="t1", goal_id="g1"):
    from research_agent.state import SearchTask

    return SearchTask(key=key, goal_id=goal_id, query=query, depth=0)


def test_make_mcp_tool_converts_text_content_to_evidence():
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[_FakeContentBlock(text="fact one")])
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search", query_arg_name="query")

    evidence = tool(_task(query="redis vs cassandra", key="t1", goal_id="g1"))

    assert len(evidence) == 1
    assert evidence[0].content == "fact one"
    assert evidence[0].source == "mcp"
    assert evidence[0].task_key == "t1"
    assert evidence[0].goal_id == "g1"
    assert bridge.calls == [("search", {"query": "redis vs cassandra"}, 30.0)]


def test_make_mcp_tool_produces_one_evidence_item_per_text_block():
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[
        _FakeContentBlock(text="fact A"), _FakeContentBlock(text="fact B")])
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search")

    evidence = tool(_task())
    assert [e.content for e in evidence] == ["fact A", "fact B"]


def test_make_mcp_tool_skips_non_text_content_blocks():
    """A content block with no .text (e.g. an image) is skipped, not an
    error -- Evidence in this build is text-only."""
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[
        _FakeContentBlock(text=None), _FakeContentBlock(text="only this one")])
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search")

    evidence = tool(_task())
    assert len(evidence) == 1
    assert evidence[0].content == "only this one"


def test_make_mcp_tool_returns_empty_list_on_tool_reported_error():
    """isError=True is a TOOL-level failure (the server ran fine, the
    tool itself reported nothing useful) -- treated as "no results",
    not raised as an exception."""
    from research_agent.tools.mcp_client import make_mcp_tool

    result = _FakeCallToolResult(content=[_FakeContentBlock(text="ignored")], isError=True)
    bridge = _FakeBridgeForToolParsing(result)
    tool = make_mcp_tool(bridge, "search")

    assert tool(_task()) == []


def test_make_mcp_tool_uses_the_configured_query_arg_name():
    from research_agent.tools.mcp_client import make_mcp_tool

    bridge = _FakeBridgeForToolParsing(_FakeCallToolResult())
    tool = make_mcp_tool(bridge, "search", query_arg_name="search_text")

    tool(_task(query="hello"))
    assert bridge.calls[0][1] == {"search_text": "hello"}


def test_make_mcp_tool_content_is_capped_at_800_chars():
    """Same slicing cap tools/corpus_search.py's corpus_search uses --
    one enormous content block shouldn't dominate the compile prompt."""
    from research_agent.tools.mcp_client import make_mcp_tool

    long_text = "x" * 2000
    bridge = _FakeBridgeForToolParsing(_FakeCallToolResult(content=[_FakeContentBlock(text=long_text)]))
    tool = make_mcp_tool(bridge, "search")

    evidence = tool(_task())
    assert len(evidence[0].content) == 800


def test_mcp_bridge_surfaces_a_clear_error_for_a_nonexistent_command():
    """A bad command must fail fast and clearly (FileNotFoundError, in
    practice), not hang -- proven against the REAL subprocess-spawning
    path, not a mock (a mock could never demonstrate a real spawn
    failure)."""
    from research_agent.tools.mcp_client import MCPBridge

    bridge = MCPBridge(command="this-command-does-not-exist-anywhere",
                       args=[], env_allowlist=[], startup_timeout_seconds=5.0)
    try:
        with __import__("pytest").raises(Exception):
            bridge.call_tool("search", {"query": "x"}, timeout_seconds=5.0)
    finally:
        bridge.close()  # must not itself raise, even after a failed start


def test_mcp_tool_round_trips_through_a_real_stdio_server():
    """The genuine end-to-end proof: a real subprocess, real MCP protocol,
    real stdio pipes, using tests/fixtures/mcp_echo_server.py (a ~20-line
    FastMCP server shipped in this repo, deterministic, no external
    dependencies of its own). This is what caught the anyio cancel-scope
    bug in MCPBridge.close() that no amount of mocking would have --
    see this section's module-level note for the full story."""
    import pathlib
    import sys

    from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool

    server_path = str(pathlib.Path(__file__).parent / "fixtures" / "mcp_echo_server.py")
    bridge = MCPBridge(command=sys.executable, args=[server_path], env_allowlist=[])
    tool = make_mcp_tool(bridge, "search", query_arg_name="query")
    try:
        evidence = tool(_task(query="redis vs cassandra", key="t1", goal_id="g1"))
        assert len(evidence) == 1
        assert evidence[0].content == "canned result for: redis vs cassandra"
        assert evidence[0].source == "mcp"

        # A second call on the SAME bridge proves the connection is
        # actually PERSISTENT (reused), not re-spawned per call -- the
        # whole point of the background-loop design over asyncio.run()
        # per call (see MCPBridge's own docstring).
        evidence2 = tool(_task(query="second query", key="t2", goal_id="g1"))
        assert evidence2[0].content == "canned result for: second query"
    finally:
        bridge.close()  # must succeed cleanly -- this is the regression check


# --- scripts/mcp_corpus_server.py ---
#
# A real MCP server wrapping the EXISTING tools/corpus_search.py tool --
# built because a fair question ("the MCP server just has to call the
# corpus tools, right?") pointed out that tests/fixtures/mcp_echo_server.py
# proves the wiring but returns nothing genuinely useful. This section
# tests scripts/mcp_corpus_server.py's OWN wrapping logic (hits_for_query,
# search) via a fake corpus tool substituted directly into the module's
# _corpus_tool global -- deliberately NOT importing it in a way that would
# trigger the real lazy QdrantStore/OpenSearchStore construction (a real
# run showed that path takes real network round trips to build, correctly
# degrading if unreachable but far too slow for a unit test to eat on
# every run -- see _get_corpus_tool's docstring in that file for the fix
# that made import alone instant).


def _load_mcp_corpus_server():
    import importlib.util
    import pathlib

    script_path = (pathlib.Path(__file__).parent.parent
                  / "scripts" / "mcp_corpus_server.py")
    spec = importlib.util.spec_from_file_location("mcp_corpus_server", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_corpus_tool_returning(*contents):
    """A stand-in for tools/corpus_search.py's own returned closure --
    same ToolFn shape (task in, Evidence list out), fixed canned content
    regardless of the task's actual query, purely for testing
    mcp_corpus_server.py's OWN wrapping logic in isolation."""
    from research_agent.state import Evidence, Volatility

    def tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="corpus",
                         content=c, score=0.9, volatility=Volatility.SEMI_STABLE)
                for c in contents]
    return tool


def test_mcp_corpus_server_imports_instantly_without_a_live_backend():
    """Regression guard for the real bug a manual test caught: importing
    this module must NOT eagerly build a real QdrantStore/OpenSearchStore
    CONNECTION (that used to take ~10s of retry/backoff even though it
    degrades gracefully) -- _corpus_tool must start as None.

    The wall-clock threshold below is deliberately loose (30s, not the
    original 2s). A later fix (see mcp_corpus_server.py's module
    docstring, "First-import gotcha") made this module eagerly IMPORT
    qdrant_client/opensearchpy -- not connect to them -- at module load
    time, on purpose, to avoid a real ~120s stall that happened when
    their first import instead happened lazily on a worker thread during
    a live tool call. That eager import alone genuinely costs several
    seconds (confirmed: ~9s in one environment) even with no live
    backend reachable, which is an accepted, intentional trade-off: a few
    seconds of slower importability in exchange for never silently
    hanging on a real request. This test still fails fast (30s, not
    unbounded) if that cost balloons far beyond what a plain package
    import should cost, and still asserts _corpus_tool stays None -- the
    thing this test actually guards against (an eager CONNECTION attempt)
    is unchanged.
    """
    import time

    t0 = time.time()
    mod = _load_mcp_corpus_server()
    elapsed = time.time() - t0

    assert mod._corpus_tool is None
    assert elapsed < 30.0, f"import took {elapsed}s -- far more than a plain package import should cost"


def test_hits_for_query_wraps_the_corpus_tool_correctly():
    mod = _load_mcp_corpus_server()
    mod._corpus_tool = _fake_corpus_tool_returning("hit one", "hit two")

    result = mod.hits_for_query("redis vs cassandra")

    assert result == ["hit one", "hit two"]


def test_hits_for_query_constructs_a_valid_search_task():
    """The corpus tool receives a real SearchTask, not a bare string --
    confirms the wrapping actually goes through this repo's normal
    SearchTask/Evidence contract, not some shortcut."""
    mod = _load_mcp_corpus_server()
    seen_tasks = []

    def capturing_tool(task):
        seen_tasks.append(task)
        return []

    mod._corpus_tool = capturing_tool
    mod.hits_for_query("my query")

    assert len(seen_tasks) == 1
    assert seen_tasks[0].query == "my query"
    assert seen_tasks[0].key  # non-empty
    assert seen_tasks[0].goal_id  # non-empty


def test_mcp_corpus_server_search_function_matches_hits_for_query():
    """search() (the actual @mcp.tool()-decorated function FastMCP
    exposes) is `async def` (P2-13 Tier 3 concurrency fix: the blocking
    call is offloaded to a thread pool so FastMCP's single event loop
    isn't held up -- see README.md Limitations #6) but must still be a
    thin wrapper -- same result as calling hits_for_query directly, just
    awaited."""
    import asyncio

    mod = _load_mcp_corpus_server()
    mod._corpus_tool = _fake_corpus_tool_returning("a", "b", "c")

    assert asyncio.run(mod.search("q")) == mod.hits_for_query("q") == ["a", "b", "c"]


def test_get_corpus_tool_only_builds_once():
    """The lazy-singleton pattern: _get_corpus_tool must not rebuild on
    every call once _corpus_tool is already set."""
    mod = _load_mcp_corpus_server()
    sentinel = _fake_corpus_tool_returning("x")
    mod._corpus_tool = sentinel

    first = mod._get_corpus_tool()
    second = mod._get_corpus_tool()

    assert first is sentinel
    assert second is sentinel


# --- P2-13 follow-up: evidence_by_source telemetry + worker.done source log ---
#
# Direct answer to "is there any indication content was retrieved via
# MCP": before this, there wasn't a deterministic one -- only an
# indirect, LLM-dependent hint (whether the compiled report's own
# citations happened to preserve a "[goal | mcp | score]" tag). These
# tests cover the two concrete signals added instead.


def test_telemetry_evidence_by_source_reflects_the_standard_test_fixture(graph):
    """The standard `graph` fixture's own fake_tool tags its evidence
    source="fake" (see conftest.py) -- confirms evidence_by_source counts
    whatever string is ACTUALLY on each Evidence item, not a hardcoded
    list of expected sources like "corpus"/"mcp"/"memory"."""
    from research_agent.state import ResearchState

    result = graph.invoke(ResearchState(raw_query="q"),
                          config={"configurable": {"thread_id": "evidence-by-source-fake"}})

    telemetry = result["telemetry"]
    assert telemetry["evidence_by_source"] == {"fake": telemetry["evidence_items"]}


def test_telemetry_evidence_by_source_distinguishes_mcp_from_corpus(settings):
    """The scenario this was actually built for: a tool tagging its
    evidence source="mcp" (tools/mcp_client.py::make_mcp_tool always does
    this) must show up as such in telemetry, distinctly from any other
    source -- proven here with a minimal stand-in tool rather than a full
    MCPBridge, since only the source-counting behavior is under test."""
    from langgraph.checkpoint.memory import MemorySaver

    from research_agent.llm.client import StubClient
    from research_agent.llm.router import FallbackRouter
    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.orchestration.graph import build_graph
    from research_agent.state import Evidence, ResearchState, Volatility
    from research_agent.storage.qdrant_store import QdrantStore

    def mcp_shaped_tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="mcp",
                         content=f"mcp fact about {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    router = FallbackRouter([StubClient()], quality_threshold=0.6)
    memory = SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                            settings.memory_top_k, 90.0, 14.0)
    graph = build_graph(router, mcp_shaped_tool, memory, settings, MemorySaver())

    result = graph.invoke(ResearchState(raw_query="q"),
                          config={"configurable": {"thread_id": "evidence-by-source-mcp"}})

    telemetry = result["telemetry"]
    assert telemetry["evidence_by_source"] == {"mcp": telemetry["evidence_items"]}
    assert "corpus" not in telemetry["evidence_by_source"]


def test_worker_done_log_line_reports_the_tools_actual_source(graph, caplog):
    """Per-task, real-time visibility in a --debug trace: which tool
    answered THIS specific task, not just the run-level aggregate."""
    import logging as _logging

    from research_agent.state import ResearchState

    with caplog.at_level(_logging.INFO):
        graph.invoke(ResearchState(raw_query="q"),
                    config={"configurable": {"thread_id": "worker-done-source"}})

    done_lines = [r for r in caplog.records if r.message == "worker.done"]
    assert done_lines, "expected at least one worker.done log line"
    for line in done_lines:
        assert line.event_fields["source"] == "fake"  # matches conftest.py's fake_tool


def test_mcp_bridge_survives_many_concurrent_first_calls():
    """Regression test for a REAL bug a live run caught: multiple threads
    calling call_tool() at nearly the same moment, before the bridge has
    finished connecting, used to make every thread EXCEPT the one that
    created the background thread skip the readiness wait entirely and
    crash with AttributeError ('NoneType' object has no attribute
    'call_tool') -- exactly reproducing LangGraph fanning several
    search_worker instances out concurrently for one gather-cycle
    superstep. Uses the REAL fixture server (tests/fixtures/
    mcp_echo_server.py) and REAL threads -- a mock could not have caught
    this, since the bug was a genuine race between real OS threads."""
    import pathlib
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool

    server_path = str(pathlib.Path(__file__).parent / "fixtures" / "mcp_echo_server.py")
    bridge = MCPBridge(command=sys.executable, args=[server_path], env_allowlist=[])
    tool = make_mcp_tool(bridge, "search", query_arg_name="query")

    def run_one(i):
        return tool(_task(query=f"concurrent query {i}", key=f"t{i}", goal_id="g1"))

    try:
        # 8 threads all calling the SAME bridge for the first time at
        # once -- exactly the shape that crashed before the fix, at a
        # concurrency level at least as high as this codebase's own
        # MAX_FANOUT default ever produces in one gather cycle.
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run_one, range(8)))

        for i, evidence in enumerate(results):
            assert len(evidence) == 1
            assert evidence[0].content == f"canned result for: concurrent query {i}"
    finally:
        bridge.close()


def test_get_corpus_tool_builds_exactly_once_under_real_concurrent_load():
    """Regression test for a REAL bug a live run caught: the original
    _get_corpus_tool had no lock around its "if _corpus_tool is None:
    build it" check. FastMCP dispatches concurrent tool calls to worker
    threads (this server's search() does blocking Qdrant/OpenSearch
    calls), so six search_worker tasks firing at once -- this codebase's
    normal gather-cycle fan-out -- meant six threads could all see
    _corpus_tool is None SIMULTANEOUSLY and all six would build their OWN
    separate QdrantStore/OpenSearchStore AT THE SAME TIME, turning one
    measured ~13s cold start into six concurrently-competing ones that
    blew past a 30s client-side timeout. Uses REAL threads and a slow
    fake build function (not the real Qdrant/OpenSearch, which isn't
    available in this test environment) to prove only ONE build ever
    happens no matter how many threads race in at once -- the actual
    mechanism under test is the lock, not the real retrieval backend."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    mod = _load_mcp_corpus_server()
    build_count = []
    build_lock = threading.Lock()  # only protects the counter itself, not _get_corpus_tool

    def slow_fake_build():
        with build_lock:
            build_count.append(1)
        time.sleep(0.2)  # long enough that concurrent callers would
                          # overlap if the real lock weren't doing its job
        return _fake_corpus_tool_returning("built")

    mod._build_corpus_tool = slow_fake_build

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: mod._get_corpus_tool(), range(8)))

    assert len(build_count) == 1, (
        f"expected exactly 1 build, got {len(build_count)} -- "
        "the thundering-herd race is back")
    assert all(r is results[0] for r in results), "every caller must get the SAME instance"


def test_mcp_bridge_timeout_error_is_actually_informative():
    """Regression test for a fair, direct complaint: a real failure showed
    up in this codebase's own D-16 failure log as bare "reason=
    TimeoutError" with zero further detail -- concurrent.futures.
    TimeoutError's own message is EMPTY (confirmed: str(TimeoutError())
    == ""), so there was genuinely nothing else to show. This uses a REAL
    server (tests/fixtures/mcp_slow_server.py, which sleeps 5s) and a
    deliberately short 1s timeout to trigger the real timeout path fast
    and deterministically, then checks the raised exception's message
    actually names the tool, the arguments, and how long it waited --
    not just the exception's class name."""
    import pathlib
    import sys

    from research_agent.tools.mcp_client import MCPBridge

    server_path = str(pathlib.Path(__file__).parent / "fixtures" / "mcp_slow_server.py")
    # sys.executable, not a hardcoded "python3" -- FOUND BY A REAL FAILURE:
    # on Windows there's typically no "python3" on PATH at all (the
    # official installer only provides "python.exe"), so this fell back to
    # whichever OTHER Python happened to resolve from PATH -- a completely
    # different interpreter than the venv running pytest, missing the mcp
    # package entirely, which crashed the subprocess immediately (surfacing
    # as a confusing "McpError: Connection closed" rather than the
    # ModuleNotFoundError that was the real cause, visible only in the
    # subprocess's own captured stderr). Every other MCP test in this file
    # already used sys.executable correctly; this one test didn't.
    bridge = MCPBridge(command=sys.executable, args=[server_path], env_allowlist=[])
    try:
        try:
            bridge.call_tool("search", {"query": "redis vs cassandra"}, timeout_seconds=1.0)
            assert False, "expected a TimeoutError"
        except TimeoutError as exc:
            message = str(exc)
            assert message, "the whole point of this fix: the message must NOT be empty"
            assert "search" in message
            assert "redis vs cassandra" in message
            assert "1.0" in message  # the configured timeout, visible in the message
    finally:
        bridge.close()


# ---------------------------------------------------------------------------
# P2-14 — typed specialist workers (D-25)
# ---------------------------------------------------------------------------


def _p214_settings(mcp_enabled: bool):
    from research_agent.config import Settings

    return Settings(_env_file=None, llm_mode="stub", hitl_enabled=False,
                    mcp_enabled=mcp_enabled,
                    qdrant_url="http://127.0.0.1:1",
                    postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


def _p214_memory(settings):
    from research_agent.memory.semantic_memory import SemanticMemory
    from research_agent.storage.qdrant_store import QdrantStore

    return SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
                          settings.memory_top_k, 90.0, 14.0)


def test_cap_and_filter_keeps_a_hint_that_is_in_allowed_tool_hints():
    from research_agent.agents.task_utils import cap_and_filter
    from research_agent.state import ResearchState

    raw = [{"query": "q1", "goal_id": "g1", "priority": 1, "tool_hint": "mcp"}]
    tasks, rejected = cap_and_filter(raw, ResearchState(raw_query="q"), depth=0,
                                     max_fanout=5, allowed_tool_hints=frozenset({"mcp"}))
    assert rejected == 0
    assert tasks[0].tool_hint == "mcp"


def test_cap_and_filter_resets_a_hint_not_in_allowed_tool_hints():
    """The core validation this item exists for: a hint the LLM emitted
    but that isn't actually wired into THIS run must never survive into
    a SearchTask -- it's silently reset to the default, not rejected as
    malformed (the request itself was well-formed; the hint just isn't
    available right now)."""
    from research_agent.agents.task_utils import cap_and_filter
    from research_agent.state import ResearchState

    raw = [{"query": "q1", "goal_id": "g1", "priority": 1, "tool_hint": "mcp"}]
    tasks, rejected = cap_and_filter(raw, ResearchState(raw_query="q"), depth=0,
                                     max_fanout=5, allowed_tool_hints=frozenset())
    assert rejected == 0
    assert tasks[0].tool_hint == ""


def test_cap_and_filter_default_call_with_no_allowed_tool_hints_arg_is_unchanged():
    """Backward compatibility: a caller that doesn't even know about
    allowed_tool_hints yet (the exact old call signature) still gets
    tool_hint="" on everything -- byte-identical pre-P2-14 behavior."""
    from research_agent.agents.task_utils import cap_and_filter
    from research_agent.state import ResearchState

    raw = [{"query": "q1", "goal_id": "g1", "priority": 1}]
    tasks, rejected = cap_and_filter(raw, ResearchState(raw_query="q"), depth=0, max_fanout=5)
    assert tasks[0].tool_hint == ""


def test_build_graph_without_mcp_tool_never_registers_the_specialist_node(graph):
    """graph fixture (conftest.py) never passes mcp_tool -- confirms the
    default shape is completely unchanged from before P2-14."""
    node_names = set(graph.get_graph().nodes.keys())
    assert "mcp_search_worker" not in node_names
    assert "search_worker" in node_names


def test_build_graph_with_mcp_tool_registers_the_specialist_node():
    from langgraph.checkpoint.memory import MemorySaver

    from research_agent.llm.client import StubClient
    from research_agent.llm.router import FallbackRouter
    from research_agent.orchestration.graph import build_graph

    settings = _p214_settings(mcp_enabled=True)
    router = FallbackRouter([StubClient()], quality_threshold=0.6)
    memory = _p214_memory(settings)

    def fake_tool(task):
        return []

    g = build_graph(router, fake_tool, memory, settings, MemorySaver(), mcp_tool=fake_tool)
    node_names = set(g.get_graph().nodes.keys())
    assert "mcp_search_worker" in node_names
    assert "search_worker" in node_names


def test_p2_14_mixed_backlog_routes_to_both_specialists_end_to_end():
    """The real, definitive proof: one full graph run, one task hinted
    "mcp", one task with no hint -- each must land on the RIGHT tool, and
    the resulting evidence must show BOTH sources in telemetry. Uses a
    custom StubClient subclass (only overriding the TASK=expand response;
    every other prompt still gets StubClient's normal canned behavior)
    rather than a full FallbackRouter fake, so this test exercises the
    REAL agents/planning.py::task_expander_node ->
    task_utils.py::cap_and_filter -> orchestration/graph.py::
    dispatch_tasks chain end to end, not a shortcut around it."""
    import json

    from langgraph.checkpoint.memory import MemorySaver

    from research_agent.llm.client import StubClient
    from research_agent.llm.router import FallbackRouter
    from research_agent.orchestration.graph import build_graph
    from research_agent.state import Evidence, ResearchState, Volatility

    class _HintingStubClient(StubClient):
        def complete(self, messages, temperature=0.2):
            last = messages[-1]["content"]
            if "TASK=expand" in last:
                return json.dumps({"tasks": [
                    {"query": "corpus query", "goal_id": "g1", "priority": 2},
                    {"query": "mcp query", "goal_id": "g2", "priority": 2, "tool_hint": "mcp"},
                ]})
            return super().complete(messages, temperature)

    settings = _p214_settings(mcp_enabled=True)
    router = FallbackRouter([_HintingStubClient()], quality_threshold=0.6)
    memory = _p214_memory(settings)

    corpus_calls = []
    mcp_calls = []

    def fake_corpus_tool(task):
        corpus_calls.append(task)
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="corpus",
                         content=f"corpus result for {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    def fake_mcp_tool(task):
        mcp_calls.append(task)
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="mcp",
                         content=f"mcp result for {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    g = build_graph(router, fake_corpus_tool, memory, settings, MemorySaver(),
                    mcp_tool=fake_mcp_tool)
    result = g.invoke(ResearchState(raw_query="q"),
                      config={"configurable": {"thread_id": "p214-mixed-backlog"}})

    assert len(corpus_calls) == 1
    assert corpus_calls[0].query == "corpus query"
    assert corpus_calls[0].tool_hint == ""
    assert len(mcp_calls) == 1
    assert mcp_calls[0].query == "mcp query"
    assert mcp_calls[0].tool_hint == "mcp"

    telemetry = result["telemetry"]
    assert telemetry["evidence_by_source"].get("corpus", 0) >= 1
    assert telemetry["evidence_by_source"].get("mcp", 0) >= 1


def test_p2_14_with_mcp_disabled_the_llm_is_never_even_told_about_it():
    """settings.mcp_enabled=False (the default) -- confirms the PROMPT
    itself carries no tool_hint schema at all, not just that no task
    happens to use it. Proven by asserting the actual prompt text sent
    to the router never mentions "tool_hint"."""
    from research_agent.config import Settings
    from research_agent.prompts import templates
    from research_agent.state import Goal

    settings = _p214_settings(mcp_enabled=False)
    available = frozenset({"mcp"}) if settings.mcp_enabled else frozenset()
    assert available == frozenset()

    msgs = templates.expand_tasks([Goal(goal_id="g1", description="x")], 5,
                                  available_tool_hints=available)
    assert "tool_hint" not in msgs[1]["content"]