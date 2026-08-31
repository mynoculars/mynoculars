"""
tests/unit/test_memory_semantic.py — memory/semantic_memory.py.

Covers: decay_factor() itself (D-24's volatility-aware half-life math),
its numeric parity with Qdrant's server-side exp_decay formula semantics
(P2-10 — proves the FORMULA is correct, not that a live server executes
it this way; see the parity section below for that caveat stated in
full), SemanticMemory.retrieve()'s server-side-decay on/off gate (P2-10),
and SemanticMemory.store_run()'s content-hash dedup (P2-15). Does NOT
cover QdrantStore itself — see test_storage_qdrant_store.py — these
tests fake the store's search()/search_with_decay()/upsert_texts()
methods to isolate SemanticMemory's own logic.
"""

from research_agent.memory.semantic_memory import SemanticMemory, decay_factor
from research_agent.state import Evidence, Volatility
from research_agent.storage.qdrant_store import content_id as _content_id

# ---------------------------------------------------------------------------
# decay_factor — the Python parity oracle (D-24)
# ---------------------------------------------------------------------------


def test_stable_facts_never_decay():
    assert decay_factor(365, Volatility.STABLE, 90, 14) == 1.0


def test_volatile_decays_faster_than_semi_stable():
    semi = decay_factor(30, Volatility.SEMI_STABLE, 90, 14)
    vol = decay_factor(30, Volatility.VOLATILE, 90, 14)
    assert vol < semi < 1.0


def test_half_life_is_half():
    assert decay_factor(14, Volatility.VOLATILE, 90, 14) == __import__("pytest").approx(0.5)


# ---------------------------------------------------------------------------
# Numeric parity: server-side formula math vs. the Python parity oracle
# (P2-10). IMPORTANT LIMITATION, stated plainly: this proves the FORMULA
# decay_factor() implements is mathematically identical to Qdrant's
# DOCUMENTED exp_decay semantics (output = midpoint ** (|x-target|/scale)).
# It does NOT prove a live Qdrant server actually executes it this way —
# that step needs a live run against a real server; see
# test_storage_qdrant_store.py's own module docstring for the matching
# caveat on the query-construction half of P2-10.
# ---------------------------------------------------------------------------


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
    for age_days in (0.0, 1.0, 45.0, 90.0, 180.0, 365.0):
        server_value = _qdrant_exp_decay_semantics(age_days, half_life_days=90.0)
        python_value = decay_factor(age_days, Volatility.SEMI_STABLE,
                                    half_life_semi=90.0, half_life_volatile=14.0)
        assert abs(server_value - python_value) < 1e-9, (
            f"mismatch at age_days={age_days}: server={server_value} python={python_value}")


def test_formula_math_matches_python_decay_factor_for_volatile():
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


# ---------------------------------------------------------------------------
# SemanticMemory.retrieve — server-side decay gate on/off (P2-10)
# ---------------------------------------------------------------------------


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
# SemanticMemory.store_run — content-identity dedup (P2-15)
#
# Scope note, stated plainly: identity here is EXACT content-hash only.
# There is no separate Evidence.superseded_by field — "supersession" is
# achieved simply by upsert-by-id overwriting the same point in place
# (see store_run's own docstring for the full reasoning). Near-duplicate/
# paraphrased-fact detection remains explicitly out of scope.
# ---------------------------------------------------------------------------


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
    return [Evidence(task_key="t1", goal_id=goal_id, source="corpus", content=content,
                     score=0.9, volatility=Volatility.SEMI_STABLE)]


def test_store_run_passes_a_content_based_id_fn():
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
    store = _FakeUpsertStore()
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    mem.store_run("first query", _one_evidence("same fact", goal_id="g1"))
    mem.store_run("second query, different run", _one_evidence("same fact", goal_id="g7"))

    (items1, id_fn1), (items2, id_fn2) = store.calls
    assert id_fn1(items1[0]) == id_fn2(items2[0])


def test_store_run_id_fn_differs_for_different_content():
    store = _FakeUpsertStore()
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    mem.store_run("q", _one_evidence("fact A"))
    mem.store_run("q", _one_evidence("fact B"))

    (items1, id_fn1), (items2, id_fn2) = store.calls
    assert id_fn1(items1[0]) != id_fn2(items2[0])


def test_store_run_logs_all_items_as_new_when_nothing_preexisted(caplog):
    import logging

    store = _FakeUpsertStore()  # empty preexisting_ids -> everything is new
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    with caplog.at_level(logging.INFO):
        mem.store_run("q", _one_evidence("brand new fact"))

    stored_lines = [r for r in caplog.records if r.message == "memory.stored"]
    assert len(stored_lines) == 1
    assert stored_lines[0].event_fields["new"] == 1
    assert stored_lines[0].event_fields["overwritten"] == 0
    assert stored_lines[0].event_fields["count"] == 1


def test_store_run_logs_overwritten_when_the_id_already_existed(caplog):
    import logging

    already_there = _content_id("known fact")
    store = _FakeUpsertStore(preexisting_ids={already_there})
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    with caplog.at_level(logging.INFO):
        mem.store_run("q", _one_evidence("known fact"))

    stored_lines = [r for r in caplog.records if r.message == "memory.stored"]
    assert stored_lines[0].event_fields["new"] == 0
    assert stored_lines[0].event_fields["overwritten"] == 1


def test_store_run_splits_a_mixed_batch_correctly(caplog):
    import logging

    already_there = _content_id("old fact")
    store = _FakeUpsertStore(preexisting_ids={already_there})
    mem = SemanticMemory(store, top_k=5, half_life_semi=90.0, half_life_volatile=14.0)
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="corpus", content="old fact",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
        Evidence(task_key="t2", goal_id="g1", source="corpus", content="brand new fact",
                 score=0.9, volatility=Volatility.SEMI_STABLE),
    ]
    with caplog.at_level(logging.INFO):
        mem.store_run("q", evidence)

    stored_lines = [r for r in caplog.records if r.message == "memory.stored"]
    assert stored_lines[0].event_fields["new"] == 1
    assert stored_lines[0].event_fields["overwritten"] == 1
    assert stored_lines[0].event_fields["count"] == 2


def test_content_id_still_filters_memory_sourced_evidence_before_dedup_even_applies():
    """Regression guard: the PRE-EXISTING `fresh = [... if e.source !=
    "memory"]` filter must still run before id_fn is ever considered —
    P2-15 must not change what counts as "fresh" in the first place, only
    how fresh items get their point id."""
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


def test_store_run_drops_evidence_that_never_cleared_the_coverage_bar():
    """P205 regression (run p205.71-check): recall 0.0, every goal reported
    "no reliable data", and 16 sub-threshold items were still written to
    long-term memory. They return next run as source="memory" at raw cosine
    similarity (~0.75), which outranks anything fresh retrieval produces
    under RRF -- junk that failed the bar once comes back promoted."""
    class _Store:
        def __init__(self):
            self.written = []

        def existing_point_ids(self, ids):
            return set()

        def upsert_texts(self, texts, payloads=None, *a, **k):
            self.written = list(texts)
            return len(texts)

    store = _Store()
    mem = SemanticMemory(store, 5, 90.0, 14.0)
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="corpus",
                 content="single-leg junk", score=0.5),
        Evidence(task_key="t2", goal_id="g1", source="corpus",
                 content="genuinely fused, both legs agreed", score=0.99),
    ]
    written = mem.store_run("q", evidence, min_score=0.5)
    assert written == 1, "only the item that beat the floor may be stored"
    assert "genuinely fused" in str(store.written[0])


def test_store_run_default_keeps_every_fresh_item():
    """min_score defaults to 0.0 so existing callers are unchanged."""
    class _Store:
        def existing_point_ids(self, ids):
            return set()

        def upsert_texts(self, texts, payloads=None, *a, **k):
            return len(texts)

    mem = SemanticMemory(_Store(), 5, 90.0, 14.0)
    evidence = [Evidence(task_key="t1", goal_id="g1", source="corpus",
                         content="low", score=0.1)]
    assert mem.store_run("q", evidence) == 1


def test_store_run_never_writes_model_recollection_to_memory():
    """D-42 regression (runs p205.96/.97-check). The army run stored 28
    items of which 24 were source="model"; the next, unrelated run recalled
    five and composed an entirely military goal set for "Compare India and
    US", inheriting PLA doctrine prose into an India-vs-US report.

    Memory is for what RETRIEVAL found. Recollection stored as memory comes
    back tagged source="memory" -- indistinguishable from document-backed
    evidence -- and steers a later run before any retrieval happens."""
    class _Store:
        def __init__(self):
            self.written = []

        def existing_point_ids(self, ids):
            return set()

        def upsert_texts(self, texts, payloads=None, *a, **k):
            self.written = list(texts)
            return len(texts)

    store = _Store()
    mem = SemanticMemory(store, 5, 90.0, 14.0)
    written = mem.store_run("q", [
        Evidence(task_key="t1", goal_id="g1", source="model",
                 content="PLA doctrine prioritises forward deployment", score=0.6),
        Evidence(task_key="t2", goal_id="g1", source="corpus",
                 content="a real retrieved document", score=0.9),
        Evidence(task_key="t3", goal_id="g1", source="mcp",
                 content="a real specialist-tool hit", score=0.9),
    ])
    assert written == 2, "corpus and mcp are findings; model recollection is not"
    blob = str(store.written)
    assert "PLA doctrine" not in blob


def test_store_run_never_writes_web_snippets_to_memory():
    """Phase 4 (D-57), extending D-42's exclusion by one source.

    D-42's reason applies unchanged: anything written here comes back on a
    LATER run tagged source="memory", indistinguishable from something a
    document supported, and steers that run before any retrieval happens.

    Web adds a second, independent reason. make_web_search_tool stamps every
    snippet Volatility.VOLATILE because that is what it is -- a cached copy
    of today's search result is simply a wrong answer next month, with
    nothing in the text marking it stale. A live lookup should be repeated,
    not remembered.
    """
    class _Store:
        def __init__(self):
            self.written = []

        def existing_point_ids(self, ids):
            return set()

        def upsert_texts(self, texts, payloads=None, *a, **k):
            self.written = list(texts)
            return len(texts)

    store = _Store()
    mem = SemanticMemory(store, 5, 90.0, 14.0)
    written = mem.store_run("q", [
        Evidence(task_key="t1", goal_id="g1", source="web",
                 content="Today's headline says the rate is 6.5 percent",
                 score=0.75, url="https://example.org/rates",
                 domain="example.org"),
        Evidence(task_key="t2", goal_id="g1", source="corpus",
                 content="a real retrieved document", score=0.9),
    ])
    assert written == 1, "corpus is a finding; a web snippet is a live lookup"
    assert "Today's headline" not in str(store.written)


# ---------------------------------------------------------------------------
# D-142 — the recall relevance floor
#
# Until this landed, memory had NO floor on the way out: retrieve() took
# scored[:top_k] unconditionally, so every run inherited five remembered
# items however unrelated they were. memory_write_min_score gates what goes
# IN; nothing gated what came back OUT. Live shape (p205.280-check): five
# Redis-vs-Memcached items recalled at similarity 0.45-0.47 into a
# China-vs-India military query, leading the compile prompt, while the
# CORPUS floor at 0.55 dropped 72 of 72 dense candidates.
# ---------------------------------------------------------------------------


def test_above_floor_drops_only_what_is_below_it():
    from research_agent.memory.semantic_memory import _above_floor

    kept, dropped = _above_floor(
        [{"similarity": 0.90}, {"similarity": 0.47}, {"similarity": 0.60}], 0.60)

    assert [h["similarity"] for h in kept] == [0.90, 0.60]
    assert dropped == 1


def test_above_floor_is_inclusive_at_the_boundary():
    """>= not >. The corpus floor (retrieval/hybrid.py) is also inclusive;
    a second, subtly different boundary rule for the same kind of number is
    exactly the drift D-99 was written about."""
    from research_agent.memory.semantic_memory import _above_floor

    kept, dropped = _above_floor([{"similarity": 0.60}], 0.60)
    assert len(kept) == 1 and dropped == 0


def test_a_zero_floor_is_the_documented_disable_switch():
    """MEMORY_MIN_SIMILARITY=0.0 must reproduce pre-D-142 behaviour exactly,
    the same escape hatch MIN_SIMILARITY=0.0 already provides for the corpus
    leg."""
    from research_agent.memory.semantic_memory import _above_floor

    hits = [{"similarity": 0.01}, {"similarity": 0.0}]
    kept, dropped = _above_floor(hits, 0.0)
    assert kept == hits and dropped == 0


def test_retrieve_applies_the_floor_before_decay_not_after():
    """The floor asks about RELEVANCE, decay asks about FRESHNESS, and
    testing their product would conflate them: a stale-but-relevant fact
    should be de-ranked by decay, not deleted by a relevance gate.

    The item below is relevant (0.80) but old enough that similarity x decay
    lands under the floor. It must survive.
    """
    store = _FakeDecayStore(search_hits=[
        {"content": "relevant but old", "similarity": 0.80, "age_days": 180.0,
         "goal_id": "g1", "volatility": "volatile"},
    ])
    memory = SemanticMemory(store, top_k=5, half_life_semi=90.0,
                            half_life_volatile=14.0, min_similarity=0.60)

    out = memory.retrieve("q")

    assert len(out) == 1, "a relevant item was deleted by decay, not de-ranked"
    assert out[0].score < 0.60, "decay should still have pushed the score down"


def test_retrieve_drops_the_off_topic_recall_that_motivated_this():
    """The p205.280-check shape, reconstructed: a fresh but unrelated
    memory item at 0.46 against a 0.60 floor."""
    store = _FakeDecayStore(search_hits=[
        {"content": "Memcached scales vertically across many threads",
         "similarity": 0.46, "age_days": 1.0, "goal_id": "g1",
         "volatility": "semi_stable"},
        {"content": "PLA active personnel 2023", "similarity": 0.78,
         "age_days": 1.0, "goal_id": "g1", "volatility": "semi_stable"},
    ])
    memory = SemanticMemory(store, top_k=5, half_life_semi=90.0,
                            half_life_volatile=14.0, min_similarity=0.60)

    out = memory.retrieve("Compare the Armies of China and India")

    assert [e.content for e in out] == ["PLA active personnel 2023"]


def test_the_floor_defaults_off_on_the_constructor():
    """0.0 on __init__, 0.60 on Settings. A default of 0.60 in the
    constructor would silently change what every existing test recalls;
    assembly.py is what wires the real default in."""
    store = _FakeDecayStore(search_hits=[
        {"content": "weak", "similarity": 0.10, "age_days": 0.0,
         "goal_id": "g1", "volatility": "stable"},
    ])
    memory = SemanticMemory(store, top_k=5, half_life_semi=90.0,
                            half_life_volatile=14.0)

    assert memory.min_similarity == 0.0
    assert len(memory.retrieve("q")) == 1


def test_the_server_side_path_also_applies_the_floor():
    """Stricter there, and the docstring says so: Qdrant's "similarity" on
    that path is already similarity x decay, so there is no raw value to
    test. Applying the floor to the only number available is the honest
    reading -- but it means the two paths are NOT identical here, which is
    why the Python path stays the default."""
    store = _FakeDecayStore(decay_hits=[
        {"content": "low", "similarity": 0.30, "goal_id": "g1",
         "volatility": "semi_stable"},
        {"content": "high", "similarity": 0.82, "goal_id": "g1",
         "volatility": "semi_stable"},
    ])
    memory = SemanticMemory(store, top_k=5, half_life_semi=90.0,
                            half_life_volatile=14.0, server_side_decay=True,
                            min_similarity=0.60)

    out = memory.retrieve("q")

    assert store.search_with_decay_calls == 1
    assert [e.content for e in out] == ["high"]

