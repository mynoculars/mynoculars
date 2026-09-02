"""
tests/unit/test_tools_retrieval_chain.py — the retrieval escalation ladder
(D-38), fully offline.
"""


from research_agent.state import Evidence, SearchTask, Volatility
from research_agent.tools.model_knowledge import make_model_knowledge_tool
from research_agent.tools.retrieval_chain import _reformulate, make_retrieval_chain

FLOOR = 0.5


def _task(query="Compare Indian and Chinese army on battlefield", goal="g1"):
    return SearchTask(key=f"{goal}::{query.lower()}", query=query,
                      goal_id=goal, priority=1, depth=0)


def _ev(source, score, content="Indian and Chinese army battlefield strength"):
    def _tool(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source=source,
                         content=content, score=score,
                         volatility=Volatility.SEMI_STABLE)]
    return _tool


def _empty(task):
    return []


def test_corpus_wins_when_it_actually_answers():
    """A real document must always beat recollection when one exists."""
    def _model(task):
        raise AssertionError("model tier must not be reached")
    chain = make_retrieval_chain(_ev("corpus", 0.99), FLOOR, model=_model,
                                 reformulate=False)
    out = chain(_task())
    assert [e.source for e in out] == ["corpus"]


def test_ladder_falls_through_to_model_when_corpus_is_sub_threshold():
    """The exact live shape (runs p205.66-.81): the corpus returns real
    hits that sit AT the single-leg ceiling, so they can never cover a
    goal. Before D-38 the run ended there and reported the subject
    unanswerable."""
    chain = make_retrieval_chain(_ev("corpus", 0.5), FLOOR,
                                 model=_ev("model", 0.6), reformulate=False)
    out = chain(_task())
    assert [e.source for e in out] == ["corpus", "model"]
    assert any(e.score > FLOOR for e in out), "the goal must now be coverable"


def test_mcp_is_tried_before_the_model_tier():
    def _model(task):
        raise AssertionError("model tier must not be reached when MCP answers")
    chain = make_retrieval_chain(_ev("corpus", 0.2), FLOOR,
                                 mcp=_ev("mcp", 0.9), model=_model,
                                 reformulate=False)
    assert [e.source for e in chain(_task())] == ["corpus", "mcp"]


def test_a_dead_tier_does_not_abort_a_task_a_later_tier_can_answer():
    def _boom(task):
        raise ConnectionError("backend down")
    chain = make_retrieval_chain(_boom, FLOOR, model=_ev("model", 0.6),
                                 reformulate=False)
    assert [e.source for e in chain(_task())] == ["model"]


def test_reformulated_retry_is_retagged_onto_the_original_task_key():
    """Otherwise dedup (D-2), coverage and the D-16 failure ledger would
    all see a task the producer never emitted."""
    seen = []

    def _corpus(task):
        seen.append(task.query)
        if len(seen) == 1:
            return []
        return [Evidence(task_key=task.key, goal_id=task.goal_id,
                         source="corpus", content="Indian Chinese army battlefield",
                         score=0.9, volatility=Volatility.SEMI_STABLE)]

    original = _task()
    chain = make_retrieval_chain(_corpus, FLOOR, reformulate=True)
    out = chain(original)
    assert len(seen) == 2 and seen[0] != seen[1]
    assert out[0].task_key == original.key


def test_reformulate_strips_comparison_scaffolding():
    short = _reformulate(
        "Comparative analysis of Indian and Chinese military technology: "
        "weapons systems, cyber capabilities, and electronic warfare")
    assert "Comparative" not in short and "analysis" not in short
    assert "Indian" in short and "Chinese" in short
    assert len(short.split()) <= 6


def test_chain_preserves_the_retrieval_counter_seam():
    """agents/gathering.py drains this duck-typed attribute off the tool."""
    corpus = _ev("corpus", 0.9)
    corpus.drain_retrieval_counts = lambda: {"retrieval_dense_calls": 1}
    chain = make_retrieval_chain(corpus, FLOOR)
    assert chain.drain_retrieval_counts() == {"retrieval_dense_calls": 1}


# ---------------------------------------------------------------------------
# The model tier itself
# ---------------------------------------------------------------------------


class _Router:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, messages):
        return self.payload

    def set_node(self, name):  # pragma: no cover - must never be called
        raise AssertionError("set_node races across parallel workers")


def test_model_tier_drops_claims_the_model_itself_flags_as_shaky():
    """A low-confidence recollection that still marks a goal `covered` is
    worse than no item at all."""
    tool = make_model_knowledge_tool(_Router({"claims": [
        {"text": "solid fact", "confidence": 0.9},
        {"text": "half-remembered", "confidence": 0.2},
    ]}))
    out = tool(_task())
    assert [e.content for e in out] == ["solid fact"]


def test_model_tier_always_labels_its_source_model():
    """It must never be able to masquerade as a retrieved document."""
    tool = make_model_knowledge_tool(_Router(
        {"claims": [{"text": "a", "confidence": 1.0}]}))
    assert all(e.source == "model" for e in tool(_task()))


def test_model_tier_score_can_cover_a_goal_but_never_outranks_a_document():
    tool = make_model_knowledge_tool(_Router(
        {"claims": [{"text": "a", "confidence": 1.0}]}))
    score = tool(_task())[0].score
    assert score > FLOOR, "must be able to mark a goal covered"
    assert score < 0.99, "must never outrank a document both legs agreed on"


def test_model_tier_tolerates_malformed_claims():
    tool = make_model_knowledge_tool(_Router({"claims": [
        "a bare string is accepted", {"no_text": 1}, 42, {"text": "", "confidence": 1},
    ]}))
    assert [e.content for e in tool(_task())] == ["a bare string is accepted"]


def test_model_tier_returns_nothing_when_the_model_declines():
    tool = make_model_knowledge_tool(_Router({"claims": []}))
    assert tool(_task()) == []


# ---------------------------------------------------------------------------
# The relevance gate (run p205.90-check)
# ---------------------------------------------------------------------------


def test_a_high_scoring_but_off_topic_corpus_hit_does_not_stop_the_ladder():
    """THE regression. Live: asked for "GDP per capita India US comparison
    2023" against ten Redis documents, both legs returned the same three
    irrelevant docs, cross-leg agreement pushed them past the floor, the
    chain declared "answered", the model tier was never reached, and the
    report said the evidence did not cover GDP per capita -- the exact
    give-up this ladder exists to remove. A fixed-k search over a small
    corpus ALWAYS returns k documents; a score test alone can never fail.
    """
    off_topic = _ev("corpus", 0.99,
                    "Redis is an in-memory data store supporting hashes")
    chain = make_retrieval_chain(off_topic, FLOOR,
                                 model=_ev("model", 0.6), reformulate=False)
    task = _task("GDP per capita India US comparison 2023", "g2")
    assert "model" in [e.source for e in chain(task)], (
        "an off-topic document must not be allowed to end the search")


def test_an_on_topic_corpus_hit_still_stops_the_ladder():
    """The gate must not fire on the runs that were already working
    (p205.93-check): one shared distinctive term is enough."""
    def _model(task):
        raise AssertionError("model tier must not be reached")
    on_topic = _ev("corpus", 0.99,
                   "Redis is an in-memory data store supporting hashes")
    chain = make_retrieval_chain(on_topic, FLOOR, model=_model,
                                 reformulate=False)
    task = _task("Redis distributed architecture vs Cassandra store", "g1")
    assert [e.source for e in chain(task)] == ["corpus"]


def test_scaffolding_words_do_not_count_as_topical_overlap():
    """"comparison" appearing in both a query about armies and a document
    about Redis is not evidence of anything."""
    from research_agent.retrieval.terms import distinctive_terms
    shared = distinctive_terms("Comparison analysis of Indian army") & \
        distinctive_terms("Comparison analysis of Redis throughput")
    assert shared == set()


def test_one_broad_term_does_not_satisfy_a_long_specific_query():
    """P205 regression (run p205.101-check). "Comparative analysis Redis
    Cassandra DynamoDB petabyte scale performance scalability cost
    operational complexity" has nine distinctive terms; a Redis
    session-caching document shares exactly one ("redis"). That passed the
    gate, stopped the ladder at tier 1, left model_sourced_items at 0, and
    the report said "no retrieved evidence quantifies..." for all five
    goals -- the give-up re-entering through a too-permissive gate."""
    doc = _ev("corpus", 0.99,
              "Redis is an in-memory data store supporting rich data "
              "structures: strings, hashes, lists, sets, sorted sets")
    chain = make_retrieval_chain(doc, FLOOR, model=_ev("model", 0.6),
                                 reformulate=False)
    task = _task("Comparative analysis Redis Cassandra DynamoDB petabyte "
                 "scale performance scalability cost operational complexity",
                 "g1")
    assert "model" in [e.source for e in chain(task)]


def test_a_short_query_still_matches_on_a_single_term():
    """The bar scales with how specific the query was; a two-word query
    must not be held to a three-term overlap."""
    def _model(task):
        raise AssertionError("model tier must not be reached")
    doc = _ev("corpus", 0.99, "Redis memory overhead per key is higher")
    chain = make_retrieval_chain(doc, FLOOR, model=_model, reformulate=False)
    assert [e.source for e in chain(_task("Redis overhead", "g1"))] == ["corpus"]


def test_need_never_exceeds_the_querys_own_term_count():
    """A query with only ONE distinctive term (e.g. "key differences" --
    "key" is filtered as too short, leaving just "differences") cannot be
    held to a 2-term overlap bar -- that would make even a document
    constructed to be maximally relevant (echoing the query verbatim)
    fail the gate, which is worse than the bug being fixed. Regression
    target: test_a_real_document_still_beats_recollection (a separate,
    pre-existing integration test) broke on exactly this shape the first
    time the floor was raised from 1 to 2, before this cap was added."""
    def _model(task):
        raise AssertionError("model tier must not be reached")
    doc = _ev("corpus", 0.99, "A relevant document about key differences")
    chain = make_retrieval_chain(doc, FLOOR, model=_model, reformulate=False)
    assert [e.source for e in chain(_task("key differences", "g1"))] == ["corpus"]


def test_one_accidental_shared_word_no_longer_satisfies_a_short_query():
    """Regression target: run p205.141-check. The reformulated retry
    "Indian Army size composition Chinese PLA" (4 distinctive terms,
    need=1 under the pre-fix floor) matched a completely unrelated
    Memcached slab-allocator document on the single accidental word
    "size" -- an off-topic hit that got merged into evidence under a
    real, correctly-tagged goal_id, and later primed gap_generator's next
    cycle toward more off-topic queries. A floor of 1 reopened this hole
    for any query <=7 distinctive terms, which is EVERY reformulated
    retry by construction (_reformulate caps output at 6 words) -- this
    is not a synthetic edge case, it is the ladder's own second tier's
    normal operating range. With the floor raised to 2, a single
    accidental word is never sufficient on its own, regardless of query
    length."""
    doc = _ev("corpus", 0.99,
              "Memcached generally has lower per-key memory overhead for "
              "small opaque values due to its slab allocator, whose chunk "
              "size classes are fixed at startup.")
    chain = make_retrieval_chain(doc, FLOOR, model=_ev("model", 0.6),
                                 reformulate=False)
    task = _task("Indian Army size composition Chinese PLA", "g1")
    sources = [e.source for e in chain(task)]
    # The insufficient corpus hit is still COLLECTED (kept as context for
    # the compiler, per this module's own docstring) -- what matters is
    # that it did NOT stop the ladder, so the model tier was also reached.
    assert "model" in sources


def test_two_genuinely_shared_terms_still_satisfies_a_short_query():
    """The floor-raise (1 -> 2) must not make the gate impossible to
    clear for a short query that is genuinely on topic -- two real
    shared content words is still a strong enough signal."""
    def _model(task):
        raise AssertionError("model tier must not be reached")
    doc = _ev("corpus", 0.99,
              "The Indian Army's size and composition include roughly 1.4 "
              "million active personnel across several commands.")
    chain = make_retrieval_chain(doc, FLOOR, model=_model, reformulate=False)
    task = _task("Indian Army size composition", "g1")
    assert [e.source for e in chain(task)] == ["corpus"]




def test_short_all_caps_acronyms_survive_as_distinctive_terms():
    """Code-review finding. The old length-only rule (len > 3) threw away
    exactly the tokens carrying the most topical signal in this project's
    real traffic: "GDP growth India US 2020-2023" retained neither GDP
    nor US, and an Indian-vs-PLA query dropped PLA -- its single most
    distinctive word."""
    from research_agent.retrieval.terms import distinctive_terms
    assert "pla" in distinctive_terms("Indian Army size composition Chinese PLA")
    gdp = distinctive_terms("GDP growth India US 2020-2023")
    assert {"gdp", "us"} <= gdp


def test_bare_years_are_not_distinctive_terms():
    """Code-review finding. A standalone number is a weak topical signal
    that travels in PAIRS: "2020-2023" contributed two terms, so any
    off-topic document mentioning the same two years cleared the two-term
    overlap bar on years alone -- defeating the D-55 floor-raise for the
    date-ranged queries this project issues constantly. Mixed
    alphanumerics (pm10, 155mm) are real terms and must survive."""
    from research_agent.retrieval.terms import distinctive_terms
    terms = distinctive_terms("GDP growth India US 2020-2023")
    assert not {"2020", "2023"} & terms
    assert "pm10" in distinctive_terms("US air quality PM10 levels")


def test_year_collision_alone_no_longer_satisfies_the_topical_gate():
    """The end-to-end version of the two rules above: an off-topic Redis
    document that happens to mention the same two years as a GDP query
    must not stop the ladder."""
    doc = _ev("corpus", 0.99,
              "Redis 7.0 was released in 2022. Memcached benchmarks from "
              "2020 through 2023 show slab allocator throughput.")
    chain = make_retrieval_chain(doc, FLOOR, model=_ev("model", 0.6),
                                 reformulate=False)
    task = _task("GDP growth India US 2020-2023", "g1")
    assert "model" in [e.source for e in chain(task)]


# ---------------------------------------------------------------------------
# Phase 4 (D-57): the web tier
# ---------------------------------------------------------------------------


def test_web_is_tried_after_mcp_and_before_the_model_tier():
    """The ordering that matters. In this repo the tier-3 MCP server is
    scripts/mcp_corpus_server.py -- the corpus reached a second way -- so
    web must come after it (nothing corpus-shaped is left to try) and before
    the model (a live snippet beats recollection)."""
    def _model(task):
        raise AssertionError("model tier must not be reached when web answers")

    chain = make_retrieval_chain(_ev("corpus", 0.2), FLOOR,
                                 mcp=_ev("mcp", 0.2), web=_ev("web", 0.75),
                                 model=_model, reformulate=False)
    assert [e.source for e in chain(_task())] == ["corpus", "mcp", "web"]


def test_an_answering_mcp_tier_never_reaches_web():
    def _web(task):
        raise AssertionError("web must not be reached when mcp answers")

    chain = make_retrieval_chain(_ev("corpus", 0.2), FLOOR,
                                 mcp=_ev("mcp", 0.9), web=_web,
                                 model=_empty, reformulate=False)
    assert [e.source for e in chain(_task())] == ["corpus", "mcp"]


def test_web_none_leaves_the_ladder_byte_identical_to_before_phase_4():
    """The default. WEB_SEARCH_ENABLED=false means assembly passes web=None,
    and no pre-Phase-4 run may behave differently because of this work."""
    with_web_absent = make_retrieval_chain(
        _ev("corpus", 0.2), FLOOR, mcp=_ev("mcp", 0.2),
        model=_ev("model", 0.6), reformulate=False)
    explicit_none = make_retrieval_chain(
        _ev("corpus", 0.2), FLOOR, mcp=_ev("mcp", 0.2), web=None,
        model=_ev("model", 0.6), reformulate=False)
    assert ([e.source for e in with_web_absent(_task())]
            == [e.source for e in explicit_none(_task())]
            == ["corpus", "mcp", "model"])


def test_a_dead_web_tier_does_not_abort_a_task_the_model_can_answer():
    """DDGS is an unofficial client that can be throttled at any time. A
    ratelimited search must not cost the task -- the ladder logs and
    continues."""
    def _boom(task):
        raise RuntimeError("ratelimited")

    chain = make_retrieval_chain(_empty, FLOOR, web=_boom,
                                 model=_ev("model", 0.6), reformulate=False)
    assert [e.source for e in chain(_task())] == ["model"]


def test_sub_threshold_web_evidence_is_kept_but_does_not_stop_the_ladder():
    """Same rule every tier obeys: evidence that cannot cover a goal cannot
    end the search for one. It is still handed to the compiler as context."""
    chain = make_retrieval_chain(_empty, FLOOR, web=_ev("web", 0.4),
                                 model=_ev("model", 0.6), reformulate=False)
    assert [e.source for e in chain(_task())] == ["web", "model"]


def test_an_off_topic_web_hit_does_not_stop_the_ladder():
    """_sufficient's topical gate applies to web exactly as to every other
    tier. A search engine judges relevance against the query STRING; this
    gate judges it against the query's distinctive terms, which is what
    catches a plausible-looking result set about the wrong subject."""
    chain = make_retrieval_chain(
        _empty, FLOOR,
        web=_ev("web", 0.75, content="Recipe for sourdough bread starter"),
        model=_ev("model", 0.6), reformulate=False)
    assert [e.source for e in chain(_task())] == ["web", "model"]


def test_web_evidence_reaching_the_compiler_carries_its_url(monkeypatch):
    """url/domain must survive the ladder untouched -- the deterministic
    Sources pass has nothing else to build from."""
    def _web(task):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="web",
                         content="Indian and Chinese army battlefield strength",
                         score=0.75, volatility=Volatility.VOLATILE,
                         url="https://example.org/report", domain="example.org")]

    chain = make_retrieval_chain(_empty, FLOOR, web=_web, reformulate=False)
    out = chain(_task())
    assert out[0].url == "https://example.org/report"
    assert out[0].domain == "example.org"


# ---------------------------------------------------------------------------
# D-87: per-tier counters, drained through the existing worker seam
# ---------------------------------------------------------------------------


def _tier_task(query="Redis eviction policies and memory limits"):
    return SearchTask(key="g1::t", query=query, goal_id="g1")


def _tier_hit(content, score=0.9):
    return [Evidence(task_key="g1::t", goal_id="g1", source="corpus",
                     content=content, score=score)]


def test_the_answering_tier_is_counted_and_drained():
    chain = make_retrieval_chain(
        lambda t: _tier_hit("Redis eviction policies include allkeys-lru and "
                       "volatile-ttl memory limits."),
        min_evidence_score=0.5)

    chain(_tier_task())
    counts = chain.drain_retrieval_counts()

    assert counts["chain_answered_corpus"] == 1.0


def test_a_tier_that_raises_is_counted_as_a_failure_not_a_crash():
    def _boom(task):
        raise RuntimeError("backend down")

    chain = make_retrieval_chain(
        _boom, min_evidence_score=0.5,
        model=lambda t: _tier_hit("Redis eviction policies and memory limits "
                             "are well documented.", score=0.6))

    chain(_tier_task())
    counts = chain.drain_retrieval_counts()

    # TWO, not one, and that is the real behaviour worth pinning:
    # tier 1 (corpus) and tier 2 (the reformulated corpus retry)
    # are the SAME tool, so one dead backend fails both. A run
    # showing chain_tier_failed at twice the task count is a dead
    # corpus, not two unrelated problems -- which is exactly the
    # kind of thing this counter exists to make legible.
    assert counts["chain_tier_failed"] == 2.0
    assert counts["chain_answered_model"] == 1.0


def test_an_exhausted_ladder_is_counted():
    """No tier answered and no model tier wired -- the ladder ran out.
    Previously visible only as a WARNING log line."""
    chain = make_retrieval_chain(lambda t: [], min_evidence_score=0.5,
                                 reformulate=False)

    chain(_tier_task())
    counts = chain.drain_retrieval_counts()

    assert counts["chain_exhausted"] == 1.0


def test_counts_drain_so_each_task_reports_only_its_own():
    chain = make_retrieval_chain(
        lambda t: _tier_hit("Redis eviction policies include allkeys-lru and "
                       "volatile-ttl memory limits."),
        min_evidence_score=0.5)

    chain(_tier_task())
    first = chain.drain_retrieval_counts()
    second = chain.drain_retrieval_counts()

    assert first["chain_answered_corpus"] == 1.0
    assert second == {}


def test_the_corpus_tools_own_counters_still_come_through():
    """The seam previously forwarded the corpus tier's retrieval-leg
    counters verbatim. D-87 merges tier counts ON TOP of those -- it must
    not replace them, or P2-07's dense/keyword call counts vanish."""
    def _corpus(task):
        return _tier_hit("Redis eviction policies include allkeys-lru and "
                    "volatile-ttl memory limits.")

    _corpus.drain_retrieval_counts = lambda: {"retrieval_dense_calls": 1.0}

    chain = make_retrieval_chain(_corpus, min_evidence_score=0.5)
    chain(_tier_task())
    counts = chain.drain_retrieval_counts()

    assert counts["retrieval_dense_calls"] == 1.0
    assert counts["chain_answered_corpus"] == 1.0


# ---------------------------------------------------------------------------
# D-162 -- the model tier's provider calls must reach telemetry
# ---------------------------------------------------------------------------


def test_the_model_tier_s_router_counters_reach_the_worker_s_drain():
    """Every LLM call in this system happens in a NODE, and every node
    does `**router.drain_counters()` right after it. The model tier calls
    the router from inside a fanned-out search_worker instead, where no
    such line exists -- so its provider calls and token counts stayed on
    that worker thread's threading.local and were never merged into
    state.counters. D-86's llm_total_tokens under-reported, and D-132's
    RUN_TOKEN_BUDGET could not see model-tier spend at all."""
    from research_agent.llm.router import FallbackRouter
    from research_agent.state import SearchTask
    from research_agent.tools.model_knowledge import make_model_knowledge_tool
    from research_agent.tools.retrieval_chain import make_retrieval_chain

    class _Client:
        name = "stub"

        def set_trace_node(self, node):
            pass

        def complete(self, messages, temperature=0.2):
            return '{"claims": [{"text": "A stable fact.", "confidence": 0.9}]}'

        def complete_json(self, messages, temperature=0.0):
            import json
            return json.loads(self.complete(messages, temperature))

    router = FallbackRouter([_Client()], quality_threshold=0.6)
    chain = make_retrieval_chain(
        lambda task: [], 0.5,
        model=make_model_knowledge_tool(router), reformulate=False)

    evidence = chain(SearchTask(key="k1", query="anything", goal_id="g1"))
    counts = chain.drain_retrieval_counts()

    assert [e.source for e in evidence] == ["model"]
    assert counts.get("llm_provider_calls") == 1.0, counts
    # Draining is drain-not-peek, exactly like every node's own call.
    assert "llm_provider_calls" not in chain.drain_retrieval_counts()


def test_a_router_without_drain_counters_is_still_a_valid_model_tier():
    """Duck-typed on purpose: several tests pass a hand-written object
    with only complete_json, and a telemetry detail must not break them."""
    from research_agent.state import SearchTask
    from research_agent.tools.model_knowledge import make_model_knowledge_tool
    from research_agent.tools.retrieval_chain import make_retrieval_chain

    class _Bare:
        def set_node(self, name):
            pass

        def complete_json(self, messages):
            return {"claims": [{"text": "A stable fact.", "confidence": 0.9}]}

    chain = make_retrieval_chain(
        lambda task: [], 0.5,
        model=make_model_knowledge_tool(_Bare()), reformulate=False)

    assert [e.source for e in chain(SearchTask(key="k", query="q", goal_id="g1"))] == ["model"]
    assert chain.drain_retrieval_counts() is not None
