"""
tests/unit/test_tools_retrieval_chain.py — the retrieval escalation ladder
(D-38), fully offline.
"""

import pytest

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
    from research_agent.tools.retrieval_chain import _distinctive_terms
    shared = _distinctive_terms("Comparison analysis of Indian army") & \
        _distinctive_terms("Comparison analysis of Redis throughput")
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


