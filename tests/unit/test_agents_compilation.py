"""
tests/unit/test_agents_compilation.py — agents/compilation.py::compiler_node.

WHY THIS FILE EXISTS: compile_report's prompt explicitly asks for Markdown
prose, but a live trace showed a fallback provider (Mistral, reached after
a quality-reject bounced the call off the primary provider) answering with
a ```json ...``` block anyway. Nothing on that call path stripped the
fence — the identical defensive pattern already existed for the JSON-mode
call sites (llm/client.py::_extract_json) but was never carried to this
one. These tests cover the fix (llm/client.py::strip_code_fence) and its
wiring into compiler_node.
"""

from research_agent.agents.compilation import build_compiler_node
from research_agent.config import Settings
from research_agent.llm.client import strip_code_fence
from research_agent.state import Evidence, Goal, ResearchState

# D-85 added `settings` to build_compiler_node. Deliberately LIVE mode
# rather than stub: the D-85 provenance pass is gated off in stub mode, so
# a stub Settings would skip it entirely and these tests would stop saying
# anything about it. In live mode the pass genuinely runs against each
# state below -- all of which have no goals and no evidence, so it must
# no-op -- which makes every exact-equality assertion in this file double
# as proof that the no-op path leaves the report byte-identical.
_SETTINGS = Settings(_env_file=None, llm_mode="live")


class _FakeRouter:
    """Just enough of FallbackRouter's interface for compiler_node."""

    def __init__(self, reply: str):
        self._reply = reply
        self.node = None

    def set_node(self, node):
        self.node = node

    def complete(self, messages, temperature=0.2):
        return self._reply

    def drain_counters(self):
        return {}


# ---------------------------------------------------------------------------
# strip_code_fence — pure function
# ---------------------------------------------------------------------------

def test_strip_code_fence_removes_json_tagged_fence():
    text = '```json\n{"a": 1}\n```'
    assert strip_code_fence(text) == '{"a": 1}'


def test_strip_code_fence_removes_bare_fence():
    text = "```\n# Report\nSome prose.\n```"
    assert strip_code_fence(text) == "# Report\nSome prose."


def test_strip_code_fence_removes_any_language_tag():
    text = "```markdown\n# Report\n```"
    assert strip_code_fence(text) == "# Report"


def test_strip_code_fence_is_a_noop_on_unfenced_text():
    text = "# Report\n\nJust plain markdown, no fence at all."
    assert strip_code_fence(text) == text


def test_strip_code_fence_does_not_touch_a_fence_in_the_middle():
    """Only a fence WRAPPING the whole reply should be removed — a code
    block the model legitimately included as part of the report body must
    survive untouched."""
    text = "# Report\n\nExample query:\n```sql\nSELECT 1;\n```\n\nMore prose."
    assert strip_code_fence(text) == text


def test_strip_code_fence_handles_punctuated_language_tags():
    """A regression: the first implementation used \\w+ for the language
    tag, which stops at the first non-word character. A tag like c++ or
    objective-c left a stray fragment (e.g. "-c\\n...") in the output
    instead of the actual content."""
    assert strip_code_fence("```c++\nint x = 1;\n```") == "int x = 1;"
    assert strip_code_fence("```objective-c\n[foo bar];\n```") == "[foo bar];"


def test_strip_code_fence_leaves_a_single_line_pseudo_fence_untouched():
    """```hello``` has no newline separating a "tag" from content, so per
    CommonMark it is not a real fenced code block -- just three literal
    backticks. The first implementation's greedy \\w+ swallowed "hello" as
    a tag and returned an empty string; the correct behaviour is to leave
    it as unfenced content."""
    text = "```hello```"
    assert strip_code_fence(text) == text


def test_strip_code_fence_handles_a_closing_fence_glued_to_content():
    """No newline before the closing ``` (a fully single-line model
    reply) must still be recognised and stripped."""
    assert strip_code_fence('```json\n{"a":1}```') == '{"a":1}'


def test_strip_code_fence_tolerates_trailing_whitespace_on_delimiter_lines():
    assert strip_code_fence("```json   \n{}\n```   ") == "{}"


def test_strip_code_fence_only_strips_the_outer_wrap_around_two_blocks():
    """Two fenced blocks back to back: only the outermost open/close pair
    is a genuine wrap around the whole reply. The inner fence markers are
    content, not delimiters, and must survive."""
    text = "```json\n{}\n```\n```python\nx=1\n```"
    assert strip_code_fence(text) == "{}\n```\n```python\nx=1"


def test_strip_code_fence_is_a_noop_with_only_an_opening_marker():
    """A fence with no matching close is not a real wrap -- leave it
    alone rather than guessing where content should end."""
    text = "```json\n{\"a\": 1}"
    assert strip_code_fence(text) == text


def test_strip_code_fence_on_empty_string():
    assert strip_code_fence("") == ""


# ---------------------------------------------------------------------------
# compiler_node — wiring
# ---------------------------------------------------------------------------

def test_compiler_node_strips_a_fence_the_model_added_despite_instructions():
    """Regression for the exact live-trace symptom: a fallback provider
    ignored the "write Markdown, not JSON, no fence" instruction and wrapped
    its answer in ```json anyway. The fence must not leak into
    final_report."""
    router = _FakeRouter('```json\n{"title": "x", "findings": {}}\n```')
    node = build_compiler_node(router, _SETTINGS)
    state = ResearchState(raw_query="q")

    result = node(state)

    assert result["final_report"] == '{"title": "x", "findings": {}}'
    assert "```" not in result["final_report"]


def test_compiler_node_normalises_the_citation_form_before_anything_reads_it():
    """P205 regression (run p205.253-check), at the pipeline level rather
    than the function level. Order is the whole point: D-99 has to run
    before clean_citations, the Sources block, the D-66 gate and the D-91
    audit, because every one of those reads citations through
    `cited_goal_ids` and sees `(g1)` as no citation at all. If this ever
    moves later in the chain, this test is what says so."""
    router = _FakeRouter("## 1. Personnel (g1)\n\nChina fields more troops.")
    node = build_compiler_node(router, _SETTINGS)
    state = ResearchState(
        raw_query="q",
        goals=[Goal(goal_id="g1", description="personnel")],
        evidence=[Evidence(task_key="t", goal_id="g1", source="corpus",
                           content="China fields many troops.", score=0.9)])

    result = node(state)

    assert "[g1]" in result["final_report"]
    assert "(g1)" not in result["final_report"]
    assert result["last_compile_guardrails"][
        "citations_form_normalised"] == 1.0


def test_compiler_node_leaves_clean_markdown_unchanged():
    router = _FakeRouter("# Report\n\nRedis is fast. [g1 | corpus | score=0.90]")
    node = build_compiler_node(router, _SETTINGS)
    state = ResearchState(raw_query="q")

    result = node(state)

    assert result["final_report"] == "# Report\n\nRedis is fast. [g1 | corpus | score=0.90]"


def test_compiler_node_abort_path_is_unaffected_by_fence_stripping():
    """The two non-LLM report shapes (abort, planning_error) never call
    strip_code_fence at all — confirm the fix didn't touch that branch."""
    router = _FakeRouter("unused")
    node = build_compiler_node(router, _SETTINGS)
    state = ResearchState(raw_query="q", abort_reason="reviewer aborted")

    result = node(state)

    assert "aborted by human reviewer" in result["final_report"]
    assert router.node is None  # router.set_node was never called on this path


# ---------------------------------------------------------------------------
# D-43: deterministic citation repair
# ---------------------------------------------------------------------------

from research_agent.guardrails.citations import (  # noqa: E402
    clean_citations, normalise_citation_form, residual_paste_sites)


def _g(goal_id):
    return Goal(goal_id=goal_id, description=f"desc {goal_id}")


def _e(goal_id, content, source="corpus", score=0.9):
    return Evidence(task_key="t", goal_id=goal_id, source=source,
                    content=content, score=score)


def test_pasted_evidence_text_is_stripped_from_the_prose():
    """Live (run p205.95-check): the compiler ran the source sentence
    straight into the claim -- '...whole session blobRedis is an in-memory
    data store...' -- unreadable and unattributable."""
    body = ("Redis is an in-memory data store supporting rich data "
            "structures: strings, hashes, lists, sets.")
    report = f"Sessions map naturally to hashes{body} and update partially."
    cleaned, counters = clean_citations(report, [_g("g1")], [_e("g1", body)])
    assert body not in cleaned
    assert counters["citations_pasted_evidence_removed"] == 1.0


def test_citations_to_goals_with_no_evidence_are_dropped():
    """A [gN] marker asserts goal N's retrieved evidence supports the
    sentence. If goal N retrieved nothing, that is false on its face."""
    report = "Cassandra scales linearly [g2]. Redis is in-memory [g1]."
    cleaned, counters = clean_citations(
        report, [_g("g1"), _g("g2")], [_e("g1", "x" * 50)])
    assert "[g2]" not in cleaned
    assert "[g1]" in cleaned, "an evidenced goal keeps its citation"
    assert counters["citations_to_unevidenced_goals"] == 1.0


def test_short_evidence_is_never_stripped_as_a_pasted_citation():
    """Short fragments collide with ordinary prose; a pasted citation is
    always a whole retrieved sentence."""
    report = "Redis is fast and reliable."
    cleaned, counters = clean_citations(report, [_g("g1")], [_e("g1", "fast")])
    assert cleaned == report
    assert counters == {}


def test_a_clean_report_is_returned_untouched():
    report = "Redis is in-memory [g1]. It shards via clustering [g1]."
    cleaned, counters = clean_citations(
        report, [_g("g1")], [_e("g1", "y" * 60)])
    assert cleaned == report
    assert counters == {}


def test_corpus_recall_ignores_off_topic_documents():
    """P205 regression (runs p205.99/.100-check): corpus_recall reported 1.0
    against a ten-document Redis corpus for queries about armies and about
    India vs the US. Off-topic hits still cleared score > 0.5 via cross-leg
    agreement, so the metric added specifically as the honesty counterpart
    to recall was fooled exactly the way the ladder's sufficiency test used
    to be. It must apply the same topical gate (D-39)."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="Compare Indian and Chinese army on battlefield",
        goals=[Goal(goal_id="g1",
                    description="Compare Indian and Chinese army strength")],
        evidence=[
            Evidence(task_key="t1", goal_id="g1", source="corpus", score=0.99,
                     content="Redis is an in-memory data store using hashes"),
            Evidence(task_key="t2", goal_id="g1", source="model", score=0.6,
                     content="The Indian Army fields 1.2 million personnel"),
        ],
        recall_score=1.0, iteration_depth=1)
    telemetry = node(state)["telemetry"]
    assert telemetry["recall"] == 1.0, "the goal IS covered -- by the model tier"
    assert telemetry["corpus_recall"] == 0.0, (
        "no DOCUMENT covered it; a Redis page is not evidence about armies")
    assert telemetry["model_sourced_items"] == 1


def test_telemetry_reports_grounded_score_and_hedge_specific_count():
    """Guardrail G2/G3 surface in telemetry: grounded_score is whatever
    progress_checker_node last wrote to state (this node just reads it
    back, D-12), and hedge_specific_items is counted fresh from
    state.evidence, same pattern evidence_by_source already uses."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="q", goals=[Goal(goal_id="g1", description="d")],
        evidence=[
            Evidence(task_key="t1", goal_id="g1", source="model", score=0.6,
                     content="fabricated", hedge_specific=True),
            Evidence(task_key="t2", goal_id="g1", source="model", score=0.6,
                     content="hedged already", hedge_specific=False),
        ],
        grounded_score=0.25)
    telemetry = node(state)["telemetry"]
    assert telemetry["grounded_score"] == 0.25
    assert telemetry["hedge_specific_items"] == 1


def test_telemetry_warns_on_floor_starvation(caplog):
    """Guardrail G1: telemetry_node must WARN when the run-level dense
    candidate drop ratio clears settings.retrieval_floor_warn_ratio --
    the exact silent-starvation shape run p205.131-check hit."""
    import logging as _logging

    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(
        Settings(_env_file=None, retrieval_floor_warn_ratio=0.8))
    state = ResearchState(
        raw_query="q", goals=[],
        counters={"retrieval_dense_candidates": 10.0,
                 "retrieval_dropped_by_floor": 10.0})
    with caplog.at_level(_logging.WARNING):
        telemetry = node(state)["telemetry"]
    assert telemetry["retrieval_floor_drop_ratio"] == 1.0
    assert any("retrieval.floor_starvation" in r.message for r in caplog.records)


def test_telemetry_does_not_warn_when_floor_is_doing_its_job():
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(
        Settings(_env_file=None, retrieval_floor_warn_ratio=0.8))
    state = ResearchState(
        raw_query="q", goals=[],
        counters={"retrieval_dense_candidates": 100.0,
                 "retrieval_dropped_by_floor": 5.0})
    telemetry = node(state)["telemetry"]
    assert telemetry["retrieval_floor_drop_ratio"] == 0.05


# ---------------------------------------------------------------------------
# Guardrail G4 (P205 Phase 2): quality-judge failure-rate telemetry
# ---------------------------------------------------------------------------


def test_telemetry_warns_when_the_quality_judge_never_actually_judges(caplog):
    """Regression target: every live run this session showed
    llm_quality_calls_failed == llm_quality_calls (2/2) -- the judge
    fail-open path is correct, but a 100% failure rate had nothing
    distinguishing it from a 100% genuine-1.0 run."""
    import logging as _logging

    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(
        Settings(_env_file=None, quality_judge_warn_ratio=0.5))
    state = ResearchState(
        raw_query="q", goals=[],
        counters={"llm_quality_calls": 2.0, "llm_quality_calls_failed": 2.0})
    with caplog.at_level(_logging.WARNING):
        telemetry = node(state)["telemetry"]
    assert telemetry["llm_quality_failure_ratio"] == 1.0
    assert any("quality.judge_unreliable" in r.message for r in caplog.records)


def test_telemetry_does_not_warn_when_the_judge_is_mostly_working():
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(
        Settings(_env_file=None, quality_judge_warn_ratio=0.5))
    state = ResearchState(
        raw_query="q", goals=[],
        counters={"llm_quality_calls": 10.0, "llm_quality_calls_failed": 1.0})
    telemetry = node(state)["telemetry"]
    assert telemetry["llm_quality_failure_ratio"] == 0.1


def test_telemetry_never_warns_when_the_judge_was_never_called():
    """llm_quality_calls == 0 means "nothing to report", not "100%
    failure" -- same 0/0 guard as G1's retrieval_floor_drop_ratio."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(raw_query="q", goals=[], counters={})
    telemetry = node(state)["telemetry"]
    assert telemetry["llm_quality_failure_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Guardrail G7 (P205 Phase 3, observability only): run-level call budget
# ---------------------------------------------------------------------------


def test_telemetry_warns_when_llm_provider_calls_clears_the_budget(caplog):
    """Observational only -- this WARNs, it does not change routing or
    abort the run. See settings.run_call_budget_warn's own comment for
    why this is a WARNING and not a circuit breaker."""
    import logging as _logging

    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None, run_call_budget_warn=10))
    state = ResearchState(raw_query="q", goals=[],
                          counters={"llm_provider_calls": 10.0})
    with caplog.at_level(_logging.WARNING):
        node(state)
    assert any("run.call_budget_high" in r.message for r in caplog.records)


def test_telemetry_does_not_warn_below_the_call_budget():
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None, run_call_budget_warn=40))
    state = ResearchState(raw_query="q", goals=[],
                          counters={"llm_provider_calls": 18.0})
    telemetry = node(state)["telemetry"]
    assert telemetry["llm_provider_calls"] == 18


def test_telemetry_never_warns_on_the_call_budget_when_no_calls_were_made():
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(raw_query="q", goals=[], counters={})
    telemetry = node(state)["telemetry"]
    assert telemetry["llm_provider_calls"] == 0


def test_corpus_recall_counts_a_genuinely_on_topic_document():
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="Compare Redis and Memcached",
        goals=[Goal(goal_id="g1", description="Compare Redis throughput")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.99,
                           content="Redis throughput is limited per shard")],
        recall_score=1.0, iteration_depth=1)
    assert node(state)["telemetry"]["corpus_recall"] == 1.0


def test_properly_separated_evidence_text_is_never_stripped():
    """P205 regression (run p205.107-check, "Compare Redis vs Memcached for
    production systems" -- the first query fully covered by the corpus).
    Retrieval was perfect (corpus_recall 1.0, 36 corpus items) and this
    function deleted SIX whole sections of the finished report, shipping
    "### Scalability" and "### Security" as empty headings. On an in-corpus
    query the compiler legitimately states corpus sentences almost word for
    word -- that is what a grounded report is. The defect being guarded
    against was never quoting; it was the missing delimiter."""
    body = ("Redis supports primary-replica replication, Sentinel failover, "
            "and Redis Cluster sharding out of the box.")
    report = f"### Failover\n{body} Memcached has none."
    cleaned, counters = clean_citations(
        report, [_g("g3")], [_e("g3", body)])
    assert cleaned == report
    assert counters == {}


def test_a_parenthesised_goal_id_is_normalised_to_a_bracket():
    """P205 regression (run p205.253-check, "Compare Armies of China and
    India"). The compiler wrote its goal ids in PARENTHESES -- "## 1.
    Military Personnel Strength (g1)" through "(g4)" -- and because every
    reader of citations matches `[gN]` and nothing else, that report had
    no citations at all as far as the system was concerned. The D-66 gate
    failed it twice, two revision cycles and an E4 escalation were spent,
    the Sources block listed 0 of 78 web items, and D-91 audited nothing.
    Normalising the form recovers all four."""
    report = ("## 1. Personnel (g1)\n\nChina fields more troops.\n"
              "## 2. Budgets (g2)\n\nChina spends more.\n")

    fixed, counters = normalise_citation_form(report, [_g("g1"), _g("g2")])

    assert "[g1]" in fixed and "[g2]" in fixed
    assert "(g1)" not in fixed
    assert counters["citations_form_normalised"] == 2.0


def test_a_multi_goal_citation_becomes_one_marker_per_goal():
    """`[g1, g4]` is one of the four forms live runs produced. Splitting
    rather than dropping matters: the sentence really does cite both, and
    the Sources block attributes per goal."""
    fixed, counters = normalise_citation_form(
        "Both forces grew [g1, g4]. Spending rose (g2; g3).",
        [_g("g1"), _g("g2"), _g("g3"), _g("g4")])

    assert fixed == "Both forces grew [g1] [g4]. Spending rose [g2] [g3]."
    assert counters["citations_form_normalised"] == 2.0


def test_a_citation_carrying_its_own_metadata_is_reduced_to_the_goal_id():
    """`[g1 | corpus | score=0.50]` was live-observed too, and it is the
    shape D-91 has to strip figures out of afterwards. Reducing it here
    means that defence is a backstop rather than the only thing standing
    between a score and the figure audit."""
    fixed, _ = normalise_citation_form(
        "The PLA is large [g1 | corpus | score=0.50].", [_g("g1")])

    assert fixed == "The PLA is large [g1]."


def test_prose_that_merely_looks_like_a_citation_is_left_alone():
    """The whole safety property. A delimiter only counts when what is
    inside it is goal ids end to end -- otherwise this would rewrite the
    model's sentences."""
    for text in ("(g1 is the largest force)",
                 "see section (g1 and the rest)",
                 "the g1 goal covers personnel"):
        fixed, counters = normalise_citation_form(text, [_g("g1")])
        assert fixed == text, text
        assert counters == {}


def test_a_goal_that_does_not_exist_in_this_run_is_not_invented():
    """A block naming no known goal is left entirely alone. Rewriting it
    would manufacture a citation out of prose, and it would buy nothing:
    clean_citations drops markers for goals with no evidence anyway."""
    fixed, counters = normalise_citation_form("(g9)", [_g("g1")])

    assert fixed == "(g9)"
    assert counters == {}


def test_an_already_correct_citation_is_a_no_op():
    fixed, counters = normalise_citation_form(
        "Redis is in-memory [g1]. It shards [g2].", [_g("g1"), _g("g2")])

    assert fixed == "Redis is in-memory [g1]. It shards [g2]."
    assert counters == {}


def test_residual_paste_sites_counts_what_the_guard_left_behind():
    """`citations_pasted_evidence_removed` says what came out; it says
    nothing about what stayed. Live (run p205.253-check) 21 runs were
    removed and four glued pastes still shipped, and no number in the run
    record distinguished that from a clean report."""
    body = ("Redis is an in-memory data store supporting rich data "
            "structures: strings, hashes, lists, sets.")
    report = f"Sessions map to hashes{body}"

    assert residual_paste_sites(report, [_e("g1", body)]) == 1

    cleaned, _ = clean_citations(report, [_g("g1")], [_e("g1", body)])
    assert residual_paste_sites(cleaned, [_e("g1", body)]) == 0, \
        "a report the guard fully cleaned must report zero residue"


def test_residual_paste_sites_does_not_count_a_delimited_quotation():
    """Same detector as the remover, so the two numbers can never
    disagree about what a paste is -- including the p205.107 case the
    remover deliberately leaves alone."""
    body = ("Redis supports primary-replica replication, Sentinel failover, "
            "and Redis Cluster sharding out of the box.")

    assert residual_paste_sites(f"### Failover\n\n{body} Memcached has none.",
                                [_e("g3", body)]) == 0


def test_a_span_lifted_out_of_a_long_snippet_is_stripped():
    """P205 regression (run p205.251-check, "Compare Armies of China and
    India"). EVERY paragraph of the shipped report carried glued source
    text -- "...and strategic support forcesChina's armed forces have
    over 2.1 million active personnel." -- and this function removed
    nothing at all: citations_pasted_evidence_removed never appeared in
    the telemetry.

    The cause was the whole-body exact match (D-45). It required an
    evidence item's ENTIRE content to appear in the prose, which holds
    for short corpus chunks and essentially never for a web snippet: the
    model lifts a SPAN out of the middle of a long snippet, not the whole
    snippet. D-96 matches any verbatim run of six or more words."""
    snippet = ("Military Strength: China vs India China (ranked #2 globally) "
               "holds a stronger overall military position than India. "
               "China fields 2,535,000 active troops vs 3,068,000 for India, "
               "backed by 1,155,000 reserves and 1,616,050 paramilitary.")
    report = ("The PLA maintains roughly 2.1 million active personnel across "
              "all servicesChina fields 2,535,000 active troops vs 3,068,000 "
              "for India.")

    cleaned, counters = clean_citations(
        report, [_g("g1")], [_e("g1", snippet)])

    assert "China fields 2,535,000" not in cleaned
    assert cleaned.endswith("across all services.")
    assert counters["citations_pasted_evidence_removed"] == 1.0


def test_a_run_of_pasted_sentences_is_stripped_not_just_the_glued_one():
    """P205 regression (run p205.251-check). The model does not paste once
    and stop -- it emitted "...support forces" + <source sentence A> + " "
    + <source sentence B>. Only A is glued to the claim, so a check that
    anchored on glue alone removed A and shipped B, leaving the report
    barely more readable than before."""
    a = "China's armed forces have over 2.1 million active personnel."
    b = "China fields 2,535,000 active troops for the ground component."
    report = f"The PLA spans all services{a} {b} India differs."

    cleaned, counters = clean_citations(
        report, [_g("g1")], [_e("g1", f"{a} {b}")])

    assert a not in cleaned and b not in cleaned
    assert cleaned == "The PLA spans all services. India differs."
    assert counters["citations_pasted_evidence_removed"] == 1.0, \
        "one RUN, not one per span"


def test_a_paste_run_is_only_absorbed_after_a_confirmed_glue():
    """The continuation rule above is what makes D-96 safe rather than a
    second p205.107-check. It can only ever start from a glue site, so an
    evidence sentence quoted with proper delimiters anywhere else in the
    report is never reached."""
    body = ("Redis supports primary-replica replication, Sentinel failover, "
            "and Redis Cluster sharding out of the box.")
    report = f"### Failover\n\n{body} Memcached has none."

    cleaned, counters = clean_citations(report, [_g("g3")], [_e("g3", body)])

    assert cleaned == report
    assert counters == {}


def test_a_short_verbatim_collision_is_not_treated_as_a_paste():
    """Six words is the floor. Ordinary prose collides with evidence over
    short runs all the time, and a guardrail that deletes on those is
    worse than no guardrail."""
    report = "Sessions map to hashesRedis is fast for that."

    cleaned, counters = clean_citations(
        report, [_g("g1")], [_e("g1", "Redis is fast for that. " + "x" * 60)])

    assert cleaned == report
    assert counters == {}


def test_removing_a_paste_never_welds_two_sentences_together():
    """The removed run took its own terminal '.' with it, so the claim
    would otherwise run straight into whatever followed. The punctuation
    is carried over from the run rather than invented -- and only where
    the next character actually opens a new sentence."""
    paste = "The PLAGF is estimated to have a deployed force of 975,000 troops."
    ev = [_e("g1", paste)]

    welded, _ = clean_citations(
        f"It deploys forces in operational units{paste} India differs.",
        [_g("g1")], ev)
    assert welded == "It deploys forces in operational units. India differs."

    # The claim's own punctuation already follows: nothing is added.
    doubled, _ = clean_citations(
        f"It deploys forces in operational units{paste}.",
        [_g("g1")], ev)
    assert doubled == "It deploys forces in operational units."

    # The sentence continues in lowercase: it was never two sentences.
    flowing, _ = clean_citations(
        f"It deploys forces in operational units{paste} and holds them.",
        [_g("g1")], ev)
    assert flowing == "It deploys forces in operational units and holds them."


def test_evidence_text_glued_to_a_claim_is_still_stripped():
    """The real signature: the source sentence running into the claim with
    no boundary, e.g. "...the whole session blobRedis is an in-memory..."."""
    body = ("Redis is an in-memory data store supporting rich data "
            "structures: strings, hashes, lists, sets.")
    cleaned, counters = clean_citations(
        f"Sessions map to hashes without rewriting the whole blob{body}",
        [_g("g1")], [_e("g1", body)])
    assert body not in cleaned
    assert cleaned == "Sessions map to hashes without rewriting the whole blob"
    assert counters["citations_pasted_evidence_removed"] == 1.0


def test_corpus_recall_does_not_count_a_web_sourced_item():
    """The telemetry half of the Phase 4 grounding lock (D-57).

    corpus_recall (D-43) measures how many goals a real DOCUMENT covered.
    telemetry_node tests `source in ("corpus", "mcp")`; "web" is
    deliberately absent, so a run answered entirely from the web reports
    recall 1.0 alongside corpus_recall 0.0 -- exactly the honest split D-43
    was built to surface. Tagging web results "mcp" instead would have made
    that number silently meaningless.
    """
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="Compare Redis and Memcached",
        goals=[Goal(goal_id="g1", description="Compare Redis throughput")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="web",
                           score=0.75, url="https://example.org/redis",
                           domain="example.org",
                           content="Redis throughput is limited per shard")],
        recall_score=1.0, iteration_depth=1)
    telemetry = node(state)["telemetry"]
    assert telemetry["corpus_recall"] == 0.0, (
        'if this fails, check whether "web" was added to _doc_sources in '
        "telemetry_node")


# ---------------------------------------------------------------------------
# D-66 -- the deterministic zero-citation gate
# ---------------------------------------------------------------------------


class _JudgeShouldNotBeCalled:
    """Fake router whose complete_json raises -- proves the deterministic
    gate short-circuits BEFORE ever asking the LLM judge to score a report
    already known to fail."""

    def set_node(self, node):
        pass

    def complete_json(self, messages, temperature=0.0):
        raise AssertionError("router.complete_json must not be called "
                             "when the zero-citation gate already failed "
                             "the report")

    def drain_counters(self):
        return {}


class _CannedJudge:
    """Fake router that always returns a canned passing critique -- used
    to prove the gate is SKIPPED (the judge IS reached) in the cases
    where D-66 should not apply."""

    def __init__(self):
        self.called = False

    def set_node(self, node):
        pass

    def complete_json(self, messages, temperature=0.0):
        self.called = True
        return {"passed": True, "score": 0.9, "notes": []}

    def drain_counters(self):
        return {}


def _make_critic_state(final_report="# Report\n\nNo citations here."):
    return ResearchState(
        raw_query="Compare Redis and Memcached",
        goals=[Goal(goal_id="g1", description="Compare Redis throughput")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.9, content="Redis is single-threaded")],
        final_report=final_report)


def test_zero_citation_gate_fails_the_report_without_calling_the_judge():
    """D-66: a served report with evidence available but zero [gN]
    citations must fail deterministically -- never reach the LLM judge,
    which has nothing to fail an uncited report ON (zero claims means
    zero unsupported claims)."""
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    node = build_critic_node(_JudgeShouldNotBeCalled(),
                             Settings(_env_file=None, llm_mode="live"))
    update = node(_make_critic_state())
    assert update["critique_passed"] is False
    assert update["critique_notes"], "must explain why it failed"
    assert "cites no evidence" in update["critique_notes"][0]


def test_zero_citation_gate_is_skipped_in_stub_mode():
    """StubClient's fixed placeholder report never carries [gN] markers by
    design -- the gate must not fire in stub mode, or every offline test
    using the canned stub report would fail critique."""
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    judge = _CannedJudge()
    node = build_critic_node(judge, Settings(_env_file=None, llm_mode="stub"))
    node(_make_critic_state())
    assert judge.called, "stub mode must reach the judge, not the gate"


def test_zero_citation_gate_is_skipped_when_nothing_was_ever_retrieved():
    """A report citing nothing because NOTHING was retrieved is not this
    failure mode -- gated on state.evidence being non-empty."""
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    judge = _CannedJudge()
    node = build_critic_node(judge, Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(raw_query="q", goals=[], evidence=[],
                          final_report="# Report\n\nNothing was found.")
    node(state)
    assert judge.called, "no evidence retrieved must still reach the judge"


def test_zero_citation_gate_is_skipped_when_citations_are_present():
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    judge = _CannedJudge()
    node = build_critic_node(judge, Settings(_env_file=None, llm_mode="live"))
    state = _make_critic_state(final_report="# Report\n\nRedis is fast [g1].")
    node(state)
    assert judge.called, "a report that DID cite must still reach the judge"


def test_telemetry_warns_when_a_report_ships_with_no_citations(caplog):
    """D-66 backstop: if a zero-citation report still ships (HITL off,
    revision budget exhausted), telemetry_node must WARN -- the last line
    of sight after the critic gate."""
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(
        raw_query="q", goals=[Goal(goal_id="g1", description="d")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.9, content="x")],
        final_report="# Report\n\nNo citations here.")
    with caplog.at_level(_logging.WARNING):
        node(state)
    warned = [r for r in caplog.records
             if "report.shipped_with_no_citations" in r.message]
    assert warned, "expected a WARNING when the shipped report cites nothing"


def test_telemetry_does_not_warn_in_stub_mode(caplog):
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="stub"))
    state = ResearchState(
        raw_query="q", goals=[Goal(goal_id="g1", description="d")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.9, content="x")],
        final_report="# Report (stub mode)\n\nNo citations here.")
    with caplog.at_level(_logging.WARNING):
        node(state)
    warned = [r for r in caplog.records
             if "report.shipped_with_no_citations" in r.message]
    assert not warned, "stub mode must never trigger this warning"


def test_telemetry_does_not_warn_when_the_report_cites_something(caplog):
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(
        raw_query="q", goals=[Goal(goal_id="g1", description="d")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.9, content="x")],
        final_report="# Report\n\nSupported by evidence [g1].")
    with caplog.at_level(_logging.WARNING):
        node(state)
    warned = [r for r in caplog.records
             if "report.shipped_with_no_citations" in r.message]
    assert not warned


# ---------------------------------------------------------------------------
# D-85: the provenance notice, wired through compiler_node and read back by
# telemetry_node
# ---------------------------------------------------------------------------


def _ungrounded_state():
    """The p205.246-check shape: every goal covered, none of it grounded."""
    return ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[_g("g1"), _g("g2")],
        evidence=[_e("g1", "The PLA fields about two million troops.",
                     source="web", score=0.7),
                  _e("g2", "The Indian Army fields about 1.2 million.",
                     source="web", score=0.7)])


def test_compiler_node_annotates_an_ungrounded_report_in_live_mode():
    from research_agent.guardrails.grounding import NOTICE_MARKER

    router = _FakeRouter("# Report\n\nBoth armies are large [g1] [g2].")
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live"))

    result = node(_ungrounded_state())

    assert NOTICE_MARKER in result["final_report"]
    assert result["counters"]["grounding_notice_inserted"] == 1.0


def test_compiler_node_does_not_annotate_in_stub_mode():
    """Same gate D-66's zero-citation check and telemetry_node's
    report.shipped_with_no_citations backstop already use. StubClient's
    fixed placeholder report proves the graph executes offline; it models
    nothing about where evidence came from, so annotating it would be
    noise in the one mode that is deliberately not a real answer."""
    from research_agent.guardrails.grounding import NOTICE_MARKER

    router = _FakeRouter("# Report\n\nBoth armies are large [g1] [g2].")
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="stub"))

    result = node(_ungrounded_state())

    assert NOTICE_MARKER not in result["final_report"]
    assert "grounding_notice_inserted" not in result["counters"]


def test_telemetry_reports_whether_the_notice_actually_shipped():
    """D-59's rule: read from the SHIPPED report, never from a counter --
    compiler_node runs once per revision and its counters merge
    additively, so a counter would describe the compile attempts rather
    than the artifact the reader received."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.guardrails.grounding import annotate_ungrounded_report

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = _ungrounded_state()

    bare = node(state.model_copy(update={
        "final_report": "# Report\n\nBoth armies are large [g1] [g2]."}))
    assert bare["telemetry"]["grounding_notice_shipped"] is False

    annotated, _ = annotate_ungrounded_report(
        "# Report\n\nBoth armies are large [g1] [g2].",
        state.goals, state.evidence, 0.5, 0.5)
    withnotice = node(state.model_copy(update={"final_report": annotated}))
    assert withnotice["telemetry"]["grounding_notice_shipped"] is True


def test_telemetry_warns_when_an_ungrounded_report_ships(caplog):
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = _ungrounded_state().model_copy(update={
        "final_report": "# Report\n\nBoth armies are large [g1] [g2]."})
    with caplog.at_level(_logging.WARNING):
        node(state)
    warned = [r for r in caplog.records
              if "report.shipped_ungrounded" in r.message]
    assert warned, "a run whose corpus contributed nothing must be visible"
    assert warned[0].event_fields["notice_shipped"] is False


def test_telemetry_does_not_warn_when_the_corpus_did_ground_the_run(caplog):
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(
        raw_query="Redis eviction",
        goals=[_g("g1")],
        evidence=[Evidence(task_key="t", goal_id="g1", source="corpus",
                           content="desc g1 eviction policies explained",
                           score=0.9)],
        final_report="# Report\n\nGrounded [g1].")
    with caplog.at_level(_logging.WARNING):
        node(state)
    assert not [r for r in caplog.records
                if "report.shipped_ungrounded" in r.message]


# ---------------------------------------------------------------------------
# D-88: guardrail counts scoped to the SHIPPED report, not summed across
# every compile attempt
# ---------------------------------------------------------------------------


def test_compiler_returns_compile_scoped_guardrail_counts():
    """The same numbers reach state two ways: additively via `counters`
    (a legitimate "how much repair did this whole RUN need" signal) and
    replace-on-write via `last_compile_guardrails` (what THIS report
    needed). Only the second can describe the artifact."""
    router = _FakeRouter("# Report\n\nBoth armies are large [g1] [g2].")
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live"))

    result = node(_ungrounded_state())

    assert result["last_compile_guardrails"]["grounding_notice_inserted"] == 1.0
    # Router counters are genuinely run-cumulative and have no per-report
    # meaning, so they must NOT be duplicated into the compile-scoped view.
    assert "llm_node_calls" not in result["last_compile_guardrails"]
    assert "llm_provider_calls" not in result["last_compile_guardrails"]


def test_telemetry_reads_the_report_scoped_field_not_the_summed_counter():
    """The D-88 contract, pinned directly: given a `counters` dict inflated
    by three compile attempts and a `last_compile_guardrails` describing
    only the last, telemetry must report the last.

    This is the exact defect D-59 found for web_sources_listed -- live, a
    two-revision run reported 44 listed against a report containing 34.
    Every one of those numbers was arithmetically correct and none of them
    described what the reader actually received."""
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(
        raw_query="q",
        goals=[_g("g1")],
        evidence=[_e("g1", "desc g1 content", source="corpus", score=0.9)],
        final_report="# Report\n\nGrounded [g1].",
        # Three compile attempts' worth of repairs...
        counters={"citations_pasted_evidence_removed": 9.0},
        # ...but the shipped report needed three.
        last_compile_guardrails={"citations_pasted_evidence_removed": 3.0})

    telemetry = node(state)["telemetry"]

    assert telemetry["last_compile_guardrails"] == {
        "citations_pasted_evidence_removed": 3}


def test_telemetry_surfaces_token_totals_and_tier_answers():
    """D-86 and D-87 together: what the run cost, and which tier of the
    D-38 ladder actually answered it."""
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(
        raw_query="q",
        goals=[_g("g1")],
        evidence=[_e("g1", "desc g1 content", source="corpus", score=0.9)],
        final_report="# Report\n\nGrounded [g1].",
        counters={"llm_prompt_tokens": 4023.0, "llm_completion_tokens": 2068.0,
                  "chain_answered_corpus": 4.0, "chain_answered_web": 2.0,
                  "chain_answered_model": 0.0, "chain_tier_failed": 1.0})

    telemetry = node(state)["telemetry"]

    assert telemetry["llm_prompt_tokens"] == 4023
    assert telemetry["llm_completion_tokens"] == 2068
    assert telemetry["llm_total_tokens"] == 6091
    # A tier that answered nothing is omitted rather than reported as 0 --
    # the dict names what DID answer, so it stays readable as the ladder
    # grows.
    assert telemetry["tier_answers"] == {"corpus": 4, "web": 2}
    assert telemetry["chain_tier_failures"] == 1


# ---------------------------------------------------------------------------
# D-91: the cited-figure audit, wired into telemetry_node
# ---------------------------------------------------------------------------


def _figure_state(report):
    return ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[_g("g1")],
        evidence=[_e("g1", "desc g1 -- the PLA fields about 2,000,000 "
                     "personnel", source="corpus", score=0.9)],
        final_report=report)


def test_telemetry_reports_an_unsupported_cited_figure(caplog):
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = _figure_state(
        "# R\n\nThe PLA fields 2,300,000 active personnel [g1].\n")

    with caplog.at_level(_logging.WARNING):
        telemetry = node(state)["telemetry"]

    assert telemetry["cited_figures_checked"] == 1
    assert telemetry["cited_figures_unsupported"] == 1
    assert telemetry["unsupported_figures"] == [
        {"figure": "2300000", "goals": ["g1"]}]
    assert [r for r in caplog.records
            if "report.unsupported_cited_figures" in r.message]


def test_telemetry_is_quiet_when_every_cited_figure_checks_out(caplog):
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="live"))
    state = _figure_state(
        "# R\n\nThe PLA fields 2,000,000 active personnel [g1].\n")

    with caplog.at_level(_logging.WARNING):
        telemetry = node(state)["telemetry"]

    assert telemetry["cited_figures_checked"] == 1
    assert telemetry["cited_figures_unsupported"] == 0
    assert telemetry["unsupported_figures"] == []
    assert not [r for r in caplog.records
                if "report.unsupported_cited_figures" in r.message]


def test_the_figure_audit_is_gated_off_in_stub_mode(caplog):
    """Same gate as its two neighbouring shipped-report checks (D-66,
    D-85): StubClient's fixed placeholder report carries no [gN] markers
    by design, so there is nothing here to audit."""
    import logging as _logging
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node(Settings(_env_file=None, llm_mode="stub"))
    state = _figure_state(
        "# R\n\nThe PLA fields 2,300,000 active personnel [g1].\n")

    with caplog.at_level(_logging.WARNING):
        telemetry = node(state)["telemetry"]

    assert telemetry["cited_figures_checked"] == 0
    assert telemetry["cited_figures_unsupported"] == 0
    assert not [r for r in caplog.records
                if "report.unsupported_cited_figures" in r.message]


# ---------------------------------------------------------------------------
# D-95: the D-91-triggered semantic judge in critic_node
# ---------------------------------------------------------------------------


class _JudgeRouter(_FakeRouter):
    """A router whose complete_json answers the verify_figures call."""

    def __init__(self, unsupported=None, raises=None):
        super().__init__("unused")
        self._unsupported = unsupported or []
        self._raises = raises
        self.json_calls = 0

    def complete_json(self, messages, temperature=0.0):
        self.json_calls += 1
        # Raise on the JUDGE call only (the first). A fake that
        # raised on every call would also break the ordinary
        # critique that follows a fail-open, and the test would be
        # asserting on its own fake rather than on the code.
        if self._raises and self.json_calls == 1:
            raise self._raises
        if self.json_calls == 1:
            return {"unsupported": self._unsupported}
        return {"passed": True, "notes": []}


def _figure_critic_state():
    """A draft claiming 2,300,000 against evidence that says 2,000,000."""
    return ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[_g("g1")],
        evidence=[_e("g1", "desc g1 -- the PLA fields about 2,000,000 "
                     "personnel", source="corpus", score=0.9)],
        final_report="# R\n\nThe PLA fields 2,300,000 personnel [g1].\n")


def _settings(**kw):
    base = {"_env_file": None, "llm_mode": "live", "max_revisions": 2,
            "claim_verification_enabled": True}
    base.update(kw)
    return Settings(**base)


def test_a_confirmed_unsupported_figure_fails_the_critique():
    """A rewrite CAN fix this -- the compiler can drop or hedge the figure
    -- which is exactly why this may gate where D-85's notice may not."""
    from research_agent.agents.compilation import build_critic_node

    router = _JudgeRouter(unsupported=["2300000"])
    result = build_critic_node(router, _settings())(_figure_critic_state())

    assert result["critique_passed"] is False
    assert "2300000" in result["critique_notes"][0]
    assert result["counters"]["claim_figures_confirmed"] == 1.0
    assert router.json_calls == 1, "the judge call, and not the main critique"


def test_a_cleared_figure_falls_through_to_the_normal_critique():
    """The judge exists to REDUCE false positives: evidence saying "about
    2,000,000" may well support "2,300,000" badly, but evidence saying
    "roughly a seventh" genuinely supports "14.7%". Anything the judge
    does not name is treated as supported."""
    from research_agent.agents.compilation import build_critic_node

    router = _JudgeRouter(unsupported=[])
    result = build_critic_node(router, _settings())(_figure_critic_state())

    # Two JSON calls: the judge cleared it, then the real critique ran.
    assert router.json_calls == 2
    assert "critique_passed" in result


class _PromptCapturingJudge(_JudgeRouter):
    """_JudgeRouter that keeps the prompt it was handed."""

    def __init__(self, unsupported=None):
        super().__init__(unsupported=unsupported)
        self.judge_prompt = ""

    def complete_json(self, messages, temperature=0.0):
        if self.json_calls == 0:
            self.judge_prompt = messages[-1]["content"]
        return super().complete_json(messages, temperature)


def test_the_judge_is_shown_deduplicated_evidence():
    """D-175, the p205.303-check defect.

    templates.verify_figures shows the judge at most 8 evidence items per
    flagged figure. Byte-identical duplicates each consume one of those
    slots, so evidence the figure actually rests on can be crowded out of
    the prompt entirely -- D-46's defect, in the one path added after
    D-46 was fixed. Live, six of the eight slots shown for one figure
    were three corpus items repeated twice.

    The audit itself is deliberately NOT deduplicated: it folds evidence
    into a set of figures per goal, where a duplicate cannot change the
    verdict. Only the bounded prompt cares.
    """
    from research_agent.agents.compilation import build_critic_node

    noise = "An unrelated corpus item about cache eviction."
    state = ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[_g("g1")],
        evidence=([_e("g1", noise, source="corpus", score=0.9)] * 8
                  + [_e("g1", "The PLA fields about 2,000,000 personnel",
                        source="corpus", score=0.9)]),
        final_report="# R\n\nThe PLA fields 2,300,000 personnel [g1].\n")

    router = _PromptCapturingJudge(unsupported=[])
    build_critic_node(router, _settings())(state)

    assert router.judge_prompt.count(noise) == 1, "duplicates collapsed"
    assert "2,000,000" in router.judge_prompt, (
        "the evidence the figure rests on survived the 8-item cap")


def _misattributed_state():
    """975,000 retrieved under g3; the report cites g1 (p205.308-check)."""
    return ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[_g("g1"), _g("g3")],
        evidence=[
            _e("g1", "China's armed forces have over 2.1 million active "
                     "personnel.", source="web", score=0.9),
            _e("g3", "the People's Liberation Army Ground Force (PLAGF) is "
                     "estimated to have a deployed force of 975,000 troops.",
               source="web", score=0.71)],
        final_report=("# R\n\nThe PLAGF is estimated to deploy roughly "
                      "975,000 troops [g1].\n"))


def test_a_misattributed_figure_is_never_put_to_the_judge():
    """D-179. verify_figures shows the judge the CITED goal's evidence and
    asks whether it supports the figure "in any form" -- for a figure
    filed under another goal that is a question whose answer is fixed
    before it is asked. Live, the judge confirmed 975000 as unsupported
    while the run held an item reading "a deployed force of 975,000
    troops", and the compiler deleted a true figure."""
    from research_agent.agents.compilation import build_critic_node

    router = _JudgeRouter(unsupported=["975000"])
    result = build_critic_node(router, _settings())(_misattributed_state())

    assert router.json_calls == 1, "the ordinary critique only -- no judge call"
    assert "975000" not in str(result.get("critique_notes", ""))


def test_a_misattribution_alone_does_not_spend_a_revision():
    """A citation pointing at the wrong goal is a smaller fault than an
    invented number, and the run already reports it."""
    from research_agent.agents.compilation import build_critic_node

    # _VerdictRouter, not _JudgeRouter: with no judge call to make,
    # the single JSON call IS the ordinary critique and must be
    # answered as one.
    router = _VerdictRouter([], passed=True)
    result = build_critic_node(router, _settings())(_misattributed_state())

    assert router.json_calls == 1
    assert result["critique_passed"] is True
    assert result["revision_count"] == 1, "one critic pass, no revision"


def test_the_gate_is_off_by_default():
    """D-54: the false-positive rate has not been measured on real
    reports, so this ships inert -- the same posture
    contradiction_detection_enabled takes."""
    from research_agent.agents.compilation import build_critic_node

    router = _JudgeRouter(unsupported=["2300000"])
    settings = _settings(claim_verification_enabled=False)

    build_critic_node(router, settings)(_figure_critic_state())

    assert router.json_calls == 1, "only the ordinary critique call"


def test_a_judge_that_raises_fails_open(caplog):
    """A verification call that goes wrong must never manufacture a
    failure -- the same posture score_answer takes."""
    import logging as _logging
    from research_agent.agents.compilation import build_critic_node

    router = _JudgeRouter(raises=RuntimeError("provider down"))
    with caplog.at_level(_logging.WARNING):
        result = build_critic_node(router, _settings())(_figure_critic_state())

    # Fail open: nothing confirmed, and the run proceeds to the
    # ORDINARY critique rather than dying on a verification call.
    assert "claim_figures_confirmed" not in (result.get("counters") or {})
    assert result["critique_passed"] is True
    assert router.json_calls == 2
    assert [r for r in caplog.records
            if "critic.claim_verification_failed" in r.message]


def test_the_judge_cannot_invent_a_finding_of_its_own():
    """The deterministic pass owns WHAT MAY BE ACCUSED; the judge may only
    confirm or clear. Letting it add findings would hand an LLM the power
    to fail a report over something no mechanical check ever saw -- the
    exact inversion of this package's own rule."""
    from research_agent.agents.compilation import _confirm_unsupported_figures

    router = _JudgeRouter(unsupported=["9999", "2300000"])
    flagged = [{"figure": "2300000", "goals": ["g1"], "sentence": "s"}]

    assert _confirm_unsupported_figures(router, flagged, []) == ["2300000"]


def test_nothing_flagged_means_no_judge_call_at_all():
    """A clean report never pays for this gate."""
    from research_agent.agents.compilation import _confirm_unsupported_figures

    router = _JudgeRouter(unsupported=["x"])

    assert _confirm_unsupported_figures(router, [], []) == []
    assert router.json_calls == 0


# ---------------------------------------------------------------------------
# D-130 (P6-3) -- the disabled-provider skip reaches the run record
# ---------------------------------------------------------------------------


def test_telemetry_reports_hops_skipped_for_a_dead_provider():
    """The counter exists at the router boundary; this is the half that
    makes it readable in an agent_runs row and the RESULT block. A signal
    recorded where nobody looks is the defect D-105 and D-108 both were.

    Pure pass-through, per D-12: this node adds up what the router
    recorded and invents nothing."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(raw_query="q",
                          goals=[Goal(goal_id="g1", description="d")],
                          counters={"llm_disabled_skips": 2.0,
                                    "llm_context_skips": 1.0})

    telemetry = node(state)["telemetry"]
    assert telemetry["llm_disabled_skips"] == 2
    assert telemetry["llm_context_skips"] == 1, (
        "the two skips are different facts and must not be merged")


def test_telemetry_reports_zero_disabled_skips_on_a_healthy_run():
    """0 is the every-run value, and it has to be present rather than
    absent -- an omitted key reads as "not measured", which is what D-103
    made recall NULL to say honestly."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    telemetry = node(ResearchState(raw_query="q"))["telemetry"]
    assert telemetry["llm_disabled_skips"] == 0


# ---------------------------------------------------------------------------
# D-131 (P6-2) -- the evidence budget is applied where prompts are built
# ---------------------------------------------------------------------------


def _many_evidence(n, goal_ids=("g1", "g2"), chars=500):
    # DISTINCT content per item, deliberately: dedupe_evidence runs first
    # in both nodes and would collapse identical text to one item per
    # goal, leaving nothing for the budget to do.
    return [Evidence(task_key=f"t{i}", goal_id=goal_ids[i % len(goal_ids)],
                     source="web", score=0.7,
                     content=f"item {i}: " + "x" * chars)
            for i in range(n)]


class _PromptCapturingRouter:
    """Records the transcript each node actually sent."""

    def __init__(self, json_payload=None, text="# Report\n\nBody [g1]."):
        self.prompts = {}
        self._json = json_payload or {"passed": True, "score": 0.9, "notes": []}
        self._text = text

    def set_node(self, node):
        self._node = node

    def drain_counters(self):
        return {}

    def complete(self, messages):
        self.prompts["compile"] = messages
        return self._text

    def complete_json(self, messages):
        self.prompts["critique"] = messages
        return self._json


def _evidence_chars(messages):
    """Characters of the <evidence> block in a built prompt."""
    body = messages[-1]["content"]
    start = body.index("<evidence>")
    return len(body[start:body.index("</evidence>", start)])


def test_compiler_node_bounds_the_evidence_it_sends():
    """Run p205.267-check put 30,199 characters of evidence into one
    compile prompt. compile_report has no bound of its own -- this node
    is where the budget is applied."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    settings = Settings(_env_file=None, llm_mode="live",
                        prompt_evidence_max_chars=4000)
    node = build_compiler_node(router, settings)
    state = ResearchState(
        raw_query="q",
        goals=[Goal(goal_id="g1", description="d1"),
               Goal(goal_id="g2", description="d2")],
        evidence=_many_evidence(60))

    out = node(state)

    assert _evidence_chars(router.prompts["compile"]) < 5000
    assert out["counters"]["evidence_prompt_dropped"] > 0
    # D-88: the same number, scoped to the report that actually shipped.
    assert out["last_compile_guardrails"]["evidence_prompt_dropped"] > 0


def test_the_compiler_prompt_still_carries_every_goal_after_budgeting():
    """A budget that dropped a goal entirely would trade a token problem
    for a coverage one."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live",
                                                prompt_evidence_max_chars=4000))
    node(ResearchState(raw_query="q",
                       goals=[Goal(goal_id="g1", description="d1"),
                              Goal(goal_id="g2", description="d2")],
                       evidence=_many_evidence(60)))

    block = router.prompts["compile"][-1]["content"]
    assert "[g1 |" in block and "[g2 |" in block


def test_critic_node_is_bounded_by_the_same_rule():
    """templates.critique used to keep `evidence[-60:]` -- a tail slice
    that, after a third gather lap, keeps the lap that found least. D-46
    is what that costs: a critic judging a report against evidence it was
    never shown."""
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_critic_node(router, Settings(_env_file=None, llm_mode="live",
                                              prompt_evidence_max_chars=4000))
    state = ResearchState(
        raw_query="q", final_report="# R\n\nclaim [g1]",
        goals=[Goal(goal_id="g1", description="d1"),
               Goal(goal_id="g2", description="d2")],
        evidence=_many_evidence(60))

    node(state)

    assert _evidence_chars(router.prompts["critique"]) < 5000


def test_a_disabled_budget_leaves_the_compile_prompt_unbounded():
    """0 restores the pre-D-131 prompt exactly -- the documented escape
    hatch, so it must genuinely change nothing."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live",
                                                prompt_evidence_max_chars=0))
    out = node(ResearchState(raw_query="q",
                             goals=[Goal(goal_id="g1", description="d1")],
                             evidence=_many_evidence(60, goal_ids=("g1",))))

    assert _evidence_chars(router.prompts["compile"]) > 25000
    assert "evidence_prompt_dropped" not in out["counters"]


def test_a_small_run_is_byte_identical_with_the_budget_on():
    """Every guardrail in this codebase carries this test: with nothing to
    trim, the pass must be invisible."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    state = ResearchState(raw_query="q",
                          goals=[Goal(goal_id="g1", description="d1")],
                          evidence=_many_evidence(3, goal_ids=("g1",)))

    bounded = _PromptCapturingRouter()
    build_compiler_node(bounded, Settings(_env_file=None, llm_mode="live",
                                          prompt_evidence_max_chars=12000))(state)
    unbounded = _PromptCapturingRouter()
    build_compiler_node(unbounded, Settings(_env_file=None, llm_mode="live",
                                            prompt_evidence_max_chars=0))(state)

    assert bounded.prompts["compile"] == unbounded.prompts["compile"]


# ---------------------------------------------------------------------------
# D-132 (P6-4) -- the compiler's own budget check, the notice, the record
# ---------------------------------------------------------------------------


def test_compiler_node_flags_a_budget_spent_in_the_revision_loop():
    """The case progress_checker cannot see: a run that converged in one
    lap and then spent its budget compiling and re-compiling."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    settings = Settings(_env_file=None, llm_mode="live",
                        run_deadline_seconds=1.0)
    node = build_compiler_node(router, settings)
    state = ResearchState(raw_query="q", run_started_at=1.0,
                          goals=[Goal(goal_id="g1", description="d")],
                          evidence=_many_evidence(3, goal_ids=("g1",)))

    out = node(state)

    assert out["budget_exhausted"] == "deadline"


def test_a_truncated_run_says_so_in_the_report():
    """D-85's argument in a second place: telemetry is read by whoever
    runs the agent, the report by whoever asked the question."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live"))
    state = ResearchState(raw_query="q", budget_exhausted="deadline",
                          goals=[Goal(goal_id="g1", description="d")],
                          evidence=_many_evidence(3, goal_ids=("g1",)))

    out = node(state)

    assert out["final_report"].startswith("> **Run stopped early")
    assert out["last_compile_guardrails"]["truncation_notice_inserted"] == 1


def test_the_stopped_early_notice_sits_above_the_provenance_one():
    """Ordering is deliberate -- "this run was stopped early" changes how
    a reader weighs everything below it, the provenance notice
    included."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live"))
    # Ungrounded (web-only evidence) AND truncated -> both notices apply.
    state = ResearchState(raw_query="q", budget_exhausted="tokens",
                          goals=[Goal(goal_id="g1", description="d")],
                          evidence=_many_evidence(3, goal_ids=("g1",)))

    report = node(state)["final_report"]

    assert report.index("Run stopped early") < report.index("Provenance notice")


def test_a_run_inside_its_budget_ships_no_notice():
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None, llm_mode="live"))
    out = node(ResearchState(raw_query="q",
                             goals=[Goal(goal_id="g1", description="d")],
                             evidence=_many_evidence(3, goal_ids=("g1",))))

    assert "Run stopped early" not in out["final_report"]
    assert "truncation_notice_inserted" not in out["last_compile_guardrails"]


def test_telemetry_records_what_stopped_the_run_and_what_it_spent():
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="q", goals=[Goal(goal_id="g1", description="d")],
        budget_exhausted="tokens", run_started_at=1.0, paused_seconds=68.0,
        final_report="> **Run stopped early — inserted automatically.**\n\n# R")

    telemetry = node(state)["telemetry"]

    assert telemetry["run_budget_exhausted"] == "tokens"
    assert telemetry["run_paused_seconds"] == 68.0
    assert telemetry["run_elapsed_seconds"] > 0
    # D-59: read from the ARTIFACT, not from a counter that would sum
    # every compile attempt.
    assert telemetry["truncation_notice_shipped"] is True


def test_telemetry_on_an_ordinary_run_reports_no_budget_stop():
    """None, not a string -- "finished on its own terms" must be
    distinguishable from "stopped by something" at a glance."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    telemetry = node(ResearchState(raw_query="q"))["telemetry"]

    assert telemetry["run_budget_exhausted"] is None
    assert telemetry["run_elapsed_seconds"] == 0.0
    assert telemetry["truncation_notice_shipped"] is False


# ---------------------------------------------------------------------------
# D-137 — the glue signature without the verbatim confirmation
# ---------------------------------------------------------------------------

def test_a_paraphrased_restatement_glued_to_a_claim_is_separated():
    """The live shape, from run p205.277-check's shipped report. This
    sentence appears NOWHERE in the evidence block -- the model wrote its
    own restatement and welded it on where a [gN] marker belongs -- so the
    verbatim paste guard cannot see it."""
    from research_agent.guardrails.citations import repair_glued_sentences

    report = ("The establishment of a new mountain strike corps is reported "
              "as part of efforts along the disputed Himalayan borderIndia "
              "raised a new mountain strike corps to strengthen its defence "
              "along its disputed border with China.")

    repaired, counters = repair_glued_sentences(report)

    assert "Himalayan border. India raised" in repaired
    assert counters["citations_glued_sentences_repaired"] == 1.0


def test_every_glued_site_in_one_paragraph_is_separated():
    from research_agent.guardrails.citations import (repair_glued_sentences,
                                                     residual_glue_sites)

    report = ("China leads on personnel and equipment across the boardIndia "
              "fields the second largest ground force in the world today. "
              "Both navies are expanding their carrier fleets quicklyIndia "
              "operates fewer hulls than China does at present.")

    repaired, counters = repair_glued_sentences(report)

    assert counters["citations_glued_sentences_repaired"] == 2.0
    assert residual_glue_sites(repaired) == 0


def test_a_correctly_formatted_report_is_returned_byte_identical():
    """The common path. Not "similar" -- identical, and with no counter,
    so a clean run's guardrail block is unchanged."""
    from research_agent.guardrails.citations import repair_glued_sentences

    report = ("# Report\n\nChina fields more personnel [g1]. India fields "
              "fewer but is modernising quickly [g2].\n")

    assert repair_glued_sentences(report) == (report, {})


def test_camel_case_names_are_never_split():
    """The signature has to survive ordinary prose. Each of these carries
    the lowercase-then-capital shape the weak signal matches on, and none
    of them is a missing sentence boundary."""
    from research_agent.guardrails.citations import repair_glued_sentences

    for text in (
            "eBay is a large online marketplace with many active sellers.",
            "We use LinkedIn to recruit engineers across several regions.",
            "The McKinsey report was published last year for its clients.",
            "PayPal processes payments for merchants in many countries.",
            "The iPhone remains the best selling handset in some markets.",
            "Run PostgreSQL migrations before deploying the service today.",
    ):
        assert repair_glued_sentences(text) == (text, {}), text


def test_a_digit_before_the_capital_is_never_split():
    """Live near-miss: "Type 054B frigate" carries the shape and is a
    model number. The left side must be LETTERS."""
    from research_agent.guardrails.citations import repair_glued_sentences

    text = "A new Type 054B frigate launched with a vertical launch system."

    assert repair_glued_sentences(text) == (text, {})


def test_two_words_that_merely_lost_a_space_are_left_alone():
    """The run must be a whole SENTENCE. Six words is the floor, so a
    missing space inside a phrase is not turned into a full stop."""
    from research_agent.guardrails.citations import repair_glued_sentences

    text = "The forces deployed acrossChina and India."

    assert repair_glued_sentences(text) == (text, {})


def test_a_url_is_never_punctuated():
    """A link's path carries the same shape, and a full stop inserted into
    one breaks it."""
    from research_agent.guardrails.citations import repair_glued_sentences

    text = ("See https://www.orfonline.org/researchIndia and China have "
            "both expanded their naval forces considerably.")

    assert repair_glued_sentences(text) == (text, {})


def test_a_verbatim_paste_is_removed_rather_than_punctuated():
    """Ordering check. clean_citations runs the verbatim strip first, so
    the stronger verdict gets first refusal at a site both could claim."""
    from research_agent.guardrails.citations import (clean_citations,
                                                     repair_glued_sentences)

    source = ("India fields roughly 1.45 million active personnel across "
              "three services today.")
    report = "China leads on raw numbers" + source
    evidence = [_e("g1", source)]

    cleaned, counters = clean_citations(report, [_g("g1")], evidence)
    repaired, glue_counters = repair_glued_sentences(cleaned)

    assert counters["citations_pasted_evidence_removed"] == 1.0
    assert source not in repaired
    assert glue_counters == {}


def test_residual_glue_sites_reads_the_shipped_artifact():
    """The counter that closes the honesty gap: two live reports shipped
    with 9 and 22 welded joins while citations_residual_paste_sites read
    0, because a paste was the only thing it could see."""
    from research_agent.guardrails.citations import residual_glue_sites

    report = ("China leads on personnel and equipment across the boardIndia "
              "fields the second largest ground force in the world today.")

    assert residual_glue_sites(report) == 1
    assert residual_glue_sites("") == 0


def test_compiler_node_separates_glued_sentences_before_shipping():
    """Wiring: the repair has to run inside the node, not only as a pure
    function nobody calls."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _FakeRouter(
        "# Report\n\nChina leads on personnel and equipment across the "
        "boardIndia fields the second largest ground force today [g1].\n")
    node = build_compiler_node(router, Settings(_env_file=None,
                                                llm_mode="live"))

    out = node(ResearchState(raw_query="q",
                             goals=[Goal(goal_id="g1", description="d1")],
                             evidence=[_e("g1", "unrelated evidence text")]))

    assert "board. India fields" in out["final_report"]
    assert out["counters"]["citations_glued_sentences_repaired"] == 1.0
    assert out["last_compile_guardrails"][
        "citations_glued_sentences_repaired"] == 1.0


# ---------------------------------------------------------------------------
# D-138 — the critique notes entering a compile prompt
# ---------------------------------------------------------------------------

def test_compiler_node_bounds_the_critique_notes_it_sends():
    """Live (p205.277-check) the third compile opened with 16 notes /
    4,947 chars about three different drafts, and dropped its citations
    entirely."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None,
                                                llm_mode="live"))
    notes = [f"note {i} " + "x" * 400 for i in range(20)]

    out = node(ResearchState(raw_query="q",
                             goals=[Goal(goal_id="g1", description="d1")],
                             evidence=[_e("g1", "text")],
                             critique_notes=notes))

    body = router.prompts["compile"][-1]["content"]
    assert "note 19" in body          # newest kept
    assert "note 0 " not in body      # oldest, about a superseded draft
    assert out["counters"]["critique_notes_dropped"] > 0


def test_an_ordinary_revision_carries_every_note():
    """The bound sits above the largest verdict yet observed (10 notes /
    2,722 chars), so a real critique is never truncated."""
    from research_agent.agents.compilation import build_compiler_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_compiler_node(router, Settings(_env_file=None,
                                                llm_mode="live"))
    notes = [f"Goal g{i}: the report claims something unsupported. FAIL."
             for i in range(10)]

    out = node(ResearchState(raw_query="q",
                             goals=[Goal(goal_id="g1", description="d1")],
                             evidence=[_e("g1", "text")],
                             critique_notes=notes))

    body = router.prompts["compile"][-1]["content"]
    for note in notes:
        assert note in body
    assert "critique_notes_dropped" not in out["counters"]


# ---------------------------------------------------------------------------
# D-139 — the critic judges what the MODEL wrote
# ---------------------------------------------------------------------------

def test_the_critic_is_not_shown_the_provenance_notice():
    """Live (p205.276-check) three of six notes demanded the removal of a
    notice the compiler never wrote and cannot remove."""
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_critic_node(router, Settings(_env_file=None,
                                              llm_mode="live"))
    report = ("> **Provenance notice — inserted automatically, not written "
              "by the model.**\n> None of this report's goals are supported "
              "by a document.\n\n# Report\n\nChina leads [g1].\n")

    node(ResearchState(raw_query="q", final_report=report,
                       goals=[Goal(goal_id="g1", description="d1")],
                       evidence=[_e("g1", "text")]))

    body = router.prompts["critique"][-1]["content"]
    assert "Provenance notice" not in body
    assert "China leads [g1]." in body


def test_the_critic_is_not_shown_the_sources_block():
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_critic_node(router, Settings(_env_file=None,
                                              llm_mode="live"))
    report = ("# Report\n\nChina leads [g1].\n\n## Sources\n\n"
              "1. [g1] A Title (example.com) — https://example.com/a\n")

    node(ResearchState(raw_query="q", final_report=report,
                       goals=[Goal(goal_id="g1", description="d1")],
                       evidence=[_e("g1", "text")]))

    body = router.prompts["critique"][-1]["content"]
    assert "example.com" not in body
    assert "China leads [g1]." in body


def test_the_zero_citation_gate_still_reads_the_shipped_report():
    """Deliberately unchanged (D-139). append_web_sources only lists goals
    the PROSE cites, so a report citing nothing has no Sources block for
    the gate to miscount -- and the gate must keep failing that report."""
    from research_agent.agents.compilation import build_critic_node
    from research_agent.config import Settings

    router = _PromptCapturingRouter()
    node = build_critic_node(router, Settings(_env_file=None,
                                              llm_mode="live"))

    out = node(ResearchState(raw_query="q",
                             final_report="# Report\n\nChina leads.\n",
                             goals=[Goal(goal_id="g1", description="d1")],
                             evidence=[_e("g1", "text")]))

    assert out["critique_passed"] is False
    assert "cites no evidence" in out["critique_notes"][0]


# ---------------------------------------------------------------------------
# D-144 -- the whole attribution repair, at the pipeline level
#
# Runs p205.276, p205.277 and p205.280 each shipped a report with ZERO
# [gN] markers against 35-100 evidence items, and the damage was never
# confined to the missing markers: three guardrails read attribution
# through the same function and failed together and silently.
#
# p205.280-check, 100 evidence items:
#     evidence_cited            0
#     web_sources_listed        0   of 58 items, 33 distinct domains
#     web_sources_suppressed   58
#     cited_figures_checked     0   (D-91 audited nothing)
#     -> E4, approved by a human because the prose read fine
# ---------------------------------------------------------------------------

_UNCITED_REPORT = (
    "# Comparative Analysis of China's and India's Armed Forces\n"
    "\n"
    "## 1. Size and Composition of Active-Duty Military Forces\n"
    "\n"
    "India's active force is reported at approximately 1.45 million "
    "personnel, while China's active force is significantly larger.\n")


def _army_state():
    goals = [Goal(goal_id="g1",
                  description="Compare the size and composition of the "
                              "active-duty military forces of China and India")]
    evidence = [
        Evidence(task_key="t1", goal_id="g1", source="web", score=0.75,
                 content="India Military Strength — India ranks #4 with 1.45 "
                         "million active military personnel",
                 url="https://a.example/india", domain="a.example"),
        Evidence(task_key="t2", goal_id="g1", source="web", score=0.71,
                 content="China military personnel by type 2023 — active "
                         "military personnel 2,035,000",
                 url="https://b.example/china", domain="b.example"),
    ]
    return ResearchState(raw_query="Compare the Armies of China and India",
                         goals=goals, evidence=evidence)


def test_an_uncited_report_leaves_the_compiler_cited_and_sourced():
    """The p205.280-check failure, end to end. Both halves must recover:
    the markers, and the Sources block that was gated on them."""
    node = build_compiler_node(_FakeRouter(_UNCITED_REPORT), _SETTINGS)

    result = node(_army_state())
    report = result["final_report"]

    assert "[g1]" in report, "the attachment pass did not fire"
    assert "## Sources" in report
    assert "https://a.example/india" in report
    assert result["counters"]["citations_attached"] >= 1
    assert result["counters"]["web_sources_listed"] >= 1


def test_the_rescue_is_visible_in_the_per_report_guardrail_block():
    """D-88's last_compile_guardrails is what a reader consults to tell a
    report the model cited from one that was repaired. If the rescue were
    invisible there, this would be a nicer lie rather than a fix."""
    node = build_compiler_node(_FakeRouter(_UNCITED_REPORT), _SETTINGS)

    result = node(_army_state())

    assert result["last_compile_guardrails"]["citations_attached"] >= 1


def test_a_report_the_model_cited_itself_is_not_touched_by_the_pass():
    """The no-op path, which is the healthy one."""
    cited = (
        "## 1. Size and Composition\n"
        "\n"
        "India's active force is approximately 1.45 million personnel. [g1]\n")
    node = build_compiler_node(_FakeRouter(cited), _SETTINGS)

    result = node(_army_state())

    from research_agent.guardrails.claims import (cited_goal_ids_in_prose,
                                                  report_body)

    assert "citations_attached" not in result["counters"]
    assert cited_goal_ids_in_prose(result["final_report"]) == {"g1"}
    # One marker in, one marker out: the pass added nothing to the prose.
    assert report_body(result["final_report"]).count("[g1]") == 1


def test_a_report_nothing_can_be_attached_to_still_lists_its_sources():
    """The fallback that stops 33 real domains disappearing because of a
    formatting miss. The prose here shares no distinctive term with any
    evidence, so attachment correctly declines -- and the Sources block
    must still ship, labelled for what it is."""
    from research_agent.guardrails.sources import UNCITED_NOTE

    unrelated = ("## Overview\n"
                 "\n"
                 "Several considerations follow from the preceding discussion "
                 "of the wider topic.\n")
    node = build_compiler_node(_FakeRouter(unrelated), _SETTINGS)

    result = node(_army_state())
    report = result["final_report"]

    assert "citations_attached" not in result["counters"]
    assert "## Sources" in report
    assert UNCITED_NOTE in report
    assert result["counters"]["web_sources_listed_uncited"] >= 1


def test_evidence_cited_counts_prose_not_the_sources_block():
    """The latent defect D-144 had to fix first: every Sources entry begins
    "1. [g1] " by construction, so a whole-report read reported an uncited
    report as cited -- which would have made this count wrong, the D-66
    gate silent, and telemetry's backstop agree with both."""
    from research_agent.guardrails.claims import cited_goal_ids_in_prose

    unrelated = ("## Overview\n"
                 "\n"
                 "Several considerations follow from the preceding discussion "
                 "of the wider topic.\n")
    node = build_compiler_node(_FakeRouter(unrelated), _SETTINGS)

    report = node(_army_state())["final_report"]

    assert "[g1]" in report, "the Sources block does carry markers"
    assert cited_goal_ids_in_prose(report) == set(), "but the prose does not"



# ---------------------------------------------------------------------------
# D-146 -- telemetry, by concern
#
# telemetry_node was 531 lines, of which roughly 350 were one dict literal
# interleaved with the comments explaining each field. Nothing in it could
# be tested without running the whole node. These three builders are the
# part that reads nothing but state.counters.
# ---------------------------------------------------------------------------


def test_llm_metrics_reads_only_counters_and_defaults_everything():
    from research_agent.reporting.telemetry import llm_metrics

    empty = llm_metrics({})

    assert empty["llm_node_calls"] == 0
    assert empty["llm_total_tokens"] == 0
    assert set(empty) == {
        "llm_node_calls", "llm_provider_calls", "llm_fallback_hops",
        "llm_quality_calls", "llm_quality_calls_failed",
        "llm_quality_rejections", "llm_prompt_tokens",
        "llm_completion_tokens", "llm_total_tokens", "llm_context_skips",
        # D-153: which provider was skipped, not just how many times.
        "context_skips_by_provider",
        "llm_disabled_skips"}


def test_llm_total_tokens_is_the_sum_of_the_two_halves():
    from research_agent.reporting.telemetry import llm_metrics

    out = llm_metrics({"llm_prompt_tokens": 19437,
                        "llm_completion_tokens": 5164})

    assert out["llm_total_tokens"] == 24601, "p205.280-check's own figures"


def test_retrieval_metrics_derives_tier_answers_from_the_chain_counters():
    from research_agent.reporting.telemetry import retrieval_metrics

    out = retrieval_metrics({"chain_answered_web": 12,
                              "chain_answered_corpus": 0,
                              "retrieval_dense_calls": 24})

    assert out["tier_answers"] == {"web": 12}, "a zero tier is not an answer"
    assert out["retrieval_dense_calls"] == 24


def test_run_metrics_is_the_whole_run_tally():
    from research_agent.reporting.telemetry import run_metrics

    out = run_metrics({"search_calls": 12, "memory_hits": 5,
                        "revision_cycles": 2})

    assert out["search_calls"] == 12
    assert out["search_failures"] == 0
    assert out["memory_hits"] == 5
    assert out["revision_cycles"] == 2


def test_the_three_builders_emit_disjoint_keys():
    """They are merged with ** into one dict; overlapping keys would mean a
    field silently taking whichever builder ran last."""
    from research_agent.agents.compilation import (llm_metrics,
                                                   retrieval_metrics,
                                                   run_metrics)

    keys = [set(fn({})) for fn in (llm_metrics, retrieval_metrics,
                                   run_metrics)]
    assert not (keys[0] & keys[1])
    assert not (keys[0] & keys[2])
    assert not (keys[1] & keys[2])


def test_they_are_pure_and_do_not_mutate_the_counters_they_read():
    from research_agent.agents.compilation import (llm_metrics,
                                                   retrieval_metrics,
                                                   run_metrics)

    counters = {"llm_node_calls": 6, "search_calls": 12,
                "chain_answered_web": 12}
    before = dict(counters)

    for fn in (llm_metrics, retrieval_metrics, run_metrics):
        fn(counters)

    assert counters == before



class _VerdictRouter(_FakeRouter):
    """A router whose critique call fails the report on the notes given."""

    def __init__(self, notes, passed=False):
        super().__init__("unused")
        self._notes = notes
        self._passed = passed
        self.json_calls = 0

    def complete_json(self, messages, temperature=0.0):
        self.json_calls += 1
        return {"passed": self._passed, "notes": self._notes}


def _corroboration_state(report, evidence):
    return ResearchState(
        raw_query="Compare Armies of China and India",
        goals=[_g("g1")],
        evidence=evidence,
        final_report=report)


# The p205.287 shape: the report rounds, the evidence carries the figure,
# and the critic calls the rounding unfaithful.
_ROUNDING_EVIDENCE = [
    _e("g1", "The PLA is estimated at approximately 2 million to 2.1 "
             "million active personnel."),
    _e("g1", "The force was formally established in 1948."),
]
_ROUNDING_REPORT = ("# R\n\nThe PLA fields approximately 2 million "
                    "personnel [g1], and was founded in 1948 [g1].\n")
_ROUNDING_NOTES = [
    "Unfaithful: the report claims 'approximately 2 million personnel'. "
    "The evidence states 'approximately 2 million to 2.1 million', so the "
    "report's figure omits the upper bound.",
    "Unfaithful: the report dates the founding to 1948; no evidence item "
    "supports 1948.",
]


def test_critic_node_resolves_a_verdict_the_evidence_refutes():
    """D-155, end to end. Both notes dispute a figure the evidence the
    critic was SHOWN actually contains, and D-91 flagged nothing on the
    same report. Before this, the run took the LLM's word and spent a
    revision -- or, twice in three live runs, an E4 escalation."""
    from research_agent.agents.compilation import build_critic_node

    router = _VerdictRouter(_ROUNDING_NOTES)
    result = build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(_ROUNDING_REPORT, _ROUNDING_EVIDENCE))

    assert result["critique_passed"] is True
    assert result["counters"]["critique_notes_dismissed"] == 2.0
    assert "critique_notes" not in result, \
        "a passing verdict must not accumulate notes into the next cycle"


def test_critic_node_leaves_an_unadjudicatable_note_alone():
    """A coverage finding is what the LLM critic is FOR, and one of them
    stops the whole resolution."""
    from research_agent.agents.compilation import build_critic_node

    notes = _ROUNDING_NOTES + ["The report never addresses goal g1's "
                               "second half."]
    router = _VerdictRouter(notes)
    result = build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(_ROUNDING_REPORT, _ROUNDING_EVIDENCE))

    assert result["critique_passed"] is False
    assert result["critique_notes"] == notes
    assert "critique_notes_dismissed" not in result["counters"]


def test_a_clean_pass_is_untouched():
    """The common path. No counter, no log line, nothing added."""
    from research_agent.agents.compilation import build_critic_node

    router = _VerdictRouter([], passed=True)
    result = build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(_ROUNDING_REPORT, _ROUNDING_EVIDENCE))

    assert result["critique_passed"] is True
    assert "critique_notes_dismissed" not in result["counters"]


def test_the_counterweight_defers_to_the_audit_even_with_the_gate_off():
    """D-178. D-155's stated safety property, made real by default.

    Same shape as the resolution test above -- both notes dispute a
    figure the evidence contains, so they WOULD be dismissed -- except
    the report also states a figure no evidence supports, which D-91
    flags. Before D-178 the audit only ran when
    claim_verification_enabled was set, so with the flag off (the
    shipped default) `audit_flagged` was 0 on every run and the
    counterweight was free to overrule the critic on a report the
    deterministic check would have condemned. It never learned there
    was anything to defer to.

    Live: p205.304-check ran with the flag unset and shipped a report
    scoring cited_figures_unsupported: 2.
    """
    from research_agent.agents.compilation import build_critic_node

    report = ("# R\n\nThe PLA fields approximately 2 million personnel "
              "[g1], and was founded in 1948 [g1]. It operates 4,317 main "
              "battle tanks [g1].\n")
    router = _VerdictRouter(_ROUNDING_NOTES)

    result = build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(report, _ROUNDING_EVIDENCE))

    assert result["critique_passed"] is False, \
        "the audit flagged 4,317; the critic's failure must stand"
    assert "critique_notes_dismissed" not in result["counters"]


def test_the_gate_off_still_costs_no_model_call():
    """D-178 buys the audit, not a call. audit_cited_figures is string
    matching over one report -- the flag gates the JUDGE and the GATING,
    and with it off neither happens however much the audit flags."""
    from research_agent.agents.compilation import build_critic_node

    report = ("# R\n\nThe PLA fields approximately 2 million personnel "
              "[g1]. It operates 4,317 main battle tanks [g1].\n")
    router = _JudgeRouter(unsupported=["4317"])

    build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(report, _ROUNDING_EVIDENCE))

    assert router.json_calls == 1, "only the ordinary critique call"


class _ViolationsRouter(_FakeRouter):
    """A critic answering the D-181 `violations` key."""

    def __init__(self, entries, passed=False):
        super().__init__("unused")
        self._entries, self._passed = entries, passed

    def complete_json(self, messages, temperature=0.0):
        return {"passed": self._passed, "score": 0.7,
                "violations": self._entries}


def test_the_critic_reads_the_violations_key():
    """D-181. The contract field, not the legacy one."""
    from research_agent.agents.compilation import build_critic_node

    router = _ViolationsRouter(["g1: the figure 8,000 is not supported."])
    result = build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(_ROUNDING_REPORT, _ROUNDING_EVIDENCE))

    assert result["critique_passed"] is False
    assert result["critique_notes"] == ["g1: the figure 8,000 is not supported."]


def test_an_affirmation_never_becomes_an_instruction_to_the_next_compile():
    """p205.308-check: 21 of 23 entries said a claim was faithful, and
    templates.compile renders every entry under "Address every note"."""
    from research_agent.agents.compilation import build_critic_node

    router = _ViolationsRouter([
        "g1: evidence shows 14,55,550 active troops, so the claim is faithful.",
        "g1: the ranking is not supported by any evidence item.",
    ])
    result = build_critic_node(router, _settings(
        claim_verification_enabled=False))(
            _corroboration_state(_ROUNDING_REPORT, _ROUNDING_EVIDENCE))

    assert result["critique_notes"] == [
        "g1: the ranking is not supported by any evidence item."]
    assert result["counters"]["critique_affirmations_dropped"] == 1.0


def test_the_verdict_is_checked_against_the_evidence_the_critic_saw():
    """Not state.evidence: the critic is budgeted (D-131), so an item
    dropped from its prompt is an item it could not have read. Falsifying
    a note against evidence the critic never saw would be a different and
    much weaker claim."""
    import research_agent.agents.compilation as comp

    seen = {}
    original = comp.resolve_verdict

    def spy(passed, notes, evidence, audit_flagged):
        seen["evidence"] = list(evidence)
        return original(passed, notes, evidence, audit_flagged)

    comp.resolve_verdict = spy
    try:
        router = _VerdictRouter(_ROUNDING_NOTES)
        comp.build_critic_node(router, _settings(
            claim_verification_enabled=False, prompt_evidence_max_chars=80))(
                _corroboration_state(_ROUNDING_REPORT, _ROUNDING_EVIDENCE))
    finally:
        comp.resolve_verdict = original

    assert len(seen["evidence"]) < len(_ROUNDING_EVIDENCE), \
        "the budget must have dropped an item for this test to mean anything"


# ---------------------------------------------------------------------------
# D-162 -- the glue repair must not split a proper noun
# ---------------------------------------------------------------------------


def test_glue_repair_leaves_camelcase_product_names_alone():
    """The signature was measured against eBay / LinkedIn / McKinsey /
    PayPal / iPhone / PostgreSQL, all of which carry three lowercase
    letters or fewer before the capital and so could never match. Names
    with four or more were never tested, and were being split:

        standardised on TensorFlow -> standardised on Tensor. Flow
    """
    from research_agent.guardrails.citations import (repair_glued_sentences,
                                                     residual_glue_sites)

    for report in (
        "The team standardised on TensorFlow for training and on PowerPoint "
        "for the quarterly deck [g1].",
        "Routing data is taken from OpenStreetMap, and payment rails run "
        "through MasterCard settlement [g2].",
        "Deployment is scripted in PowerShell and documented in SharePoint "
        "for the operations team [g1].",
    ):
        out, counters = repair_glued_sentences(report)
        assert out == report, out
        assert counters == {}
        assert residual_glue_sites(report) == 0


def test_glue_repair_still_repairs_a_genuinely_glued_sentence():
    """The discriminator is the TOKEN'S first letter, not the boundary: a
    glued sentence ends on an ordinary lowercase word, a CamelCase proper
    noun is capitalised at the start. This is the case that must keep
    working."""
    from research_agent.guardrails.citations import repair_glued_sentences

    out, counters = repair_glued_sentences(
        "The cache holds session objects in memoryThe eviction policy is "
        "LRU by default [g1].")

    assert out == ("The cache holds session objects in memory. The eviction "
                   "policy is LRU by default [g1].")
    assert counters == {"citations_glued_sentences_repaired": 1.0}


def test_glue_repair_skips_fenced_code():
    """`maxmemoryPolicy` in an ini block is a key, not two sentences."""
    from research_agent.guardrails.citations import repair_glued_sentences

    report = ("Configuration below [g1].\n\n```ini\n"
              "maxmemoryPolicy allkeys-lru and six more words follow here.\n```\n")
    out, counters = repair_glued_sentences(report)

    assert out == report
    assert counters == {}


def test_citations_attached_describes_the_shipped_report_not_the_whole_run():
    """D-162. `state.counters` SUMS across every compile (merge_counters),
    so a run whose FIRST draft needed D-144's rescue and whose FINAL draft
    the model cited itself reported the stale 1 from the first draft.
    confidence.py caps that at 60 (MODERATE) with the reason "the model
    wrote none of them" -- about a report where the model wrote all of
    them. Its neighbour `evidence_cited` was already read from the
    artifact; this now is too."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="Compare Redis and Memcached",
        goals=[Goal(goal_id="g1", description="Compare them")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.99, content="Redis is an in-memory store")],
        final_report="# Report\n\nRedis persists to disk [g1].\n",
        recall_score=1.0, iteration_depth=1,
        # the first compile attached one marker; the LAST attached none
        counters={"citations_attached": 1.0},
        last_compile_guardrails={})

    telemetry = node(state)["telemetry"]

    assert telemetry["citations_attached"] == 0, (
        "the shipped report's own markers were written by the model")
    assert telemetry["confidence"]["band"] != "MODERATE" or not [
        r for r in telemetry["confidence"]["reasons"]
        if "attached deterministically" in r], (
        "a self-cited report must not be capped as machine-attributed")


def test_citations_attached_still_reports_a_real_rescue():
    """The other direction: when the SHIPPED report's markers really were
    attached by D-144, the cap must still fire."""
    from research_agent.agents.compilation import build_telemetry_node
    from research_agent.config import Settings

    node = build_telemetry_node(Settings(_env_file=None))
    state = ResearchState(
        raw_query="Compare Redis and Memcached",
        goals=[Goal(goal_id="g1", description="Compare them")],
        evidence=[Evidence(task_key="t1", goal_id="g1", source="corpus",
                           score=0.99, content="Redis is an in-memory store")],
        final_report="# Report\n\nRedis persists to disk [g1].\n",
        recall_score=1.0, iteration_depth=1,
        counters={"citations_attached": 5.0},
        last_compile_guardrails={"citations_attached": 1.0})

    telemetry = node(state)["telemetry"]

    assert telemetry["citations_attached"] == 1
