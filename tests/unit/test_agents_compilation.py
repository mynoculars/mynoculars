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

from research_agent.guardrails.citations import clean_citations  # noqa: E402


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
