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
from research_agent.llm.client import strip_code_fence
from research_agent.state import Evidence, Goal, ResearchState


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
    node = build_compiler_node(router)
    state = ResearchState(raw_query="q")

    result = node(state)

    assert result["final_report"] == '{"title": "x", "findings": {}}'
    assert "```" not in result["final_report"]


def test_compiler_node_leaves_clean_markdown_unchanged():
    router = _FakeRouter("# Report\n\nRedis is fast. [g1 | corpus | score=0.90]")
    node = build_compiler_node(router)
    state = ResearchState(raw_query="q")

    result = node(state)

    assert result["final_report"] == "# Report\n\nRedis is fast. [g1 | corpus | score=0.90]"


def test_compiler_node_abort_path_is_unaffected_by_fence_stripping():
    """The two non-LLM report shapes (abort, planning_error) never call
    strip_code_fence at all — confirm the fix didn't touch that branch."""
    router = _FakeRouter("unused")
    node = build_compiler_node(router)
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

    node = build_telemetry_node()
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


def test_corpus_recall_counts_a_genuinely_on_topic_document():
    from research_agent.agents.compilation import build_telemetry_node

    node = build_telemetry_node()
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
