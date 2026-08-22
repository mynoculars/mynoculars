"""
tests/unit/test_guardrails_dedup.py — guardrails/dedup.py::dedupe_evidence.

Covers: the per-goal collapse itself, the boundaries it deliberately does
NOT cross (across goals, near-identical text), which copy survives, order
stability, and the empty-counters convention. Does NOT cover the compiler
node's call site (see test_agents_compilation.py) — this file tests the
pure function only.
"""

from research_agent.guardrails.dedup import dedupe_evidence
from research_agent.state import Evidence


def _ev(goal_id, content, score=0.5, source="corpus"):
    return Evidence(task_key=f"t-{goal_id}-{score}-{source}", goal_id=goal_id,
                    source=source, content=content, score=score)


def test_identical_content_under_one_goal_collapses_to_one():
    # The live shape: the same corpus sentence arriving again as an MCP hit,
    # then again on the next gather lap.
    items = [_ev("g1", "Redis supports replication.", 0.50, "corpus"),
             _ev("g1", "Redis supports replication.", 0.50, "mcp"),
             _ev("g1", "Redis supports replication.", 0.48, "corpus")]
    kept, counters = dedupe_evidence(items)
    assert len(kept) == 1
    assert counters["evidence_deduplicated"] == 2.0


def test_same_sentence_under_different_goals_is_kept_for_each():
    # NOT a duplicate: a sentence genuinely covering two goals is two facts
    # about coverage, and the compiler needs both goal tags to cite either.
    items = [_ev("g1", "Redis supports replication."),
             _ev("g3", "Redis supports replication.")]
    kept, counters = dedupe_evidence(items)
    assert [e.goal_id for e in kept] == ["g1", "g3"]
    assert counters == {}


def test_highest_scoring_copy_survives():
    items = [_ev("g1", "same text", 0.48, "corpus"),
             _ev("g1", "same text", 0.75, "mcp"),
             _ev("g1", "same text", 0.50, "corpus")]
    kept, _ = dedupe_evidence(items)
    assert len(kept) == 1
    assert kept[0].score == 0.75
    assert kept[0].source == "mcp"


def test_surviving_copy_holds_its_original_position():
    # A later, higher-scoring duplicate replaces the earlier one IN PLACE —
    # it must not jump to the end and reorder the evidence block.
    items = [_ev("g1", "first", 0.4),
             _ev("g1", "dupe", 0.4),
             _ev("g1", "last", 0.4),
             _ev("g1", "dupe", 0.9)]
    kept, _ = dedupe_evidence(items)
    assert [e.content for e in kept] == ["first", "dupe", "last"]
    assert kept[1].score == 0.9


def test_ties_keep_the_first_seen_so_the_pass_is_order_stable():
    items = [_ev("g1", "same", 0.5, "corpus"), _ev("g1", "same", 0.5, "mcp")]
    kept, _ = dedupe_evidence(items)
    assert kept[0].source == "corpus"


def test_near_identical_text_is_not_collapsed():
    # Only BYTE-identical content collapses. Deciding two differently-worded
    # sentences are "the same claim" is a semantic judgement, which this
    # package's own rule keeps out of deterministic guardrails.
    items = [_ev("g1", "Redis supports replication."),
             _ev("g1", "Redis supports replication and failover.")]
    kept, counters = dedupe_evidence(items)
    assert len(kept) == 2
    assert counters == {}


def test_no_duplicates_returns_empty_counters_not_a_zero():
    # Same only-present-when-nonzero convention D-45's citation counters use,
    # so a clean run's telemetry gains no key at all.
    kept, counters = dedupe_evidence([_ev("g1", "a"), _ev("g2", "b")])
    assert len(kept) == 2
    assert counters == {}


def test_empty_input_is_a_clean_no_op():
    assert dedupe_evidence([]) == ([], {})


def test_state_evidence_is_never_mutated():
    # The whole safety argument for this pass is that it shapes the PROMPT
    # and nothing else — every telemetry figure still counts what was
    # genuinely retrieved.
    items = [_ev("g1", "same", 0.5), _ev("g1", "same", 0.9)]
    kept, _ = dedupe_evidence(items)
    assert len(items) == 2
    assert len(kept) == 1


def test_live_shape_from_run_p205_211_collapses_as_expected():
    # Reconstructed from the real prompt: one sentence, five goals, three
    # arrivals each (corpus, mcp, second lap). 15 items in, 5 out — one per
    # goal, which is what the compiler should actually have been shown.
    sentence = ("Redis supports primary-replica replication, Sentinel failover, "
                "and Redis Cluster sharding out of the box.")
    items = [_ev(f"g{n}", sentence, score, src)
             for n in range(1, 6)
             for score, src in ((0.50, "corpus"), (0.50, "mcp"), (0.48, "corpus"))]
    kept, counters = dedupe_evidence(items)
    assert len(kept) == 5
    assert {e.goal_id for e in kept} == {"g1", "g2", "g3", "g4", "g5"}
    assert counters["evidence_deduplicated"] == 10.0
