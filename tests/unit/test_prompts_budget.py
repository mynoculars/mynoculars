"""
tests/unit/test_prompts_budget.py — prompts/budget.py::budget_evidence
(D-131, P6-2).

Covers the three properties the pass exists for -- it BOUNDS the prompt,
it is FAIR across goals (a goal with one hit is never crowded out by a
goal with forty), and it is DETERMINISTIC (compiler_node runs again on
every revision and two passes over the same evidence must build the same
prompt) -- plus the disabled path and the byte-identical path.

Does NOT cover where the budget is APPLIED; see
test_agents_compilation.py for compiler_node/critic_node wiring it in.
"""

import logging

from research_agent.prompts.budget import budget_evidence
from research_agent.state import Evidence, Goal


def _goals(n):
    return [Goal(goal_id=f"g{i}", description=f"goal {i}") for i in range(1, n + 1)]


def _ev(goal_id, key, chars=200, score=0.7, source="web"):
    return Evidence(task_key=key, goal_id=goal_id, source=source,
                    content="x" * chars, score=score)


def test_evidence_that_fits_is_returned_untouched():
    """The byte-identical rule every guardrail in this codebase follows:
    with nothing to drop, the input comes back as it went in and no
    counter is emitted at all."""
    items = [_ev("g1", "a"), _ev("g2", "b")]
    kept, counters = budget_evidence(items, _goals(2), 12000)

    assert kept == items
    assert counters == {}


def test_an_oversized_block_is_cut_to_the_budget():
    items = [_ev("g1", f"t{i}", chars=500) for i in range(40)]
    kept, counters = budget_evidence(items, _goals(1), 3000)

    spent = sum(len(e.content) + 48 for e in kept)
    assert spent <= 3000
    assert counters["evidence_prompt_dropped"] == float(len(items) - len(kept))


def test_a_goal_with_one_hit_is_never_crowded_out():
    """THE property a tail slice does not have. g2 has a single item and
    it arrives LAST -- `evidence[-60:]` would keep it by luck and
    `evidence[:60]` would drop it; round-robin keeps it by construction."""
    items = [_ev("g1", f"t{i}", chars=400) for i in range(60)]
    items.append(_ev("g2", "lonely", chars=400))

    kept, _ = budget_evidence(items, _goals(2), 2000)

    assert any(e.task_key == "lonely" for e in kept)
    assert {e.goal_id for e in kept} == {"g1", "g2"}


def test_every_goal_gets_its_best_item_before_any_goal_gets_a_second():
    items = []
    for goal in range(1, 5):
        for i in range(10):
            items.append(_ev(f"g{goal}", f"g{goal}-t{i}", chars=200))

    kept, _ = budget_evidence(items, _goals(4), 4 * 248)

    assert len({e.goal_id for e in kept}) == 4
    assert len(kept) == 4


def test_within_a_goal_the_strongest_item_wins():
    items = [_ev("g1", "weak", score=0.51), _ev("g1", "strong", score=0.95)]
    kept, _ = budget_evidence(items, _goals(1), 260)

    assert [e.task_key for e in kept] == ["strong"]


def test_a_document_breaks_a_tie_against_a_snippet_at_the_same_score():
    """D-38's invariant, applied only where scores are genuinely equal --
    in practice they separate these already (a two-leg corpus hit reaches
    ~1.0, web is banded 0.60-0.75)."""
    items = [_ev("g1", "snippet", score=0.7, source="web"),
             _ev("g1", "document", score=0.7, source="corpus")]
    kept, _ = budget_evidence(items, _goals(1), 260)

    assert [e.task_key for e in kept] == ["document"]


def test_survivors_keep_their_original_order():
    """The prompt's SHAPE must not change, only its membership -- so the
    diff between a budgeted and an unbudgeted prompt is items missing,
    never items moved."""
    items = [_ev("g1", "first", score=0.6), _ev("g2", "second", score=0.9),
             _ev("g1", "third", score=0.95)]
    kept, _ = budget_evidence(items, _goals(2), 2 * 248)

    assert [e.task_key for e in kept] == ["second", "third"]


def test_memory_namespaced_goal_ids_still_get_a_share():
    """P2-02 tags recalled evidence "memory::gN", which is deliberately
    never a CURRENT goal id -- but it is still real evidence the compiler
    is shown, so it must not be dropped wholesale for being absent from
    state.goals."""
    items = [_ev("g1", f"t{i}", chars=300) for i in range(10)]
    items.append(_ev("memory::g3", "recalled", chars=300, source="memory"))

    kept, _ = budget_evidence(items, _goals(1), 1200)

    assert any(e.task_key == "recalled" for e in kept)


def test_the_pass_is_deterministic():
    """compiler_node runs again on every revision; two passes over the
    same evidence must produce the same prompt, or D-88's report-scoped
    counters compare two different things."""
    items = [_ev(f"g{(i % 3) + 1}", f"t{i}", chars=250, score=0.5 + (i % 5) / 100)
             for i in range(30)]

    first, c1 = budget_evidence(items, _goals(3), 3000)
    second, c2 = budget_evidence(items, _goals(3), 3000)

    assert [e.task_key for e in first] == [e.task_key for e in second]
    assert c1 == c2


def test_a_zero_budget_disables_the_pass_entirely():
    """The documented escape hatch, matching MIN_SIMILARITY=0.0 and
    WEB_SEARCH_MAX_PER_DOMAIN=0. config.warn_on_unbounded_prompt_budget
    is what stops it being silent."""
    items = [_ev("g1", f"t{i}", chars=5000) for i in range(20)]
    kept, counters = budget_evidence(items, _goals(1), 0)

    assert kept == items
    assert counters == {}


def test_an_oversized_single_item_does_not_starve_the_rest():
    """An item that cannot fit is SKIPPED and the walk continues -- a
    later, smaller item for another goal must not be denied by one giant
    item ahead of it in the queue."""
    items = [_ev("g1", "giant", chars=9000, score=0.99),
             _ev("g2", "small", chars=200, score=0.6)]

    kept, _ = budget_evidence(items, _goals(2), 1000)

    assert [e.task_key for e in kept] == ["small"]


def test_the_trim_is_logged_with_what_it_kept(caplog):
    items = [_ev("g1", f"t{i}", chars=400) for i in range(30)]

    with caplog.at_level(logging.INFO):
        budget_evidence(items, _goals(1), 2000)

    rec = [r for r in caplog.records
           if "guardrail.evidence_budgeted" in r.message]
    assert rec
    f = rec[0].event_fields
    assert f["limit"] == 2000
    assert f["kept"] + f["dropped"] == 30
    assert f["chars"] <= 2000
    assert f["goals_represented"] == 1


def test_an_empty_evidence_list_is_not_a_special_case():
    assert budget_evidence([], _goals(2), 12000) == ([], {})


# ---------------------------------------------------------------------------
# D-138 — budget_notes
# ---------------------------------------------------------------------------

def test_a_short_note_list_is_returned_unchanged():
    """The common path: one critique's verdict, entire, with no counter."""
    from research_agent.prompts.budget import budget_notes

    notes = ["Goal g1: unsupported figure. FAIL.",
             "Goal g2: the comparison is not evidenced. FAIL."]

    assert budget_notes(notes) == (notes, {})


def test_the_newest_notes_survive():
    """Notes accumulate across revisions, so the ones appended LAST are
    the ones the CURRENT draft failed on. The earlier ones describe a
    draft that has already been rewritten."""
    from research_agent.prompts.budget import budget_notes

    notes = [f"note {i}" for i in range(30)]

    kept, counters = budget_notes(notes)

    assert kept[-1] == "note 29"
    assert "note 0" not in kept
    assert counters["critique_notes_dropped"] == 30 - len(kept)


def test_the_kept_notes_stay_in_their_original_order():
    """The prompt still reads oldest to newest; only the head is lost."""
    from research_agent.prompts.budget import budget_notes

    kept, _ = budget_notes([f"note {i}" for i in range(30)])

    assert kept == sorted(kept, key=lambda n: int(n.split()[1]))


def test_the_character_bound_stops_a_few_very_long_notes():
    """Live, one note ran to 874 characters. A count-only bound would let
    twelve of those into a prompt the evidence budget then has to share."""
    from research_agent.prompts.budget import budget_notes

    notes = ["x" * 900 for _ in range(12)]

    kept, counters = budget_notes(notes)

    assert sum(len(n) for n in kept) <= 3500
    assert counters["critique_notes_dropped"] > 0


def test_the_largest_verdict_yet_observed_passes_through_whole():
    """p205.277-check's first critique: 10 notes, 2,722 characters. The
    bounds are set above it deliberately, so a real critique is never
    truncated -- only superseded ones are dropped."""
    from research_agent.prompts.budget import budget_notes

    notes = ["n" * 272 for _ in range(10)]

    assert budget_notes(notes) == (notes, {})


def test_empty_and_blank_notes_are_handled():
    from research_agent.prompts.budget import budget_notes

    assert budget_notes([]) == ([], {})
    assert budget_notes(["", None, "real note"]) == (["real note"], {})


# ---------------------------------------------------------------------------
# D-142 — memory shares ONE round-robin bucket
#
# P2-02 namespaces recalled evidence as "memory::gN", which must stay. But
# the round-robin allocates a slot per BUCKET per lap, so those namespaced
# ids were read as several independent goals, each guaranteed a slot in lap
# 1 ahead of the second item for any real goal. _SOURCE_RANK ranks memory
# last precisely to stop this, and never got to apply because it only
# breaks ties WITHIN a bucket.
# ---------------------------------------------------------------------------


def _bev(goal_id, source, score, size=100):
    """D-142's own fixture. Deliberately NOT the module's `_ev` above --
    that one defaults source="web" and takes a task_key, and these tests
    are specifically about provenance, so they name it explicitly."""
    from research_agent.state import Evidence, Volatility

    return Evidence(task_key=f"{goal_id}-{source}-{score}", goal_id=goal_id,
                    source=source, content="x" * size, score=score,
                    volatility=Volatility.SEMI_STABLE)


def test_memory_pseudo_goals_collapse_to_one_bucket():
    from research_agent.prompts.budget import MEMORY_BUCKET, _bucket

    assert _bucket("memory::g1") == MEMORY_BUCKET
    assert _bucket("memory::g4") == MEMORY_BUCKET
    assert _bucket("g1") == "g1"


def test_recall_no_longer_claims_a_slot_per_remembered_goal():
    """The p205.280-check shape, reconstructed: four memory items filed
    under four DIFFERENT remembered goals, against one real goal, in a
    budget that fits four items.

    Before D-142 those four namespaced ids were four buckets, so lap 1
    handed a slot to each and the real goal got exactly one -- three of the
    four slots went to recall. With one shared bucket the two compete
    fairly, so recall can take at most half.
    """
    from research_agent.prompts.budget import _cost, budget_evidence
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="China vs India force size")]
    evidence = (
        [_bev(f"memory::g{i}", "memory", 0.46) for i in range(1, 5)]
        + [_bev("g1", "web", 0.75) for _ in range(4)]
    )

    kept, counters = budget_evidence(evidence, goals,
                                     max_chars=4 * _cost(evidence[0]))

    by_source = {}
    for e in kept:
        by_source[e.source] = by_source.get(e.source, 0) + 1
    assert len(kept) == 4
    assert by_source.get("memory", 0) == 2, by_source
    assert by_source.get("web", 0) == 2, by_source
    assert counters["evidence_prompt_dropped"] == 4


def test_more_real_goals_shrink_recalls_share_not_grow_it():
    """The property that actually matters. Recall holds ONE bucket, so its
    share falls as the run has more real goals to cover -- the opposite of
    the pre-D-142 behaviour, where more remembered goals meant more slots
    for recall regardless of what the run was researching."""
    from research_agent.prompts.budget import _cost, budget_evidence
    from research_agent.state import Goal

    goals = [Goal(goal_id=f"g{i}", description=f"goal {i}") for i in range(1, 5)]
    evidence = (
        [_bev(f"memory::g{i}", "memory", 0.46) for i in range(1, 5)]
        + [_bev(f"g{i}", "web", 0.75) for i in range(1, 5)]
    )

    kept, _ = budget_evidence(evidence, goals,
                              max_chars=5 * _cost(evidence[0]))

    memory_kept = [e for e in kept if e.source == "memory"]
    assert len(memory_kept) == 1, [e.goal_id for e in kept]


def test_within_the_shared_bucket_the_best_recall_wins():
    """_SOURCE_RANK already ranked memory last; it only ever broke ties
    INSIDE a bucket, which is why one bucket per pseudo-goal meant it never
    applied. With one bucket, score ordering inside it finally does."""
    from research_agent.prompts.budget import _cost, budget_evidence
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="d")]
    weak = _bev("memory::g1", "memory", 0.41)
    strong = _bev("memory::g9", "memory", 0.79)
    evidence = [weak, strong, _bev("g1", "web", 0.75)]

    kept, _ = budget_evidence(evidence, goals, max_chars=2 * _cost(weak))

    assert strong in kept and weak not in kept


def test_real_goals_still_get_one_slot_each_before_anyone_gets_two():
    """The round-robin's actual guarantee, unchanged by D-142."""
    from research_agent.prompts.budget import _cost, budget_evidence
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="a"), Goal(goal_id="g2", description="b")]
    evidence = [_bev("g1", "web", 0.9), _bev("g1", "web", 0.8),
                _bev("g2", "web", 0.7)]
    two_items = 2 * _cost(evidence[0])

    kept, _ = budget_evidence(evidence, goals, max_chars=two_items)

    assert {e.goal_id for e in kept} == {"g1", "g2"}

