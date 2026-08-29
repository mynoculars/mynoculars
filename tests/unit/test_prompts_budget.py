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
