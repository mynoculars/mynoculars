"""
prompts/budget.py — bound how much evidence may enter ONE prompt (D-131).

WHY THIS EXISTS
    `compile_report` inlined EVERY evidence item with no bound of any
    kind, and `critique` bounded itself with a bare tail slice
    (`evidence[-60:]`). Measured on a p205.267-check-shaped request --
    4 goals, 97 evidence items -- that compile transcript is 32,873
    characters, ~8,200 estimated tokens, of which 30,199 are the evidence
    block. The run it came from spent 237 seconds, compiled three times
    and escalated.

    Two costs, and as with guardrails/dedup.py the second is the one that
    matters:

    1. Tokens, paid again on every revision pass and (before D-129) again
       on every scoring call. p205.246's compile that actually SUCCEEDED
       ran at 4,023 prompt tokens; p205.267's was half as large again
       before the answer.

    2. A TAIL SLICE IS NOT A BUDGET. `evidence[-60:]` keeps whatever
       happened to arrive last -- which, after a third gather lap, is the
       lap that found least. D-46 exists because the critic could not see
       the evidence a claim rested on; a slice that silently drops the
       first 37 items of 97 can reintroduce exactly that, and would do it
       invisibly.

WHAT THIS DOES
    Round-robin across goals, best-first within each goal, until the
    character budget is spent. Every goal that has evidence gets its
    strongest item before any goal gets a second one, so a goal with one
    hit is never crowded out by a goal with forty.

WHAT IT DELIBERATELY DOES NOT DO
    - It does NOT touch ResearchState.evidence. Only the list handed to a
      PROMPT is trimmed, exactly as dedupe_evidence already works, so
      recall, grounded_score, corpus_recall, evidence_by_source and every
      other honesty figure keeps counting what was actually retrieved,
      and the post-compile guardrails still see the full evidence.
    - It does NOT rank by relevance. Scores come from retrieval; this
      pass reorders nothing and judges nothing. A dropped item is dropped
      for arithmetic, not for merit.
    - It does NOT summarise, truncate or rewrite an item's content.
      Half a sentence of evidence is worse than none: a compiler cannot
      cite what it cannot read, and a mid-word cut is what FIX-1 already
      proved a model scores against.

WHY CHARACTERS AND NOT TOKENS
    llm/client.py::estimate_prompt_tokens is ~4 chars/token and says
    itself that it is approximate. A budget expressed in the unit that is
    actually measured, rather than in one derived from it by a constant,
    is the honest version -- and the two differ only by that constant.
"""

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

from research_agent.logging_setup import log_event
from research_agent.state import Evidence, Goal

logger = logging.getLogger(__name__)

# What one rendered evidence LINE costs beyond its content: the
# "- [g1 | corpus | score=0.60] " tag prompts/templates.py writes in front
# of it, plus the newline. Measured against compile_report's own format
# rather than guessed, and deliberately a small over-estimate: budgeting
# on content alone would let 30 items smuggle in ~1,400 unaccounted
# characters.
_PER_ITEM_OVERHEAD = 48

# Provenance order, used ONLY to break a tie between two items with the
# SAME score. D-38's invariant is that a real document beats weaker
# provenance; in practice scores already separate these (a two-leg corpus
# hit reaches ~1.0, a web result is banded 0.60-0.75, model recollection
# sits at 0.60), so this decides genuinely equal cases and nothing else.
_SOURCE_RANK = {"corpus": 0, "mcp": 0, "web": 1, "model": 2, "memory": 3}


def _cost(item: Evidence) -> int:
    """What this item costs the prompt, tag included."""
    return len(item.content) + _PER_ITEM_OVERHEAD


def budget_evidence(evidence: Iterable[Evidence], goals: Sequence[Goal],
                    max_chars: int) -> Tuple[List[Evidence], Dict[str, float]]:
    """Return (kept, counters) -- the evidence that fits, in INPUT order.

    CALLED BY   agents/compilation.py::compiler_node and critic_node,
                immediately after dedupe_evidence and on the same
                prompt-only copy.
    RETURNS     `kept` in the ORIGINAL input order, so the prompt's shape
                is unchanged and only its membership differs. `counters`
                carries "evidence_prompt_dropped" and is EMPTY when
                nothing was dropped -- the only-present-when-nonzero
                convention dedup and D-45's citation counters both use.

    max_chars <= 0 DISABLES the budget and returns the input unchanged.
    That is the documented way to reproduce pre-D-131 behaviour
    deliberately, the same escape hatch MIN_SIMILARITY=0.0 and
    WEB_SEARCH_MAX_PER_DOMAIN=0 already provide -- and
    config.warn_on_unbounded_prompt_budget says so at startup rather than
    letting it happen silently.

    ORDERING, in full, because every part of it is load-bearing:
      - goals are walked in `goals` order (g1, g2, ...), then any goal_id
        present in the evidence but not in `goals` -- memory items carry
        "memory::gN" by P2-02 and are real evidence the compiler still
        sees, just never a CURRENT goal;
      - within a goal: score descending, then provenance (_SOURCE_RANK),
        then first-seen. Fully deterministic, which matters because
        compiler_node runs again on every revision and two passes over
        the same evidence must build the same prompt;
      - allocation is round-robin: every goal's best item, then every
        goal's second, and so on. An item that does not fit is skipped
        and the walk CONTINUES -- a later, smaller item for another goal
        should not be denied by one oversized item ahead of it.
    """
    items = list(evidence)
    if max_chars <= 0 or not items:
        return items, {}
    total = sum(_cost(e) for e in items)
    if total <= max_chars:
        return items, {}

    order = {e_id: i for i, e_id in enumerate(
        dict.fromkeys([g.goal_id for g in goals]
                      + [e.goal_id for e in items]))}
    by_goal: Dict[str, List[Tuple[int, Evidence]]] = {}
    for index, item in enumerate(items):
        by_goal.setdefault(item.goal_id, []).append((index, item))
    for goal_id in by_goal:
        by_goal[goal_id].sort(
            key=lambda pair: (-pair[1].score,
                              _SOURCE_RANK.get(pair[1].source, 9),
                              pair[0]))

    kept_indices: set = set()
    spent = 0
    # Deepest stack across all goals -- the round-robin runs that many
    # laps, taking at most one item per goal per lap.
    for lap in range(max(len(v) for v in by_goal.values())):
        for goal_id in sorted(by_goal, key=lambda g: order.get(g, 10 ** 6)):
            stack = by_goal[goal_id]
            if lap >= len(stack):
                continue
            index, item = stack[lap]
            if spent + _cost(item) > max_chars:
                continue
            kept_indices.add(index)
            spent += _cost(item)

    kept = [item for i, item in enumerate(items) if i in kept_indices]
    dropped = len(items) - len(kept)
    if not dropped:
        return kept, {}

    log_event(logger, "guardrail.evidence_budgeted", dropped=dropped,
              kept=len(kept), chars=spent, limit=max_chars,
              goals_represented=len({e.goal_id for e in kept}))
    return kept, {"evidence_prompt_dropped": float(dropped)}
