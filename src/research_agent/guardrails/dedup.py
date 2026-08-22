"""
guardrails/dedup.py — collapse byte-identical evidence before it reaches a prompt.

WHY THIS EXISTS
    The retrieval ladder's tiers 1-3 all resolve to the SAME ingested
    documents (D-38, and the README's own ladder diagram says so plainly):
    corpus, the reformulated corpus retry, and MCP are three ways of
    reaching one corpus. Every hit therefore reappears under a second
    `source` tag with byte-identical `content`, and once the gather loop
    runs several laps against the same goals it reappears again per lap.

    Counted directly from run p205.211's compiler prompt (45,063 chars /
    10,626 prompt tokens, for the query "Compare India and US"):

        26 x  "Redis supports primary-replica replication, Sentinel..."
        22 x  "Both Redis and Memcached support per-key TTL..."
        13 x  "Redis ships rich introspection: SLOWLOG, MEMORY USAGE..."
         6 x  "Redis is an in-memory data store supporting rich data..."

    Four sentences, 67 occurrences, in a report about two countries.

    Two distinct costs, and the second is the one that matters:

    1. Tokens. Better than half that prompt was repeated text, paid for on
       every compile AND every revision cycle.

    2. FABRICATED CONSENSUS. This is exactly the argument
       websearch/filtering.py::cap_by_domain already makes for web results
       -- five hits from one site read to the compiler as five independent
       sources agreeing, and the retrieved-item count looks identical
       either way. The corpus/MCP path has the same failure at a higher
       multiple and had no equivalent guard. A model shown one sentence
       twenty-six times is being told, by sheer repetition, that this is
       the most corroborated fact in the evidence block.

WHAT IT DELIBERATELY DOES NOT DO
    - It does NOT touch ResearchState.evidence. Only the list handed to a
      PROMPT is collapsed. recall_score, grounded_score, corpus_recall,
      grounding_ratio, evidence_by_source and every other telemetry figure
      keep counting exactly what was actually retrieved, so nothing in the
      honesty rail moves because of a prompt-shaping pass. The post-
      processing guardrails (clean_citations, enforce_hedging,
      append_web_sources) likewise still see the full state.evidence.
    - It does NOT deduplicate ACROSS goals. The same sentence genuinely
      covering g1 and g3 is two real facts about coverage, and the
      compiler needs both goal tags to cite either. Dedup is keyed on
      (goal_id, content).
    - It does NOT normalise text before hashing. Only byte-identical
      content collapses. Two sentences differing by a word are two claims,
      and deciding otherwise is a semantic judgement this package's own
      rule (deterministic checks only) says does not belong here.

WHICH COPY SURVIVES
    The highest-scoring one; ties keep the first seen, so the pass is
    order-stable and repeatable. Since the collapsed items are
    byte-identical by construction, the surviving copy differs only in its
    `source` tag and score -- and keeping the highest score is the choice
    that cannot make an item look weaker to the compiler than the
    retrieval actually justified.
"""

import logging
from typing import Dict, Iterable, List, Tuple

from research_agent.logging_setup import log_event
from research_agent.state import Evidence
from research_agent.storage.qdrant_store import content_id

logger = logging.getLogger(__name__)


def dedupe_evidence(evidence: Iterable[Evidence]) -> Tuple[List[Evidence], Dict[str, float]]:
    """Collapse byte-identical evidence within each goal.

    CALLED BY   agents/compilation.py::compiler_node and critic_node, on
                the evidence list they pass to prompts/templates.py --
                never on the state itself (see the module docstring).
    RETURNS     (kept, counters). `counters` carries
                "evidence_deduplicated" (how many items were dropped) and
                is EMPTY when nothing was dropped, so a run with no
                duplicates adds no key to telemetry at all -- the same
                only-present-when-nonzero convention D-45's citation
                counters use.

    Identity is (goal_id, content_id(content)) -- content_id being the
    uuid5-of-content function this codebase already uses for Qdrant point
    ids and memory dedup, so "the same document" means the same thing here
    as it does in storage.
    """
    kept: List[Evidence] = []
    # index maps an identity to the POSITION of the surviving item in
    # `kept`, so a later, higher-scoring duplicate can replace it in place
    # and preserve the original ordering rather than moving to the end.
    index: Dict[Tuple[str, str], int] = {}
    dropped = 0

    for item in evidence:
        identity = (item.goal_id, content_id(item.content))
        seen_at = index.get(identity)
        if seen_at is None:
            index[identity] = len(kept)
            kept.append(item)
            continue
        dropped += 1
        if item.score > kept[seen_at].score:
            kept[seen_at] = item

    if not dropped:
        return kept, {}

    log_event(logger, "guardrail.evidence_deduplicated",
              dropped=dropped, kept=len(kept))
    return kept, {"evidence_deduplicated": float(dropped)}
