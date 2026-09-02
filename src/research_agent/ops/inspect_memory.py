"""
scripts/inspect_memory.py -- "what does the agent actually remember?" (D-90).

Purpose:
    Read the semantic-memory collection directly, without running a
    research query. Two modes:

        --query "..."   what memory would RECALL for that question, in the
                        exact order memory_retrieve would see it -- the
                        same similarity x decay rerank, via the same
                        SemanticMemory object the graph uses.
        (no --query)    a summary of the whole collection: how many points,
                        how old, which volatility classes, which past
                        queries put them there.

Why this exists:
    Long-term memory is the one store in this system with NO read path
    outside a full run. Qdrant's corpus collection is inspectable through
    ingest and through any retrieval trace; Postgres run history is a
    table you can SELECT from; OpenSearch has its own API. Memory is
    written only by memory_writer (after a PASSED critique, D-24) and read
    only by memory_retrieve (the second node of every run) -- so the only
    ways to see what it holds were to run a whole research query and read
    a debug trace, or to open Qdrant by hand.

    That matters more than convenience, because memory is the one store
    that can silently steer a LATER, unrelated run: goal_manager is handed
    recalled evidence as "relevant facts from earlier research" BEFORE it
    composes this run's goals. D-42 exists because exactly that happened
    -- an army run stored 24 model-tier items, and the next, unrelated
    query inherited a military goal set. The fix stopped model/web
    recollection entering memory at all; this makes what DID enter
    legible without a run.

Read-only, always:
    Nothing here writes, deletes, or upserts. Deleting decayed points is
    scripts/gc_memory.py's job and keeps its own --yes gate; this script
    deliberately has no destructive mode at all, so it is safe to run
    against a production collection with no flags to get wrong.

Exit codes:
    0  Qdrant was reached (even if the collection turned out to be empty)
    1  Qdrant could not be reached
"""

import argparse
import sys
import time
from collections import Counter

# D-157: the `sys.path.insert(..., '<repo>/src')` bootstrap that stood
# here is gone, and so is the reason for it. This module lives INSIDE
# the package now, so `research_agent` is importable by definition --
# from a checkout on PYTHONPATH, from an editable install, and from a
# wheel, without any of them being a special case. scripts/ keeps a
# thin launcher of the same name for `python scripts/<name>.py`.

from research_agent.config import get_settings              # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402
from research_agent.memory.semantic_memory import SemanticMemory  # noqa: E402
from research_agent.state import Volatility                  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore   # noqa: E402


def summarize(store: QdrantStore, now: float = None) -> dict:
    """Aggregate the whole memory collection into countable facts.

    CALLED BY   main(), below. Split out for the same reason
    gc_memory.py::find_gc_candidates is: so the interesting logic is
    testable without argparse and print noise wrapped around it.

    READS       every point, via scroll_all() -- a full scan, not a
                top-k search, because a summary that only saw the nearest
                neighbours to some arbitrary probe would describe a
                sample rather than the collection.
    RETURNS     a plain dict of counted facts. Like telemetry_node (D-12),
                this aggregates what is there and invents nothing -- no
                judgement about whether the memory is "good", which is
                not something a counter can know.

    A point missing "volatility" defaults to SEMI_STABLE, matching
    SemanticMemory.retrieve's and gc_memory.py's identical fallback for
    the same payload gap -- one behaviour for a missing field, not a
    third separate guess.
    """
    now = now if now is not None else time.time()
    volatilities = Counter()
    source_queries = Counter()
    ages_days = []
    total = 0
    for point in store.scroll_all():
        total += 1
        volatilities[str(point.get("volatility", Volatility.SEMI_STABLE.value))] += 1
        source_queries[str(point.get("source_query", "(unknown)"))] += 1
        created_at = float(point.get("created_at", now))
        ages_days.append((now - created_at) / 86400.0)
    return {
        "points": total,
        "by_volatility": dict(volatilities),
        "distinct_source_queries": len(source_queries),
        "top_source_queries": source_queries.most_common(10),
        "oldest_days": round(max(ages_days), 2) if ages_days else 0.0,
        "newest_days": round(min(ages_days), 2) if ages_days else 0.0,
    }


def main(argv=None) -> int:
    """Parse arguments and print either a recall preview or a summary."""
    p = argparse.ArgumentParser(
        description="Inspect the agent's long-term semantic memory. Read-only.")
    p.add_argument("--query", default="",
                   help="show what memory would RECALL for this question, "
                        "using the same similarity x decay rerank a real run "
                        "would (omit for a whole-collection summary)")
    p.add_argument("--top-k", type=int, default=0,
                   help="how many recalled items to show with --query "
                        "(default: settings.memory_top_k, i.e. exactly what "
                        "a real run would receive)")
    p.add_argument("--full", action="store_true",
                   help="print each recalled item's full content instead of "
                        "a one-line preview")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    store = QdrantStore(settings.qdrant_url, settings.memory_collection)
    if not store.available:
        print(f"Qdrant:     UNREACHABLE — cannot read "
              f"'{settings.memory_collection}'")
        return 1

    if args.query:
        # The REAL SemanticMemory, not a reimplementation of its ranking:
        # whatever this prints is exactly what memory_retrieve_node would
        # hand goal_manager for the same question, decay and namespacing
        # included. A second, similar-looking ranking here could disagree
        # with the live one, which would make this tool actively
        # misleading -- the failure mode gc_memory.py avoids the same way
        # by reusing decay_factor rather than recomputing it.
        memory = SemanticMemory(
            store,
            args.top_k or settings.memory_top_k,
            settings.decay_half_life_days_semi_stable,
            settings.decay_half_life_days_volatile,
            server_side_decay=settings.memory_server_side_decay)
        recalled = memory.retrieve(args.query)
        print(f"Recall preview (collection={settings.memory_collection})")
        print(f"  query : {args.query!r}")
        print(f"  items : {len(recalled)} "
              f"(top_k={args.top_k or settings.memory_top_k})")
        if not recalled:
            print("\n  Nothing recalled. Either memory is empty, or nothing "
                  "in it is similar enough to this question.")
            return 0
        print()
        for item in recalled:
            # goal_id is shown as stored-and-namespaced ("memory::g3"),
            # because that IS what a run sees (P2-02) -- printing the bare
            # original would hide the very transformation that stops a
            # remembered fact satisfying a current goal by id collision.
            print(f"  score={item.score:.4f}  {item.volatility.value:<11} "
                  f"{item.goal_id}")
            content = item.content if args.full else item.content[:160]
            print(f"      {content}")
        return 0

    facts = summarize(store)
    print(f"Memory summary (collection={settings.memory_collection})")
    print(f"  points                  : {facts['points']}")
    if not facts["points"]:
        print("\n  Memory is empty. Nothing has passed a critique and been "
              "written yet (D-24), or the collection was reset.")
        return 0
    print(f"  by volatility           : {facts['by_volatility']}")
    print(f"  distinct source queries : {facts['distinct_source_queries']}")
    print(f"  age range (days)        : {facts['newest_days']} .. "
          f"{facts['oldest_days']}")
    print("\n  Most frequent source queries (which run wrote these):")
    for query, count in facts["top_source_queries"]:
        print(f"    {count:>4}  {query[:90]!r}")
    print("\n  Recall preview for any question:  "
          "python scripts/inspect_memory.py --query \"...\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
