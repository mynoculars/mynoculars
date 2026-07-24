"""
scripts/gc_memory.py — Remove semantic-memory points that have decayed
past the point of being useful.

Purpose:
    P2-10's server-side decay and the pre-existing Python decay_factor()
    path (memory/semantic_memory.py) both DEPRIORITIZE stale points at
    RETRIEVAL time -- neither one ever deletes anything. Left alone, the
    Qdrant memory collection grows without bound for the life of a
    deployment: every passed run's fresh evidence adds points (P2-15's
    content-identity dedup in store_run collapses EXACT repeats, but a
    genuinely different fact is still a genuinely new point, forever).
    This script is the separate, explicit cleanup step: it computes each
    point's CURRENT decayed relevance using the exact same decay_factor()
    math the retrieval path already trusts (never a second implementation
    of the same formula -- see memory/semantic_memory.py), and removes
    points that have decayed below a threshold.

Usage:
    python scripts/gc_memory.py --dry-run
    python scripts/gc_memory.py --yes
    python scripts/gc_memory.py --yes --threshold 0.02

Scope:
    Only ever touches settings.memory_collection (Qdrant). Never touches
    the corpus collection, OpenSearch, or Postgres -- those are
    reset_stores.py's job, a different (much blunter) tool for a
    different situation (wiping everything to re-ingest from scratch, not
    ongoing steady-state pruning of an otherwise-healthy collection).

Threshold:
    A point's decay_factor() (age + volatility aware, same half-lives as
    everywhere else -- settings.decay_half_life_days_semi_stable /
    _volatile) below --threshold (default 0.05, i.e. under 5% of its
    original relevance) is a GC candidate. A STABLE-volatility point's
    decay_factor() is always exactly 1.0 (see that function's own
    docstring), so stable facts are NEVER collected by this script,
    regardless of age -- this is deliberate, not an oversight: D-24's own
    reasoning for stable facts is that they shouldn't fade at all.

Safety:
    - Destructive. Requires --yes, or it only prints the plan (matching
      reset_stores.py's exact --dry-run/--yes convention).
    - An unreachable Qdrant is reported and treated as a non-zero exit,
      never a crash.

Exit codes:
    0  Qdrant was reached (and points were GC'd, unless --dry-run)
    1  Qdrant could not be reached
"""

import argparse
import sys
import time

sys.path.insert(0, "src")

from research_agent.config import get_settings              # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402
from research_agent.memory.semantic_memory import decay_factor  # noqa: E402
from research_agent.state import Volatility                  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore   # noqa: E402


def find_gc_candidates(store: QdrantStore, half_life_semi: float,
                       half_life_volatile: float, threshold: float,
                       now: float = None) -> list:
    """Return [(point_id, decay_value, content_preview), ...] for every
    point whose CURRENT decay_factor() is below `threshold`.

    CALLED BY   main(), below. Split out as its own function so it's
                testable without argparse/print noise around it.
    READS       every point in store's collection, via scroll_all()
                (P2-15) -- this is a full-collection scan, not a
                top-k search, deliberately: GC needs to see everything to
                decide what to remove, the same reasoning scroll_all's own
                docstring gives.
    CALLS       decay_factor() -- the SAME function
                memory/semantic_memory.py::SemanticMemory.retrieve already
                uses to rerank live queries. Reusing it here (rather than
                recomputing similar math a second time) is deliberate:
                whatever this script decides to delete is EXACTLY what the
                retrieval path already considers all-but-invisible, not a
                separately-tuned threshold that could disagree with it.
    RETURNS     a plain list of tuples, oldest/most-decayed-first isn't
                guaranteed -- callers that want a specific order should
                sort the result themselves.

    A point missing "volatility" in its payload defaults to SEMI_STABLE,
    matching SemanticMemory.retrieve's own fallback for the same case
    (memory/semantic_memory.py) -- consistent behavior for a payload gap,
    not a second, different guess about what an untagged item probably is.
    """
    now = now if now is not None else time.time()
    candidates = []
    for point in store.scroll_all():
        volatility = Volatility(point.get("volatility", Volatility.SEMI_STABLE.value))
        created_at = float(point.get("created_at", now))
        age_days = (now - created_at) / 86400.0
        value = decay_factor(age_days, volatility, half_life_semi, half_life_volatile)
        if value < threshold:
            preview = str(point.get("content", ""))[:80]
            candidates.append((point["id"], value, preview))
    return candidates


def main(argv=None) -> int:
    """Parse arguments, print the plan, and GC the memory collection."""
    p = argparse.ArgumentParser(
        description="Remove semantic-memory points decayed below a relevance threshold.")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="decay_factor() cutoff below which a point is removed (default 0.05)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without changing anything")
    p.add_argument("--yes", action="store_true",
                   help="required to actually perform the deletions")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    store = QdrantStore(settings.qdrant_url, settings.memory_collection)
    if not store.available:
        print(f"Qdrant:     UNREACHABLE — cannot GC '{settings.memory_collection}'")
        return 1

    candidates = find_gc_candidates(
        store, settings.decay_half_life_days_semi_stable,
        settings.decay_half_life_days_volatile, args.threshold)

    print(f"GC plan (collection={settings.memory_collection}, threshold={args.threshold})")
    print(f"  {len(candidates)} point(s) decayed below threshold")
    for point_id, value, preview in candidates[:10]:
        print(f"    {value:.4f}  {point_id}  {preview!r}")
    if len(candidates) > 10:
        print(f"    ... and {len(candidates) - 10} more")
    print()

    if not args.yes and not args.dry_run:
        print("Refusing to delete anything without --yes. "
              "Re-run with --dry-run to preview, or --yes to proceed.")
        return 0

    if args.dry_run:
        print("Dry run complete. Nothing was changed.")
        return 0

    if not candidates:
        print("Nothing to delete.")
        return 0

    deleted = store.delete_points([c[0] for c in candidates])
    print(f"Deleted {deleted} point(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
