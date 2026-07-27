"""
scripts/reset_stores.py — Return every store to a pristine, re-ingestable state.

Purpose:
    The core build has no idempotent ingest: Qdrant points are keyed by a fresh
    uuid4 on every write (storage/qdrant_store.py::upsert_texts), so re-running
    scripts/ingest_sample_data.py DUPLICATES the corpus instead of overwriting
    it, and every passed run appends another copy of its evidence to semantic
    memory. This script is the supported way to get back to a known-empty state
    before re-ingesting.

Usage:
    python reset_stores.py --dry-run
    python reset_stores.py --yes
    python reset_stores.py --yes --keep-memory
    python reset_stores.py --yes --qdrant --opensearch

    This file lives at the REPO ROOT, not under scripts/ (only
    ingest_sample_data.py is there) -- a stale "scripts/reset_stores.py"
    path lived in this docstring and in reset_stores.bat until a live run
    surfaced it. No PYTHONPATH needed either: a repo-relative "src" is put
    on sys.path (resolved from __file__, not the CWD)
    two lines below does that already, on every OS.

Scope (endpoints and names always come from Settings / .env — never hardcoded):
    Qdrant      drop collection CORPUS_INDEX
                drop collection MEMORY_COLLECTION      (unless --keep-memory)
    OpenSearch  delete index    CORPUS_INDEX
    Postgres    drop table      agent_runs             (app-owned run history)
                drop tables     checkpoint_writes, checkpoint_blobs,
                                checkpoints, checkpoint_migrations
                                (LangGraph PostgresSaver-owned; recreated by
                                 saver.setup() on the next run)

Safety:
    - Destructive. Requires --yes, or it only prints the plan.
    - --dry-run prints the plan and exits 0 without touching anything.
    - Each store is independent: an unreachable store is reported and skipped,
      never fatal, matching the graceful-degradation policy elsewhere.

Exit codes:
    0  every requested store was reached (and reset, unless --dry-run)
    1  at least one requested store could not be reached — applies to --dry-run
       too, which is deliberate: an unreachable store is exactly what you want
       a preview to tell you.
"""

import argparse
import pathlib
import sys

# Resolve "src" RELATIVE TO THIS FILE, never relative to the current
# working directory. `sys.path.insert(0, "src")` only resolved when the
# process happened to be launched from the repo root -- not guaranteed
# for a script launched as an MCP_SERVER_COMMAND subprocess, from a
# Windows shortcut or scheduled task, or from any other directory --
# and failed with an opaque ModuleNotFoundError when it did not.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from research_agent.config import get_settings              # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402

# LangGraph checkpointer tables, child-first so FK/dependency order is safe.
# Source of truth: langgraph.checkpoint.postgres.base (CREATE TABLE statements).
_CHECKPOINT_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)

# Owned by this repo's application code (storage/postgres.py::record_run).
_APP_TABLES = ("agent_runs",)


def reset_qdrant(url: str, collections: list, dry_run: bool) -> bool:
    """Delete the named Qdrant collections. Returns False if Qdrant is down."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=url, timeout=5)
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001
        print(f"Qdrant:     UNREACHABLE ({type(exc).__name__}) — skipped")
        return False

    for name in collections:
        if name not in existing:
            print(f"Qdrant:     collection '{name}' absent — nothing to do")
            continue
        if dry_run:
            print(f"Qdrant:     WOULD DELETE collection '{name}'")
            continue
        client.delete_collection(name)
        print(f"Qdrant:     deleted collection '{name}'")
    return True


def reset_opensearch(url: str, index: str, username: str, password: str,
                     use_ssl: bool, verify_certs: bool, dry_run: bool) -> bool:
    """Delete the corpus index. Returns False if OpenSearch is down."""
    try:
        from opensearchpy import OpenSearch
        kwargs = {"hosts": [url], "timeout": 5}
        if use_ssl:
            kwargs["use_ssl"] = True
            kwargs["verify_certs"] = verify_certs
            if not verify_certs:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        if username:
            kwargs["http_auth"] = (username, password)
        client = OpenSearch(**kwargs)
        client.info()
    except Exception as exc:  # noqa: BLE001
        print(f"OpenSearch: UNREACHABLE ({type(exc).__name__}) — skipped")
        return False

    if not client.indices.exists(index=index):
        print(f"OpenSearch: index '{index}' absent — nothing to do")
        return True
    if dry_run:
        print(f"OpenSearch: WOULD DELETE index '{index}'")
        return True
    client.indices.delete(index=index)
    print(f"OpenSearch: deleted index '{index}'")
    return True


def reset_postgres(dsn: str, dry_run: bool) -> bool:
    """Drop app run-history and LangGraph checkpointer tables.

    The checkpointer tables are recreated by PostgresSaver.setup() on the next
    run (storage/postgres.py::get_checkpointer); agent_runs is recreated by
    record_run's CREATE TABLE IF NOT EXISTS. Nothing here needs a migration.
    """
    tables = _APP_TABLES + _CHECKPOINT_TABLES
    try:
        import psycopg
        with psycopg.connect(dsn, autocommit=True) as conn:
            if dry_run:
                for t in tables:
                    print(f"Postgres:   WOULD DROP TABLE IF EXISTS {t}")
                return True
            for t in tables:
                conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
                print(f"Postgres:   dropped {t}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Postgres:   UNREACHABLE ({type(exc).__name__}) — skipped")
        return False


def main(argv=None) -> int:
    """Parse arguments, print the plan, and reset the selected stores."""
    p = argparse.ArgumentParser(
        description="Reset Qdrant / OpenSearch / Postgres to a pristine state.")
    p.add_argument("--qdrant", action="store_true", help="reset Qdrant only")
    p.add_argument("--opensearch", action="store_true", help="reset OpenSearch only")
    p.add_argument("--postgres", action="store_true", help="reset Postgres only")
    p.add_argument("--keep-memory", action="store_true",
                   help="do NOT drop the Qdrant semantic-memory collection")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without changing anything")
    p.add_argument("--yes", action="store_true",
                   help="required to actually perform the deletions")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    # No explicit store flags -> all three.
    do_all = not (args.qdrant or args.opensearch or args.postgres)
    do_qdrant = do_all or args.qdrant
    do_opensearch = do_all or args.opensearch
    do_postgres = do_all or args.postgres

    collections = [settings.corpus_index]
    if not args.keep_memory:
        collections.append(settings.memory_collection)

    print("Reset plan")
    print(f"  Qdrant     {settings.qdrant_url}      -> {collections if do_qdrant else 'skipped'}")
    print(f"  OpenSearch {settings.opensearch_url}  -> "
          f"{[settings.corpus_index] if do_opensearch else 'skipped'}")
    print(f"  Postgres   {settings.postgres_dsn.split('@')[-1]}  -> "
          f"{list(_APP_TABLES + _CHECKPOINT_TABLES) if do_postgres else 'skipped'}")
    print()

    if not args.yes and not args.dry_run:
        print("Refusing to delete anything without --yes. "
              "Re-run with --dry-run to preview, or --yes to proceed.")
        return 0

    ok = True
    if do_qdrant:
        ok &= reset_qdrant(settings.qdrant_url, collections, args.dry_run)
    if do_opensearch:
        ok &= reset_opensearch(settings.opensearch_url, settings.corpus_index,
                               settings.opensearch_username,
                               settings.opensearch_password,
                               settings.opensearch_use_ssl,
                               settings.opensearch_verify_certs,
                               args.dry_run)
    if do_postgres:
        ok &= reset_postgres(settings.postgres_dsn, args.dry_run)

    print()
    if args.dry_run:
        print("Dry run complete. Nothing was changed.")
    else:
        print("Reset complete. Next: PYTHONPATH=src python scripts/ingest_sample_data.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())