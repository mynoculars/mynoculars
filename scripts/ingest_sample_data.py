"""
scripts/ingest_sample_data.py — Load the sample corpus into both stores.

Purpose:
    One-shot ingest of sample_data/corpus.jsonl into OpenSearch (BM25 leg)
    and Qdrant (dense leg) so the hybrid retriever has something to search.

Usage:
    PYTHONPATH=src python scripts/ingest_sample_data.py

Behavior:
    Skips (with a clear message) any store that is unreachable — you can
    run one leg only and the retriever degrades accordingly. Safe to
    re-run: both legs are idempotent on unchanged content (see
    research_agent.storage.qdrant_store.content_id -- P2-15 promoted this
    script's OWN content_id() to that shared, canonical location, since
    memory/semantic_memory.py needed the identical scheme -- for the
    Qdrant half of that guarantee).
"""

import json
import pathlib
import sys

sys.path.insert(0, "src")

from research_agent.config import get_settings          # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402
from research_agent.storage.opensearch_store import OpenSearchStore  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore, content_id  # noqa: E402

# P2-15: content_id used to be defined locally, right here, as
# content_id(item: dict) -> uuid5-of-item["content"]. memory/
# semantic_memory.py needed the IDENTICAL identity scheme for its own
# dedup (P2-15), so it's now a single shared function in
# storage/qdrant_store.py instead of two copies of the same logic --
# see that module for the full rationale. This script's call site below
# just wraps it: content_id expects a plain string, this script's items
# are dicts, so id_fn extracts ["content"] before calling through.


def main() -> int:
    """Ingest the corpus; print per-store outcomes; return exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)

    corpus_path = pathlib.Path(__file__).parent.parent / "sample_data" / "corpus.jsonl"
    docs = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(docs)} sample documents")

    keyword = OpenSearchStore(
        settings.opensearch_url, settings.corpus_index,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs)
    n_kw = keyword.ingest(docs)
    print(f"OpenSearch: {'indexed ' + str(n_kw) if n_kw else 'SKIPPED (unreachable)'}")

    dense = QdrantStore(settings.qdrant_url, settings.corpus_index)
    # P2-03: id_fn=content_id makes this idempotent — re-running on an
    # unchanged corpus.jsonl overwrites the same 10 points in place instead
    # of piling up a fresh set of duplicates every time.
    n_dense = dense.upsert_texts(docs, id_fn=lambda item: content_id(item["content"]))
    print(f"Qdrant:     {'embedded ' + str(n_dense) if n_dense else 'SKIPPED (unreachable)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
