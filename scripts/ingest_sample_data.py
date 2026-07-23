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
    content_id() below for the Qdrant half of that guarantee).
"""

import json
import pathlib
import sys
import uuid

sys.path.insert(0, "src")

from research_agent.config import get_settings          # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402
from research_agent.storage.opensearch_store import OpenSearchStore  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore  # noqa: E402


def content_id(item: dict) -> str:
    """Deterministic Qdrant point id derived from an item's content.

    CALLED BY   main(), below, passed as upsert_texts's id_fn (P2-03).
    WHY THIS EXISTS: previously this script called
    dense.upsert_texts(docs) with no id_fn, so QdrantStore.upsert_texts
    fell back to its default str(uuid.uuid4()) — a FRESH random id every
    single call. Re-running this script on an unchanged corpus.jsonl
    therefore duplicated all 10 points on the dense leg every time,
    while OpenSearch (deterministic _id=str(i) in OpenSearchStore.ingest)
    stayed correctly idempotent — an asymmetry that meant simply
    re-running this script for a fresh checkout silently bloated the
    dense index with duplicate copies of the same 10 documents, degrading
    fusion quality more with every re-run.

    uuid.uuid5(NAMESPACE, content) is deterministic: the SAME content
    string always produces the SAME UUID, and it's a valid Qdrant point id
    (Qdrant accepts unsigned ints or UUID strings, never an arbitrary
    string) — unlike a raw hash digest, which Qdrant would reject.
    Content is the only input, deliberately: this corpus has no other
    stable natural key, and a changed content string is the one case
    where getting a genuinely new id (rather than overwriting) is
    actually correct — it's semantically a different document version.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, item["content"]))


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
    n_dense = dense.upsert_texts(docs, id_fn=content_id)
    print(f"Qdrant:     {'embedded ' + str(n_dense) if n_dense else 'SKIPPED (unreachable)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
