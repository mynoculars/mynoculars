"""
scripts/ingest_sample_data.py — Load the sample corpus into both stores.

Purpose:
    One-shot ingest of sample_data/corpus.jsonl into OpenSearch (BM25 leg)
    and Qdrant (dense leg) so the hybrid retriever has something to search.

Usage:
    PYTHONPATH=src python scripts/ingest_sample_data.py

Behavior:
    Skips (with a clear message) any store that is unreachable — you can
    run one leg only and the retriever degrades accordingly.
"""

import json
import pathlib
import sys

sys.path.insert(0, "src")

from research_agent.config import get_settings          # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402
from research_agent.storage.opensearch_store import OpenSearchStore  # noqa: E402
from research_agent.storage.qdrant_store import QdrantStore  # noqa: E402


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
    n_dense = dense.upsert_texts(docs)
    print(f"Qdrant:     {'embedded ' + str(n_dense) if n_dense else 'SKIPPED (unreachable)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
