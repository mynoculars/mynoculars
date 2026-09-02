"""
research_agent/ops/ingest.py — Load a corpus into both retrieval stores.

Purpose:
    One-shot ingest of a JSONL corpus into OpenSearch (BM25 leg) and
    Qdrant (dense leg) so the hybrid retriever has something to search.

Usage:
    PYTHONPATH=src python scripts/ingest_sample_data.py    # from a checkout
    research-agent-ingest --corpus /path/to/corpus.jsonl   # once installed

Behavior:
    Skips (with a clear message) any store that is unreachable — you can
    run one leg only and the retriever degrades accordingly. Safe to
    re-run: both legs are idempotent on unchanged content (see
    research_agent.storage.qdrant_store.content_id -- P2-15 promoted this
    script's OWN content_id() to that shared, canonical location, since
    memory/semantic_memory.py needed the identical scheme -- for the
    Qdrant half of that guarantee).
"""

import argparse
import json
import pathlib
import sys

# D-157: the `sys.path.insert(..., '<repo>/src')` bootstrap that stood
# here is gone, and so is the reason for it. This module lives INSIDE
# the package now, so `research_agent` is importable by definition --
# from a checkout on PYTHONPATH, from an editable install, and from a
# wheel, without any of them being a special case. scripts/ keeps a
# thin launcher of the same name for `python scripts/<name>.py`.

from research_agent.config import get_settings          # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402
from research_agent.ops._paths import repo_file          # noqa: E402
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


def main(argv=None) -> int:
    """Ingest the corpus; print per-store outcomes; return exit code.

    D-157: the corpus path used to be `__file__.parent.parent /
    "sample_data" / "corpus.jsonl"` -- a hardcoded walk out of `scripts/`
    into the checkout, which is precisely the assumption an installed
    package cannot make. It is now an ARGUMENT, defaulting to this
    repository's sample corpus when there IS a repository and required
    otherwise.

    That is not a packaging workaround; it is what this command always
    needed. Nobody deploying this ingests ten documents about Redis --
    they ingest their own corpus, and until now the only way to do that
    was to edit this repo's sample file in place.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    default_corpus = repo_file("sample_data", "corpus.jsonl")
    parser.add_argument(
        "--corpus", type=pathlib.Path, default=default_corpus,
        required=default_corpus is None,
        help="JSONL file, one document per line, each with 'content' plus "
             "any payload keys (title, topic). Defaults to this "
             "repository's sample_data/corpus.jsonl when run from a "
             "checkout; required from an installed package, which has no "
             "repository to default to.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    corpus_path = args.corpus
    if not corpus_path.is_file():
        # Naming the path beats a bare FileNotFoundError traceback, and
        # keeps "there is no repository" (handled above, by making the
        # argument required) distinct from "that file is not there".
        print(f"No corpus file at {corpus_path}", file=sys.stderr)
        return 1
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
