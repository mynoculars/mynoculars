"""
tests/unit/test_storage_opensearch_store.py -- storage/opensearch_store.py.

Covers: the degrade-don't-die __init__ policy, ensure_index's mapping,
ingest's content-based id scheme (shared with qdrant_store's content_id,
so both retrieval legs address the same document identically), the
malformed-document positional-id fallback, and search's BM25 result shape.

IMPORTANT LIMITATION, stated plainly (same posture as
test_storage_qdrant_store.py): none of these tests run against a real
OpenSearch server. A MagicMock stands in for the opensearchpy client, so
what is verified here is the SHAPE of what this module sends and returns
-- not that a real OpenSearch server accepts it. Before trusting this in
production, run against a real instance and compare.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("opensearchpy")

from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.qdrant_store import content_id


def _mock_store(index="test_index"):
    """An OpenSearchStore constructed already-degraded (probe=False, no
    socket -- D-140), then forced "available" with a MagicMock standing in
    for the real opensearch-py client -- lets the API-CALLING code paths be
    tested without a real server, same pattern as
    test_storage_qdrant_store.py's _mock_store."""
    store = OpenSearchStore("http://127.0.0.1:1", index, probe=False)
    assert store.available is False  # sanity: constructed degraded
    store.available = True
    store._client = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Degrade-don't-die
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_unreachable_host_degrades_instead_of_raising():
    """The ONE test in this suite that still opens a real socket (D-140).

    Every other store construction here uses probe=False, which reaches the
    same degraded state without any network I/O. This test is the reason
    that is safe: it proves the real probe path still degrades rather than
    raising, so the cheap path and the real path agree.

    TEST-NET-1 (RFC 5737) rather than a localhost port: unroutable
    everywhere, by definition, whereas 127.0.0.1:1's behaviour depends on
    the OS and (on Windows) on WinNAT port reservations. Marked `slow`
    because "unroutable" means the client waits out its own timeout -- run
    it with the suite, skip it with `-m "not slow"` in a tight loop.
    """
    from tests.conftest import UNROUTABLE_URL

    store = OpenSearchStore(UNROUTABLE_URL, "test_index")
    assert store.available is False


def test_search_returns_empty_list_when_degraded():
    store = OpenSearchStore("http://127.0.0.1:1", "test_index", probe=False)
    assert store.search("query", top_k=5) == []


def test_ingest_returns_zero_when_degraded():
    store = OpenSearchStore("http://127.0.0.1:1", "test_index", probe=False)
    assert store.ingest([{"content": "x"}]) == 0


def test_ensure_index_is_a_noop_when_degraded():
    store = OpenSearchStore("http://127.0.0.1:1", "test_index", probe=False)
    store.ensure_index()  # must not raise


# ---------------------------------------------------------------------------
# ensure_index
# ---------------------------------------------------------------------------


def test_ensure_index_creates_the_mapping_when_missing():
    store = _mock_store()
    store._client.indices.exists.return_value = False
    store.ensure_index()
    store._client.indices.create.assert_called_once()
    _, kwargs = store._client.indices.create.call_args
    fields = kwargs["body"]["mappings"]["properties"]
    assert set(fields) == {"content", "title", "topic"}


def test_ensure_index_does_not_recreate_an_existing_index():
    store = _mock_store()
    store._client.indices.exists.return_value = True
    store.ensure_index()
    store._client.indices.create.assert_not_called()


# ---------------------------------------------------------------------------
# ingest -- content-based id scheme, shared with qdrant_store
# ---------------------------------------------------------------------------


def test_ingest_ids_documents_by_content_not_position():
    """The SAME uuid5 content identity scheme storage/qdrant_store.py
    uses for its point ids -- so both retrieval legs address a document
    the same way, and a reordered/deleted corpus line does not leave
    stale documents at now-unused positions."""
    store = _mock_store()
    store._client.indices.exists.return_value = True
    doc = {"content": "Redis is an in-memory data store", "title": "t"}
    store.ingest([doc])
    _, kwargs = store._client.index.call_args
    assert kwargs["id"] == content_id(doc["content"])


def test_ingest_falls_back_to_positional_id_for_a_malformed_document():
    """A doc with no 'content' key must still ingest, keyed by position,
    rather than raising -- a malformed corpus line should not abort the
    whole ingest run."""
    store = _mock_store()
    store._client.indices.exists.return_value = True
    store.ingest([{"title": "no content field"}])
    _, kwargs = store._client.index.call_args
    assert kwargs["id"] == "0"


def test_ingest_returns_the_document_count():
    store = _mock_store()
    store._client.indices.exists.return_value = True
    docs = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
    assert store.ingest(docs) == 3


def test_ingest_refreshes_the_index_for_read_your_own_writes():
    store = _mock_store()
    store._client.indices.exists.return_value = True
    store.ingest([{"content": "a"}])
    store._client.indices.refresh.assert_called_once_with(index="test_index")


def test_ingest_is_a_noop_on_an_empty_document_list():
    store = _mock_store()
    assert store.ingest([]) == 0
    store._client.index.assert_not_called()


# ---------------------------------------------------------------------------
# search -- BM25 result shape
# ---------------------------------------------------------------------------


def test_search_attaches_bm25_score_to_each_result():
    store = _mock_store()
    store._client.search.return_value = {"hits": {"hits": [
        {"_source": {"content": "Redis benchmarks"}, "_score": 4.2},
    ]}}
    results = store.search("redis", top_k=5)
    assert results == [{"content": "Redis benchmarks", "bm25_score": 4.2}]


def test_search_returns_empty_list_on_no_hits():
    store = _mock_store()
    store._client.search.return_value = {"hits": {"hits": []}}
    assert store.search("nonsense query", top_k=5) == []


def test_search_sends_a_match_query_against_the_content_field():
    store = _mock_store()
    store._client.search.return_value = {"hits": {"hits": []}}
    store.search("redis throughput", top_k=7)
    _, kwargs = store._client.search.call_args
    assert kwargs["body"]["size"] == 7
    assert kwargs["body"]["query"] == {"match": {"content": "redis throughput"}}
