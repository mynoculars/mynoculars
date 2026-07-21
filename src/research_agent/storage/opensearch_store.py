"""
storage/opensearch_store.py — Keyword (BM25) search abstraction.

Purpose:
    All OpenSearch I/O lives here: index bootstrap, document ingest, BM25
    query. Serves the sparse/keyword leg of retrieval over the sample corpus.

Responsibilities:
    - Lazy connect + graceful degradation (same policy as qdrant_store:
      available=False and no-op rather than crash — a reference build must
      run on a laptop with nothing else installed).

Design decision (why BM25 here instead of Qdrant sparse vectors):
    OpenSearch gives BM25 with zero modeling work and demonstrates the
    "each storage system gets its own module" boundary the project brief
    requires. Server-side hybrid fusion inside Qdrant (design doc D-27) is
    deferred to the full build; this core build fuses in Python
    (retrieval/hybrid.py) where a learner can read the RRF math directly.
"""

import logging
from typing import Any, Dict, List

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


class OpenSearchStore:
    """Thin, failure-tolerant wrapper over opensearch-py."""

    def __init__(self, url: str, index: str, username: str = "",
                 password: str = "", use_ssl: bool = False,
                 verify_certs: bool = False, tracer: Any = None,
                 trace_label: str = "OPENSEARCH (BM25)"):
        """Connect lazily; mark unavailable instead of raising."""
        self.index = index
        self.available = False
        self._client = None
        try:
            from opensearchpy import OpenSearch
            kwargs = {"hosts": [url], "timeout": 5}
            if use_ssl:
                # verify_certs=False + self-signed cert -> opensearch-py emits a
                # noisy InsecureRequestWarning per call; silence it deliberately.
                kwargs["use_ssl"] = True
                kwargs["verify_certs"] = verify_certs
                if not verify_certs:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            if username:
                kwargs["http_auth"] = (username, password)
            self._client = OpenSearch(**kwargs)
            self._client.info()  # liveness probe
            self.available = True
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "opensearch.unavailable", level=logging.WARNING,
                      reason=type(exc).__name__)
        self._tracer = tracer
        self._label = trace_label

    def ensure_index(self) -> None:
        """Create the corpus index with a plain text mapping if missing."""
        if not self.available:
            return
        if not self._client.indices.exists(self.index):
            self._client.indices.create(self.index, body={
                "mappings": {"properties": {
                    "content": {"type": "text"},
                    "title": {"type": "text"},
                    "topic": {"type": "keyword"},
                }}})

    def ingest(self, docs: List[Dict[str, Any]]) -> int:
        """Index documents (dicts with content/title/topic). Returns count."""
        if not self.available or not docs:
            return 0
        self.ensure_index()
        for i, doc in enumerate(docs):
            self._client.index(self.index, body=doc, id=str(i))
        self._client.indices.refresh(self.index)
        return len(docs)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 match query. Returns docs + 'bm25_score'; [] when degraded."""
        if not self.available:
            return []
        res = self._client.search(index=self.index, body={
            "size": top_k,
            "query": {"match": {"content": query}},
        })
        out: List[Dict[str, Any]] = []
        for hit in res["hits"]["hits"]:
            doc = dict(hit["_source"])
            doc["bm25_score"] = float(hit["_score"])
            out.append(doc)
        if self._tracer is not None:
            self._tracer.record_retrieval(self._label, query, out)
        return out
