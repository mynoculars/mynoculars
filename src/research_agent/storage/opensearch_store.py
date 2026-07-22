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

Python mechanics used in this file, if any of this is new to you:
    **kwargs  (kwargs = {...}; ...; OpenSearch(**kwargs))
        Building a dict named `kwargs` and then calling a function with
        "**kwargs" UNPACKS that dict into individual keyword arguments —
        OpenSearch(**{"hosts": [...], "timeout": 5}) is exactly the same as
        writing OpenSearch(hosts=[...], timeout=5) directly. This pattern is
        used below because the exact set of arguments to pass depends on
        runtime configuration (SSL on or off, a username set or not) — the
        dict is built up conditionally first, THEN unpacked into the call
        in one place, rather than writing many different OpenSearch(...)
        call variations for every combination of settings.
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
        """Connect lazily; mark unavailable instead of raising.

        CALLED BY   cli.py::build_app_and_settings, once, for the corpus
                    index (there is only one OpenSearch index in this
                    codebase — memory never touches OpenSearch, only
                    Qdrant).
        WRITES      self.available — see qdrant_store.py's __init__ for the
                    identical degrade-don't-die policy.
        """
        self.index = index
        self.available = False
        self._client = None
        try:
            from opensearchpy import OpenSearch
            # Build up the connection arguments as a plain dict FIRST,
            # conditionally adding keys, then unpack it all at once into
            # the actual OpenSearch(...) constructor call below — see the
            # module docstring's **kwargs explanation.
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
        """Create the corpus index with a plain text mapping if missing.

        CALLED BY   ingest() below — automatically, so a fresh OpenSearch
                    instance gets its index created the first time
                    documents are loaded, with no separate setup step
                    needed elsewhere.
        """
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
        """Index documents (dicts with content/title/topic). Returns count.

        CALLED BY   scripts/ingest_sample_data.py — the one-time, offline
                    corpus-loading script. Nothing at RUNTIME (i.e. no node
                    in agents/*.py) ever writes to OpenSearch; only reads
                    (via search() below) happen during a live run.
        """
        if not self.available or not docs:
            return 0
        self.ensure_index()
        # enumerate(docs) pairs each document with its position (0, 1, 2,
        # ...); str(i) becomes that document's OpenSearch _id. Because this
        # id is DETERMINISTIC (always the same for the same position in the
        # same file), re-running ingest on an unchanged corpus.jsonl
        # OVERWRITES the existing documents in place rather than creating
        # duplicates — unlike Qdrant's upsert_texts, which always generates
        # a brand-new random id per call. That asymmetry is a documented
        # difference between the two stores in this codebase.
        for i, doc in enumerate(docs):
            self._client.index(self.index, body=doc, id=str(i))
        # indices.refresh(...) forces OpenSearch to make the just-indexed
        # documents immediately searchable. Without it, there can be a
        # short delay before new documents show up in search results — this
        # call trades a little bit of write performance for immediate
        # read-your-own-writes consistency, which matters for a one-shot
        # ingest script that's typically followed right away by a query.
        self._client.indices.refresh(self.index)
        return len(docs)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 match query. Returns docs + 'bm25_score'; [] when degraded.

        CALLED BY   retrieval/hybrid.py::HybridRetriever.search — the
                    keyword-search half of every hybrid retrieval call.
        """
        if not self.available:
            return []
        res = self._client.search(index=self.index, body={
            "size": top_k,
            # {"match": {"content": query}} is OpenSearch's standard BM25
            # full-text query — "find documents whose `content` field best
            # matches these words", ranked by the classic BM25 relevance
            # formula. `title` and `topic` are stored in the index (see
            # ensure_index's mapping above) but never queried here.
            "query": {"match": {"content": query}},
        })
        out: List[Dict[str, Any]] = []
        # res["hits"]["hits"] is OpenSearch's standard response shape: a
        # list of hit objects, each with the original document under
        # "_source" and its relevance score under "_score".
        for hit in res["hits"]["hits"]:
            doc = dict(hit["_source"])
            doc["bm25_score"] = float(hit["_score"])
            out.append(doc)
        if self._tracer is not None:
            self._tracer.record_retrieval(self._label, query, out)
        return out
