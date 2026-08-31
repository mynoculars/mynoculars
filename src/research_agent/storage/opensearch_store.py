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
from research_agent.storage.qdrant_store import _retrieval_trace_fields, content_id

logger = logging.getLogger(__name__)


class OpenSearchStore:
    """Thin, failure-tolerant wrapper over opensearch-py."""

    def __init__(self, url: str, index: str, username: str = "",
                 password: str = "", use_ssl: bool = False,
                 verify_certs: bool = False, tracer: Any = None,
                 trace_label: str = "OPENSEARCH (BM25)", probe: bool = True):
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
        if not probe:
            # D-140, same contract as QdrantStore: construct already-
            # degraded without opening a socket. See that class for the
            # measurement that motivated it.
            self._tracer = tracer
            self._label = trace_label
            return
        try:
            from opensearchpy import OpenSearch
            # Build up the connection arguments as a plain dict FIRST,
            # conditionally adding keys, then unpack it all at once into
            # the actual OpenSearch(...) constructor call below — see the
            # module docstring's **kwargs explanation.
            # pool_maxsize=20: default urllib3 pool is small (often 1),
            # which under concurrent search_worker fan-out (D-13,
            # settings.max_fanout, default 6) causes real, harmless-but-
            # noisy "Connection pool is full, discarding connection"
            # warnings -- each discarded connection still WORKS, just
            # forces a fresh TCP+TLS handshake next time instead of
            # reuse. 20 covers max_fanout's default with real headroom
            # for a future higher setting, at negligible memory cost.
            kwargs = {"hosts": [url], "timeout": 5, "pool_maxsize": 20}
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
        if not self._client.indices.exists(index=self.index):
            self._client.indices.create(index=self.index, body={
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
        # content_id(doc["content"]) — the SAME uuid5 content identity
        # scheme storage/qdrant_store.py uses for its point ids, so BOTH
        # legs of hybrid retrieval now address a document the same way.
        #
        # This replaces a POSITIONAL str(i) id. Positional ids are
        # idempotent only for a byte-identical corpus: delete or reorder a
        # line in corpus.jsonl and re-ingest, and OpenSearch overwrites by
        # position while Qdrant overwrites by content — leaving stale
        # documents at now-unused positions and silently drifting the two
        # legs apart, which surfaces as a fused ranking neither leg agrees
        # with. One identity scheme for both stores removes that class of
        # divergence, and makes the id usable as an RRF join key if the
        # title-based key in retrieval/hybrid.py is ever replaced.
        #
        # A doc with no "content" falls back to its position, so a
        # malformed corpus line still ingests rather than raising.
        for i, doc in enumerate(docs):
            doc_id = content_id(doc["content"]) if doc.get("content") else str(i)
            self._client.index(index=self.index, body=doc, id=doc_id)
        # indices.refresh(...) forces OpenSearch to make the just-indexed
        # documents immediately searchable. Without it, there can be a
        # short delay before new documents show up in search results — this
        # call trades a little bit of write performance for immediate
        # read-your-own-writes consistency, which matters for a one-shot
        # ingest script that's typically followed right away by a query.
        self._client.indices.refresh(index=self.index)
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
        log_event(logger, "retrieval.raw", source=self._label, query=query,
                  hit_count=len(out), **_retrieval_trace_fields(self._tracer, out))
        return out
