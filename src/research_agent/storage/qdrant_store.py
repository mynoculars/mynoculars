"""
storage/qdrant_store.py — Vector store abstraction (semantic memory backend).

Purpose:
    All Qdrant I/O lives here: collection bootstrap, upsert with payload,
    similarity search. The memory module composes decay logic ON TOP of this
    — storage stays policy-free.

Responsibilities:
    - Lazy connect + graceful degradation: if Qdrant is unreachable, expose
      available=False and no-op, so the agent still runs (memory-off mode)
      instead of crashing at import time.
    - Embed text via fastembed (local ONNX model, no API key needed).

Design decisions:
    - fastembed over an embedding API: keeps the repo runnable offline and
      free; tradeoff is a one-time model download (~100 MB) on first use.
    - Payload schema mirrors the design doc (D-24): content, volatility,
      created_at, source_query. Supersession links are deferred to the full
      build (documented in README Limitations).

This ONE class backs TWO different collections in this codebase — the
corpus (fresh retrieval, written by scripts/ingest_sample_data.py) and
semantic memory (cross-run recall, written by memory/semantic_memory.py) —
both are just "a QdrantStore pointed at a different collection name." See
cli.py::build_app_and_settings for where each instance is constructed.

Python mechanics used in this file, if any of this is new to you:
    imports INSIDE functions (e.g. "from qdrant_client import QdrantClient"
    appears inside __init__, not at the top of the file)
        Normally Python imports live at the top of a file, run once when the
        module is first loaded. Here they're deliberately placed INSIDE
        methods so that importing THIS FILE never requires qdrant-client to
        be installed or reachable — the import (and therefore any error it
        could raise) only happens when a QdrantStore is actually
        constructed, wrapped in the try/except below. This is part of the
        graceful-degradation design: a machine with no Qdrant installed can
        still import and run every other part of this codebase.
    zip(items, vectors)
        Pairs up two lists element-by-element: zip([a,b,c], [1,2,3]) yields
        (a,1), then (b,2), then (c,3). Used in upsert_texts below to walk
        through each original item dict together with the embedding vector
        that was computed FOR that item, in the same order.
    dict comprehension:  {**item, "created_at": time.time()}
        The "**" here means "unpack all of this dict's key-value pairs into
        the new dict literal being built." So {**item, "created_at": ...}
        creates a BRAND NEW dict containing every key/value already in
        `item`, PLUS one more key ("created_at") added on top — without
        modifying the original `item` dict at all.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


class QdrantStore:
    """Thin, failure-tolerant wrapper over qdrant-client."""

    def __init__(self, url: str, collection: str, tracer: Any = None,
                 trace_label: str = "QDRANT (dense)"):
        """Connect lazily; mark unavailable instead of raising.

        CALLED BY   cli.py::build_app_and_settings, twice per run — once
                    for the corpus collection, once for the memory
                    collection (each with a different `collection` name
                    and `trace_label`).
        WRITES      self.available — every other method on this class
                    checks this flag FIRST and no-ops (returns [] or 0)
                    if it's False, rather than attempting a network call
                    that would just fail again.
        """
        self.collection = collection
        self.available = False
        self._client = None
        self._embedder = None
        try:
            # Import happens here, not at module load time — see the
            # module docstring's explanation of why.
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=url, timeout=5)
            self._client.get_collections()  # cheap liveness probe
            self.available = True
        except Exception as exc:  # noqa: BLE001 — degrade, don't die
            # ANY problem here (Qdrant not installed, not running, wrong
            # URL, network timeout...) is treated identically: log a
            # warning and leave self.available as False. The rest of this
            # class is written so that being unavailable is a normal,
            # silently-handled state, never a crash.
            log_event(logger, "qdrant.unavailable", level=logging.WARNING,
                      reason=type(exc).__name__)
        self._tracer = tracer
        self._label = trace_label

    # -- internals ----------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed with a lazily-loaded local fastembed model.

        CALLED BY   ensure_collection, upsert_texts, and search below —
                    every method on this class that needs to turn text into
                    a vector.
        "Lazily-loaded" means the actual TextEmbedding() model (which
        involves downloading and initializing an ONNX model on first use)
        is only constructed the FIRST time _embed is ever called on this
        instance — self._embedder starts as None in __init__ above, and
        this method fills it in exactly once, then reuses it on every
        subsequent call.
        """
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding()  # default small English model
        # self._embedder.embed(texts) returns an iterator of numpy arrays,
        # one per input text; list(v) converts each array into a plain
        # Python list of floats so the rest of this codebase (which has no
        # numpy dependency) can work with ordinary lists.
        return [list(v) for v in self._embedder.embed(texts)]

    def ensure_collection(self) -> None:
        """Create the collection if missing (dimension probed from embedder).

        CALLED BY   upsert_texts and search below, at the START of each —
                    so a collection is created automatically on first use,
                    with no separate "setup" step required anywhere else in
                    this codebase.
        """
        if not self.available:
            return
        from qdrant_client import models
        # {c.name for c in self._client.get_collections().collections} is a
        # SET COMPREHENSION — like a list comprehension, but builds a set
        # (unordered, no duplicates) instead of a list. Here it collects
        # every existing collection's name so we can cheaply check
        # membership ("in existing") below.
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            # Embed a single throwaway string just to discover how many
            # numbers long the embedding model's output vectors are — the
            # collection has to be created with that exact dimension
            # up front; there's no other API to ask the model directly.
            dim = len(self._embed(["probe"])[0])
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    # -- public API ---------------------------------------------------------

    def upsert_texts(self, items: List[Dict[str, Any]]) -> int:
        """Store items: each dict needs 'content' plus arbitrary payload keys.

        CALLED BY   scripts/ingest_sample_data.py (loading the sample
                    corpus, once, offline) and
                    memory/semantic_memory.py::SemanticMemory.store_run
                    (after every run whose critique passed).
        READS       nothing from ResearchState — this class has no
                    knowledge of the graph; it only ever sees plain dicts.
        WRITES      new points into this instance's Qdrant collection —
                    one per item in `items`, each with a brand-new random
                    id (see the "id=str(uuid.uuid4())" line below — this is
                    exactly why re-running ingest on the SAME corpus
                    produces DUPLICATE points rather than overwriting the
                    previous ones, a documented limitation of this build).

        Returns number stored (0 when degraded).
        """
        if not self.available or not items:
            return 0
        from qdrant_client import models
        self.ensure_collection()
        # Embed every item's "content" field in ONE batched call (more
        # efficient than calling _embed once per item), then zip() the
        # original items back up with their corresponding vectors — see
        # the module docstring for exactly what zip() does here.
        vectors = self._embed([i["content"] for i in items])
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                # {**item, "created_at": time.time()} — see the module
                # docstring's dict-comprehension explanation. This builds a
                # payload containing every field the caller supplied in
                # `item`, PLUS a fresh timestamp recording when this point
                # was written (used later by decay_factor in
                # memory/semantic_memory.py to compute an item's age).
                payload={**item, "created_at": time.time()},
            )
            for item, vec in zip(items, vectors)
        ]
        self._client.upsert(self.collection, points=points)
        return len(points)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Similarity search; returns payload dicts + 'similarity' + 'age_days'.

        CALLED BY   retrieval/hybrid.py::HybridRetriever.search (for the
                    corpus collection) and
                    memory/semantic_memory.py::SemanticMemory.retrieve (for
                    the memory collection) — the two callers pass in
                    QdrantStore instances pointed at different collections,
                    but call this exact same method.
        RETURNS     [] immediately when degraded — callers need no
                    availability checks of their own; a QdrantStore always
                    behaves like "a store with zero results" rather than
                    raising, whether it's actually down or genuinely just
                    has no matches.

        Returns [] when degraded — callers need no availability checks.
        """
        if not self.available:
            return []
        self.ensure_collection()
        vec = self._embed([query])[0]
        # query_points(...) is the actual Qdrant similarity search call:
        # "find the `top_k` points whose stored vectors are closest to
        # `vec`". `.points` unwraps the response object down to the plain
        # list of hit objects.
        hits = self._client.query_points(self.collection, query=vec, limit=top_k).points
        now = time.time()
        out: List[Dict[str, Any]] = []
        for h in hits:
            # dict(h.payload or {}) makes a plain, mutable copy of this
            # hit's stored payload (falling back to an empty dict if, for
            # some reason, a point was stored with no payload at all) —
            # the two lines below then ADD extra computed fields onto that
            # copy without touching whatever Qdrant itself returned.
            payload: Dict[str, Any] = dict(h.payload or {})
            payload["similarity"] = float(h.score)
            payload["age_days"] = (now - float(payload.get("created_at", now))) / 86400.0
            out.append(payload)
        if self._tracer is not None:
            self._tracer.record_retrieval(self._label, query, out)
        return out
