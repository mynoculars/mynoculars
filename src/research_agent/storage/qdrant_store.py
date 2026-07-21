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
        """Connect lazily; mark unavailable instead of raising."""
        self.collection = collection
        self.available = False
        self._client = None
        self._embedder = None
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=url, timeout=5)
            self._client.get_collections()  # cheap liveness probe
            self.available = True
        except Exception as exc:  # noqa: BLE001 — degrade, don't die
            log_event(logger, "qdrant.unavailable", level=logging.WARNING,
                      reason=type(exc).__name__)
        self._tracer = tracer
        self._label = trace_label

    # -- internals ----------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed with a lazily-loaded local fastembed model."""
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding()  # default small English model
        return [list(v) for v in self._embedder.embed(texts)]

    def ensure_collection(self) -> None:
        """Create the collection if missing (dimension probed from embedder)."""
        if not self.available:
            return
        from qdrant_client import models
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            dim = len(self._embed(["probe"])[0])
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    # -- public API ---------------------------------------------------------

    def upsert_texts(self, items: List[Dict[str, Any]]) -> int:
        """Store items: each dict needs 'content' plus arbitrary payload keys.

        Returns number stored (0 when degraded)."""
        if not self.available or not items:
            return 0
        from qdrant_client import models
        self.ensure_collection()
        vectors = self._embed([i["content"] for i in items])
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={**item, "created_at": time.time()},
            )
            for item, vec in zip(items, vectors)
        ]
        self._client.upsert(self.collection, points=points)
        return len(points)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Similarity search; returns payload dicts + 'similarity' + 'age_days'.

        Returns [] when degraded — callers need no availability checks."""
        if not self.available:
            return []
        self.ensure_collection()
        vec = self._embed([query])[0]
        hits = self._client.query_points(self.collection, query=vec, limit=top_k).points
        now = time.time()
        out: List[Dict[str, Any]] = []
        for h in hits:
            payload: Dict[str, Any] = dict(h.payload or {})
            payload["similarity"] = float(h.score)
            payload["age_days"] = (now - float(payload.get("created_at", now))) / 86400.0
            out.append(payload)
        if self._tracer is not None:
            self._tracer.record_retrieval(self._label, query, out)
        return out
