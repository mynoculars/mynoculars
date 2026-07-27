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

import datetime
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)


def content_id(content: str) -> str:
    """Deterministic Qdrant point id derived from raw text content.

    CALLED BY   scripts/ingest_sample_data.py (as id_fn=lambda item:
                content_id(item["content"]), P2-03) and
                memory/semantic_memory.py::SemanticMemory.store_run (same
                wrapper shape, P2-15). ONE canonical identity scheme for
                BOTH callers -- there used to be a second, duplicate
                uuid5-based content_id(item) defined locally inside
                ingest_sample_data.py; P2-15 needed the identical scheme
                for memory, so this promotes it here instead of writing a
                second copy (the two collections -- corpus vs memory --
                never share a point-id namespace anyway, since Qdrant
                scopes ids per collection, so reusing the exact same
                function for both is safe as well as simpler).

    uuid.uuid5(NAMESPACE_URL, content) is deterministic: the SAME string
    always produces the SAME UUID, and it's a valid Qdrant point id (Qdrant
    only accepts unsigned ints or UUID strings, never an arbitrary string)
    -- unlike a raw hash digest, which Qdrant would reject outright.

    Content is the ONLY input, deliberately -- this is an exact-match
    identity scheme, not fuzzy/semantic deduplication. Two evidence items
    with byte-identical text collapse to one point (their payload's
    goal_id/source_query is simply whichever store_run call happened most
    recently -- those fields are diagnostic metadata only, never used to
    filter retrieval, so overwriting them on a repeat is harmless). Two
    items that are semantically the same fact but WORDED differently
    (e.g. paraphrased by a different LLM provider on a different run) get
    DIFFERENT ids and remain separate points -- catching that case would
    need semantic/fuzzy matching, an open problem explicitly out of scope
    here (see PHASE2_TIER3_IMPLEMENTATION_PLAN.md's P2-15 risk note).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, content))


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
        # Guards the lazy TextEmbedding() build in _embed below. ONE
        # QdrantStore is shared across every parallel search_worker, so
        # without this every worker in a MAX_FANOUT fan-out can observe
        # self._embedder is None SIMULTANEOUSLY and each build its own ONNX
        # model — turning one cold start into N concurrent, competing cold
        # starts on the same CPU. This is the identical thundering-herd
        # scripts/mcp_corpus_server.py::_get_corpus_tool already documents
        # fixing one layer up, with the same remedy.
        self._embedder_lock = threading.Lock()
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
            with self._embedder_lock:
                # Re-check inside the lock: the losing threads of a race
                # arrive here after the winner has already built it, and
                # must reuse that instance rather than build a second one.
                if self._embedder is None:
                    from fastembed import TextEmbedding
                    self._embedder = TextEmbedding()  # default small English model
        # self._embedder.embed(texts) returns an iterator of numpy arrays,
        # one per input text; list(v) converts each array into a plain
        # Python list of floats so the rest of this codebase (which has no
        # numpy dependency) can work with ordinary lists.
        return [list(v) for v in self._embedder.embed(texts)]

    def ensure_collection(self) -> None:
        """Create the collection if missing (dimension probed from embedder),
        and make sure the P2-10 payload indexes exist either way.

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
        # P2-10: run on EVERY call, not just when the collection was just
        # created — Qdrant's create_payload_index is idempotent (re-creating
        # an index that already exists is a no-op, not an error), so this
        # costs nothing extra and guarantees the indexes exist even for a
        # collection that already existed from a pre-P2-10 run. D-27 calls
        # these indexes "mandatory ... created before ingest"; running this
        # unconditionally, right here, is what makes that true regardless of
        # collection age.
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> None:
        """Create the payload indexes D-27 requires, if they don't exist yet.

        CALLED BY   ensure_collection() above, on every call (see its
                    comment for why "every call" and not just "on create"
                    is the right choice here).
        WRITES      two Qdrant payload indexes on THIS collection:
                    - "created_at_iso" (datetime) — lets a FormulaQuery
                      reference document age server-side via
                      DatetimeKeyExpression, which requires an actual
                      RFC3339-typed field. Deliberately a SEPARATE field
                      from the existing "created_at" (a raw Unix-epoch
                      float, written since before P2-10 and still what the
                      Python-side decay_factor path in
                      memory/semantic_memory.py reads) — changing that
                      field's type would break every already-ingested point
                      and the still-supported Python fallback path. Adding
                      a second field is purely additive; nothing existing
                      changes shape.
                    - "volatility" (keyword) — lets search_with_decay's
                      per-branch Filter match efficiently instead of a full
                      collection scan per query.
        NOTE        points written BEFORE this shipped have neither
                    "created_at_iso" in their payload NOR were indexed
                    retroactively by this call — an index only accelerates
                    lookups on a field going forward; it doesn't backfill
                    missing payload data on old points. Old points simply
                    won't match a DatetimeKeyExpression built against a key
                    they don't have. This codebase's existing practice for
                    a corpus-shape change is scripts/reset_stores.py /
                    reset_stores.bat — wipe and re-ingest — rather than an
                    in-place migration; the same applies here.
        Safe to call when degraded (no-ops) or when the indexes already
        exist (Qdrant treats re-creation as a no-op, not an error) — but
        any OTHER problem (bad collection name, network hiccup) is caught
        and logged rather than raised, matching this class's overall
        graceful-degradation posture: a failed index-creation attempt
        should not take the run down, only cost some query performance.
        """
        if not self.available:
            return
        from qdrant_client import models
        try:
            self._client.create_payload_index(
                self.collection, field_name="created_at_iso",
                field_schema=models.PayloadSchemaType.DATETIME)
            self._client.create_payload_index(
                self.collection, field_name="volatility",
                field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception as exc:  # noqa: BLE001 — degrade, don't die
            log_event(logger, "qdrant.index_creation_failed", level=logging.WARNING,
                      collection=self.collection, reason=type(exc).__name__)

    # -- public API ---------------------------------------------------------

    def upsert_texts(self, items: List[Dict[str, Any]],
                     id_fn: Optional[Callable[[Dict[str, Any]], str]] = None) -> int:
        """Store items: each dict needs 'content' plus arbitrary payload keys.

        CALLED BY   scripts/ingest_sample_data.py (loading the sample
                    corpus, once, offline) and
                    memory/semantic_memory.py::SemanticMemory.store_run
                    (after every run whose critique passed).
        READS       nothing from ResearchState — this class has no
                    knowledge of the graph; it only ever sees plain dicts.
        WRITES      new points into this instance's Qdrant collection —
                    one per item in `items`.

        P2-03: id_fn is a NEW, OPTIONAL parameter -- callers that don't
        pass it get the exact same uuid4()-per-call behaviour as before,
        nothing changes for them. memory/semantic_memory.py::store_run now
        DOES pass id_fn (P2-15, storage/qdrant_store.py::content_id) --
        see that method's docstring for what changed and why.

        The problem this solves, for whoever DOES pass id_fn: with a
        random id every call, re-running corpus ingest on an UNCHANGED
        corpus.jsonl doesn't overwrite the existing points, it silently
        DUPLICATES them. Passing id_fn=lambda item: <stable hash of
        item["content"]> makes the same input always produce the same
        Qdrant point id, so re-ingesting the same content overwrites in
        place instead of piling up — the same idempotent behaviour
        OpenSearchStore.ingest already has via its deterministic str(i)
        document ids.

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
        points = []
        for item, vec in zip(items, vectors):
            # P2-10: write BOTH the original float "created_at" (unchanged
            # — the Python-side decay_factor path in
            # memory/semantic_memory.py still reads this) AND a new
            # "created_at_iso" RFC3339 string of the SAME instant, purely
            # additive, so a FormulaQuery's DatetimeKeyExpression (which
            # requires an actual datetime-typed field) has something to
            # reference without changing what any existing reader sees.
            now = time.time()
            payload = {
                **item,
                "created_at": now,
                "created_at_iso": datetime.datetime.fromtimestamp(
                    now, tz=datetime.timezone.utc).isoformat(),
            }
            points.append(models.PointStruct(
                # id_fn(item) if the caller opted in, else the ORIGINAL
                # uuid4() behaviour — this is the entire backward-
                # compatibility guarantee: no id_fn passed means byte-for-
                # byte the same behaviour as before this change.
                id=id_fn(item) if id_fn else str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            ))
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

    def search_with_decay(self, query: str, top_k: int, decay_field: str,
                          half_lives: Dict[str, Optional[float]],
                          overfetch: int = 4) -> List[Dict[str, Any]]:
        """Similarity search with volatility-aware decay applied SERVER-SIDE
        (P2-10, implements D-27), instead of Python's over-fetch-then-rerank
        in memory/semantic_memory.py::SemanticMemory.retrieve.

        CALLED BY   memory/semantic_memory.py::SemanticMemory.retrieve,
                    ONLY when settings.memory_server_side_decay is True —
                    the Python decay_factor()-based path (this method's
                    PARITY ORACLE, deliberately kept, never deleted — see
                    that module's docstring) remains the default and the
                    fallback.
        READS       nothing from ResearchState — same contract as search()
                    above; this is a plain storage-layer method.

        Parameters:
            query: the text to embed and search for.
            top_k: how many FINAL (post-decay) results to return.
            decay_field: the payload field whose VALUE selects which
                half-life applies to a given point (e.g. "volatility").
                Deliberately a parameter, not hardcoded — this class stays
                policy-free about what "decay_field" means; the caller
                (SemanticMemory) supplies the domain concept.
            half_lives: {field_value: half_life_in_days}. A value of None
                means "no decay for this category" (a flat 1.0 multiplier)
                — e.g. {"stable": None, "semi_stable": 90.0, "volatile": 14.0}.
                Every DISTINCT field_value present becomes one filtered
                branch in the formula below (see the module-level note in
                this method's body for why one branch per value, not a
                single expression, is Qdrant's own constraint here, not a
                design choice).
            overfetch: how many candidates the vector-similarity PREFETCH
                stage retrieves before the formula reranks and top_k cuts
                — mirrors the *2 over-fetch the Python path already used,
                as a multiplier instead (default 4x is more generous since
                this path pays no per-candidate Python-side cost for it).

        RETURNS     the SAME shape search() returns — payload dicts plus a
                    "similarity" key — EXCEPT that "similarity" here is
                    ALREADY the decay-adjusted final score (vector
                    similarity x decay multiplier), computed by Qdrant, not
                    a raw cosine similarity awaiting a Python-side
                    decay_factor() call. Callers that branch on
                    server_side_decay must not re-apply decay_factor() to
                    these results — see SemanticMemory.retrieve.
                    [] when degraded, same as search().

        WHY ONE BRANCH PER half_lives VALUE, NOT ONE EXPRESSION:
        Qdrant's DecayParamsExpression.scale is a plain float CONSTANT — it
        cannot reference a per-point payload field the way `x`/`target` can
        (verified against the installed qdrant-client's actual Pydantic
        model, not assumed). Three volatility classes each needing a
        DIFFERENT half-life therefore cannot be expressed as a single decay
        expression with a variable scale. The construction below instead
        SUMS one branch per distinct field_value: each branch is
        (a Filter matching that value, evaluated as a 1/0 switch) MULTIPLIED
        BY (that value's own decay expression, or a flat 1.0 if its
        half-life is None) — since exactly one branch's filter can match
        any given point, the sum behaves like a per-point "select the right
        half-life" despite `scale` itself never varying within a branch.
        """
        if not self.available or not half_lives:
            return []
        from qdrant_client import models
        self.ensure_collection()
        vec = self._embed([query])[0]

        # SECONDS_PER_DAY: Qdrant's datetime decay measures `scale` in the
        # same units the datetime difference naturally resolves to —
        # seconds, since both x and target here are real RFC3339 instants.
        # half_lives is expressed in DAYS (matching decay_factor()'s own
        # units in memory/semantic_memory.py), so this converts once.
        SECONDS_PER_DAY = 86400.0

        def _branch(field_value: str, half_life_days: Optional[float]):
            switch = models.Filter(must=[models.FieldCondition(
                key=decay_field, match=models.MatchValue(value=field_value))])
            if half_life_days is None:
                # "No decay" for this category — a flat 1.0 multiplier,
                # gated by the same switch so it only applies to points
                # that actually match this field_value.
                return models.MultExpression(mult=[switch, 1.0])
            return models.MultExpression(mult=[
                switch,
                models.ExpDecayExpression(exp_decay=models.DecayParamsExpression(
                    x=models.DatetimeKeyExpression(datetime_key="created_at_iso"),
                    target=models.DatetimeExpression(datetime="now"),
                    scale=half_life_days * SECONDS_PER_DAY,
                    midpoint=0.5,
                )),
            ])

        decay_expr = models.SumExpression(
            sum=[_branch(value, hl) for value, hl in half_lives.items()])
        formula = models.MultExpression(mult=["$score", decay_expr])

        hits = self._client.query_points(
            self.collection,
            prefetch=models.Prefetch(query=vec, limit=max(top_k * overfetch, top_k)),
            query=models.FormulaQuery(formula=formula),
            limit=top_k,
        ).points

        now = time.time()
        out: List[Dict[str, Any]] = []
        for h in hits:
            payload: Dict[str, Any] = dict(h.payload or {})
            # h.score here IS the formula's output ($score x decay) —
            # already the final, decay-adjusted value, not a raw cosine
            # similarity. Callers must treat it as such (see RETURNS above).
            payload["similarity"] = float(h.score)
            payload["age_days"] = (now - float(payload.get("created_at", now))) / 86400.0
            out.append(payload)
        if self._tracer is not None:
            self._tracer.record_retrieval(self._label, query, out)
        return out

    def scroll_all(self, batch_size: int = 256) -> List[Dict[str, Any]]:
        """Return every point in this collection as {"id": ..., **payload}.

        CALLED BY   scripts/gc_memory.py (P2-15) -- a batch job needs to
                    see EVERY point to decide what's decayed past keeping,
                    unlike search()/search_with_decay() above, which only
                    ever look at the top-k candidates for one query.
        READS       nothing from ResearchState -- same contract as every
                    other method here.
        RETURNS     a flat list of payload dicts, each with its Qdrant
                    point id folded in under the "id" key (payloads
                    otherwise never carry their own id -- callers that
                    need to act on specific points, like gc_memory.py
                    deciding what to pass to delete_points(), need it
                    surfaced explicitly). [] when degraded or empty.

        WHY PAGINATED, NOT ONE CALL: Qdrant's scroll() API is itself
        paginated (this is Qdrant's own design, not a choice made here) --
        each call returns at most `batch_size` records plus a
        next_page_offset token, or None once there's nothing left. The
        while loop below just follows that token until it runs out, same
        idea as following a "next page" link, one page at a time.
        """
        if not self.available:
            return []
        self.ensure_collection()
        out: List[Dict[str, Any]] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                self.collection, limit=batch_size, offset=offset,
                with_payload=True, with_vectors=False)
            for r in records:
                out.append({"id": r.id, **(r.payload or {})})
            if offset is None:
                break
        return out

    def delete_points(self, ids: List[str]) -> int:
        """Delete the given point ids. Returns how many were REQUESTED
        (Qdrant's delete response doesn't confirm a count back, only that
        the operation completed -- see the try/except below for what
        happens if it doesn't).

        CALLED BY   scripts/gc_memory.py (P2-15), only after its own
                    --yes gate -- this method itself has no confirmation
                    logic of its own; that responsibility stays in the
                    CALLING script, matching reset_stores.py's existing
                    --dry-run/--yes convention rather than duplicating a
                    second confirmation mechanism inside the storage layer.
        WRITES      removes points from this collection. Nothing else.

        Fails open (returns 0, logs a warning) rather than raising --
        matches this class's overall graceful-degradation posture: a
        failed cleanup attempt should not crash whatever called it.
        """
        if not self.available or not ids:
            return 0
        try:
            self._client.delete(self.collection, points_selector=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001 -- degrade, don't die
            log_event(logger, "qdrant.delete_points_failed", level=logging.WARNING,
                      collection=self.collection, reason=type(exc).__name__)
            return 0

    def existing_point_ids(self, ids: List[str]) -> set:
        """Return the subset of `ids` that already exist in this collection.

        CALLED BY   memory/semantic_memory.py::SemanticMemory.store_run
                    (P2-15 follow-up) -- BEFORE calling upsert_texts, so it
                    can log how many of this run's fresh items are brand
                    new points versus overwrites of an already-known fact
                    (content_id makes that distinction meaningful -- see
                    that function's docstring).
        READS       nothing from ResearchState -- plain id list in,
                    matching id set out.
        CALLS       self._client.retrieve(..., with_payload=False,
                    with_vectors=False) -- the cheapest possible query
                    that still tells us existence: neither the payload
                    nor the vector is needed, only WHICH of the requested
                    ids Qdrant actually has (retrieve() only returns
                    records that exist; ids not found are simply absent
                    from the result, never an error).

        Fails open (returns an empty set, logs a warning) rather than
        raising -- if this check can't run, store_run's own upsert_texts
        call still proceeds unaffected; the only cost is a less accurate
        "new vs overwritten" log line, never a broken write.
        """
        if not self.available or not ids:
            return set()
        try:
            records = self._client.retrieve(
                self.collection, ids=ids, with_payload=False, with_vectors=False)
            return {r.id for r in records}
        except Exception as exc:  # noqa: BLE001 -- degrade, don't die
            log_event(logger, "qdrant.existing_point_ids_failed", level=logging.WARNING,
                      collection=self.collection, reason=type(exc).__name__)
            return set()
