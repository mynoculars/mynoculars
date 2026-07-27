"""
tests/unit/test_storage_qdrant_store.py — storage/qdrant_store.py.

Covers: payload index creation (P2-10), upsert_texts' created_at/
created_at_iso fields, search_with_decay's FormulaQuery construction
(P2-10 — see the IMPORTANT LIMITATION note below), existing_point_ids/
scroll_all/delete_points (P2-15), and content_id's determinism/shape.
Does NOT cover SemanticMemory's own logic that CALLS these methods —
see test_memory_semantic.py, which fakes this module's classes instead.

IMPORTANT LIMITATION, stated plainly: none of these tests run against a
real Qdrant server — this whole suite is offline by design (see
conftest.py's module docstring), and no live Qdrant was reachable in the
environment these were written in either. What IS verified here:
  1. The FormulaQuery/DecayParamsExpression/etc. objects construct
     without a Pydantic validation error against the ACTUAL installed
     qdrant-client version (not guessed, not copied from the design
     doc's Appendix C, which a prior external review is on record for
     having invented API symbols in).
  2. The exact shape of what gets sent (index field names/types, filter
     values, scale/midpoint numbers) matches what P2-10 was scoped to do.
  3. A NUMERIC PARITY check (test_memory_semantic.py, not here): if
     Qdrant evaluates an exp_decay formula per its documented semantics,
     the value it would return is mathematically identical to
     decay_factor()'s Python output for the same age/half-life. This
     proves the FORMULA is correct; it does NOT prove Qdrant's server
     actually executes it this way — that step needs a live run against
     a real server (see internal/PHASE-2_TIER-3_PLAN.md's P2-10 risk
     note).
Before trusting this in production: run search_with_decay against a real
Qdrant (client 1.18.0 / server 1.17.1 confirmed compatible) and compare
its output to the Python path on the same corpus.
"""

import datetime as _dt
import uuid as _uuid
from unittest.mock import MagicMock

import pytest

# See tests/unit/test_gc_memory.py for why this is a skip, not a failure.
pytest.importorskip("qdrant_client")

from research_agent.storage.qdrant_store import QdrantStore, content_id


def _mock_store(collection="test_collection"):
    """A QdrantStore with a real (degraded, unreachable) __init__ pass, then
    forced "available" with a MagicMock in place of the real qdrant-client
    connection — lets us test the Qdrant-API-CALLING code paths (something
    no existing test in this suite does; every prior QdrantStore test only
    ever exercised the degraded no-op path or a standalone helper function)
    without a real server."""
    store = QdrantStore("http://127.0.0.1:1", collection)
    assert store.available is False  # sanity: really did fail to connect
    store.available = True
    store._client = MagicMock()
    store._embedder = MagicMock()
    # A fresh 3-float vector per call — NOT a fixed return_value, which
    # would hand back the SAME (single-use) iterator object on every call
    # and silently break the second of any two _embed() calls in one test.
    store._embedder.embed = MagicMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    # get_collections()... .collections defaults to an empty iterator on a
    # MagicMock, so ensure_collection() always takes the "create it" path
    # — fine for these tests, which only care about what's SENT, not the
    # collection-exists check.
    return store


# ---------------------------------------------------------------------------
# Payload indexes (P2-10)
# ---------------------------------------------------------------------------


def test_ensure_payload_indexes_creates_the_two_required_indexes():
    from qdrant_client import models

    store = _mock_store()
    store.ensure_payload_indexes()

    calls = store._client.create_payload_index.call_args_list
    assert len(calls) == 2
    fields = {c.kwargs["field_name"]: c.kwargs["field_schema"] for c in calls}
    assert fields["created_at_iso"] == models.PayloadSchemaType.DATETIME
    assert fields["volatility"] == models.PayloadSchemaType.KEYWORD


def test_ensure_payload_indexes_is_a_noop_when_degraded():
    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    store.ensure_payload_indexes()  # must not raise


def test_ensure_payload_indexes_fails_open_on_client_error(caplog):
    import logging

    store = _mock_store()
    store._client.create_payload_index = MagicMock(side_effect=RuntimeError("qdrant is down"))
    with caplog.at_level(logging.WARNING):
        store.ensure_payload_indexes()  # must not raise
    assert any("qdrant.index_creation_failed" in r.message for r in caplog.records)


def test_ensure_collection_calls_ensure_payload_indexes_every_time():
    """P2-10: unlike collection creation itself (only on first use),
    payload-index creation must run on EVERY ensure_collection() call —
    it's what guarantees the indexes exist even for a collection that
    already existed from before P2-10 shipped."""
    store = _mock_store()
    store.ensure_collection()
    store.ensure_collection()
    assert store._client.create_payload_index.call_count == 4  # 2 indexes x 2 calls


# ---------------------------------------------------------------------------
# upsert_texts
# ---------------------------------------------------------------------------


def test_upsert_texts_writes_both_created_at_and_created_at_iso():
    store = _mock_store()
    store.upsert_texts([{"content": "fact one"}])

    upsert_call = store._client.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[-1]
    payload = points[0].payload
    assert "created_at" in payload and isinstance(payload["created_at"], float)
    assert "created_at_iso" in payload
    # Round-trips as a real ISO/RFC3339 instant — this is exactly the shape
    # DatetimeKeyExpression needs; a malformed string here would silently
    # never match server-side, not raise loudly, so this is worth checking.
    _dt.datetime.fromisoformat(payload["created_at_iso"])


# ---------------------------------------------------------------------------
# search_with_decay (P2-10) — see module docstring for the parity caveat
# ---------------------------------------------------------------------------


def test_search_with_decay_returns_empty_list_when_degraded():
    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    out = store.search_with_decay("q", top_k=3, decay_field="volatility",
                                  half_lives={"stable": None, "semi_stable": 90.0})
    assert out == []


def test_search_with_decay_builds_a_valid_formula_query_and_correct_scales():
    """The real test of the P2-10 risk this item was flagged for: does the
    FormulaQuery this method builds actually validate against the REAL
    installed qdrant-client Pydantic models (not a hand-rolled dict that
    only LOOKS right)? A ValidationError here would surface immediately —
    this test doesn't even need to inspect the object's shape to prove
    that much, though it does so anyway for the scale-math check below."""
    from qdrant_client import models

    store = _mock_store()
    fake_point = MagicMock()
    fake_point.payload = {"content": "x", "volatility": "semi_stable",
                          "created_at": 1700000000.0,
                          "created_at_iso": "2023-11-14T22:13:20+00:00"}
    fake_point.score = 0.42
    fake_response = MagicMock()
    fake_response.points = [fake_point]
    store._client.query_points = MagicMock(return_value=fake_response)

    out = store.search_with_decay(
        "q", top_k=3, decay_field="volatility",
        half_lives={"stable": None, "semi_stable": 90.0, "volatile": 14.0})

    # Constructed without error (no ValidationError raised above) AND
    # produced a result in the expected shape.
    assert out == [{
        "content": "x", "volatility": "semi_stable",
        "created_at": 1700000000.0, "created_at_iso": "2023-11-14T22:13:20+00:00",
        "similarity": 0.42,
        "age_days": out[0]["age_days"],  # computed from wall-clock "now"; not asserting an exact value
    }]

    call = store._client.query_points.call_args
    formula = call.kwargs["query"]
    assert isinstance(formula, models.FormulaQuery)
    outer = formula.formula
    assert isinstance(outer, models.MultExpression)
    assert outer.mult[0] == "$score"
    decay_sum = outer.mult[1]
    assert isinstance(decay_sum, models.SumExpression)
    assert len(decay_sum.sum) == 3  # one branch per half_lives entry

    # Find the semi_stable branch and check its scale is EXACTLY
    # half_life_days * 86400 seconds — the conversion this method's
    # docstring promises.
    semi_branch = next(
        b for b in decay_sum.sum
        if b.mult[0].must[0].match.value == "semi_stable")
    exp_decay = semi_branch.mult[1]
    assert isinstance(exp_decay, models.ExpDecayExpression)
    assert exp_decay.exp_decay.scale == 90.0 * 86400.0
    assert exp_decay.exp_decay.midpoint == 0.5
    assert exp_decay.exp_decay.x.datetime_key == "created_at_iso"
    assert exp_decay.exp_decay.target.datetime == "now"

    # The "stable" branch (half_life=None) must be a flat 1.0 multiplier,
    # not a decay expression — confirms the None-means-no-decay contract.
    stable_branch = next(
        b for b in decay_sum.sum
        if b.mult[0].must[0].match.value == "stable")
    assert stable_branch.mult[1] == 1.0


def test_search_with_decay_prefetch_overfetches_by_the_given_multiplier():
    from qdrant_client import models

    store = _mock_store()
    fake_response = MagicMock()
    fake_response.points = []
    store._client.query_points = MagicMock(return_value=fake_response)

    store.search_with_decay("q", top_k=5, decay_field="volatility",
                            half_lives={"stable": None}, overfetch=4)

    call = store._client.query_points.call_args
    prefetch = call.kwargs["prefetch"]
    assert isinstance(prefetch, models.Prefetch)
    assert prefetch.limit == 20  # top_k(5) * overfetch(4)
    assert call.kwargs["limit"] == 5  # final cut is still just top_k


# ---------------------------------------------------------------------------
# existing_point_ids (P2-15)
# ---------------------------------------------------------------------------


def test_existing_point_ids_returns_only_the_ids_qdrant_actually_has():
    class _Rec:
        def __init__(self, id_):
            self.id = id_

    store = _mock_store()
    store._client.retrieve = MagicMock(return_value=[_Rec("id1"), _Rec("id3")])

    result = store.existing_point_ids(["id1", "id2", "id3"])

    assert result == {"id1", "id3"}
    call = store._client.retrieve.call_args
    assert call.args[0] == "test_collection"
    assert call.kwargs["ids"] == ["id1", "id2", "id3"]
    assert call.kwargs["with_payload"] is False
    assert call.kwargs["with_vectors"] is False


def test_existing_point_ids_returns_empty_set_when_degraded():
    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    assert store.existing_point_ids(["id1"]) == set()


def test_existing_point_ids_fails_open_on_client_error(caplog):
    import logging

    store = _mock_store()
    store._client.retrieve = MagicMock(side_effect=RuntimeError("qdrant is down"))
    with caplog.at_level(logging.WARNING):
        result = store.existing_point_ids(["id1"])

    assert result == set()
    assert any("qdrant.existing_point_ids_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# scroll_all / delete_points (P2-15)
# ---------------------------------------------------------------------------


def test_scroll_all_follows_pagination_until_offset_is_none():
    store = _mock_store()
    r1, r2, r3 = MagicMock(), MagicMock(), MagicMock()
    r1.id, r1.payload = "id1", {"content": "a"}
    r2.id, r2.payload = "id2", {"content": "b"}
    r3.id, r3.payload = "id3", {"content": "c"}
    store._client.scroll = MagicMock(side_effect=[([r1, r2], "page2token"), ([r3], None)])

    out = store.scroll_all(batch_size=2)

    assert out == [
        {"id": "id1", "content": "a"},
        {"id": "id2", "content": "b"},
        {"id": "id3", "content": "c"},
    ]
    assert store._client.scroll.call_count == 2


def test_scroll_all_returns_empty_list_when_degraded():
    store = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert store.available is False
    assert store.scroll_all() == []


def test_delete_points_calls_client_delete_with_the_given_ids():
    store = _mock_store()
    n = store.delete_points(["id1", "id2"])

    assert n == 2
    call = store._client.delete.call_args
    assert call.args[0] == "test_collection"
    assert call.kwargs["points_selector"] == ["id1", "id2"]


def test_delete_points_is_a_noop_on_empty_list_or_when_degraded():
    degraded = QdrantStore("http://127.0.0.1:1", "test_collection")
    assert degraded.delete_points(["id1"]) == 0  # degraded -> no client call possible

    store = _mock_store()
    assert store.delete_points([]) == 0
    assert store._client.delete.called is False


def test_delete_points_fails_open_on_client_error(caplog):
    import logging

    store = _mock_store()
    store._client.delete = MagicMock(side_effect=RuntimeError("qdrant is down"))
    with caplog.at_level(logging.WARNING):
        n = store.delete_points(["id1"])

    assert n == 0
    assert any("qdrant.delete_points_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# content_id (P2-03 follow-up / P2-15)
# ---------------------------------------------------------------------------


def test_content_id_is_deterministic_for_same_content():
    assert content_id("Redis is an in-memory data store.") == \
        content_id("Redis is an in-memory data store.")


def test_content_id_differs_for_different_content():
    assert content_id("Redis is an in-memory data store.") != \
        content_id("Cassandra is a distributed database.")


def test_content_id_is_a_valid_qdrant_point_id_shape():
    # Qdrant point ids must be an unsigned int or a UUID string — a raw
    # hash digest would be rejected outright. uuid.uuid5(...) guarantees
    # this shape; this test would fail loudly if that ever changed to a
    # plain hexdigest by mistake.
    result = content_id("anything")
    parsed = _uuid.UUID(result)  # raises ValueError if not a real UUID
    assert str(parsed) == result
