"""
tests/unit/test_gc_memory.py — scripts/gc_memory.py's find_gc_candidates
(P2-15).

gc_memory.py is a standalone operational script (like reset_stores.py,
which has no pytest coverage of its own at all) -- but unlike
reset_stores.py, its actual decision logic (find_gc_candidates) is a
pure function worth testing directly, so it's loaded by file path here,
the same way scripts/ingest_sample_data.py's content_id() used to be
before it moved into the regular package (see test_storage_qdrant_store.py's
own history for that migration).
"""

import time
from unittest.mock import MagicMock

import pytest

# qdrant-client is an OPTIONAL runtime dependency (storage/qdrant_store.py
# imports it lazily, inside methods, precisely so the rest of the codebase
# runs without it). A minimal install should therefore SKIP these, not
# fail them -- a hard failure reports a missing optional extra as a
# broken test suite.
pytest.importorskip("qdrant_client")

from research_agent.storage.qdrant_store import QdrantStore


def _mock_store(collection="test_collection"):
    """Same shape as test_storage_qdrant_store.py's own _mock_store —
    kept local here rather than shared, since this file's only need is a
    fake .scroll() to drive, not the full range of QdrantStore behavior
    that file exercises. _embedder must still be mocked even though no
    test here asserts anything about it: scroll_all() calls
    ensure_collection() first, which probes the embedder for vector
    dimensions when the (mocked) client reports no existing collection."""
    store = QdrantStore("http://127.0.0.1:1", collection, probe=False)
    assert store.available is False
    store.available = True
    store._client = MagicMock()
    store._embedder = MagicMock()
    store._embedder.embed = MagicMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    return store


def _load_gc_script():
    import importlib.util
    import pathlib

    script_path = (pathlib.Path(__file__).parent.parent.parent
                  / "scripts" / "gc_memory.py")
    spec = importlib.util.spec_from_file_location("gc_memory", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_gc_candidates_flags_old_volatile_points():
    gc_memory = _load_gc_script()
    store = _mock_store()
    now = time.time()
    old_volatile = MagicMock(id="old_volatile",
                             payload={"content": "stale fact", "volatility": "volatile",
                                     "created_at": now - 200 * 86400.0})
    store._client.scroll = MagicMock(return_value=([old_volatile], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert len(candidates) == 1
    assert candidates[0][0] == "old_volatile"


def test_find_gc_candidates_spares_fresh_points():
    gc_memory = _load_gc_script()
    store = _mock_store()
    now = time.time()
    fresh = MagicMock(id="fresh_volatile",
                      payload={"content": "new fact", "volatility": "volatile",
                              "created_at": now - 1 * 86400.0})
    store._client.scroll = MagicMock(return_value=([fresh], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert candidates == []


def test_find_gc_candidates_never_flags_stable_points_regardless_of_age():
    """D-24's own reasoning: stable facts don't fade at all -- decay_factor
    returns exactly 1.0 for Volatility.STABLE regardless of age (see that
    function's docstring), so no threshold ever catches one."""
    gc_memory = _load_gc_script()
    store = _mock_store()
    now = time.time()
    ancient_stable = MagicMock(id="old_stable",
                               payload={"content": "old but stable", "volatility": "stable",
                                       "created_at": now - 5000 * 86400.0})
    store._client.scroll = MagicMock(return_value=([ancient_stable], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert candidates == []


def test_find_gc_candidates_defaults_missing_volatility_to_semi_stable():
    """Consistency with SemanticMemory.retrieve's own fallback for the
    same payload gap -- not a second, different guess."""
    gc_memory = _load_gc_script()
    store = _mock_store()
    now = time.time()
    # No "volatility" key at all in the payload. 500 days at the default
    # 90-day semi_stable half-life decays to 0.5**(500/90) ~= 0.011, well
    # past the 0.05 threshold (200 days, tried first, only decays to
    # ~0.214 -- still well ABOVE threshold, which is exactly why this
    # needed a real number check rather than an assumed one).
    untagged = MagicMock(id="untagged",
                         payload={"content": "no volatility tag",
                                 "created_at": now - 500 * 86400.0})
    store._client.scroll = MagicMock(return_value=([untagged], None))

    candidates = gc_memory.find_gc_candidates(
        store, half_life_semi=90.0, half_life_volatile=14.0, threshold=0.05, now=now)

    assert len(candidates) == 1
    assert candidates[0][0] == "untagged"
