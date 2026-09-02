"""
tests/unit/test_inspect_memory.py -- scripts/inspect_memory.py's
summarize() (D-90).

Same shape and same reasoning as test_gc_memory.py: inspect_memory.py is
a standalone operational script, but its actual aggregation logic is a
pure function worth testing directly, so the module is loaded by file
path. The --query path is deliberately NOT re-tested here -- it delegates
wholesale to SemanticMemory.retrieve, which tests/unit/test_memory_semantic.py
already covers, and re-asserting its ranking here would be a second copy
of that contract free to drift from the first.
"""

import importlib.util
import time
from unittest.mock import MagicMock

import pytest

# qdrant-client is an OPTIONAL runtime dependency (storage/qdrant_store.py
# imports it lazily so the rest of the codebase runs without it), so a
# minimal install should SKIP these rather than report a missing optional
# extra as a broken suite.
pytest.importorskip("qdrant_client")

from research_agent.storage.qdrant_store import QdrantStore


def _mock_store(points):
    """A QdrantStore whose scroll_all() yields exactly `points`.

    summarize() only ever calls scroll_all(), so this stubs that directly
    rather than driving the underlying client's pagination -- which
    test_storage_qdrant_store.py already covers on its own.
    """
    store = QdrantStore("http://127.0.0.1:1", "test_collection", probe=False)
    assert store.available is False
    store.available = True
    store._client = MagicMock()
    store._embedder = MagicMock()
    store.scroll_all = lambda: iter(points)
    return store


def _load():
    # D-157: the implementation moved into the package
    # (research_agent.ops.inspect_memory); scripts/ now holds a thin
    # launcher, and loading THAT would exercise a six-line shim.
    # find_spec locates the module WITHOUT executing it, and the fresh
    # module object below is deliberate: several tests here assert on
    # module-level caching, which a shared sys.modules entry would carry
    # from one test into the next.
    origin = importlib.util.find_spec("research_agent.ops.inspect_memory").origin
    spec = importlib.util.spec_from_file_location(
        "inspect_memory", origin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_counts_points_volatility_and_source_queries():
    now = time.time()
    store = _mock_store([
        {"id": "1", "content": "a", "volatility": "semi_stable",
         "source_query": "Compare Redis and Memcached",
         "created_at": now - 86400},
        {"id": "2", "content": "b", "volatility": "semi_stable",
         "source_query": "Compare Redis and Memcached",
         "created_at": now - 2 * 86400},
        {"id": "3", "content": "c", "volatility": "volatile",
         "source_query": "Compare Armies of China and India",
         "created_at": now - 10 * 86400},
    ])

    facts = _load().summarize(store, now=now)

    assert facts["points"] == 3
    assert facts["by_volatility"] == {"semi_stable": 2, "volatile": 1}
    assert facts["distinct_source_queries"] == 2
    assert facts["top_source_queries"][0] == ("Compare Redis and Memcached", 2)
    assert facts["newest_days"] == 1.0
    assert facts["oldest_days"] == 10.0


def test_summarize_defaults_missing_volatility_to_semi_stable():
    """The same fallback SemanticMemory.retrieve and gc_memory.py both
    apply to the identical payload gap -- one behaviour for a missing
    field, not a third separate guess about what an untagged item is."""
    now = time.time()
    store = _mock_store([{"id": "1", "content": "a", "created_at": now}])

    facts = _load().summarize(store, now=now)

    assert facts["by_volatility"] == {"semi_stable": 1}


def test_summarize_of_an_empty_collection_is_all_zeroes_not_an_error():
    """A fresh install has written nothing yet (memory only accepts a
    PASSED critique, D-24). That is a normal state, not a failure, and
    must not divide by zero on the age range."""
    facts = _load().summarize(_mock_store([]), now=time.time())

    assert facts["points"] == 0
    assert facts["by_volatility"] == {}
    assert facts["oldest_days"] == 0.0
    assert facts["newest_days"] == 0.0


def test_summarize_tolerates_a_point_with_no_created_at():
    """Treated as "written just now" rather than raising -- consistent
    with gc_memory.py's identical `point.get("created_at", now)` default
    for the same gap."""
    now = time.time()
    store = _mock_store([{"id": "1", "content": "a", "source_query": "q"}])

    facts = _load().summarize(store, now=now)

    assert facts["points"] == 1
    assert facts["oldest_days"] == 0.0


def test_main_reports_an_unreachable_qdrant_as_exit_1_not_a_crash():
    """Same posture as gc_memory.py and reset_stores.py: an unreachable
    store is reported and exits non-zero, never a traceback."""
    inspect_memory = _load()
    store = QdrantStore("http://127.0.0.1:1", "test_collection", probe=False)
    assert store.available is False
    inspect_memory.QdrantStore = lambda *a, **kw: store

    assert inspect_memory.main([]) == 1
