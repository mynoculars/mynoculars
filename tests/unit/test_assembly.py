"""
tests/unit/test_assembly.py -- assembly.py::build_app_and_settings, the
project's one dependency-graph constructor.

Covers ONLY the conditional wiring branches (mcp_enabled, web_search_enabled)
and the returned AppBundle's shape -- NOT genuine connectivity to Qdrant/
OpenSearch/Postgres, which each storage module's own test file already
covers for its fail-open behavior. The heavy I/O constructors are
monkeypatched here so this stays fast and offline; get_checkpointer in
particular retries against a real socket for several seconds if left
unpatched, which is why this was previously untested at the unit level --
only ever exercised end-to-end via the CLI/API themselves.
"""

from research_agent import assembly
from research_agent.config import Settings


class _FakeGraph:
    def get_state(self, config):
        return None


def _patch_heavy_deps(monkeypatch, mcp_enabled=False, web_search_enabled=False):
    """Replace every constructor build_app_and_settings calls that would
    otherwise touch a real socket or spawn a real subprocess, with the
    cheapest fake that satisfies its call signature."""
    monkeypatch.setattr(assembly, "get_settings",
                        lambda: Settings(_env_file=None, llm_mode="stub",
                                        mcp_enabled=mcp_enabled,
                                        web_search_enabled=web_search_enabled))
    monkeypatch.setattr(assembly, "QdrantStore", lambda *a, **k: object())
    monkeypatch.setattr(assembly, "OpenSearchStore", lambda *a, **k: object())
    monkeypatch.setattr(assembly, "get_checkpointer", lambda dsn: (object(), False))
    monkeypatch.setattr(assembly, "build_graph", lambda *a, **k: _FakeGraph())
    if mcp_enabled or web_search_enabled:
        monkeypatch.setattr(assembly, "MCPBridge", lambda *a, **k: object())


def test_mcp_disabled_by_default_builds_no_mcp_wiring(monkeypatch):
    _patch_heavy_deps(monkeypatch, mcp_enabled=False)
    bundle = assembly.build_app_and_settings()
    assert bundle.mcp_bridge is None


def test_mcp_enabled_builds_an_mcp_bridge(monkeypatch):
    _patch_heavy_deps(monkeypatch, mcp_enabled=True)
    bundle = assembly.build_app_and_settings()
    assert bundle.mcp_bridge is not None


def test_web_search_disabled_by_default_builds_no_web_bridge(monkeypatch):
    _patch_heavy_deps(monkeypatch, web_search_enabled=False)
    bundle = assembly.build_app_and_settings()
    assert bundle.web_mcp_bridge is None


def test_web_search_enabled_builds_a_second_independent_bridge(monkeypatch):
    """D-57: a SEPARATE MCPBridge from the corpus one, not a reuse --
    both fields must be independently settable."""
    _patch_heavy_deps(monkeypatch, mcp_enabled=True, web_search_enabled=True)
    bundle = assembly.build_app_and_settings()
    assert bundle.mcp_bridge is not None
    assert bundle.web_mcp_bridge is not None
    assert bundle.mcp_bridge is not bundle.web_mcp_bridge


def test_returned_bundle_carries_every_field_by_name():
    """P2-08: durable/checkpointer/router must all be reachable by name
    on the returned bundle -- the exact gap that used to make
    close_checkpointer's cleanup unreachable from either caller."""
    from research_agent.assembly import AppBundle
    bundle = AppBundle(app=object(), settings=object(), durable=False,
                       checkpointer=object())
    assert bundle.mcp_bridge is None       # has a default
    assert bundle.web_mcp_bridge is None   # has a default
    assert bundle.router is None           # has a default
    assert bundle.durable is False


def test_build_app_and_settings_returns_the_durable_flag_from_the_checkpointer(monkeypatch):
    _patch_heavy_deps(monkeypatch)
    monkeypatch.setattr(assembly, "get_checkpointer", lambda dsn: (object(), True))
    bundle = assembly.build_app_and_settings()
    assert bundle.durable is True
