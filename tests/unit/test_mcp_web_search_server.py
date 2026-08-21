"""
tests/unit/test_mcp_web_search_server.py -- scripts/mcp_web_search_server.py's
OWN wrapping logic (hits_for_query, web_search, _get_provider), plus the MCP
wire shape a `-> list[dict]` tool actually produces.

Mirrors tests/unit/test_mcp_corpus_server.py in structure and intent: the
server's logic is exercised with a FAKE provider substituted into the
module's `_provider` global, never against a real search engine. This suite
makes no network call and does not require `ddgs` to be installed.

The one test that DOES spawn a real subprocess uses
tests/fixtures/mcp_web_search_echo_server.py, not the real server, for
exactly that reason -- see that fixture's docstring.
"""

import asyncio
import importlib.util
import pathlib
import sys
import threading
import time

import pytest

# mcp is an OPTIONAL extra (only reached when the web tier is enabled). Skip
# rather than fail on a minimal install -- same posture as
# tests/unit/test_mcp_corpus_server.py and tests/unit/test_gc_memory.py.
#
# NOTE what is deliberately NOT skipped on: `ddgs`. The server module guards
# its own eager ddgs import (see its "First-import gotcha" docstring), so
# this entire file runs without that package present. If a future change
# makes ddgs mandatory just to IMPORT the server, this suite will fail on a
# minimal install -- which is the intended alarm, not a nuisance.
pytest.importorskip("mcp")

from research_agent.config import Settings  # noqa: E402
from research_agent.websearch import WebResult  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def _load_server():
    """Load scripts/mcp_web_search_server.py as a module, by path.

    Same loader shape as test_mcp_corpus_server.py's: the scripts/ directory
    is not a package, so the file is loaded from its resolved path rather
    than imported by name.
    """
    script_path = REPO_ROOT / "scripts" / "mcp_web_search_server.py"
    spec = importlib.util.spec_from_file_location(
        "mcp_web_search_server", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProvider:
    """Stand-in for a real SearchProvider -- fixed canned results, no I/O.

    Records how many times it was constructed and searched, so the
    singleton/concurrency tests below can assert on both.
    """

    builds = 0

    def __init__(self, results=None, raises=None, delay=0.0):
        FakeProvider.builds += 1
        self._results = results
        self._raises = raises
        self._delay = delay
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self._delay:
            time.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        if self._results is not None:
            return list(self._results)
        return [
            WebResult(title=f"T{i}", url=f"https://s{i}.example.com/p",
                      snippet=f"snippet {i}", rank=i + 1, engine="fake")
            for i in range(max_results)
        ]


@pytest.fixture
def server(monkeypatch):
    """The server module with a deterministic Settings and no real provider.

    get_settings is patched rather than relying on the ambient environment,
    for the reason tests/conftest.py's `settings` fixture documents at
    length: Settings(_env_file=None, ...) does NOT insulate against real OS
    environment variables, and a developer who exported WEB_SEARCH_* earlier
    in the same shell would otherwise silently change what these tests
    assert.
    """
    module = _load_server()
    settings = Settings(_env_file=None, min_evidence_score=0.5,
                        web_search_max_results=5, web_search_min_score=0.60,
                        web_search_max_score=0.75, web_search_max_per_domain=2)
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    module._provider = None
    FakeProvider.builds = 0
    return module


# ---------------------------------------------------------------------------
# Import-time behaviour
# ---------------------------------------------------------------------------


def test_importing_the_server_builds_no_provider_and_opens_no_client():
    """Regression guard, matching the corpus server's own: importing this
    module must not construct a provider (and so must not construct an HTTP
    client, or resolve WEB_SEARCH_PROVIDER at all). _provider must start
    None.

    The wall-clock threshold is deliberately loose, for the same reason the
    corpus server's is: this module eagerly IMPORTS ddgs when present (the
    documented first-import stall fix), which genuinely costs real time. The
    test still fails fast if that cost balloons far past what a package
    import should be.
    """
    t0 = time.time()
    module = _load_server()
    elapsed = time.time() - t0
    assert module._provider is None
    assert elapsed < 30.0, f"import took {elapsed:.1f}s -- far past a plain import"


def test_the_server_imports_without_ddgs_installed(monkeypatch):
    """The try/except around the eager ddgs import is what lets this whole
    suite run on a minimal install. If someone 'simplifies' it to a bare
    import, this fails."""
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "ddgs" or name.startswith("ddgs."):
            raise ImportError("ddgs is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    module = _load_server()
    assert module.ddgs is None
    assert module._provider is None


# ---------------------------------------------------------------------------
# hits_for_query -- the entire implementation
# ---------------------------------------------------------------------------


def test_hits_for_query_returns_the_documented_payload_keys(server):
    server._provider = FakeProvider()
    payload = server.hits_for_query("redis vs memcached", max_results=3)
    assert len(payload) == 3
    for item in payload:
        assert set(item) == {"title", "url", "snippet", "rank", "engine",
                             "domain", "score"}


def test_hits_for_query_scores_descend_with_rank(server):
    server._provider = FakeProvider()
    payload = server.hits_for_query("q", max_results=4)
    scores = [p["score"] for p in payload]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 4


def test_every_returned_score_sits_inside_the_configured_band(server):
    """The band invariant, enforced at the boundary that actually emits it.
    Below the floor and the D-17 coverage gate rejects a genuinely retrieved
    result; above the ceiling and a snippet outranks a fused corpus hit,
    inverting D-38."""
    server._provider = FakeProvider()
    payload = server.hits_for_query("q", max_results=5)
    assert all(0.60 <= p["score"] <= 0.75 for p in payload)
    assert payload[0]["score"] == pytest.approx(0.75)
    assert payload[-1]["score"] == pytest.approx(0.60)


def test_hits_for_query_defaults_max_results_from_settings(server):
    server._provider = FakeProvider()
    server.hits_for_query("q")
    assert server._provider.calls == [("q", 5)]


def test_hits_for_query_clamps_an_oversized_request_instead_of_failing(server):
    """A caller asking for 500 results is a caller bug, but failing the whole
    search over it is worse than quietly serving the configured maximum."""
    server._provider = FakeProvider()
    server.hits_for_query("q", max_results=500)
    assert server._provider.calls == [("q", 5)]


def test_hits_for_query_clamps_a_nonpositive_request_to_one(server):
    server._provider = FakeProvider()
    server.hits_for_query("q", max_results=0)
    assert server._provider.calls == [("q", 1)]


def test_duplicate_urls_are_dropped_before_scoring(server):
    server._provider = FakeProvider(results=[
        WebResult(title="A", url="https://a.com/x", snippet="s", rank=1, engine="f"),
        WebResult(title="B", url="https://a.com/x", snippet="s", rank=2, engine="f"),
        WebResult(title="C", url="https://b.com/y", snippet="s", rank=3, engine="f"),
    ])
    payload = server.hits_for_query("q", max_results=5)
    assert [p["url"] for p in payload] == ["https://a.com/x", "https://b.com/y"]


def test_one_domain_cannot_masquerade_as_several_sources(server):
    """The reason cap_by_domain exists: five hits from one site read to the
    compiler as five independent sources agreeing."""
    server._provider = FakeProvider(results=[
        WebResult(title=f"T{i}", url=f"https://seo.com/{i}", snippet="s",
                  rank=i + 1, engine="f") for i in range(5)
    ])
    payload = server.hits_for_query("q", max_results=5)
    assert len(payload) == 2, "web_search_max_per_domain=2 in this fixture"
    assert {p["domain"] for p in payload} == {"seo.com"}


def test_the_band_is_interpolated_across_survivors_not_raw_results(server):
    """Scoring runs AFTER both filters, deliberately. If it ran first, the
    worst SURVIVOR would not carry the floor and the spread would silently
    compress by however many duplicates the engine happened to return."""
    server._provider = FakeProvider(results=[
        WebResult(title="A", url="https://a.com/1", snippet="s", rank=1, engine="f"),
        WebResult(title="dup", url="https://a.com/1", snippet="s", rank=2, engine="f"),
        WebResult(title="dup2", url="https://a.com/1", snippet="s", rank=3, engine="f"),
        WebResult(title="B", url="https://b.com/1", snippet="s", rank=4, engine="f"),
    ])
    payload = server.hits_for_query("q", max_results=5)
    assert [p["score"] for p in payload] == [pytest.approx(0.75),
                                             pytest.approx(0.60)]


def test_ranks_are_renumbered_after_filtering_with_no_gaps(server):
    server._provider = FakeProvider(results=[
        WebResult(title="A", url="https://a.com/1", snippet="s", rank=1, engine="f"),
        WebResult(title="dup", url="https://a.com/1", snippet="s", rank=2, engine="f"),
        WebResult(title="B", url="https://b.com/1", snippet="s", rank=3, engine="f"),
    ])
    assert [p["rank"] for p in server.hits_for_query("q")] == [1, 2]


def test_domain_is_materialized_on_the_wire(server):
    """domain is a derived property on WebResult; the agent side receives
    plain JSON and has no object to ask, so it must be a real key."""
    server._provider = FakeProvider(results=[
        WebResult(title="A", url="https://www.Arxiv.org/abs/1", snippet="s",
                  rank=1, engine="f")])
    assert server.hits_for_query("q")[0]["domain"] == "arxiv.org"


def test_an_empty_result_set_is_an_empty_payload_not_an_error(server):
    """"The engine ran and found nothing" is a normal outcome -- the same way
    an empty corpus result is normal in tools/corpus_search.py."""
    server._provider = FakeProvider(results=[])
    assert server.hits_for_query("q") == []


def test_a_provider_failure_propagates_rather_than_reporting_no_results(server):
    """The load-bearing distinction. Returning [] on a broken engine would
    report a transport failure as "the web has no answer", and the retrieval
    ladder would quietly escalate to the model tier as if that were true."""
    server._provider = FakeProvider(raises=RuntimeError("ratelimited"))
    with pytest.raises(RuntimeError, match="ratelimited"):
        server.hits_for_query("q")


# ---------------------------------------------------------------------------
# Lazy singleton + concurrency
# ---------------------------------------------------------------------------


def test_get_provider_only_builds_once(server, monkeypatch):
    monkeypatch.setattr(server, "_build_provider", lambda: FakeProvider())
    first = server._get_provider()
    second = server._get_provider()
    assert first is second
    assert FakeProvider.builds == 1


def test_get_provider_builds_exactly_once_under_real_concurrent_load(server, monkeypatch):
    """The corpus server learned this the expensive way (one ~13s cold start
    became six competing ones). Here the cost of losing the lock is six HTTP
    clients and six simultaneous sessions arriving at one endpoint -- the
    shape most likely to be throttled."""
    def slow_build():
        time.sleep(0.05)
        return FakeProvider()

    monkeypatch.setattr(server, "_build_provider", slow_build)
    seen = []
    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()
        seen.append(server._get_provider())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeProvider.builds == 1
    assert len({id(p) for p in seen}) == 1


def test_concurrent_searches_do_not_serialize(server):
    """web_search is `async def` + a thread-pool offload. A plain `def`
    handler would have FastMCP call it inline on its single event loop,
    serializing every concurrent request -- and this codebase's normal
    gather-cycle fan-out is six tasks at once."""
    server._provider = FakeProvider(delay=0.15)

    async def run_six():
        return await asyncio.gather(*[
            server.web_search(f"q{i}", max_results=2) for i in range(6)])

    t0 = time.time()
    results = asyncio.run(run_six())
    elapsed = time.time() - t0

    assert len(results) == 6
    assert all(len(r) == 2 for r in results)
    # Six 0.15s calls summed is 0.9s. Genuinely concurrent should land near
    # 0.15s; the threshold is loose enough not to be flaky on a busy machine
    # while still failing outright if the calls serialize.
    assert elapsed < 0.6, f"6 concurrent calls took {elapsed:.2f}s -- serialized"


def test_the_async_tool_matches_hits_for_query(server):
    """web_search is a thin wrapper, not a reimplementation."""
    server._provider = FakeProvider()
    direct = server.hits_for_query("q", max_results=3)
    server._provider = FakeProvider()
    through_tool = asyncio.run(server.web_search("q", max_results=3))
    assert direct == through_tool


# ---------------------------------------------------------------------------
# The MCP wire shape, over a real stdio connection
# ---------------------------------------------------------------------------


def test_a_list_dict_tool_puts_results_under_structured_content_result():
    """Locks the shape the agent side will parse, verified against the
    ACTUALLY INSTALLED FastMCP rather than assumed -- the same standard
    scripts/mcp_corpus_server.py held itself to for its list[str] return.

    A `-> list[dict]` tool emits BOTH channels:
      structuredContent == {"result": [ {...}, ... ]}
      content           == one TextContent block per item, JSON-encoded

    structuredContent is what tools/mcp_client.py must read, because it is
    the only channel carrying `score` as a real number rather than as text
    to be re-parsed. If a future SDK changes this, the agent-side parser
    must change with it -- and this test is where that shows up first.
    """
    import json

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    fixture = REPO_ROOT / "tests" / "fixtures" / "mcp_web_search_echo_server.py"

    async def call():
        params = StdioServerParameters(
            command=sys.executable, args=[str(fixture)], env={})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(
                    "web_search", {"query": "hello", "max_results": 3})

    result = asyncio.run(call())

    assert getattr(result, "isError", False) is False
    structured = getattr(result, "structuredContent", None)
    assert isinstance(structured, dict)
    assert "result" in structured, "the list is wrapped under a 'result' key"

    items = structured["result"]
    assert len(items) == 3
    assert set(items[0]) == {"title", "url", "snippet", "rank", "engine",
                             "domain", "score"}
    assert [i["rank"] for i in items] == [1, 2, 3]
    assert items[0]["score"] > items[-1]["score"]
    assert isinstance(items[0]["score"], float), \
        "score must survive as a number, not become a string"

    # The text-block fallback: one block per item, each parseable as JSON.
    texts = [b.text for b in result.content if getattr(b, "text", None)]
    assert len(texts) == 3
    assert json.loads(texts[0])["url"] == items[0]["url"]
