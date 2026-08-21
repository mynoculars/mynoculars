"""
tests/unit/test_websearch_provider.py — websearch/provider.py's types and
normalization, plus websearch/ddgs_provider.py's wrapping logic.

Fully offline, like every other test in this suite (see tests/conftest.py's
module docstring). No network, and no requirement that `ddgs` be installed:
DDGSProvider is exercised against a FAKE ddgs module injected into
sys.modules, which is what makes the "only this module imports ddgs" claim
in websearch/__init__.py testable rather than merely asserted.
"""

import sys
import types

import pytest

from research_agent.websearch import (
    WebResult,
    as_payload,
    coerce_results,
    registrable_domain,
)


# ---------------------------------------------------------------------------
# WebResult
# ---------------------------------------------------------------------------


def test_web_result_forbids_extra_fields():
    """D-29's extra="forbid" posture applies here too: a typo'd field name
    must fail at construction, not create a silently-wrong object."""
    with pytest.raises(Exception):
        WebResult(title="t", url="https://a.com/x", snippet="s", rank=1,
                  engine="ddg_text", scoer=0.7)  # noqa - deliberate typo


def test_web_result_rank_must_be_one_based():
    """rank=0 is rejected. The whole scoring formula reads rank as a 1-based
    ordinal (scoring.rank_to_score); admitting 0 would let an off-by-one
    reach the interpolation silently."""
    with pytest.raises(Exception):
        WebResult(title="t", url="https://a.com/x", snippet="s", rank=0,
                  engine="ddg_text")


def test_web_result_domain_is_derived_not_stored():
    r = WebResult(title="t", url="https://www.Example.com:8443/a/b?q=1",
                  snippet="s", rank=1, engine="ddg_text")
    assert r.domain == "example.com"


# ---------------------------------------------------------------------------
# registrable_domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,expected", [
    ("https://arxiv.org/abs/1234", "arxiv.org"),
    ("https://www.bbc.co.uk/news", "bbc.co.uk"),
    ("http://EXAMPLE.COM/Path", "example.com"),
    ("https://user:pw@secure.example.org/x", "secure.example.org"),
    ("https://host.example.net:8080/", "host.example.net"),
    ("", ""),
    ("not-a-url", ""),
])
def test_registrable_domain_normalizes(url, expected):
    assert registrable_domain(url) == expected


def test_registrable_domain_returns_empty_instead_of_raising_on_junk():
    """A malformed URL inside a third-party response is a data problem, not
    a reason to fail the whole search -- see the function's docstring."""
    assert registrable_domain("https://[unclosed") == ""


# ---------------------------------------------------------------------------
# coerce_results
# ---------------------------------------------------------------------------


def test_coerce_results_assigns_one_based_ranks_in_order():
    raw = [{"title": "A", "href": "https://a.com/1", "body": "aa"},
           {"title": "B", "href": "https://b.com/2", "body": "bb"},
           {"title": "C", "href": "https://c.com/3", "body": "cc"}]
    out = coerce_results(raw, engine="ddg_text")
    assert [r.rank for r in out] == [1, 2, 3]
    assert [r.title for r in out] == ["A", "B", "C"]
    assert all(r.engine == "ddg_text" for r in out)


def test_coerce_results_drops_rows_with_no_url():
    raw = [{"title": "A", "href": "https://a.com/1", "body": "aa"},
           {"title": "no url", "href": "", "body": "bb"},
           {"title": "C", "href": "https://c.com/3", "body": "cc"}]
    out = coerce_results(raw, engine="ddg_text")
    assert [r.title for r in out] == ["A", "C"]


def test_coerce_results_drops_rows_with_neither_title_nor_snippet():
    """Nothing to cite and nothing to read -- carrying it would produce an
    Evidence item with empty content occupying a compile-prompt slot."""
    raw = [{"title": "", "href": "https://a.com/1", "body": ""},
           {"title": "B", "href": "https://b.com/2", "body": ""}]
    out = coerce_results(raw, engine="ddg_text")
    assert [r.title for r in out] == ["B"]


def test_coerce_results_reranks_after_dropping_so_there_is_no_gap():
    """Rule 2 in coerce_results' docstring: a dropped row must not leave a
    hole in the ranking, or scoring.rank_to_score interpolates against a
    total that does not match the items being scored."""
    raw = [{"title": "", "href": "", "body": ""},
           {"title": "B", "href": "https://b.com/2", "body": "bb"},
           {"title": "", "href": "", "body": ""},
           {"title": "D", "href": "https://d.com/4", "body": "dd"}]
    out = coerce_results(raw, engine="ddg_text")
    assert [r.rank for r in out] == [1, 2]


def test_coerce_results_collapses_whitespace_in_scraped_text():
    raw = [{"title": "  A   long\n title ", "href": "https://a.com/1",
            "body": "line one\n\n   line two  "}]
    out = coerce_results(raw, engine="ddg_text")
    assert out[0].title == "A long title"
    assert out[0].snippet == "line one line two"


def test_coerce_results_accepts_both_href_body_and_url_snippet_key_names():
    """DDGS uses href/body; a keyed API is far more likely to use
    url/snippet. Accepting both keeps a second provider from needing its
    own copy of this normalization."""
    raw = [{"title": "A", "url": "https://a.com/1", "snippet": "aa"}]
    out = coerce_results(raw, engine="other")
    assert out[0].url == "https://a.com/1" and out[0].snippet == "aa"


def test_coerce_results_enforces_max_results_even_if_the_engine_overshoots():
    raw = [{"title": f"T{i}", "href": f"https://s{i}.com/", "body": "b"}
           for i in range(10)]
    out = coerce_results(raw, engine="ddg_text", max_results=3)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# as_payload -- the cross-process wire shape
# ---------------------------------------------------------------------------


def test_as_payload_materializes_domain_for_the_process_boundary():
    """domain is a PROPERTY on WebResult but must be a real key on the wire:
    the agent side receives plain JSON and has no object to ask."""
    r = WebResult(title="T", url="https://www.arxiv.org/abs/1", snippet="s",
                  rank=2, engine="ddg_text")
    payload = as_payload(r, score=0.68)
    assert payload == {"title": "T", "url": "https://www.arxiv.org/abs/1",
                       "snippet": "s", "rank": 2, "engine": "ddg_text",
                       "domain": "arxiv.org", "score": 0.68}


def test_as_payload_keys_are_the_contract_with_the_agent_side():
    """Locked deliberately. These key names are what tools/mcp_client.py
    will parse out of structuredContent; changing one here without changing
    it there is exactly the drift this assertion exists to catch."""
    r = WebResult(title="T", url="https://a.com/", snippet="s", rank=1,
                  engine="e")
    assert set(as_payload(r, 0.7)) == {
        "title", "url", "snippet", "rank", "engine", "domain", "score"}


# ---------------------------------------------------------------------------
# DDGSProvider -- against a fake ddgs module, never the real one
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ddgs(monkeypatch):
    """Install a stand-in `ddgs` module in sys.modules.

    Deliberately NOT pytest.importorskip("ddgs"): this suite must exercise
    DDGSProvider's own wrapping logic on a minimal install where the real
    package is absent, and must never make a network call even where it is
    present. The fake records the kwargs it was called with so the tests
    below can assert the region/safesearch/max_results plumbing.
    """
    calls = []

    class FakeDDGS:
        def __init__(self, timeout=None):
            self.timeout = timeout
            calls.append(("init", {"timeout": timeout}))

        def text(self, query, region=None, safesearch=None, max_results=None):
            calls.append(("text", {"query": query, "region": region,
                                   "safesearch": safesearch,
                                   "max_results": max_results}))
            return [{"title": f"R{i}", "href": f"https://s{i}.example.com/p",
                     "body": f"snippet {i}"} for i in range(max_results or 3)]

    module = types.ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    return calls


def test_ddgs_provider_returns_ranked_web_results(fake_ddgs):
    from research_agent.websearch import DDGSProvider

    provider = DDGSProvider(region="in-en", safesearch="moderate")
    results = provider.search("redis vs memcached", max_results=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert all(r.engine == "ddg_text" for r in results)


def test_ddgs_provider_passes_region_and_safesearch_through(fake_ddgs):
    from research_agent.websearch import DDGSProvider

    DDGSProvider(region="in-en", safesearch="off",
                 timeout_seconds=7.5).search("q", max_results=2)
    init_kwargs = next(kw for name, kw in fake_ddgs if name == "init")
    text_kwargs = next(kw for name, kw in fake_ddgs if name == "text")
    assert init_kwargs["timeout"] == 7.5
    assert text_kwargs["region"] == "in-en"
    assert text_kwargs["safesearch"] == "off"
    assert text_kwargs["max_results"] == 2


def test_ddgs_provider_returns_empty_for_a_blank_query_rather_than_raising(fake_ddgs):
    """An empty query is an upstream slip. Raising would turn it into a
    D-16 task failure; returning [] lets the ladder escalate, which is the
    right outcome either way."""
    from research_agent.websearch import DDGSProvider

    assert DDGSProvider().search("   ", max_results=5) == []


def test_ddgs_provider_does_not_catch_engine_failures(monkeypatch):
    """provider.SearchProvider.search's contract: [] means "ran, found
    nothing"; an exception means "could not run" and must propagate. A
    provider that swallows its own errors makes those two states
    indistinguishable -- the same defect the old hardcoded MCP score=1.0
    and MIN_EVIDENCE_SCORE=0.0 both were."""
    class ExplodingDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, *a, **kw):
            raise RuntimeError("ratelimited")

    module = types.ModuleType("ddgs")
    module.DDGS = ExplodingDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)

    from research_agent.websearch import DDGSProvider

    with pytest.raises(RuntimeError, match="ratelimited"):
        DDGSProvider().search("q", max_results=3)


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------


def test_build_provider_constructs_ddgs(fake_ddgs):
    from research_agent.websearch import DDGSProvider, build_provider

    assert isinstance(build_provider("ddgs"), DDGSProvider)
    assert isinstance(build_provider("  DDGS  "), DDGSProvider)


def test_build_provider_rejects_an_unknown_name_instead_of_falling_back(fake_ddgs):
    """A silent fallback would mean WEB_SEARCH_PROVIDER=tavily on a build
    with no Tavily support quietly runs DuckDuckGo -- the same silent
    misconfiguration config.py::warn_on_likely_env_typos exists to kill."""
    from research_agent.websearch import build_provider

    with pytest.raises(ValueError, match="Unknown WEB_SEARCH_PROVIDER"):
        build_provider("tavily")


# ---------------------------------------------------------------------------
# The package-level import contract
# ---------------------------------------------------------------------------


def test_importing_the_package_does_not_require_ddgs(monkeypatch):
    """The load-bearing claim in websearch/__init__.py: everything except
    ddgs_provider.py must import on a minimal install. Simulated by hiding
    ddgs from sys.modules and blocking its import outright."""
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "ddgs" or name.startswith("ddgs."):
            raise ImportError("ddgs is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "ddgs", None)
    monkeypatch.delitem(sys.modules, "ddgs")
    monkeypatch.setattr(builtins, "__import__", blocking_import)

    import importlib

    module = importlib.reload(
        importlib.import_module("research_agent.websearch"))
    assert module.WebResult is not None
    assert module.rank_to_score(1, 1, 0.6, 0.75) == 0.75


def test_package_getattr_still_raises_on_a_genuine_typo():
    import research_agent.websearch as ws

    with pytest.raises(AttributeError):
        ws.WebReslt  # noqa - deliberate typo
