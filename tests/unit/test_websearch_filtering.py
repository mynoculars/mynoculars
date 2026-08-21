"""
tests/unit/test_websearch_filtering.py — websearch/filtering.py.

The point of these two filters is not tidiness. Five hits from one site read
to the compiler as five independent sources agreeing -- corroboration that
does not exist -- and `web_search_results: 5` looks identical in telemetry
either way. These tests lock the behaviour that stops that.
"""

from research_agent.websearch import WebResult, cap_by_domain, dedupe_by_url


def _r(rank, url, title="T"):
    return WebResult(title=title, url=url, snippet="s", rank=rank,
                     engine="ddg_text")


# ---------------------------------------------------------------------------
# dedupe_by_url
# ---------------------------------------------------------------------------


def test_dedupe_keeps_the_first_occurrence_of_each_url():
    results = [_r(1, "https://a.com/x", "first"),
               _r(2, "https://b.com/y"),
               _r(3, "https://a.com/x", "duplicate")]
    out = dedupe_by_url(results)
    assert [r.url for r in out] == ["https://a.com/x", "https://b.com/y"]
    assert out[0].title == "first", "the better-ranked copy must survive"


def test_dedupe_reranks_the_survivors_with_no_gaps():
    results = [_r(1, "https://a.com/x"), _r(2, "https://a.com/x"),
               _r(3, "https://b.com/y")]
    assert [r.rank for r in dedupe_by_url(results)] == [1, 2]


def test_dedupe_treats_a_differing_query_string_as_a_different_page():
    """Exact URL match only, deliberately -- see the function's docstring.
    Two URLs differing by a query string are often genuinely different
    pages, and guessing wrong drops real evidence."""
    results = [_r(1, "https://a.com/p"), _r(2, "https://a.com/p?page=2")]
    assert len(dedupe_by_url(results)) == 2


def test_dedupe_on_an_empty_list_is_an_empty_list():
    assert dedupe_by_url([]) == []


# ---------------------------------------------------------------------------
# cap_by_domain
# ---------------------------------------------------------------------------


def test_cap_allows_at_most_n_per_domain():
    results = [_r(1, "https://seo.com/1"), _r(2, "https://seo.com/2"),
               _r(3, "https://seo.com/3"), _r(4, "https://other.org/1")]
    out = cap_by_domain(results, max_per_domain=2)
    assert [r.url for r in out] == ["https://seo.com/1", "https://seo.com/2",
                                    "https://other.org/1"]


def test_cap_keeps_the_best_ranked_hits_from_a_capped_domain():
    """Input is best-first, so the cap trims a dominant site's TAIL, never
    its best hit."""
    results = [_r(1, "https://seo.com/best"), _r(2, "https://seo.com/worse"),
               _r(3, "https://seo.com/worst")]
    out = cap_by_domain(results, max_per_domain=1)
    assert [r.url for r in out] == ["https://seo.com/best"]


def test_cap_collapses_www_and_bare_host_onto_one_domain():
    """registrable_domain strips a leading www., so these two are one site
    -- otherwise the cap is trivially defeated by a subdomain prefix."""
    results = [_r(1, "https://www.seo.com/1"), _r(2, "https://seo.com/2")]
    assert len(cap_by_domain(results, max_per_domain=1)) == 1


def test_cap_reranks_the_survivors_with_no_gaps():
    results = [_r(1, "https://a.com/1"), _r(2, "https://a.com/2"),
               _r(3, "https://b.com/1")]
    assert [r.rank for r in cap_by_domain(results, max_per_domain=1)] == [1, 2]


def test_cap_of_zero_or_less_disables_the_cap_entirely():
    """The documented way to reproduce uncapped behaviour deliberately --
    same posture as min_similarity=0.0 for pre-P2-01 retrieval."""
    results = [_r(i + 1, f"https://a.com/{i}") for i in range(5)]
    assert len(cap_by_domain(results, max_per_domain=0)) == 5
    assert len(cap_by_domain(results, max_per_domain=-1)) == 5


def test_cap_never_suppresses_results_whose_url_has_no_parseable_domain():
    """Grouping every unparseable URL under one empty-string key would let
    one malformed row suppress an unrelated malformed row -- a filter doing
    damage on data it does not understand."""
    results = [_r(1, "mailto:a@b"), _r(2, "mailto:c@d"),
               _r(3, "https://real.com/x")]
    out = cap_by_domain(results, max_per_domain=1)
    assert len(out) == 3


def test_cap_preserves_relative_order_across_domains():
    results = [_r(1, "https://a.com/1"), _r(2, "https://b.com/1"),
               _r(3, "https://a.com/2"), _r(4, "https://c.com/1")]
    out = cap_by_domain(results, max_per_domain=1)
    assert [r.url for r in out] == ["https://a.com/1", "https://b.com/1",
                                    "https://c.com/1"]


def test_the_two_filters_compose_in_either_readable_order():
    """Composition sanity: dedupe then cap is what the server does; the
    result must not depend on a subtle interaction between the two."""
    results = [_r(1, "https://a.com/1"), _r(2, "https://a.com/1"),
               _r(3, "https://a.com/2"), _r(4, "https://b.com/1")]
    out = cap_by_domain(dedupe_by_url(results), max_per_domain=1)
    assert [r.url for r in out] == ["https://a.com/1", "https://b.com/1"]
    assert [r.rank for r in out] == [1, 2]
