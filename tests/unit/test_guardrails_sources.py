"""
tests/unit/test_guardrails_sources.py -- guardrails/sources.py (D-57).

The deterministic attribution pass. Its most important property is the one
tested FIRST: when there is no cited web evidence -- which is every run with
WEB_SEARCH_ENABLED false, i.e. the default -- the report must come back
BYTE-IDENTICAL, not merely similar.
"""

from research_agent.guardrails.sources import append_web_sources
from research_agent.state import Evidence, Volatility

REPORT = "# Findings\n\nRedis is single-threaded per shard [g1].\n"


def _web(goal_id="g1", url="https://example.org/a", domain="example.org",
         title="Redis internals", score=0.75):
    return Evidence(task_key="t1", goal_id=goal_id, source="web",
                    content=f"{title} — a snippet of the page text",
                    score=score, volatility=Volatility.VOLATILE,
                    url=url, domain=domain)


def _corpus(goal_id="g1"):
    return Evidence(task_key="t1", goal_id=goal_id, source="corpus",
                    content="A real ingested document", score=0.9)


# ---------------------------------------------------------------------------
# The no-op paths -- the common case today
# ---------------------------------------------------------------------------


def test_no_evidence_at_all_returns_the_report_byte_identical():
    out, counters = append_web_sources(REPORT, [])
    assert out == REPORT
    assert counters["web_sources_listed"] == 0.0


def test_no_web_evidence_returns_the_report_byte_identical():
    """Every run with WEB_SEARCH_ENABLED false takes this path. A trailing
    newline difference here would be a visible diff in every report the
    system has ever produced."""
    out, _ = append_web_sources(REPORT, [_corpus(), _corpus("g2")])
    assert out == REPORT


def test_web_evidence_with_no_url_is_ignored():
    """A url is the entire point of the section; an item without one has
    nothing to link to."""
    item = _web()
    item = item.model_copy(update={"url": None})
    out, _ = append_web_sources(REPORT, [item])
    assert out == REPORT


def test_web_evidence_for_an_uncited_goal_is_not_listed():
    """A Sources list is a claim about what backed THIS report. Listing a
    page the compiler never drew on is misattribution -- the reader
    reasonably assumes a listed source supported something they read."""
    out, counters = append_web_sources(REPORT, [_web(goal_id="g9")])
    assert out == REPORT
    assert counters["web_sources_suppressed"] == 1.0
    assert counters["web_sources_listed"] == 0.0


# ---------------------------------------------------------------------------
# The section itself
# ---------------------------------------------------------------------------


def test_a_cited_web_source_is_appended():
    out, counters = append_web_sources(REPORT, [_web()])
    assert "## Sources" in out
    assert "https://example.org/a" in out
    assert "Redis internals" in out
    assert counters["web_sources_listed"] == 1.0


def test_the_prose_above_is_left_completely_untouched():
    """D-40's [gN]-only prose rule is NOT relaxed by this pass. The section
    sits below the report; nothing above it changes."""
    out, _ = append_web_sources(REPORT, [_web()])
    assert out.startswith(REPORT.rstrip())
    prose = out.split("## Sources")[0]
    assert "https://" not in prose


def test_only_the_title_half_of_the_content_becomes_the_link_label():
    """make_web_search_tool stores "Title — snippet"; only the title is
    wanted as a label, since the snippet already informed the prose."""
    out, _ = append_web_sources(REPORT, [_web(title="PLA modernization")])
    line = [ln for ln in out.splitlines() if ln.startswith("1.")][0]
    assert "PLA modernization" in line
    assert "a snippet of the page text" not in line


def test_the_domain_is_shown_alongside_the_label():
    out, _ = append_web_sources(REPORT, [_web(domain="arxiv.org")])
    assert "(arxiv.org)" in out


def test_the_goal_marker_is_carried_into_the_listing():
    """So a reader can tie a listed source back to the claim that cites it,
    using the same [gN] vocabulary the prose already uses."""
    report = "Claim [g2].\n"
    out, _ = append_web_sources(report, [_web(goal_id="g2")])
    assert "[g2]" in out.split("## Sources")[1]


def test_duplicate_urls_are_listed_once_keeping_the_best_score():
    """The same page can legitimately be returned under two goals; listing
    it twice would imply two independent sources."""
    report = "A [g1]. B [g2].\n"
    out, counters = append_web_sources(report, [
        _web(goal_id="g1", score=0.60, title="Lower"),
        _web(goal_id="g2", score=0.75, title="Higher")])
    assert counters["web_sources_listed"] == 1.0
    assert "Higher" in out and "Lower" not in out


def test_sources_are_ordered_best_first():
    """rank_to_score maps rank onto the band monotonically, so ordering by
    score means the engine's own ranking survives to the reader."""
    report = "A [g1].\n"
    out, _ = append_web_sources(report, [
        _web(url="https://c.com/", title="Third", score=0.60),
        _web(url="https://a.com/", title="First", score=0.75),
        _web(url="https://b.com/", title="Second", score=0.67)])
    body = out.split("## Sources")[1]
    assert body.index("First") < body.index("Second") < body.index("Third")


def test_an_empty_title_falls_back_to_the_domain_not_the_url():
    """A raw URL at typical search-result length is unreadable as a label."""
    item = _web().model_copy(update={"content": " — just a snippet"})
    out, _ = append_web_sources(REPORT, [item])
    line = [ln for ln in out.splitlines() if ln.startswith("1.")][0]
    assert "example.org" in line


def test_corpus_and_model_evidence_are_never_listed():
    """Corpus documents live in the operator's own ingested corpus and are
    addressable by their own identifiers, not URLs. Model recollection has
    no source to cite -- which is why D-49/D-51 hedge it instead."""
    model = Evidence(task_key="t1", goal_id="g1", source="model",
                     content="Recollected claim", score=0.6)
    out, counters = append_web_sources(REPORT, [_corpus(), model])
    assert out == REPORT
    assert counters["web_sources_listed"] == 0.0


def test_a_mixed_batch_lists_only_the_web_half():
    out, counters = append_web_sources(REPORT, [_corpus(), _web(), _corpus()])
    assert counters["web_sources_listed"] == 1.0
    assert out.count("https://") == 1


def test_the_section_ends_with_exactly_one_trailing_newline():
    """Deterministic whitespace: a heading landing one line below the last
    paragraph in one run and three in the next is a diff nobody wants."""
    out, _ = append_web_sources("Body [g1].\n\n\n\n", [_web()])
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
    assert "Body [g1].\n\n## Sources" in out

# ---------------------------------------------------------------------------
# D-59 — a listed source must actually be about the goal it is filed under
#
# Live regression: run p205.203-check. A drifted gather cycle retrieved web
# results about Redis monitoring and tagged them g1 -- a real,
# correctly-formed goal id. The report cited [g1] for its own (economic)
# reasons, so every one of those pages was listed as a source of an
# India-vs-US report. The prose was clean; the Sources section was not.
# ---------------------------------------------------------------------------
def _web_item(goal_id, content, url, score=0.7):
    from research_agent.state import Evidence
    return Evidence(task_key="t", goal_id=goal_id, source="web",
                    content=content, score=score, url=url,
                    domain=url.split("/")[2])


def test_off_topic_web_evidence_is_not_listed_as_a_source():
    from research_agent.guardrails.sources import append_web_sources
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1",
                  description="GDP growth inflation unemployment India US")]
    evidence = [
        _web_item("g1", "India vs United States Economy Comparison — compare GDP, "
                   "inflation and unemployment", "https://country-compare.com/x",
             score=0.75),
        _web_item("g1", "Redis Monitoring 101 — key metrics, alerts and "
                   "observability", "https://signoz.io/blog/redis-monitoring/",
             score=0.74),
    ]
    report = "Growth diverged over the period [g1].\n"
    out, counters = append_web_sources(report, evidence, goals)
    assert "country-compare.com" in out
    assert "signoz.io" not in out, "an off-topic page must not claim support"
    assert counters["web_sources_listed"] == 1
    assert counters["web_sources_suppressed"] == 1


def test_topical_gate_is_skipped_when_goals_are_not_supplied():
    """The pre-D-59 signature stays byte-compatible: without goals there is
    nothing to test topicality against, so cited-goal membership decides
    alone, exactly as before."""
    from research_agent.guardrails.sources import append_web_sources

    evidence = [_web_item("g1", "Redis Monitoring 101 — metrics",
                     "https://signoz.io/blog/redis-monitoring/")]
    out, counters = append_web_sources("Claim [g1].\n", evidence)
    assert "signoz.io" in out
    assert counters["web_sources_listed"] == 1


def test_a_goal_with_no_distinctive_terms_leaves_its_sources_alone():
    """An untestable claim is left alone rather than resolved against the
    item -- the same posture _sufficient takes when a query yields no
    distinctive terms."""
    from research_agent.guardrails.sources import append_web_sources
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="the of and")]  # all filler
    evidence = [_web_item("g1", "Anything at all", "https://example.com/a")]
    out, _ = append_web_sources("Claim [g1].\n", evidence, goals)
    assert "example.com" in out


# ---------------------------------------------------------------------------
# D-59 — count_listed_sources reads the artifact, not the accumulator
# ---------------------------------------------------------------------------
def test_count_listed_sources_counts_entries_in_the_shipped_report():
    from research_agent.guardrails.sources import (append_web_sources,
                                                   count_listed_sources)
    from research_agent.state import Goal

    goals = [Goal(goal_id="g1", description="economy growth India US")]
    evidence = [_web_item("g1", f"economy growth report {i}",
                          f"https://example{i}.com/a", score=0.9 - i / 100)
                for i in range(3)]
    out, counters = append_web_sources("Claim [g1].\n", evidence, goals)
    assert count_listed_sources(out) == 3 == counters["web_sources_listed"]


def test_count_listed_sources_is_zero_without_a_sources_section():
    """The path every WEB_SEARCH_ENABLED=false run takes."""
    from research_agent.guardrails.sources import count_listed_sources
    assert count_listed_sources("# Report\n\nJust prose [g1].\n") == 0
    # A report that merely mentions the words must not be miscounted.
    assert count_listed_sources("We list 1. [g1] nothing here\n") == 0
