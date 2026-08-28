"""
tests/unit/test_analyze_runs.py -- scripts/analyze_runs.py's summarize()
(D-92).

Same shape as test_gc_memory.py and test_inspect_memory.py: the script is
standalone and operational, but its aggregation logic is a pure function
worth testing directly, so the module is loaded by file path. `load_runs`
is deliberately NOT tested here -- it is a single parameterised SELECT,
and testing it would mean either a live Postgres (the suite is offline by
design, D-33) or a mock of psycopg deep enough to be testing the mock.
"""

import importlib.util
import pathlib


def _load():
    script_path = (pathlib.Path(__file__).parent.parent.parent
                   / "scripts" / "analyze_runs.py")
    spec = importlib.util.spec_from_file_location("analyze_runs", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(**telemetry):
    return {"id": 1, "thread_id": "t", "query": "q", "recall": 1.0,
            "telemetry": telemetry, "created_at": "2026-08-27"}


def test_tier_answers_are_summed_across_runs():
    """The headline question this script exists for: which tier of the
    D-38 ladder actually answers, over a run series rather than one
    trace."""
    facts = _load().summarize([
        _run(tier_answers={"corpus": 4, "web": 2}),
        _run(tier_answers={"web": 6}),
    ])

    assert facts["tier_answers"] == {"web": 8, "corpus": 4}
    assert facts["runs"] == 2


def test_corpus_grounding_is_counted_per_run_not_per_goal():
    """"How often did the corpus ground ANYTHING" is a different question
    from "what was the mean corpus_recall" -- a series of near-misses and
    a series of total misses can share a mean."""
    facts = _load().summarize([
        _run(corpus_recall=0.0),
        _run(corpus_recall=0.5),
        _run(corpus_recall=1.0),
    ])

    assert facts["runs_with_any_corpus_grounding"] == 2
    assert facts["mean_corpus_recall"] == 0.5


def test_honesty_signals_are_counted():
    facts = _load().summarize([
        _run(grounding_notice_shipped=True, cited_figures_unsupported=2,
             escalations=[{"trigger": "E3", "action": "approve"}]),
        _run(grounding_notice_shipped=False, cited_figures_unsupported=0),
    ])

    assert facts["runs_shipping_provenance_notice"] == 1
    assert facts["runs_with_unsupported_figures"] == 1
    assert facts["runs_with_escalations"] == 1


def test_token_totals_only_count_runs_that_reported_them():
    """Rows written before D-86 carry no token fields at all. Averaging
    over ALL runs would quietly halve the mean for every history that
    spans the change."""
    facts = _load().summarize([
        _run(llm_prompt_tokens=4000, llm_completion_tokens=2000),
        _run(),  # a pre-D-86 row
    ])

    assert facts["token_runs"] == 1
    assert facts["mean_total_tokens_per_run"] == 6000


def test_rows_from_older_revisions_do_not_crash_the_report():
    """Every field is read with a default precisely so a cross-run report
    still works across a schema that grew -- which is exactly when history
    is most worth looking at."""
    facts = _load().summarize([_run(), _run(intent="Comparison")])

    assert facts["runs"] == 2
    assert facts["tier_answers"] == {}
    assert facts["mean_recall"] is None
    assert facts["mean_total_tokens_per_run"] is None
    assert facts["intents"] == {"Comparison": 1}


def test_an_empty_history_is_all_zeroes_not_an_error():
    facts = _load().summarize([])

    assert facts["runs"] == 0
    assert facts["mean_recall"] is None
    assert facts["tier_answers"] == {}


def test_main_reports_an_unreachable_postgres_as_exit_1(monkeypatch):
    """Same posture as gc_memory.py and inspect_memory.py: report and exit
    non-zero, never a traceback. Covers psycopg absent, server down, and
    table missing with one message."""
    analyze_runs = _load()
    monkeypatch.setattr(analyze_runs, "load_runs",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("connection refused")))

    assert analyze_runs.main([]) == 1


# ---------------------------------------------------------------------------
# D-104 -- failed runs (D-103) are separated before anything is counted
# ---------------------------------------------------------------------------


def _failed(failure_type="ProviderChainExhausted", **failure):
    """A D-103 failure row, in the shape cli.py::_failure_record writes."""
    failure.setdefault("message", "provider chain exhausted")
    return {"id": 2, "thread_id": "t", "query": "q", "recall": None,
            "telemetry": {"run_outcome": "failed",
                          "failure": {"type": failure_type, **failure}},
            "created_at": "2026-08-27"}


def test_a_failed_run_is_counted_but_never_aggregated():
    """The whole point of the split: a row with no telemetry must not
    dilute a rate computed from telemetry."""
    facts = _load().summarize([
        _run(corpus_recall=1.0, grounding_notice_shipped=True),
        _failed(),
    ])

    assert facts["runs"] == 2
    assert facts["completed_runs"] == 1
    assert facts["failed_runs"] == 1
    # 1 of 1 COMPLETED run, not 1 of 2 rows.
    assert facts["runs_with_any_corpus_grounding"] == 1
    assert facts["runs_shipping_provenance_notice"] == 1
    assert facts["mean_corpus_recall"] == 1.0


def test_failures_are_grouped_by_type():
    """'How often do we lose a run to provider exhaustion' is a count, and
    a count needs a field -- which is why _failure_record writes the
    exception type rather than only a formatted message."""
    facts = _load().summarize([
        _failed("ProviderChainExhausted"),
        _failed("ProviderChainExhausted"),
        _failed("GraphRecursionError"),
    ])

    assert facts["failed_runs"] == 3
    assert facts["failures_by_type"] == {"ProviderChainExhausted": 2,
                                         "GraphRecursionError": 1}


def test_provider_outcomes_are_counted_across_failed_runs():
    """A history where `primary` fails every time says something no single
    run can -- which is the reason D-101's chain is recorded at all."""
    facts = _load().summarize([
        _failed(chain=[["primary", "HTTPStatusError"],
                       ["mistral", "ReadTimeout"]]),
        _failed(chain=[["primary", "HTTPStatusError"],
                       ["mistral", "TruncatedGenerationError"]]),
    ])

    assert facts["failed_provider_outcomes"]["primary HTTPStatusError"] == 2
    assert facts["failed_provider_outcomes"]["mistral ReadTimeout"] == 1


def test_a_malformed_chain_entry_does_not_crash_the_report():
    """Same posture as every other field here: a row written by hand, or
    by a future revision, must not take down a history report."""
    facts = _load().summarize([_failed(chain=[["primary"], "nonsense", None])])

    assert facts["failed_runs"] == 1
    assert facts["failed_provider_outcomes"] == {}


def test_a_history_with_no_failures_reports_zero_not_absence():
    facts = _load().summarize([_run(recall=1.0)])

    assert facts["failed_runs"] == 0
    assert facts["completed_runs"] == 1
    assert facts["failures_by_type"] == {}


def test_rows_written_before_D_103_classify_as_completed():
    """`run_outcome` absent means completed, which is already true of the
    entire pre-D-103 history. This is the property that lets one rule
    classify old and new rows alike."""
    module = _load()

    assert module.is_failed({}) is False
    assert module.is_failed(None) is False
    assert module.is_failed({"recall": 0.0}) is False
    assert module.is_failed({"run_outcome": "failed"}) is True


# ---------------------------------------------------------------------------
# D-105 -- 14.6's follow-up: a report that cited no web source at all
# ---------------------------------------------------------------------------


def test_a_run_citing_no_web_source_while_suppressing_some_is_flagged():
    """Run p205.253-check carried web_sources_listed 0 against
    web_sources_suppressed 78 the whole time, and nobody read it."""
    facts = _load().summarize([
        _run(web_sources_listed=0, web_sources_suppressed=78),
        _run(web_sources_listed=29, web_sources_suppressed=1),
    ])

    assert facts["runs_listing_no_cited_web_sources"] == 1


def test_a_run_with_no_web_evidence_at_all_is_not_flagged():
    """0 listed and 0 suppressed is a corpus-only run, not the D-99 shape.
    Flagging it would make the counter fire on every offline run and mean
    nothing."""
    facts = _load().summarize([
        _run(web_sources_listed=0, web_sources_suppressed=0),
        _run(),  # a row predating the counters entirely
    ])

    assert facts["runs_listing_no_cited_web_sources"] == 0
