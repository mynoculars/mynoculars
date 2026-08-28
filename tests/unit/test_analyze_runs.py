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
