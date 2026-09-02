"""
tests/unit/test_eval_suite.py -- scripts/eval_suite.py's pure half (D-136).

Same shape as test_analyze_runs.py, test_gc_memory.py and
test_inspect_memory.py: the script is standalone and operational, but its
loading, grading and diffing logic are pure functions worth testing
directly, so the module is loaded by file path.

`run_case` and `main`'s live half are deliberately NOT tested here. They
need a real model and real stores, and D-33's rule is that this suite is
offline -- a golden-set harness that quietly needed four services to run
the offline suite would be exactly the thing it exists to prevent.
"""

import importlib.util
import json

import pytest


def _load():
    # D-157: the implementation moved into the package
    # (research_agent.ops.eval_suite); scripts/ now holds a thin
    # launcher, and loading THAT would exercise a six-line shim.
    # find_spec locates the module WITHOUT executing it, and the fresh
    # module object below is deliberate: several tests here assert on
    # module-level caching, which a shared sys.modules entry would carry
    # from one test into the next.
    origin = importlib.util.find_spec("research_agent.ops.eval_suite").origin
    spec = importlib.util.spec_from_file_location(
        "eval_suite", origin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, *lines):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_cases_reads_a_well_formed_set(tmp_path):
    mod = _load()
    path = _write(tmp_path,
                  json.dumps({"id": "a", "query": "q1"}),
                  "",
                  "# a comment line",
                  json.dumps({"id": "b", "query": "q2", "expect": {}}))

    cases = mod.load_cases(path)
    assert [c["id"] for c in cases] == ["a", "b"]


def test_load_cases_names_the_line_of_a_malformed_case(tmp_path):
    """A golden set with a typo must fail loudly at load time, not
    silently run seven of eight cases."""
    mod = _load()
    path = _write(tmp_path, json.dumps({"id": "a", "query": "q"}), "{not json")

    with pytest.raises(ValueError) as exc:
        mod.load_cases(path)
    assert ":2:" in str(exc.value)


def test_load_cases_requires_an_id_and_a_query(tmp_path):
    mod = _load()
    path = _write(tmp_path, json.dumps({"query": "q"}))

    with pytest.raises(ValueError) as exc:
        mod.load_cases(path)
    assert "'id'" in str(exc.value)


def test_load_cases_rejects_duplicate_ids(tmp_path):
    """Ids key the baseline file: two cases sharing one would make a
    baseline silently describe whichever ran last."""
    mod = _load()
    path = _write(tmp_path, json.dumps({"id": "a", "query": "q1"}),
                  json.dumps({"id": "a", "query": "q2"}))

    with pytest.raises(ValueError) as exc:
        mod.load_cases(path)
    assert "duplicate" in str(exc.value)


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


def _case(**expect):
    return {"id": "c", "query": "q", "expect": expect}


def test_a_band_passes_inside_and_fails_outside():
    mod = _load()

    assert mod.grade(_case(min_corpus_recall=0.5),
                     {"corpus_recall": 0.75})["passed"]
    assert not mod.grade(_case(min_corpus_recall=0.5),
                         {"corpus_recall": 0.25})["passed"]
    assert mod.grade(_case(max_corpus_recall=0.0),
                     {"corpus_recall": 0.0})["passed"]
    assert not mod.grade(_case(max_total_tokens=1000),
                         {"llm_total_tokens": 4000})["passed"]


def test_an_absent_telemetry_field_fails_rather_than_passes():
    """"Not measured" is not "fine" -- the exact confusion D-103 removed
    from the recall column. A run that never reached telemetry_node must
    not satisfy an expectation by omission."""
    mod = _load()
    result = mod.grade(_case(min_recall=0.5), {})

    assert not result["passed"]
    assert result["checks"][0]["actual"] is None


def test_expect_tiers_accepts_a_subset_and_rejects_an_outsider():
    mod = _load()
    allowed = _case(expect_tiers=["corpus", "mcp"])

    assert mod.grade(allowed, {"tier_answers": {"corpus": 6}})["passed"]
    assert mod.grade(allowed, {"tier_answers": {"corpus": 4, "mcp": 2}})["passed"]
    assert not mod.grade(allowed, {"tier_answers": {"web": 6}})["passed"]


def test_expect_tiers_fails_when_no_tier_answered_at_all():
    """A run where nothing is recorded as having answered is not a run
    that answered from the allowed set, whatever its recall says."""
    mod = _load()
    assert not mod.grade(_case(expect_tiers=["corpus"]),
                         {"tier_answers": {}, "recall": 1.0})["passed"]


def test_a_required_notice_is_checked_both_ways():
    """The off-corpus cases turn on this: the requirement is not that the
    run answers, it is that the report SAYS the corpus contributed
    nothing (D-85)."""
    mod = _load()
    want = _case(require_grounding_notice=True)

    assert mod.grade(want, {"grounding_notice_shipped": True})["passed"]
    assert not mod.grade(want, {"grounding_notice_shipped": False})["passed"]

    forbid = _case(require_grounding_notice=False)
    assert mod.grade(forbid, {"grounding_notice_shipped": False})["passed"]
    assert not mod.grade(forbid, {"grounding_notice_shipped": True})["passed"]


def test_a_case_with_no_expectations_passes_with_no_checks():
    """Useful while adding a query whose band you have not decided yet --
    it runs and reports numbers without asserting anything."""
    mod = _load()
    result = mod.grade({"id": "c", "query": "q"}, {"recall": 0.1})

    assert result["passed"] and result["checks"] == []


def test_every_failed_check_is_named_not_just_counted():
    mod = _load()
    result = mod.grade(_case(min_recall=0.9, max_total_tokens=10),
                       {"recall": 0.2, "llm_total_tokens": 99})

    missed = [c["name"] for c in result["checks"] if not c["ok"]]
    assert missed == ["min_recall", "max_total_tokens"]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_a_baseline_keeps_a_fixed_field_set():
    """A baseline that changes shape every time a telemetry field is
    added is a baseline nobody keeps."""
    mod = _load()
    entry = mod.baseline_entry({"recall": 1.0, "llm_total_tokens": 4023,
                                "some_new_field": "ignored"})

    assert set(entry) == set(mod._BASELINE_FIELDS)
    assert entry["recall"] == 1.0 and entry["llm_total_tokens"] == 4023


def test_the_diff_reports_only_what_moved():
    """A regression report that prints everything is a report nobody
    reads."""
    mod = _load()
    baseline = {"a": mod.baseline_entry({"recall": 1.0, "corpus_recall": 0.8,
                                         "llm_total_tokens": 4000})}
    results = [{"id": "a", "telemetry": {"recall": 1.0, "corpus_recall": 0.2,
                                         "llm_total_tokens": 4000}}]

    moved = mod.diff_against_baseline(baseline, results)

    assert [m["field"] for m in moved] == ["corpus_recall"]
    assert moved[0]["was"] == 0.8 and moved[0]["now"] == 0.2
    assert moved[0]["delta"] == -0.6


def test_the_diff_names_a_case_that_appeared_or_vanished():
    """A golden set that grew or shrank must say so, rather than silently
    comparing fewer cases than you think."""
    mod = _load()
    baseline = {"gone": mod.baseline_entry({"recall": 1.0})}
    results = [{"id": "fresh", "telemetry": {"recall": 1.0}}]

    moved = mod.diff_against_baseline(baseline, results)
    fields = {(m["id"], m["field"]) for m in moved}

    assert ("fresh", "(new)") in fields
    assert ("gone", "(missing)") in fields


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_the_report_names_each_missed_expectation():
    mod = _load()
    results = [{"id": "c", "telemetry": {"recall": 0.2},
                "grade": mod.grade(_case(min_recall=0.9), {"recall": 0.2})}]

    text = mod.format_report(results, graded=True)

    assert "[FAIL] c" in text
    assert "MISS min_recall: expected 0.9, got 0.2" in text


def test_the_report_says_when_grading_was_skipped():
    """Stub mode runs the harness and grades nothing; a reader must not
    mistake "ran" for "passed"."""
    mod = _load()
    text = mod.format_report([{"id": "c", "telemetry": {}}], graded=False)

    assert "grading SKIPPED (stub mode)" in text
    assert "[ran] c" in text
    assert "PASS" not in text


def test_an_errored_case_is_reported_as_an_error_not_a_result():
    mod = _load()
    text = mod.format_report(
        [{"id": "c", "error": "ConnectionError: backend down"}], graded=True)

    assert "[ERROR] c" in text
    assert "backend down" in text


# ---------------------------------------------------------------------------
# the shipped golden set itself
# ---------------------------------------------------------------------------


def test_the_shipped_golden_set_loads_and_states_why_each_case_exists():
    """The file ships with the repo and is the first thing a reader
    copies for their own corpus -- a case with no stated reason teaches
    them to write one the same way."""
    mod = _load()
    cases = mod.load_cases(mod.DEFAULT_GOLDEN)

    assert len(cases) >= 6
    for case in cases:
        assert case.get("why"), f"{case['id']} does not say why it exists"
        assert case.get("expect"), f"{case['id']} asserts nothing"


def test_every_expectation_in_the_shipped_set_is_one_the_grader_knows():
    """A typo'd expectation key would be silently ignored by grade() --
    the case would pass while asserting nothing at all."""
    mod = _load()
    known = ({name for name, _f, _k in mod._NUMERIC_CHECKS}
             | {"expect_tiers", "require_grounding_notice",
                "require_critique_passed", "require_truncation_notice"})

    for case in mod.load_cases(mod.DEFAULT_GOLDEN):
        unknown = set(case["expect"]) - known
        assert not unknown, f"{case['id']} uses unknown key(s): {unknown}"
