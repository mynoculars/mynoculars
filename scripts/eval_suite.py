"""
scripts/eval_suite.py -- the golden-set regression harness (D-136).

Purpose:
    Run a fixed set of queries against THIS deployment and check each
    run's telemetry against expectations written down in advance, so a
    change to a prompt, a threshold or the retrieval ladder shows up as a
    named, reproducible difference rather than as a feeling about the
    last report someone read.

WHY THIS EXISTS, stated against what already exists:
    The test suite is 900+ tests and entirely OFFLINE by design (D-33) --
    it proves the graph's mechanics and can say nothing about answer
    quality. `scripts/analyze_runs.py` (D-92) reads real runs but has no
    notion of what a run SHOULD have produced: it counts, and counting is
    deliberately not judging. Between them sat the gap this closes --
    every threshold in this project (min_similarity, min_evidence_score,
    grounded_recall_target, and now three Phase 6 budgets) is tuned
    against a handful of remembered runs, and nothing re-checks the ones
    that used to work.

WHAT AN EXPECTATION MAY SAY, and what it deliberately may not:
    Every check below reads a telemetry field the graph ALREADY records
    (D-12) and compares it to a band. There is no LLM judge here, no
    similarity-to-a-reference-answer, no scored rubric -- adding one
    would make this harness's own verdict as arguable as the thing it is
    measuring, and this project already has one fail-open judge it does
    not fully trust (evaluation/quality.py's own docstring says so).

    So a case cannot assert "the report is good". It asserts things like
    "the corpus answered this one", "this one shipped the provenance
    notice", "no cited figure was unsupported", "this did not cost more
    than N tokens". Mechanical, reproducible, and honest about being
    narrower than quality.

WHAT IT WILL NOT DO:
    - It is NOT a pytest test and must never become one. D-33's rule: the
      suite is offline, and a check that needs a live model, Qdrant,
      OpenSearch and Postgres has no business silently skipping (or
      silently running) inside that guarantee.
    - It does NOT grade in stub mode. LLM_MODE=stub produces a fixed
      placeholder report; grading it would measure StubClient. The
      harness still RUNS there, which is a useful smoke test of the
      harness itself, and says plainly that grading was skipped.
    - It writes NOTHING to the corpus, and by default nothing to the
      production memory collection -- see --memory-collection below.

Usage:
    python scripts/eval_suite.py --run
    python scripts/eval_suite.py --run --save-baseline eval-baseline.json
    python scripts/eval_suite.py --run --baseline eval-baseline.json
    python scripts/eval_suite.py --run --case in-corpus-comparison
    python scripts/eval_suite.py --list

Exit codes:
    0  every case met its expectations (or stub mode: harness ran, no grading)
    1  at least one case missed an expectation
    2  the harness could not run at all (bad config, unreachable services,
       HITL enabled, empty golden file)
"""

import argparse
import json
import os
import pathlib
import sys
import time
import uuid

# Resolve "src" RELATIVE TO THIS FILE, never the current working
# directory -- same reasoning every other script in this directory
# documents.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "sample_data" / "golden_queries.jsonl"

# Where an eval run's semantic memory goes by default.
#
# THIS IS NOT TIDINESS. memory_writer stores a passed run's fresh
# evidence (D-24), and memory_retrieve feeds it back into goal
# composition on LATER runs. Point the harness at the production
# collection and every eval run changes the conditions the NEXT eval run
# is measured under -- the measurement contaminates its own subject, and
# a drift caused by the harness reads exactly like a drift it detected.
# Live precedent that this is not theoretical: D-42 exists because one
# run's stored recollection re-framed a later, unrelated run's goals.
#
# Overridable with --memory-collection (pass MEMORY_COLLECTION's real
# value to deliberately measure the production memory's influence).
EVAL_MEMORY_COLLECTION = "agent_eval_memory"


# ---------------------------------------------------------------------------
# Pure logic -- loading, grading, diffing. No I/O, no graph, no database.
# Tested directly by tests/unit/test_eval_suite.py, the same split
# gc_memory.py, inspect_memory.py and analyze_runs.py already use.
# ---------------------------------------------------------------------------


def load_cases(path):
    """Read a golden-set JSONL file into a list of case dicts.

    RAISES ValueError naming the offending line for malformed JSON or a
    case missing `id`/`query` -- a golden set with a typo must fail
    loudly at load time, not silently run seven of eight cases.
    """
    cases = []
    seen = set()
    for number, line in enumerate(pathlib.Path(path).read_text(
            encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON -- {exc}")
        if not case.get("id") or not case.get("query"):
            raise ValueError(f"{path}:{number}: every case needs an 'id' and "
                             f"a 'query'")
        if case["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate case id "
                             f"{case['id']!r} -- ids key the baseline file, "
                             f"so they have to be unique")
        seen.add(case["id"])
        cases.append(case)
    return cases


# Every supported expectation, as (key, telemetry field, comparison).
# A band the run must sit inside; anything absent from a case's `expect`
# block is simply not checked. Adding a knob here is deliberately cheap
# and adding a JUDGEMENT is deliberately impossible.
_NUMERIC_CHECKS = (
    ("min_recall", "recall", "min"),
    ("max_recall", "recall", "max"),
    ("min_corpus_recall", "corpus_recall", "min"),
    ("max_corpus_recall", "corpus_recall", "max"),
    ("min_grounded_score", "grounded_score", "min"),
    ("max_grounded_score", "grounded_score", "max"),
    ("min_grounding_ratio", "grounding_ratio", "min"),
    ("min_evidence_items", "evidence_items", "min"),
    ("max_unsupported_figures", "cited_figures_unsupported", "max"),
    ("max_total_tokens", "llm_total_tokens", "max"),
    ("max_provider_calls", "llm_provider_calls", "max"),
    ("max_elapsed_seconds", "run_elapsed_seconds", "max"),
)


def grade(case, telemetry):
    """Check one run's telemetry against one case's expectations.

    RETURNS {"id", "passed", "checks": [...]} where each check is
    {"name", "expected", "actual", "ok"}. A case with no `expect` block
    is reported as passed with zero checks -- useful while adding a
    query you have not yet decided the band for.

    A telemetry field that is ABSENT (an older run, a run that never
    reached telemetry_node) fails the check rather than passing it: the
    expectation was not met, and treating "not measured" as "fine" is the
    exact confusion D-103 removed from the recall column.
    """
    checks = []
    expect = case.get("expect") or {}

    for key, field, kind in _NUMERIC_CHECKS:
        if key not in expect:
            continue
        want = expect[key]
        actual = telemetry.get(field)
        if actual is None:
            ok = False
        elif kind == "min":
            ok = float(actual) >= float(want)
        else:
            ok = float(actual) <= float(want)
        checks.append({"name": key, "expected": want, "actual": actual,
                       "ok": ok})

    if "expect_tiers" in expect:
        allowed = set(expect["expect_tiers"])
        answered = set((telemetry.get("tier_answers") or {}).keys())
        # Empty `answered` fails: a run where no tier is recorded as
        # having answered is not a run that answered from the allowed
        # set, whatever its recall says.
        ok = bool(answered) and answered <= allowed
        checks.append({"name": "expect_tiers", "expected": sorted(allowed),
                       "actual": sorted(answered), "ok": ok})

    for key, field in (("require_grounding_notice", "grounding_notice_shipped"),
                       ("require_critique_passed", "critique_passed"),
                       ("require_truncation_notice", "truncation_notice_shipped")):
        if key not in expect:
            continue
        want = bool(expect[key])
        actual = telemetry.get(field)
        checks.append({"name": key, "expected": want, "actual": actual,
                       "ok": bool(actual) == want})

    return {"id": case["id"], "passed": all(c["ok"] for c in checks),
            "checks": checks}


# What a baseline stores per case: the numbers worth watching move.
# Deliberately NOT the whole telemetry dict -- a baseline that changes
# shape every time a field is added is a baseline nobody keeps.
_BASELINE_FIELDS = ("recall", "corpus_recall", "grounded_score",
                    "grounding_ratio", "evidence_items", "llm_total_tokens",
                    "llm_provider_calls", "cited_figures_unsupported",
                    "run_elapsed_seconds")


def baseline_entry(telemetry):
    """The slice of one run's telemetry a baseline keeps."""
    return {f: telemetry.get(f) for f in _BASELINE_FIELDS}


def diff_against_baseline(baseline, results):
    """Compare this run's numbers to a stored baseline, per case.

    RETURNS a list of {"id", "field", "was", "now", "delta"} for every
    field that MOVED. Unchanged fields are omitted -- a regression report
    that prints everything is a report nobody reads.

    A case absent from the baseline is reported once with field "(new)",
    and a baseline case absent from this run with field "(missing)", so
    a golden set that grew or shrank says so rather than silently
    comparing fewer cases than you think.
    """
    moved = []
    for result in results:
        was = baseline.get(result["id"])
        if was is None:
            moved.append({"id": result["id"], "field": "(new)",
                          "was": None, "now": None, "delta": None})
            continue
        now = baseline_entry(result.get("telemetry") or {})
        for field in _BASELINE_FIELDS:
            a, b = was.get(field), now.get(field)
            if a == b:
                continue
            delta = None
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                delta = round(b - a, 3)
            moved.append({"id": result["id"], "field": field,
                          "was": a, "now": b, "delta": delta})
    seen = {r["id"] for r in results}
    for case_id in baseline:
        if case_id not in seen:
            moved.append({"id": case_id, "field": "(missing)",
                          "was": None, "now": None, "delta": None})
    return moved


def format_report(results, graded):
    """Render the human-readable result block. Pure, so a test can read
    it rather than a terminal."""
    lines = ["", "=== GOLDEN SET ===",
             f"{len(results)} case(s), grading "
             + ("ON" if graded else "SKIPPED (stub mode)")]
    for result in results:
        grade_result = result.get("grade")
        telemetry = result.get("telemetry") or {}
        if result.get("error"):
            status = "ERROR"
        elif not graded or grade_result is None:
            status = "ran"
        else:
            status = "PASS" if grade_result["passed"] else "FAIL"
        lines.append("")
        lines.append(f"[{status}] {result['id']}")
        if result.get("error"):
            lines.append(f"    {result['error']}")
            continue
        lines.append(
            f"    recall {telemetry.get('recall')}  "
            f"corpus_recall {telemetry.get('corpus_recall')}  "
            f"grounded {telemetry.get('grounded_score')}  "
            f"tiers {sorted((telemetry.get('tier_answers') or {}).keys())}")
        lines.append(
            f"    tokens {telemetry.get('llm_total_tokens')}  "
            f"provider_calls {telemetry.get('llm_provider_calls')}  "
            f"elapsed {telemetry.get('run_elapsed_seconds')}s  "
            f"unsupported_figures {telemetry.get('cited_figures_unsupported')}")
        for check in (grade_result or {}).get("checks", []):
            if not check["ok"]:
                lines.append(f"    MISS {check['name']}: expected "
                             f"{check['expected']}, got {check['actual']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The live half -- everything below this line needs real services.
# ---------------------------------------------------------------------------


def run_case(app, settings, case, run_id):
    """Invoke the graph once for one case and return its telemetry.

    Each case gets its OWN thread_id, always fresh: reusing one would
    accumulate the previous case's evidence and counters into this one
    (D-20, and the live evidence in assembly.reject_if_thread_in_use).
    The run_id prefix keeps a whole eval sweep greppable in the logs and
    in agent_runs.

    An exception is caught and reported as that case's error rather than
    ending the sweep -- one dead case must not cost the other seven,
    which is the same posture the retrieval ladder takes toward a dead
    tier.
    """
    from research_agent.state import ResearchState

    thread_id = f"eval-{run_id}-{case['id']}"
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.recursion_limit}
    started = time.time()
    try:
        result = app.invoke(ResearchState(raw_query=case["query"]),
                            config=config)
    except Exception as exc:  # noqa: BLE001 -- one case, not the sweep
        return {"id": case["id"], "thread_id": thread_id, "telemetry": {},
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(time.time() - started, 1)}
    if "__interrupt__" in result:
        # Reachable only if HITL was enabled after the guard in main()
        # read it. Reported, never answered: a harness that auto-approves
        # its own escalations is measuring a system nobody runs.
        return {"id": case["id"], "thread_id": thread_id, "telemetry": {},
                "error": "run paused for human review (HITL); eval cases are "
                         "not auto-approved",
                "wall_seconds": round(time.time() - started, 1)}
    return {"id": case["id"], "thread_id": thread_id,
            "telemetry": result.get("telemetry") or {},
            "report_chars": len(result.get("final_report") or ""),
            "wall_seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Golden-set regression harness. Runs fixed queries "
                    "against THIS deployment and checks each run's "
                    "telemetry against expectations written down in "
                    "advance.")
    p.add_argument("--golden", default=str(DEFAULT_GOLDEN),
                   help=f"golden-set JSONL (default {DEFAULT_GOLDEN.name})")
    p.add_argument("--run", action="store_true",
                   help="execute the cases (otherwise nothing runs)")
    p.add_argument("--list", action="store_true",
                   help="print the cases and their expectations, run nothing")
    p.add_argument("--case", action="append", default=[],
                   help="run only this case id (repeatable)")
    p.add_argument("--save-baseline", default="",
                   help="write this sweep's numbers to a baseline file")
    p.add_argument("--baseline", default="",
                   help="compare this sweep against a baseline file")
    p.add_argument("--json", default="",
                   help="write the full result set as JSON to this path")
    p.add_argument("--memory-collection", default=EVAL_MEMORY_COLLECTION,
                   help="Qdrant collection eval runs read and write "
                        f"(default {EVAL_MEMORY_COLLECTION}; pass your real "
                        "MEMORY_COLLECTION to measure production memory's "
                        "influence deliberately)")
    args = p.parse_args(argv)

    try:
        cases = load_cases(args.golden)
    except (OSError, ValueError) as exc:
        print(f"golden set: UNREADABLE -- {exc}")
        return 2
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("golden set: no cases to run")
        return 2

    if args.list or not args.run:
        for case in cases:
            print(f"{case['id']}")
            print(f"    query  : {case['query']}")
            print(f"    why    : {case.get('why', '(not stated)')}")
            print(f"    expect : {json.dumps(case.get('expect') or {})}")
        if not args.run:
            print("\n(nothing ran -- pass --run to execute against this "
                  "deployment)")
        return 0

    # Set BEFORE get_settings() is first called: Settings reads the
    # environment once and get_settings() is @lru_cache'd, so this is the
    # only point at which the eval sweep can redirect memory away from
    # the production collection. See EVAL_MEMORY_COLLECTION above for
    # why that matters more than it looks.
    os.environ["MEMORY_COLLECTION"] = args.memory_collection

    from research_agent.assembly import build_app_and_settings
    from research_agent.config import get_settings

    settings = get_settings()
    if settings.hitl_enabled:
        print("HITL_ENABLED is true. An eval sweep cannot answer its own "
              "escalations, and auto-approving them would measure a system "
              "nobody runs. Set HITL_ENABLED=false for the sweep.")
        return 2

    graded = settings.llm_mode != "stub"
    try:
        bundle = build_app_and_settings()
    except Exception as exc:  # noqa: BLE001 -- report, never traceback
        print(f"could not build the app: {type(exc).__name__}: {exc}")
        return 2

    run_id = uuid.uuid4().hex[:8]
    print(f"eval sweep {run_id}: {len(cases)} case(s), llm_mode="
          f"{settings.llm_mode}, memory_collection={args.memory_collection}")

    results = []
    try:
        for case in cases:
            print(f"  running {case['id']} ...", flush=True)
            result = run_case(bundle.app, settings, case, run_id)
            if graded and not result.get("error"):
                result["grade"] = grade(case, result["telemetry"])
            results.append(result)
    finally:
        # Same close order cli.py's finally block uses, for the same
        # reasons -- this script opens exactly what a CLI run opens.
        from research_agent.storage.postgres import close_checkpointer
        close_checkpointer(bundle.checkpointer)
        if bundle.mcp_bridge is not None:
            bundle.mcp_bridge.close()
        if bundle.web_mcp_bridge is not None:
            bundle.web_mcp_bridge.close()
        if bundle.router is not None:
            bundle.router.close()

    print(format_report(results, graded))

    if args.baseline:
        try:
            baseline = json.loads(pathlib.Path(args.baseline).read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"\nbaseline: UNREADABLE -- {exc}")
            baseline = None
        if baseline is not None:
            moved = diff_against_baseline(baseline, results)
            print("\n=== VS BASELINE ===")
            if not moved:
                print("nothing moved")
            for row in moved:
                delta = "" if row["delta"] is None else f"  ({row['delta']:+})"
                print(f"  {row['id']:<28} {row['field']:<24} "
                      f"{row['was']} -> {row['now']}{delta}")

    if args.save_baseline:
        payload = {r["id"]: baseline_entry(r.get("telemetry") or {})
                   for r in results if not r.get("error")}
        pathlib.Path(args.save_baseline).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {args.save_baseline} "
              f"({len(payload)} case(s))")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"results written to {args.json}")

    errored = [r for r in results if r.get("error")]
    if not graded:
        print("\ngrading SKIPPED: LLM_MODE=stub produces a fixed placeholder "
              "report, so grading it would measure StubClient. The sweep "
              "still exercised the harness end to end.")
        # A case that ERRORED still failed, even ungraded -- found by the
        # first smoke run of this script, which reported two dead cases
        # and exited 0. "Grading was skipped" is not "nothing went
        # wrong", and an exit code that cannot tell them apart is the
        # same defect D-103 removed from the recall column.
        if errored:
            print(f"{len(errored)} case(s) did not complete; the harness ran "
                  f"but the runs did not.")
            return 1
        return 0

    failures = [r for r in results
                if r.get("error") or not (r.get("grade") or {}).get("passed")]
    print(f"\n{len(results) - len(failures)}/{len(results)} case(s) met "
          f"their expectations")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
