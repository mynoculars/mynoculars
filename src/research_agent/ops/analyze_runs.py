"""
scripts/analyze_runs.py -- cross-run analysis over agent_runs (D-92).

Purpose:
    Answer questions no single run can: which retrieval tier actually
    answers most often, how often the corpus grounds anything, what a run
    typically costs in tokens, how often a report ships with a figure its
    evidence never mentioned. Read-only, over data this codebase has been
    persisting all along.

WHY THIS EXISTS, AND WHY IT IS NOT "STRATEGY MEMORY":
    The obvious way to make a harness improve across runs is a second
    memory collection recording which strategies worked -- proposed, and
    deliberately not built. It would need new storage, a new write path, a
    new decay model, and a new set of failure modes, all to derive
    conclusions from data.

    But the data already exists. `storage/postgres.py::record_run` has
    been writing one `agent_runs` row per completed run since P2-08 --
    thread_id, query, recall, and the FULL telemetry dict as JSONB -- and
    that table's own docstring said plainly, until this script existed,
    that "nothing else in this codebase reads that table back
    afterward". After D-85..D-91 that telemetry carries `tier_answers`,
    `corpus_recall`, `grounded_score`, token totals,
    `grounding_notice_shipped` and `cited_figures_unsupported`.

    Since then the table has grown two more things this script reads:
    D-103 added a row per FAILED run (recall NULL, `run_outcome`
    "failed"), and D-106 added what the quality judge actually scored.
    Both are aggregated below -- see the Failures and Quality judge
    blocks in main()'s report.

    So the cheapest honest version of strategy memory is a READER, not a
    writer. Everything below is a query over rows that were already
    there. If these numbers eventually justify a real learning loop, they
    are also the evidence base that would justify it -- which is the
    order D-54 asks for: measure first, build against what you measured.

Read-only, always:
    No INSERT, no UPDATE, no DELETE, no DDL. This script cannot damage a
    run history, so there is no --yes gate to get wrong. It does not even
    create the table: a missing `agent_runs` means no run has completed
    yet, which is reported rather than fixed.

Exit codes:
    0  Postgres was reached (even if there were no rows to analyse)
    1  Postgres could not be reached, or psycopg is not installed
"""

import argparse
import json
import sys
from collections import Counter
from typing import Callable, NamedTuple, Tuple

# D-157: the `sys.path.insert(..., '<repo>/src')` bootstrap that stood
# here is gone, and so is the reason for it. This module lives INSIDE
# the package now, so `research_agent` is importable by definition --
# from a checkout on PYTHONPATH, from an editable install, and from a
# wheel, without any of them being a special case. scripts/ keeps a
# thin launcher of the same name for `python scripts/<name>.py`.

from research_agent.config import get_settings              # noqa: E402
from research_agent.logging_setup import configure_logging  # noqa: E402


def load_runs(dsn: str, limit: int = 200, query_like: str = ""):
    """Fetch recent agent_runs rows, newest first.

    RETURNS a list of dicts: {id, thread_id, query, recall, telemetry,
    created_at}. `telemetry` is already decoded -- psycopg returns JSONB
    as a dict, but a row written by an older revision (or by hand) could
    hold a string, so that case is decoded rather than crashing the whole
    report over one bad row.

    RAISES nothing of its own. A missing table, an unreachable server and
    an absent psycopg all surface to main(), which reports them as a
    clean exit 1 -- the same posture gc_memory.py and inspect_memory.py
    take toward an unreachable Qdrant.
    """
    import psycopg

    sql = ("SELECT id, thread_id, query, recall, telemetry, created_at "
           "FROM agent_runs ")
    params: tuple = ()
    if query_like:
        # Parameterised, never f-string-interpolated -- see record_run's
        # own comment on why this matters even for a local tool.
        sql += "WHERE query ILIKE %s "
        params = (f"%{query_like}%",)
    sql += "ORDER BY id DESC LIMIT %s"
    params = params + (limit,)

    out = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        for row in conn.execute(sql, params).fetchall():
            telemetry = row[4]
            if isinstance(telemetry, str):
                try:
                    telemetry = json.loads(telemetry)
                except ValueError:
                    telemetry = {}
            out.append({"id": row[0], "thread_id": row[1], "query": row[2],
                        "recall": row[3], "telemetry": telemetry or {},
                        "created_at": row[5]})
    return out


# S-14: the aggregators, as DATA.
#
# summarize() used to declare 26 accumulator locals, mutate all of them in
# one 120-line loop at cyclomatic complexity 44, and assemble them into a
# dict 40 lines further down. Adding one metric meant touching three widely
# separated places and hoping you found all three -- and this file gains a
# metric every time the graph does.
#
# Same shape reporting/pipeline.py::REPORT_PASSES already established for
# the report post-processing steps: the sequence is a named tuple of small
# pieces rather than a long body, and each piece is independently readable.
# Each aggregator owns three things and nothing else -- its initial state,
# how one completed run folds into that state, and how that state renders
# into output keys. Ordered as the output dict reads, so the JSON a person
# looks at keeps the grouping the comments describe.
#
# FAILED ROWS NEVER REACH AN AGGREGATOR. D-104: a failed run (D-103)
# carries no telemetry to aggregate, so it is separated out in the loop
# below before any fold is called, rather than diluting every denominator.


class _Aggregator(NamedTuple):
    """One independent slice of the cross-run report.

    init    -> a fresh mutable state for this aggregator
    fold    -> (state, telemetry) -> None; one COMPLETED run folded in
    render  -> (state) -> dict of output keys, merged in declaration order
    """

    name: str
    init: Callable[[], dict]
    fold: Callable[[dict, dict], None]
    render: Callable[[dict], dict]


def _mean(values):
    return round(sum(values) / len(values), 3) if values else None


# --- D-106 -----------------------------------------------------------------
# The judge has been decisive across five live runs and explicable in none:
# the only thing ever recorded was that a score was below the threshold, so
# "is 0.6 in the right place" had no data behind it. Accumulated here, which
# is what makes the question answerable at all -- one run's two judgements
# never could.

def _fold_quality(st: dict, t: dict) -> None:
    judged = int(t.get("llm_quality_scores_judged") or 0)
    if not judged:
        return
    st["runs"] += 1
    st["judged"] += judged
    st["rejections"] += int(t.get("llm_quality_rejections") or 0)
    # The run recorded a MEAN, so the run's total is mean x judged --
    # reconstructed rather than stored, because a per-run sum would be a
    # second number that could disagree with the mean already there.
    mean_score = t.get("llm_quality_score_mean")
    if mean_score is not None:
        st["total"] += float(mean_score) * judged
    for name, count in (t.get("llm_quality_bands") or {}).items():
        st["bands"][str(name)] += int(count)


def _render_quality(st: dict) -> dict:
    return {"quality_runs": st["runs"],
            "quality_judgements": st["judged"],
            "quality_rejections": st["rejections"],
            "mean_quality_score": (round(st["total"] / st["judged"], 3)
                                   if st["judged"] else None),
            "quality_bands": {name: st["bands"][name]
                              for _upper, name in _BAND_ORDER
                              if st["bands"][name]}}


# --- D-145 -----------------------------------------------------------------
# The composed verdict is stored per run inside the telemetry JSONB, so a
# history of it costs nothing to read. What it answers that no single run
# can: is the band distribution moving, and do the caps that fire most often
# match the defects being worked on.

def _fold_confidence(st: dict, t: dict) -> None:
    confidence = t.get("confidence") or {}
    if not confidence.get("band"):
        return
    st["runs"] += 1
    st["bands"][str(confidence["band"])] += 1
    st["total"] += int(confidence.get("score") or 0)
    for cap in confidence.get("caps") or []:
        st["caps"][str(cap)] += 1


def _render_confidence(st: dict) -> dict:
    return {"confidence_runs": st["runs"],
            "mean_confidence": (round(st["total"] / st["runs"], 1)
                                if st["runs"] else None),
            "confidence_bands": dict(st["bands"].most_common()),
            "confidence_caps": dict(st["caps"].most_common())}


# --- D-144 -----------------------------------------------------------------
# A rescued report and a self-cited one must never look the same in a
# history either.

def _fold_attribution(st: dict, t: dict) -> None:
    attached = int(t.get("citations_attached") or 0)
    if attached:
        st["runs"] += 1
        st["attached"] += attached


def _render_attribution(st: dict) -> dict:
    return {"runs_with_attached_citations": st["runs"],
            "citations_attached": st["attached"]}


# --- D-105 -----------------------------------------------------------------
# web_sources_listed 0 against a non-zero web_sources_suppressed is the
# loudest possible statement that a report cited nothing the Sources block
# could attribute. Run p205.253-check carried 0 / 78 in its telemetry the
# whole time and nobody read it. Counted here so nobody has to.

def _fold_silent_sources(st: dict, t: dict) -> None:
    listed = int(t.get("web_sources_listed") or 0)
    suppressed = int(t.get("web_sources_suppressed") or 0)
    if listed == 0 and suppressed > 0:
        st["runs"] += 1


def _fold_shape(st: dict, t: dict) -> None:
    """Which tier answered, and what kind of question was asked."""
    for tier, count in (t.get("tier_answers") or {}).items():
        st["tiers"][tier] += int(count)
    if t.get("intent"):
        st["intents"][str(t["intent"])] += 1


def _fold_recall(st: dict, t: dict) -> None:
    """recall, corpus_recall, and how often a document grounded anything."""
    corpus_recall = t.get("corpus_recall")
    if corpus_recall is not None:
        st["corpus"].append(float(corpus_recall))
        if float(corpus_recall) > 0.0:
            st["grounded_runs"] += 1
    if t.get("recall") is not None:
        st["recall"].append(float(t["recall"]))


def _render_recall(st: dict) -> dict:
    return {"mean_recall": _mean(st["recall"]),
            "mean_corpus_recall": _mean(st["corpus"]),
            "runs_with_any_corpus_grounding": st["grounded_runs"]}


def _fold_report_flags(st: dict, t: dict) -> None:
    """Three per-run booleans, counted as run tallies."""
    if t.get("grounding_notice_shipped"):
        st["notice"] += 1
    if int(t.get("cited_figures_unsupported") or 0) > 0:
        st["unsupported"] += 1
    if t.get("escalations"):
        st["escalating"] += 1


def _fold_tokens(st: dict, t: dict) -> None:
    """D-86: what the runs actually COST, over rows that reported it."""
    pt = int(t.get("llm_prompt_tokens") or 0)
    ct = int(t.get("llm_completion_tokens") or 0)
    if pt or ct:
        st["runs"] += 1
        st["prompt"] += pt
        st["completion"] += ct


def _render_tokens(st: dict) -> dict:
    return {"token_runs": st["runs"],
            "prompt_tokens": st["prompt"],
            "completion_tokens": st["completion"],
            "mean_total_tokens_per_run": (
                round((st["prompt"] + st["completion"]) / st["runs"])
                if st["runs"] else None)}


AGGREGATORS: Tuple[_Aggregator, ...] = (
    _Aggregator("quality",
                lambda: {"runs": 0, "judged": 0, "rejections": 0,
                         "total": 0.0, "bands": Counter()},
                _fold_quality, _render_quality),
    _Aggregator("confidence",
                lambda: {"runs": 0, "total": 0, "bands": Counter(),
                         "caps": Counter()},
                _fold_confidence, _render_confidence),
    _Aggregator("attribution",
                lambda: {"runs": 0, "attached": 0},
                _fold_attribution, _render_attribution),
    _Aggregator("silent_sources",
                lambda: {"runs": 0},
                _fold_silent_sources,
                lambda st: {"runs_listing_no_cited_web_sources": st["runs"]}),
    _Aggregator("shape",
                lambda: {"tiers": Counter(), "intents": Counter()},
                _fold_shape,
                lambda st: {"intents": dict(st["intents"].most_common()),
                            "tier_answers": dict(st["tiers"].most_common())}),
    _Aggregator("recall",
                lambda: {"recall": [], "corpus": [], "grounded_runs": 0},
                _fold_recall, _render_recall),
    _Aggregator("report_flags",
                lambda: {"notice": 0, "unsupported": 0, "escalating": 0},
                _fold_report_flags,
                lambda st: {"runs_shipping_provenance_notice": st["notice"],
                            "runs_with_unsupported_figures": st["unsupported"],
                            "runs_with_escalations": st["escalating"]}),
    _Aggregator("tokens",
                lambda: {"runs": 0, "prompt": 0, "completion": 0},
                _fold_tokens, _render_tokens),
)


def summarize(runs) -> dict:
    """Aggregate a list of runs into counted facts.

    Pure -- no I/O -- so it is testable without a database, the same
    split gc_memory.py::find_gc_candidates and
    inspect_memory.py::summarize already use.

    D-12's rule applies here exactly as it does inside the graph: this
    counts what the runs recorded and invents nothing. There is no
    judgement about whether a corpus_recall of 0.2 is "bad" -- that
    depends on the corpus, and a script cannot know it.

    Every field is read with .get() and a default: telemetry rows written
    before D-85..D-91 simply lack the newer keys, and a cross-run report
    that crashes on the first old row would be useless exactly when you
    most want history.

    THE BODY IS NOW A LOOP OVER AGGREGATORS (S-14). Failure classification
    stays here, inline, because it is the one thing that decides whether a
    row reaches the aggregators at all -- everything else is in the tuple
    above, one small function per metric group.
    """
    failures = Counter()             # failure type -> count
    failure_chains = Counter()       # "provider outcome" -> count
    failed_runs = 0
    states = {a.name: a.init() for a in AGGREGATORS}

    for run in runs:
        t = run["telemetry"]
        if is_failed(t):
            failed_runs += 1
            failure = t.get("failure") or {}
            failures[str(failure.get("type") or "unknown")] += 1
            for pair in failure.get("chain") or []:
                # Each pair is [provider, outcome] -- exactly what
                # cli.py::_failure_record wrote. Counted per provider so a
                # history can show that `primary` fails every time while
                # the cloud hops rarely do.
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    failure_chains[f"{pair[0]} {pair[1]}"] += 1
            # Nothing below applies to a row with no telemetry in it.
            continue
        for agg in AGGREGATORS:
            agg.fold(states[agg.name], t)

    facts = {
        "runs": len(runs),
        # D-104: rows analysed, minus the ones that never produced
        # telemetry. Every rate printed by main() divides by THIS.
        "completed_runs": len(runs) - failed_runs,
        "failed_runs": failed_runs,
        "failures_by_type": dict(failures.most_common()),
        "failed_provider_outcomes": dict(failure_chains.most_common()),
    }
    for agg in AGGREGATORS:
        facts.update(agg.render(states[agg.name]))
    return facts


# D-106: the router's own band order, restated here rather than imported.
# This script is loaded BY FILE PATH (see its tests) and deliberately
# depends on nothing in research_agent except config and logging -- and a
# report over historical rows must keep working against band names those
# rows were written with, not against whatever the current router defines.
_BAND_ORDER = ((0.2, "very_low"), (0.4, "low"), (0.6, "mid"),
               (0.8, "high"), (1.01, "very_high"))


def is_failed(telemetry) -> bool:
    """Is this row a D-103 failed-run record?

    The contract, stated once here and relied on everywhere else in this
    file: a row is failed IFF its telemetry says so. Absence means
    completed, which is already true of every row written before D-103 --
    so this classifies the whole history correctly, not just new rows.
    """
    return (telemetry or {}).get("run_outcome") == "failed"


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.0f}%" if whole else "n/a"


def main(argv=None) -> int:
    """Parse arguments, load runs, print the report."""
    p = argparse.ArgumentParser(
        description="Cross-run analysis over the agent_runs table. Read-only.")
    p.add_argument("--limit", type=int, default=200,
                   help="how many recent runs to analyse (default 200)")
    p.add_argument("--query-like", default="",
                   help="only runs whose query contains this text "
                        "(case-insensitive) -- e.g. --query-like compare")
    p.add_argument("--list", action="store_true",
                   help="also print one line per run, newest first")
    p.add_argument("--json", action="store_true",
                   help="machine-readable summary instead of the report")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        runs = load_runs(settings.postgres_dsn, args.limit, args.query_like)
    except Exception as exc:  # noqa: BLE001 -- report, never traceback
        # Covers all three realistic failures with one message: psycopg
        # absent, server unreachable, table not yet created. Naming the
        # exception type keeps them distinguishable without three
        # near-identical branches.
        print(f"Postgres:   UNREADABLE — {type(exc).__name__}: {exc}")
        print("            (no completed run has been recorded yet if the "
              "agent_runs table simply does not exist)")
        return 1

    facts = summarize(runs)

    if args.json:
        print(json.dumps(facts, indent=2, default=str))
        return 0

    print(f"Run history ({facts['runs']} run(s)"
          + (f", query ILIKE %{args.query_like}%" if args.query_like else "")
          + ")")
    if not facts["runs"]:
        print("\n  Nothing recorded yet. agent_runs gains a row per CLI run --"
              "\n  one per completed run, and since D-103 one per failed run "
              "too.")
        return 0

    # D-104: printed FIRST, and before anything that divides by
    # completed_runs, so a report whose rates are computed over a subset
    # says so before it shows them rather than after.
    if facts["failed_runs"]:
        print()
        print("Failures")
        print("-" * 62)
        print(f"  runs that did not finish: {facts['failed_runs']}"
              f" / {facts['runs']}"
              f"  ({_pct(facts['failed_runs'], facts['runs'])})")
        for name, count in facts["failures_by_type"].items():
            print(f"    {name:<38} {count}")
        if facts["failed_provider_outcomes"]:
            print("  provider outcomes on those runs (D-101):")
            for name, count in facts["failed_provider_outcomes"].items():
                print(f"    {name:<38} {count}")
        print("  Every rate below is out of the "
              f"{facts['completed_runs']} completed run(s).")

    print()
    print("Retrieval")
    print("-" * 62)
    print(f"  tier answers            : {facts['tier_answers'] or '(none recorded)'}")
    print(f"  mean recall             : {facts['mean_recall']}")
    print(f"  mean corpus_recall      : {facts['mean_corpus_recall']}")
    done = facts["completed_runs"]
    print(f"  runs the corpus grounded: {facts['runs_with_any_corpus_grounding']}"
          f" / {done}"
          f"  ({_pct(facts['runs_with_any_corpus_grounding'], done)})")
    print()
    print("Honesty")
    print("-" * 62)
    print(f"  shipped provenance notice        : "
          f"{facts['runs_shipping_provenance_notice']} / {done}"
          f"  ({_pct(facts['runs_shipping_provenance_notice'], done)})")
    print(f"  had unsupported cited figures    : "
          f"{facts['runs_with_unsupported_figures']} / {done}"
          f"  ({_pct(facts['runs_with_unsupported_figures'], done)})")
    print(f"  escalated to a human             : "
          f"{facts['runs_with_escalations']} / {done}")
    # D-105: 14.6's follow-up. Not a rate -- any non-zero count here is
    # worth opening the run for, so it prints as a flagged line rather
    # than a percentage that rounds a single occurrence to 0%.
    if facts["runs_listing_no_cited_web_sources"]:
        print(f"  !! cited NO web source, yet suppressed some: "
              f"{facts['runs_listing_no_cited_web_sources']} / {done}")
        print("     (web_sources_listed 0 with web_sources_suppressed > 0 --"
              " the D-99 shape: a report whose citations nothing could read)")
    # D-144: a rescue must be visible in a history, not just in one run.
    if facts["runs_with_attached_citations"]:
        print(f"  citations attached deterministically: "
              f"{facts['runs_with_attached_citations']} / {done} run(s), "
              f"{facts['citations_attached']} marker(s)")
        print("     (D-144 fired -- those reports cited nothing until it did)")
    print()
    # D-145: the composed verdict, aggregated. One run's band says whether
    # THAT report is trustworthy; the distribution says whether the system
    # is getting better, and the cap tally says at what.
    print("Confidence (D-145)")
    print("-" * 62)
    if facts["confidence_runs"]:
        print(f"  runs scored                      : "
              f"{facts['confidence_runs']} / {done}")
        print(f"  mean score                       : "
              f"{facts['mean_confidence']}%")
        for band, count in facts["confidence_bands"].items():
            print(f"    {band:<12}                   : {count}"
                  f"  ({_pct(count, facts['confidence_runs'])})")
        if facts["confidence_caps"]:
            print("  what capped them (most common first):")
            for cap, count in facts["confidence_caps"].items():
                print(f"    {count:>3} x {cap}")
    else:
        print("  no run in this window carries a confidence verdict")
        print("  (rows written before D-145 simply lack the field)")
    print()
    print("Quality judge")
    print("-" * 62)
    if facts["quality_judgements"]:
        print(f"  runs the judge scored   : {facts['quality_runs']} / {done}")
        print(f"  judgements              : {facts['quality_judgements']}"
              f"   (mean {facts['mean_quality_score']})")
        print(f"  below threshold         : {facts['quality_rejections']}"
              f" / {facts['quality_judgements']}"
              f"  ({_pct(facts['quality_rejections'], facts['quality_judgements'])})")
        print(f"  distribution            : {facts['quality_bands']}")
        print("  Bands are fixed (<0.2 / <0.4 / <0.6 / <0.8 / rest) and do")
        print("  NOT move with LLM_QUALITY_THRESHOLD -- that is what lets")
        print("  them show whether the threshold sits in the right place.")
    else:
        print("  no run has recorded a judgement yet (D-106 and later only).")
        print("  A run records nothing here when the judge failed open every")
        print("  time -- check llm_quality_calls_failed, not this section.")
    print()
    print("Cost")
    print("-" * 62)
    if facts["token_runs"]:
        print(f"  runs reporting tokens   : {facts['token_runs']} / {done}"
              "   (older rows predate D-86)")
        print(f"  prompt / completion     : {facts['prompt_tokens']:,}"
              f" / {facts['completion_tokens']:,}")
        print(f"  mean total per run      : "
              f"{facts['mean_total_tokens_per_run']:,}")
    else:
        print("  no run has recorded token totals yet (D-86 and later only)")
    print()
    print(f"Intents: {facts['intents'] or '(none recorded)'}")

    if args.list:
        print()
        print("Runs, newest first")
        print("-" * 62)
        for run in runs:
            t = run["telemetry"]
            if is_failed(t):
                # D-104: a failed row printed in the completed format
                # reads as "recall=? corpus=? tiers={}", which is exactly
                # how a very BAD run looks. They are different events and
                # the listing must not blur them.
                failure = t.get("failure") or {}
                print(f"  #{run['id']:<5} {str(run['created_at'])[:19]}  "
                      f"FAILED {failure.get('type', 'unknown')}"
                      + (f" at {failure['node']}" if failure.get("node") else ""))
                chain = failure.get("chain") or []
                if chain:
                    print("         "
                          + " -> ".join(f"{p[0]} {p[1]}" for p in chain
                                        if isinstance(p, (list, tuple))
                                        and len(p) == 2))
            else:
                print(f"  #{run['id']:<5} {str(run['created_at'])[:19]}  "
                      f"recall={t.get('recall', '?')} "
                      f"corpus={t.get('corpus_recall', '?')} "
                      f"tiers={t.get('tier_answers', {})}")
            print(f"         {str(run['query'])[:88]!r}")

    print()
    print("  Read these against your own corpus. A low mean corpus_recall is")
    print("  a fact about the QUERIES you ran versus the documents you")
    print("  ingested -- it is not, on its own, a defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
