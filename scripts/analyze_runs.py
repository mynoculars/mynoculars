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
    that table's own docstring says plainly that "nothing else in this
    codebase reads that table back afterward". After D-85..D-91 that
    telemetry carries `tier_answers`, `corpus_recall`, `grounded_score`,
    token totals, `grounding_notice_shipped` and
    `cited_figures_unsupported`.

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
import pathlib
import sys
from collections import Counter

# Resolve "src" RELATIVE TO THIS FILE, never the current working
# directory -- same reasoning every other script in this directory
# documents.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

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
    """
    tiers = Counter()
    intents = Counter()
    grounded_runs = 0        # runs where a document grounded ANYTHING
    notice_runs = 0          # runs that shipped the D-85 provenance notice
    unsupported_runs = 0     # runs with >=1 D-91 unsupported cited figure
    escalating_runs = 0
    prompt_tokens = 0
    completion_tokens = 0
    token_runs = 0           # runs that reported tokens at all (D-86+)
    recalls, corpus_recalls = [], []

    for run in runs:
        t = run["telemetry"]
        for tier, count in (t.get("tier_answers") or {}).items():
            tiers[tier] += int(count)
        if t.get("intent"):
            intents[str(t["intent"])] += 1
        corpus_recall = t.get("corpus_recall")
        if corpus_recall is not None:
            corpus_recalls.append(float(corpus_recall))
            if float(corpus_recall) > 0.0:
                grounded_runs += 1
        if t.get("recall") is not None:
            recalls.append(float(t["recall"]))
        if t.get("grounding_notice_shipped"):
            notice_runs += 1
        if int(t.get("cited_figures_unsupported") or 0) > 0:
            unsupported_runs += 1
        if t.get("escalations"):
            escalating_runs += 1
        pt = int(t.get("llm_prompt_tokens") or 0)
        ct = int(t.get("llm_completion_tokens") or 0)
        if pt or ct:
            token_runs += 1
            prompt_tokens += pt
            completion_tokens += ct

    def mean(values):
        return round(sum(values) / len(values), 3) if values else None

    return {
        "runs": len(runs),
        "intents": dict(intents.most_common()),
        "tier_answers": dict(tiers.most_common()),
        "mean_recall": mean(recalls),
        "mean_corpus_recall": mean(corpus_recalls),
        "runs_with_any_corpus_grounding": grounded_runs,
        "runs_shipping_provenance_notice": notice_runs,
        "runs_with_unsupported_figures": unsupported_runs,
        "runs_with_escalations": escalating_runs,
        "token_runs": token_runs,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "mean_total_tokens_per_run": (
            round((prompt_tokens + completion_tokens) / token_runs)
            if token_runs else None),
    }


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
        print("\n  Nothing recorded yet. agent_runs gains a row per completed "
              "run, from the CLI or the API.")
        return 0

    print()
    print("Retrieval")
    print("-" * 62)
    print(f"  tier answers            : {facts['tier_answers'] or '(none recorded)'}")
    print(f"  mean recall             : {facts['mean_recall']}")
    print(f"  mean corpus_recall      : {facts['mean_corpus_recall']}")
    print(f"  runs the corpus grounded: {facts['runs_with_any_corpus_grounding']}"
          f" / {facts['runs']}"
          f"  ({_pct(facts['runs_with_any_corpus_grounding'], facts['runs'])})")
    print()
    print("Honesty")
    print("-" * 62)
    print(f"  shipped provenance notice        : "
          f"{facts['runs_shipping_provenance_notice']} / {facts['runs']}"
          f"  ({_pct(facts['runs_shipping_provenance_notice'], facts['runs'])})")
    print(f"  had unsupported cited figures    : "
          f"{facts['runs_with_unsupported_figures']} / {facts['runs']}"
          f"  ({_pct(facts['runs_with_unsupported_figures'], facts['runs'])})")
    print(f"  escalated to a human             : "
          f"{facts['runs_with_escalations']} / {facts['runs']}")
    print()
    print("Cost")
    print("-" * 62)
    if facts["token_runs"]:
        print(f"  runs reporting tokens   : {facts['token_runs']} / {facts['runs']}"
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
