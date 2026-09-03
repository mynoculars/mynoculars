"""
reporting/scores.py -- the five Langfuse scores a finished run reports.

WHY THIS FILE EXISTS (S-17). cli.py::_run and api/server.py::_record_scores
each carried their own copy of this, and the two bodies were identical
line for line: same five metrics, same guards, same grounding_ratio
comment. Two copies of a reporting contract is the drift this codebase
avoids everywhere else -- M-1's rule applied to scores rather than to a
predicate -- and the failure mode is quiet: add a sixth score to one
interface and runs served by the other silently stop being comparable.

WHY HERE AND NOT IN langfuse/. D-35 keeps the langfuse package free of
business shapes: it exposes thin, always-safe functions and never learns
what a telemetry key means. This function knows telemetry keys, so it is
a reporting module that CALLS that thin API, exactly like every other
business module does.

D-12 holds: every score below repeats a number telemetry already
computed. Nothing here derives a new one.
"""

from typing import Any, Dict

from research_agent import langfuse as lf


def emit_run_scores(thread_id: str, telemetry: Dict[str, Any]) -> None:
    """Emit this run's scores. Safe on an empty or partial telemetry dict.

    CALLED BY   cli.py::_run at end of run, and api/server.py::
                _record_scores once a request settles as "done".
    RETURNS     None. Every lf.* call is fail-open by construction (see
                langfuse/observer.py), so this cannot fail a run.

    EVERY FIELD IS GUARDED, and the guards are not interchangeable:
    `in telemetry` for the two that are legitimately 0.0 or False, and
    truthiness for the two that are DIVISORS. A run that ends without
    reaching telemetry_node has none of them, and scoring zeros for it
    would put a measurement in the record that nothing measured.
    """
    if not telemetry:
        return
    if "recall" in telemetry:
        lf.score(thread_id, "recall", telemetry["recall"])
    if "critique_passed" in telemetry:
        lf.score(thread_id, "critique_passed", bool(telemetry["critique_passed"]))
    if telemetry.get("evidence_items", 0) and telemetry.get("goals", 0):
        lf.score(thread_id, "evidence_per_goal",
                 telemetry["evidence_items"] / telemetry["goals"])
    if telemetry.get("search_calls", 0):
        lf.score(thread_id, "memory_hit_rate",
                 telemetry.get("memory_hits", 0) / telemetry["search_calls"])
    # Trendable across prompt revisions -- which is what prompt_name/
    # prompt_version tagging on every generation was for. The comment
    # carries WHICH goals were unevidenced, so a low score in the Langfuse
    # UI is actionable without opening the run's logs.
    if "grounding_ratio" in telemetry:
        unevidenced = telemetry.get("goals_without_evidence") or []
        lf.score(thread_id, "grounding_ratio", telemetry["grounding_ratio"],
                 comment=f"unevidenced={','.join(unevidenced) or 'none'}")
    # DELIBERATELY NO log_event HERE. An earlier draft of this extraction
    # added one, and a byte-comparison of a rendered narrative caught it:
    # the line appeared in every run's logs/run-*.txt where neither of the
    # two call sites had ever emitted one. A refactor that claims to
    # preserve behaviour does not get to add output, however useful the
    # output might be -- that is a separate change, made on purpose.
