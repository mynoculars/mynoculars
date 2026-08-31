"""
reporting/confidence.py -- one honest number for "how much should I trust
this report" (D-145).

Purpose:
    Compose the signals telemetry ALREADY carries into a single band and
    percentage, plus the reasons that drove it, so a reader does not have
    to integrate eight numbers by eye.

THE PROBLEM THIS SOLVES. Run p205.280-check emitted every one of these and
combined none of them:

    grounding_ratio            1.0     every goal has SOME evidence
    recall                     1.0
    corpus_recall              0.0     nothing from the curated corpus
    grounded_score             0.0     no goal met the evidence floor
    retrieval_floor_drop_ratio 1.0     72 of 72 dense candidates dropped
    llm_quality_score_mean     0.067   the judge rejected all 3 compiles
    critique_passed            False   the critic never accepted it
    cited_figures_unsupported  0 of 0  the audit could not run at all
    escalations                E4 -> approve

Every signal was bad. The RESULT block printed six of them as separate raw
numbers, the reader was left to integrate, and the report -- a confident,
well-formatted comparative analysis -- shipped because a human approved
the escalation. That is precisely the failure a confidence figure exists
to prevent.

WHY CAPS AND NOT A WEIGHTED MEAN. A weighted average over those numbers
scores p205.280-check somewhere near 0.5, because recall and
grounding_ratio were both 1.0 and would have carried it. But "the report
cites nothing despite 100 evidence items" is not a quantity to be averaged
against other quantities -- it is a fact that invalidates the rest. So the
model here is: start from a base built out of the graded signals, then
apply CAPS, each of which is a condition under which no amount of good
news elsewhere should be able to raise the verdict. The lowest cap wins.

WHY A BAND, NOT A BARE PERCENTAGE. A bare "31%" implies a calibration this
project does not have; there is no labelled corpus of good and bad reports
to fit against. A band with its reasons listed is defensible: it says what
it observed and what that means, and the percentage is an ordering aid
inside the band rather than a probability. scripts/eval_suite.py's eight
golden queries are the calibration substrate -- the bands are set so its
in-corpus cases land HIGH/MODERATE and its off-corpus cases land LOW.

WHAT THIS IS NOT. It is NOT the quality judge. evaluation/quality.py scores
a raw answer to decide fallback routing, is fail-open by design, and its
own docstring says it is not a substitute for evidence-grounded
verification. Its mean is ONE input here, weighted accordingly.

D-12 HOLDS. Every value read below was recorded by some other node. This
module aggregates; it does not invent, and it makes no LLM call.

CALLED BY   agents/compilation.py::telemetry_node (which puts `confidence`
            in the telemetry dict, so it reaches the agent_runs row and
            scripts/analyze_runs.py for free) and cli.py::_fmt_result_summary
            (which prints the band, the percentage and the top reasons).
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Band thresholds, as a percentage. Ordered high to low; the first band
# whose floor the score reaches is the verdict.
BANDS: Tuple[Tuple[int, str], ...] = (
    (75, "HIGH"),
    (50, "MODERATE"),
    (25, "LOW"),
    (0, "UNRELIABLE"),
)

# Each cap is a ceiling, in percent, that a named condition imposes. The
# LOWEST applicable cap wins -- a cap is "no amount of good news elsewhere
# should raise the verdict past here", so they cannot be traded off.
CAP_NO_CITATIONS = 15
CAP_CRITIQUE_FAILED = 45
CAP_FLOOR_STARVED = 45
CAP_UNSUPPORTED_FIGURES = 40
# D-144's attachment pass only runs when the model cited NOTHING, so a
# nonzero citations_attached means every marker in the prose was written by
# this codebase, not by the model. That is a rescue and it is far better
# than shipping uncited -- but it is a machine's term-overlap judgement
# standing in for the writer's own attribution, and a report resting on it
# has not earned HIGH.
CAP_ATTRIBUTION_SYNTHESISED = 60
# The Sources block shipped under D-144's uncited note: real pages, listed
# honestly as "retrieved", but not tied to any claim the reader can check.
CAP_SOURCES_UNCITED = 45
CAP_PLANNING_ERROR = 10
CAP_ABORTED = 0


def _band(score: int) -> str:
    for floor, name in BANDS:
        if score >= floor:
            return name
    return "UNRELIABLE"


def _num(telemetry: Dict, key: str, default: float = 0.0) -> float:
    """Read a numeric telemetry field, tolerating absent and None.

    `.get` with a default throughout, never `[]`: an interrupted or
    degraded run reaches this with a partial telemetry dict, and staying
    readable exactly then is the point -- the same reasoning that made
    cli.py::_fmt_result_summary use `.get` for every field it prints.
    """
    value = telemetry.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_report(telemetry: Dict) -> Dict[str, object]:
    """-> {"band", "score", "reasons", "caps"}. Pure; no I/O, no LLM.

    `reasons` is ordered worst-first and is the part meant to be READ --
    the number is an ordering aid, the reasons are the finding.
    """
    reasons: List[str] = []
    caps: List[Tuple[int, str]] = []

    evidence_items = _num(telemetry, "evidence_items")
    cited = _num(telemetry, "evidence_cited")
    attached = _num(telemetry, "citations_attached")

    # ---- the graded part -------------------------------------------------
    # Four signals, each contributing up to its own share of 100. They are
    # deliberately about DIFFERENT questions, which is why they are summed
    # rather than averaged: grounding asks "was there evidence", corpus
    # recall asks "was it curated", the critic asks "is the report faithful",
    # and the judge asks "was the raw answer usable".
    grounding = min(1.0, _num(telemetry, "grounding_ratio"))
    corpus_recall = min(1.0, _num(telemetry, "corpus_recall"))
    judge_mean = _num(telemetry, "llm_quality_score_mean", default=-1.0)

    score = 0.0
    score += 30.0 * grounding
    score += 30.0 * corpus_recall
    if telemetry.get("critique_passed") is True:
        score += 25.0
    if judge_mean < 0:
        # The judge was never asked -- the common case with a single
        # provider, or a chain that never needed a second opinion. Absence
        # of a judgement is not a bad judgement, so this contributes its
        # share neutrally rather than scoring zero.
        score += 7.5
    else:
        score += 15.0 * min(1.0, judge_mean)

    if corpus_recall == 0.0 and evidence_items:
        reasons.append("no goal was answered from the ingested corpus")
    missing = telemetry.get("goals_without_evidence") or []
    if missing:
        reasons.append(f"{len(missing)} goal(s) retrieved no evidence at all")

    # ---- the caps --------------------------------------------------------
    if telemetry.get("abort_reason"):
        caps.append((CAP_ABORTED, "the run was aborted by a human reviewer"))
    if telemetry.get("planning_error"):
        caps.append((CAP_PLANNING_ERROR, "goal composition failed; no "
                                         "retrieval was attempted"))
    if evidence_items and not cited:
        # The p205.280-check condition. D-144's attachment pass usually
        # prevents it now; when even that finds nothing attachable, the
        # report genuinely rests on nothing a reader can check.
        caps.append((CAP_NO_CITATIONS,
                     f"the report cites no evidence despite "
                     f"{int(evidence_items)} item(s) retrieved"))
    if telemetry.get("critique_passed") is False:
        caps.append((CAP_CRITIQUE_FAILED,
                     "the critic never accepted the report"))
    unsupported = _num(telemetry, "cited_figures_unsupported")
    if unsupported:
        caps.append((CAP_UNSUPPORTED_FIGURES,
                     f"{int(unsupported)} cited figure(s) appear in no cited "
                     f"evidence"))
    if _num(telemetry, "retrieval_floor_drop_ratio") >= 0.8:
        # D-152: the same ratio means two different things, and the wording
        # has to say which.
        #
        # For a query the corpus DOES cover, dropping 80%+ of dense
        # candidates means the floor is set too high and real evidence is
        # being discarded -- D-42's failure mode, and a genuine defect.
        # For a query the corpus does NOT cover, the identical ratio is
        # the floor doing exactly its job. Live (p205.282-check) a
        # China-vs-India question against a Redis corpus reported
        # "retrieval was starved", which reads as a misconfiguration to
        # fix and was not one.
        #
        # tier_answers is what separates them: it names which tier
        # actually answered. If the corpus tiers answered nothing and
        # another tier did, the corpus simply has no material on this
        # subject. The CAP is unchanged either way -- an answer with no
        # corpus behind it is LOW whichever the cause -- only the reason
        # text differs, because only the remedy differs.
        tiers = telemetry.get("tier_answers") or {}
        corpus_answered = any(int(count or 0) > 0
                              for tier, count in tiers.items()
                              if str(tier).startswith(("corpus", "mcp")))
        if tiers and not corpus_answered:
            caps.append((CAP_FLOOR_STARVED,
                         "the corpus has no material on this subject; the "
                         "answer rests on other tiers"))
        else:
            caps.append((CAP_FLOOR_STARVED,
                         "the relevance floor dropped at least 80% of dense "
                         "candidates; real evidence may be being discarded"))
    if attached:
        caps.append((CAP_ATTRIBUTION_SYNTHESISED,
                     f"all {int(attached)} citation(s) were attached "
                     f"deterministically; the model wrote none of them"))
    if _num(telemetry, "web_sources_listed_uncited"):
        caps.append((CAP_SOURCES_UNCITED,
                     "sources are listed as retrieved, not as support for "
                     "any specific claim"))

    if caps:
        ceiling, _ = min(caps)
        score = min(score, float(ceiling))
        # Worst cap first, then the graded observations.
        reasons = [reason for _, reason in sorted(caps)] + reasons

    # ---- annotations that do not change the number -----------------------
    # Facts a reader must see next to the verdict, but which are not
    # themselves evidence of unreliability.
    for escalation in telemetry.get("escalations") or []:
        if escalation.get("action") == "approve":
            reasons.append(f"shipped on a human approval of "
                           f"{escalation.get('trigger', '?')}, not on a clean "
                           f"pass")

    final = max(0, min(100, int(round(score))))
    return {"band": _band(final), "score": final, "reasons": reasons,
            "caps": [name for _, name in sorted(caps)]}


def format_line(confidence: Dict[str, object], max_reasons: int = 2) -> str:
    """One RESULT-block line: the band, the number, and why.

    Two reasons by default. The full list is in telemetry; this line has to
    survive next to eight others in a terminal, and a verdict nobody reads
    is worth as little as no verdict.
    """
    reasons = list(confidence.get("reasons") or [])
    tail = "; ".join(reasons[:max_reasons])
    more = len(reasons) - max_reasons
    if more > 0:
        tail += f" (+{more} more)"
    head = f"{confidence.get('band', '?')} ({confidence.get('score', 0)}%)"
    return f"{head}  — {tail}" if tail else head
