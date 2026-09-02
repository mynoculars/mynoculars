"""
tools/model_knowledge.py — the LLM itself as a retrieval tier (D-38).

Purpose:
    Turn the answering model's own parametric knowledge into ordinary
    Evidence, so a goal the corpus cannot serve is still answered instead
    of being reported as unanswerable.

Why this exists:
    Before this module the system had exactly ONE retrieval tier. If
    hybrid corpus search returned nothing above the quality floor, the
    run had no other move: progress_checker could never mark the goal
    covered, gap_generator gave up, and compile_report was under an
    explicit instruction NEVER to use the model's own knowledge. A run
    against a corpus that simply did not contain the subject therefore
    terminated in "the retrieved evidence does not cover it" no matter
    how well the model itself knew the answer. That is a retrieval
    limitation being reported to the user as an absence of knowledge,
    which is the wrong answer to give.

    A missing corpus document does not mean the information cannot be
    obtained. This tier obtains it.

Honesty invariants — this tier is additive, never disguised:
    - Every item it produces carries source="model". It is NEVER labelled
      "corpus". evidence_by_source in telemetry therefore shows exactly
      how much of a report rests on parametric knowledge.
    - compile_report (prompts/templates.py) requires model-sourced claims
      to be attributed in the report itself, and telemetry reports
      corpus_recall separately from recall so "we answered it" and "the
      corpus answered it" can never be confused.
    - It is the LAST tier tried (see tools/retrieval_chain.py), so a real
      document always wins over recollection when one exists.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from research_agent.guardrails.retrieval import passes_evidence_gate
from research_agent.logging_setup import log_event
from research_agent.prompts import templates
from research_agent.state import Evidence, SearchTask, Volatility

logger = logging.getLogger(__name__)

# Guardrail G3: a claim's own self-reported `confidence` field does not
# catch false precision -- live evidence (run p205.131-check) shows the
# model reporting confidence>=0.5 on fabricated figures the critic later
# rejected, e.g. "India's population grew from approximately 900 million
# in 1970 to over 1.4 billion in 2020" -- a precise quantity pinned to a
# precise date, stated as fact, present in no evidence item. This is a
# cheap, deterministic pre-flag -- not a filter, not a rewrite -- so the
# compiler prompt (prompts/templates.py::compile_report) can be told
# explicitly which model-tier claims carry that risk and hedge them,
# rather than relying on the compiler to remember on every single claim.
# Deliberately narrow: a bare date ("in 1979") or a bare rounded figure
# ("about 35%") alone is common and unremarkable; it is specifically the
# PAIRING of the two that reads as verified fact while being
# unverifiable recollection. A claim naming a specific year with no
# accompanying quantity (e.g. "the policy was introduced in 1979") is
# NOT caught by this heuristic -- catching that class reliably would
# need real date/event extraction, which is exactly the kind of
# judgment call this codebase's philosophy reserves for an LLM (the
# critic), not a regex.
#
# P205.134 follow-up: three of five claims the critic rejected as
# fabricated in that run slipped past the ORIGINAL unit list --
# "500 GW of non-fossil capacity by 2030", "1.9 metric tons" CO2 per
# capita in 2022, "8.0 global hectares" per capita as of 2021. Same
# year+quantity pairing this guard already targets, just units the
# original list didn't cover (%/million/billion/trillion/per-X only --
# no energy, mass, or area units at all). Added here rather than
# widening to a catch-all "any unit word" pattern, which would start
# flagging ordinary counts ("3 goals", "12 states") that carry no false
# precision risk -- these are specifically the units this system's own
# domains (economic, demographic, environmental/energy) actually use.
_SPECIFIC_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
# `%` is matched bare, with NO trailing \b: \b checks for a transition
# between a word char and a non-word char, and "%" is itself a non-word
# char -- so `%\b` only succeeds when a word character immediately
# follows the "%" (e.g. "50%increase"), which real prose essentially
# never does ("50% increase" has a space, and space is ALSO non-word,
# so no boundary transition exists there either). This bug predates the
# P205.134 unit additions below -- rechecked here while widening the
# rest of the pattern, since it means the ORIGINAL percentage branch
# had effectively never matched normal text with a space after the
# sign. Every other alternative below ends in a word character, so it
# keeps the trailing \b (needed to stop "million" matching inside
# "billionaire", etc.) -- only the symbol gets the exemption.
_SPECIFIC_NUMBER = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|(?:"
    r"percent|million|billion|trillion|"
    r"per\s+(?:square\s+)?(?:km|kilometer|mile|capita|year)|"
    # Energy: GW/MW/kW and their -h (watt-hour) variants, spelled out or
    # abbreviated -- "500 GW", "3.2 megawatts", "40 kWh".
    r"[gmk]w(?:h)?s?|gigawatts?|megawatts?|kilowatts?|"
    # Mass: metric tons/tonnes -- "1.9 metric tons", "40 tonnes".
    r"(?:metric\s+)?tonnes?|(?:metric\s+)?tons?|"
    # Area: hectares, including "global hectares" (the ecological-
    # footprint unit) -- the "global" is optional so both match.
    r"(?:global\s+)?hectares?|"
    # Air quality / concentration: micrograms or milligrams per cubic
    # metre (both the proper micro sign U+00B5 and the Greek mu U+03BC
    # show up in the wild, plus a plain ASCII "u" fallback some sources
    # use instead of either), and ppm/ppb -- "100 \u00b5g/m\u00b3", "45 ppm".
    # P205.136 follow-up: "Delhi's annual average exceeding 100 "
    # "\u00b5g/m\u00b3 in multiple years" reached a shipped report unflagged --
    # same year+quantity pairing this guard already targets, just yet
    # another domain-specific unit (environmental/air-quality, following
    # the same pattern that added energy/mass/area units above) that the
    # list didn't cover yet.
    r"(?:[\u00b5\u03bcu]g|mg)\s*/\s*m(?:\u00b3|3)|ppm|ppb)\b)",
    re.IGNORECASE)


def _looks_overspecific(text: str) -> bool:
    """True if `text` states a specific year AND a specific quantity --
    the exact combination (a precise figure pinned to a precise date)
    that reads as verified fact but, for the model-knowledge tier, is
    unverifiable recollection. Either alone is common and unremarkable
    (a year on its own, a rounded percentage on its own); it's the pair
    that carries false precision. Pure string check, no LLM call --
    consistent with this codebase's "deterministic where possible"
    guardrail philosophy (see guardrails/__init__.py).
    """
    return overspecific_span(text) is not None


def overspecific_span(text: str) -> Optional[str]:
    """Return the exact quantity substring _looks_overspecific matched
    (e.g. "500 GW", "6.7%", "1.9 metric tons"), or None if `text` isn't
    overspecific. Not private (no leading underscore, unlike this
    module's other helpers): guardrails/hedging.py -- Guardrail G3's
    enforcement half (P205.135 follow-up) -- needs the literal matched
    text, not just the boolean _looks_overspecific gives, so it can
    search the COMPILED REPORT for that same substring and hedge it if
    the compiler ignored the ATTRIBUTION RULE instruction and stated it
    as flat fact anyway. Kept here rather than duplicated in hedging.py
    so the two guardrails can never define "overspecific" differently.
    """
    if not _SPECIFIC_YEAR.search(text):
        return None
    m = _SPECIFIC_NUMBER.search(text)
    return m.group(0) if m else None

# Model-sourced evidence must clear settings.min_evidence_score (0.5) so it
# can actually mark a goal covered -- otherwise this tier would produce
# evidence the coverage rule ignores, the gather loop would never converge,
# and we would be back to the failure this module exists to remove. It sits
# deliberately BELOW the ~1.0 a document both retrieval legs agreed on, so
# corpus evidence always outranks recollection in the compiler's context.
DEFAULT_MODEL_SCORE = 0.6

# The bonus in the score formula below. Named because D-163's admission
# rule and the score are now the same arithmetic and must stay so.
CONFIDENCE_BONUS = 0.05

# D-163: the confidence a claim must beat before this tier will even keep
# it, INDEPENDENT of scoring. The derived rule below subsumes this at the
# shipped thresholds; it survives for the low-floor configuration, where
# `MIN_EVIDENCE_SCORE=0.1` would otherwise admit a claim the model itself
# says it half-remembers. The module's stated principle -- an item the
# model disowns is worse than no item -- is not a function of anyone's
# threshold.
MIN_SELF_REPORTED_CONFIDENCE = 0.5


def score_for_confidence(confidence: float,
                         score: float = DEFAULT_MODEL_SCORE) -> float:
    """The Evidence.score a claim at `confidence` receives.

    Extracted (D-163) so the admission rule can ASK this function rather
    than re-derive its algebra -- the same M-1 reasoning that put
    has_grounded_evidence and passes_evidence_gate in one place.
    """
    return round(min(score, score * confidence + CONFIDENCE_BONUS), 4)


def make_model_knowledge_tool(router: Any, score: float = DEFAULT_MODEL_SCORE,
                              max_claims: int = 4,
                              min_evidence_score: float = 0.5):
    """Build a ToolFn backed by the LLM's own knowledge.

    CALLED BY   assembly.py, as the final tier handed to
                tools/retrieval_chain.py::make_retrieval_chain.
    RETURNS     a plain ToolFn — Callable[[SearchTask], List[Evidence]] —
                so it is interchangeable with corpus_search and the MCP
                tool and needs no new node, edge, or worker type.

    Deliberately does NOT call router.set_node(): one router instance is
    shared across every parallel search_worker (llm/router.py keeps its
    counters in a threading.local for exactly this reason), and set_node
    mutates provider-level trace state that is not per-thread. The node
    label is already carried by the log line below.
    """

    def model_knowledge(task: SearchTask) -> List[Evidence]:
        result: Dict[str, Any] = router.complete_json(
            templates.model_knowledge(task.query, max_claims))
        claims = result.get("claims") or []
        evidence: List[Evidence] = []
        dropped_inert = 0
        for claim in claims[:max_claims]:
            # The model is asked for {"text": ..., "confidence": 0..1}. A
            # bare string is accepted too: a malformed item is data, not a
            # crash (the same D-16 posture the search worker already takes).
            if isinstance(claim, str):
                text, confidence = claim, 1.0
            elif isinstance(claim, dict):
                text = str(claim.get("text", "")).strip()
                try:
                    confidence = float(claim.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
            else:
                continue
            if not text:
                continue
            # Low-confidence recollection is dropped rather than scored
            # down: an item the model itself flags as shaky is worse than
            # no item, because it can still mark a goal covered.
            if confidence < MIN_SELF_REPORTED_CONFIDENCE:
                continue
            item_score = score_for_confidence(confidence, score)
            # D-163: AND it must be able to do the job it was admitted for.
            #
            # The comment above states this tier's whole safety argument --
            # a shaky item is dangerous BECAUSE it can still mark a goal
            # covered. That argument silently stopped applying to half the
            # admitted band. With the shipped 0.6/0.5 pair the score of a
            # claim is `min(0.6, 0.6*conf + 0.05)`, and the coverage gate
            # is a strict `>`, so:
            #
            #     conf 0.50 -> 0.35     conf 0.75 -> 0.50   (cannot cover)
            #     conf 0.76 -> 0.506    conf 1.00 -> 0.60   (covers)
            #
            # Everything from 0.50 to 0.75 was retrieved, prompted, made
            # citable -- and could never converge a goal. The module's own
            # header says the opposite in as many words: model evidence
            # "must clear settings.min_evidence_score ... so it can
            # actually mark a goal covered -- otherwise this tier would
            # produce evidence the coverage rule ignores [and] the gather
            # loop would never converge". It did exactly that, and the
            # loop paid for it: agents/gathering.py's `ladder_exhausted`
            # is `not settings.model_knowledge_enabled`, so an
            # inert-but-enabled tier also stopped the no-strong-evidence
            # escalation from firing.
            #
            # ASKING THE GATE rather than inverting its algebra: the bar
            # to be admitted and the bar to be useful are now one
            # comparison, made by the same predicate progress_checker_node
            # uses, so they cannot drift apart again the way two constants
            # did. Re-tuning MIN_EVIDENCE_SCORE or MODEL_KNOWLEDGE_SCORE
            # moves both at once and needs no second edit here.
            #
            # THE ALTERNATIVE IS CLOSED, NOT PENDING (D-165). Band-scoring
            # this tier -- mapping confidence onto [floor, ceiling] the way
            # websearch/scoring.py maps rank, so that everything admitted
            # covers by construction -- was weighed in full and rejected:
            # it would let a claim the model half-disowns converge a goal,
            # and the band does not fit between min_evidence_score (0.5)
            # and the web tier's floor (0.60) in any case. See DECISIONS.md
            # "Closed, not pending". Do not re-derive it from first
            # principles here; it has been derived.
            if not passes_evidence_gate(item_score, min_evidence_score):
                dropped_inert += 1
                continue
            evidence.append(Evidence(
                task_key=task.key, goal_id=task.goal_id, source="model",
                content=text[:800],
                # Scale within a narrow band so a confident recollection
                # still cannot outrank a document both legs agreed on.
                score=item_score,
                volatility=Volatility.SEMI_STABLE,
                # Guardrail G3 -- deterministic, evaluated on the same
                # text the model just produced, independent of (and not
                # replacing) the confidence gate above.
                hedge_specific=_looks_overspecific(text)))
        flagged = sum(1 for e in evidence if e.hedge_specific)
        # D-163: `dropped_inert` is reported, never silent. A tier that
        # keeps discarding claims for being unable to clear the coverage
        # floor is telling you the two thresholds are mismatched, and
        # config.py::warn_on_model_knowledge_inert says so at startup for
        # the total case; this says it per call for the partial one.
        #
        # THE REMEDY IT POINTS AT IS TUNING THOSE TWO SETTINGS. A nonzero
        # count here is not evidence for band-scoring this tier -- that
        # design is closed, see D-165 -- it is evidence that
        # MODEL_KNOWLEDGE_SCORE and MIN_EVIDENCE_SCORE disagree about
        # what this tier is for.
        log_event(logger, "tool.model_knowledge", task=task.key,
                  claims=len(evidence), asked=len(claims),
                  dropped_inert=dropped_inert,
                  hedge_specific=flagged)
        return evidence

    # D-162: expose the router's own counters so the ladder can drain them
    # (tools/retrieval_chain.py explains what was being lost and why the
    # existing seam is the right one). getattr, not a direct reference:
    # `router` is duck-typed here -- several tests pass a hand-written
    # object with only complete_json -- and a tool that required a full
    # FallbackRouter would break them for a telemetry detail.
    drain = getattr(router, "drain_counters", None)
    if drain is not None:
        model_knowledge.drain_router_counts = drain  # type: ignore[attr-defined]
    return model_knowledge
