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
from typing import Any, Dict, List

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
_SPECIFIC_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_SPECIFIC_NUMBER = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|trillion|"
    r"per\s+(?:square\s+)?(?:km|kilometer|mile|capita|year))\b",
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
    return bool(_SPECIFIC_YEAR.search(text) and _SPECIFIC_NUMBER.search(text))

# Model-sourced evidence must clear settings.min_evidence_score (0.5) so it
# can actually mark a goal covered -- otherwise this tier would produce
# evidence the coverage rule ignores, the gather loop would never converge,
# and we would be back to the failure this module exists to remove. It sits
# deliberately BELOW the ~1.0 a document both retrieval legs agreed on, so
# corpus evidence always outranks recollection in the compiler's context.
DEFAULT_MODEL_SCORE = 0.6


def make_model_knowledge_tool(router: Any, score: float = DEFAULT_MODEL_SCORE,
                              max_claims: int = 4):
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
            if confidence < 0.5:
                continue
            evidence.append(Evidence(
                task_key=task.key, goal_id=task.goal_id, source="model",
                content=text[:800],
                # Scale within a narrow band so a confident recollection
                # still cannot outrank a document both legs agreed on.
                score=round(min(score, score * confidence + 0.05), 4),
                volatility=Volatility.SEMI_STABLE,
                # Guardrail G3 -- deterministic, evaluated on the same
                # text the model just produced, independent of (and not
                # replacing) the confidence gate above.
                hedge_specific=_looks_overspecific(text)))
        flagged = sum(1 for e in evidence if e.hedge_specific)
        log_event(logger, "tool.model_knowledge", task=task.key,
                  claims=len(evidence), asked=len(claims),
                  hedge_specific=flagged)
        return evidence

    return model_knowledge
