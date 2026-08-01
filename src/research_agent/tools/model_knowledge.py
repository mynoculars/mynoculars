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
from typing import Any, Dict, List

from research_agent.logging_setup import log_event
from research_agent.prompts import templates
from research_agent.state import Evidence, SearchTask, Volatility

logger = logging.getLogger(__name__)

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
                volatility=Volatility.SEMI_STABLE))
        log_event(logger, "tool.model_knowledge", task=task.key,
                  claims=len(evidence), asked=len(claims))
        return evidence

    return model_knowledge
