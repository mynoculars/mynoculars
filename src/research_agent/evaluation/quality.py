"""
evaluation/quality.py — Cheap answer-quality signal for fallback routing.

Purpose:
    Give the FallbackRouter a 0..1 score for a free-text answer so it can
    decide whether the primary model's output is good enough to keep.

Responsibilities:
    - score_answer(): ask the SAME model to rate its own answer, parse the
      score, and degrade gracefully (assume acceptable) when scoring itself
      fails — a broken scorer must never take down a working answer path.

Limitation (stated, not hidden):
    Self-evaluation is optimistic. It reliably catches catastrophic output
    (empty, off-task, truncated) — which is the failure mode of a small
    local model that this exists to guard — but will not catch subtle
    factual errors. A second-model judge is the documented future upgrade.
"""

import logging
from typing import List

from research_agent.llm.client import ChatClient, Message
from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

_SCORING_PROMPT = (
    "TASK=quality\n"
    "You wrote the answer below for the preceding request. Rate how well it "
    "answers the request on a 0.0-1.0 scale. Respond ONLY with JSON: "
    '{"score": <float>}\n\nANSWER:\n'
)


def score_answer(client: ChatClient, request_messages: List[Message], answer: str) -> float:
    """Return a 0..1 self-evaluated quality score for `answer`.

    Parameters:
        client: the model that produced the answer (it also scores it).
        request_messages: the original request transcript, for context.
        answer: the produced answer text.

    Returns:
        Parsed score clamped to [0, 1]; 1.0 if scoring itself errors
        (fail-open by design — see module header).
    """
    try:
        result = client.complete_json(
            request_messages + [{"role": "user", "content": _SCORING_PROMPT + answer[:4000]}]
        )
        score = float(result.get("score", 1.0))
        return max(0.0, min(1.0, score))
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "quality.score_failed", reason=type(exc).__name__)
        return 1.0
