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

# A fixed prompt template, built once at import time (not inside the
# function below) since it never changes between calls — only the answer
# text at the very end varies per call. "TASK=quality" is the tag
# llm/client.py::StubClient looks for, exactly like every other prompt in
# prompts/templates.py.
_SCORING_PROMPT = (
    "TASK=quality\n"
    "You wrote the answer below for the preceding request. Rate how well it "
    "answers the request on a 0.0-1.0 scale. Respond ONLY with JSON: "
    '{"score": <float>}\n\nANSWER:\n'
)


def score_answer(client: ChatClient, request_messages: List[Message], answer: str) -> float:
    """Return a 0..1 self-evaluated quality score for `answer`.

    CALLED BY   llm/router.py::FallbackRouter._passes_quality, which in
                turn is only called from complete() (never complete_json())
                — see router.py for why only free-text calls go through
                the quality gate.
    READS       nothing from ResearchState — this is a standalone utility
                function with no knowledge of the graph at all.
    CALLS       client.complete_json(...) — note this asks the SAME client
                that produced `answer` to now grade its own work, by
                appending one more "user" message (the scoring prompt plus
                the answer text) onto the ORIGINAL request transcript, so
                the model sees the full context it was originally
                responding to.
    RETURNS     a float clamped to the [0, 1] range.

    Parameters:
        client: the model that produced the answer (it also scores it).
        request_messages: the original request transcript, for context.
        answer: the produced answer text.

    Returns:
        Parsed score clamped to [0, 1]; 1.0 if scoring itself errors
        (fail-open by design — see module header).
    """
    try:
        # request_messages + [...] CONCATENATES two lists into a new one —
        # this does not modify request_messages itself, it builds a fresh
        # list containing all of request_messages's items followed by the
        # one new scoring-prompt message.
        # answer[:4000] is a SLICE taking only the first 4000 characters of
        # `answer` — a cheap guard against sending an enormous answer back
        # to the model just to be scored, which would cost tokens for very
        # little benefit once the answer is already long enough to judge.
        result = client.complete_json(
            request_messages + [{"role": "user", "content": _SCORING_PROMPT + answer[:4000]}]
        )
        # result.get("score", 1.0): read the "score" key if the model's
        # JSON included one, otherwise default to 1.0 (treat a missing
        # field the same as "couldn't be scored, assume it's fine").
        score = float(result.get("score", 1.0))
        # max(0.0, min(1.0, score)) is the standard two-step "clamp" idiom:
        # min(1.0, score) caps the value at 1.0 from above; max(0.0, ...)
        # then floors THAT result at 0.0 from below. Net effect: whatever
        # `score` was, the returned value is guaranteed to sit inside
        # [0.0, 1.0], even if the model returned something out of range
        # like 1.5 or -3.
        return max(0.0, min(1.0, score))
    except Exception as exc:  # noqa: BLE001
        # Anything going wrong here — the model erroring, the JSON not
        # parsing, "score" being some non-numeric value that float() can't
        # convert — lands here. The module docstring's "Limitation" section
        # explains WHY this returns 1.0 (fail-open) rather than 0.0
        # (fail-closed): a broken scoring call should never be allowed to
        # accidentally reject a perfectly good answer.
        log_event(logger, "quality.score_failed", reason=type(exc).__name__)
        return 1.0
