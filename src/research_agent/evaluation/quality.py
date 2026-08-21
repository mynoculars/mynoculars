"""
evaluation/quality.py — Cheap answer-quality signal for fallback routing.

Purpose:
    Give the FallbackRouter a 0..1 score for a free-text answer so it can
    decide whether the current provider's output is good enough to keep.

Responsibilities:
    - score_answer(): ask a JUDGE model to rate an answer, parse the score,
      and degrade gracefully (assume acceptable) when scoring itself fails
      — a broken scorer must never take down a working answer path.

History (P2-11):
    This used to ask the SAME provider that wrote the answer to grade its
    own work — cheap, but optimistic: a real run showed both the self-score
    AND the critic pass a report whose Cassandra/DynamoDB sections cited no
    retrieved evidence at all. The scorer is now always a DIFFERENT
    provider — specifically the next one in FallbackRouter's chain (see
    llm/router.py::FallbackRouter._score_quality, the only caller, which
    only invokes this when a next provider exists in the first place, so a
    judge is always available whenever this runs). Still no new external
    dependency — just a different existing provider doing the judging.

    Follow-up (same P2-11 batch): a real run showed the judge itself can be
    the one that's unavailable (Gemini rate-limited with a 429 right after
    Mistral served the answer) — fail-open handled that correctly (the
    answer was kept), but nothing distinguished "judge said 1.0" from
    "judge couldn't be reached, defaulted to 1.0" in telemetry. The optional
    `on_score_failed` callback below exists so llm/router.py can count that
    case separately (`llm_quality_calls_failed`) without changing this
    function's return type or its fail-open contract for any other caller.

Limitation (stated, not hidden):
    A different LLM judging free text is still an LLM judgement, not
    ground truth — it catches confidently-wrong or unsupported output far
    more reliably than same-model self-scoring did, but it is not a
    substitute for evidence-grounded verification. That remains the
    critic's job (D-22, agents/compilation.py::critic_node) — the two stay
    separate judges answering separate questions: this gate asks "is this
    raw answer usable", the critic asks "is this compiled report faithful
    and complete". One judge per question, per the router's own design note.
"""

import logging
from typing import Callable, List, Optional

from research_agent.llm.client import ChatClient, Message
from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

# A fixed prompt template, built once at import time (not inside the
# function below) since it never changes between calls — only the answer
# text at the very end varies per call. "TASK=quality" is the tag
# llm/client.py::StubClient looks for, exactly like every other prompt in
# prompts/templates.py.
#
# P2-11: reworded from "You wrote the answer below" (accurate when this was
# self-scoring) to "Another model wrote the answer below" (accurate now that
# `judge` is never the model that produced `answer`) — the wording is part
# of the prompt a real model reads, so it needs to match reality.
_SCORING_PROMPT = (
    "TASK=quality\n"
    "Another model wrote the answer below for the preceding request. Rate how "
    "well it answers the request on a 0.0-1.0 scale. Respond ONLY with JSON: "
    '{"score": <float>}\n\nANSWER:\n'
)

# FIX-1 (run p205.211 root cause, first link in the chain). The judge used to
# be handed `answer[:4000]` -- a raw character slice. A compiled report is
# routinely 7,000-11,000 characters, so the judge saw a report that stopped
# DEAD MID-WORD and correctly scored it as incomplete. Observed live: a
# 10,103-char report scored 0.45, a 4,670-char one scored 0.35, both against a
# 0.6 threshold, on consecutive runs. The gate was not measuring answer
# quality, it was measuring how far past 4,000 characters the answer ran --
# i.e. it penalised the compiler for being thorough, which is the opposite of
# its purpose.
#
# Two changes, both needed:
#   - a cap large enough that a normal compiled report is judged WHOLE
#     (nothing observed across the p205 runs exceeds ~11k chars),
#   - and when the cap IS hit, cut on a paragraph/line boundary and say so
#     explicitly, so an excerpt never reads as a truncated answer.
_MAX_ANSWER_CHARS = 16000
_EXCERPT_NOTE = (
    "\n\n[EXCERPT ENDS HERE. The answer above was shortened to fit this "
    "scoring request; it is NOT the end of the answer. Judge only the "
    "excerpt shown, and do NOT lower the score for appearing incomplete.]"
)


def _excerpt_for_judging(answer: str, max_chars: int = _MAX_ANSWER_CHARS) -> str:
    """Return `answer` whole, or a boundary-cut excerpt that says it is one.

    CALLED BY   score_answer below, on every scoring call.
    WHY         see the _MAX_ANSWER_CHARS comment above: a silent mid-word
                cut made the judge score length, not quality.
    """
    if len(answer) <= max_chars:
        return answer
    head = answer[:max_chars]
    # Prefer a paragraph break, then any line break, then a sentence end --
    # only accept one that lands in the last quarter of the excerpt, so a
    # document with no breaks near the cap doesn't collapse to a stub.
    floor = int(max_chars * 0.75)
    for sep in ("\n\n", "\n", ". "):
        idx = head.rfind(sep)
        if idx >= floor:
            head = head[:idx + len(sep)].rstrip()
            break
    return head + _EXCERPT_NOTE


def score_answer(judge: ChatClient, request_messages: List[Message], answer: str,
                 on_score_failed: Optional[Callable[[], None]] = None) -> float:
    """Return a 0..1 quality score for `answer`, as judged by `judge`.

    CALLED BY   llm/router.py::FallbackRouter._score_quality, which in
                turn is only called from complete() (never complete_json())
                — see router.py for why only free-text calls go through
                the quality gate.
    READS       nothing from ResearchState — this is a standalone utility
                function with no knowledge of the graph at all.
    CALLS       judge.complete_json(...) — note `judge` is (P2-11) always a
                DIFFERENT ChatClient than the one that produced `answer`:
                the caller always passes the NEXT provider in the fallback
                chain, never the answering one, by appending one more
                "user" message (the scoring prompt plus the answer text)
                onto the ORIGINAL request transcript, so the judge sees the
                full context the answer was originally responding to.
    RETURNS     a float clamped to the [0, 1] range.

    Parameters:
        judge: the model asked to grade the answer — never the model that
            produced it (see module docstring's P2-11 history note).
        request_messages: the original request transcript, for context.
        answer: the produced answer text.
        on_score_failed: optional zero-argument callback invoked ONLY on
            the fail-open path below (judge errored / bad JSON / score
            wasn't numeric) — never on a genuine low score. Lets a caller
            (llm/router.py) count "couldn't be scored" separately from
            "scored low" without this function's return type changing.
            None (the default) means "don't bother" — every existing
            caller that doesn't pass this keeps behaving identically.

    Returns:
        Parsed score clamped to [0, 1]; 1.0 if scoring itself errors
        (fail-open by design — see module header).
    """
    try:
        # request_messages + [...] CONCATENATES two lists into a new one —
        # this does not modify request_messages itself, it builds a fresh
        # list containing all of request_messages's items followed by the
        # one new scoring-prompt message.
        # _excerpt_for_judging (FIX-1) replaces a bare answer[:4000] slice.
        # The cap still exists — an enormous answer shouldn't be resent in
        # full just to be scored — but it is now large enough to pass a
        # normal report whole, and when it does bite it cuts on a boundary
        # and TELLS the judge it is an excerpt. See the constant's comment
        # above for the live traces that made this necessary.
        result = judge.complete_json(
            request_messages
            + [{"role": "user", "content": _SCORING_PROMPT + _excerpt_for_judging(answer)}]
        )
        # result.get("score", 1.0): read the "score" key if the judge's
        # JSON included one, otherwise default to 1.0 (treat a missing
        # field the same as "couldn't be scored, assume it's fine").
        score = float(result.get("score", 1.0))
        # max(0.0, min(1.0, score)) is the standard two-step "clamp" idiom:
        # min(1.0, score) caps the value at 1.0 from above; max(0.0, ...)
        # then floors THAT result at 0.0 from below. Net effect: whatever
        # `score` was, the returned value is guaranteed to sit inside
        # [0.0, 1.0], even if the judge returned something out of range
        # like 1.5 or -3.
        return max(0.0, min(1.0, score))
    except Exception as exc:  # noqa: BLE001
        # Anything going wrong here — the judge erroring, the JSON not
        # parsing, "score" being some non-numeric value that float() can't
        # convert — lands here. The module docstring's "Limitation" section
        # explains WHY this returns 1.0 (fail-open) rather than 0.0
        # (fail-closed): a broken scoring call should never be allowed to
        # accidentally reject a perfectly good answer.
        log_event(logger, "quality.score_failed", reason=type(exc).__name__)
        if on_score_failed is not None:
            on_score_failed()
        return 1.0
