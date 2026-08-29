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

Prompt budget (D-129, P6-1):
    The judge is no longer sent the request verbatim. `_digest_request`
    keeps every system message, drops the fenced evidence block from the
    last user message and caps what remains -- measured on a 97-item
    compile transcript, that is 32,873 characters down to ~3,200, before
    the answer is added. This is a COST change, not a policy change: the
    threshold, the fail-open contract, the callbacks and the returned
    value are all untouched. It is, however, a change to a prompt a real
    model reads, so scores recorded either side of it are not one
    population -- the same caveat D-106 attached to adding `reason`.

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
from typing import Callable, Dict, List, Optional, Tuple

from research_agent.guardrails.fencing import EVIDENCE_SPAN_RE
from research_agent.llm.client import (ChatClient, Message,
                                       classify_http_failure,
                                       estimate_prompt_tokens)
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
# D-106: `reason` added. The score alone has been decisive in five
# consecutive live runs and diagnostic in none of them -- p205.254-check
# rejected one compile at 0.5 and the next at 0.1, forcing the hop into
# the truncation that caused the E4, and nothing anywhere recorded WHY
# either number was chosen. A score you cannot interrogate is a number you
# can only tune blindly, which is exactly what this project's own D-54
# ordering forbids: measure first, build against what you measured.
#
# `score` is still FIRST and still the only required key. A judge that
# ignores the new field, and StubClient's canned {"score": 0.9}, both
# still parse exactly as before -- reason is read with .get() and its
# absence is not an error.
#
# THIS IS A PROMPT CHANGE, and it is recorded as one: asking for a
# justification can itself move a model's scores. Scores recorded before
# and after this change are therefore not one population, and a
# distribution that spans it should not be read as though they were.
# Taken deliberately: the alternative is another five runs of the judge
# being decisive and unexplained.
_SCORING_PROMPT = (
    "TASK=quality\n"
    "Another model wrote the answer below for the preceding request. Rate how "
    "well it answers the request on a 0.0-1.0 scale. Respond ONLY with JSON: "
    '{"score": <float>, "reason": "<one short sentence naming the single '
    'biggest thing that decided the score>"}\n\nANSWER:\n'
)

# Cap on the reason as it reaches a log line. A judge that ignores "one
# short sentence" must not be able to turn one JSON log record into a
# page -- the same reasoning cli.py::_failure_record truncates on.
_MAX_REASON_CHARS = 240

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


# D-129 (P6-1): how much of the ORIGINAL request the judge is shown.
#
# WHY A CAP EXISTS AT ALL. score_answer used to hand the judge
# `request_messages` VERBATIM, and for the only caller that reaches it
# (compiler_node's free-text call) that IS the entire compile prompt,
# evidence block included. Measured on a p205.267-check-shaped run --
# 4 goals, 97 evidence items -- the transcript is 32,873 characters,
# ~8,200 estimated tokens, of which 30,199 are the evidence block alone.
# Add the answer being judged and ONE scoring call costs ~12,000 tokens;
# three of them in a run is how p205.262/.264 met Gemini's 429 and how
# p205.267 met grok's quota, with `llm_quality_calls_failed == 3` and the
# gate inert on every attempt.
#
# WHY REMOVING THE EVIDENCE IS NOT A WEAKER GATE. This gate asks "is
# this raw answer usable" -- it has never been the grounding check, and
# treating it as one would contradict both this file's own module
# docstring and D-22/D-46, which put evidence-support judgment on the
# CRITIC. The critic still receives the evidence in full; nothing about
# what it sees changes here. A judge shown 30,000 characters of snippets
# to answer "does this read as an answer to the request" is paying for
# context it was never asked to use.
#
# WHAT THE JUDGE LOSES, stated rather than glossed: with the evidence
# gone it cannot notice that a specific claim is unsupported, and past a
# 3,000-character request it stops seeing the citation-format and
# attribution rules that trail the compile prompt. Neither was ever
# enforced here -- D-66's zero-citation gate, D-91's figure audit and
# the critic own those, deterministically and with the evidence in hand.
#
# 3000 IS DERIVED, NOT GUESSED: that same measured prompt's non-evidence
# body -- task line, question, per-goal coverage verdicts, citation and
# attribution rules -- is 2,674 characters, so a real compile request
# survives whole and an arbitrarily long one is still bounded. The HEAD
# is the end worth keeping: the question and the goals lead the prompt,
# the formatting rules trail it.
_MAX_REQUEST_CHARS = 3000

# Substituted IN PLACE of each removed span, so the judge is told the
# block was withheld rather than left to conclude the answer cites
# nothing. Same reasoning as _EXCERPT_NOTE above: an omission the judge
# cannot see is an omission it scores against.
_EVIDENCE_OMITTED_NOTE = (
    "\n[The retrieved-evidence block was removed from this scoring request. "
    "Judge the answer AS AN ANSWER TO THE REQUEST; do NOT lower the score "
    "because the evidence behind it is not shown here.]\n"
)
_REQUEST_EXCERPT_NOTE = (
    "\n[REQUEST EXCERPT ENDS HERE. The request continued with formatting "
    "instructions that do not affect this judgement.]"
)


def _boundary_cut(text: str, max_chars: int) -> str:
    """Cut `text` to at most max_chars on a natural boundary where one is
    close enough to the cap.

    CALLED BY   _excerpt_for_judging (the answer) and _digest_request
                (the request), below -- ONE implementation of "shorten
                this without stopping mid-word", not two that can drift.
                Extracted unchanged from _excerpt_for_judging, which is
                where the rule was first derived (FIX-1).

    Prefers a paragraph break, then any line break, then a sentence end,
    and only accepts one landing in the last quarter of the head -- so a
    block with no break anywhere near the cap is cut at the cap rather
    than collapsing to a stub.
    """
    head = text[:max_chars]
    floor = int(max_chars * 0.75)
    for sep in ("\n\n", "\n", ". "):
        idx = head.rfind(sep)
        if idx >= floor:
            return head[:idx + len(sep)].rstrip()
    return head


def _strip_evidence(text: str) -> Tuple[str, int]:
    """Replace every fenced evidence span with a note; return (text, chars
    removed).

    Uses guardrails/fencing.py's own EVIDENCE_SPAN_RE rather than a
    second literal copy of the delimiter -- see that module for why
    non-greedy matching is safe against content the fence already
    neutralised.
    """
    removed = sum(len(m.group(0)) for m in EVIDENCE_SPAN_RE.finditer(text))
    if not removed:
        return text, 0
    return EVIDENCE_SPAN_RE.sub(_EVIDENCE_OMITTED_NOTE, text), removed


def _digest_request(request_messages: List[Message],
                    max_chars: int = _MAX_REQUEST_CHARS
                    ) -> Tuple[List[Message], Dict[str, int]]:
    """Build the bounded context the judge sees instead of the whole
    request (D-129).

    CALLED BY   score_answer below, once per scoring call.
    RETURNS     (messages, stats) -- messages is every SYSTEM message
                verbatim plus at most ONE abridged user message; stats
                carries the three character counts the log line reports.

    SYSTEM MESSAGES ARE KEPT WHOLE, and that is load-bearing rather than
    tidy: prompts/templates.py::_SYSTEM is what tells the model to answer
    with ONLY a JSON object, which is exactly what complete_json then has
    to parse. Dropping it to save ~500 characters would buy tokens with
    parse failures. It is also OUR text, bounded by construction, and
    carries the never-obey-evidence rule.

    THE LAST USER MESSAGE IS THE REQUEST. Every builder in
    prompts/templates.py emits a two-message transcript and StubClient
    already reads `messages[-1]` for the same reason. Taking the last one
    rather than joining all of them means a multi-turn transcript's cap
    can never discard the turn actually being answered.

    An empty transcript yields no context messages at all -- the caller
    then sends the scoring prompt alone, which is what several existing
    callers (and tests passing []) already relied on.
    """
    system = [m for m in request_messages if m.get("role") == "system"]
    users = [m for m in request_messages if m.get("role") == "user"]
    request_text = (users[-1].get("content") or "") if users else ""
    digest, evidence_removed = _strip_evidence(request_text)
    if len(digest) > max_chars:
        digest = _boundary_cut(digest, max_chars) + _REQUEST_EXCERPT_NOTE
    stats = {"request_chars": len(request_text),
             "digest_chars": len(digest),
             "evidence_chars_removed": evidence_removed}
    if not digest.strip():
        return list(system), stats
    return list(system) + [{"role": "user", "content":
                            "REQUEST BEING ANSWERED (abridged):\n" + digest}], stats


def _excerpt_for_judging(answer: str, max_chars: int = _MAX_ANSWER_CHARS) -> str:
    """Return `answer` whole, or a boundary-cut excerpt that says it is one.

    CALLED BY   score_answer below, on every scoring call.
    WHY         see the _MAX_ANSWER_CHARS comment above: a silent mid-word
                cut made the judge score length, not quality.

    DELIBERATELY NOT TIGHTENED BY D-129: the answer is the thing being
    judged, and FIX-1's own live evidence (a 10,103-character report
    scored 0.45 for looking incomplete) is what a smaller cap here would
    reintroduce. D-129 cuts the REQUEST, which the judge was never asked
    to evaluate; it leaves the answer alone.
    """
    if len(answer) <= max_chars:
        return answer
    return _boundary_cut(answer, max_chars) + _EXCERPT_NOTE


def score_answer(judge: ChatClient, request_messages: List[Message], answer: str,
                 on_score_failed: Optional[Callable[[], None]] = None,
                 on_scored: Optional[Callable[[float, str], None]] = None
                 ) -> float:
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
                chain, never the answering one.
                D-129: what the judge receives is the request DIGEST
                (_digest_request — system messages verbatim, the last user
                message with its evidence block removed and capped at
                _MAX_REQUEST_CHARS) plus one "user" message carrying the
                scoring prompt and the answer. It used to be the ORIGINAL
                transcript verbatim, which for the compiler meant resending
                the whole evidence block on every scoring call.
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
        on_scored: optional callback invoked with (clamped_score, reason)
            ONLY when the judge genuinely returned a score (D-106) — never
            on the fail-open path, which is the whole point. The fabricated
            1.0 this function returns when scoring breaks is not a
            judgement, and folding it into a score distribution would make
            a dead judge look like a generous one. `reason` is "" when the
            judge did not supply one. Same optional-callback shape, and the
            same rationale, as on_score_failed above: the caller gets the
            detail it needs without this function's return type changing.

    Returns:
        Parsed score clamped to [0, 1]; 1.0 if scoring itself errors
        (fail-open by design — see module header).
    """
    # D-129: the judge is sent a BOUNDED digest of the request, not the
    # request itself -- system messages whole, the last user message with
    # its evidence block removed and capped. See _digest_request and
    # _MAX_REQUEST_CHARS above for what that costs and what it does not.
    # Built OUTSIDE the try: this is pure string work over data already in
    # hand, and folding it into the fail-open path would let a bug here
    # masquerade as a judge failure.
    context, digest_stats = _digest_request(request_messages)
    # _excerpt_for_judging (FIX-1) replaces a bare answer[:4000] slice.
    # The cap still exists -- an enormous answer shouldn't be resent in
    # full just to be scored -- but it is now large enough to pass a
    # normal report whole, and when it does bite it cuts on a boundary
    # and TELLS the judge it is an excerpt. See the constant's comment
    # above for the live traces that made this necessary.
    judge_messages = context + [
        {"role": "user", "content": _SCORING_PROMPT + _excerpt_for_judging(answer)}]
    # Logged BEFORE the call, unconditionally, so a scoring call that then
    # fails still records what it was about to spend -- the same reasoning
    # tools/retrieval_chain.py::chain.attempt states for logging an attempt
    # rather than only its success. estimate_prompt_tokens is D-93's own
    # ~4-chars-per-token estimator, reused rather than re-derived; it is
    # approximate and only ever read as a magnitude.
    log_event(logger, "quality.request_digested",
              judge=getattr(judge, "name", None),
              estimated_prompt_tokens=estimate_prompt_tokens(judge_messages),
              answer_chars=len(answer), **digest_stats)
    try:
        result = judge.complete_json(judge_messages)
        # result.get("score", 1.0): read the "score" key if the judge's
        # JSON included one, otherwise default to 1.0 (treat a missing
        # field the same as "couldn't be scored, assume it's fine").
        score = float(result.get("score", 1.0))
        # str() before slicing: a judge that answers with a number, a list
        # or null here would otherwise raise INSIDE the try and be
        # misreported as a scoring failure -- turning a usable score into
        # a fail-open 1.0 over a cosmetic field.
        reason = str(result.get("reason") or "")[:_MAX_REASON_CHARS]
        # max(0.0, min(1.0, score)) is the standard two-step "clamp" idiom:
        # min(1.0, score) caps the value at 1.0 from above; max(0.0, ...)
        # then floors THAT result at 0.0 from below. Net effect: whatever
        # `score` was, the returned value is guaranteed to sit inside
        # [0.0, 1.0], even if the judge returned something out of range
        # like 1.5 or -3.
        clamped = max(0.0, min(1.0, score))
        if on_scored is not None:
            on_scored(clamped, reason)
        return clamped
    except Exception as exc:  # noqa: BLE001
        # Anything going wrong here — the judge erroring, the JSON not
        # parsing, "score" being some non-numeric value that float() can't
        # convert — lands here. The module docstring's "Limitation" section
        # explains WHY this returns 1.0 (fail-open) rather than 0.0
        # (fail-closed): a broken scoring call should never be allowed to
        # accidentally reject a perfectly good answer.
        # D-110: the status code where there is one. `reason` alone said
        # "HTTPStatusError" and nothing more, which across five runs was
        # not enough to tell a dead judge from a busy one. httpx puts the
        # response on the exception; getattr twice rather than importing
        # httpx here, so this stays true for any client that raises
        # something response-shaped and harmless for one that does not.
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        # D-119: the same classification the provider call itself records,
        # repeated HERE because this is where the consequence lands. A
        # reader of `quality.score_failed` should not have to go and find
        # the matching llm.http_error to learn that the gate was inert
        # because an account has no credits. The body excerpt is short --
        # the full one is on the llm.http_error line -- but enough to name
        # the cause without a second lookup.
        kind = hint = None
        body = None
        if isinstance(status, int):
            kind, hint = classify_http_failure(status)
            text = getattr(response, "text", None)
            if isinstance(text, str) and text:
                body = text[:300]
        log_event(logger, "quality.score_failed", level=logging.WARNING,
                  reason=type(exc).__name__,
                  judge=getattr(judge, "name", None), status=status,
                  kind=kind, hint=hint, body=body,
                  effect="the quality gate did not run for this call "
                         "(fail-open: the answer was kept unscored)")
        if on_score_failed is not None:
            on_score_failed()
        return 1.0
