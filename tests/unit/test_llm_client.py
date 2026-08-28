"""
tests/unit/test_llm_client.py — llm/client.py's StubClient / _extract_json.

Deliberately thin: StubClient itself is exercised implicitly throughout
this whole suite (it IS the default LLM for every offline test), and its
per-TASK canned responses are covered by whatever test actually needs
that behavior (e.g. test_orchestration_graph.py, the integration tests).
This file covers only _extract_json's own parsing robustness, which has
no other natural home.
"""

import json
import logging

import httpx
import pytest

from research_agent.llm.client import (
    OpenAICompatibleClient, TruncatedGenerationError, _extract_json,
    _truncate_at_sentinel)


def test_prompt_tag_covers_every_node_that_calls_an_llm():
    """Every node this codebase's graph.py actually wires an LLM through
    is expected to be in the table. Written against the KNOWN node names
    from orchestration/graph.py rather than iterating PROMPT_VERSIONS
    itself, so a typo'd or accidentally-removed entry is caught -- testing
    against the registry's own keys would just prove it equals itself."""
    from research_agent.llm.client import _prompt_tag_for_node
    llm_calling_nodes = {"classify", "goal_manager", "task_expander",
                          "gap_generator", "compiler", "critic"}
    for node in llm_calling_nodes:
        tag = _prompt_tag_for_node(node)
        assert "prompt_name" in tag and "prompt_version" in tag, (
            f"{node!r} calls an LLM but has no PROMPT_VERSIONS entry")


def test_prompt_tag_is_empty_not_placeholder_for_an_untagged_node():
    """merger sometimes calls detect_contradictions and sometimes calls
    nothing -- it is deliberately absent from the table (see the
    registry's own docstring) rather than mis-tagged. Absent keys, not a
    placeholder value, so a Langfuse query grouping by prompt_name can
    tell "tagged" from "untagged" apart."""
    from research_agent.llm.client import _prompt_tag_for_node
    assert _prompt_tag_for_node("merger") == {}
    assert _prompt_tag_for_node(None) == {}
    assert _prompt_tag_for_node("some_future_node") == {}


def test_prompt_tag_values_match_the_templates_registry_exactly():
    from research_agent.llm.client import _prompt_tag_for_node
    from research_agent.prompts.templates import PROMPT_VERSIONS
    for node, (name, version) in PROMPT_VERSIONS.items():
        assert _prompt_tag_for_node(node) == {
            "prompt_name": name, "prompt_version": version}


def test_stub_json_fence_tolerance():
    """Regression guard for _extract_json's fence stripping."""
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


# ---------------------------------------------------------------------------
# P205 regression: a LEADING sentinel must not annihilate the response
# ---------------------------------------------------------------------------


def test_leading_sentinel_does_not_destroy_the_whole_response():
    """Live (runs p205.67/.70/.71-check): the critic node produced a
    sentinel at index 0 followed by the real answer -- raw_chars 3898,
    kept_chars 0. complete() returned "", complete_json() raised
    JSONDecodeError, and EVERY structured critic call fell back to the
    secondary provider despite the local model answering correctly."""
    raw = '<|im_end|>{"passed": false, "score": 0.2, "notes": ["thin"]}'
    kept = _truncate_at_sentinel(raw)
    assert kept == '{"passed": false, "score": 0.2, "notes": ["thin"]}'
    assert _extract_json(kept)["passed"] is False


def test_trailing_sentinel_behaviour_is_unchanged():
    """The overwhelmingly common case must stay byte-identical."""
    assert _truncate_at_sentinel('{"intent": "Comparison"}<|im_end|>') == \
        '{"intent": "Comparison"}'


def test_leading_sentinel_then_answer_then_runaway_keeps_only_the_answer():
    raw = '<|im_end|>{"passed": true}<|im_end|>\nsystem\nfake continuation'
    assert _truncate_at_sentinel(raw) == '{"passed": true}'


def test_response_of_nothing_but_sentinels_is_returned_raw_not_empty():
    """Returning "" would look like a successful call that produced
    nothing; the caller should see the real (useless) response instead."""
    assert _truncate_at_sentinel("<|im_end|><|eot_id|>") != ""


def test_json_path_recovers_the_answer_after_a_leading_fragment():
    """P205 regression (runs p205.98/.100-check): the local model emitted a
    short fragment, a sentinel, then the real JSON -- raw_chars 2895
    kept_chars 21, raw_chars 1153 kept_chars 18. complete() keeps the first
    segment (correct for prose), so every structured call raised
    JSONDecodeError and paid a fallback hop despite a valid answer being
    present."""
    import threading

    raw = 'Here is the answer:<|im_end|>{"goals": [{"goal_id": "g1"}]}'

    class _Client(OpenAICompatibleClient):
        def __init__(self):  # noqa: D107 - bypass HTTP setup entirely
            self._raw = threading.local()

        def complete(self, messages, temperature=0.2):
            # Exactly what the real complete() does: stash the untruncated
            # text, return the truncated one.
            self._raw.text = raw
            return _truncate_at_sentinel(raw)

    client = _Client()
    assert client.complete([]) == "Here is the answer:", "prose rule unchanged"
    assert client.complete_json([]) == {"goals": [{"goal_id": "g1"}]}


def test_free_text_path_still_kills_a_runaway_continuation():
    """The JSON fix must not weaken the prose rule: everything after the
    model's first end-of-turn is hallucinated continuation."""
    assert _truncate_at_sentinel(
        "the real report<|im_end|>\nsystem\nfake second conversation"
    ) == "the real report"


def test_sentinel_segments_returns_every_non_empty_run():
    from research_agent.llm.client import sentinel_segments
    assert sentinel_segments("a<|im_end|><|eot_id|>b") == ["a", "b"]


# ---------------------------------------------------------------------------
# Guardrail G6 (P205 Phase 2): max_tokens generation budget
# ---------------------------------------------------------------------------


def _client_with_mock_transport(handler, **kwargs):
    """Build a real OpenAICompatibleClient, then swap its httpx.Client
    for one backed by a MockTransport -- this exercises the REAL
    complete() request-construction path, not a reimplementation of it."""
    client = OpenAICompatibleClient("primary", "http://x", "key",
                                    "model-x", **kwargs)
    client._http = httpx.Client(transport=httpx.MockTransport(handler),
                                base_url="http://x")
    return client


def test_complete_sends_max_tokens_when_configured():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}]})

    client = _client_with_mock_transport(handler, max_tokens=777)
    client.complete([{"role": "user", "content": "hi"}])
    assert captured["body"]["max_tokens"] == 777


def test_complete_omits_max_tokens_when_not_configured():
    """Backward compatibility: a caller that doesn't pass max_tokens at
    all (the pre-G6 shape) must produce a byte-identical request body --
    no max_tokens key at all, not max_tokens=None serialized in."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}]})

    client = _client_with_mock_transport(handler)
    client.complete([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in captured["body"]



# ---------------------------------------------------------------------------
# FIX-2 — finish_reason == "length" is a failure, not an answer
#
# Diagnosed from run p205.211: gemini-3.5-flash returned 162 completion tokens
# ending mid-number, and the client handed that fragment back as a finished
# report because nothing ever looked at finish_reason.
# ---------------------------------------------------------------------------


def test_length_truncated_generation_raises_instead_of_returning_a_fragment():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "India's growth moderates to around 6"},
                         "finish_reason": "length"}]})

    client = _client_with_mock_transport(handler, max_tokens=4096)
    with pytest.raises(TruncatedGenerationError):
        client.complete([{"role": "user", "content": "compile"}])


# ---------------------------------------------------------------------------
# D-102 -- WHOSE ceiling truncated the generation
#
# Diagnosed from run p205.254-check: gemini-3.5-flash reported
# finish_reason=length at completion_tokens 616 and 2150 against
# max_tokens 8192, and the log line reported "max_tokens: 8192" both
# times -- reading as "we hit our own limit" and sending the operator to
# the wrong config file.
# ---------------------------------------------------------------------------


def _truncating_handler(completion_tokens, reasoning_tokens=None):
    def handler(request):
        usage = {"prompt_tokens": 100, "completion_tokens": completion_tokens}
        if reasoning_tokens is not None:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "cut off mid-"},
                         "finish_reason": "length"}],
            "usage": usage})
    return handler


def test_cap_is_attributed_to_the_provider_when_our_budget_was_not_reached():
    """The p205.254-check shape: we asked for 8192, the provider stopped
    at 616. Our budget was demonstrably not the binding constraint, and
    the message must not send anyone to LLM_MAX_TOKENS."""
    client = _client_with_mock_transport(_truncating_handler(616),
                                         max_tokens=8192)
    with pytest.raises(TruncatedGenerationError) as exc:
        client.complete([{"role": "user", "content": "compile"}])
    assert "PROVIDER's own ceiling" in str(exc.value)
    assert "completion_tokens=616" in str(exc.value)


def test_cap_is_attributed_to_us_when_the_provider_reached_our_number():
    """The other half: when the provider really did stop at the number we
    sent, raising LLM_MAX_TOKENS IS the right advice and the message must
    say so."""
    client = _client_with_mock_transport(_truncating_handler(4096),
                                         max_tokens=4096)
    with pytest.raises(TruncatedGenerationError) as exc:
        client.complete([{"role": "user", "content": "compile"}])
    assert "OUR max_tokens=4096" in str(exc.value)


def test_reasoning_tokens_count_toward_our_budget_when_reported():
    """A reasoning model spends the output budget on tokens
    completion_tokens does not report. 600 completion + 3496 reasoning
    IS our 4096 -- attributing that to the provider would be wrong."""
    client = _client_with_mock_transport(
        _truncating_handler(600, reasoning_tokens=3496), max_tokens=4096)
    with pytest.raises(TruncatedGenerationError) as exc:
        client.complete([{"role": "user", "content": "compile"}])
    assert "OUR max_tokens=4096" in str(exc.value)
    assert "reasoning_tokens=3496" in str(exc.value)


def test_reasoning_tokens_are_absent_from_the_message_when_unreported():
    """Providers with no reasoning concept omit the field entirely. The
    message must not invent a zero for them."""
    client = _client_with_mock_transport(_truncating_handler(616),
                                         max_tokens=8192)
    with pytest.raises(TruncatedGenerationError) as exc:
        client.complete([{"role": "user", "content": "compile"}])
    assert "reasoning_tokens" not in str(exc.value)


def test_stop_finish_reason_is_returned_normally():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "a complete answer"},
                         "finish_reason": "stop"}]})

    client = _client_with_mock_transport(handler)
    assert client.complete([{"role": "user", "content": "x"}]) == "a complete answer"


def test_absent_finish_reason_is_returned_normally():
    """Providers that omit finish_reason entirely (and every existing test
    fixture in this file) must behave exactly as before the check existed."""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}]})

    client = _client_with_mock_transport(handler)
    assert client.complete([{"role": "user", "content": "x"}]) == "hi"


def test_length_after_a_sentinel_trim_is_kept_not_raised():
    """A runaway generation that reached its own end-of-turn and was then cut
    off at the token limit has a COMPLETE answer before the sentinel — the
    `length` describes the discarded tail. _truncate_at_sentinel already
    salvages it, so raising here would throw away a good answer."""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "the real answer<|im_end|>then junk that ran"},
                         "finish_reason": "length"}]})

    client = _client_with_mock_transport(handler)
    assert client.complete([{"role": "user", "content": "x"}]) == "the real answer"


def test_truncated_generation_error_makes_the_router_fall_back():
    """End-to-end intent: a length-truncated provider must behave exactly
    like a transport error as far as FallbackRouter is concerned."""
    from research_agent.llm.router import FallbackRouter

    def truncating(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "cut off mid-"},
                         "finish_reason": "length"}]})

    class _Good:
        name = "good"
        def complete(self, messages, temperature=0.2):
            return "complete answer"
        def complete_json(self, messages, temperature=0.0):
            return {"score": 0.9}

    chain = FallbackRouter([_client_with_mock_transport(truncating), _Good()],
                           quality_threshold=0.6)
    assert chain.complete([{"role": "user", "content": "x"}]) == "complete answer"


# ---------------------------------------------------------------------------
# D-93: recognising a context-overflow rejection
# ---------------------------------------------------------------------------

from research_agent.llm.client import (estimate_prompt_tokens,  # noqa: E402
                                       looks_like_context_overflow)


def test_context_overflow_bodies_are_recognised():
    """Without this every one of these arrives as
    `llm.fallback reason=HTTPStatusError`, indistinguishable from a 429 or
    a dead port -- and reads as flakiness when it is deterministic."""
    for body in ("the request exceeds context length of 1536",
                 "This model's maximum context length is 1536 tokens",
                 "prompt is too long",
                 "ERROR: too many tokens in prompt"):
        assert looks_like_context_overflow(body), body


def test_unrelated_errors_are_not_mistaken_for_overflow():
    """A false positive here would mislabel a transient failure as a
    permanent one and send someone to change `-c` for nothing."""
    for body in ("rate limit exceeded", "internal server error",
                 "invalid api key", "", None):
        assert not looks_like_context_overflow(body)


def test_token_estimation_is_roughly_four_characters_per_token():
    """Approximate on purpose -- the alternative is a tokenizer dependency
    to answer "is this obviously too big for a 1536-token window"."""
    assert 900 <= estimate_prompt_tokens(
        [{"role": "user", "content": "x" * 4000}]) <= 1100
    assert estimate_prompt_tokens([]) == 0


# ---------------------------------------------------------------------------
# D-107 -- a routine sentinel trim is not a runaway
#
# p205.254-check logged this at WARNING on all five local-primary calls,
# discarding 10, 10, 11, 11 and 11 characters. Five WARNINGs a run for a
# non-event is how a real one gets scrolled past.
# ---------------------------------------------------------------------------


def _sentinel_handler(tail):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "the real answer<|im_end|>" + tail},
                         "finish_reason": "stop"}]})
    return handler


def test_a_bare_sentinel_trim_logs_at_info_not_warning(caplog):
    """The observed shape: the model emits its own end-of-turn token as
    literal text and nothing follows it."""
    client = _client_with_mock_transport(_sentinel_handler(""))

    with caplog.at_level(logging.INFO):
        assert client.complete([{"role": "user", "content": "x"}]) == "the real answer"

    events = [r for r in caplog.records
              if "llm.truncated_runaway_generation" in r.message]
    assert events and all(r.levelno == logging.INFO for r in events)


def test_a_real_runaway_still_warns(caplog):
    """What the guard was built for: an entire fabricated continuation."""
    client = _client_with_mock_transport(_sentinel_handler("z" * 400))

    with caplog.at_level(logging.INFO):
        client.complete([{"role": "user", "content": "x"}])

    events = [r for r in caplog.records
              if "llm.truncated_runaway_generation" in r.message]
    assert events and all(r.levelno == logging.WARNING for r in events)


def test_the_level_follows_the_discarded_size_not_the_response_size():
    """An absolute threshold, not a ratio. The live trims ran 1% to 18%
    of the response depending only on how SHORT the response was, so a
    ratio would have flagged the 54-character classify call and cleared
    the identical 1,084-character one."""
    from research_agent.llm.client import _RUNAWAY_WARN_CHARS

    assert _RUNAWAY_WARN_CHARS > len("<|im_end|>"), "a bare sentinel is quiet"
    assert _RUNAWAY_WARN_CHARS < 400, "a fabricated continuation is not"


def test_an_untrimmed_response_logs_nothing_at_all(caplog):
    """Unchanged: the overwhelming majority of calls trim nothing."""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "a clean answer"},
                         "finish_reason": "stop"}]})

    client = _client_with_mock_transport(handler)
    with caplog.at_level(logging.INFO):
        client.complete([{"role": "user", "content": "x"}])

    assert not [r for r in caplog.records
                if "llm.truncated_runaway_generation" in r.message]


# ---------------------------------------------------------------------------
# D-110 -- a 4xx that is not a context overflow is still worth reading
#
# Runs p205.260/.261: gemini returned 4xx on every call it was given, and
# the entire run record said "HTTPStatusError". A retired model name
# (404), a bad key (401/403) and an exhausted quota (429) were
# indistinguishable, and the quality gate had been inert for five runs
# with no way to learn why from the logs.
# ---------------------------------------------------------------------------


def _status_handler(status, body):
    def handler(request):
        return httpx.Response(status, text=body)
    return handler


def test_a_plain_4xx_logs_its_status_and_body(caplog):
    client = _client_with_mock_transport(
        _status_handler(404, '{"error": {"message": "model not found"}}'))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(httpx.HTTPStatusError):
            client.complete([{"role": "user", "content": "x"}])

    rec = [r for r in caplog.records if "llm.http_error" in r.message]
    assert rec, "a 4xx must not vanish into a bare exception class name"
    # log_event puts the payload on record.event_fields (extra=), never
    # in the formatted message -- see logging_setup.py::log_event.
    fields = rec[0].event_fields
    assert fields["status"] == 404
    assert "model not found" in fields["body"]
    assert fields["provider"] == "primary" and fields["model"] == "model-x"


def test_the_status_code_is_what_separates_the_realistic_causes(caplog):
    """404 wrong model, 401/403 bad key, 429 quota. The point of logging
    the number is that these need different fixes."""
    for status in (401, 403, 429, 500):
        caplog.clear()
        client = _client_with_mock_transport(_status_handler(status, "nope"))
        with caplog.at_level(logging.WARNING):
            with pytest.raises(httpx.HTTPStatusError):
                client.complete([{"role": "user", "content": "x"}])
        assert any(r.event_fields["status"] == status for r in caplog.records
                   if "llm.http_error" in r.message)


def test_a_context_overflow_still_gets_its_own_specific_event(caplog):
    """D-93's event carries the estimated/configured token fields that the
    generic one cannot. The new branch must not swallow it."""
    client = _client_with_mock_transport(
        _status_handler(400, "the request exceeds the maximum context length"),
        context_tokens=1536)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(httpx.HTTPStatusError):
            client.complete([{"role": "user", "content": "x"}])

    msgs = " ".join(r.message for r in caplog.records)
    assert "llm.context_overflow" in msgs
    assert "llm.http_error" not in msgs, "one event per failure, not two"


def test_a_2xx_logs_no_http_error_at_all(caplog):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]})

    client = _client_with_mock_transport(handler)
    with caplog.at_level(logging.WARNING):
        client.complete([{"role": "user", "content": "x"}])

    assert not [r for r in caplog.records if "llm.http_error" in r.message]


# ---------------------------------------------------------------------------
# D-119 -- a status is a number; an operator needs the failure CLASS
#
# Run p205.265-check: a 403 from xAI meant "this team has no credits" -- an
# account problem, not a bug, not transient, and not fixable by a retry.
# ---------------------------------------------------------------------------


def test_the_statuses_an_operator_acts_on_are_each_named():
    from research_agent.llm.client import classify_http_failure

    assert classify_http_failure(401)[0] == "auth_failed"
    assert classify_http_failure(403)[0] == "permission_denied"
    assert classify_http_failure(404)[0] == "model_or_endpoint_not_found"
    assert classify_http_failure(429)[0] == "quota_or_rate_limit"
    for status in (500, 502, 503):
        assert classify_http_failure(status)[0] == "provider_unavailable"


def test_an_unmapped_status_gets_no_invented_advice():
    """An unrecognised status still reports its number and its body. What
    it must not do is guess at a remedy."""
    from research_agent.llm.client import classify_http_failure

    kind, hint = classify_http_failure(418)
    assert kind == "http_error"
    assert hint == ""


def test_the_403_that_started_this_is_logged_with_kind_hint_and_body(caplog):
    """The exact shape of run p205.265-check's failure, end to end."""
    body = ('{"code":"permission-denied","error":"Your newly created team '
            'doesn\'t have any credits or licenses yet."}')
    client = _client_with_mock_transport(
        lambda request: httpx.Response(403, text=body))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(httpx.HTTPStatusError):
            client.complete([{"role": "user", "content": "compile"}])

    f = [r for r in caplog.records if "llm.http_error" in r.message][0].event_fields
    assert f["status"] == 403
    assert f["kind"] == "permission_denied"
    assert "credits" in f["hint"]
    assert "no credits" in f["body"] or "credits or licenses" in f["body"]
