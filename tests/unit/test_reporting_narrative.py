"""
tests/unit/test_reporting_narrative.py -- reporting/narrative.py's
NarrativeBufferHandler and flush_narrative contracts.

NarrativeFormatter's own rendering (grouping by node, loop-boundary
detection, per-event-type sections) is already exercised end-to-end via
tests/unit/test_tracing.py's Tracer tests -- real log_event() calls
through a real Tracer, asserting on the rendered file's content. This
file covers the layer BELOW that: the buffering/pop contract
NarrativeBufferHandler provides, and flush_narrative's None-vs-path
return contract, neither of which test_tracing.py exercises directly
(it always goes through a full Tracer with real events to flush).
"""

import logging

from research_agent.logging_setup import run_id_var
from research_agent.reporting import narrative as narrative_module
from research_agent.reporting.narrative import NarrativeBufferHandler, _Event


def test_buffer_handler_groups_events_by_run_id():
    handler = NarrativeBufferHandler()
    record_a = logging.LogRecord("x", logging.INFO, "", 0, "msg-a", None, None)
    record_a.event_fields = {"run_id": "run-a"}
    record_b = logging.LogRecord("x", logging.INFO, "", 0, "msg-b", None, None)
    record_b.event_fields = {"run_id": "run-b"}
    handler.emit(record_a)
    handler.emit(record_b)
    a_events = handler.pop("run-a")
    b_events = handler.pop("run-b")
    assert len(a_events) == 1 and a_events[0].msg == "msg-a"
    assert len(b_events) == 1 and b_events[0].msg == "msg-b"


def test_buffer_handler_pop_drains_the_buffer():
    """pop() must remove what it returns -- a second pop for the same
    run_id must come back empty, not repeat the same events."""
    handler = NarrativeBufferHandler()
    record = logging.LogRecord("x", logging.INFO, "", 0, "msg", None, None)
    record.event_fields = {"run_id": "run-a"}
    handler.emit(record)
    first = handler.pop("run-a")
    second = handler.pop("run-a")
    assert len(first) == 1
    assert second == []


def test_buffer_handler_drops_an_event_with_no_correlatable_run_id():
    """No event_fields run_id AND no ambient run_id_var set -- nothing to
    correlate this line to, so it must be dropped rather than buffered
    under some placeholder key."""
    handler = NarrativeBufferHandler()
    record = logging.LogRecord("x", logging.INFO, "", 0, "msg", None, None)
    token = run_id_var.set("")
    try:
        handler.emit(record)
    finally:
        run_id_var.reset(token)
    assert handler.pop("") == []


def test_buffer_handler_falls_back_to_the_ambient_run_id_var():
    """A record with no explicit run_id in its event_fields must still be
    correlated via the ambient run_id_var -- the fallback every live call
    site relies on (only a few tests set run_id explicitly per-event)."""
    handler = NarrativeBufferHandler()
    record = logging.LogRecord("x", logging.INFO, "", 0, "msg", None, None)
    record.event_fields = {}
    token = run_id_var.set("ambient-run")
    try:
        handler.emit(record)
    finally:
        run_id_var.reset(token)
    assert len(handler.pop("ambient-run")) == 1


def test_flush_narrative_returns_none_when_narrative_logging_was_never_enabled(monkeypatch):
    monkeypatch.setattr(narrative_module, "_narrative_handler", None)
    assert narrative_module.flush_narrative("some-run") is None


def test_flush_narrative_returns_none_for_a_run_with_no_buffered_events():
    handler = NarrativeBufferHandler()
    import research_agent.reporting.narrative as mod
    original = mod._narrative_handler
    mod._narrative_handler = handler
    try:
        assert mod.flush_narrative("never-logged-anything") is None
    finally:
        mod._narrative_handler = original


def test_flush_narrative_writes_a_file_and_returns_its_path(tmp_path):
    handler = NarrativeBufferHandler()
    handler._buffers["run-x"] = [_Event(0.0, "mod::fn:1", "node.enter",
                                        {"node": "classify"}, "INFO")]
    import research_agent.reporting.narrative as mod
    original = mod._narrative_handler
    mod._narrative_handler = handler
    handler.setFormatter(narrative_module.NarrativeFormatter())
    try:
        path = mod.flush_narrative("run-x", log_dir=str(tmp_path))
        assert path is not None
        assert (tmp_path / "run-run-x.txt").exists()
    finally:
        mod._narrative_handler = original


# ---------------------------------------------------------------------------
# D-108 / D-109 -- the narrative shows the judge, and names its own events
# ---------------------------------------------------------------------------


def test_the_telemetry_block_reports_what_the_judge_decided():
    """The execution narrative is the OTHER per-run surface, and it had
    the same gap as the RESULT block: attempts and failures, never the
    judgement."""
    from research_agent.reporting.narrative import NarrativeFormatter

    rendered = NarrativeFormatter()._render_telemetry({
        "llm_quality_calls": 3, "llm_quality_calls_failed": 0,
        "llm_quality_scores_judged": 3, "llm_quality_score_mean": 0.5,
        "llm_quality_rejections": 2,
        "llm_quality_bands": {"very_low": 1, "mid": 1, "very_high": 1}})

    assert "Quality checks" in rendered, "the attempt count stays"
    assert "Quality judge:" in rendered
    assert "mean 0.5" in rendered


def test_the_telemetry_block_still_renders_for_a_pre_D_106_run():
    """Older rows carry no judgement keys at all."""
    from research_agent.reporting.narrative import NarrativeFormatter

    rendered = NarrativeFormatter()._render_telemetry({"llm_quality_calls": 0})

    assert "Quality judge:" in rendered


def test_every_event_this_codebase_warns_on_has_a_prose_label():
    """D-109. The _PROSE table falls back to a raw dotted code for an
    unlisted event -- deliberate, and honest, but a WARNING an operator
    is meant to act on should not be the thing that reads as an internal
    identifier next to prose neighbours."""
    from research_agent.reporting.narrative import NarrativeFormatter

    prose = NarrativeFormatter._PROSE
    for event in ("llm.quality_scored", "llm.quality_reject",
                  "llm.truncated_by_token_limit",
                  "llm.truncated_runaway_generation",
                  "llm.skipped_for_context", "llm.context_overflow",
                  "quality.judge_unreliable", "run_history.skipped"):
        assert event in prose, f"{event} renders as a raw dotted code"
        assert not prose[event].startswith("llm."), "a label, not the code"
