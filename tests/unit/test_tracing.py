"""
tests/unit/test_tracing.py — tracing.py's Tracer / NullTracer.

Post-unification (single instrumentation path, see logging_setup.py's
module docstring): Tracer no longer has record_llm()/record_retrieval()
methods of its own — it only turns narrative capture on and flushes one
run's buffer. These tests exercise that through the SAME log_event() call
business code now makes directly, then flush() the narrative and check the
file's content — the actual path llm/client.py and storage/*.py exercise.
"""

import logging

from research_agent.logging_setup import log_event, run_id_var


def test_tracer_enables_capture_and_flushes(tmp_path):
    from research_agent.tracing import Tracer

    t = Tracer("run-test", log_dir=str(tmp_path))
    assert t.enabled is True

    logger = logging.getLogger("research_agent.test_tracing")
    logger.setLevel(logging.INFO)
    token = run_id_var.set("run-test")
    try:
        # Same shape llm/client.py::OpenAICompatibleClient.complete emits,
        # with the tracer-only fields attached (t.enabled is True above).
        log_event(logger, "llm.call", provider="LOCAL PRIMARY (x)",
                  model="x", node="classify", latency_s=1.5,
                  prompt_tokens=10, completion_tokens=3,
                  prompt_messages=[{"role": "user", "content": "hello"}],
                  response='{"ok":1}')
        # Same shape storage/qdrant_store.py::QdrantStore.search emits.
        log_event(logger, "retrieval.raw", source="QDRANT (dense)",
                  query="redis vs memcached", hit_count=1,
                  hits=[{"title": "doc", "similarity": 0.9}])
    finally:
        run_id_var.reset(token)

    path = t.flush()
    assert path is not None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "LLM REQUEST" in text
    assert "classify" in text
    assert "SEARCH RESULTS" in text
    assert "redis vs memcached" in text


def test_null_tracer_is_noop(tmp_path):
    from research_agent.tracing import NullTracer

    t = NullTracer()
    assert t.enabled is False
    assert t.flush() is None


def test_disabled_tracer_omits_heavy_fields_from_flush(tmp_path):
    """A caller that checks `tracer.enabled` before attaching
    prompt_messages/response/hits (exactly what llm/client.py and
    storage/*.py do) never buffers anything for a NullTracer, since
    NullTracer never enables narrative capture in the first place."""
    from research_agent.tracing import NullTracer

    t = NullTracer()
    logger = logging.getLogger("research_agent.test_tracing")
    logger.setLevel(logging.INFO)
    token = run_id_var.set("run-disabled")
    try:
        trace_fields = ({"prompt_messages": [], "response": "x"}
                        if t.enabled else {})
        log_event(logger, "llm.call", provider="x", model="x", node=None,
                  latency_s=0.0, prompt_tokens=None, completion_tokens=None,
                  **trace_fields)
    finally:
        run_id_var.reset(token)
    assert t.flush() is None
