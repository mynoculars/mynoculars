"""
tests/unit/test_tracing.py — tracing.py's Tracer / NullTracer.

Covers the debug tracer used behind --debug / DEBUG_TRACE: recording an
LLM call and a retrieval hit, then flushing to a real file with the
expected content, and NullTracer's no-op guarantee when tracing is off.
"""


def test_tracer_records_and_flushes(tmp_path):
    from research_agent.tracing import Tracer

    t = Tracer("run-test", log_dir=str(tmp_path))
    t.record_llm("LOCAL PRIMARY (x)", "classify",
                 [{"role": "user", "content": "hello"}], '{"ok":1}', 10, 3, 1.5)
    t.record_retrieval("QDRANT (dense)", "redis vs memcached",
                       [{"title": "doc", "similarity": 0.9}])
    path = t.flush()
    assert path is not None
    # ResourceWarning fix: the file object from a bare open(...).read() is
    # only closed whenever the garbage collector gets around to it -- a
    # `with` block closes it deterministically the moment this line ends.
    # This only became visible once filterwarnings=always (pytest.ini)
    # stopped Python's default "once per location" dedup from hiding it.
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "RETRIEVED FROM LOCAL PRIMARY (X)" in text
    assert "node=classify" in text
    assert "RETRIEVED FROM QDRANT (DENSE)" in text
    assert "redis vs memcached" in text


def test_null_tracer_is_noop(tmp_path):
    from research_agent.tracing import NullTracer

    t = NullTracer()
    assert t.enabled is False
    t.record_llm("x", None, [], "", None, None, 0.0)
    t.record_retrieval("x", "q", [])
    assert t.flush() is None
