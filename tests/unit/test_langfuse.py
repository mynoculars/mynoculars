"""
tests/unit/test_langfuse.py — Phase 3 Langfuse observability module.

WHY THIS FILE EXISTS: this is offline-only, matching the rest of
tests/unit/ (no real Langfuse project, no network). It proves the two
guarantees Phase 3 actually depends on:

  1. Disabled (the default) is genuinely zero-cost: no SDK import
     attempted, no client built, every thin call a silent no-op.
  2. Enabled-but-broken (no SDK installed, bad credentials, a client
     that raises) degrades the same way -- never propagates an
     exception into the caller.

It does NOT test against a live Langfuse project (there isn't one in
this environment) -- see the module's own docstrings for what was
additionally confirmed by hand against the real, installed SDK.
"""

import sys
import types

import pytest

from research_agent.config import Settings
from research_agent.langfuse import client as lf_client
from research_agent.langfuse.helpers import thread_id_from_config, traced_node
from research_agent.langfuse.masking import (
    MODE_ALL,
    MODE_OFF,
    MODE_PATTERNS,
    build_mask,
    redact_text,
    resolve_mode,
)
from research_agent.langfuse.observer import Observer
from research_agent.langfuse.pricing import TokenUsage, calculate_cost


def _settings(**overrides):
    return Settings(**overrides)


# ---------------------------------------------------------------------------
# client.build_client -- the one place the SDK is ever imported
# ---------------------------------------------------------------------------

def test_build_client_returns_none_when_disabled():
    assert lf_client.build_client(_settings(langfuse_enabled=False)) is None


def test_build_client_returns_none_without_credentials():
    s = _settings(langfuse_enabled=True, langfuse_public_key="",
                  langfuse_secret_key="")
    assert lf_client.build_client(s) is None


def test_build_client_returns_none_if_sdk_import_fails(monkeypatch):
    """Simulates `langfuse` not being installed at all -- disabled must
    degrade to None, never raise, regardless of what settings say."""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "langfuse":
            raise ImportError("no langfuse installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                  langfuse_secret_key="sk")
    assert lf_client.build_client(s) is None


def test_build_client_returns_none_if_construction_raises(monkeypatch):
    """A real `langfuse` module whose Langfuse(...) constructor raises
    (bad host, version mismatch, whatever) must still degrade to None."""
    fake_module = types.ModuleType("langfuse")

    class _ExplodingLangfuse:
        def __init__(self, **kwargs):
            raise RuntimeError("boom")

    fake_module.Langfuse = _ExplodingLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                  langfuse_secret_key="sk")
    assert lf_client.build_client(s) is None


def test_build_client_returns_client_on_success(monkeypatch):
    fake_module = types.ModuleType("langfuse")
    built_kwargs = {}

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            built_kwargs.update(kwargs)

    fake_module.Langfuse = _FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                  langfuse_secret_key="sk", langfuse_host="https://example.test")
    client = lf_client.build_client(s)
    assert isinstance(client, _FakeLangfuse)
    assert built_kwargs["host"] == "https://example.test"
    # Masking must reach the constructor, not just exist as a module.
    assert callable(built_kwargs["mask"])


def test_build_client_passes_no_mask_when_masking_is_off(monkeypatch):
    """mask=None makes the SDK skip masking entirely rather than call an
    identity function, so "off" must produce None, not a callable."""
    fake_module = types.ModuleType("langfuse")
    built_kwargs = {}

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            built_kwargs.update(kwargs)

    fake_module.Langfuse = _FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    s = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                  langfuse_secret_key="sk", langfuse_mask_mode="off")
    lf_client.build_client(s)
    assert built_kwargs["mask"] is None


# ---------------------------------------------------------------------------
# masking -- what actually leaves the process
# ---------------------------------------------------------------------------

def test_mask_mode_off_returns_no_callable():
    assert build_mask(_settings(langfuse_mask_mode="off")) is None


def test_mask_default_is_patterns_not_off():
    """A safety control that ships inert is the mistake this default
    exists to avoid -- assert it explicitly."""
    assert _settings().langfuse_mask_mode == MODE_PATTERNS
    assert build_mask(_settings()) is not None


def test_redact_text_catches_each_supported_shape():
    assert "[REDACTED:email]" in redact_text("write to a.b+x@example.co.uk today")
    assert "[REDACTED:bearer]" in redact_text("Authorization: Bearer abcdef1234567890")
    assert "[REDACTED:api_key]" in redact_text("key sk-abcdefghij0123456789 used")
    assert "[REDACTED:card]" in redact_text("paid with 4111 1111 1111 1111 ok")


def test_redact_text_leaves_ordinary_research_prose_alone():
    """A pattern that fires on corpus text would destroy the traces this
    package exists to produce -- years and plain figures must survive."""
    prose = "In 1994 the study of 250 subjects reported a 12.5% increase."
    assert redact_text(prose) == prose


def test_mask_patterns_preserves_structure_and_non_pii():
    mask = build_mask(_settings(langfuse_mask_mode="patterns"))
    out = mask(data={"messages": [{"role": "user",
                                   "content": "mail me at x@y.com"}],
                     "depth": 3, "ok": True, "none": None})
    assert out["depth"] == 3
    assert out["ok"] is True
    assert out["none"] is None
    assert out["messages"][0]["role"] == "user"
    assert "x@y.com" not in out["messages"][0]["content"]
    assert "[REDACTED:email]" in out["messages"][0]["content"]


def test_mask_all_removes_every_string_leaf_but_keeps_shape():
    mask = build_mask(_settings(langfuse_mask_mode="all"))
    out = mask(data={"report": "a long compiled report",
                     "tokens": 1200,
                     "goals": ["g1", "g2"]})
    assert out["report"] == "[REDACTED]"
    assert out["tokens"] == 1200
    assert out["goals"] == ["[REDACTED]", "[REDACTED]"]


def test_mask_stringifies_and_redacts_unknown_objects():
    """The SDK would serialize an unknown object with default=str, which
    would carry its repr through unmasked -- so it must be redacted here."""
    class _Thing:
        def __repr__(self):
            return "<Thing owner=a@b.com>"

    mask = build_mask(_settings(langfuse_mask_mode="patterns"))
    out = mask(data={"thing": _Thing()})
    assert "a@b.com" not in out["thing"]


def test_mask_survives_a_repr_that_raises():
    class _Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    mask = build_mask(_settings(langfuse_mask_mode="patterns"))
    assert mask(data=_Hostile()) == "[REDACTED]"


def test_mask_caps_recursion_depth():
    """Deeply nested payloads must not turn masking into runaway
    recursion on the request path."""
    payload = "leaf@example.com"
    for _ in range(200):
        payload = {"n": payload}
    mask = build_mask(_settings(langfuse_mask_mode="patterns"))
    out = mask(data=payload)          # must not raise
    flat = str(out)
    assert "leaf@example.com" not in flat
    assert "[REDACTED:depth]" in flat


def test_unrecognized_mask_mode_fails_toward_more_redaction():
    """An unparseable setting must never silently mean "send everything"."""
    assert resolve_mode("nonsense") == MODE_PATTERNS
    assert resolve_mode("") == MODE_PATTERNS
    assert resolve_mode(None) == MODE_PATTERNS
    assert resolve_mode("OFF") == MODE_OFF
    assert resolve_mode("  All ") == MODE_ALL
    assert build_mask(_settings(langfuse_mask_mode="nonsense")) is not None


# ---------------------------------------------------------------------------
# pricing.calculate_cost -- never hardcoded, settings-driven
# ---------------------------------------------------------------------------

def test_cost_is_zero_for_local_primary_by_default():
    s = _settings()
    cost = calculate_cost(s, "primary", TokenUsage(prompt_tokens=1000, completion_tokens=500))
    assert cost.total_usd == 0.0


def test_cost_uses_configured_rates_not_hardcoded_ones():
    s = _settings(langfuse_price_mistral_in_per_1m=2.0,
                 langfuse_price_mistral_out_per_1m=6.0)
    cost = calculate_cost(s, "mistral", TokenUsage(prompt_tokens=1_000_000,
                                                    completion_tokens=1_000_000))
    assert cost.input_cost_usd == 2.0
    assert cost.output_cost_usd == 6.0
    assert cost.total_usd == 8.0


def test_cost_is_none_for_an_unrecognized_provider():
    assert calculate_cost(_settings(), "some_new_provider", TokenUsage()) is None


# ---------------------------------------------------------------------------
# Observer -- every method is a safe no-op with client=None
# ---------------------------------------------------------------------------

def test_observer_with_no_client_never_raises():
    obs = Observer(client=None, settings=_settings())
    assert obs.enabled is False
    # None of these should raise, regardless of arguments:
    obs.start_trace("t1", "run")
    obs.span("t1", "x")
    obs.generation("t1", "g", provider="primary", model="m")
    obs.event("t1", "e")
    obs.score("t1", "s", 0.5)
    obs.end_trace("t1")
    obs.flush()
    obs.shutdown()


def test_observer_degrades_when_the_client_itself_raises():
    """Every SDK call raises RuntimeError -- Observer must swallow every
    single one and never propagate."""
    class _ExplodingClient:
        def __getattr__(self, name):
            def _raise(*a, **kw):
                raise RuntimeError(f"{name} failed")
            return _raise

    obs = Observer(client=_ExplodingClient(), settings=_settings())
    assert obs.enabled is True
    obs.start_trace("t1", "run")
    obs.span("t1", "x")
    obs.generation("t1", "g", provider="primary", model="m")
    obs.event("t1", "e")
    obs.score("t1", "s", 0.5)
    obs.end_trace("t1")
    obs.flush()
    obs.shutdown()  # must not raise even though _client.shutdown() raises too


def test_observer_span_uses_a_real_trace_context(monkeypatch):
    """Confirms the real call shape reaches the client: a deterministic
    trace_id derived from thread_id via create_trace_id(seed=...), and
    start_observation/... .end() actually invoked."""
    calls = []

    class _FakeObservation:
        def end(self, **kwargs):
            calls.append(("end", kwargs))

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-for-{seed}"

        def start_observation(self, **kwargs):
            calls.append(("start_observation", kwargs))
            return _FakeObservation()

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs.span("thread-abc", "my-span", input={"a": 1}, output={"b": 2})

    assert calls[0][0] == "start_observation"
    assert calls[0][1]["name"] == "my-span"
    assert calls[0][1]["trace_context"].get("trace_id") == "trace-for-thread-abc"
    assert calls[1][0] == "end"


def test_observer_nests_observations_under_the_open_root_span():
    """With a root span open, every later observation must carry the
    root's span id as parent_span_id -- that is what makes the trace a
    tree instead of a flat list of siblings."""
    calls = []

    class _FakeObservation:
        def __init__(self, span_id):
            self.id = span_id

        def end(self, **kwargs):
            pass

        def update(self, **kwargs):
            pass

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-for-{seed}"

        def start_observation(self, **kwargs):
            calls.append(kwargs)
            return _FakeObservation("root-span-id")

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs.start_trace("t1", "research_run")
    obs.span("t1", "node:classify")
    obs.generation("t1", "llm", provider="primary", model="m")
    obs.event("t1", "memory.retrieved")

    # The root itself must NOT be parented under anything.
    assert "parent_span_id" not in calls[0]["trace_context"]
    # Everything after it must be.
    for kwargs in calls[1:]:
        assert kwargs["trace_context"]["parent_span_id"] == "root-span-id"
        assert kwargs["trace_context"]["trace_id"] == "trace-for-t1"


def test_observer_does_not_nest_once_the_root_is_closed():
    """After end_trace the root is gone, so an observation must fall back
    to a flat, trace-only context rather than raising or reusing a stale
    parent id."""
    calls = []

    class _FakeObservation:
        id = "root-span-id"

        def end(self, **kwargs):
            pass

        def update(self, **kwargs):
            pass

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-for-{seed}"

        def start_observation(self, **kwargs):
            calls.append(kwargs)
            return _FakeObservation()

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs.start_trace("t1", "research_run")
    obs.end_trace("t1")
    obs.span("t1", "late-span")

    assert "parent_span_id" not in calls[-1]["trace_context"]


def test_observer_nesting_degrades_when_the_root_handle_has_no_id():
    """A root handle without a usable `.id` must produce the old flat
    behavior, not an exception and not a garbage parent_span_id."""
    calls = []

    class _IdlessObservation:
        def end(self, **kwargs):
            pass

        def update(self, **kwargs):
            pass

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-for-{seed}"

        def start_observation(self, **kwargs):
            calls.append(kwargs)
            return _IdlessObservation()

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs.start_trace("t1", "research_run")
    obs.span("t1", "x")

    assert "parent_span_id" not in calls[-1]["trace_context"]


def test_observer_generation_computes_cost_and_usage(monkeypatch):
    calls = []

    class _FakeObservation:
        def end(self, **kwargs):
            pass

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-{seed}"

        def start_observation(self, **kwargs):
            calls.append(kwargs)
            return _FakeObservation()

    s = _settings(langfuse_price_mistral_in_per_1m=1.0,
                 langfuse_price_mistral_out_per_1m=3.0)
    obs = Observer(client=_FakeClient(), settings=s)
    obs.generation("t1", "gen", provider="mistral", model="mistral-small",
                   prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert calls[0]["as_type"] == "generation"
    assert calls[0]["usage_details"] == {"input": 1_000_000, "output": 1_000_000,
                                         "total": 2_000_000}
    assert calls[0]["cost_details"]["total"] == 4.0


def test_observer_score_passes_trace_id_directly():
    calls = []

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-{seed}"

        def create_score(self, **kwargs):
            calls.append(kwargs)

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs.score("thread-xyz", "recall", 0.85, comment="depth=1")

    assert calls[0]["name"] == "recall"
    assert calls[0]["value"] == 0.85
    assert calls[0]["trace_id"] == "trace-thread-xyz"


def test_observer_end_trace_is_a_noop_without_a_matching_start():
    """Calling end_trace for a thread_id that never had start_trace called
    must not raise -- a documented no-op, not a KeyError."""
    class _FakeClient:
        def create_trace_id(self, *, seed):
            return "x"

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs.end_trace("never-started")  # must not raise


# ---------------------------------------------------------------------------
# helpers.traced_node -- the one change point that instruments every node
# ---------------------------------------------------------------------------

def test_thread_id_from_config_extracts_the_configurable_thread_id():
    config = {"configurable": {"thread_id": "run-abc"}, "recursion_limit": 60}
    assert thread_id_from_config(config) == "run-abc"


def test_thread_id_from_config_falls_back_when_missing():
    assert thread_id_from_config(None) == "unknown"
    assert thread_id_from_config({}) == "unknown"


def test_traced_node_preserves_the_wrapped_functions_return_value():
    def fake_node(state):
        return {"field": state["x"] + 1}

    spans = []

    class _FakeObserver:
        def span(self, thread_id, name, **kwargs):
            spans.append((thread_id, name, kwargs))

    wrapped = traced_node(lambda: _FakeObserver(), "fake", fake_node)
    result = wrapped({"x": 1}, config={"configurable": {"thread_id": "t1"}})

    assert result == {"field": 2}
    assert spans[0][0] == "t1"
    assert spans[0][1] == "node:fake"


def test_traced_node_still_spans_and_reraises_on_exception():
    def failing_node(state):
        raise ValueError("boom")

    spans = []

    class _FakeObserver:
        def span(self, thread_id, name, **kwargs):
            spans.append(kwargs)

    wrapped = traced_node(lambda: _FakeObserver(), "fake", failing_node)
    with pytest.raises(ValueError):
        wrapped({"x": 1}, config=None)

    assert spans[0]["level"] == "ERROR"
    assert "boom" in spans[0]["metadata"]["error"]


def test_traced_node_works_with_no_config_argument_at_all():
    """LangGraph calls a node with just (state) if it doesn't declare a
    config parameter of its own -- traced_node's wrapper must still work
    when called that way (config defaults to None)."""
    def fake_node(state):
        return {"ok": True}

    class _NullObserver:
        def span(self, *a, **kw):
            pass

    wrapped = traced_node(lambda: _NullObserver(), "fake", fake_node)
    assert wrapped({"x": 1}) == {"ok": True}


# ---------------------------------------------------------------------------
# __init__.py -- the public, business-module-facing surface
# ---------------------------------------------------------------------------

def test_init_module_is_safe_before_init_from_settings_is_ever_called():
    """Import-time safety: a test or script that imports research_agent.
    langfuse and calls a thin function without ever calling
    init_from_settings must not crash -- get_observer() lazily builds a
    disabled Observer."""
    import research_agent.langfuse as lf
    lf._observer = None  # simulate "never initialized" for this test
    lf._init_settings = None
    assert lf.is_enabled() is False
    lf.span("t1", "x")  # must not raise
    lf.generation("t1", "g", provider="primary", model="m")
    lf.event("t1", "e")
    lf.score("t1", "s", 1.0)
    lf.start_trace("t1", "run")
    lf.end_trace("t1")
    lf.flush()
    lf.shutdown()


def test_init_from_settings_wires_a_real_disabled_observer():
    import research_agent.langfuse as lf
    obs = lf.init_from_settings(_settings(langfuse_enabled=False))
    assert obs.enabled is False
    assert lf.get_observer() is obs


# ---------------------------------------------------------------------------
# Regression tests for the reviewed-and-confirmed fixes -- batch 1 (bugs)
# ---------------------------------------------------------------------------

def test_traced_node_forwards_config_when_the_node_declares_it():
    """Fix for #1: a node that DOES declare config must actually receive
    it -- the bug was that fn(state) was always called regardless."""
    def node_with_config(state, config):
        return {"thread_id": config["configurable"]["thread_id"]}

    class _NullObserver:
        def span(self, *a, **kw):
            pass

    wrapped = traced_node(lambda: _NullObserver(), "cfg", node_with_config)
    result = wrapped({"x": 1}, config={"configurable": {"thread_id": "t1"}})
    assert result == {"thread_id": "t1"}


def test_traced_node_still_omits_config_for_a_plain_single_arg_node():
    """The common case (12 of 13 real nodes): fn(state) only, never
    fn(state, config) -- forwarding unconditionally would TypeError here."""
    def plain_node(state):
        return {"ok": True}

    class _NullObserver:
        def span(self, *a, **kw):
            pass

    wrapped = traced_node(lambda: _NullObserver(), "plain", plain_node)
    assert wrapped({"x": 1}, config={"configurable": {"thread_id": "t1"}}) == {"ok": True}


def test_traced_node_forwards_config_even_via_kwargs_catchall():
    """A node written as def n(state, **kwargs) should also receive
    config, since **kwargs would silently swallow it otherwise."""
    received = {}

    def kwargs_node(state, **kwargs):
        received.update(kwargs)
        return {}

    class _NullObserver:
        def span(self, *a, **kw):
            pass

    wrapped = traced_node(lambda: _NullObserver(), "kw", kwargs_node)
    wrapped({"x": 1}, config={"configurable": {"thread_id": "t1"}})
    assert received.get("config") == {"configurable": {"thread_id": "t1"}}


def test_observer_shutdown_ends_every_still_open_root_span():
    """Fix for #2 (backstop half): a root span that never went through
    end_trace() must still be .end()ed on shutdown -- otherwise, in v4's
    OTel model, it's never exported at all, not just incomplete."""
    ended = []

    class _FakeRoot:
        def end(self, **kwargs):
            ended.append("ended")

    class _FakeClient:
        def shutdown(self):
            pass

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs._roots["t1"] = _FakeRoot()
    obs._roots["t2"] = _FakeRoot()
    obs.shutdown()

    assert ended == ["ended", "ended"]
    assert obs._roots == {}


def test_observer_shutdown_survives_a_root_that_raises_on_end():
    """One bad root.end() must not stop the others from being ended, or
    stop client.shutdown() from still being called."""
    ended = []

    class _ExplodingRoot:
        def end(self, **kwargs):
            raise RuntimeError("boom")

    class _FakeRoot:
        def end(self, **kwargs):
            ended.append("ended")

    shutdown_called = []

    class _FakeClient:
        def shutdown(self):
            shutdown_called.append(True)

    obs = Observer(client=_FakeClient(), settings=_settings())
    obs._roots["bad"] = _ExplodingRoot()
    obs._roots["good"] = _FakeRoot()
    obs.shutdown()  # must not raise

    assert ended == ["ended"]
    assert shutdown_called == [True]


def test_build_client_passes_environment_to_the_constructor():
    """Fix for #12 (constructor half): environment must be a first-class
    kwarg to Langfuse(...), not only buried in per-trace metadata --
    that's what the Langfuse UI's environment filter actually reads."""
    built_kwargs = {}

    fake_module = types.ModuleType("langfuse")

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            built_kwargs.update(kwargs)

    fake_module.Langfuse = _FakeLangfuse
    import sys as _sys
    _sys.modules["langfuse"] = fake_module

    s = _settings(langfuse_enabled=True, langfuse_public_key="pk",
                 langfuse_secret_key="sk", langfuse_environment="staging")
    lf_client.build_client(s)

    assert built_kwargs["environment"] == "staging"
    del _sys.modules["langfuse"]


def test_cost_clamps_a_misconfigured_negative_rate_to_zero():
    """Fix for #14: a typo'd negative LANGFUSE_PRICE_* must not produce a
    negative dollar figure."""
    s = _settings(langfuse_price_mistral_in_per_1m=-5.0,
                 langfuse_price_mistral_out_per_1m=-2.0)
    cost = calculate_cost(s, "mistral", TokenUsage(prompt_tokens=1_000_000,
                                                    completion_tokens=1_000_000))
    assert cost.input_cost_usd == 0.0
    assert cost.output_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Regression tests for the reviewed-and-confirmed fixes -- batch 2 (enrichment)
# ---------------------------------------------------------------------------

def _fake_client_and_propagate_attributes(monkeypatch):
    """Shared fixture-ish helper: a fake langfuse module with a fake
    propagate_attributes context manager, wired in place of the real
    import inside Observer.start_trace/end_trace. Returns (calls list,
    fake client instance)."""
    calls = []

    class _FakeCtx:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            calls.append(("enter", self.kwargs))
            return self

        def __exit__(self, *exc):
            calls.append(("exit", None))
            return False

    def fake_propagate_attributes(**kwargs):
        return _FakeCtx(**kwargs)

    fake_module = types.ModuleType("langfuse")
    fake_module.propagate_attributes = fake_propagate_attributes
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    class _FakeRoot:
        def end(self, **kwargs):
            calls.append(("root_end", None))

        def update(self, **kwargs):
            calls.append(("root_update", kwargs))

    class _FakeClient:
        def create_trace_id(self, *, seed):
            return f"trace-{seed}"

        def start_observation(self, **kwargs):
            calls.append(("start_observation", kwargs))
            return _FakeRoot()

    return calls, _FakeClient()


def test_start_trace_enters_propagate_attributes_with_session_id(monkeypatch):
    """Fix for #12 (propagation half): start_trace must open a
    propagate_attributes context carrying session_id=thread_id, entered
    BEFORE the root span is created (ordering is load-bearing -- see
    module docstring)."""
    calls, client = _fake_client_and_propagate_attributes(monkeypatch)
    obs = Observer(client=client, settings=_settings(langfuse_environment="staging"))

    obs.start_trace("thread-42", "research_run", input={"query": "x"})

    assert calls[0] == ("enter", {"session_id": "thread-42", "trace_name": "research_run",
                                  "environment": "staging"})
    assert calls[1][0] == "start_observation"
    assert "thread-42" in obs._session_contexts


def test_end_trace_exits_the_matching_propagate_attributes_context(monkeypatch):
    calls, client = _fake_client_and_propagate_attributes(monkeypatch)
    obs = Observer(client=client, settings=_settings())

    obs.start_trace("t1", "run")
    obs.end_trace("t1", output={"done": True})

    assert calls[-1] == ("exit", None)
    assert "t1" not in obs._session_contexts
    assert "t1" not in obs._roots


def test_start_trace_does_not_leak_the_context_if_start_observation_fails(monkeypatch):
    """The bug I caught in my own review before shipping it: if
    start_observation() raises AFTER propagate_attributes was already
    entered, the context must still be exited -- otherwise it leaks for
    the life of the thread."""
    calls, _ = _fake_client_and_propagate_attributes(monkeypatch)

    class _ExplodingClient:
        def create_trace_id(self, *, seed):
            return f"trace-{seed}"

        def start_observation(self, **kwargs):
            raise RuntimeError("network hiccup")

    obs = Observer(client=_ExplodingClient(), settings=_settings())
    obs.start_trace("t1", "run")  # must not raise (caught by _safe)

    assert ("enter", {"session_id": "t1", "trace_name": "run", "environment": "development"}) in calls
    assert ("exit", None) in calls  # the context WAS closed despite the failure
    assert "t1" not in obs._session_contexts  # and never recorded as open


def test_shutdown_closes_any_still_open_session_contexts(monkeypatch):
    """Backstop half of the same fix: a context left open by a crashed
    run (end_trace never reached) must still be closed on shutdown."""
    calls, client = _fake_client_and_propagate_attributes(monkeypatch)
    obs = Observer(client=client, settings=_settings())

    obs.start_trace("t1", "run")
    calls.clear()
    obs.shutdown()

    assert ("exit", None) in calls
    assert obs._session_contexts == {}
    assert obs._roots == {}


def test_observer_has_a_lock_guarding_its_dicts():
    """Fix for #3: cheap, direct assertion that the hardening is
    actually present, not just that behavior happens to look right."""
    import threading as _threading
    obs = Observer(client=None, settings=_settings())
    assert isinstance(obs._lock, _threading.Lock().__class__)
