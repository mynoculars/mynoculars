"""
tests/unit/test_cli_failure_paths.py -- what main() does when a run dies.

Covers D-100 (the tracer is flushed even when the run raised) and D-101
(provider-chain exhaustion gets a diagnosable message and exit code 4,
not a raw traceback). Both were diagnosed from run p205.254-check, which
ended in a 40-line LangGraph traceback and wrote no logs/run-*.txt at all
-- the run that most needed a debug trace was the one that never got one.

Its own file rather than appended to test_cli_result_summary.py or
test_cli_hitl_wall_time.py: those two cover pure formatting helpers with
no process wiring, and these drive main() end to end through a faked
AppBundle. Same delivery reasoning recorded in DECISIONS.md D-62/D-63.
"""

import json

import pytest
from langgraph.errors import GraphRecursionError

from research_agent import cli
from research_agent.assembly import AppBundle
from research_agent.llm.client import TruncatedGenerationError
from research_agent.llm.router import ProviderChainExhausted


class _FakeApp:
    """Enough of a compiled graph for main() to reach invoke().

    get_state must return something whose .values is falsy, or the D-20
    guard (assembly.reject_if_thread_in_use) short-circuits the run with
    exit code 3 before invoke() is ever called.
    """

    def __init__(self, raises):
        self._raises = raises

    class _EmptySnapshot:
        values = {}

    def get_state(self, config):
        return self._EmptySnapshot()

    def invoke(self, *args, **kwargs):
        raise self._raises


class _RecordingTracer:
    """Counts flush() calls and reports a path the first time, exactly as
    the real Tracer does -- flush_narrative POPS the run's buffered
    events, so a second call finds nothing."""

    def __init__(self):
        self.flushes = 0

    @property
    def enabled(self):
        return True

    def flush(self):
        self.flushes += 1
        return "logs/run-fake.txt" if self.flushes == 1 else None


@pytest.fixture
def wired(monkeypatch, settings):
    """main() with a faked bundle, and a tracer we can interrogate.

    Returns a callable: run(exception) -> (exit_code, tracer). The tracer
    is ALSO parked on run.tracer, and every record_failed_run call on
    run.recorded, so a test whose exception propagates out of main() --
    and therefore never gets a return value -- can still assert what the
    finally and except blocks did.
    """
    def run(exc):
        tracer = _RecordingTracer()
        run.tracer = tracer
        run.recorded = []
        monkeypatch.setattr(cli, "Tracer", lambda _thread_id: tracer)
        monkeypatch.setattr(cli, "NullTracer", lambda: tracer)
        monkeypatch.setattr(
            cli, "build_app_and_settings",
            lambda tracer=None: AppBundle(app=_FakeApp(exc), settings=settings,
                                          durable=True, checkpointer=None))
        # record_run would try to reach Postgres; it is never reached on
        # any path under test here, but stubbing it keeps a future edit
        # from silently opening a socket in a unit test.
        monkeypatch.setattr(cli, "record_run", lambda *a, **kw: None)
        # D-103 added a second recorder on the path every test here
        # takes. The FIXTURE owns the spy rather than each test patching
        # it: run() is what installs the monkeypatches, so a test that
        # patched record_failed_run before calling wired() would have its
        # patch silently replaced here and see zero calls.
        monkeypatch.setattr(
            cli, "record_failed_run",
            lambda *a, **kw: run.recorded.append(a) or 1)
        code = cli.main(["a query", "--thread-id", "fake", "--debug"])
        return code, tracer
    return run


# ---------------------------------------------------------------------------
# D-101 -- provider-chain exhaustion is an operational event, not a crash
# ---------------------------------------------------------------------------


def _exhausted():
    exc = ProviderChainExhausted(
        "compiler",
        [("primary", "HTTPStatusError"),
         ("mistral", "ReadTimeout"),
         ("gemini", "TruncatedGenerationError")],
        "text")
    # What `raise ... from last_exc` sets, reproduced directly so this
    # test does not have to route a real chain to get a __cause__.
    exc.__cause__ = TruncatedGenerationError(
        "gemini/gemini-3.5-flash stopped at the token limit")
    return exc


def test_chain_exhaustion_exits_4_instead_of_raising(wired):
    code, _ = wired(_exhausted())
    assert code == 4


def test_chain_exhaustion_prints_every_provider_and_its_failure(wired, capsys):
    wired(_exhausted())
    err = capsys.readouterr().err
    assert "primary HTTPStatusError" in err
    assert "mistral ReadTimeout" in err
    assert "gemini TruncatedGenerationError" in err
    assert "compiler" in err


def test_chain_exhaustion_surfaces_the_underlying_cause(wired, capsys):
    """The chain summary says WHICH providers failed; only the last
    provider's own message says WHY in the detail an operator needs
    (D-102's cap attribution lives in that string)."""
    wired(_exhausted())
    err = capsys.readouterr().err
    assert "stopped at the token limit" in err


def test_the_recursion_limit_still_has_its_own_distinct_exit_code(wired):
    """4 must not swallow 2. The operator action differs: 2 means look at
    the graph's budgets, 4 means look at the providers."""
    code, _ = wired(GraphRecursionError("limit"))
    assert code == 2


def test_an_unrecognised_exception_still_propagates(wired):
    """D-101 narrows the raw-traceback path, it does not close it. An
    error nobody has a diagnosis for must still reach the operator in
    full rather than becoming a bare exit code."""
    with pytest.raises(ZeroDivisionError):
        wired(ZeroDivisionError("something genuinely unexpected"))


# ---------------------------------------------------------------------------
# D-100 -- the tracer is flushed on the failure path too
# ---------------------------------------------------------------------------


def test_the_tracer_is_flushed_when_the_run_raised(wired):
    """p205.254-check's actual gap: tracer.flush() sat on _run's happy
    path, so the only run that produced no trace was the one that needed
    it. main()'s finally already closed every OTHER resource."""
    _, tracer = wired(_exhausted())
    assert tracer.flushes == 1


def test_the_tracer_is_flushed_even_for_an_exception_nobody_handles(wired):
    """finally, not a second except clause -- the flush must not depend
    on main() recognising the exception type. This is the case that
    matters most: an unrecognised error is exactly when a trace is worth
    having."""
    with pytest.raises(ZeroDivisionError):
        wired(ZeroDivisionError("boom"))
    assert wired.tracer.flushes == 1


def test_the_crash_trace_path_is_reported_on_stderr(wired, capsys):
    """stdout may hold a half-written report on this path; the trace
    location is diagnostic output."""
    wired(_exhausted())
    captured = capsys.readouterr()
    assert "logs/run-fake.txt" in captured.err
    assert "logs/run-fake.txt" not in captured.out



# ---------------------------------------------------------------------------
# D-103 -- a failed run enters the history
# ---------------------------------------------------------------------------


def test_a_failed_run_is_recorded_once_with_no_invented_recall(wired):
    """The gap D-103 closes: record_run sat on the happy path, so the run
    you most want in the history was the one that never got a row. Once,
    not once per except clause -- a ProviderChainExhausted passes through
    _run's except AND main()'s handler."""
    wired(_exhausted())

    assert len(wired.recorded) == 1
    _dsn, thread_id, query, failure = wired.recorded[0]
    assert thread_id == "fake"
    assert query == "a query"
    assert failure["type"] == "ProviderChainExhausted"


def test_the_recorder_is_reached_for_an_exception_main_does_not_handle(wired):
    """Recorded in _run's except, not in main()'s handlers, precisely so
    an unrecognised failure is still history."""
    with pytest.raises(ZeroDivisionError):
        wired(ZeroDivisionError("boom"))

    assert len(wired.recorded) == 1
    assert wired.recorded[0][3]["type"] == "ZeroDivisionError"


def test_a_database_failure_while_recording_does_not_mask_the_run_failure(
        wired, monkeypatch):
    """record_failed_run swallows its own errors exactly as record_run
    does. If that ever stopped being true, the original exception -- the
    one worth seeing -- would be replaced by a database error. Patched
    AFTER wired() has installed its own, which is why this one takes the
    exception rather than the fixture's spy."""
    code, _ = wired(_exhausted())
    assert code == 4

    monkeypatch.setattr(cli, "record_failed_run", lambda *a, **kw: None)
    code, _ = wired(_exhausted())
    assert code == 4


def test_a_completed_run_with_no_recall_records_null_not_zero(monkeypatch,
                                                              settings):
    """D-103's other half. telemetry.get("recall", 0.0) wrote a number
    nothing measured, and a run that genuinely retrieved nothing recorded
    the identical value. The column is nullable."""
    recorded = []

    class _FinishingApp(_FakeApp):
        def invoke(self, *args, **kwargs):
            return {"telemetry": {"goals": 2}, "final_report": "r"}

    tracer = _RecordingTracer()
    monkeypatch.setattr(cli, "Tracer", lambda _t: tracer)
    monkeypatch.setattr(
        cli, "build_app_and_settings",
        lambda tracer=None: AppBundle(app=_FinishingApp(None),
                                      settings=settings, durable=True,
                                      checkpointer=None))
    monkeypatch.setattr(cli, "record_run",
                        lambda *a, **kw: recorded.append(a) or 1)

    cli.main(["a query", "--thread-id", "fake"])

    assert recorded, "a completed run still records its row"
    assert recorded[0][3] is None, "recall must be NULL, not a fabricated 0.0"


def test_failure_record_captures_the_chain_node_and_cause():
    record = cli._failure_record(_exhausted())

    assert record["node"] == "compiler"
    assert record["mode"] == "text"
    assert record["chain"] == [["primary", "HTTPStatusError"],
                               ["mistral", "ReadTimeout"],
                               ["gemini", "TruncatedGenerationError"]]
    assert record["cause"]["type"] == "TruncatedGenerationError"


def test_failure_record_is_json_serialisable():
    """It is about to be json.dumps'd into a JSONB column. Tuples would
    round-trip as lists, so they are converted at the point of writing --
    what is stored equals what is read back."""
    json.dumps(cli._failure_record(_exhausted()))
    json.dumps(cli._failure_record(ZeroDivisionError("boom")))


def test_failure_record_of_an_ordinary_exception_carries_no_chain_keys():
    """Only ProviderChainExhausted has a chain. Emitting an empty one for
    everything else would make `failed_provider_outcomes` look measured
    when nothing measured it."""
    record = cli._failure_record(GraphRecursionError("limit"))

    assert record["type"] == "GraphRecursionError"
    assert "chain" not in record and "node" not in record


def test_failure_record_truncates_an_unbounded_provider_message():
    """A row is a record, not a log file."""
    record = cli._failure_record(RuntimeError("x" * 5000))

    assert len(record["message"]) == 500
