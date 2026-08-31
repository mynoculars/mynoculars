"""
tests/conftest.py — Shared fixtures: offline graph with stub LLM + fake tool.

Every test runs fully offline: StubClient for the LLM, an in-process fake
retrieval tool, degraded (off) memory, and an in-memory checkpointer.

"Offline" now means it literally, too (D-140). It previously meant "no
test talks to a service that answers" -- which is not the same thing as
"no test opens a socket", and the difference was the single largest cost
in the suite. The `off_memory` fixture built a real QdrantStore per test
against 127.0.0.1:1, and that constructor probes: QdrantClient(url,
timeout=5) runs a version check with its own retry loop, then
get_collections() probes liveness. Measured at 235 ms per construction on
Linux, where the port refuses instantly -- 28 constructions, 6.6 s of a
24.4 s run, for connections no test wanted. The same 1,019 tests took
925 s (15 min 25 s) on Windows: a refused connect there does not
necessarily return a fast RST, because Docker Desktop's WinNAT reserves
port-exclusion ranges and a connect into one retransmits SYN until the
timeout expires.

Two changes close it:
  - stores are constructed with probe=False, which sets the same degraded
    flag every method already checks, without opening anything;
  - `settings` and `off_memory` are SESSION-scoped. They are immutable
    value objects -- no test assigns to a field, and the tests that need a
    variant already build one with .model_copy() -- and they were being
    rebuilt 222 and 43 times respectively.

The failed-connect path is still covered, once, by the storage tests.
Where a real connect attempt is wanted, use UNROUTABLE_URL below rather
than a localhost port.
"""

import json
import os

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.state import Evidence, SearchTask, Volatility
from research_agent.storage.qdrant_store import QdrantStore


class RejectingCriticStub(StubClient):
    """StubClient whose critic ALWAYS fails the report.

    Shared between tests/integration/test_failure_paths.py (critique
    exhaustion) and tests/integration/test_hitl_escalation.py (E4, which
    is triggered BY critique exhaustion) -- defined here, once, rather
    than one of those files importing it from the other.
    """

    def complete(self, messages, temperature=0.2):
        last = messages[-1]["content"]
        if "TASK=critique" in last:
            return json.dumps({"passed": False, "score": 0.2,
                               "notes": ["missing tradeoff analysis"]})
        return super().complete(messages, temperature)


@pytest.fixture(scope="session", autouse=True)
def _no_ambient_config():
    """Nothing in this suite may read the developer's real configuration
    (D-147). Autouse and session-scoped, so it holds for every test.

    THE FAILURE THIS CLOSES. The suite is documented as fully offline, and
    the `settings` fixture below explains at length why it passes
    _env_file=None rather than trusting the ambient environment. But 43
    tests never reach that fixture: they construct Settings() through
    config.get_settings(), either directly or by importing
    scripts/mcp_corpus_server.py and scripts/mcp_web_search_server.py,
    both of which call get_settings() AT MODULE IMPORT. Those 43 read the
    real .env in the current working directory.

    Reproduced exactly, from one line in a developer's own .env:

        LLM_PRIMARY_CONTEXT_TOKENS=8,876
        -> 26 failed, 1087 passed, 17 errors

    A thousands separator in a file the test suite has no business
    reading took out 3.8% of the suite, and every one of the 43 failures
    reported a pydantic ValidationError against a field none of them
    tests. The value itself is now accepted (D-148, config.py) -- but the
    isolation is the real fix, because the next malformed line will be a
    different one.

    TWO SOURCES, BOTH CLOSED, because closing one is what made this
    survive so long:

      - the .env FILE: model_config's env_file is set to None for the
        session. tests/unit/test_config.py already solved this for itself
        (D-79) by chdir'ing into an empty tmp_path; this is the same idea
        applied once, globally, instead of per-file.
      - REAL OS ENVIRONMENT VARIABLES: every name that maps to a Settings
        field is removed for the session. Settings(_env_file=None) does
        NOT insulate against these -- pydantic-settings always checks
        os.environ first -- which is the hazard the `settings` fixture
        below documents with the HITL_ENABLED example. That fixture made
        one field structurally safe; this makes all of them safe.

    A test that WANTS a value set uses monkeypatch.setenv, which runs
    after this fixture and is undone at its own teardown, so nothing here
    takes anything away from a test that asks for it.
    """
    from research_agent.config import Settings, get_settings

    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None

    names = {name.upper() for name in Settings.model_fields}
    removed = {key: os.environ.pop(key) for key in list(os.environ)
               if key in names}
    # get_settings is lru_cached; a cached Settings built before this
    # fixture ran would carry exactly the ambient values it exists to
    # exclude.
    get_settings.cache_clear()
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original_env_file
        os.environ.update(removed)
        get_settings.cache_clear()


# RFC 5737 TEST-NET-1: reserved for documentation, routed nowhere by
# anybody. The correct choice for a test that must attempt and fail a real
# connection -- unlike a localhost port, whose behaviour depends on the OS,
# on what else is bound, and (on Windows) on Hyper-V/WinNAT port
# reservations. Always pair it with an explicit short timeout: unroutable
# means "no answer", not "refused".
UNROUTABLE_URL = "http://192.0.2.1:6333"


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings with tight bounds and no env-file surprises.

    hitl_enabled=False is passed EXPLICITLY here, not left to the field's
    own default. Reason (found via a real failure, not theoretical):
    Settings(_env_file=None, ...) only skips reading a .env FILE — it does
    NOTHING to insulate against real OS environment variables, which
    pydantic-settings always checks first regardless of _env_file. A
    developer who ran `$env:HITL_ENABLED = "true"` earlier in the SAME
    shell session (e.g. for manual live testing) and then ran pytest in
    that same window would silently get hitl_enabled=True here too — and
    tests that specifically exercise the HITL-OFF path (this fixture is
    used by tests that expect NO interrupt) would instead pause via
    interrupt() and never reach telemetry_node, producing a confusing
    KeyError on state.telemetry (which stays at its default {} for an
    interrupted run) instead of a clear assertion failure. Passing it
    explicitly here makes that class of failure structurally impossible,
    regardless of what's sitting in whoever's shell.

    SESSION-SCOPED (D-140). No test assigns to a field on this object, and
    the tests that need a variant already build one with .model_copy(), so
    rebuilding it 222 times per run bought nothing.
    """
    return Settings(_env_file=None, llm_mode="stub", max_depth=2,
                    max_fanout=4, max_revisions=2, hitl_enabled=False,
                    qdrant_url="http://127.0.0.1:1", postgres_dsn="postgresql://x:x@127.0.0.1:1/x",
                    opensearch_url="http://127.0.0.1:1")


@pytest.fixture
def stub_router() -> FallbackRouter:
    """Router in stub mode: deterministic, no fallback."""
    return FallbackRouter([StubClient()], quality_threshold=0.6)


@pytest.fixture
def fake_tool():
    """Retrieval tool returning one high-score evidence item per task."""

    def tool(task: SearchTask):
        return [Evidence(task_key=task.key, goal_id=task.goal_id, source="fake",
                         content=f"fact about {task.query}", score=0.9,
                         volatility=Volatility.SEMI_STABLE)]

    return tool


@pytest.fixture(scope="session")
def off_memory(settings) -> SemanticMemory:
    """Memory in degraded (off) mode — the state every offline test wants.

    D-140: probe=False rather than an unreachable URL. The DEGRADED STATE
    is what these tests exercise, and it is reached identically either way
    -- but only one of the two routes opens a socket and waits for it to
    fail. Session-scoped for the same reason `settings` is: a degraded
    store holds no per-test state.
    """
    return SemanticMemory(QdrantStore(settings.qdrant_url, "test", probe=False),
                          settings.memory_top_k, 90.0, 14.0)


@pytest.fixture
def graph(stub_router, fake_tool, off_memory, settings):
    """The full compiled workflow, offline."""
    return build_graph(stub_router, fake_tool, off_memory, settings, MemorySaver())


def _extract_failure_reason(longreprtext: str) -> str:
    """Pull the actual assertion/exception message out of pytest's
    longreprtext, not just its last line.

    CALLED BY   pytest_terminal_summary, below, once per failed/errored
                report.
    WHY NOT "just take the last line": pytest's longreprtext ends with a
    LOCATION summary line (e.g. "tests/foo.py:2: AssertionError"), not the
    actual message -- the real text ("AssertionError: <your message>")
    is a few lines earlier, prefixed with "E " (pytest's own convention
    for highlighting the error line in a traceback). This walks the text
    looking for the FIRST such "E "-prefixed line and returns it with the
    prefix stripped; if none is found (an unusual failure shape), it
    falls back to the last non-empty line rather than returning nothing.
    """
    text = (longreprtext or "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("E "):
            return stripped[2:].strip()
    return lines[-1] if lines else "(no details captured)"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print an explicit "X out of Y tests passed" line after pytest's own
    summary -- requested because pytest's default final line ("97 passed,
    31 warnings in 76.94s") reports a bare pass count with no visible
    denominator, which reads ambiguously at a glance (is 97 all of them,
    or 97 out of some larger number that scrolled off?).

    CALLED BY   pytest itself, automatically, once per test session --
                this is a standard pytest hook (see
                https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_terminal_summary),
                not a mechanism specific to this codebase.
    READS       terminalreporter.stats, a dict pytest maintains internally
                mapping outcome name ("passed", "failed", "error",
                "skipped", "xfailed", "xpassed") to the list of test
                reports with that outcome -- this just counts each list's
                length rather than re-running or re-collecting anything.
    WRITES      one line to the terminal, via terminalreporter.write_line
                (the same output stream pytest's own summary uses, so it
                appears in the normal `-q` run, not just verbose mode).
    """
    stats = terminalreporter.stats
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    error = len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    xfailed = len(stats.get("xfailed", []))
    xpassed = len(stats.get("xpassed", []))
    total = passed + failed + error + skipped + xfailed + xpassed
    terminalreporter.write_line(f"{passed} out of {total} tests passed")

    # Explicit failed/error/skipped counts and per-test failure details --
    # this was a real gap in the first version of this hook: it computed
    # `failed` and `error` above but never actually printed them, so a run
    # with real failures would say "94 out of 97 tests passed" and leave
    # the other 3 to be inferred by subtraction, with no indication of
    # WHICH tests or WHY. pytest's own default reporting already shows
    # this correctly elsewhere (the "short test summary info" section,
    # and full tracebacks above it) -- this doesn't replace that, it just
    # means this hook's own block is self-contained rather than silently
    # dropping the one thing (failures) that matters most when it happens.
    if failed or error:
        parts = [f"{failed} failed", f"{error} errored"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if xfailed:
            parts.append(f"{xfailed} xfailed")
        if xpassed:
            parts.append(f"{xpassed} xpassed")
        terminalreporter.write_line("  " + ", ".join(parts))
        for outcome in ("failed", "error"):
            for report in stats.get(outcome, []):
                reason = _extract_failure_reason(report.longreprtext)
                terminalreporter.write_line(f"  {outcome.upper()}: {report.nodeid} -- {reason}")
    elif skipped or xfailed or xpassed:
        parts = []
        if skipped:
            parts.append(f"{skipped} skipped")
        if xfailed:
            parts.append(f"{xfailed} xfailed")
        if xpassed:
            parts.append(f"{xpassed} xpassed")
        terminalreporter.write_line("  " + ", ".join(parts))

    # Itemized warnings, one line per OCCURRENCE -- pytest's own built-in
    # "warnings summary" section (printed automatically, just above this
    # hook's output) groups every occurrence of the SAME warning message
    # under one entry with a bare count (e.g. "tests/test_tier3.py: 22
    # warnings"), which is the right thing for a quick scan but doesn't
    # answer "which 22 tests, and what did each one actually say". This
    # prints the full, ungrouped list instead: terminalreporter.stats
    # holds every captured WarningReport under the "warnings" key,
    # regardless of how pytest's own summary chooses to group them for
    # display -- reading straight from that list, one line per report,
    # is what actually gives "at least a one-liner for each warning".
    warnings = stats.get("warnings", [])
    if not warnings:
        return

    # P8-2: under xdist this list is assembled from N worker processes, and
    # a warning that fires once PER PROCESS therefore contributes a count
    # that depends on how many workers ran. Measured on the same suite,
    # same commit: 57 warnings at -n0, 55 at -n4, 52 at -n8. Printing a
    # bare per-occurrence total under those conditions is a number that
    # looks like a regression signal and is not one.
    #
    # So: when distributed, group identical (location, message) pairs, sort
    # them, and print each with its own count -- deterministic in ORDER and
    # in MEMBERSHIP, with the varying part isolated to a per-line integer
    # that is explained. Serial runs keep the original ungrouped listing,
    # which is what a single process can honestly claim.
    distributed = getattr(config.option, "numprocesses", None)
    entries = []
    for w in warnings:
        location = w.get_location(config) or "?"
        # message can be multi-line (e.g. a traceback-style warning); only
        # the first line is shown so this stays ONE line per warning. The
        # full text is still in pytest's own grouped summary above.
        first_line = w.message.splitlines()[0] if w.message else ""
        entries.append(f"{location}: {first_line}")

    terminalreporter.write_line("")
    if distributed:
        from collections import Counter

        grouped = Counter(entries)
        terminalreporter.write_line(
            f"Distinct warnings ({len(grouped)} distinct, {len(entries)} "
            f"occurrences across {distributed} worker(s) -- occurrence "
            f"counts vary with worker count, distinct set does not):")
        for entry, count in sorted(grouped.items()):
            suffix = f"  (x{count})" if count > 1 else ""
            terminalreporter.write_line(f"  {entry}{suffix}")
    else:
        terminalreporter.write_line(
            f"Individual warnings ({len(entries)} total, ungrouped):")
        for entry in entries:
            terminalreporter.write_line(f"  {entry}")
