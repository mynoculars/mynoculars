"""
tests/conftest.py — Shared fixtures: offline graph with stub LLM + fake tool.

Every test runs fully offline: StubClient for the LLM, an in-process fake
retrieval tool, degraded (off) memory, and an in-memory checkpointer.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.config import Settings
from research_agent.llm.client import StubClient
from research_agent.llm.router import FallbackRouter
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.state import Evidence, SearchTask, Volatility
from research_agent.storage.qdrant_store import QdrantStore


@pytest.fixture
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


@pytest.fixture
def off_memory(settings) -> SemanticMemory:
    """Memory over an unreachable Qdrant — exercises degraded (off) mode."""
    return SemanticMemory(QdrantStore(settings.qdrant_url, "test"),
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
    if warnings:
        terminalreporter.write_line("")
        terminalreporter.write_line(f"Individual warnings ({len(warnings)} total, ungrouped):")
        for w in warnings:
            location = w.get_location(config) or "?"
            # message can be multi-line (e.g. a traceback-style warning);
            # only the first line is shown here so this stays ONE line
            # per warning, matching what was actually asked for -- the
            # full message is still in pytest's own grouped summary above
            # for anyone who needs the complete text.
            first_line = w.message.splitlines()[0] if w.message else ""
            terminalreporter.write_line(f"  {location}: {first_line}")
