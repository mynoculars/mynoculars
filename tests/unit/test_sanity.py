"""
tests/unit/test_sanity.py -- scripts/sanity.py's two pure functions (D-158).

sanity.py is the offline pre-demo gate, and everything about it that can
be wrong WITHOUT running a subprocess lives in `build_steps` (what runs,
and in what order) and `format_summary` (what the last line says). Both
are pure, so both are tested directly, the same file-path loading idiom
tests/unit/test_gc_memory.py uses for scripts/gc_memory.py.

What is deliberately NOT tested here: whether ruff or pytest actually
pass. This suite IS one of the steps -- a test asserting its own result
would be circular, and the gate's whole point is to be run, not mocked.
"""

import importlib.util
import sys


def _load_sanity():
    # D-157: the implementation moved into the package
    # (research_agent.ops.sanity); scripts/ now holds a thin
    # launcher, and loading THAT would exercise a six-line shim.
    # find_spec locates the module WITHOUT executing it, and the fresh
    # module object below is deliberate: several tests here assert on
    # module-level caching, which a shared sys.modules entry would carry
    # from one test into the next.
    origin = importlib.util.find_spec("research_agent.ops.sanity").origin
    spec = importlib.util.spec_from_file_location(
        "sanity", origin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_runs_in_cost_order():
    """Ordering IS the design: ruff catches in one second what the suite
    would report twenty seconds later as a confusing import error, and
    the offline run is only worth attempting once the suite is green."""
    sanity = _load_sanity()

    names = [step.name for step in sanity.build_steps()]

    assert names == ["ruff", "tests", "offline run"]


def test_lint_can_be_skipped_without_disturbing_the_rest():
    sanity = _load_sanity()

    names = [step.name for step in sanity.build_steps(lint=False)]

    assert names == ["tests", "offline run"]


def test_quick_deselects_the_slow_marker_and_nothing_else():
    """--quick must drop the socket-timeout test by MARKER, never by
    naming a file: a path-based skip silently stops matching the moment
    that test moves."""
    sanity = _load_sanity()

    quick = {s.name: s.argv for s in sanity.build_steps(quick=True)}
    full = {s.name: s.argv for s in sanity.build_steps(quick=False)}

    assert quick["tests"] == full["tests"] + ["-m", "not slow"]
    assert quick["offline run"] == full["offline run"]


def test_every_step_runs_this_interpreter_not_whichever_python_is_on_path():
    """A venv is the normal way to run this repo, and `python` on PATH is
    not reliably the venv's. Only the optional external tool (ruff) is
    resolved by name, and that one is guarded by shutil.which."""
    sanity = _load_sanity()

    for step in sanity.build_steps():
        if step.optional_tool:
            continue
        assert step.argv[0] == sys.executable, step.name


def test_only_the_offline_run_is_isolated_from_the_developer_s_env_file():
    """`Settings` reads `.env` relative to the CWD, so a run launched from
    the repo root inherits MCP_ENABLED/WEB_SEARCH_ENABLED from whatever
    the developer has configured -- and then fails because those servers
    are not running, which is check_services.py's question, not this
    gate's. Found by running the gate, not by reasoning about it.

    ruff and pytest must NOT be isolated: both need the repo root as
    their working directory to find the code they are checking."""
    sanity = _load_sanity()

    isolated = {s.name: s.isolated for s in sanity.build_steps()}

    assert isolated == {"ruff": False, "tests": False, "offline run": True}


def test_the_summary_says_failed_when_anything_failed():
    sanity = _load_sanity()

    text = sanity.format_summary([("ruff", "PASS", 1.0),
                                  ("tests", "FAIL", 18.4)])

    assert "FAILED: tests" in text
    assert "PASSED" not in text


def test_a_skipped_optional_tool_is_reported_rather_than_counted_as_a_pass():
    """The distinction the gate exists to preserve: "ruff is not
    installed" and "ruff found nothing" must never print the same."""
    sanity = _load_sanity()

    text = sanity.format_summary([("ruff", "SKIP", 0.0),
                                  ("tests", "PASS", 18.4)])

    assert "with 1 step(s) skipped: ruff" in text
    assert "Every step ran." not in text


def test_a_clean_gate_says_so_unambiguously():
    sanity = _load_sanity()

    text = sanity.format_summary([("ruff", "PASS", 1.0),
                                  ("tests", "PASS", 18.4),
                                  ("offline run", "PASS", 0.9)])

    assert "PASSED. Every step ran." in text
    assert "FAILED" not in text
