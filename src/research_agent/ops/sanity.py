"""
research_agent/ops/sanity.py -- the pre-demo gate, in one command (D-158).

Purpose:
    Run the checks that are cheap, offline and deterministic, in cost
    order, and stop at the first one that fails.

    THIS IS NOT A REPLACEMENT FOR CI, and an earlier version of this
    docstring wrongly said it was (D-160). `.github/workflows/tests.yml`
    runs the suite on every push and answers "did that commit break
    anything". This answers a different question, and it is the one a
    walkthrough actually depends on: "is the machine I am standing at,
    with the working tree I have right now, in a state I can demo from" --
    including uncommitted edits, which CI by definition never sees, and
    including the offline run itself, which CI does not perform.

WHY THIS EXISTS AS A SCRIPT AND NOT AS A README PARAGRAPH:
    The four commands below were already documented, in four different
    sections of OPERATIONS.md, and a person about to present is exactly
    the person who runs three of them and forgets the fourth. The ORDER
    also matters and prose cannot enforce it -- ruff takes a second and
    catches a typo'd import that would otherwise surface as a confusing
    test error 20 seconds later.

WHAT IT DELIBERATELY DOES NOT DO:
    - It does NOT touch a store, a model, or the network. Every step is
      offline, which is the same guarantee the test suite itself holds
      (D-33), so a failure here is always the code and never the
      environment. Live-service verification is a different question with
      a different tool: scripts/check_services.py, which this script
      points at rather than absorbing.
    - It does NOT grade an answer. scripts/eval_suite.py is the golden-set
      harness and it needs a live deployment; a gate that silently skips
      the interesting half is worse than one that says what it covers.
    - It installs nothing. A missing `ruff` is reported as SKIPPED with the
      one command that fixes it, never treated as a pass and never
      pip-installed behind your back.

Usage:
    python scripts/sanity.py              # ruff, tests, offline run
    python scripts/sanity.py --quick      # skip the slow-marked tests
    python scripts/sanity.py --no-lint    # tests and run only

Exit codes:
    0  every step that ran, passed
    1  a step failed (its own output is above the summary)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

from research_agent.ops._paths import repo_root

# D-157: None when this runs from an installed package. Unlike every
# other command in this package, that is FATAL here rather than a
# different default -- see main(), which says so and exits. ruff needs
# files to lint and pytest needs a tests/ directory; a gate that
# "succeeded" against neither would be the most dangerous kind of green.
REPO_ROOT = repo_root()
SRC = REPO_ROOT / "src" if REPO_ROOT is not None else None

# The offline demo query. Deliberately the one OPERATIONS Step 1b and
# run.bat both use, so a person who has read either recognises the output
# rather than wondering whether this script runs something else.
DEMO_QUERY = "Compare Redis and Memcached for session caching"


class Step:
    """One check: a name, a command, and whether a miss is fatal.

    `optional` is what separates "ruff is not installed" from "ruff found
    a violation". The first is a missing tool and reports SKIPPED; the
    second is a real finding and fails the gate. Nothing else in this
    file needs to know the difference.
    """

    def __init__(self, name: str, argv: list, why: str, optional_tool: str = "",
                 isolated: bool = False):
        self.name = name
        self.argv = argv
        self.why = why
        self.optional_tool = optional_tool
        # `isolated` runs the step in a scratch directory instead of the
        # repo root. Exactly one step needs it, and the reason is not
        # tidiness -- see build_steps' note on the offline run.
        self.isolated = isolated


def build_steps(quick: bool = False, lint: bool = True) -> list:
    """The gate's contents, in COST ORDER. Pure -- builds argv lists and
    returns them, runs nothing, so a test can assert the shape and the
    ordering without executing a single check."""
    steps = []
    if lint:
        steps.append(Step(
            "ruff", ["ruff", "check", "."],
            "a typo'd import surfaces here in one second, or as a "
            "confusing test error twenty seconds later",
            optional_tool="ruff"))
    pytest_argv = [sys.executable, "-m", "pytest", "tests/", "-q"]
    if quick:
        # The marker pytest.ini already defines. -m "not slow" drops the
        # tests that wait out a real socket timeout, which is most of the
        # suite's wall clock and none of its coverage of the graph.
        pytest_argv += ["-m", "not slow"]
    steps.append(Step(
        "tests", pytest_argv,
        "the suite is fully offline, so a failure here is a code "
        "regression and never an environment problem"))
    steps.append(Step(
        "offline run", [sys.executable, "-m", "research_agent.cli", DEMO_QUERY],
        "proves the wiring end to end: L1, no services, no API keys "
        "(see OPERATIONS.md Step 1b for what a healthy result looks like)",
        # RUNS IN A SCRATCH DIRECTORY, and this is the load-bearing part
        # of the whole script. `Settings` reads `.env` RELATIVE TO THE CWD
        # (config.py's `SettingsConfigDict(env_file=".env")`), so a step
        # launched from the repo root inherits whatever the developer's
        # own .env says -- and a working .env on this project routinely
        # says MCP_ENABLED=true, WEB_SEARCH_ENABLED=true, HITL_ENABLED=true.
        # The gate would then be asserting that two MCP servers happen to
        # be running, which is a different question with its own tool
        # (scripts/check_services.py) and no business failing a code gate.
        # Measured, not theorised: the first run of this script failed
        # here for exactly that reason.
        #
        # A scratch CWD rather than a longer list of pinned env vars: the
        # list would need extending every time a setting is added, and
        # forgetting one puts the leak back silently. No file is read from
        # the CWD other than .env, and the run writes only logs/, which
        # belongs in the scratch directory too.
        isolated=True))
    return steps


def format_summary(results: list) -> str:
    """Render the final block. Pure: takes (name, outcome, seconds) and
    returns text, so the wording is testable and this script's last line
    of output can never disagree with its own exit code."""
    width = max((len(name) for name, _, _ in results), default=4)
    lines = ["", "=" * 60, "SANITY", "=" * 60]
    for name, outcome, seconds in results:
        lines.append(f"  {outcome:<8} {name:<{width}}  {seconds:5.1f}s")
    failed = [name for name, outcome, _ in results if outcome == "FAIL"]
    skipped = [name for name, outcome, _ in results if outcome == "SKIP"]
    lines.append("=" * 60)
    if failed:
        lines.append(f"FAILED: {', '.join(failed)} -- output is above.")
    elif skipped:
        lines.append(f"PASSED, with {len(skipped)} step(s) skipped: "
                     f"{', '.join(skipped)}.")
    else:
        lines.append("PASSED. Every step ran.")
    return "\n".join(lines)


def run(step: Step, env: dict) -> tuple:
    """Execute one step. Returns (outcome, seconds)."""
    if step.optional_tool and shutil.which(step.optional_tool) is None:
        print(f"\n--- {step.name}: SKIPPED ({step.optional_tool} is not "
              f"installed -- `pip install {step.optional_tool}`)")
        return "SKIP", 0.0
    print(f"\n--- {step.name}: {' '.join(step.argv)}")
    print(f"    ({step.why})")
    started = time.monotonic()
    if step.isolated:
        with tempfile.TemporaryDirectory(prefix="sanity-") as scratch:
            completed = subprocess.run(step.argv, cwd=scratch, env=env)
    else:
        completed = subprocess.run(step.argv, cwd=REPO_ROOT, env=env)
    elapsed = time.monotonic() - started
    return ("PASS" if completed.returncode == 0 else "FAIL"), elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline pre-demo gate: ruff, the test suite, and one "
                    "L1 run. Stops at the first failure.")
    parser.add_argument("--quick", action="store_true",
                        help="skip slow-marked tests (the ones that wait out "
                             "a real socket timeout)")
    parser.add_argument("--no-lint", action="store_true",
                        help="skip ruff")
    args = parser.parse_args(argv)

    if REPO_ROOT is None:
        # D-157. Every other command in research_agent.ops works from an
        # installed package; this one cannot, and says so rather than
        # inventing a degraded meaning for itself. There is nothing to
        # lint and no tests/ directory to run -- a gate that reported
        # PASSED against neither would be worse than no gate.
        print("scripts/sanity.py needs a checkout: it lints this "
              "repository's source and runs its test suite, and an "
              "installed package contains neither. Run it from a clone.",
              file=sys.stderr)
        return 1

    # PYTHONPATH=src, set HERE rather than demanded of the caller: this
    # script is the thing someone runs when they are in a hurry, and
    # "ModuleNotFoundError: research_agent" is the single most common way
    # this repo wastes that person's next five minutes. An installed
    # package does not need it and is not harmed by it.
    env = dict(os.environ)
    env["PYTHONPATH"] = (str(SRC) + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else str(SRC))
    # The offline run must be offline no matter what .env says. Overriding
    # here, in the CHILD's environment only, means a developer whose .env
    # points at a live model still gets a deterministic gate -- and their
    # .env is not edited to achieve it.
    env["LLM_MODE"] = "stub"

    results = []
    for step in build_steps(quick=args.quick, lint=not args.no_lint):
        outcome, seconds = run(step, env)
        results.append((step.name, outcome, seconds))
        if outcome == "FAIL":
            break  # cost order is pointless if a failure does not stop it

    print(format_summary(results))
    return 1 if any(outcome == "FAIL" for _, outcome, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
