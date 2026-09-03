"""
tests/unit/test_packaging_contract.py -- the declared public interface is
real (D-161).

`pyproject.toml` names two things as this project's public interface that
no test had ever checked, and both are the kind that break silently:

    the .env setting NAMES in config.py
    every console script's NAME and arguments (all ten, since D-157)

A setting documented in `.env.example` with no matching `Settings` field
does NOT raise -- `model_config` sets `extra="ignore"` deliberately, so
that a stray environment variable from an unrelated tool cannot stop the
process. The cost of that choice is that a key which LOOKS configured
does nothing at all, and the person who set it has no way to find out.
`warn_on_likely_env_typos` catches a fixed list of plausible
mis-spellings; it cannot catch a key that is spelled exactly right and
simply has nothing reading it.

An entry point naming a module that moved, or a `main` that was renamed,
fails at `pip install` time for a CONSUMER and never for us -- the test
suite imports modules directly and would not notice.

Both checks read the SAME files the packaging metadata reads, so neither
can pass by agreeing with a copy of itself.
"""

import pathlib
import re
import sys
import tomllib

import pytest

from research_agent.config import Settings

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


# Keys in .env.example with no Settings field behind them, PINNED rather
# than tolerated. The assertion below tests equality, not membership, so
# fixing one of these fails this test and forces the list to be updated
# deliberately -- the same shape as
# test_config.py::test_max_task_retries_no_longer_exists_at_all.
#
# EMPTY, and that is the point. It held exactly one entry --
# API_ASYNC_WORKERS, D-134's async-run switch -- from D-161 until now.
# That decision record describes `POST /research {"wait": false}`,
# `GET /result/{thread_id}` and a bounded worker pool, and names
# `api/server.py` and `config.py` as where it lives; neither the field
# nor the route exists in this tree. D-161 pinned it here rather than
# editing four documents on the strength of one snapshot (D-160's rule).
# THE CHOICE HAS NOW BEEN MADE: the key is gone from `.env.example` and
# README/OPERATIONS say the feature is designed and not built, so there
# is nothing left to pin. The design itself survives in DECISIONS.md
# D-134; if it is built, add the Settings field FIRST and the
# `.env.example` key second, and this set stays empty.
KNOWN_UNMAPPED_ENV_KEYS: set = set()

# Settings fields deliberately absent from `.env.example`. Empty, and the
# test below asserts equality, so leaving one out is a decision someone has
# to record here rather than an omission that goes unnoticed.
KNOWN_UNDOCUMENTED_SETTINGS: set = set()


def test_every_env_example_key_maps_to_a_real_setting():
    """A key in `.env.example` that no field reads is documentation of a
    control that does not exist."""
    from dotenv import dotenv_values

    keys = set(dotenv_values(REPO_ROOT / ".env.example").keys())
    fields = {name.upper() for name in Settings.model_fields}

    unmapped = keys - fields

    assert unmapped == KNOWN_UNMAPPED_ENV_KEYS, (
        f"unmapped keys changed: {sorted(unmapped)}. A key here is read by "
        f"nobody -- extra='ignore' means setting it is silently a no-op. "
        f"Add the Settings field, remove the key, or (only with a reason) "
        f"add it to KNOWN_UNMAPPED_ENV_KEYS.")


def test_every_setting_is_documented_in_env_example():
    """The OTHER direction, which nothing checked (S-20).

    The test above catches a key with no field behind it. It cannot catch
    the inverse -- a real, readable Settings field that `.env.example`
    never mentions -- and that is the direction this project actually
    drifts, because a field is added in `config.py` where the code needs
    it and the example file is a separate edit nobody is forced to make.

    IT HAD FOUND SEVEN when this test was written: MEMORY_WRITE_MIN_SCORE,
    QUALITY_JUDGE_WARN_RATIO, MODEL_KNOWLEDGE_SCORE,
    QUERY_REFORMULATION_ENABLED, MAX_ESCALATIONS, RUN_CALL_BUDGET_WARN and
    CLAIM_VERIFICATION_ENABLED. Two of those (QUALITY_JUDGE_WARN_RATIO,
    RUN_CALL_BUDGET_WARN) are documented in README's own guardrails table,
    so the README offered a control the example file did not, and
    MODEL_KNOWLEDGE_SCORE has a dedicated startup warning
    (warn_on_model_knowledge_inert, D-163) for a misconfiguration you could
    not reach `.env.example` to cause.

    Equality, not subset, and the allowlist is empty: a field deliberately
    left out has to be named here, so the choice is made rather than
    forgotten -- the same shape as KNOWN_UNMAPPED_ENV_KEYS above.
    """
    from dotenv import dotenv_values

    keys = set(dotenv_values(REPO_ROOT / ".env.example").keys())
    fields = {name.upper() for name in Settings.model_fields}

    undocumented = fields - keys

    assert undocumented == KNOWN_UNDOCUMENTED_SETTINGS, (
        f"undocumented settings changed: {sorted(undocumented)}. A field "
        f"with no key in .env.example is a tunable nobody reading that "
        f"file can discover. Add the key with a comment saying what it "
        f"guards, or (only with a reason) add it to "
        f"KNOWN_UNDOCUMENTED_SETTINGS.")


def test_every_console_script_resolves_to_a_real_module_and_main():
    """Ten names since D-157, each one a promise to a consumer's
    Dockerfile. A moved module or a renamed `main` breaks `pip install`
    for them and nothing here would otherwise notice.

    find_spec locates the module WITHOUT importing it, deliberately: two
    of these ARE the MCP servers, and importing one constructs a FastMCP
    instance and a thread pool as an import side effect. The `main`
    check is therefore a source scan rather than a getattr, which cannot
    catch a `main` that exists but fails at runtime -- that is what the
    module's own tests are for."""
    import importlib.util

    scripts = _pyproject()["project"]["scripts"]
    assert scripts, "pyproject declares no console scripts"

    for command, target in scripts.items():
        module_name, _, attr = target.partition(":")
        assert attr, f"{command}: entry point {target!r} names no attribute"

        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin, (
            f"{command}: entry point points at {module_name!r}, which is "
            f"not importable")

        source = pathlib.Path(spec.origin).read_text(encoding="utf-8")
        assert re.search(rf"^def {re.escape(attr)}\(", source, re.MULTILINE), (
            f"{command}: {module_name} defines no {attr}()")


def test_the_console_scripts_cover_every_ops_and_server_module():
    """The other direction: a command added to research_agent.ops or
    research_agent.servers and NOT given an entry point is invisible to
    an installed package -- which is the exact defect D-157 existed to
    fix, reintroduced one module at a time.

    `sanity` is the deliberate exception and says so in its own module:
    it lints this repository and runs its suite, so it cannot work from
    an install and is reachable only through scripts/."""
    targets = {t.partition(":")[0] for t in _pyproject()["project"]["scripts"].values()}
    exempt = {"research_agent.ops.sanity", "research_agent.ops._paths"}

    for package in ("ops", "servers"):
        directory = REPO_ROOT / "src" / "research_agent" / package
        for module_path in sorted(directory.glob("*.py")):
            if module_path.name == "__init__.py":
                continue
            dotted = f"research_agent.{package}.{module_path.stem}"
            if dotted in exempt:
                continue
            assert dotted in targets, (
                f"{dotted} has no console script, so an installed package "
                f"cannot run it (D-157). Add one to [project.scripts], or "
                f"add it to this test's `exempt` set with a reason.")


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib is 3.11+")
def test_requirements_and_pyproject_do_not_disagree_on_a_pin():
    """requirements.txt is the dev pin-set and pyproject is what a
    consumer resolves against; pyproject's own note says "if you add a
    dependency, add it to both or they will drift". This checks the one
    kind of drift that actually breaks an install: the same package
    pinned to incompatible ranges in the two files.

    Deliberately narrow -- it does NOT require the two lists to be equal.
    requirements.txt legitimately carries dev-only entries (pytest,
    pytest-xdist) and the extras legitimately carry what requirements.txt
    installs unconditionally."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    req = {}
    for line in text.splitlines():
        # Strip the comment AND the environment marker: `pywin32>=311;
        # platform_system == "Windows"` is the same pin as pyproject's
        # `pywin32>=311` under its own marker, and comparing one with the
        # marker against one without reports a drift that is not there.
        line = line.split("#")[0].split(";")[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\])?\s*(.*)$", line)
        if line and match and match.group(3):
            req[match.group(1).lower()] = match.group(3).replace(" ", "")

    project = _pyproject()["project"]
    declared = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)

    for entry in declared:
        entry = entry.split(";")[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\])?\s*(.*)$", entry)
        if not match or not match.group(3):
            continue
        name, spec = match.group(1).lower(), match.group(3).replace(" ", "")
        if name in req:
            assert req[name] == spec, (
                f"{name}: requirements.txt pins {req[name]!r} but "
                f"pyproject declares {spec!r} -- a consumer and a "
                f"developer would install different versions")


def test_no_env_example_key_is_assigned_twice():
    """D-162: `LLM_FALLBACK_BASE_URL` was assigned twice, and dotenv takes
    the LAST -- so editing the first one, the one under its own heading
    and the first a reader meets, did nothing at all. It was invisible
    only because both values happened to be identical.

    No warning could have caught it: both spellings are correct, so
    `warn_on_likely_env_typos` has nothing to compare them against. A
    duplicate key is only ever a mistake in a file whose whole job is to
    be copied and edited."""
    from collections import Counter

    keys = [line.split("=", 1)[0].strip()
            for line in (REPO_ROOT / ".env.example").read_text(
                encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line]

    duplicates = {key: n for key, n in Counter(keys).items() if n > 1}

    assert not duplicates, (
        f"assigned more than once in .env.example: {duplicates}. dotenv "
        f"keeps the last, so every earlier assignment is silently dead.")
