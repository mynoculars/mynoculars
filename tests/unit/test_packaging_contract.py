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
# API_ASYNC_WORKERS is D-134's async-run switch. That decision record
# describes `POST /research {"wait": false}`, `GET /result/{thread_id}`
# and a bounded worker pool, and names `api/server.py` and `config.py` as
# where it lives -- but in this tree `config.py` declares no such field
# and `api/server.py` registers no such route, so the key in
# `.env.example` is inert: setting it to any value does nothing, and
# nothing says so. Either the feature lands and this entry goes, or the
# key goes; this list is here so that choice gets made rather than
# forgotten again.
KNOWN_UNMAPPED_ENV_KEYS = {"API_ASYNC_WORKERS"}


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
