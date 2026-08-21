"""
tests/unit/test_check_services.py -- scripts/check_services.py's CONFIGURATION
handling, not its live probes.

Scope, deliberately narrow: the only thing tested here is that the script
agrees with assembly.py about what a valid configuration IS. Nothing in this
file starts a subprocess, opens a socket, or reaches a service -- the live
paths are what the script itself exists to exercise (D-33), and the suite is
offline by design.

Why this file exists at all (D-58): check_mcp() used to hard-FAIL on an empty
MCP_SERVER_COMMAND, while assembly.py resolved that same empty value to
sys.executable and ran happily. So the health check reported a fault on a
configuration that works -- worse than no check, because it sends you hunting
for a problem that is not there. These tests pin the two in agreement.
"""

import importlib.util
import pathlib
import sys

import pytest

from research_agent.config import Settings

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_services", REPO_ROOT / "scripts" / "check_services.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cs():
    return _load()


# ---------------------------------------------------------------------------
# Disabled means SKIPPED, never FAIL
# ---------------------------------------------------------------------------


def test_mcp_disabled_is_skipped_not_failed(cs):
    """MCP_ENABLED=false is this repo's default and a correct, working
    configuration. Reporting it as a failure would train people to ignore the
    output."""
    status = cs.check_mcp(Settings(_env_file=None, mcp_enabled=False))
    assert status.ok is True and status.skipped is True


def test_web_search_disabled_is_skipped_not_failed(cs):
    status = cs.check_web_search(Settings(_env_file=None, web_search_enabled=False))
    assert status.ok is True and status.skipped is True


# ---------------------------------------------------------------------------
# D-58: an empty server command is VALID, and means sys.executable
# ---------------------------------------------------------------------------


def test_check_mcp_does_not_reject_an_empty_server_command(cs, monkeypatch):
    """THE regression this file was written for.

    Asserts on the DETAIL string rather than on ok/latency, so the test needs
    no subprocess: whatever the outcome of the live spawn, the one thing that
    must never come back is the old "is empty -- misconfigured" verdict.
    """
    captured = {}

    class _Bridge:
        def __init__(self, command, args, env_allowlist):
            captured["command"] = command
            captured["args"] = args

        def call_tool(self, *a, **kw):
            raise RuntimeError("no live server in tests")

        def close(self):
            pass

    monkeypatch.setattr(
        "research_agent.tools.mcp_client.MCPBridge", _Bridge)

    status = cs.check_mcp(Settings(
        _env_file=None, mcp_enabled=True, mcp_server_command="",
        mcp_server_args="scripts/mcp_corpus_server.py"))

    assert "misconfigured" not in status.detail, (
        "an empty MCP_SERVER_COMMAND is valid since D-58 -- assembly.py "
        "resolves it to sys.executable, and this check must agree")
    assert captured["command"] == sys.executable, (
        "check_mcp must resolve the command through resolve_server_command, "
        "the same helper assembly.py uses -- otherwise the check exercises a "
        "different launch path than a real run")


def test_check_mcp_resolves_args_against_the_repo_root(cs, monkeypatch):
    """A relative MCP_SERVER_ARGS must not depend on the current working
    directory -- MCPBridge never sets StdioServerParameters.cwd."""
    captured = {}

    class _Bridge:
        def __init__(self, command, args, env_allowlist):
            captured["args"] = args

        def call_tool(self, *a, **kw):
            raise RuntimeError("no live server in tests")

        def close(self):
            pass

    monkeypatch.setattr("research_agent.tools.mcp_client.MCPBridge", _Bridge)
    monkeypatch.chdir(REPO_ROOT.parent)

    cs.check_mcp(Settings(_env_file=None, mcp_enabled=True,
                          mcp_server_command="",
                          mcp_server_args="scripts/mcp_corpus_server.py"))

    assert captured["args"] == [
        str(REPO_ROOT / "scripts" / "mcp_corpus_server.py")]


def test_both_mcp_checks_agree_about_an_empty_command(cs, monkeypatch):
    """The corpus and web-search checks must not disagree about what a valid
    command is -- they front the same MCPBridge and the same resolution
    helpers, and a difference between them is always a bug in one of them."""
    class _Bridge:
        def __init__(self, command, args, env_allowlist):
            pass

        def call_tool(self, *a, **kw):
            raise RuntimeError("no live server in tests")

        def close(self):
            pass

    monkeypatch.setattr("research_agent.tools.mcp_client.MCPBridge", _Bridge)

    corpus = cs.check_mcp(Settings(
        _env_file=None, mcp_enabled=True, mcp_server_command=""))
    web = cs.check_web_search(Settings(
        _env_file=None, web_search_enabled=True, web_mcp_server_command=""))

    assert ("misconfigured" in corpus.detail) == ("misconfigured" in web.detail)