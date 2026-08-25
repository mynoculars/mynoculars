"""
tests/unit/test_check_services.py -- scripts/check_services.py's
CONFIGURATION handling, not its live probes.

Scope, deliberately narrow: the only thing tested here is that
check_mcp/check_web_search wire the right Settings fields into MCPBridge
and treat disabled/skipped correctly. Nothing in this file opens a real
socket or reaches a service -- the live paths are what the script itself
exists to exercise (D-33), and the suite is offline by design.

D-76 rewrite: this file previously covered D-58's "empty
MCP_SERVER_COMMAND resolves to sys.executable" agreement between
check_mcp and assembly.py -- that whole command/args/subprocess concept
no longer exists (MCPBridge connects to a standalone server URL only).
These tests now pin the URL-based equivalent: check_mcp/check_web_search
must construct MCPBridge with exactly settings.mcp_server_url /
settings.web_mcp_server_url, and nothing else.
"""

import importlib.util
import pathlib

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


class _FakeBridge:
    """Captures the url MCPBridge was constructed with, and raises on
    any call_tool -- these tests are about CONFIGURATION wiring, not
    about a real connection succeeding or failing."""

    def __init__(self, url):
        self.url = url

    def call_tool(self, *a, **kw):
        raise RuntimeError("no live server in tests")

    def close(self):
        pass


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
# D-76: check_mcp/check_web_search connect to the configured URL, nothing
# is spawned
# ---------------------------------------------------------------------------


def test_check_mcp_connects_to_the_configured_url(cs, monkeypatch):
    """check_mcp must construct MCPBridge with exactly
    settings.mcp_server_url -- the same field assembly.py reads to wire
    the real bridge, so this check exercises the real configuration
    rather than a lookalike that could pass while a real run fails."""
    monkeypatch.setattr("research_agent.tools.mcp_client.MCPBridge", _FakeBridge)

    status = cs.check_mcp(Settings(
        _env_file=None, mcp_enabled=True,
        mcp_server_url="http://127.0.0.1:8765/mcp"))

    assert "http://127.0.0.1:8765/mcp" in status.detail


def test_check_web_search_connects_to_the_configured_url(cs, monkeypatch):
    monkeypatch.setattr("research_agent.tools.mcp_client.MCPBridge", _FakeBridge)

    status = cs.check_web_search(Settings(
        _env_file=None, web_search_enabled=True,
        web_mcp_server_url="http://127.0.0.1:8766/mcp"))

    assert "http://127.0.0.1:8766/mcp" in status.detail


def test_check_mcp_failure_message_points_at_starting_the_standalone_server(cs, monkeypatch):
    """A FAIL here most likely means the standalone server just isn't
    running -- the message should say so, not describe a spawn failure
    (which is no longer possible; nothing is spawned, D-76)."""
    class _RaisingBridge:
        def __init__(self, url):
            pass

        def call_tool(self, *a, **kw):
            raise ConnectionRefusedError("no one listening")

        def close(self):
            pass

    monkeypatch.setattr("research_agent.tools.mcp_client.MCPBridge", _RaisingBridge)

    status = cs.check_mcp(Settings(
        _env_file=None, mcp_enabled=True,
        mcp_server_url="http://127.0.0.1:8765/mcp"))

    assert status.ok is False
    assert "standalone" in status.detail.lower()
