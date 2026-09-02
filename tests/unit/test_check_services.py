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
    # D-157: the implementation moved into the package
    # (research_agent.ops.check_services); scripts/ now holds a thin
    # launcher, and loading THAT would exercise a six-line shim.
    # find_spec locates the module WITHOUT executing it, and the fresh
    # module object below is deliberate: several tests here assert on
    # module-level caching, which a shared sys.modules entry would carry
    # from one test into the next.
    origin = importlib.util.find_spec("research_agent.ops.check_services").origin
    spec = importlib.util.spec_from_file_location(
        "check_services", origin)
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


# ---------------------------------------------------------------------------
# D-81: check_api_server must read /health's BODY, not just its status code
# ---------------------------------------------------------------------------


class _FakeResponse:
    """The two methods check_api_server actually calls on an httpx
    response. Deliberately minimal -- this file opens no sockets (see the
    module docstring), so a real httpx.Response would be more machinery
    than the assertion needs."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        # Every case under test is a real HTTP 200; the unreachable-server
        # case is already covered by check_api_server's own except branch
        # and needs no fake to reach it.
        return None

    def json(self) -> dict:
        return self._payload


def _patch_health(monkeypatch, payload: dict) -> None:
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeResponse(payload))


def test_api_health_200_with_an_error_body_is_a_failure(cs, monkeypatch):
    """D-81 regression, straight from tmp/console-output.txt.

    D-78 makes /health answer HTTP 200 even when the app bundle failed to
    build, precisely so liveness stays reachable and can report WHY --
    the failure lives in the BODY. check_api_server called
    raise_for_status() and nothing else, so it reported a server whose
    every /research and /resume call returns 503 as [PASS]:

        [PASS] FastAPI server ... {'status': 'error', 'detail':
               'ValueError: MCPBridge requires a url (D-76: ...)'}
    """
    _patch_health(monkeypatch, {
        "status": "error",
        "detail": "ValueError: MCPBridge requires a url (D-76: standalone "
                  "Streamable HTTP server only)"})

    status = cs.check_api_server("http://127.0.0.1:8000")

    assert status.ok is False
    assert "MCPBridge requires a url" in status.detail, (
        "the operator needs the actual build error, not just a FAIL")
    assert "503" in status.detail, (
        "and needs to know what it means for the other endpoints")


def test_api_health_200_with_an_ok_body_still_passes(cs, monkeypatch):
    """The healthy path is unchanged by D-81 -- including `durable`, which
    P2-08 surfaces here so a run degraded to in-memory checkpointing is
    visible without reading logs."""
    _patch_health(monkeypatch, {"status": "ok", "llm_mode": "stub",
                                "durable": True})

    status = cs.check_api_server("http://127.0.0.1:8000")

    assert status.ok is True
    assert "durable=True" in status.detail


def test_api_health_body_without_a_status_field_is_not_trusted(cs, monkeypatch):
    """Defense in depth: a 200 from something that is NOT this endpoint
    (a proxy's own health page, a captive portal) has no "status" field.
    Absent is not ok -- the test is `!= "ok"`, never `== "error"`."""
    _patch_health(monkeypatch, {"hello": "from some other service"})

    assert cs.check_api_server("http://127.0.0.1:8000").ok is False


# ---------------------------------------------------------------------------
# D-89: tool discovery via MCP tools/list
# ---------------------------------------------------------------------------


class _ToolsBridge:
    def __init__(self, names=None, raises=None):
        self._names = names or []
        self._raises = raises

    def list_tools(self, timeout_seconds=30.0):
        if self._raises:
            raise self._raises
        return self._names


def test_discovery_reports_what_the_server_actually_exposes(cs):
    suffix, warning = cs._discover_tools(
        _ToolsBridge(["search", "healthcheck"]), "search", 5.0)

    assert "search" in suffix and "healthcheck" in suffix
    assert warning == ""


def test_discovery_warns_when_the_configured_tool_is_absent(cs):
    """A typo in MCP_TOOL_NAME, or a URL pointed at the wrong server, used
    to surface only as a per-TASK failure once retrieval was underway."""
    suffix, warning = cs._discover_tools(
        _ToolsBridge(["web_search"]), "search", 5.0)

    assert "web_search" in suffix
    assert "search" in warning and "NOT" in warning


def test_discovery_degrades_silently_when_the_server_cannot_list(cs):
    """Discovery is extra information about a server that has ALREADY
    answered a real tool call. An SDK or server without tools/list is not
    a broken deployment and must never turn a PASS into a FAIL."""
    assert cs._discover_tools(
        _ToolsBridge(raises=RuntimeError("no tools/list")), "search", 5.0) == ("", "")
    assert cs._discover_tools(_ToolsBridge([]), "search", 5.0) == ("", "")


# ---------------------------------------------------------------------------
# D-111 -- the fallback providers are checked too
# ---------------------------------------------------------------------------


def test_an_unconfigured_provider_is_skipped_not_failed():
    """FallbackRouter omits a keyless provider from the chain entirely, so
    its absence is a choice rather than an outage."""
    status = _load().check_llm_fallback("gemini", "http://x", "", "some-model")

    assert status.skipped is True
    assert status.ok is True


def test_a_4xx_reports_the_status_and_the_body(monkeypatch):
    """The whole reason this check exists: 404 (retired model name),
    401/403 (bad key) and 429 (quota) need different fixes, and the run
    logs could not tell them apart."""
    mod = _load()
    import httpx

    def fake_post(url, **kw):
        return httpx.Response(404, text='{"error": {"message": "model not found"}}')

    monkeypatch.setattr(httpx, "post", fake_post)
    status = mod.check_llm_fallback("gemini", "http://x", "k", "gemini-9.9-flash")

    assert status.ok is False
    assert "404" in status.detail
    assert "model not found" in status.detail
    assert "gemini-9.9-flash" in status.detail


def test_a_working_provider_passes_and_names_the_model(monkeypatch):
    mod = _load()
    import httpx

    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(
        200, json={"choices": [{"message": {"content": "pong"}}]}))
    status = mod.check_llm_fallback("mistral", "http://x", "k", "mistral-small")

    assert status.ok is True and status.skipped is False
    assert "mistral-small" in status.detail


def test_a_transport_failure_is_reported_not_raised(monkeypatch):
    """Same posture as every other check here: report, never traceback."""
    mod = _load()
    import httpx

    def boom(url, **kw):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "post", boom)
    status = mod.check_llm_fallback("gemini", "http://x", "k", "m")

    assert status.ok is False
    assert "ConnectError" in status.detail


def test_the_probe_is_a_chat_completion_not_a_models_listing(monkeypatch):
    """A /models listing can succeed against a good key while the
    CONFIGURED model name is retired -- which is one of the failures this
    check exists to catch."""
    mod = _load()
    import httpx
    seen = {}

    def fake_post(url, **kw):
        seen["url"] = url
        seen["model"] = kw.get("json", {}).get("model")
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    mod.check_llm_fallback("gemini", "http://x/v1", "k", "the-model")

    assert seen["url"].endswith("/chat/completions")
    assert seen["model"] == "the-model"
