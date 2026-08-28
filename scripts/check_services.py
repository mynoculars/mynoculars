"""
scripts/check_services.py -- "Is everything up?" in one command.

Checks the live dependencies this repo's Level 2/3 setup (plus two
optional, opt-in components) needs (OPERATIONS.md's own "60-Second Is
Everything Up? Check" walks through the original four by hand, one
command at a time -- this script automates that check and adds a clear
PASS/FAIL/SKIP summary):

    1. Qdrant       (dense vector store)
    2. OpenSearch   (keyword/BM25 store)
    3. Postgres     (checkpointer + run history)
    4. LLM primary  (the local Llama/Qwen inference engine, llama.cpp's
                      server or equivalent, OpenAI-compatible /v1/models)
    5. MCP server   (connects to the ALREADY-RUNNING standalone server at
                      MCP_SERVER_URL, ONLY if MCP_ENABLED=true in .env;
                      SKIPPED, not failed, when it's off, since that's this
                      repo's default and a correct, working configuration.
                      D-83: this paragraph used to describe a fresh stdio
                      subprocess spawned per invocation from
                      MCP_SERVER_COMMAND -- D-76 deleted that transport
                      outright. Nothing in this repo spawns an MCP server
                      any more; you start and stop it yourself, so a PASS
                      here means "that server is up and answering", not
                      "the spawn mechanism works")
    6. Web search   (the SECOND standalone MCP server, at
                      WEB_MCP_SERVER_URL, ONLY if WEB_SEARCH_ENABLED=true;
                      SKIPPED when off. This row is the ONLY live
                      verification of the web-search path anywhere in this
                      repo -- the test suite stops at a fake provider by
                      design)
    7. FastAPI server (api/server.py's own /health endpoint -- this is
                      a separate, optional way to run this codebase
                      alongside the CLI, not required for L1/L2/L3;
                      pass --skip-api if you only ever use the CLI. Note
                      D-78: /health answers HTTP 200 even when the app
                      bundle FAILED to build, carrying that failure in the
                      response BODY -- so this check reads the body's own
                      "status" field, never just the HTTP status code)

Deliberately NOT a pytest test in tests/ -- this repo's test suite is
proudly, entirely offline (see OPERATIONS.md "Running and Interpreting
the Test Suite"); a test that depends on live services has no place
silently running (or silently skipping) inside that guarantee. This is
a standalone, opt-in script instead, exactly like scripts/reset_stores.py
and scripts/ingest_sample_data.py.

Usage:
    python scripts/check_services.py
    python scripts/check_services.py --json     # machine-readable output
    python scripts/check_services.py --quiet    # only print failures
    python scripts/check_services.py --skip-api                  # CLI-only workflow, no FastAPI server
    python scripts/check_services.py --api-url http://127.0.0.1:9000  # non-default uvicorn port

Exit code: 0 if every ATTEMPTED service is reachable (a SKIPPED check,
e.g. MCP when MCP_ENABLED=false, never affects this), 1 if any single
one failed. This makes it usable as a precondition in your own scripts:
    python scripts/check_services.py && python -m research_agent.cli "..."
"""
import argparse
import pathlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Resolve "src" RELATIVE TO THIS FILE, never relative to the current
# working directory. `sys.path.insert(0, "src")` only resolved when the
# process happened to be launched from the repo root -- not guaranteed
# for a script launched as an MCP_SERVER_COMMAND subprocess, from a
# Windows shortcut or scheduled task, or from any other directory --
# and failed with an opaque ModuleNotFoundError when it did not.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from research_agent.config import get_settings  # noqa: E402


@dataclass
class ServiceStatus:
    name: str
    ok: bool
    detail: str
    latency_s: Optional[float] = None
    skipped: bool = False


def check_qdrant(url: str) -> ServiceStatus:
    """A collections listing is Qdrant's own cheapest liveness probe --
    the same one storage/qdrant_store.py::QdrantStore.__init__ uses."""
    t0 = time.time()
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=url, timeout=5)
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        return ServiceStatus(
            "Qdrant", True,
            f"{url} -- {len(names)} collection(s): {', '.join(names) or '(none)'}",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus("Qdrant", False, f"{url} -- {type(e).__name__}: {e}",
                              time.time() - t0)


def check_opensearch(url: str, username: str, password: str,
                      use_ssl: bool, verify_certs: bool) -> ServiceStatus:
    """A cluster health check is OpenSearch's own cheapest liveness probe."""
    t0 = time.time()
    try:
        from opensearchpy import OpenSearch
        kwargs = {"hosts": [url], "timeout": 5}
        if username:
            kwargs["http_auth"] = (username, password)
        if use_ssl:
            kwargs["use_ssl"] = True
            kwargs["verify_certs"] = verify_certs
        client = OpenSearch(**kwargs)
        health = client.cluster.health()
        return ServiceStatus(
            "OpenSearch", True,
            f"{url} -- cluster status: {health.get('status', 'unknown')}",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus("OpenSearch", False, f"{url} -- {type(e).__name__}: {e}",
                              time.time() - t0)


def check_postgres(dsn: str) -> ServiceStatus:
    """SELECT 1 is the cheapest real round trip; also confirms auth,
    not just that the port is open."""
    t0 = time.time()
    try:
        import psycopg
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        # Don't echo the DSN verbatim -- it contains credentials.
        safe = dsn.split("@")[-1] if "@" in dsn else dsn
        return ServiceStatus("Postgres", True, f"...@{safe} -- SELECT 1 OK",
                              time.time() - t0)
    except Exception as e:
        safe = dsn.split("@")[-1] if "@" in dsn else dsn
        return ServiceStatus("Postgres", False, f"...@{safe} -- {type(e).__name__}: {e}",
                              time.time() - t0)


def check_llm_primary(base_url: str) -> ServiceStatus:
    """/v1/models is the OpenAI-compatible endpoint every provider this
    codebase talks to (llama.cpp's server included) is expected to
    expose -- see llm/client.py. A real chat completion isn't needed
    just to prove the engine is up and serving."""
    t0 = time.time()
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        model_ids = [m.get("id", "?") for m in data.get("data", [])]
        return ServiceStatus(
            "LLM primary", True,
            f"{base_url} -- model(s): {', '.join(model_ids) or '(none reported)'}",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus("LLM primary", False, f"{base_url} -- {type(e).__name__}: {e}",
                              time.time() - t0)


def check_llm_fallback(name: str, base_url: str, api_key: str,
                       model: str) -> ServiceStatus:
    """One configured fallback provider, probed the way the agent uses it.

    D-111. Until this existed the only LLM row was `check_llm_primary`, so
    two of the three providers in the default chain were never checked by
    the script whose entire job is "which services are actually reachable
    right now". Live (runs p205.260/.261) gemini failed on every call it
    was given for at least five consecutive runs and nothing here would
    have said so.

    A real `/chat/completions` POST, not the `/models` listing
    `check_llm_primary` uses, and the difference is the point: a listing
    can succeed against a perfectly good key while the CONFIGURED model
    name is retired, which is one of the failures this is meant to catch.
    One token of output is enough to prove the model answers.

    Reports the status code on a 4xx/5xx, which is what separates the
    three realistic causes -- 404 a wrong or retired model name,
    401/403 a bad key or a disabled API, 429 an exhausted quota.

    A provider with no API key is SKIPPED, not failed: FallbackRouter
    omits it from the chain entirely (see from_settings), so an
    unconfigured provider is a choice rather than an outage.
    """
    label = f"LLM {name}"
    if not api_key:
        return ServiceStatus(label, True, "no API key set -- not in the chain",
                             None, skipped=True)
    t0 = time.time()
    try:
        import httpx
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "ping"}]},
            timeout=15)
        if resp.status_code >= 400:
            # The provider's own error text, capped. This is the line that
            # answers "why is it failing"; without it the caller is back
            # to guessing from an exception class name.
            return ServiceStatus(
                label, False,
                f"{model} -- HTTP {resp.status_code}: {resp.text[:200]}",
                time.time() - t0)
        return ServiceStatus(label, True, f"{model} -- answered",
                             time.time() - t0)
    except Exception as e:  # noqa: BLE001 -- report, never traceback
        return ServiceStatus(label, False, f"{model} -- {type(e).__name__}: {e}",
                             time.time() - t0)


def _discover_tools(bridge, configured: str, timeout: float):
    """Ask a bridge what tools its server exposes (D-89).

    Returns (suffix, warning) -- a string to append to the PASS detail
    naming what the server actually offers, and a non-empty warning when
    the CONFIGURED tool name is absent from that list.

    Never raises and never turns a working service into a FAIL: discovery
    is extra information about a server that has ALREADY answered a real
    tool call by the time this runs. An SDK or server that does not
    support tools/list is not a broken deployment, so it degrades to a
    silent no-op -- the same graceful-degradation posture every storage
    module in this repo takes.
    """
    try:
        names = bridge.list_tools(timeout_seconds=timeout)
    except Exception:  # noqa: BLE001 -- discovery is a bonus, never a gate
        return "", ""
    if not names:
        return "", ""
    suffix = f" [server exposes: {', '.join(sorted(names))}]"
    if configured not in names:
        return suffix, (f" -- WARNING: configured tool {configured!r} is NOT "
                        f"among them; this run answered, but check the "
                        f"configured name against that list")
    return suffix, ""


def check_mcp(settings) -> ServiceStatus:
    """Only runs when settings.mcp_enabled is True -- MCP_ENABLED=false
    (this repo's default) is a correct, working configuration, not a
    failure, so that case is reported as SKIPPED rather than FAIL.

    D-76: MCP is a real STANDING service, exactly like Qdrant/OpenSearch/
    Postgres/the LLM engine above -- an independent, already-running
    server at settings.mcp_server_url, started and stopped by you,
    entirely separately from this script or any `research_agent` run.
    This check CONNECTS to that server and calls the configured tool
    once; unlike every OTHER check in this script, a FAIL here most
    likely means the standalone server simply isn't running -- see
    OPERATIONS.md's "Running the MCP servers standalone" for how to
    start it.
    """
    if not settings.mcp_enabled:
        return ServiceStatus("MCP server", True, "MCP_ENABLED=false -- skipped, not checked",
                              skipped=True)

    from research_agent.tools.mcp_client import MCPBridge

    t0 = time.time()
    bridge = MCPBridge(url=settings.mcp_server_url)
    try:
        result = bridge.call_tool(
            settings.mcp_tool_name,
            {settings.mcp_query_arg_name: "health check"},
            timeout_seconds=settings.mcp_call_timeout_seconds)
        item_count = len(getattr(result, "content", []) or [])
        suffix, warning = _discover_tools(
            bridge, settings.mcp_tool_name, settings.mcp_call_timeout_seconds)
        return ServiceStatus(
            "MCP server", True,
            f"{settings.mcp_server_url} -- "
            f"tool '{settings.mcp_tool_name}' responded, {item_count} content item(s)"
            f"{suffix}{warning}",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus(
            "MCP server", False,
            f"{settings.mcp_server_url} -- "
            f"{type(e).__name__}: {e} "
            "(most likely: nothing is listening at MCP_SERVER_URL -- start "
            "the standalone server; see OPERATIONS.md 'Running the MCP "
            "servers standalone')",
            time.time() - t0)
    finally:
        bridge.close()


def check_web_search(settings) -> ServiceStatus:
    """Phase 4 (D-57). Same standing-service model as check_mcp above
    (D-76) -- read that docstring first.

    THE ONE THING THIS CHECK CATCHES THAT NOTHING ELSE DOES: the
    standalone web-search server is the only part of the system that
    makes OUTBOUND INTERNET requests. If ITS OWN environment (set however
    you normally would, on the machine/terminal running it -- this
    process no longer configures it, D-76) is missing proxy variables
    behind a corporate proxy, every search fails as an opaque timeout and
    no unit test can tell you -- the entire suite is offline by design.
    This is where that shows up, which is exactly what D-33 put this
    script here for.

    A PASS means a live query reached a real search engine and came back
    scored. That makes this the ONLY verification of the live DDGS path
    in the repo; the unit tests deliberately stop at a fake provider.
    """
    if not settings.web_search_enabled:
        return ServiceStatus("Web search (MCP)", True,
                              "WEB_SEARCH_ENABLED=false -- skipped, not checked",
                              skipped=True)

    from research_agent.tools.mcp_client import MCPBridge

    t0 = time.time()
    bridge = MCPBridge(url=settings.web_mcp_server_url)
    try:
        result = bridge.call_tool(
            settings.web_mcp_tool_name,
            {settings.web_mcp_query_arg_name: "site reliability engineering"},
            timeout_seconds=settings.web_mcp_call_timeout_seconds)
        # Read structuredContent, the channel make_web_search_tool reads --
        # so this check exercises the same path a real run does rather than
        # a lookalike.
        structured = getattr(result, "structuredContent", None) or {}
        items = structured.get("result") if isinstance(structured, dict) else None
        items = items if isinstance(items, list) else []
        domains = len({i.get("domain") for i in items if isinstance(i, dict)})
        if not items:
            return ServiceStatus(
                "Web search (MCP)", False,
                f"tool '{settings.web_mcp_tool_name}' responded but returned NO "
                "results for a deliberately broad query. Most likely the engine "
                "is throttling this host, or the standalone server's own "
                "process has no network route -- check its environment "
                "includes your proxy variables if you are behind one",
                time.time() - t0)
        return ServiceStatus(
            "Web search (MCP)", True,
            f"{settings.web_mcp_server_url} -- "
            f"tool '{settings.web_mcp_tool_name}' returned {len(items)} scored "
            f"result(s) across {domains} domain(s) via "
            f"WEB_SEARCH_PROVIDER={settings.web_search_provider}"
            f"{_discover_tools(bridge, settings.web_mcp_tool_name, settings.web_mcp_call_timeout_seconds)[0]}"
            " (the ONLY live verification of the search path in this repo -- "
            "the unit suite stops at a fake provider by design)",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus(
            "Web search (MCP)", False,
            f"{settings.web_mcp_server_url} -- "
            f"{type(e).__name__}: {e} "
            "(most likely: nothing is listening at WEB_MCP_SERVER_URL -- "
            "start the standalone server; see OPERATIONS.md 'Running the "
            "MCP servers standalone')",
            time.time() - t0)
    finally:
        bridge.close()


def check_api_server(base_url: str) -> ServiceStatus:
    """GET /health -- api/server.py's own liveness endpoint. Optional
    component: this codebase runs perfectly well via the CLI alone, so a
    FAIL here just means "the FastAPI server isn't running right now",
    not that anything is broken -- pass --skip-api if you never run it.

    D-81: TWO different failures are possible here and only one of them is
    an HTTP-level failure.

      1. The process is not up / not reachable -- caught by the except
         below, as it always was.
      2. The process IS up but its app bundle FAILED to build, so every
         /research and /resume call returns 503. D-78 made /health answer
         **HTTP 200** in exactly this case, on purpose: liveness must stay
         reachable so it can report WHY the deeper build failed. The
         failure therefore lives in the response BODY ("status": "error",
         plus a "detail"), never in the status code.

    raise_for_status() cannot see case 2, so this function used to report
    a completely unusable server as [PASS] -- inverting the entire point
    of D-78. Live evidence (tmp/console-output.txt):

        [PASS] FastAPI server http://127.0.0.1:8000/health --
               {'status': 'error', 'detail': 'ValueError: MCPBridge
                requires a url (D-76: ...)'}

    The one tool whose job is to answer "is anything down" was answering
    "everything is fine" about a server that could not serve a single
    request. Reading the field D-78 actually sets is the whole fix.
    """
    t0 = time.time()
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # D-81 case 2. `!= "ok"` rather than `== "error"`: an unrecognised
        # status is not something to pass optimistically, and a body with
        # no status field at all is not this endpoint answering.
        if data.get("status") != "ok":
            return ServiceStatus(
                "FastAPI server", False,
                f"{base_url}/health -- reachable, but the server's app "
                f"bundle FAILED to build: {data.get('detail') or data}. "
                f"Every /research and /resume call returns 503 until the "
                f"underlying config is fixed and the server restarted "
                f"(D-78; see the server's own startup log for the full "
                f"traceback)",
                time.time() - t0)
        durable = data.get("durable", "?")
        return ServiceStatus(
            "FastAPI server", True,
            f"{base_url}/health -- {data} (durable={durable})",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus(
            "FastAPI server", False,
            f"{base_url}/health -- {type(e).__name__}: {e} "
            "(not required for CLI-only use -- see --skip-api)",
            time.time() - t0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    parser.add_argument("--skip-api", action="store_true",
                       help="don't check the FastAPI server (api/server.py) at all")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                       help="base URL for the FastAPI server (default: uvicorn's own "
                            "default host/port -- override if you started it with "
                            "--port or a different host)")
    args = parser.parse_args()

    settings = get_settings()

    results = [
        check_qdrant(settings.qdrant_url),
        check_opensearch(settings.opensearch_url, settings.opensearch_username,
                          settings.opensearch_password, settings.opensearch_use_ssl,
                          settings.opensearch_verify_certs),
        check_postgres(settings.postgres_dsn),
        check_llm_primary(settings.llm_primary_base_url),
        # D-111: the same two rows FallbackRouter.from_settings builds the
        # chain from, in the same order, so this table and the chain can
        # never disagree about who is configured.
        check_llm_fallback("mistral", settings.llm_mistral_base_url,
                           settings.llm_mistral_api_key, settings.llm_mistral_model),
        # D-114: named from settings, so this row cannot say "gemini"
        # while probing something else.
        check_llm_fallback(settings.llm_fallback_name,
                           settings.llm_fallback_base_url,
                           settings.llm_fallback_api_key, settings.llm_fallback_model),
        check_mcp(settings),
        check_web_search(settings),
    ]
    if not args.skip_api:
        results.append(check_api_server(args.api_url))

    # SKIPPED checks never count against readiness -- only real FAILs do.
    all_ok = all(r.ok or r.skipped for r in results)

    if args.json:
        print(json.dumps({"all_ok": all_ok, "services": [asdict(r) for r in results]},
                          indent=2))
        return 0 if all_ok else 1

    if not args.quiet:
        print("Service health check")
        print("=" * 60)

    for r in results:
        if args.quiet and r.ok:
            continue
        mark = "SKIP" if r.skipped else ("PASS" if r.ok else "FAIL")
        latency = f"  ({r.latency_s:.2f}s)" if r.latency_s is not None else ""
        print(f"[{mark}] {r.name:<15} {r.detail}{latency}")

    if not args.quiet:
        print("=" * 60)
        if all_ok:
            skipped = [r.name for r in results if r.skipped]
            suffix = f" ({', '.join(skipped)} skipped)" if skipped else ""
            print(f"All checked services reachable.{suffix}")
        else:
            failed = [r.name for r in results if not r.ok and not r.skipped]
            print(f"NOT READY -- failed: {', '.join(failed)}")
            print("See OPERATIONS.md 'Troubleshooting Common Errors' for the "
                  "specific failure signatures above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())