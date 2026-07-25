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
    5. MCP server   (spawns a FRESH, throwaway instance of whatever
                      MCP_SERVER_COMMAND points at, ONLY if MCP_ENABLED=true
                      in .env; SKIPPED, not failed, when it's off, since
                      that's this repo's default and a correct, working
                      configuration. Unlike the four checks above, there
                      is NO persistent MCP server to check -- cli.py
                      itself spawns one fresh per invocation, on demand,
                      via stdio (D-30); a PASS here means "the spawn
                      mechanism works right now", not "a server is up")
    6. FastAPI server (api/server.py's own /health endpoint -- this is
                      a separate, optional way to run this codebase
                      alongside the CLI, not required for L1/L2/L3;
                      pass --skip-api if you only ever use the CLI)

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
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, "src")

from research_agent.config import get_settings, split_csv  # noqa: E402


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


def check_mcp(settings) -> ServiceStatus:
    """Only runs when settings.mcp_enabled is True -- MCP_ENABLED=false
    (this repo's default) is a correct, working configuration, not a
    failure, so that case is reported as SKIPPED rather than FAIL.

    IMPORTANT DIFFERENCE FROM EVERY OTHER CHECK IN THIS SCRIPT: Qdrant/
    OpenSearch/Postgres/the LLM engine are all real STANDING services --
    either already running somewhere or not, independent of this script.
    MCP (D-30: stdio transport for local servers) is NOT a standing
    service at all -- cli.py spawns a fresh subprocess via MCPBridge on
    first real use, per CLI invocation, and it's torn down when that run
    ends. There is nothing persistent to "check on" the way you'd check
    a database is up. So this function does the only thing that's
    actually meaningful: spawns its OWN throwaway instance of the real
    configured subprocess (MCP_SERVER_COMMAND/MCP_SERVER_ARGS) via
    MCPBridge -- the exact same class cli.py uses -- calls the
    configured tool once with a throwaway query, then closes it. A PASS
    here means "the spawn-and-call mechanism works right now, on this
    machine, with this config" -- NOT "there is an MCP server currently
    running that a real cli.py invocation would reuse." There is no such
    persistent server to find, by design (see DECISIONS.md D-30's note
    on the not-yet-built Streamable HTTP variant, which WOULD be a real
    standing service if it existed).
    """
    if not settings.mcp_enabled:
        return ServiceStatus("MCP server", True, "MCP_ENABLED=false -- skipped, not checked",
                              skipped=True)

    if not settings.mcp_server_command:
        return ServiceStatus("MCP server", False,
                              "MCP_ENABLED=true but MCP_SERVER_COMMAND is empty -- "
                              "misconfigured, not a live-service failure")

    from research_agent.tools.mcp_client import MCPBridge

    t0 = time.time()
    bridge = MCPBridge(
        command=settings.mcp_server_command,
        args=split_csv(settings.mcp_server_args),
        env_allowlist=split_csv(settings.mcp_server_env_allowlist),
    )
    try:
        result = bridge.call_tool(
            settings.mcp_tool_name,
            {settings.mcp_query_arg_name: "health check"},
            timeout_seconds=settings.mcp_call_timeout_seconds)
        item_count = len(getattr(result, "content", []) or [])
        return ServiceStatus(
            "MCP server", True,
            f"{settings.mcp_server_command} {settings.mcp_server_args} -- "
            f"tool '{settings.mcp_tool_name}' responded, {item_count} content item(s) "
            "(spawned fresh for this check -- MCP has no persistent server; "
            "this confirms the on-demand subprocess+stdio path works, not that "
            "a server is 'up' the way Qdrant/OpenSearch/Postgres/the LLM are)",
            time.time() - t0)
    except Exception as e:
        return ServiceStatus(
            "MCP server", False,
            f"{settings.mcp_server_command} {settings.mcp_server_args} -- "
            f"{type(e).__name__}: {e} "
            "(this is a REAL failure -- the on-demand spawn mechanism itself is "
            "broken, e.g. a bad command/path or a crash in mcp_corpus_server.py; "
            "it is NOT a 'no server running' situation, since none is ever "
            "supposed to be running independently -- see this function's own "
            "docstring)",
            time.time() - t0)
    finally:
        bridge.close()


def check_api_server(base_url: str) -> ServiceStatus:
    """GET /health -- api/server.py's own liveness endpoint. Optional
    component: this codebase runs perfectly well via the CLI alone, so a
    FAIL here just means "the FastAPI server isn't running right now",
    not that anything is broken -- pass --skip-api if you never run it."""
    t0 = time.time()
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
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
        check_mcp(settings),
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