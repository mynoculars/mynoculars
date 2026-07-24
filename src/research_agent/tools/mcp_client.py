"""
tools/mcp_client.py — MCP-mediated tool, alongside tools/corpus_search.py.

Purpose:
    Implements D-26 (tool mediation via MCP) and D-30 (the specific
    transport/security constraints for that mediation) as a SECOND ToolFn
    implementation (agents/gathering.py::ToolFn), proving the seam
    corpus_search.py's own docstring has claimed since the core build
    shipped: "the graph-level tool-calling pattern is identical, so
    upgrading the plumbing to MCP later touches only this module."
    agents/gathering.py and orchestration/graph.py needed ZERO changes to
    support this -- see cli.py for the one wiring line that chooses
    between the two.

Responsibilities:
    - MCPBridge: owns one persistent background event loop + one
      persistent stdio-connected ClientSession, for the process's
      lifetime. Exists because every OTHER node in this codebase is a
      plain synchronous function (confirmed: no `async def` anywhere in
      agents/*.py), but the MCP SDK is async-only -- something has to
      bridge the two without converting the whole graph to async.
    - make_mcp_tool(): builds the actual ToolFn closure, matching
      corpus_search.make_corpus_tool()'s exact shape and calling
      convention so agents/gathering.py::search_worker cannot tell the
      difference between the two tools.

Design decision -- WHERE the async/sync bridge lives (the P2-13 risk
flagged before writing any code here):
    Two options existed: (1) bridge inside this module, keeping
    search_worker and the whole graph synchronous, or (2) convert
    search_worker (and callers) to async so LangGraph's async node support
    could be used directly. (1) was chosen -- smaller, more contained
    change, consistent with "preserve existing seams" and "don't rewrite
    when incremental suffices". Every other file in this codebase is
    UNCHANGED by this file's existence.

Design decision -- A PERSISTENT background loop, not asyncio.run() per
call:
    A stdio-connected MCP server is a real subprocess with real startup
    cost; spawning a fresh one per tool call would be wasteful, and
    asyncio.run() tears down its event loop when it returns, which would
    invalidate a ClientSession/transport tied to that loop. So the
    subprocess and the ClientSession are established ONCE (lazily, on
    first use) and kept alive on a dedicated background thread running
    its own event loop for as long as the process lives; every synchronous
    ToolFn call submits its coroutine onto THAT SAME loop via
    asyncio.run_coroutine_threadsafe(...), which is safe to call from any
    thread (including however LangGraph happens to execute concurrent
    search_worker instances for one gather-cycle superstep) -- asyncio
    objects themselves are only ever touched from the one thread that
    owns their loop.

D-30 constraints, implemented exactly as specified (not re-litigated
here):
    - stdio transport only, in this build -- Streamable HTTP is D-30's
      other allowed transport for a REMOTE server, but is not implemented
      here; adding it later is a new MCPBridge variant, not a change to
      make_mcp_tool's contract. SSE is explicitly PROHIBITED by D-30 and
      never used anywhere in this module.
    - Explicit per-server env ALLOWLIST, never a blanket os.environ
      passthrough -- confirmed necessary: mcp.StdioServerParameters's own
      `env` field defaults to None, which mcp's stdio_client interprets as
      "inherit the parent's FULL environment" if left unset. This module
      NEVER leaves it unset; _build_subprocess_env always constructs an
      explicit dict from only the allowlisted names.
    - AsyncExitStack for stdio server lifecycle, so the subprocess and its
      streams are guaranteed to be torn down together, in the right order,
      even if something raises partway through connecting.
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional

from research_agent.logging_setup import log_event
from research_agent.state import Evidence, SearchTask, Volatility

logger = logging.getLogger(__name__)


def _build_subprocess_env(env_allowlist: List[str]) -> Dict[str, str]:
    """Build the subprocess env dict from an explicit allowlist of names.

    CALLED BY   MCPBridge._connect, below.
    WHY THIS EXISTS (D-30): mcp.StdioServerParameters's own `env` field
    defaults to None, and mcp's stdio_client treats that as "inherit the
    parent process's FULL environment" -- which could leak this process's
    entire environment (API keys, DB credentials, everything in .env)
    into a third-party MCP server subprocess. This function is the ONLY
    place that ever builds the `env` dict passed to StdioServerParameters
    in this module, and it only ever includes names explicitly listed in
    `env_allowlist` that are ALSO actually set in this process's own
    environment -- an allowlisted name that happens not to be set is
    simply absent from the result, never an error.
    """
    return {name: os.environ[name] for name in env_allowlist if name in os.environ}


class MCPBridge:
    """Owns one persistent background event loop + one persistent
    stdio-connected ClientSession, for the process's lifetime.

    CALLED BY   make_mcp_tool, below, which wraps this in a plain
                synchronous ToolFn closure -- nothing outside this module
                (and make_mcp_tool) ever touches an MCPBridge directly.
    LIFECYCLE   Nothing happens at construction time -- the background
                thread, event loop, subprocess, and ClientSession are all
                created lazily, on the FIRST call_tool() call (see
                start()). Call close() when done (cli.py does this at
                shutdown, mirroring how storage/postgres.py's checkpointer
                is explicitly closed); an MCPBridge that's never started
                has nothing to close.
    """

    def __init__(self, command: str, args: Optional[List[str]] = None,
                 env_allowlist: Optional[List[str]] = None,
                 startup_timeout_seconds: float = 30.0):
        self._command = command
        self._args = args or []
        self._env_allowlist = env_allowlist or []
        self._startup_timeout = startup_timeout_seconds
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None  # type: Optional["mcp.ClientSession"]
        self._exit_stack = None  # type: Optional["contextlib.AsyncExitStack"]
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._shutdown_event = None  # type: Optional[asyncio.Event]
                                      # created inside _serve() itself (an
                                      # asyncio.Event must be created on
                                      # the loop it will be awaited on)

    def start(self) -> None:
        """Start the background loop + connect, if not already started;
        wait for readiness EVERY call, regardless of who created it.

        FIXED BUG (found via a real concurrent run, not a test): the
        original version guarded THREAD CREATION and the READINESS WAIT
        with the same `if self._thread is not None: return` check inside
        the lock -- meaning only the ONE caller that actually created the
        background thread ever waited for it to become ready. Every OTHER
        thread racing in at nearly the same time (exactly what happens
        when LangGraph fans multiple search_worker instances out
        concurrently for one gather-cycle superstep) saw
        `self._thread is not None` already true and returned IMMEDIATELY,
        then proceeded straight to call_tool()'s
        `self._session.call_tool(...)` while `self._session` was still
        None (the background thread was still mid-connect) --
        `AttributeError: 'NoneType' object has no attribute 'call_tool'`.
        A real run with 6 concurrent search_worker calls showed exactly
        this: 5 failed with AttributeError within milliseconds, while the
        1 that actually created the thread waited correctly (and, in that
        run, eventually hit a separate, legitimate timeout -- see
        call_tool's own docstring note on cold-start latency against a
        real corpus-backed server).

        THE FIX: only THREAD CREATION is guarded by the lock/already-
        started check below; the readiness wait itself now runs
        UNCONDITIONALLY, for every single call to start(), from every
        thread. self._ready is a threading.Event -- once set, .wait()
        returns immediately for any caller, including ones that arrive
        long after the thread was created, so this costs nothing extra
        for the common case (already-ready bridge, every later call's
        wait returns instantly) while fixing the race for calls that
        arrive DURING startup.
        """
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run_loop, name="mcp-bridge", daemon=True)
                self._thread.start()
        if not self._ready.wait(timeout=self._startup_timeout):
            raise TimeoutError(
                f"MCP server '{self._command}' did not become ready within "
                f"{self._startup_timeout}s")
        if self._start_error is not None:
            raise self._start_error

    def _run_loop(self) -> None:
        """Entry point for the background thread: owns the event loop for
        as long as this MCPBridge lives. Everything that touches
        self._session happens on THIS thread -- individual call_tool()
        invocations submit their OWN separate coroutines here via
        run_coroutine_threadsafe (fine, proven safe below), but connect
        and disconnect deliberately do NOT: see _serve()'s docstring for
        why they must share a single task, not just a single loop.

        The `finally: self._loop.close()` below matters -- an earlier
        version omitted it, and a real test run surfaced
        `ResourceWarning: unclosed event loop` (pytest.ini's
        filterwarnings=always is what made it visible at all -- see that
        file's own history). run_until_complete() returning (however it
        returns -- normally, or via the except above) does NOT release
        the loop's own resources by itself; closing it explicitly, on
        THIS thread, right before the thread function ends, is what
        actually does. Safe to close here specifically because close()
        (called from a DIFFERENT thread) only ever sets self._loop = None
        AFTER self._thread.join() returns -- i.e. after this entire
        function has already finished -- so there is no race between the
        two.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 -- surfaced to start()
            self._start_error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        """Connect, then BLOCK (inside this same coroutine/task) until
        close() signals shutdown, then disconnect -- all in one task.

        WHY THIS SHAPE, not "connect in one run_until_complete call, then
        run_forever(), then disconnect via a SEPARATE run_coroutine_threadsafe
        call" (the first version of this method, before a real end-to-end
        test against tests/fixtures/mcp_echo_server.py caught the bug):
        the MCP SDK's stdio_client/ClientSession context managers are built
        on anyio cancel scopes, which anyio requires to be exited from the
        EXACT SAME asyncio Task they were entered in -- not just the same
        event loop. Splitting connect and disconnect into two separately-
        submitted coroutines put them in two different Tasks on the same
        loop, which LOOKS fine (same loop, same thread) but anyio itself
        raises `RuntimeError: Attempted to exit cancel scope in a
        different task than it was entered in`. Keeping the whole
        connect -> wait -> disconnect sequence in one coroutine, submitted
        once via run_until_complete, is what actually satisfies that
        constraint. Concurrent call_tool() invocations are unaffected --
        each is its own SEPARATE coroutine that only calls
        self._session.call_tool(...), never touches the exit stack's
        cancel scopes, and is safe to run in a different task than
        _serve()'s (confirmed empirically: this is exactly what already
        happens on every call_tool() call, submitted independently of
        this method, and it works correctly).
        """
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._exit_stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self._command, args=self._args,
                env=_build_subprocess_env(self._env_allowlist))
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params))
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream))
            await self._session.initialize()
        except BaseException as exc:  # noqa: BLE001 -- surfaced to start()
            self._start_error = exc
            self._ready.set()
            return

        self._shutdown_event = asyncio.Event()
        self._ready.set()
        await self._shutdown_event.wait()  # blocks here until close() signals
        await self._exit_stack.aclose()  # SAME task as the enter_async_context
                                          # calls above -- this is the fix.

    def call_tool(self, name: str, arguments: Dict[str, Any],
                 timeout_seconds: float = 30.0):
        """Call an MCP tool by name, blocking THIS (caller's) thread until
        the result comes back or `timeout_seconds` elapses.

        CALLED BY   the ToolFn closure make_mcp_tool builds, below --
                    exactly once per SearchTask, same calling convention
                    as tools/corpus_search.py's corpus_search().
        RETURNS     mcp.types.CallToolResult -- the caller (make_mcp_tool)
                    is responsible for turning that into Evidence objects;
                    this method knows nothing about this codebase's domain
                    model, only about talking to the MCP server.

        May raise (McpError for a protocol-level failure, TimeoutError if
        the call doesn't complete in time, or whatever start() itself
        raised if the server never became ready) -- deliberately NOT
        caught here. Matches tools/corpus_search.py's own convention:
        agents/gathering.py::search_worker owns turning a tool exception
        into a D-16 "failed" record, not the tool itself.

        OPERATIONAL NOTE (found via a real run against
        scripts/mcp_corpus_server.py, not a bug in this method): the
        DEFAULT timeout_seconds=30.0 (and settings.mcp_call_timeout_seconds's
        own 30.0 default) can be too tight for a real corpus-backed MCP
        server's FIRST call after the subprocess starts -- that call is
        what lazily builds the real QdrantStore/OpenSearchStore/
        HybridRetriever INSIDE the server process (see
        scripts/mcp_corpus_server.py::_get_corpus_tool), which can itself
        take real time (fastembed's embedding model load, real network
        round trips to Qdrant/OpenSearch) well beyond what a trivial echo
        server needs. Several search_worker calls arriving concurrently
        against the SAME persistent stdio session can also serialize
        behind one another (one stdio pipe, processed one request at a
        time by a typical MCP server), compounding the effective wait for
        whichever calls land later in that queue. If you see TimeoutError
        failures against a real corpus-backed server, raise
        MCP_CALL_TIMEOUT_SECONDS well above the 30s default (e.g. 120)
        before assuming something is broken.
        """
        # error=str(exc)[:300] follow-up: bare concurrent.futures.TimeoutError
        # has an EMPTY message (confirmed: str(TimeoutError()) == "") -- a
        # real failure showed up in this codebase's own D-16 failure log as
        # "reason=TimeoutError" with NOTHING else to go on: not which tool,
        # not what was being asked, not how long it actually waited. That
        # was correctly called out as unacceptable error reporting. This
        # method now builds a real, specific message BEFORE re-raising, and
        # logs it immediately too (not just relying on whatever the caller
        # eventually does with the exception) -- so a --debug trace shows
        # the actual diagnostic even if some future caller ever changes how
        # it logs failures.
        self.start()

        # DIAGNOSTIC INSTRUMENTATION (found necessary via a real, otherwise
        # unexplained failure: a single, non-concurrent call against a real
        # server timed out with ZERO server-side output at all -- ruling
        # out both retrieval-stack speed and concurrency as causes, and
        # leaving open whether the request ever actually got scheduled to
        # run on THIS bridge's own event loop in the first place, versus
        # being scheduled but never getting a response). _traced_call
        # wraps the real call_tool coroutine and logs the instant it
        # actually STARTS EXECUTING on the loop (not just the instant it
        # was submitted) and the instant it finishes (success or
        # exception) -- these log_events fire on the bridge's OWN
        # background thread, so if "mcp.call_tool_task_started" never
        # appears at all, the coroutine never even began running (a
        # scheduling/loop problem, not a transport/server problem); if it
        # appears but "mcp.call_tool_task_finished" never does, the
        # request was sent but no response ever arrived (points at the
        # transport or the server's own request routing instead).
        async def _traced_call():
            log_event(logger, "mcp.call_tool_task_started", tool=name)
            try:
                result = await self._session.call_tool(name, arguments)
                log_event(logger, "mcp.call_tool_task_finished", tool=name, outcome="success")
                return result
            except BaseException as exc:
                log_event(logger, "mcp.call_tool_task_finished", tool=name,
                          outcome="exception", reason=type(exc).__name__)
                raise

        future = asyncio.run_coroutine_threadsafe(_traced_call(), self._loop)
        started_at = time.time()
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            elapsed = time.time() - started_at
            # args_preview: show enough of the actual arguments to be
            # useful (which query, which task) without risking an
            # unbounded string if some future caller passes something huge.
            args_preview = str(arguments)[:200]
            message = (
                f"MCP call_tool(name={name!r}, arguments={args_preview}) "
                f"timed out after {elapsed:.1f}s (limit={timeout_seconds}s). "
                f"This does NOT necessarily mean the server is broken -- see "
                f"call_tool's own docstring for how to isolate whether the "
                f"bottleneck is the underlying tool's real work (time it "
                f"directly, bypassing MCP) or the MCP transport/session "
                f"layer itself (e.g. how many requests are actually "
                f"in-flight to this same bridge concurrently right now).")
            log_event(logger, "mcp.call_tool_timed_out", level=logging.WARNING,
                      tool=name, elapsed_s=round(elapsed, 1),
                      timeout_s=timeout_seconds, arguments=args_preview)
            raise TimeoutError(message) from None

    def close(self) -> None:
        """Tear down the session, subprocess, and background loop/thread.

        CALLED BY   cli.py, once, at process shutdown -- mirrors how
                    storage/postgres.py's checkpointer is explicitly
                    closed there. Safe to call on a bridge that was never
                    started (no-op) or twice (second call also a no-op).

        Signals self._shutdown_event rather than directly closing the
        exit stack from here -- see _serve()'s docstring for exactly why
        that distinction matters (an earlier version of this method did
        close the exit stack directly, from a separately-submitted
        coroutine, and a real end-to-end test against a live stdio
        server caught the anyio cancel-scope violation that caused).
        Setting the event via call_soon_threadsafe is what actually lets
        _serve()'s own task (already waiting on that event) proceed to
        run its own `await self._exit_stack.aclose()` -- in the SAME task
        it connected in, satisfying anyio's constraint.
        """
        if self._loop is None or self._thread is None:
            return
        if self._shutdown_event is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
        joined_in_time = True
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            joined_in_time = False
            log_event(logger, "mcp.close_timed_out", level=logging.WARNING,
                      command=self._command)
        self._loop = None
        self._thread = None
        self._shutdown_event = None
        if not joined_in_time:
            # Best-effort cleanup only -- the thread is a daemon thread
            # (see start()), so it will not block process exit even if
            # it never finishes; this just means the subprocess/session
            # teardown didn't complete cleanly within the timeout.
            return


def make_mcp_tool(bridge: MCPBridge, tool_name: str,
                  query_arg_name: str = "query",
                  call_timeout_seconds: float = 30.0):
    """Build the MCP-mediated tool bound to a bridge and a specific
    server-side tool name.

    Mirrors tools/corpus_search.py::make_corpus_tool's exact closure
    pattern and calling convention -- see that module's docstring for the
    general shape every build_*_node/*_tool function in this codebase
    shares. cli.py picks EITHER this or make_corpus_tool's result to pass
    into build_graph as the one `tool: ToolFn` argument; nothing else in
    the graph knows or cares which.

    Parameters:
        bridge: an MCPBridge, not yet necessarily started (start() is
            called lazily on first use, inside bridge.call_tool()).
        tool_name: the MCP server's own tool name to invoke (servers can
            expose more than one tool; this build calls exactly one,
            named by settings.mcp_tool_name).
        query_arg_name: the argument name the server's tool schema expects
            for the search string -- not every server will call it
            "query", so this is configurable rather than hardcoded.
        call_timeout_seconds: passed straight through to
            bridge.call_tool.

    Returns:
        callable(task: SearchTask) -> List[Evidence]. May raise -- same
        contract as make_corpus_tool's returned function; the worker owns
        turning a raised exception into a D-16 failure record.
    """

    def mcp_search(task: SearchTask) -> List[Evidence]:
        """The actual tool function every search_worker invocation calls,
        when settings.mcp_enabled has wired THIS tool in at cli.py
        instead of corpus_search.py's.

        CALLED BY   agents/gathering.py::search_worker -- exactly once
                    per SearchTask, in that worker's own try/except (see
                    tools/corpus_search.py's corpus_search for the
                    identical contract: this function may raise, and
                    does not catch anything itself).
        CALLS       bridge.call_tool(tool_name, {query_arg_name: task.query})
                    -- one round trip to the MCP server.
        RETURNS     one Evidence object per text content block in the
                    server's response, each stamped with this task's
                    identity (task_key, goal_id) exactly like
                    corpus_search's own Evidence construction, tagged
                    source="mcp" so downstream nodes/telemetry can tell
                    MCP-sourced evidence apart from corpus/memory sourced
                    evidence if that distinction ever matters.
        """
        result = bridge.call_tool(
            tool_name, {query_arg_name: task.query},
            timeout_seconds=call_timeout_seconds)

        if getattr(result, "isError", False):
            # A TOOL-LEVEL failure (the server ran the tool, but the tool
            # itself reported an error) is NOT the same as a protocol
            # error (McpError, raised, not returned) -- this is data
            # coming back cleanly, just saying "no results here". Treated
            # as "nothing found", same as any other empty result, not as
            # a D-16 failure -- a tool that legitimately has nothing to
            # say for this query is a normal outcome, not a broken one.
            log_event(logger, "mcp.tool_reported_error", tool=tool_name)
            return []

        evidence: List[Evidence] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if not text:
                # Non-text content blocks (images, embedded resources) --
                # this build's Evidence model is text-only (content: str),
                # so anything without a .text attribute is simply skipped
                # rather than raising; a server that returns e.g. an image
                # alongside text still contributes whatever text it also
                # sent.
                continue
            evidence.append(Evidence(
                task_key=task.key,
                goal_id=task.goal_id,
                source="mcp",
                content=text[:800],  # same cap corpus_search.py uses
                score=1.0,  # MCP tools return no ranking score of their
                            # own in this build's minimal schema; a future
                            # server that DOES report one could populate
                            # this from structuredContent instead.
                volatility=Volatility.SEMI_STABLE,
            ))
        return evidence

    return mcp_search
