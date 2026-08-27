"""
tools/mcp_client.py — MCP-mediated tool, alongside tools/corpus_search.py.

Purpose:
    Implements D-26 (tool mediation via MCP) as a SECOND ToolFn
    implementation (agents/gathering.py::ToolFn), proving the seam
    corpus_search.py's own docstring has claimed since the core build
    shipped: "the graph-level tool-calling pattern is identical, so
    upgrading the plumbing to MCP later touches only this module."
    agents/gathering.py and orchestration/graph.py needed ZERO changes to
    support this -- see cli.py for the one wiring line that chooses
    between the two.

Responsibilities:
    - MCPBridge: owns one persistent background event loop + one
      persistent Streamable-HTTP-connected ClientSession, for the
      process's lifetime. Exists because every OTHER node in this
      codebase is a plain synchronous function (confirmed: no
      `async def` anywhere in agents/*.py), but the MCP SDK is
      async-only -- something has to bridge the two without converting
      the whole graph to async.
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
    Reconnecting per call would add real network round-trip cost on every
    single tool invocation, and asyncio.run() tears down its event loop
    when it returns, which would invalidate a ClientSession/transport tied
    to that loop. So the connection and the ClientSession are established
    ONCE (lazily, on first use) and kept alive on a dedicated background
    thread running its own event loop for as long as the process lives;
    every synchronous ToolFn call submits its coroutine onto THAT SAME
    loop via asyncio.run_coroutine_threadsafe(...), which is safe to call
    from any thread (including however LangGraph happens to execute
    concurrent search_worker instances for one gather-cycle superstep) --
    asyncio objects themselves are only ever touched from the one thread
    that owns their loop.

Transport (D-30, D-75, D-76): Streamable HTTP only.
    D-30 originally scoped this module to stdio (a local server this
    process spawns and owns) and documented Streamable HTTP as a
    deferred remote-server variant. D-75 built the HTTP variant out
    alongside stdio, config-selectable. D-76 removed stdio entirely: this
    codebase now ALWAYS connects to an independent, already-running MCP
    server over HTTP -- nothing is spawned, ever, by this process. See
    the module's own history in DECISIONS.md (D-30, D-75, D-76) for why
    the shape changed twice. SSE is not, and has never been, a supported
    value -- D-30 prohibited it outright from the start.

    Why HTTP-only rather than a config choice: a config flag that can
    still select stdio invites exactly the coupling D-76 removed it to
    avoid -- "start it once, leave it running, point one or more runs at
    it, stop it whenever you want" is not compatible with a mode where
    this process might instead spawn and own a subprocess. One transport,
    always independent, is the simpler invariant to reason about and the
    one actually wanted operationally.
"""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional

from research_agent.logging_setup import log_event
from research_agent.state import Evidence, SearchTask, Volatility

logger = logging.getLogger(__name__)


class MCPBridge:
    """Owns one persistent background event loop + one persistent
    Streamable-HTTP-connected ClientSession, for the process's lifetime.
    A pure network client -- nothing is ever spawned by this process
    (D-76); the server this connects to is independent, already running,
    and entirely unaffected by close() below.

    CALLED BY   make_mcp_tool, below, which wraps this in a plain
                synchronous ToolFn closure -- nothing outside this module
                (and make_mcp_tool) ever touches an MCPBridge directly.
    LIFECYCLE   Nothing happens at construction time -- the background
                thread, event loop, connection, and ClientSession are all
                created lazily, on the FIRST call_tool() call (see
                start()). Call close() when done (cli.py does this at
                shutdown, mirroring how storage/postgres.py's checkpointer
                is explicitly closed); an MCPBridge that's never started
                has nothing to close. close() only ever closes THIS
                process's own connection -- the server itself is a
                separate, independent process, started and stopped by
                you, not by anything in this codebase.
    """

    def __init__(self, url: str, startup_timeout_seconds: float = 30.0):
        """
        Parameters:
            url: the standalone MCP server's HTTP endpoint, e.g.
                "http://127.0.0.1:8765/mcp". Required -- validated eagerly
                at construction, not deferred to the first call_tool(): a
                config with no URL should fail loudly at startup, not
                three tool calls into a run.
            startup_timeout_seconds: how long to wait for the initial
                connection handshake before raising TimeoutError.
        """
        if not url:
            raise ValueError("MCPBridge requires a url (D-76: standalone "
                             "Streamable HTTP server only)")
        self._url = url
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
                f"MCP server '{self._url}' did not become ready within "
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
        test caught the bug): the MCP SDK's client/ClientSession context
        managers are built on anyio cancel scopes, which anyio requires to
        be exited from the EXACT SAME asyncio Task they were entered in --
        not just the same event loop. Splitting connect and disconnect
        into two separately-submitted coroutines put them in two different
        Tasks on the same loop, which LOOKS fine (same loop, same thread)
        but anyio itself raises `RuntimeError: Attempted to exit cancel
        scope in a different task than it was entered in`. Keeping the
        whole connect -> wait -> disconnect sequence in one coroutine,
        submitted once via run_until_complete, is what actually satisfies
        that constraint. Concurrent call_tool() invocations are unaffected
        -- each is its own SEPARATE coroutine that only calls
        self._session.call_tool(...), never touches the exit stack's
        cancel scopes, and is safe to run in a different task than
        _serve()'s (confirmed empirically: this is exactly what already
        happens on every call_tool() call, submitted independently of
        this method, and it works correctly).
        """
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        # streamable_http_client yields a THIRD value (a get-session-id
        # callback) that ClientSession itself has no use for; only the
        # read/write streams matter here. Named `streamable_http_client`,
        # not the older `streamablehttp_client` (still present in the
        # installed SDK but emits a DeprecationWarning -- confirmed
        # against the actual installed mcp 1.29.0, not assumed from docs).
        from mcp.client.streamable_http import streamable_http_client

        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream, _get_session_id = (
                await self._exit_stack.enter_async_context(
                    streamable_http_client(self._url)))
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
        server's FIRST call after the server process starts -- that call
        is what lazily builds the real QdrantStore/OpenSearchStore/
        HybridRetriever INSIDE the server process (see
        scripts/mcp_corpus_server.py::_get_corpus_tool), which can itself
        take real time (fastembed's embedding model load, real network
        round trips to Qdrant/OpenSearch) well beyond what a trivial echo
        server needs. Several search_worker calls arriving concurrently
        against the SAME persistent connection can also serialize behind
        one another depending on the server's own concurrency model,
        compounding the effective wait for whichever calls land later in
        that queue. If you see TimeoutError failures against a real
        corpus-backed server, raise MCP_CALL_TIMEOUT_SECONDS well above
        the 30s default (e.g. 120) before assuming something is broken.
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

        # Read self._loop ONCE into a local: close() (running on another
        # thread) sets it to None, and run_coroutine_threadsafe(coro, None)
        # raises an opaque AttributeError instead of a usable message.
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                f"MCP bridge for '{self._url}' is closed; cannot call "
                f"tool {name!r}")
        future = asyncio.run_coroutine_threadsafe(_traced_call(), loop)
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

    def list_tools(self, timeout_seconds: float = 30.0) -> List[str]:
        """Ask the server which tools it actually exposes. Returns names.

        D-89. Until this existed, a configured tool name that the server
        does not offer -- a typo in MCP_TOOL_NAME, a server upgraded to a
        new tool name, the corpus URL pointed at the web-search server by
        mistake -- surfaced only as a per-TASK failure, once retrieval was
        already underway, as whatever error the server chose to return.
        The tool surface was never observable at all: a server offering
        three tools where this build binds one looked identical to a
        server offering exactly the one.

        CALLED BY   scripts/check_services.py, which reports the real list
                    and flags a configured name that is missing from it.
        NOT called from assembly.py, deliberately: D-76 makes the server an
        independent process that may legitimately be started AFTER the
        agent, so making startup depend on it being reachable would trade
        one failure mode for a worse one. Discovery belongs in the health
        check, which is the thing you run when you want to know.

        Same transport mechanics as call_tool above -- start() first, then
        submit onto the bridge's own loop and block this thread for the
        result. Raises whatever the session raises; the caller decides how
        to report it.
        """
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                f"MCP bridge for '{self._url}' is closed; cannot list tools")
        future = asyncio.run_coroutine_threadsafe(
            self._session.list_tools(), loop)
        started_at = time.time()
        try:
            result = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            # Same reasoning as call_tool's own handler: a bare
            # concurrent.futures.TimeoutError carries an EMPTY message, so
            # re-raise with one that says what timed out and for how long.
            elapsed = time.time() - started_at
            raise TimeoutError(
                f"MCP list_tools against {self._url!r} timed out after "
                f"{elapsed:.1f}s (limit={timeout_seconds}s)") from None
        return [getattr(t, "name", "") for t in getattr(result, "tools", None) or []]

    def close(self) -> None:
        """Tear down this process's session and background loop/thread.
        The independent server itself is never touched (D-76).

        CALLED BY   cli.py, once, at process shutdown -- mirrors how
                    storage/postgres.py's checkpointer is explicitly
                    closed there. Safe to call on a bridge that was never
                    started (no-op) or twice (second call also a no-op).

        Signals self._shutdown_event rather than directly closing the
        exit stack from here -- see _serve()'s docstring for exactly why
        that distinction matters (an earlier version of this method did
        close the exit stack directly, from a separately-submitted
        coroutine, and a real end-to-end test caught the anyio
        cancel-scope violation that caused).
        Setting the event via call_soon_threadsafe is what actually lets
        _serve()'s own task (already waiting on that event) proceed to
        run its own `await self._exit_stack.aclose()` -- in the SAME task
        it connected in, satisfying anyio's constraint.
        """
        if self._loop is None or self._thread is None:
            return
        if self._shutdown_event is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            # Best-effort cleanup only -- the thread is a daemon thread (see
            # start()), so it will not block process exit even if it never
            # finishes; this just means this connection's own teardown
            # didn't complete cleanly within the timeout. The SERVER is
            # unaffected either way (D-76) -- this is purely about this
            # process's own background thread.
            log_event(logger, "mcp.close_timed_out", level=logging.WARNING,
                      url=self._url)
        self._loop = None
        self._thread = None
        self._shutdown_event = None
        # Reset the readiness handshake too, not just the thread/loop
        # handles. Leaving _ready SET and _session STALE meant a bridge
        # that was closed and then started again would sail past
        # start()'s _ready.wait() instantly and call into a dead session
        # -- the same class of race start()'s own docstring documents
        # fixing for concurrent first calls, but on the reuse path.
        self._ready.clear()
        self._session = None
        self._exit_stack = None
        self._start_error = None


def make_mcp_tool(bridge: MCPBridge, tool_name: str,
                  query_arg_name: str = "query",
                  call_timeout_seconds: float = 30.0,
                  unscored_score: float = 0.0):
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
        unscored_score: the Evidence.score stamped on every item this
            server returns, since this build's minimal MCP schema carries
            no per-hit ranking score. cli.py passes
            settings.min_evidence_score. This MUST NOT be raised above
            that threshold: agents/gathering.py::progress_checker_node
            gates coverage on `e.score > settings.min_evidence_score`, so
            a fabricated high score marks every touched goal covered on
            the first cycle, forcing recall=1.0 and making the gap loop
            and the E2/E3 escalations structurally unreachable. That is
            exactly the defect P2-01 fixed for the corpus path
            (MIN_EVIDENCE_SCORE=0.0 made the same predicate inert); this
            parameter exists so the MCP path cannot reintroduce it. A
            server that DOES report real relevance should populate this
            per item from structuredContent instead.

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
                # Was a hardcoded 1.0, which cleared ANY min_evidence_score
                # and silently defeated the coverage gate -- see
                # unscored_score in this factory's docstring.
                score=unscored_score,
                volatility=Volatility.SEMI_STABLE,
            ))
        return evidence

    return mcp_search


# ---------------------------------------------------------------------------
# Phase 4 (D-57): the web-search tool
# ---------------------------------------------------------------------------


def make_web_search_tool(
    bridge: MCPBridge,
    tool_name: str = "web_search",
    query_arg_name: str = "query",
    call_timeout_seconds: float = 45.0,
) -> Any:
    """Build the retrieval tool for the web-search MCP server.

    CALLED BY   assembly.py::build_app_and_settings, once, when
                settings.web_search_enabled is true.
    RETURNS     a plain function with corpus_search's signature --
                (SearchTask) -> List[Evidence] -- so the retrieval ladder
                treats it identically to every other tier.

    WHY THIS IS A SEPARATE FACTORY RATHER THAN A `source=` PARAMETER ON
    make_mcp_tool ABOVE, which was the obvious alternative:

      1. The two servers have genuinely DIFFERENT response schemas.
         make_mcp_tool reads plain text content blocks, one Evidence per
         block, and stamps a single flat score (unscored_score) because a
         corpus server's text blocks carry no per-item score. This one reads
         structuredContent, where every item carries its OWN score, url and
         domain. Bending one function to do both would mean a runtime branch
         on response shape inside the tool every search_worker calls.
      2. make_mcp_tool is the proven Phase 1-3 path. Adding parameters to it
         means every existing MCP run inherits whatever this Phase 4 work
         gets wrong. Leaving it byte-identical means the corpus path cannot
         regress from this change at all -- which is worth more than the
         handful of shared lines a merged implementation would save.

    Both factories still share the ENTIRE transport: one MCPBridge class,
    one connection lifecycle, one timeout mechanism. What differs is only
    the parsing, which is exactly the part that genuinely differs.

    THE WIRE SHAPE THIS PARSES, verified against the installed FastMCP
    rather than assumed (see tests/unit/test_mcp_web_search_server.py::
    test_a_list_dict_tool_puts_results_under_structured_content_result,
    which spawns a real server and asserts it):

        result.structuredContent == {"result": [ {...}, {...} ]}

    each item carrying title, url, snippet, rank, engine, domain, score --
    the shape websearch/provider.py::as_payload defines. structuredContent
    is read in preference to the text blocks because it is the only channel
    where `score` survives as a NUMBER; the text blocks carry each item as
    JSON text, which is a usable fallback and is handled below, but a value
    re-parsed out of prettified JSON is a worse contract than one that was
    never stringified.

    RAISES whatever bridge.call_tool raises. Deliberately not caught, the
    same contract corpus_search and mcp_search hold: agents/gathering.py::
    search_worker owns turning an exception into a D-16 failure record.
    """

    def _items_from_result(result: Any) -> List[Dict[str, Any]]:
        """Pull the result list out of whichever channel carried it.

        Preference order is structuredContent, then JSON-decoded text
        blocks. A server that sends neither yields [] -- which reads as
        "found nothing" and lets the ladder escalate, rather than raising
        and burning the task as a D-16 failure over a shape problem.
        """
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            items = structured.get("result")
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]

        # Fallback: one JSON-encoded item per text block. Reached when an
        # SDK version stops populating structuredContent, or when a
        # differently-built server returns text only. Silently tolerating a
        # block that is not JSON matters here -- a server that also emits a
        # human-readable preamble block should not break the whole call.
        out: List[Dict[str, Any]] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except (ValueError, TypeError):
                continue
            if isinstance(decoded, dict):
                out.append(decoded)
            elif isinstance(decoded, list):
                out.extend(i for i in decoded if isinstance(i, dict))
        return out

    def web_search(task: SearchTask) -> List[Evidence]:
        """One web search for one SearchTask.

        CALLED BY   agents/gathering.py::search_worker, exactly once per
                    task, in that worker's own try/except.
        """
        result = bridge.call_tool(
            tool_name, {query_arg_name: task.query},
            timeout_seconds=call_timeout_seconds)

        if getattr(result, "isError", False):
            # Same distinction make_mcp_tool draws: a TOOL-level error is
            # data arriving cleanly to say "this did not work", not a
            # protocol failure. Treated as "nothing found" so the ladder
            # escalates to the model tier rather than failing the task.
            log_event(logger, "web_search.tool_reported_error", tool=tool_name)
            return []

        evidence: List[Evidence] = []
        skipped_unscored = 0
        for item in _items_from_result(result):
            snippet = (item.get("snippet") or "").strip()
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip() or None
            if not (snippet or title):
                continue

            # A missing or unparseable score is DROPPED, not defaulted.
            # There is no defensible default: too high silently defeats the
            # D-17 coverage gate (the hardcoded 1.0 this module already
            # shipped once and had to fix -- see unscored_score in
            # make_mcp_tool's docstring), and too low makes a genuinely
            # retrieved result unable to cover a goal while still consuming
            # a slot in the compile prompt. Losing an item is the smaller
            # and, crucially, the VISIBLE failure: it is counted and logged
            # below rather than quietly mis-weighted.
            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                skipped_unscored += 1
                continue
            # Clamp rather than reject: Evidence.score is bounded [0,1] by
            # its own Field constraint, and a server returning 1.0000001
            # through float round-tripping should not raise a
            # ValidationError inside a worker.
            score = min(max(score, 0.0), 1.0)

            # Title and snippet joined, because each alone loses something:
            # the snippet is the substance, the title is often the only
            # place the actual subject is named. Same 800-char cap
            # corpus_search.py and make_mcp_tool use, so no one tier can
            # crowd the compile prompt.
            content = f"{title} — {snippet}".strip(" —") if title else snippet

            evidence.append(Evidence(
                task_key=task.key,
                goal_id=task.goal_id,
                # NOT "mcp", even though this arrived over MCP. "mcp" is
                # tested for set-membership in agents/gathering.py::
                # progress_checker_node and agents/compilation.py::
                # telemetry_node as a proxy for "a real DOCUMENT backed
                # this" -- see state.py::Evidence's docstring. Tagging web
                # results "mcp" would make every snippet count toward
                # grounded_score and corpus_recall, restoring exactly the
                # blindness D-43 and D-47 exist to expose.
                source="web",
                content=content[:800],
                score=score,
                # Web content is volatile by nature -- a live page today is
                # a stale answer next month, with nothing in the text
                # saying so. This also drives memory decay, though
                # store_run excludes source="web" outright (D-57), so the
                # value matters for D-51's hedging pass rather than for
                # long-term storage.
                volatility=Volatility.VOLATILE,
                url=url,
                domain=(item.get("domain") or "").strip() or None,
            ))

        if skipped_unscored:
            log_event(logger, "web_search.dropped_unscored_items",
                      level=logging.WARNING, count=skipped_unscored,
                      tool=tool_name,
                      reason="server returned an item with no usable 'score'; "
                             "no default is safe, so the item was dropped")
        log_event(logger, "web_search.evidence_built",
                  count=len(evidence), task_key=task.key,
                  domains=len({e.domain for e in evidence if e.domain}))
        return evidence

    return web_search
