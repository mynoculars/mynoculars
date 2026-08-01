"""
assembly.py — the application's dependency graph, in one place.

Purpose:
    Construct every object the graph needs (LLM router, both retrieval
    stores, the memory wrapper, the checkpointer, the optional MCP
    bridge) from Settings, and hand them back as one AppBundle.

WHY THIS FILE EXISTS SEPARATELY FROM cli.py: this function used to live
in cli.py, which meant api/server.py -- a long-running HTTP service --
imported its entire startup path from a module named "cli". That was
merely odd while the API was a demonstration of the seam; it becomes a
genuinely confusing dependency once another project consumes the HTTP
interface, and an actively wrong one if the API is ever packaged
separately (a server package importing a command-line module).

Nothing about the assembly itself changed in the move. cli.py still
re-exports both names, so `from research_agent.cli import
build_app_and_settings` keeps working for any caller that has not been
updated -- see cli.py's own import line.

CALLED BY   cli.py::main (once per command-line invocation) and
            api/server.py (once at module-import time, so the FastAPI
            app's _graph and _settings are built once when uvicorn loads
            the module and reused across every request).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from research_agent import langfuse as lf
from research_agent.config import get_settings, split_csv
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import configure_logging, log_event
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.postgres import get_checkpointer
from research_agent.storage.qdrant_store import QdrantStore
from research_agent.tools.corpus_search import make_corpus_tool
from research_agent.tools.model_knowledge import make_model_knowledge_tool
from research_agent.tools.retrieval_chain import make_retrieval_chain
from research_agent.tools.mcp_client import MCPBridge, make_mcp_tool
from research_agent.tracing import NullTracer


class AppBundle(NamedTuple):
    """Everything build_app_and_settings assembles (P2-08).

    Replaces the old bare (app, settings) 2-tuple with named fields so
    `durable` and `checkpointer` are no longer silently dropped by callers
    who only unpack the first two — the exact gap this item exists to
    close (`durable` was previously computed inside build_app_and_settings
    and never returned at all; see get_checkpointer in storage/postgres.py
    for what durable=False actually means operationally).

    CONSUME THESE BY NAME (bundle.app, bundle.settings, ...), never by
    tuple-unpacking. api/server.py unpacked four names from what had become
    a five-field bundle and raised ValueError at import, taking the entire
    HTTP interface down; named access cannot break that way when a field is
    added, which is the whole reason this is a NamedTuple and not a tuple.
    """

    app: object
    settings: object
    durable: bool
    checkpointer: object
    mcp_bridge: object = None  # P2-13: an MCPBridge if settings.mcp_enabled,
                               # else None -- see build_app_and_settings and
                               # main()'s finally block, which closes this
                               # exactly like close_checkpointer(checkpointer)
                               # does, when it's not None.
    router: object = None      # the FallbackRouter, so main()'s finally block
                               # can close its providers' httpx clients --
                               # previously unreachable from either caller,
                               # exactly like checkpointer was before P2-08.


def build_app_and_settings(tracer=None):
    """Assemble the full application (shared by CLI and API).

    This function is the ENTIRE dependency graph of this project laid out
    in one place: every object every node in agents/*.py eventually uses
    (the LLM router, both retrieval stores, the memory wrapper, the
    checkpointer) is constructed HERE and nowhere else. Nothing inside
    orchestration/graph.py or agents/*.py ever imports config.py, httpx,
    qdrant_client, or opensearchpy directly — they receive already-built
    objects as arguments instead (the closure pattern explained across
    agents/*.py's docstrings). This is what makes swapping in fake objects
    for tests, or reusing the exact same wiring from api/server.py, as
    simple as calling this one function.

    CALLED BY   main() below, and api/server.py at module-import time (so
                the FastAPI app's _graph and _settings are built once, when
                uvicorn loads the module, and reused across every incoming
                HTTP request).

    Parameters:
        tracer: optional Tracer for debug tracing (threaded into the LLM chain
            and both retrieval stores). None -> no tracing overhead.

    Returns:
        AppBundle(app, settings, durable, checkpointer) — every dependency
        wired from Settings, with graceful degradation applied by each
        storage module. P2-08: `durable` and `checkpointer` are now part of
        the return value (previously `durable` was computed here and
        silently discarded, and `checkpointer` wasn't returned at all,
        making close_checkpointer's cleanup unreachable from either
        caller).
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    # Phase 3: build the process-wide Langfuse Observer here, alongside
    # everything else this function wires up. LANGFUSE_ENABLED=false (the
    # default) makes this a zero-cost no-op -- see langfuse/client.py's
    # build_client() for exactly what "disabled" guarantees.
    lf.init_from_settings(settings)
    # `tracer or NullTracer()` — if the caller passed a real Tracer, use it;
    # if they passed None (the default), fall back to a NullTracer instead,
    # so every line below can hand `tracer` to a constructor unconditionally
    # without an `if tracer is not None` check at every call site.
    tracer = tracer or NullTracer()
    # debug reuses tracer.enabled rather than being a second, separately-
    # supplied flag: a real Tracer means --debug (or DEBUG_TRACE=true) was
    # on, and that's exactly when we also want every node's "node.enter"
    # line to fire. Deriving it here means graph.py and every agents/*.py
    # builder gets one unambiguous boolean, with no risk of it drifting out
    # of sync with whether tracing is actually on.
    debug = tracer.enabled

    router = FallbackRouter.from_settings(settings, tracer=tracer)
    # Two SEPARATE QdrantStore instances get created in this function — one
    # here (for the corpus, used inside HybridRetriever below) and one
    # further down (for semantic memory) — pointed at two different
    # collection names on the SAME Qdrant server. See storage/
    # qdrant_store.py's docstring for why one class serves both roles.
    dense = QdrantStore(settings.qdrant_url, settings.corpus_index, tracer=tracer)
    keyword = OpenSearchStore(
        settings.opensearch_url, settings.corpus_index,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        tracer=tracer)
    # P2-01: settings.min_similarity is the retrieval-time floor on the
    # dense leg — see retrieval/hybrid.py for what it filters and why.
    # P2-14 (D-25) follow-up to P2-13: the corpus tool is now ALWAYS
    # built and is ALWAYS the default -- settings.mcp_enabled no longer
    # swaps it out wholesale (that was P2-13's original, simpler
    # behavior, before P2-14's tool_hint routing existed to make a
    # genuine choice possible). Instead, when mcp_enabled, the MCP tool
    # is built ADDITIONALLY, as a second, ADDRESSABLE specialist --
    # reachable only when a task's tool_hint == "mcp" (see
    # orchestration/graph.py::dispatch_tasks and
    # agents/planning.py::task_expander_node /
    # agents/gathering.py::gap_generator_node, which are the only two
    # places that ever set that hint, and only when settings.mcp_enabled
    # is what allowed it in the first place). With mcp_enabled off (the
    # default), mcp_tool stays None and build_graph registers no extra
    # node at all -- behavior is unchanged from every run before this.
    corpus_tool = make_corpus_tool(
        HybridRetriever(dense, keyword, min_similarity=settings.min_similarity))
    mcp_bridge = None
    mcp_tool = None
    if settings.mcp_enabled:
        mcp_bridge = MCPBridge(
            command=settings.mcp_server_command,
            args=split_csv(settings.mcp_server_args),
            env_allowlist=split_csv(settings.mcp_server_env_allowlist),
        )
        mcp_tool = make_mcp_tool(
            mcp_bridge, settings.mcp_tool_name,
            query_arg_name=settings.mcp_query_arg_name,
            call_timeout_seconds=settings.mcp_call_timeout_seconds,
            # This build's MCP schema carries no per-hit relevance score,
            # so evidence from it must not be able to SATISFY the coverage
            # gate on its own -- see make_mcp_tool's own docstring for what
            # the previous hardcoded 1.0 did to recall.
            unscored_score=settings.min_evidence_score)
    memory = SemanticMemory(
        QdrantStore(settings.qdrant_url, settings.memory_collection, tracer=tracer,
                    trace_label="QDRANT (semantic memory)"),
        settings.memory_top_k,
        settings.decay_half_life_days_semi_stable,
        settings.decay_half_life_days_volatile,
        server_side_decay=settings.memory_server_side_decay,  # P2-10
    )
    checkpointer, durable = get_checkpointer(settings.postgres_dsn)
    if not durable:
        # P2-08: previously this was visible only as a WARNING log line
        # from get_checkpointer itself (checkpointer.memory_fallback) — a
        # caller that doesn't read logs had no way to know. Surfacing it
        # here too means both cli.py and api/server.py can act on it
        # (print a banner, put it in /health) without re-deriving it.
        log_event(logging.getLogger(__name__), "app.degraded_checkpointing",
                  level=logging.WARNING)
    # D-38: the DEFAULT tool every search_worker receives is no longer the
    # bare corpus tool -- it is the escalation ladder
    # corpus -> reformulated corpus -> mcp -> model knowledge. The graph,
    # the worker contract (D-6/D-15) and the Send-fanout are untouched:
    # this is still one ToolFn. mcp_tool is ALSO still passed separately,
    # so an explicit tool_hint="mcp" (D-25) keeps routing straight to the
    # specialist node, bypassing the ladder, exactly as before.
    model_tool = (make_model_knowledge_tool(router, settings.model_knowledge_score)
                  if settings.model_knowledge_enabled else None)
    tool = make_retrieval_chain(
        corpus_tool, settings.min_evidence_score,
        mcp=mcp_tool, model=model_tool,
        reformulate=settings.query_reformulation_enabled)
    app = build_graph(router, tool, memory, settings, checkpointer, debug=debug,
                     mcp_tool=mcp_tool)  # P2-14: None unless settings.mcp_enabled
    return AppBundle(app=app, settings=settings, durable=durable,
                     checkpointer=checkpointer, mcp_bridge=mcp_bridge,
                     router=router)
