"""
cli.py — Command-line entry point.

Purpose:
    Wire real dependencies, run one research query, print the report and
    telemetry. The only place (besides api/server.py) where everything is
    assembled — nodes stay wiring-free.

Responsibilities:
    - Argument parsing, dependency assembly, single invoke, run-history row.

Usage:
    python -m research_agent.cli "Compare Redis and Memcached for session caching"
    python -m research_agent.cli "..." --thread-id my-run-1
"""

import argparse
import json
import sys
import uuid

from langgraph.types import Command

from research_agent.config import get_settings
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import configure_logging, run_id_var
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.orchestration.graph import build_graph
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.state import ResearchState
from research_agent.storage.opensearch_store import OpenSearchStore
from research_agent.storage.postgres import get_checkpointer, record_run
from research_agent.storage.qdrant_store import QdrantStore
from research_agent.tools.corpus_search import make_corpus_tool


def build_app_and_settings():
    """Assemble the full application (shared by CLI and API).

    Returns:
        (compiled_graph, settings) — every dependency wired from Settings,
        with graceful degradation applied by each storage module.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    router = FallbackRouter.from_settings(settings)
    dense = QdrantStore(settings.qdrant_url, settings.corpus_index)
    keyword = OpenSearchStore(
        settings.opensearch_url, settings.corpus_index,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs)
    tool = make_corpus_tool(HybridRetriever(dense, keyword))
    memory = SemanticMemory(
        QdrantStore(settings.qdrant_url, settings.memory_collection),
        settings.memory_top_k,
        settings.decay_half_life_days_semi_stable,
        settings.decay_half_life_days_volatile,
    )
    checkpointer, durable = get_checkpointer(settings.postgres_dsn)
    app = build_graph(router, tool, memory, settings, checkpointer)
    return app, settings


def main(argv=None) -> int:
    """Run one query end to end; returns process exit code."""
    parser = argparse.ArgumentParser(description="Agentic research agent (core build)")
    parser.add_argument("query", help="The research question")
    parser.add_argument("--thread-id", default=None,
                        help="Run identity (fresh UUID per run by default; "
                             "reuse an old id to resume — design D-20)")
    args = parser.parse_args(argv)

    app, settings = build_app_and_settings()
    thread_id = args.thread_id or f"run-{uuid.uuid4().hex[:12]}"
    run_id_var.set(thread_id)  # correlate every log line to this run

    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": settings.recursion_limit}
    result = app.invoke(ResearchState(raw_query=args.query), config=config)

    # HITL loop (D-23): an interrupted run surfaces "__interrupt__" instead
    # of finishing. Show the review payload, collect a decision, resume under
    # the SAME thread_id (D-20). Blocking stdin IS the timeout policy for a
    # CLI — the deferred operational decision, resolved per-interface.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== HUMAN REVIEW REQUIRED ===")
        print(json.dumps(payload, indent=2, default=str))
        action = ""
        while action not in ("approve", "redirect", "abort"):
            action = input("action [approve/redirect/abort]: ").strip().lower()
        guidance = input("guidance: ").strip() if action == "redirect" else ""
        result = app.invoke(Command(resume={"action": action, "guidance": guidance}),
                            config=config)

    print(result["final_report"])
    print("\n--- telemetry ---")
    print(json.dumps(result["telemetry"], indent=2))
    record_run(settings.postgres_dsn, thread_id, args.query,
               result["telemetry"].get("recall", 0.0), result["telemetry"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
