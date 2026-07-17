"""
agents/compilation.py — Phase 3/4 nodes: compile, critique, persist, measure.

Purpose:
    Turn gathered evidence into the final report, judge it, write memory,
    and aggregate telemetry.

Responsibilities:
    - compiler_node: compose the Markdown report; on revision passes it
      receives the critic's notes so the rewrite is grounded (D-22), never
      a blind retry.
    - critic_node: judge faithfulness/completeness ONLY (evidence
      sufficiency is the progress checker's job — one judge per question).
      Bounded by max_revisions; exhaustion triggers the E4 escalation STUB
      (a log line in this core build; interrupt() in the full design).
    - memory_writer_node: persist fresh evidence after a PASSED critique
      (D-24 — bad reports never feed long-term memory).
    - telemetry_node: aggregate node-accumulated counters into the final
      telemetry record (D-12: the logger aggregates, it never invents).
"""

import logging
from typing import Any, Dict

from research_agent.config import Settings
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import log_event
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.prompts import templates
from research_agent.state import ResearchState

logger = logging.getLogger(__name__)


def build_compiler_node(router: FallbackRouter):
    """Build the report compiler."""

    def compiler_node(state: ResearchState) -> Dict[str, Any]:
        if state.abort_reason:
            # Human abort (D-23): terminal, explicit, still reaches END.
            report = (f"# Research Report — aborted by human reviewer\n\n"
                      f"**Query:** {state.raw_query}\n\n"
                      f"**Reason:** {state.abort_reason}\n")
            return {"final_report": report}
        if state.planning_error:
            # D-21 path: an explicit, diagnosable error report beats silence.
            report = (f"# Research Report — planning failed\n\n"
                      f"**Query:** {state.raw_query}\n\n"
                      f"**Problem:** {state.planning_error}\n\n"
                      f"No retrieval was attempted. Rephrase the query and retry.")
            return {"final_report": report}
        report = router.complete(templates.compile_report(
            state.raw_query, state.goals, state.evidence, state.critique_notes))
        return {"final_report": report, "counters": {"llm_calls": 1}}

    return compiler_node


def build_critic_node(router: FallbackRouter, settings: Settings):
    """Build the report critic (bounded self-critique loop, D-22)."""

    def critic_node(state: ResearchState) -> Dict[str, Any]:
        if state.planning_error or state.abort_reason:
            # Nothing to judge on the error/abort paths; wave them through so
            # the run still terminates at telemetry with a clear report.
            return {"critique_passed": True}
        result = router.complete_json(templates.critique(
            state.raw_query, state.final_report, state.goals))
        passed = bool(result.get("passed", False))
        notes = [str(n) for n in result.get("notes", [])]
        revision = state.revision_count + 1
        update: Dict[str, Any] = {
            "critique_passed": passed,
            "revision_count": revision,
            "counters": {"llm_calls": 1, "revision_cycles": 1},
        }
        if not passed:
            update["critique_notes"] = notes  # accumulates via reducer
            if revision >= settings.max_revisions:
                if settings.hitl_enabled:
                    # D-23: raise E4 — the graph will interrupt for a human.
                    update["escalation_trigger"] = "E4"
                    log_event(logger, "escalation.raised", trigger="E4",
                              revisions=revision)
                else:
                    # Stub path (HITL disabled): log loudly and ship the
                    # report marked unreviewed — never silently as "good".
                    log_event(logger, "escalation.stub", level=logging.WARNING,
                              trigger="E4", revisions=revision, notes=notes)
        log_event(logger, "node.critique", passed=passed, revision=revision)
        return update

    return critic_node


def build_memory_writer_node(memory: SemanticMemory):
    """Build the memory write-back node (runs only after a passed critique)."""

    def memory_writer_node(state: ResearchState) -> Dict[str, Any]:
        written = memory.store_run(state.raw_query, state.evidence)
        return {"counters": {"memory_writes": float(written)}}

    return memory_writer_node


def build_telemetry_node():
    """Build the telemetry aggregator — pure aggregation, no invention."""

    def telemetry_node(state: ResearchState) -> Dict[str, Any]:
        c = state.counters
        telemetry = {
            "intent": state.classification.get("intent"),
            "goals": len(state.goals),
            "iterations": state.iteration_depth,
            "evidence_items": len(state.evidence),
            "recall": round(state.recall_score, 3),
            "llm_calls": int(c.get("llm_calls", 0)),
            "search_calls": int(c.get("search_calls", 0)),
            "search_failures": int(c.get("search_failures", 0)),
            "memory_hits": int(c.get("memory_hits", 0)),
            "memory_writes": int(c.get("memory_writes", 0)),
            "revision_cycles": int(c.get("revision_cycles", 0)),
            "critique_passed": state.critique_passed,
            "planning_error": state.planning_error,
        }
        log_event(logger, "run.telemetry", **telemetry)
        return {"telemetry": telemetry}

    return telemetry_node
