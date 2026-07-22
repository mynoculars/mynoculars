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
        """Turns gathered evidence into the deliverable. Reached from THREE
        different places in the graph: normally from route_convergence
        (gather loop finished), and as an error/abort SINK from two other
        paths — this function's first job is figuring out which case it's
        in.

        READS   state.abort_reason      — set only by human_escalation
                state.planning_error    — set only by goal_manager_node
                state.raw_query, state.goals, state.evidence,
                state.critique_notes    — non-empty ONLY on a rewrite pass
        CALLS   router.complete() — the ONLY free-text (non-JSON) LLM call
                anywhere in this codebase, and therefore the only call that
                passes through the router's quality gate (see llm/router.py).
                Skipped entirely on the abort/error short-circuits below.
        WRITES  state.final_report = <markdown string>
                state.counters["llm_calls"] += 1   (normal path only)
        NEXT    graph.py routes unconditionally to critic.

        Three distinct shapes for final_report, in priority order:
          1. abort_reason is set    -> "aborted by human reviewer" report,
                                        NO LLM call.
          2. planning_error is set  -> "planning failed" report (D-21 —
                                        diagnosable beats silent), NO LLM
                                        call.
          3. neither                -> normal path: build the evidence +
                                        goals context (this IS the RAG
                                        "context construction" step — every
                                        piece of evidence, memory and
                                        corpus alike, is inlined with no
                                        truncation or re-ranking), inject
                                        critique_notes if this is a rewrite
                                        (D-22 — grounded, not blind), call
                                        the model.

        Note that critic_node (below) runs unconditionally after this
        regardless of which of the three paths fired — see its
        short-circuit for how it avoids judging an error report.
        """
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
        router.set_node("compiler")
        report = router.complete(templates.compile_report(
            state.raw_query, state.goals, state.evidence, state.critique_notes))
        return {"final_report": report, "counters": {"llm_calls": 1}}

    return compiler_node


def build_critic_node(router: FallbackRouter, settings: Settings):
    """Build the report critic (bounded self-critique loop, D-22)."""

    def critic_node(state: ResearchState) -> Dict[str, Any]:
        """The agent judges its own output. Runs unconditionally right
        after compiler — including after compiler's own error/abort
        short-circuits, which is why the first check here matters.

        READS   state.planning_error, state.abort_reason — if either is
                set there is nothing meaningful to critique.
                Otherwise: state.raw_query, state.final_report, state.goals.
        CALLS   router.complete_json() asking ONLY two things: is the report
                faithful to its OWN stated evidence, and does it address
                every goal. The prompt explicitly forbids judging whether
                MORE research was needed — that question belongs to
                progress_checker, two phases upstream. One judge, one
                question, each — merging these two would make the loop
                unable to tell which remedy (rewrite vs. re-research) to
                apply.
        WRITES  state.critique_passed = bool
                state.revision_count += 1        (ticks on EVERY pass,
                                                  pass or fail)
                state.counters["llm_calls"] += 1, ["revision_cycles"] += 1
                IF FAILED: state.critique_notes += notes  (accumulates via
                    reducer — these are what gets injected into the NEXT
                    compile prompt, per compiler_node's D-22 grounding)
                    IF revision_count has hit settings.max_revisions:
                        hitl_enabled -> state.escalation_trigger = "E4"
                        else         -> just log a WARNING, ship the
                                        report unreviewed
        NEXT    graph.py's route_after_critique: passed -> memory_writer;
                failed + budget left -> compiler (rewrite loop); failed +
                E4 triggered -> human_escalation; failed + budget spent,
                HITL off -> telemetry directly, SKIPPING memory_writer (a
                report that failed its own quality bar never enters
                long-term memory).

        The short-circuit below (nothing to judge on error/abort paths)
        exists so those two paths still reach telemetry/END with a clean
        signal, rather than being critiqued against evidence that was
        never gathered.
        """
        if state.planning_error or state.abort_reason:
            return {"critique_passed": True}
        router.set_node("critic")
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
        """Only reachable via route_after_critique when
        state.critique_passed is True — a report that failed its own
        quality bar never gets here, and never teaches future runs
        anything (D-24).

        READS   state.raw_query, state.evidence (everything gathered this
                run — memory.store_run internally filters out anything
                already tagged source="memory", so recalled facts are never
                re-written and cannot echo themselves across runs).
        CALLS   memory.store_run(query, evidence) — embeds each fresh
                evidence item's content and upserts it into Qdrant's
                long-term memory collection. No LLM call.
        WRITES  state.counters["memory_writes"] += written_count
                (the actual write lands in Qdrant, not in graph state —
                see memory/semantic_memory.py for the point structure)
        NEXT    graph.py routes unconditionally to telemetry.
        """
        written = memory.store_run(state.raw_query, state.evidence)
        return {"counters": {"memory_writes": float(written)}}

    return memory_writer_node


def build_telemetry_node():
    """Build the telemetry aggregator — pure aggregation, no invention."""

    def telemetry_node(state: ResearchState) -> Dict[str, Any]:
        """The mandatory final node — every single path through this graph
        reaches here, whether it went through memory_writer, an abort, a
        planning error, or an exhausted critique budget.

        READS   state.classification, state.goals, state.iteration_depth,
                state.evidence, state.recall_score, state.counters (the
                additive tallies every earlier node has been contributing
                to all run), state.critique_passed, state.planning_error.
        CALLS   nothing external — pure aggregation.
        WRITES  state.telemetry = {intent, goals, iterations,
                    evidence_items, recall, llm_calls, search_calls,
                    search_failures, memory_hits, memory_writes,
                    revision_cycles, critique_passed, planning_error}
        NEXT    graph.py routes unconditionally to END. cli.py then prints
                state.final_report followed by this telemetry dict, and
                persists a summary row to Postgres (CLI only — the API
                path never calls record_run).

        D-12: this function ONLY adds up numbers other nodes already
        recorded in state.counters — it never invents a figure. That said,
        those counters are NODE-scoped, not call-scoped: llm_calls counts
        how many NODES made an LLM call, not how many provider requests
        actually went out (a node that fell through two fallback hops
        still counts as one). Read a debug trace (--debug) rather than
        this dict if you need the true call volume.
        """
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
