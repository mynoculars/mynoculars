"""
agents/gathering.py — Phase 2 nodes: search worker, merger, checker, gaps.

Purpose:
    The cyclic gather loop's node functions: parallel retrieval (map),
    evidence reconciliation (reduce), coverage measurement, gap generation.

Responsibilities:
    - search_worker: one instance per dispatched task (via Send). Wrapped in
      @validated_worker (D-15) — may return ONLY reducer-backed keys.
      Records success into completed_task_keys OR failure into
      failed_task_keys (exactly one of the two per task, D-16).
    - merger_node: contradiction flagging hook (D-18). The DETECTION
      heuristic here is minimal (explicit `contradicts` markers from tools);
      the MACHINERY (contested goals block coverage) is fully wired.
    - progress_checker_node: quality-gated, contradiction-aware coverage
      (D-17/D-18) + the iteration counter (D-3) — the loop's only clock.
    - gap_generator_node: new ranked/capped/filtered tasks for uncovered
      goals (same hygiene as the expander, shared code).
"""

import logging
from typing import Any, Callable, Dict, List

from research_agent.agents.task_utils import cap_and_filter
from research_agent.config import Settings
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import log_event
from research_agent.orchestration.contracts import validated_worker
from research_agent.prompts import templates
from research_agent.state import Evidence, ResearchState, SearchTask, WorkerPayload

logger = logging.getLogger(__name__)

ToolFn = Callable[[SearchTask], List[Evidence]]


def build_search_worker(tool: ToolFn):
    """Build the fanned-out worker node bound to a retrieval tool.

    The returned function receives WorkerPayload (not ResearchState — D-6)
    and is contract-enforced (D-15): the decorator raises immediately on any
    non-reducer return key, converting a rare concurrent failure into a
    deterministic test failure.
    """

    @validated_worker
    def search_worker(payload: WorkerPayload) -> Dict[str, Any]:
        task = payload.task
        try:
            evidence = tool(task)
            log_event(logger, "worker.done", task=task.key, items=len(evidence))
            return {
                "evidence": evidence,
                "completed_task_keys": {task.key},
                "counters": {"search_calls": 1},
            }
        except Exception as exc:  # noqa: BLE001 — failure is data, not a crash
            # D-16: failed, NOT completed. Re-emission allowed at depth >
            # task.depth, so a transient backend error costs one cycle of
            # delay, not permanent loss of this query formulation.
            log_event(logger, "worker.failed", level=logging.WARNING,
                      task=task.key, reason=type(exc).__name__)
            return {
                "failed_task_keys": {task.key: task.depth},
                "counters": {"search_failures": 1},
            }

    return search_worker


def build_merger_node():
    """Build the evidence merger / contradiction flagger."""

    def merger_node(state: ResearchState) -> Dict[str, Any]:
        # Minimal detection: honor explicit contradiction markers placed by
        # tools (none in the sample corpus). The important part is wired
        # regardless: any contradicted goal becomes contested, and contested
        # goals are excluded from coverage in the checker — which drives the
        # gap generator to seek adjudicating evidence (D-18). A semantic
        # (LLM-based) detector slots in here later without touching wiring.
        contested_goal_ids = {e.goal_id for e in state.evidence if e.contradicts}
        goals = [g.model_copy(update={"contested": g.goal_id in contested_goal_ids})
                 for g in state.goals]
        n = len(contested_goal_ids)
        if n:
            log_event(logger, "merger.contested", goals=sorted(contested_goal_ids))
        return {"goals": goals,
                "counters": {"contradictions_flagged": float(n)}}

    return merger_node


def build_progress_checker_node(settings: Settings):
    """Build the coverage/recall checker — the loop's clock (D-3)."""

    def progress_checker_node(state: ResearchState) -> Dict[str, Any]:
        goals = []
        for g in state.goals:
            # Coverage rule (§6.5): needs at least one evidence item for this
            # goal at/above the quality gate, AND the goal must not be
            # contested. min_evidence_score defaults to 0.0 (inert) until the
            # score distribution is observed — graceful-degradation default.
            has_quality_evidence = any(
                e.goal_id == g.goal_id and e.score >= settings.min_evidence_score
                for e in state.evidence)
            covered = has_quality_evidence and not g.contested
            goals.append(g.model_copy(update={"covered": covered}))

        recall = (sum(g.covered for g in goals) / len(goals)) if goals else 1.0
        depth = state.iteration_depth + 1  # D-3: exactly one tick per cycle
        log_event(logger, "node.progress", recall=round(recall, 3), depth=depth)
        return {"goals": goals, "recall_score": recall, "iteration_depth": depth}

    return progress_checker_node


def build_gap_generator_node(router: FallbackRouter, settings: Settings):
    """Build the gap generator: new tasks for uncovered goals."""

    def gap_generator_node(state: ResearchState) -> Dict[str, Any]:
        result = router.complete_json(templates.generate_gaps(
            state.goals, state.evidence, state.iteration_depth, settings.max_fanout))
        tasks = cap_and_filter(result.get("tasks", []), state,
                                depth=state.iteration_depth,
                                max_fanout=settings.max_fanout)
        log_event(logger, "node.gaps", produced=len(tasks))
        return {"pending_tasks": tasks, "counters": {"llm_calls": 1}}

    return gap_generator_node
