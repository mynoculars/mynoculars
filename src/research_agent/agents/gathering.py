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

Python mechanics used in this file, if any of this is new to you:
    @validated_worker           A DECORATOR. Writing "@something" directly
                                 above a function definition means: "take the
                                 function I just wrote, pass it into
                                 `something`, and replace my function with
                                 whatever `something` returns." It does NOT
                                 change search_worker's own code at all —
                                 it wraps an extra safety check AROUND it.
                                 See orchestration/contracts.py for what the
                                 wrapper actually does.
    ToolFn = Callable[...]       This just gives a NAME to a function
                                 signature, purely for readability in type
                                 hints below — it does not create any new
                                 behaviour, it's a label like a typedef.
    build_search_worker(tool)   This is a "closure" — a function that
                                 builds and returns ANOTHER function
                                 (search_worker), which remembers `tool`
                                 even after build_search_worker has finished
                                 running. That's how every node in this
                                 codebase receives its dependencies without
                                 global variables — see planning.py's module
                                 docstring for the full explanation of why.
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

# A type ALIAS, not a real class: "ToolFn" is just a short, readable name we
# can use in type hints below instead of repeating the whole signature
# "Callable[[SearchTask], List[Evidence]]" (= "a function that takes one
# SearchTask argument and returns a list of Evidence") every time.
ToolFn = Callable[[SearchTask], List[Evidence]]


def build_search_worker(tool: ToolFn, debug: bool = False):
    """Build the fanned-out worker node bound to a retrieval tool.

    The returned function receives WorkerPayload (not ResearchState — D-6)
    and is contract-enforced (D-15): the decorator raises immediately on any
    non-reducer return key, converting a rare concurrent failure into a
    deterministic test failure.
    """

    # The line below, "@validated_worker", is a DECORATOR (see the module
    # docstring above if that word is new). In plain terms: Python takes the
    # search_worker function defined right underneath, hands it to
    # validated_worker() as an argument, and whatever validated_worker()
    # returns becomes the NEW search_worker from this point on. The function
    # body you read below is completely unaware this wrapping exists — the
    # extra check happens outside it, in orchestration/contracts.py.
    @validated_worker
    def search_worker(payload: WorkerPayload) -> Dict[str, Any]:
        """One instance of this runs PER TASK, all in the same LangGraph
        superstep — this is the "map" half of the gather loop's map-reduce.

        READS   payload.task ONLY (a single SearchTask) — deliberately NOT
                the full ResearchState (D-6). A worker cannot see other
                workers' tasks, other goals, or anything else in state; it
                just executes the one query string it was handed.
        CALLS   the injected retrieval tool (tools/corpus_search.py in
                practice) — runs the hybrid dense+BM25 search and converts
                hits into Evidence objects. No LLM call in this node.
        WRITES  exactly ONE of the two outcomes below, never both:

            SUCCESS -> state.evidence            += the Evidence list
                       state.completed_task_keys  += {task.key}
                       state.counters["search_calls"] += 1

            FAILURE -> state.failed_task_keys[task.key] = task.depth
                       state.counters["search_failures"] += 1
                       (the task is NOT added to completed_task_keys — see
                        the D-16 note below)

        NEXT    ALL workers dispatched this cycle must land before anything
                else runs — that join is LangGraph's superstep barrier, not
                code in this file. Only once every worker has returned does
                merger_node (below) execute.

        Every returned dict is checked by @validated_worker (contracts.py)
        against WORKER_WRITABLE_KEYS before LangGraph ever sees it. Return
        any other key here and the run fails immediately and loudly, rather
        than risking a silent InvalidUpdateError under real concurrency.
        """
        task = payload.task
        if debug:
            log_event(logger, "node.enter", node="search_worker", task=task.key)
        try:
            evidence = tool(task)
            # P2-07 follow-up: drain retrieval-side boundary counts if this
            # tool exposes them (the real corpus_search does; test fakes
            # generally don't — see tools/corpus_search.py for exactly why
            # this is an optional, duck-typed capability rather than part
            # of ToolFn's required contract).
            retrieval_counts = getattr(tool, "drain_retrieval_counts", lambda: {})()
            log_event(logger, "worker.done", task=task.key, items=len(evidence))
            return {
                "evidence": evidence,
                "completed_task_keys": {task.key},
                "counters": {"search_calls": 1, **retrieval_counts},
            }
        except Exception as exc:  # noqa: BLE001 — failure is data, not a crash
            # D-16: failed, NOT completed. Re-emission allowed at depth >
            # task.depth, so a transient backend error costs one cycle of
            # delay, not permanent loss of this query formulation.
            # P2-07 follow-up: drain here too — _bump_retrieval_counts runs
            # as the FIRST line of HybridRetriever.search(), so an attempt
            # that raised partway through (e.g. a Qdrant NotFoundError on a
            # collection that was never ingested) still counted as an
            # attempted retrieval call, not a silent zero.
            retrieval_counts = getattr(tool, "drain_retrieval_counts", lambda: {})()
            log_event(logger, "worker.failed", level=logging.WARNING,
                      task=task.key, reason=type(exc).__name__)
            return {
                "failed_task_keys": {task.key: task.depth},
                "counters": {"search_failures": 1, **retrieval_counts},
            }

    return search_worker


def build_merger_node(router: FallbackRouter, settings: Settings, debug: bool = False):
    """Build the evidence merger / contradiction flagger.

    P2-12: now takes `router` and `settings`, the same closure shape
    build_gap_generator_node already uses — needed because the detector
    below is an LLM call, gated by settings.contradiction_detection_enabled.
    """

    def merger_node(state: ResearchState) -> Dict[str, Any]:
        """The "reduce" half of the gather loop. Runs once all search_worker
        instances from this cycle have landed (the superstep barrier — no
        code here waits for that; LangGraph guarantees it).

        READS   state.evidence (everything gathered so far, this run —
                includes memory evidence from turn 1 and every prior
                gather cycle's corpus evidence),
                state.goals, settings.contradiction_detection_enabled.
        CALLS   ONE LLM JSON call (templates.detect_contradictions), but
                ONLY when the gate is on AND at least one goal has 2+
                evidence items — otherwise nothing external, pure Python,
                exactly as before P2-12.
        WRITES  state.goals — same list, each Goal's `contested` flag
                (re)computed from `contested_goal_ids` (see below).
                state.counters["contradictions_flagged"] = count, plus
                "llm_node_calls" and router-drained counters whenever the
                LLM path actually ran.
        NEXT    graph.py routes unconditionally to progress_checker, which
                is the node that actually ACTS on the contested flag (a
                contested goal cannot be marked `covered`, however much
                evidence points at it).

        P2-12: two detection modes, chosen by settings.contradiction_detection_enabled.

        GATE OFF (default — unchanged from pre-P2-12 behaviour): only
        honors an explicit e.contradicts marker placed by a tool. No tool
        in this build ever sets one, so contested_goal_ids is always empty
        in this mode — this is the ORIGINAL, byte-for-byte-preserved
        fallback path, not a regression.

        GATE ON: an LLM pass over evidence grouped by goal_id
        (templates.detect_contradictions), asking which goal_ids have
        genuinely conflicting evidence. `contested_goal_ids` is read
        DIRECTLY from the model's JSON response — this node does NOT write
        onto individual Evidence.contradicts fields, because state.py's
        `evidence` field is an append-only reducer (operator.add): returning
        a rebuilt evidence list from this node would duplicate every item,
        not update it in place. Tracking contested_goal_ids as a plain set
        here is simpler and is the only thing progress_checker_node actually
        reads (via g.contested) — Evidence.contradicts itself is read
        nowhere else in this codebase once the gate is on.

        A detector call that itself errors (bad JSON, provider failure) is
        treated as "nothing contested" — the same fail-open posture
        evaluation/quality.py's score_answer already uses — so a flaky
        detector can never take the whole run down.

        This is what makes E2 reachable in a real run for the first time
        (see README Limitations — update once this is confirmed live, not
        just once these unit tests pass).
        """
        if debug:
            # merger_node makes no store call ever, and makes no LLM call
            # at all when the gate is off — a --debug trace file may still
            # never mention it in that case. This is exactly the "silent
            # node" gap this flag exists to fill.
            log_event(logger, "node.enter", node="merger")

        contested_goal_ids: set = set()
        counters: Dict[str, Any] = {}

        if settings.contradiction_detection_enabled:
            # Only worth the LLM call if at least one goal has 2+ evidence
            # items — a single item cannot contradict itself, and
            # templates.detect_contradictions already skips single-item
            # goals when building its prompt; this early-exit just avoids
            # paying for a call that would ask the model nothing useful.
            multi_evidence_goal_ids = {
                g.goal_id for g in state.goals
                if sum(1 for e in state.evidence if e.goal_id == g.goal_id) >= 2
            }
            if multi_evidence_goal_ids:
                router.set_node("merger")
                try:
                    result = router.complete_json(
                        templates.detect_contradictions(state.goals, state.evidence))
                    contested_goal_ids = set(result.get("contested_goal_ids", []))
                except Exception as exc:  # noqa: BLE001 — fail open, never crash the run
                    log_event(logger, "merger.contradiction_detection_failed",
                              level=logging.WARNING, reason=type(exc).__name__)
                counters = {"llm_node_calls": 1, **router.drain_counters()}
        else:
            # Original D-18 minimal detector, unchanged: honour an explicit
            # marker if any tool ever sets one (none do in this build today).
            contested_goal_ids = {e.goal_id for e in state.evidence if e.contradicts}

        goals = [g.model_copy(update={"contested": g.goal_id in contested_goal_ids})
                 for g in state.goals]
        n = len(contested_goal_ids)
        if n:
            log_event(logger, "merger.contested", goals=sorted(contested_goal_ids))
        counters["contradictions_flagged"] = float(n)
        return {"goals": goals, "counters": counters}

    return merger_node


def build_progress_checker_node(settings: Settings, debug: bool = False):
    """Build the coverage/recall checker — the loop's clock (D-3)."""

    def progress_checker_node(state: ResearchState) -> Dict[str, Any]:
        """The gather loop's ONLY clock. Runs right after merger, every
        single cycle, whether this is the first pass or the fifth.

        READS   state.goals (with merger's `contested` flags already set),
                state.evidence, settings.min_evidence_score,
                settings.max_depth, settings.recall_target,
                settings.hitl_enabled, state.iteration_depth.
        CALLS   nothing external — pure Python.
        WRITES  state.goals — same list, each Goal's `covered` flag set:
                    covered = (has >=1 evidence item scored at or above
                               min_evidence_score for this goal_id)
                              AND (not contested)
                state.recall_score = covered_goals / total_goals
                                     (1.0 if there are no goals at all)
                state.iteration_depth += 1   <- the ONLY place this ticks,
                                                exactly once per cycle
                IF terminally short (hitl_enabled AND depth>=max_depth AND
                recall<target): state.escalation_trigger = "E2" (a
                contested goal is blocking) or "E3" (nothing contested,
                just not enough found)
        NEXT    graph.py's route_convergence reads recall_score, depth and
                escalation_trigger to decide: human_escalation (if a
                trigger fired) -> compiler (recall high enough, OR depth
                budget spent) -> gap_generator (otherwise — go round again).

        ⚠ P2-01 FOLLOW-UP (confirmed via a live run, not a stub): raising
        min_evidence_score off 0.0 was necessary but not sufficient. With
        RRF_SQUASH=30.0 and RRF_K=60 (tools/corpus_search.py,
        retrieval/hybrid.py), a rank-0 hit under SINGLE-LEG fusion (i.e.
        whenever OpenSearch is down and only the dense leg contributes)
        squashes to EXACTLY 1/60 * 30 = 0.5 — not approximately, exactly,
        for ANY query, regardless of actual relevance. A `>=` comparison
        against a min_evidence_score of precisely 0.5 let that
        mathematically-guaranteed boundary value through every time,
        which is why a real end-to-end test still showed recall=1.0 on a
        totally out-of-corpus query even after min_evidence_score was
        raised. Strict `>` closes this specific collision regardless of
        the exact threshold chosen. This does NOT replace the real fix for
        the underlying cause — get BOTH retrieval legs actually
        contributing (fix whatever is making OpenSearch unreachable) — it
        only stops the coverage gate from being fooled by fusion-math
        artifacts in the meantime.
        """
        if debug:
            log_event(logger, "node.enter", node="progress_checker")
        goals = []
        for g in state.goals:
            # Strict `>`, not `>=` — see the docstring above. A score that
            # lands EXACTLY on min_evidence_score is, under single-leg RRF
            # fusion, indistinguishable from "ranked first among whatever
            # came back," which carries no information about actual
            # relevance. Requiring it to exceed the floor, not merely meet
            # it, closes that specific loophole.
            has_quality_evidence = any(
                e.goal_id == g.goal_id and e.score > settings.min_evidence_score
                for e in state.evidence)
            covered = has_quality_evidence and not g.contested
            goals.append(g.model_copy(update={"covered": covered}))

        recall = (sum(g.covered for g in goals) / len(goals)) if goals else 1.0
        depth = state.iteration_depth + 1  # D-3: exactly one tick per cycle
        log_event(logger, "node.progress", recall=round(recall, 3), depth=depth)
        update = {"goals": goals, "recall_score": recall, "iteration_depth": depth}
        # D-23: at terminal non-convergence the CHECK raises the trigger
        # (E2 if a contradiction blocks a goal, else E3). Routing reads it.
        # P2-09: the non-convergence CONDITION is evaluated regardless of
        # hitl_enabled now — previously E2/E3 emitted nothing at all when
        # HITL was off, unlike E1 (goal_manager_node) and E4 (critic_node),
        # which both already log an "escalation.stub" WARNING in their
        # disabled-mode branch. That asymmetry made a terminally-stuck run
        # look identical to a converged one in the logs whenever HITL
        # happened to be off — this restores parity across all four
        # triggers without changing any actual routing behaviour (only
        # settings.hitl_enabled still decides whether escalation_trigger is
        # ever SET, which is the only thing route_convergence reads).
        if depth >= settings.max_depth and recall < settings.recall_target:
            trigger = "E2" if any(g.contested for g in goals) else "E3"
            if settings.hitl_enabled:
                update["escalation_trigger"] = trigger
                log_event(logger, "escalation.raised", trigger=trigger,
                          recall=round(recall, 3))
            else:
                log_event(logger, "escalation.stub", level=logging.WARNING,
                          trigger=trigger, recall=round(recall, 3),
                          reason="depth_exhausted")
        return update

    return progress_checker_node


def build_gap_generator_node(router: FallbackRouter, settings: Settings,
                             debug: bool = False):
    """Build the gap generator: new tasks for uncovered goals."""

    def gap_generator_node(state: ResearchState) -> Dict[str, Any]:
        """The most agentic node in the whole graph: it looks at what was
        actually found, decides what is still missing, and writes new
        queries for the gap. This IS the gather loop — every trip round it
        passes through here.

        Only reached when progress_checker's route_convergence decided
        recall is still below target and depth budget remains.

        READS   state.goals (with `covered` flags from progress_checker),
                the last 10 items of state.evidence,
                state.iteration_depth, settings.max_fanout,
                state.human_guidance — non-empty ONLY after an E2/E3
                "redirect" from human_escalation.
        CALLS   one LLM JSON call: "here's what's covered and what isn't,
                write queries to close the gap" (templates.generate_gaps).
        WRITES  state.pending_tasks = [SearchTask, ...]   (replaces the
                    backlog wholesale, same as task_expander — D-2)
                state.human_guidance = ""   (consumed)
                state.counters["llm_calls"] += 1
                IF the model produced NO usable tasks AND recall is still
                below target: raises E2/E3 here too (see note below).
        NEXT    graph.py's dispatch_tasks (the SAME routing function
                task_expander uses): empty backlog -> compiler; otherwise
                -> one parallel search_worker per new task, looping back
                into the gather cycle.

        Tasks pass through the same task_utils.cap_and_filter as
        task_expander (D-13 cap / D-2 dedup / D-16 depth-gated retry), but
        called with depth=state.iteration_depth rather than 0 — that depth
        value is exactly what makes D-16's "retry only at a strictly
        greater depth" rule meaningful for tasks that failed earlier.

        WHY THIS NODE ALSO RAISES E2/E3 (progress_checker already does):
        a run can fail to converge two different ways. progress_checker
        catches "depth budget is spent" — but a run can ALSO run out of
        NEW tasks to try (every candidate query already tried or already
        failed) before depth is exhausted. That second failure mode only
        becomes visible here, after cap_and_filter has actually filtered
        the model's output down to nothing. Both are "cannot converge
        below target," so both raise the same trigger codes.
        """
        router.set_node("gap_generator")
        if debug:
            log_event(logger, "node.enter", node="gap_generator")
        result = router.complete_json(templates.generate_gaps(
            state.goals, state.evidence, state.iteration_depth, settings.max_fanout,
            guidance=state.human_guidance))
        # P2-06: same validated cap_and_filter seam task_expander_node uses.
        tasks, rejected = cap_and_filter(result.get("tasks", []), state,
                                         depth=state.iteration_depth,
                                         max_fanout=settings.max_fanout)
        log_event(logger, "node.gaps", produced=len(tasks), rejected=rejected)
        counters = {"llm_node_calls": 1, **router.drain_counters()}
        if rejected:
            counters["producer_rejects"] = float(rejected)
        update = {"pending_tasks": tasks, "human_guidance": "", "counters": counters}
        # D-23 (refined by test evidence): E3 originally guarded only the
        # depth-exhaustion exit; a run can ALSO fail to converge by running
        # out of producible tasks (the D-14 dispatch exit). Both are "cannot
        # converge below target" — so the trigger is raised here too.
        # P2-09: same disabled-mode logging parity as progress_checker_node
        # above — see that node's comment for why this branch exists.
        if not tasks and state.recall_score < settings.recall_target:
            trigger = "E2" if any(g.contested for g in state.goals) else "E3"
            if settings.hitl_enabled:
                update["escalation_trigger"] = trigger
                log_event(logger, "escalation.raised", trigger=trigger,
                          reason="task_supply_exhausted")
            else:
                log_event(logger, "escalation.stub", level=logging.WARNING,
                          trigger=trigger, reason="task_supply_exhausted")
        return update

    return gap_generator_node
