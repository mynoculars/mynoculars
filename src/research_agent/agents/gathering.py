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
from typing import Any, Callable, Dict, List, Optional

from research_agent import langfuse as lf
from research_agent.agents.escalation import raise_or_log
from research_agent.limits import (elapsed_seconds, run_budget_exhausted,
                                   tokens_used)
from research_agent.agents.task_utils import cap_and_filter
from research_agent.config import Settings
from research_agent.guardrails.retrieval import (SINGLE_LEG_SCORE_CEILING,
                                                  has_grounded_evidence,
                                                  passes_evidence_gate)
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import log_event, run_id_var
from research_agent.orchestration.contracts import validated_worker
from research_agent.prompts import templates
from research_agent.retrieval.terms import distinctive_terms
from research_agent.state import (Evidence, Goal, ResearchState, SearchTask,
                                  WorkerPayload)

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
            # P2-13 follow-up: source=... gives immediate per-task
            # visibility in a --debug trace into which tool actually
            # answered THIS task -- evidence[0].source is enough (every
            # item a single tool call produces shares one source; a task
            # that got zero evidence has none to report, hence the
            # "none" fallback rather than indexing an empty list).
            log_event(logger, "worker.done", task=task.key, items=len(evidence),
                      # Every tier of the ladder (D-38) can contribute to one
                      # task, so evidence[0] names whichever tier happened to
                      # run FIRST -- live, that reported source="corpus" for
                      # tasks the model tier actually answered. Report all of
                      # them.
                      source=",".join(sorted({e.source for e in evidence}))
                      or "none")
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
            # error=str(exc)[:300] follow-up: `reason` alone (the
            # exception CLASS name) was found to be genuinely unhelpful in
            # practice -- a real MCP-tool failure showed "reason=
            # TimeoutError" with no way to tell what timed out, how long,
            # or against what. str(exc) carries the actual message (for
            # TimeoutError raised by concurrent.futures.Future.result,
            # this is normally empty -- see tools/mcp_client.py::call_tool,
            # which now wraps that specific case with a real message
            # before it ever gets here). [:300] caps it the same way
            # every other user-facing text slice in this codebase is
            # capped (see tools/corpus_search.py's content[:800] for the
            # same idiom), so one enormous exception message can't bloat
            # a log line.
            log_event(logger, "worker.failed", level=logging.WARNING,
                      task=task.key, reason=type(exc).__name__, error=str(exc)[:300])
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
                              level=logging.WARNING, reason=type(exc).__name__,
                              error=str(exc)[:300])
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
                state.grounded_score_prev = last cycle's MEASURED
                    grounded_score (S-8, so route_convergence can detect a
                    stall between cycles), or the -1.0 "no previous cycle"
                    sentinel on the FIRST cycle, where no measurement
                    exists yet -- see the write itself, below, for why
                    that distinction is load-bearing.
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
        grounded_flags: List[bool] = []  # Guardrail G2 accumulator, see below
        # Same topical gate telemetry_node's corpus_recall already applies
        # (D-39) -- precomputed once, outside the per-goal loop, since
        # every goal's description is fixed for this whole node call.
        goal_terms = {g.goal_id: distinctive_terms(g.description)
                     for g in state.goals}
        for g in state.goals:
            # Strict `>`, not `>=` — see the docstring above. A score that
            # lands EXACTLY on min_evidence_score is, under single-leg RRF
            # fusion, indistinguishable from "ranked first among whatever
            # came back," which carries no information about actual
            # relevance. Requiring it to exceed the floor, not merely meet
            # it, closes that specific loophole.
            has_quality_evidence = any(
                e.goal_id == g.goal_id and passes_evidence_gate(e.score, settings.min_evidence_score)
                for e in state.evidence)
            covered = has_quality_evidence and not g.contested
            # Guardrail G2: a SEPARATE verdict from has_quality_evidence
            # above -- that one asks "is any item strong enough to cover
            # this goal", this one asks "is at least one of those items a
            # real document, not the model's own recollection, that is
            # ACTUALLY ABOUT this goal". A goal can be `covered` (recall
            # counts it) while being ungrounded (grounded_score does not)
            # -- that gap is exactly what run p205.131-check's
            # recall=1.0 / corpus_recall=0.0 outcome was.
            #
            # P205.132 follow-up: the FIRST version of this check tested
            # only source+score, no topic -- and a live run showed it
            # fooled the same way corpus_recall itself was once fooled
            # (see telemetry_node's own D-39 note): gap_generator emitted
            # a task tagged g1 that was actually about Redis vs Memcached
            # (the sample corpus's real content, not this run's actual
            # economic-comparison goal), it scored well over the floor,
            # and source="corpus" let it count as "grounded" for a goal
            # it has nothing to do with. grounded_score moved 0.0 -> 0.2
            # that cycle while corpus_recall correctly stayed 0.0 the
            # whole run, because ONLY corpus_recall applied the topical
            # overlap gate. Reusing that exact gate here closes the gap
            # between the two numbers instead of leaving G2 as a second,
            # weaker grounding metric that disagrees with the first.
            grounded = has_grounded_evidence(
                g.goal_id, goal_terms.get(g.goal_id, set()),
                state.evidence, settings.min_evidence_score)
            grounded_covered = covered and grounded
            goals.append(g.model_copy(update={"covered": covered}))
            # grounded_covered is accumulated here and folded into
            # grounded_score once ALL goals have been walked -- not
            # stored on the Goal model itself, to avoid a schema change
            # for a value only route_convergence (next node) needs in
            # aggregate.
            grounded_flags.append(grounded_covered)

        recall = (sum(g.covered for g in goals) / len(goals)) if goals else 1.0
        grounded = (sum(grounded_flags) / len(goals)) if goals else 1.0
        depth = state.iteration_depth + 1  # D-3: exactly one tick per cycle
        log_event(logger, "node.progress", recall=round(recall, 3),
                  grounded=round(grounded, 3), depth=depth)
        lf.score(run_id_var.get(), "recall", recall, comment=f"depth={depth}")
        lf.score(run_id_var.get(), "coverage",
                sum(g.covered for g in goals) / len(goals) if goals else 1.0)
        lf.score(run_id_var.get(), "grounded", grounded, comment=f"depth={depth}")
        # S-8: the value grounded_score HELD before this cycle's write --
        # i.e. last cycle's MEASUREMENT. route_convergence compares the new
        # value against this one to detect a stall (see its own comment).
        #
        # D-80: the first-cycle guard below is load-bearing, not defensive
        # tidying. On cycle 1 there IS no previous measurement:
        # state.grounded_score is still its construction default of 1.0
        # (state.py picks 1.0 so a run with zero goals never reads as
        # falsely ungrounded), a value this node has not yet written even
        # once. Copying it recorded a phantom "the previous cycle scored
        # 1.0", and route_convergence's stall check then read this cycle's
        # real 0.00 as a failure to improve on it -- routing to the
        # compiler at depth 1 with the entire gather loop unused, for every
        # run whose first cycle is ungrounded. Live, run p205.246-check:
        #
        #   convergence.grounding_stalled grounded=0.0 grounded_prev=1.0
        #                                 depth=1 max_depth=3
        #
        # and that run's own narrative summary read "Gather laps: 1"
        # against MAX_DEPTH=3.
        #
        # -1.0 is grounded_score_prev's OWN documented "no previous cycle
        # yet" sentinel (state.py), and route_convergence already excludes
        # it explicitly (`state.grounded_score_prev >= 0.0`). Both the
        # sentinel and the router's test for it were correct all along --
        # nothing ever WROTE one, so the branch protecting the first cycle
        # could never be taken. This writes it.
        #
        # state.iteration_depth is the value BEFORE this node's increment
        # (`depth`, above, is the incremented one), so `== 0` is exactly
        # "no gather cycle has completed yet".
        is_first_cycle = state.iteration_depth == 0
        update = {"goals": goals, "recall_score": recall,
                  "grounded_score": grounded, "iteration_depth": depth,
                  "grounded_score_prev": (-1.0 if is_first_cycle
                                          else state.grounded_score)}
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
            # D-23 bound: this branch fires on EVERY cycle once the depth
            # budget is spent, and a human redirect routes straight back
            # here -- see agents/escalation.py::raise_or_log, which folds
            # escalation_allowed() with the suppressed/stub logging.
            update.update(raise_or_log(state, settings, trigger,
                                       reason="depth_exhausted",
                                       recall=round(recall, 3)))
        # D-132: the run-budget check, at the end of a gather lap -- one
        # of exactly two places it runs (compiler_node is the other).
        # Here rather than inside search_worker deliberately: a check
        # that fired mid-fan-out would abandon retrieval already paid
        # for, and the lap ends microseconds later anyway.
        #
        # Sets a FLAG; route_convergence reads it and sends the run to
        # the compiler instead of another lap. The CHECK writes and the
        # ROUTER reads, exactly as D-23 splits escalation_trigger --
        # routing functions are pure and cannot write.
        #
        # `not state.budget_exhausted` keeps this idempotent: the first
        # lap to notice records WHICH budget was spent, and a later lap
        # cannot overwrite "deadline" with "tokens".
        if not state.budget_exhausted:
            spent = run_budget_exhausted(state, settings)
            if spent:
                update["budget_exhausted"] = spent
                log_event(logger, "run.budget_exhausted",
                          level=logging.WARNING, budget=spent, node="progress_checker",
                          elapsed_s=round(elapsed_seconds(state), 1),
                          paused_s=round(state.paused_seconds, 1),
                          tokens=tokens_used(state),
                          deadline_s=settings.run_deadline_seconds,
                          token_budget=settings.run_token_budget,
                          depth=depth, recall=round(recall, 3),
                          effect="stopping the gather loop and compiling "
                                 "from the evidence gathered so far")
        return update

    return progress_checker_node


def _uncovered_goal_has_strong_evidence(goal_id: str,
                                        evidence: List[Evidence]) -> bool:
    """True if this goal has at least one evidence item both retrieval
    legs agreed on (score strictly above the single-leg RRF ceiling).
    Pure and side-effect-free so it is trivially unit-testable without a
    running graph -- see tests/unit/test_agents_gathering.py."""
    return any(e.goal_id == goal_id and e.score > SINGLE_LEG_SCORE_CEILING
               for e in evidence)


def _goal_has_any_evidence(goal_id: str, evidence: List[Evidence]) -> bool:
    """True if this goal retrieved ANYTHING at all, at any score.

    A goal that retrieved NOTHING is the one case the strong-evidence guard
    below must NOT fire on. The guard exists to stop the model theming new
    queries on whatever irrelevant text happened to survive in the tail
    (see gap_generator_node's docstring) -- a goal with an empty tail
    presents no such risk, and a fresh query formulation is exactly the
    remedy it needs. Live (run p205.68-check): goal g2 retrieved zero
    items, which made `_uncovered_goal_has_strong_evidence` vacuously
    False, which fired the guard and escalated the whole run at depth 1
    -- the opposite of the intended behaviour."""
    return any(e.goal_id == goal_id for e in evidence)


def _no_strong_evidence_exit(state: ResearchState, settings: Settings
                             ) -> Optional[Dict[str, Any]]:
    """The pre-LLM give-up guard (S-19). Returns an update dict when this
    cycle should stop WITHOUT calling the model, else None.

    Closes the third convergence-failure mode gap_generator_node's own
    docstring describes. SINGLE_LEG_SCORE_CEILING (0.5) is not a new
    threshold -- it is the exact score an RRF rank-0 hit gets when only
    ONE retrieval leg answered, imported from the same constant
    compile_report's grounding rule uses. A goal whose best evidence sits
    at or below it has no document that both legs agreed on, which is the
    cheapest available signal that the corpus may not genuinely cover it
    -- verified against a live trace where every surviving hit for an
    uncovered goal was exactly this shape (dense: 0, keyword-only,
    fused <= 1).

    Deliberately checked BEFORE the LLM call, not after: skipping the call
    entirely means an off-topic run escalates one gather cycle earlier,
    and never hands the model a prompt built to produce exactly the
    failure mode being guarded against.

    FOUR CONDITIONS, and each one is a live defect this guard once caused:

    - `not state.human_guidance`. A human who redirects after exactly this
      guard raised E2/E3 is giving the system new information the
      retrieved evidence never had -- e.g. compare social and political
      aspects after an economic query came back weak. Applying the guard
      again after a redirect made it fire identically and silently ignore
      the guidance, re-raising the SAME escalation with the SAME reason: a
      human's explicit instruction to try something new was treated as
      more of the evidence that caused the original failure. The guard
      exists to stop the MODEL from free-associating off weak evidence; it
      was never meant to override a HUMAN telling it what to search for.
    - `not starving`. A goal that retrieved NOTHING is a retry candidate,
      never a reason to give up: it has no misleading tail to
      free-associate off, and a different query formulation is the entire
      remedy. Live, run p205.68-check: g2 retrieved zero items, which made
      the strong-evidence test vacuously False and fired this guard on the
      exact case it was never written for.
    - `ladder_exhausted`. D-38: with the model tier wired, "no strong
      evidence" is never a reason to STOP -- there is always another tier
      that has not been tried for the goals still uncovered. Giving up
      here is what turned a retrieval limitation into "this cannot be
      answered".
    - `iteration_depth > 1`. D-3/D-14: the depth budget is this loop's
      bound, and this guard must not pre-empt it on the FIRST cycle. Runs
      p205.66/67/68-check all ended at "iterations": 1 with MAX_DEPTH
      entirely unused, because this fired before the gap generator had
      produced a single new query. Give the loop one real gap cycle; if
      the evidence is STILL only single-leg afterwards, the corpus
      genuinely lacks the topic.
    """
    uncovered = [g for g in state.goals if not g.covered]
    weak_only = [g for g in uncovered
                 if _goal_has_any_evidence(g.goal_id, state.evidence)]
    starving = [g for g in uncovered if g not in weak_only]
    ladder_exhausted = not settings.model_knowledge_enabled
    if not (ladder_exhausted and weak_only and not starving
            and not state.human_guidance
            and state.iteration_depth > 1
            and not any(_uncovered_goal_has_strong_evidence(g.goal_id,
                                                            state.evidence)
                        for g in weak_only)):
        return None
    log_event(logger, "node.gaps_skipped_no_strong_evidence",
              uncovered_goals=[g.goal_id for g in weak_only],
              depth=state.iteration_depth)
    update = {"pending_tasks": [], "human_guidance": ""}
    if state.recall_score < settings.recall_target:
        trigger = "E2" if any(g.contested for g in state.goals) else "E3"
        # D-23 bound — see agents/escalation.py::raise_or_log. Without it
        # this guard is the single most reachable infinite-escalation
        # source in the graph: it fires on the FIRST gather cycle for any
        # query the corpus does not genuinely cover, and a human redirect
        # that fails to produce tasks lands right back on it.
        update.update(raise_or_log(
            state, settings, trigger,
            reason="no_strong_evidence_for_any_uncovered_goal"))
    return update


def _select_target_goals(state: ResearchState, settings: Settings) -> List[Goal]:
    """WHICH goals this cycle is actually for (D-59, S-19).

    Uncovered goals are the original and still the common case -- but they
    are not the only way gap_generator_node is reached. D-47's
    grounded-convergence gate routes here with recall ALREADY at target
    and every goal `covered`, purely because the coverage came from
    web/model evidence rather than a real document. In that state
    `uncovered` is empty, and the prompt used to render "Uncovered goals:
    (none)" while still demanding queries for them.

    That is an unanswerable instruction, and the model answered it the
    only way it could: from the evidence tail, which is the longest and
    most topically coherent block left in the prompt. Live (run
    p205.203-check) the tail was dominated by off-topic Redis corpus hits
    under an India-vs-US query, and the gap generator returned six
    consecutive Redis/Memcached queries. The model was not
    free-associating; it was reading the only subject the prompt still
    showed it.

    Naming the ungrounded goals instead gives the cycle the job it was
    actually dispatched for: find a DOCUMENT for a goal currently propped
    up by weaker provenance.
    """
    uncovered = [g for g in state.goals if not g.covered]
    if uncovered:
        return uncovered
    goal_terms = {g.goal_id: distinctive_terms(g.description)
                  for g in state.goals}
    return [g for g in state.goals
            if not has_grounded_evidence(g.goal_id,
                                         goal_terms.get(g.goal_id, set()),
                                         state.evidence,
                                         settings.min_evidence_score)]


def _escalation_for_empty_backlog(state: ResearchState, settings: Settings
                                  ) -> Dict[str, Any]:
    """E2/E3 when the model produced no usable tasks (S-19). Returns the
    keys to merge into the node's update, or {} when nothing should fire.

    D-23 (refined by test evidence): E3 originally guarded only the
    depth-exhaustion exit; a run can ALSO fail to converge by running out
    of producible tasks (the D-14 dispatch exit). Both are "cannot
    converge below target" — so the trigger is raised here too.
    P2-09: same disabled-mode logging parity as progress_checker_node.
    """
    if state.recall_score >= settings.recall_target:
        return {}
    trigger = "E2" if any(g.contested for g in state.goals) else "E3"
    if state.human_guidance:
        # A human redirected INTO this call and the guidance they gave
        # produced nothing that survived cap_and_filter (D-2 dedup / D-16
        # depth gate). Re-raising here asks them the identical question,
        # with the identical payload, having silently consumed the answer
        # they already gave — which is what "redirect does nothing" looked
        # like from outside. Fall through to the compiler instead: an
        # empty backlog is D-1's own terminal exit, and the report will
        # say honestly what was and was not retrieved. Deliberately NOT
        # raise_or_log here -- this suppression has nothing to do with the
        # D-23 review budget, so it does not belong to that helper's
        # suppressed/stub shape.
        log_event(logger, "escalation.suppressed", level=logging.WARNING,
                  trigger=trigger, reason="redirect_produced_no_tasks")
        return {}
    return raise_or_log(state, settings, trigger,
                        reason="task_supply_exhausted")


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

        A THIRD way a run fails to converge, found live and closed here:
        every uncovered goal's evidence is real but SINGLE-LEG (BM25 has no
        relevance floor -- MIN_SIMILARITY only gates the dense leg -- so a
        generic word in an off-topic query can still keyword-match the
        corpus). This node used to hand that evidence to the model as
        context regardless, and the model would write new queries themed
        on whatever subject that evidence happened to be about rather than
        on the actual uncovered goal -- e.g. asked to compare India and the
        US, it wrote "Redis and Memcached licensing models" because that
        was the only text in the tail. Those new queries then matched the
        (irrelevant) corpus at high confidence, recall cleared target, and
        the run reported success on a topic the corpus never covered. Both
        the compiler's grounding rule (Item 11) and the critic caught this
        after the fact; this closes it before it costs an LLM call and a
        gather cycle. See _uncovered_goal_has_strong_evidence below.
        """
        router.set_node("gap_generator")
        if debug:
            log_event(logger, "node.enter", node="gap_generator")

        # S-19: three decisions, three functions above. What is left here
        # is the ONE LLM call this node exists to make, and the sequence
        # around it -- which is what the docstring already describes.
        early_exit = _no_strong_evidence_exit(state, settings)
        if early_exit is not None:
            return early_exit

        target_goals = _select_target_goals(state, settings)
        if not target_goals and not state.human_guidance:
            # Nothing uncovered and nothing ungrounded: there is no gap for
            # this node to close, so there is no prompt worth paying for.
            # D-1's empty-backlog exit routes to the compiler, which is the
            # correct terminal move for a run that has genuinely converged.
            log_event(logger, "node.gaps_skipped_nothing_to_target",
                      depth=state.iteration_depth)
            return {"pending_tasks": [], "human_guidance": ""}

        # P2-14 (D-25): same reasoning as task_expander_node's identical
        # line -- settings.mcp_enabled IS the "is mcp available" signal,
        # reused directly rather than a second, separately-configured flag.
        available_tool_hints = frozenset({"mcp"}) if settings.mcp_enabled else frozenset()
        result = router.complete_json(templates.generate_gaps(
            state.goals, state.evidence, state.iteration_depth, settings.max_fanout,
            guidance=state.human_guidance, available_tool_hints=available_tool_hints,
            query=state.raw_query, target_goals=target_goals))
        # P2-06: same validated cap_and_filter seam task_expander_node uses.
        tasks, rejected = cap_and_filter(result.get("tasks", []), state,
                                         depth=state.iteration_depth,
                                         max_fanout=settings.max_fanout,
                                         allowed_tool_hints=available_tool_hints)
        log_event(logger, "node.gaps", produced=len(tasks), rejected=rejected)
        counters = {"llm_node_calls": 1, **router.drain_counters()}
        if rejected:
            counters["producer_rejects"] = float(rejected)
        update = {"pending_tasks": tasks, "human_guidance": "", "counters": counters}
        if not tasks:
            update.update(_escalation_for_empty_backlog(state, settings))
        return update

    return gap_generator_node
