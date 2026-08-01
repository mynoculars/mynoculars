"""
agents/escalation.py — Human-in-the-loop escalation node (D-23/D-28).

Purpose:
    One parametrized node serving all four escalation triggers. When a check
    fires (and HITL_ENABLED=true), the graph pauses via LangGraph's
    interrupt(), persists to the checkpointer under the run's thread_id, and
    resumes only when a human supplies an action.

Responsibilities:
    - Build a trigger-specific review payload (what the human sees).
    - interrupt() as the FIRST effectful statement — the D-28 invariant:
      this node RE-EXECUTES from its top on resume, so nothing
      non-idempotent may precede the interrupt. escalation_history is
      appended in the RESUME update for exactly this reason.
    - Map (trigger, action) -> Command(goto=..., update=...).

Resume actions (design D-28.2 taxonomy):
    approve  — accept the current state, continue forward
    redirect — inject human guidance, re-route to the producing node
    abort    — terminal error report, run still reaches telemetry/END

Termination note (design §6.9): a redirect can re-arm a loop (e.g. E4 ->
compiler -> critic fails -> E4 again), but every such cycle requires a
fresh human action — termination becomes human-bounded, which is the
point of escalation, not a defect.

Python mechanics used in this file, if any of this is new to you:
    Literal["a", "b", "c"]   This is a TYPE HINT, not executable logic. It
                             tells a type checker (and a human reader)
                             "this value can only ever be one of these exact
                             strings" — a lightweight enum. It changes
                             nothing at runtime; Python does not enforce it.
    interrupt(payload)       A special function from the langgraph library.
                             Calling it does two very different things
                             depending on whether the graph is running for
                             the first time or being RESUMED:
                               - first time: it pauses the ENTIRE graph run
                                 right here, saves everything to the
                                 checkpointer, and the whole invoke() call
                                 that started this run returns immediately
                                 to whoever called it (cli.py or the API).
                               - on resume: LangGraph calls this SAME
                                 function again from the top, but THIS time
                                 interrupt() does not pause — it immediately
                                 returns whatever the human supplied. See
                                 the big warning inside human_escalation
                                 below for why that matters so much.
    Command(goto=..., update={...})
                             Another langgraph type. Returning one from a
                             node is how that node tells LangGraph "don't
                             follow a normal edge — jump straight to the
                             node named in `goto`, and merge `update` into
                             state first." This is why human_escalation has
                             no lines for it in graph.py's g.add_edge(...)
                             calls — it decides its own destination in code
                             instead of through the graph's wiring.
"""

import logging
from typing import Any, Dict, Literal

from langgraph.types import Command, interrupt

from research_agent.logging_setup import log_event
from research_agent.state import ResearchState

logger = logging.getLogger(__name__)

# See the Literal[...] note above — this just documents, for humans and type
# checkers, the only four strings this node is ever allowed to return as a
# destination via Command(goto=...).
Destination = Literal["compiler", "gap_generator", "goal_manager", "telemetry"]


def escalation_allowed(state: ResearchState, settings) -> bool:
    """True if this run may pause for a human ONE more time (D-23 bound).

    CALLED BY   every node that can set state.escalation_trigger --
                planning.py (E1), gathering.py (E2/E3, two sites),
                compilation.py (E4). One helper, four call sites, so the
                bound cannot drift between triggers.
    READS       settings.hitl_enabled, settings.max_escalations,
                state.escalation_history (appended once per COMPLETED
                human decision, in human_escalation's resume update --
                see D-28, which is exactly what makes it a safe counter:
                a pause that is still waiting for an answer has not been
                counted yet, and re-execution on resume cannot
                double-count it).

    WHY THIS EXISTS: hitl_enabled alone is not a bound. graph.py's
    route_convergence and dispatch_tasks both test escalation_trigger
    BEFORE their terminal exits, so a check that re-raises the same
    trigger routes back into this node forever -- the depth budget
    (D-3/D-14) and the empty-backlog fallthrough (D-1) become
    unreachable, and the only remaining stop is recursion_limit, which
    ends the process with no report at all. A redirect that cannot
    change the condition that raised the trigger must eventually stop
    asking; this is where that stops.
    """
    if not settings.hitl_enabled:
        return False
    return len(state.escalation_history) < settings.max_escalations


def _payload_for(state: ResearchState) -> Dict[str, Any]:
    """The state slice a human needs to decide. Read-only — safe pre-interrupt."""
    trigger = state.escalation_trigger or "E?"
    base: Dict[str, Any] = {
        "trigger": trigger,
        "query": state.raw_query,
        "actions": ["approve", "redirect", "abort"],
    }
    if trigger == "E1":
        base["reason"] = "Goal composition produced zero goals."
        base["hint"] = "redirect: guidance is passed to goal composition."
    elif trigger in ("E2", "E3"):
        base["reason"] = ("Unresolved contradiction on a goal." if trigger == "E2"
                          else "Cannot converge: depth or task supply exhausted below recall target.")
        base["recall"] = round(state.recall_score, 3)
        base["uncovered_goals"] = [g.description for g in state.goals if not g.covered]
        base["hint"] = "redirect: guidance is passed to gap generation."
    elif trigger == "E4":
        base["reason"] = "Critique budget exhausted; report still failing."
        base["critique_notes"] = state.critique_notes[-5:]
        base["report_preview"] = state.final_report[:600]
        base["hint"] = "redirect: guidance is added to the critic's notes."
    return base


def build_escalation_node(settings, debug: bool = False):
    """Build the escalation node. `settings` reserved for future per-trigger
    policy (e.g. timeout) — deliberately unused today, keeping the signature
    stable across that change."""

    def human_escalation(state: ResearchState) -> Command[Destination]:
        """One node serving ALL FOUR escalation triggers. Reached from
        four different places: goal_manager (E1), progress_checker or
        gap_generator (E2/E3), critic (E4) — whichever node's check fired
        set state.escalation_trigger before routing sent us here.

        FIRST PASS (the graph just paused):
          READS   state.escalation_trigger — which of the 4 checks fired.
          CALLS   interrupt(_payload_for(state)) — this is a LangGraph
                  primitive, not a normal function call: it serializes the
                  payload to the checkpointer (Postgres, if reachable) and
                  the WHOLE INVOKE CALL RETURNS immediately with
                  "__interrupt__" in the result. Nothing after this line
                  runs yet. The CLI (or API) shows the payload to a human
                  and waits for approve/redirect/abort + optional guidance.

        SECOND PASS (a human answered, cli.py called invoke again with
        Command(resume=...) under the SAME thread_id):
          ⚠ THE WHOLE FUNCTION RE-EXECUTES FROM ITS TOP — this is not a
          resume-from-the-interrupted-line, it is a fresh call. That is
          why `trigger` is read again above rather than cached, and why
          nothing above the interrupt() line may have a side effect: it
          would fire twice (once on pause, once on resume) if it did.
          READS   the human's answer, now returned BY interrupt() instead
                  of pausing: {"action": ..., "guidance": ...}.
          WRITES  state.escalation_trigger = None   (clear it, or the same
                      check would just re-raise it and loop forever)
                  state.escalation_history += one entry  <- appended HERE,
                      in this resume-path update, and NOWHERE before the
                      interrupt() call — that placement is what makes
                      re-execution safe: if it were written before the
                      interrupt, the first (pausing) pass would already
                      have written it, and the second (resuming) pass
                      would write it AGAIN, double-counting one human
                      decision as two.
                  plus trigger-specific fields (planning_error cleared,
                  human_guidance set, abort_reason set, or critique_notes
                  seeded with "HUMAN REVIEWER: ...") depending on which of
                  the 12 (trigger x action) combinations below fired.
        NEXT    Command(goto=...) sends control directly to whichever node
                should react to the human's decision — see the branch
                table below for exactly which node, per trigger and action.
        """
        trigger = state.escalation_trigger or "E?"
        if debug:
            # Logging here (before interrupt()) is safe — it's not a state
            # write, so it doesn't threaten the D-28 idempotency invariant.
            # But because this whole function re-executes from the top on
            # resume (see the docstring above), this line WILL print twice
            # for one escalation: once when the run pauses, once again when
            # it resumes with the human's answer. That's expected, not a bug
            # — it's the same re-execution behaviour the docstring describes,
            # just made visible.
            log_event(logger, "node.enter", node="human_escalation", trigger=trigger)
        answer = interrupt(_payload_for(state)) or {}
        action = str(answer.get("action", "abort")).lower()
        guidance = str(answer.get("guidance", ""))

        log_event(logger, "escalation.resumed", trigger=trigger, action=action)
        # History lands in the RESUME update — never before interrupt() —
        # so re-execution can't double-append (D-28.1).
        base: Dict[str, Any] = {
            "escalation_trigger": None,  # clear, or routing re-fires the check
            "escalation_history": [{"trigger": trigger, "action": action,
                                    "guidance": guidance}],
        }

        if trigger == "E1":
            if action == "redirect":
                return Command(goto="goal_manager", update={
                    **base, "planning_error": None, "human_guidance": guidance})
            if action == "abort":
                return Command(goto="compiler", update={
                    **base, "abort_reason": f"Aborted at plan review. {guidance}".strip()})
            return Command(goto="compiler", update=base)  # approve error-report path

        if trigger in ("E2", "E3"):
            if action == "redirect":
                return Command(goto="gap_generator", update={
                    **base, "human_guidance": guidance})
            if action == "abort":
                return Command(goto="compiler", update={
                    **base, "abort_reason": f"Aborted at convergence review. {guidance}".strip()})
            return Command(goto="compiler", update=base)  # approve: ship partial

        if trigger not in ("E1", "E2", "E3", "E4"):
            # Defensive: an unrecognised trigger (including the "E?"
            # placeholder above, i.e. escalation_trigger was empty by the
            # time this node ran) used to fall through into the E4 block
            # below and, for approve/abort, return goto="telemetry" --
            # ending the run at END with final_report still "", which the
            # CLI prints as "(no report was produced)" and scores as exit
            # code 1. Route to the compiler instead: every escalation
            # path owes the caller a report, and compiler_node already
            # knows how to write an honest one from partial evidence.
            log_event(logger, "escalation.unknown_trigger", level=logging.WARNING,
                      trigger=trigger, action=action)
            return Command(goto="compiler", update=base)

        # E4
        if action == "redirect":
            # Route to gap_generator, NOT straight back to the compiler.
            # Live (run p205.103-check): the reviewer's guidance was "ask
            # for inputs from global watchdogs, UN reports of press
            # freedom, human rights abuses, democracy index" -- a request
            # for NEW EVIDENCE. Recompiling the same evidence block cannot
            # serve that, so revision 3 failed on exactly the same missing
            # support and re-raised E4. Guidance that asks for more
            # research has to reach retrieval. human_guidance is what
            # gap_generator reads (D-38); critique_notes still carries it
            # so the next compile sees the instruction too.
            return Command(goto="gap_generator", update={
                **base,
                "human_guidance": guidance,
                "critique_notes": [f"HUMAN REVIEWER: {guidance}"]})
        # approve and abort both ship without feeding memory (route decided
        # this before we got here); abort is distinguishable in history.
        return Command(goto="telemetry", update=base)

    return human_escalation
