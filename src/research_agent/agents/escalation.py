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
"""

import logging
from typing import Any, Dict, Literal

from langgraph.types import Command, interrupt

from research_agent.logging_setup import log_event
from research_agent.state import ResearchState

logger = logging.getLogger(__name__)

Destination = Literal["compiler", "gap_generator", "goal_manager", "telemetry"]


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


def build_escalation_node(settings):
    """Build the escalation node. `settings` reserved for future per-trigger
    policy (e.g. timeout) — deliberately unused today, keeping the signature
    stable across that change."""

    def human_escalation(state: ResearchState) -> Command[Destination]:
        trigger = state.escalation_trigger or "E?"
        # D-28: first effectful statement. On resume this whole function
        # re-executes; everything above this line is a pure read.
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

        # E4
        if action == "redirect":
            return Command(goto="compiler", update={
                **base, "critique_notes": [f"HUMAN REVIEWER: {guidance}"]})
        # approve and abort both ship without feeding memory (route decided
        # this before we got here); abort is distinguishable in history.
        return Command(goto="telemetry", update=base)

    return human_escalation
