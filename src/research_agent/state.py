"""
state.py — Graph state, domain entities, and concurrency-safe reducers.

Purpose:
    Single source of truth for everything the LangGraph workflow reads and
    writes, plus the reducer functions that make parallel worker writes safe.

Responsibilities:
    - Define domain entities (Goal, SearchTask, Evidence).
    - Define the shared graph state (ResearchState) and the worker payload.
    - Provide reducers for every field that parallel workers write (design
      decision D-5: a same-superstep write to a reducerless field raises
      LangGraph's InvalidUpdateError — only under parallel load, which is
      exactly why it must be impossible by construction, not by testing).

Design decisions:
    - Pydantic models with extra="forbid" (D-29): blocks accidental state
      pollution at object-construction time. Complements the runtime worker
      whitelist in orchestration/contracts.py (D-15) — two layers, two
      distinct failure modes.
    - pending_tasks is replace-on-write, NOT reducer-backed (D-2): each
      producer emits a fresh, capped, ranked backlog; the dedup key sets are
      the guard against re-execution, not backlog bookkeeping.
    - failed tasks are recorded separately from completed ones (D-16) so a
      transient retrieval failure does not permanently burn a query
      formulation: the gap generator may re-emit a failed key at a strictly
      greater depth.
"""

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def merge_key_sets(a: Set[str], b: Set[str]) -> Set[str]:
    """Set-union reducer for task dedup keys. Associative and commutative,
    so parallel worker writes merge safely in any order."""
    return (a or set()) | (b or set())


def merge_failed_keys(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """Merge failed-task records, keeping the DEEPEST failure depth per key.

    Why max: re-emission is allowed only at a depth strictly greater than the
    failure depth (D-16). Taking max is conservative — it never permits an
    earlier retry than any single worker observed — and keeps the reducer
    associative/commutative.
    """
    out = dict(a or {})
    for k, depth in (b or {}).items():
        out[k] = max(out.get(k, -1), depth)
    return out


def merge_counters(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Additive merge for telemetry counters (D-12/D-19).

    Counters must contain MONOTONIC COUNTABLES ONLY (call counts, token
    counts, flags). Never durations: two parallel workers writing 150ms and
    200ms would 'merge' to 350ms of nothing. Durations belong in log lines.
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------


class Volatility(str, Enum):
    """How quickly a remembered fact goes stale (D-24). Drives decay rate."""

    STABLE = "stable"            # historical facts, established patterns
    SEMI_STABLE = "semi_stable"  # org structure, schema versions
    VOLATILE = "volatile"        # prices, live status, personnel


class Goal(BaseModel):
    """One research goal derived from the user query."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    description: str
    covered: bool = False
    # D-18: a goal with an unresolved contradiction is contested and counts
    # as NOT covered — which automatically drives the gap generator to seek
    # adjudicating evidence.
    contested: bool = False


class SearchTask(BaseModel):
    """One unit of retrieval work, produced by the expander or gap generator."""

    model_config = ConfigDict(extra="forbid")

    key: str                      # stable dedup identity (D-2)
    query: str
    goal_id: str
    priority: int = 0             # D-13: producers rank; top MAX_FANOUT survive
    depth: int = 0                # echoed into failed-key records (D-16)


class Evidence(BaseModel):
    """One retrieved fact, from a live tool or from long-term memory."""

    model_config = ConfigDict(extra="forbid")

    task_key: str
    goal_id: str
    source: str                   # "corpus", "memory", ...
    content: str
    score: float = 0.0            # relevance; consumed by coverage rule (D-17)
    volatility: Volatility = Volatility.SEMI_STABLE
    contradicts: Optional[str] = None  # task_key of conflicting evidence (D-18)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ResearchState(BaseModel):
    """Shared state for the whole workflow.

    Concurrency rule (D-5): every field a fanned-out worker writes carries a
    reducer (the Annotated[...] fields below). Everything else is written by
    exactly one node per superstep and needs none.
    """

    model_config = ConfigDict(extra="forbid")

    # Inputs
    raw_query: str
    thread_note: str = ""

    # Planning
    classification: Dict[str, Any] = Field(default_factory=dict)
    goals: List[Goal] = Field(default_factory=list)
    planning_error: Optional[str] = None      # D-21

    # Work management
    pending_tasks: List[SearchTask] = Field(default_factory=list)  # replace-on-write (D-2)
    completed_task_keys: Annotated[Set[str], merge_key_sets] = Field(default_factory=set)
    failed_task_keys: Annotated[Dict[str, int], merge_failed_keys] = Field(default_factory=dict)

    # Results (parallel writers -> reducer-backed)
    evidence: Annotated[List[Evidence], operator.add] = Field(default_factory=list)

    # Loop control
    iteration_depth: int = 0                  # D-3: checker increments
    recall_score: float = 0.0

    # Compile & critique (D-22)
    final_report: str = ""
    critique_notes: Annotated[List[str], operator.add] = Field(default_factory=list)
    revision_count: int = 0
    critique_passed: bool = False

    # Telemetry (D-12/D-19)
    counters: Annotated[Dict[str, float], merge_counters] = Field(default_factory=dict)
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class WorkerPayload(BaseModel):
    """The payload delivered to each fanned-out search worker via Send (D-6).

    Workers receive THIS, not the full ResearchState — and may return only
    reducer-backed ResearchState keys (enforced in orchestration/contracts.py).
    """

    model_config = ConfigDict(extra="forbid")

    task: SearchTask
