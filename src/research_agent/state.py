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

If you are new to Python, read this section before the code below — every
concept it explains is used repeatedly in this file and in every node file
that reads or writes ResearchState.

    Pydantic "BaseModel"
        A Pydantic model is a class whose job is to hold DATA with type
        checking, not behaviour. You declare fields as class-level
        annotations (name: type), and Pydantic automatically writes the
        __init__ that accepts them as keyword arguments, validates their
        types at construction time, and raises a clear error if you pass
        the wrong type or a field is missing. Every class below (Goal,
        SearchTask, Evidence, ResearchState, WorkerPayload) is one of these.
        Constructing one looks like: Goal(goal_id="g1", description="...").

    model_config = ConfigDict(extra="forbid")
        A Pydantic-specific setting. By default Pydantic would silently
        accept and store extra keyword arguments you didn't declare a field
        for. "extra='forbid'" makes that an error instead — so a typo like
        Goal(goal__id="g1") (double underscore) fails LOUDLY at the moment
        you construct the object, instead of quietly creating a Goal with a
        missing goal_id and a useless extra attribute nobody reads.

    Enum (class Volatility(str, Enum): ...)
        A fixed, named set of allowed values — like an enum in Java or C#.
        Inheriting from BOTH str and Enum here means each member (e.g.
        Volatility.STABLE) behaves as a real string ("stable") wherever a
        string is expected (comparisons, JSON serialization, dict keys)
        while still being restricted to only the three values defined below.

    Optional[X]  (from the typing module)
        Shorthand for "either a value of type X, or the special value
        None". planning_error: Optional[str] = None means "this field holds
        a string once something goes wrong, and starts out as None".

    Field(default_factory=list)  /  Field(default_factory=dict)
        You cannot write "goals: List[Goal] = []" directly as a class-level
        default in Python — a single empty list object would then be SHARED
        by every instance of the class, and mutating one instance's list
        would corrupt every other instance's list too (a classic Python
        footgun). default_factory=list tells Pydantic "call list() fresh,
        once per new instance, to get its default" — so every ResearchState
        gets its OWN empty list, not a shared one.

    Annotated[SomeType, some_reducer_function]
        This is the one piece of syntax in this file that has nothing to
        do with plain Python and everything to do with LangGraph. Normally
        a type hint is just documentation for humans and type checkers —
        Python itself ignores it at runtime. LangGraph, however, actually
        INSPECTS these Annotated[...] hints on StateGraph fields: whenever
        two parallel nodes in the same "superstep" (see the reducer section
        below) both try to write to the SAME field, LangGraph looks up the
        function attached via Annotated and calls it as
        merge_function(existing_value, new_value) to decide what the
        combined value should be — instead of raising an error or silently
        picking one write and discarding the other. A field with no
        Annotated reducer can only ever be safely written by ONE node per
        superstep; if two try, LangGraph raises InvalidUpdateError.
"""

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Reducers
#
# A "reducer" here is just a plain Python function with the shape
# (existing_value, new_value) -> combined_value. LangGraph calls one of
# these automatically whenever two nodes running in the same superstep both
# write to the field it's attached to (via Annotated[...] below). None of
# these functions are called directly anywhere else in the codebase — they
# are registered as metadata on a type hint and invoked BY LangGraph.
# ---------------------------------------------------------------------------


def merge_key_sets(a: Set[str], b: Set[str]) -> Set[str]:
    """Set-union reducer for task dedup keys. Associative and commutative,
    so parallel worker writes merge safely in any order.

    CALLED BY   LangGraph itself, automatically, whenever more than one
                search_worker instance returns a "completed_task_keys" key
                in the same superstep (see agents/gathering.py). You will
                not find a direct call to merge_key_sets anywhere else.
    "|" below is Python's set-union operator: {1,2} | {2,3} == {1,2,3}.
    "(a or set())" guards against a=None, which can happen on the very
    first write when there is no existing value yet to merge with.
    """
    return (a or set()) | (b or set())


def merge_failed_keys(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """Merge failed-task records, keeping the DEEPEST failure depth per key.

    Why max: re-emission is allowed only at a depth strictly greater than the
    failure depth (D-16). Taking max is conservative — it never permits an
    earlier retry than any single worker observed — and keeps the reducer
    associative/commutative.

    CALLED BY   LangGraph itself, when more than one search_worker instance
                fails and returns a "failed_task_keys" entry in the same
                superstep.
    dict(a or {}) makes a SHALLOW COPY of the existing dict — this line does
    NOT mutate the caller's dict `a` in place; it builds a new one to return.
    ``for k, depth in (b or {}).items():`` iterates over the new dict's
    (key, value) pairs — .items() is the standard way to loop over both a
    dict's keys and values together in one go.
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

    CALLED BY   LangGraph itself, whenever more than one node in the same
                superstep returns a "counters" key — this happens on nearly
                every gather-loop cycle, since every parallel search_worker
                instance reports its own "search_calls"/"search_failures".
    out.get(k, 0) reads the current total for key k, defaulting to 0 if this
    is the first time that counter name has ever been seen.
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + v
    return out


# ---------------------------------------------------------------------------
# Domain entities
#
# These are plain data containers describing the "nouns" of a research run:
# a Goal to accomplish, a SearchTask to execute, and a piece of Evidence
# found along the way. None of them contain any logic — they are read and
# written by the node functions in agents/*.py.
# ---------------------------------------------------------------------------


class Volatility(str, Enum):
    """How quickly a remembered fact goes stale (D-24). Drives decay rate.

    Because this inherits from both str and Enum, Volatility.STABLE both
    IS the string "stable" (so it serializes to JSON cleanly and compares
    equal to "stable") AND is restricted to only these three named values.
    """

    STABLE = "stable"            # historical facts, established patterns
    SEMI_STABLE = "semi_stable"  # org structure, schema versions
    VOLATILE = "volatile"        # prices, live status, personnel


class Goal(BaseModel):
    """One research goal derived from the user query.

    Produced by agents/planning.py::goal_manager_node (the model invents
    these). Later mutated in place by two other nodes: merger_node sets
    `contested`, progress_checker_node sets `covered` — see agents/
    gathering.py for exactly how and when.
    """

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    description: str
    covered: bool = False
    # D-18: a goal with an unresolved contradiction is contested and counts
    # as NOT covered — which automatically drives the gap generator to seek
    # adjudicating evidence.
    contested: bool = False


class SearchTask(BaseModel):
    """One unit of retrieval work, produced by the expander or gap generator.

    Every SearchTask becomes exactly one parallel search_worker invocation
    (see orchestration/graph.py::dispatch_tasks, which wraps each one in a
    LangGraph `Send`) -- UNLESS tool_hint routes it to a different named
    specialist node instead (P2-14, D-25) -- see dispatch_tasks's own
    docstring for exactly how that routing decision is made.
    """

    model_config = ConfigDict(extra="forbid")

    key: str                      # stable dedup identity (D-2)
    query: str
    goal_id: str
    priority: int = 0             # D-13: producers rank; top MAX_FANOUT survive
    depth: int = 0                # echoed into failed-key records (D-16)
    # Which specialist worker handles this task, if any (D-25). "" (the
    # default) routes to search_worker. task_utils.py::cap_and_filter is
    # the only place that ever sets this to anything else, and only after
    # confirming the hint is both requested by the producer and currently
    # wired into the running graph — dispatch_tasks trusts this field
    # rather than re-validating it.
    tool_hint: str = ""


class Evidence(BaseModel):
    """One retrieved fact, from a live tool or from long-term memory.

    Producers create these with source="corpus" (tools/corpus_search.py,
    the default retrieval tool), source="mcp" (tools/mcp_client.py's
    make_mcp_tool, P2-13), source="model" (tools/model_knowledge.py, D-38's
    last tier), source="web" (tools/mcp_client.py's make_web_search_tool,
    Phase 4 / D-57), or source="memory" (memory/semantic_memory.py::retrieve,
    recalled from a past run).

    WHERE `source` IS ACTUALLY BRANCHED ON — read this before adding a new
    value, because the set membership tests below are load-bearing and are
    NOT obvious from the field's plain `str` type:

      agents/gathering.py::progress_checker_node   source in ("corpus","mcp")
      agents/compilation.py::telemetry_node        source in ("corpus","mcp")
          Both compute "is this goal backed by a real DOCUMENT" —
          grounded_score (Guardrail G2 / D-47) and corpus_recall (D-43).
          "web" is DELIBERATELY ABSENT from both. A web snippet is
          retrieval, not curation: it can COVER a goal (it counts toward
          recall) but it must not GROUND one. Keeping the two apart is the
          entire point of D-47 — a run answered wholly from the web reads
          recall 1.0 / grounded_score 0.0, which is visible rather than
          flattering. Adding "web" to either tuple would silently restore
          the recall=1.0 / corpus_recall=0.0 blindness those metrics exist
          to expose.

      memory/semantic_memory.py::store_run  source not in ("memory","model","web")
          What may enter DURABLE memory. "model" was excluded by D-42
          (recollection laundered into something a later run reads back as
          document-backed evidence); "web" is excluded for the same reason
          plus a second one — a snippet is volatile by nature, and a cached
          copy of today's search result is a stale answer tomorrow with no
          marker saying so.

    Every OTHER downstream node treats all sources identically; there is no
    per-source code path beyond the three above.
    """

    model_config = ConfigDict(extra="forbid")

    task_key: str
    goal_id: str
    source: str                   # "corpus", "mcp", "memory", ...
    content: str
    score: float = 0.0            # relevance; consumed by coverage rule (D-17)
    volatility: Volatility = Volatility.SEMI_STABLE
    contradicts: Optional[str] = None  # task_key of conflicting evidence (D-18)
    # Guardrail G3: set only by tools/model_knowledge.py, when the claim's
    # OWN text carries false-precision markers (exact years, percentages,
    # per-unit figures) the model cannot actually verify. Deterministic
    # flag, not a judgment call — see model_knowledge.py::_looks_overspecific.
    # False for every corpus/mcp/memory item; those are never flagged.
    hedge_specific: bool = False
    # Provenance for source="web" items, and only those (D-57). None for
    # everything else, rather than "", so an unset URL cannot be confused
    # with an empty one.
    #
    # WHY THESE LIVE ON Evidence: D-40 forbids URLs in report prose, but
    # attribution is not citation — the compiler still cites only [gN].
    # These two fields let a deterministic pass build a Sources section
    # from the evidence actually cited (D-51's precedent: a prompt
    # instruction to hedge was not sufficient on its own; neither would one
    # to attribute be).
    #
    # `domain` is stored, not derived, unlike websearch/provider.py's
    # WebResult. There the value lives in one process and re-deriving costs
    # nothing; here it already crossed a process boundary as JSON, and
    # re-deriving domain-from-url agent-side would be a second
    # implementation that can disagree with the server's — so the server's
    # answer is carried, not recomputed.
    url: Optional[str] = None
    domain: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ResearchState(BaseModel):
    """Shared state for the whole workflow — the ONE object every node
    function receives as its argument and returns a partial update for.

    Concurrency rule (D-5): every field a fanned-out worker writes carries a
    reducer (the Annotated[...] fields below). Everything else is written by
    exactly one node per superstep and needs none.

    HOW TO READ THIS CLASS: each field below is annotated with a comment
    naming which phase/node writes it and, where relevant, which design
    decision explains why. If you are trying to understand "who touches
    this field and when", this class plus a text-search across agents/*.py
    for the field name will answer it completely — there is no other place
    state changes happen.
    """

    model_config = ConfigDict(extra="forbid")

    # Inputs — set once, by the caller (cli.py or api/server.py), before the
    # very first node runs. Nothing inside the graph ever changes raw_query.
    raw_query: str

    # Planning — written by agents/planning.py's nodes, read by everything
    # downstream.
    classification: Dict[str, Any] = Field(default_factory=dict)
    goals: List[Goal] = Field(default_factory=list)
    planning_error: Optional[str] = None      # D-21

    # Work management — the task backlog and its bookkeeping.
    # pending_tasks has NO Annotated reducer: it is deliberately
    # REPLACE-on-write (D-2) — the producer that wrote it is always
    # rewriting the entire backlog from scratch, not adding to a shared one,
    # so only a single writer per superstep is ever expected here.
    pending_tasks: List[SearchTask] = Field(default_factory=list)  # replace-on-write (D-2)
    completed_task_keys: Annotated[Set[str], merge_key_sets] = Field(default_factory=set)
    failed_task_keys: Annotated[Dict[str, int], merge_failed_keys] = Field(default_factory=dict)

    # Results — written in parallel by every fanned-out search_worker
    # instance, hence the reducer. operator.add on two lists is just Python's
    # list concatenation ([1,2] + [3,4] == [1,2,3,4]) used as the merge rule:
    # every worker's evidence simply gets appended onto the combined list.
    evidence: Annotated[List[Evidence], operator.add] = Field(default_factory=list)

    # Loop control — the gather loop's clock and its measured progress.
    iteration_depth: int = 0                  # D-3: checker increments
    recall_score: float = 0.0
    # Guardrail G2: fraction of COVERED goals whose covering evidence
    # includes at least one corpus/mcp item (not model-only). Written by
    # progress_checker_node alongside recall_score every cycle; read by
    # route_convergence so "recall reached target" and "grounded in a
    # real document" stay two separate truths, the same way D-14 already
    # keeps recall and depth separate. 1.0 default (not 0.0) so a run
    # with zero goals — same edge case recall_score already handles —
    # never looks falsely ungrounded.
    grounded_score: float = 1.0
    # S-8: grounded_score AS READ when this cycle's progress_checker_node
    # call started -- i.e. the PREVIOUS cycle's grounded_score, not the
    # value being written this same cycle. -1.0 (outside grounded_score's
    # real [0,1] range) means "no previous cycle yet" -- route_convergence
    # uses this to give grounding exactly one gap_generator attempt before
    # comparing, rather than declaring a stall on the very first
    # below-target measurement.
    grounded_score_prev: float = -1.0

    # Human-in-the-loop escalation (D-23/D-28)
    escalation_trigger: Optional[str] = None      # E1/E2/E3/E4; set by the
    #   node whose check fired (routing fns are read-only and cannot set it)
    escalation_history: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list)  # appended in the RESUME update, never before
    #   interrupt() — the D-28 idempotency invariant made concrete
    human_guidance: str = ""                      # redirect payload for planners
    abort_reason: Optional[str] = None            # human abort -> error report

    # Compile & critique (D-22)
    final_report: str = ""
    critique_notes: Annotated[List[str], operator.add] = Field(default_factory=list)
    revision_count: int = 0
    critique_passed: bool = False

    # Telemetry (D-12/D-19) — counters accumulate additively across the
    # whole run; telemetry is the one-shot final summary built from them by
    # agents/compilation.py::telemetry_node.
    counters: Annotated[Dict[str, float], merge_counters] = Field(default_factory=dict)
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class WorkerPayload(BaseModel):
    """The payload delivered to each fanned-out search worker via Send (D-6).

    Workers receive THIS, not the full ResearchState — and may return only
    reducer-backed ResearchState keys (enforced in orchestration/contracts.py).

    Deliberately tiny: a worker cannot accidentally read (or leak) any part
    of state it has no business touching, because it is never given the
    rest of state in the first place. See orchestration/graph.py::
    dispatch_tasks for exactly how each SearchTask becomes one of these.
    """

    model_config = ConfigDict(extra="forbid")

    task: SearchTask
