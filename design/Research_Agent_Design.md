# Production Research Agent — Architecture & Design Document

| Field | Value |
|---|---|
| Document version | v3.2 |
| Status | v3.2 — Guardrails (Phases 1–3) added as a deterministic post-processing layer, D-47…D-54; logging unified onto one instrumentation path (§8); E3 semantics generalized; §2.1 topology corrected; HITL implemented |
| Date | 17 July 2026 |
| Framework | LangGraph (StateGraph, Send-based map-reduce, checkpointed, interrupt-capable) |

### Revision history

| Version | Change summary |
|---|---|
| v2.0 | Corrected design superseding the original draft; D-1 … D-12 |
| v2.1 | First external-review disposition; D-13 … D-21 |
| v3.0 | Self-critique (D-22), unified human escalation (D-23), semantic memory (D-24), Tier A multi-agent (D-25), MCP tools (D-26) |
| v3.1 | Second external-review disposition (Appendix C): D-27 … D-30 adopted; four corrected diagrams imported |
| v3.1.1 | §2.1 topology corrected: convergence "compile" branch now drawn to the main compiler (was unroutable); all former dead-end sinks (error/empty compilers, three escalation boxes) reconnected to the mandatory-sink flow via legend connectors, restoring consistency with the termination proof; goals-decision diamond de-duplicated; 3× ModularCompiler legend added. §5 Qdrant pipeline: two junction glyphs corrected. No design changes. |
| v3.1.2 | **E3 semantics generalized by implementation evidence** (core-build test): "cannot converge" now covers BOTH termination points of D-14 — depth exhaustion (raised at the convergence check) AND task-supply exhaustion (raised by the gap generator when it produces zero tasks below target; the dispatch route honors the trigger). The original E3 guarded only the depth exit, so a run dying via empty backlog below target bypassed escalation entirely. §6.8/E3 and §2.1 annotated. §12: HITL (D-23/D-28) now implemented in the core build. |
| v3.1.3 | **Logging unified onto one instrumentation path, plus a human-readable narrative view** (core-build addition, no D-number assigned yet — see §8): `tracing.py`'s separate `record_llm()`/`record_retrieval()` recorder (a second call site duplicating what `log_event()` already recorded) was retired; every module now calls `log_event()` exactly once per event, read by two independent presentation layers (machine JSON, unchanged in shape, and a new per-run narrative file). No graph topology, routing behavior, or D-xx decision changed — this is instrumentation/presentation only. |
| v3.2 | **Guardrails, Phases 1–3 (D-47…D-54)** — see §13. A new `research_agent/guardrails/` package plus checks in `agents/gathering.py`, `agents/task_utils.py`, and `agents/compilation.py::telemetry_node` close a false-convergence gap (grounded evidence must be topically relevant, not merely score above the floor), flag and then enforce hedging on model-tier claims pairing a specific year with a specific quantity, reject `gap_generator` tasks naming a nonexistent goal, and add three run-level WARNING telemetry lines (a starved retrieval floor, a failing quality judge, a high LLM call count) — deliberately observational, none are a circuit breaker. Like the D-38…D-46 range immediately before it, this range is logged in full only in `DECISIONS.md`, which remains the authoritative source for both ranges; this document records the summary and the section pointer, not a duplicate of the rationale text. No graph topology, node inventory, or existing D-xx decision changed. |

> **Scope note.** v3.1.2 remains inside the **workflow** pattern (fixed graph topology; the LLM does not choose its own control flow). Tier B dynamic supervisor delegation remains out of scope by design — see the companion dynamic-agent piece.

> **Provenance note (external review, second round).** The reviewed external document contained sound architecture-level material alongside fabricated terminology ("DeltaChannel"), a wrong HITL resume claim, outdated MCP transport guidance (HTTP+SSE), and invented Qdrant API symbols. Its diagrams are imported here only after correction; its factual errors are catalogued in Appendix C so they are not re-absorbed later.

---

## 1. Purpose and Scope

As v3.0, with two sharpened commitments: retrieval-side scoring (fusion + decay) executes inside the vector database rather than in application code (D-27), and every interrupting node carries an idempotency obligation reflecting LangGraph's actual resume semantics (D-28).

---

## 2. Architectural Overview

Phases unchanged from v3.0: **Plan** (classify → memory retrieval → goals → anomaly-checked validation → typed task expansion), **Gather** (cyclic typed-worker map-reduce with contradiction-aware merging and escalation-capable convergence), **Compile & Critique** (bounded self-revision), **Persist & Learn** (report + memory write-back).

### 2.1 Topology

```
                   [START]
                      │
                      ▼
            ┌────────────────────────┐
            │   SemanticClassifier   │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │    MemoryRetriever     │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │      GoalManager       │
            └───────────┬────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   E1: plan anomaly check (D-23)   │
        └────────┬──────────────────┬───────┘
                 │                  │ escalates
                 ▼                  ▼
                /\             ┌─────────────────────┐
               /  \            │  HumanEscalation    │
              /    \           │   (E1)      resume* │
             /      \          └─────────────────────┘
            / Goals  \   No     ┌──────────────────┐
            \Present?/─────────►│   ErrorReporter  │
             \      /           │  (logs anomaly,  │
              \    /            │   skips to END)  │
               \  /             └────────┬─────────┘
                \/                       │
                │ Yes                    ▼
                │                   ┌───────────────┐
                │                   │TelemetryLogger│
                │                   │  (error flag) │
                │                   └───────┬───────┘
                │                           ▼
                │                         [END]
                ▼
        ┌────────────────────────┐
        │      TaskExpander      │
        │  (assigns worker_type) │
        └───────────┬────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │        D-1: backlog check           │
    └──────┬──────────────────────┬───────┘
           │ backlog empty        │ tasks present
           ▼                      ▼
    ┌───────────────┐   ┌────────────────────────┐
    │ErrorReporter  │   │ typed search_worker(s) │◄────────────────┐
    │(empty rpt)►END│   │ via MCP (xN,D-25/D-26) │                 │
    └───────────────┘   └───────────┬────────────┘                 │
                                    │ (superstep join)             │
                                    ▼                              │
                     ┌─────────────────────────┐                   │
                     │  EvidenceGraphMerger    │                   │
                     │ (incl. memory conflicts)│                   │
                     └───────────┬─────────────┘                   │
                                 │                                 │
                                 ▼                                 │
                     ┌───────────────────────┐                     │
                     │  GoalProgressChecker  │                     │
                     └───────────┬───────────┘                     │
                                 │                                 │
                                 ▼                                 │
          ┌──────────────────────────────────────────┐             │
          │     E2/E3: convergence (D-14/D-23)       │             │
          └────┬─────────────┬───────────────┬───────┘             │
               │ compile     │ expand        │ escalates           │
               │             │               ▼                     │
               │             │      ┌─────────────────────┐        │
               │             │      │  HumanEscalation    │        │
               │             │      │  (E2/E3)    resume* │        │
               │             │      └─────────────────────┘        │
               │             ▼                                     │
               │   ┌─────────────────────┐                         │
               │   │ DynamicGapGenerator │                         │
               │   └──────────┬──────────┘                         │
               │              │                                    │
               │              ▼                                    │
               │   ┌───────────────────────┐  tasks ARE present    │
               │   │  D-1: backlog check   ├─────────►─────────────┘
               │   └───────────┬───────────┘    (loop continues)
               │               │ backlog empty
               │               │
               └──────────────►┤
                               │
                               ▼
                     ┌───────────────────┐
      ┌─────────────►│  ModularCompiler  │
      │              └─────────┬─────────┘
      │                        │
      │                        ▼
      │              ┌─────────────────┐
      │      (C)────►│  ReportCritic   │
      │              └────────┬────────┘
      │                       │
      │                       ▼
      │       ┌────────────────────────────────────┐
      │       │        D-22: critique check        │
      │       └──┬──────────────┬──────────────┬───┘
      │          │ FAIL, budget │ FAIL, budget │ PASS
      │          │ remains      │ exhausted    │
      │          │ (revise,     │              │
      │          │  decr cnt)   ▼              ▼
      └──────────┘   ┌───────────────────┐ ┌──────────────┐
                     │  HumanEscalation  │ │ MemoryWriter │
                     │   (E4)    resume* │ │    (D-24)    │
                     └───────────────────┘ └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │TelemetryLogger│
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                                [END]

Legend
  (C)      Error/empty-report compilers re-enter the main flow at
           ReportCritic, which detects "nothing to judge" (empty/error report)
           and BYPASSES D-22 critique check — routing directly to MemoryWriter
           → TelemetryLogger → [END]. E4 escalation ONLY triggers from D-22
           when a non-empty report fails critique with remaining budget.
  resume*  HumanEscalation resumes (approve / redirect / abort) at the
           check that raised it: E1 → plan anomaly check, E2/E3 →
           convergence check, E4 → critique check. abort → terminal error
           report → (C). Drawn as a footnote, not arrows: four different
           resume targets in one 2D flowchart produce false merges (D-28).
		   * Resume re-executes the interrupted node (LangGraph semantics, D-28)
  ModularCompiler appears 3× (error / empty / main+revise) — the SAME
  node, different entry reasons.
```

*v3.1.2 note:* the post-`DynamicGapGenerator` D-1 dispatch check additionally
routes to `HumanEscalation` (E2/E3) when the gap generator produced zero tasks
while below the recall target — see §6.8. Discovered during core-build testing:
without this, the D-14 dispatch exit silently bypassed escalation.

### 2.2 Gather phase as Plan–Map–Reduce (imported diagram)

*Imported from the external review; terminology aligned to this document's
 node names. This diagram required no correction — it independently confirms 
 the D-5 reducer requirement and D-13 producer-side planning.*

```
        ┌───────────────────────────────┐
        │ TaskExpander / GapGenerator   │
        │ (produces ranked tasks,       │
        │  ≤ MAX_FANOUT, D-13)          │
        └───────────────┬───────────────┘
                        │
                        │ conditional edge
                        │   returns Send() list
                        ▼
             ┌──────────┴──────────┐
             │    Dispatch (D-1)   │
             └──────────┬──────────┘
                        │
            ┌───────────┴──────────┐
            │                      │
            ▼                      ▼
     ┌──────┴─────┐          ┌─────┴─────┐
     │ Typed      │          │ Typed     │
     │ worker     │          │ worker    │
     │ (task 1,   │          │ (task 2,  │
     │ Worker-    │          │ Worker-   │
     │ SubState)  │          │ SubState) │
     └────┬───────┘          └──────┬────┘
          │                         │
          │  writes ONLY reducer    │
          │  -backed keys (D-5/D-15)│
          └─────────────┬───────────┘
                        │
                        ▼
             ┌──────────┴──────────┐
             │  Reducer-backed     │
             │  channels           │
             │  evidence : concat  │
             │  key sets: union    │
             │  counters: additive │
             └──────────┬──────────┘
                        │
                        ▼
            ┌───────────┴──────────┐
            │ EvidenceGraphMerger  │
            │ reconcile evidence,  │
            │ flag contradictions  │
            │ (D-18/D-24)          │
            └──────────────────────┘
```

---

## 3. Node Inventory

Unchanged from v3.0 (see v3.0 §3), with two amendments:

- **Typed workers** now carry the D-30 transport/security policy: MCP connections are stdio (local) or Streamable HTTP (remote) only; filesystem-capable tools enforce the sandbox-path invariant (§5, S-3). See §5 S-3 for full D-30 specification.
- **HumanEscalation** carries the D-28 idempotency invariant: the node re-executes from its beginning on resume, so it must perform no non-idempotent side effect before `interrupt()`.

---

## 4. State Model

The v2.1/v3.0 concurrency rule stands: every fanned-out-writer field is reducer-backed, enforced at runtime by the D-15 whitelist decorator. Implementation: `@reducer_whitelist` decorator in `state_manager.py` validates worker return keys against `ProductionGraphState` fields.

**New in v3.1 (D-29): schema-level defense in depth.** All state models (`ProductionGraphState`, `WorkerSubState`, domain entities) set `model_config = ConfigDict(extra="forbid")` (Pydantic V2 form — the `class Config:` form is deprecated). This blocks construction-time state pollution (unexpected keys silently absorbed into state objects) and complements, without replacing, D-15: `extra="forbid"` guards object construction; the D-15 whitelist guards worker returns. Two independent layers, two distinct failure modes.

Field reference otherwise unchanged from v3.0 §4.1. One recorded non-change: the `evidence` reducer remains plain list-concat; a dedup-merging reducer (external review's `combine_distinct` pattern) is a scoped candidate in §11, not adopted here — it requires an evidence-identity scheme first and changes reducer semantics, which is not a change to make as a side effect of a review pass.

---

## 5. Integration Seams

**S-3 Retrieval (amended by D-30).** MCP-mediated as in v3.0, now with an explicit transport and security policy:

*MCP architecture (imported diagram — corrected: the original presented HTTP+SSE as a current first-class remote transport; SSE was deprecated by MCP spec 2025-03-26 in favor of Streamable HTTP, with provider sunsets completed mid-2026):*

```
┌───────────────────────────────────────────────────────────────────┐
│ Host (this agent's typed worker node)                             │
│                                                                   │
│                               JSON-RPC 2.0                        │
│  ┌──────────────────┐           handshake        ┌────────────┐   │
│  │    MCP Client    │◄==========================►│ MCP Server │   │
│  │ (AsyncExitStack- │ initialize -> initialized  │ (tool      │   │
│  │  managed session)│                            │  provider) │   │
│  └────────┬─────────┘                            └─────┬──────┘   │
│           │                                            │          │
│           │  list_tools / call_tool(name, args)        │          │
│           ├───────────────────────────────────────────►│          │
│           │                                            │ executes │
│           │   structured tool results (text/images)    │ tool     │
│           │◄───────────────────────────────────────────┘          │
│           │                                                       │
│           ▼                                                       │
│  worker evidence assembly (D-15-validated return)                 │
└───────────────────────────────────────────────────────────────────┘

Transports: stdio (local subprocess)  │  Streamable HTTP (remote)
            SSE: PROHIBITED — deprecated by spec 2025-03-26, provider sunsets completed mid-2026 (D-30)
```

D-30 invariants on this seam:
1. **Transport:** stdio for local servers; Streamable HTTP for remote. HTTP+SSE is prohibited for new connections (deprecated; provider endpoints completed sunset mid-2026).
2. **Sandboxed paths:** any filesystem-capable tool validates paths as `(sandbox_root / user_path).resolve()` followed by a containment check against the resolved root — joining *before* resolving, because resolving a relative path standalone resolves it against the process CWD, which both breaks legitimate relative paths and misplaces the security boundary. Containment failure raises a typed security error, not a bare `ValueError`.
3. **Minimal subprocess environment:** stdio servers receive an explicit, minimal env allowlist — never a forwarded copy of the host's full environment, which would leak host secrets to every tool process.
4. **Lifecycle:** client sessions are AsyncExitStack-managed so server subprocesses cannot orphan on exception paths.

**S-6 Memory retrieval (amended by D-27).** Retrieval is now a single Qdrant query implementing hybrid fusion and volatility decay server-side:

*Retrieval pipeline (imported diagram — corrected: the original's flat two-prefetch + FormulaQuery structure never actually applies RRF; fusion must be a nested prefetch stage. Original's invented API symbols replaced with the real qdrant-client model names.):*

```
┌───────────────────────────────────────────────────────────────┐
│                        Search input                           │
│          (dense embedding  +  sparse/BM25 vector)             │
└───────────────────────────────┬───────────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   ┌──────────────────────┐            ┌──────────────────────┐
   │ Prefetch: dense      │            │ Prefetch: sparse     │
   │ (cosine, k≈100)      │            │ (BM25, k≈100)        │
   └──────────┬───────────┘            └───────────┬──────────┘
              │                                    │
              └─────────────────┬──────────────────┘
                                │
                                ▼
              ┌────────────────────────────────────┐
              │  Parent Prefetch: RrfQuery fusion  │
              │  S_RRF = Σ 1/(60 + rank_i)         │
              └─────────────────┬──────────────────┘
                                │
                                ▼
              ┌─────────────────┴──────────────────┐
              │  FormulaQuery (server-side, D-27)  │
              │  S_final = $score                  │
              │          + w_v · ExpDecay(age)     │
              │            (w_v, scale keyed on    │
              │             volatility class)      │
              │  [payload indexes REQUIRED on      │
              │   every formula field]             │
              └────────────────────────────────────┘
```

D-27 invariants on this seam:
1. Fusion (RRF) is a **nested prefetch stage**; the decay formula applies over the fused `$score` — never over unfused parallel prefetches.
2. The decay term is **weighted down** (w_v small): RRF scores are ~1/60-scale, and an unweighted [0,1] decay would let recency crowd out similarity entirely.
3. Decay uses the real API shape — `ExpDecayExpression(exp_decay=DecayParamsExpression(x=DatetimeKeyExpression(datetime_key="created_at"), target=DatetimeExpression(datetime=<Python datetime object>), scale=<seconds>, midpoint=...))` — with `target` as a Python `datetime.datetime` object (timezone-aware UTC), not a string or epoch seconds:
   ```python
   from datetime import datetime, timezone
   target = datetime.now(timezone.utc)
   # Passed directly to DatetimeExpression(datetime=target)
   # String ISO-8601 is NOT accepted
   ```
4. Volatility keys the parameters: `stable` → negligible weight/huge scale (near-flat), `semi_stable` → 90-day-half-life-equivalent, `volatile` → 14-day. (Three query variants or a payload-conditional weight; implementation's choice.)
5. **Payload indexes are mandatory** before ingest on every field the formula or filters touch (`created_at` datetime index; `volatility`, agent-namespace key, `superseded_by` keyword indexes) — unindexed formula fields degrade to full scans at scale.
6. Retrieval never hard-filters by age (v3.0 rule, unchanged): decay reranks; it does not exclude.

**S-7 Memory write** — unchanged from v3.0, plus: writes must populate every indexed payload field listed in D-27.5.

**S-8 Human escalation (amended by D-28).**

*Interrupt lifecycle (imported diagram — corrected: the original claimed resume "restores the thread cursor / resumes from the paused frame." LangGraph's actual semantics: the interrupted node RE-EXECUTES from its beginning, and `interrupt()` returns the resume payload on that second execution. The original's "DeltaChannel" replay mechanism does not exist and is removed.):*

```
┌──────────────────────────────────────────────────────────────┐
│              HumanEscalation node — 1st execution            │
│   [D-28: NO non-idempotent side effects may precede this]    │
└──────────────────────────────┬───────────────────────────────┘
                               │ interrupt(payload: trigger,
                               │           state slice, options)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  Engine: serializes state to checkpointer (thread_id),       │
│  suspends the run, releases execution resources              │
└──────────────────────────────┬───────────────────────────────┘
                               │ ... human reviews externally ...
                               ▼
┌──────────────────────────────────────────────────────────────┐
│        Command(resume=action) under the same thread_id       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│          HumanEscalation node — RE-EXECUTES from top         │
│   interrupt() now RETURNS the resume payload; routing        │
│   proceeds on human_resume_action                            │
└──────────────────────────────────────────────────────────────┘
```

D-28 invariants:
1. Code preceding `interrupt()` in any interrupting node runs **at least twice** (once per execution, plus once per resume). It must therefore be side-effect-free or idempotent — appending to `escalation_history`, for instance, happens in the *returned update* after resume, never before the interrupt.
2. Resume-action mapping to the standard HITL taxonomy: our `approve` = approve; our `redirect` (inject corrective guidance, re-route upstream) ≈ edit; our `abort` ≈ reject-to-terminal-error-report. The **respond** pattern (synthetic answer injected as if a tool succeeded) is excluded for SIDE-EFFECTING operations, but permitted for read-only classification correction (E1 redirect with human-provided intent). For side-effecting paths, human must choose approve/redirect/abort — a model treating a fabricated result as a real one is a correctness hazard.
3. Timeout policy remains **resolved per-interface** (§12): blocking stdin for the CLI; indefinite checkpointed persistence for the API.

---

## 6. Control Flow

§6.1–6.9 unchanged from v2.1/v3.0 (dispatch, two-point termination, coverage rule, memory staleness policy, critique loop, escalation triggers, termination proof). The termination proof gains two clauses:

1. Node re-execution on resume (D-28) does not affect termination — re-execution is bounded by the same escalation count that triggered it, and every resume path still routes to a compiler-reaching edge.
2. The E3 task-supply exhaustion path (v3.1.2) terminates because:
   - HumanEscalation returns approve, redirect, or abort
   - abort → ErrorReporter → TelemetryLogger → [END]
   - approve/redirect re-enter the loop with finite remaining depth
   - depth is bounded by max_depth, so loop must terminate

---

## 7. Multi-Agent Specialization — unchanged from v3.0 (Tier A; Tier B declined).

---

## 8. Observability — unchanged from v3.0 §8, plus one counter: `worker_contract_violations` (D-15 prod-mode filtered keys), which should alert at any nonzero value.

**Core-build addition (D-35, not adopted into this document's own §10 table —
same treatment as D-31/D-32/D-33/D-34: `DECISIONS.md` is a separate, later
consolidation and is the formal record for it):** an optional Langfuse tracing
layer, isolated entirely in `research_agent/langfuse/`, sits alongside the
counters above rather than replacing them. It is off by default
(`LANGFUSE_ENABLED=false`, zero SDK import, zero network calls) and adds no
new control-flow paths — every trace/span/generation/event/score call is a
side effect recorded around an existing node or provider call, never a
condition the graph branches on. See `DECISIONS.md` D-35 and the
implementation-level record in `internal/PHASE-3_LANG-FUSE-CHANGES.md` for
the full account (module layout, SDK-version reality check, instrumentation
coverage, cost calculation, and the known flat-span-tree limitation).

**Core-build addition, logging (no D-number assigned yet):** the counters
above and the Langfuse layer are both fed by `log_event()`, now the ONLY
function any module calls to record anything — `tracing.py`'s Tracer
previously had its own `record_llm()`/`record_retrieval()` methods, called
SEPARATELY from (and duplicating) the `log_event()` call already describing
the same event. That second path is retired; `Tracer` is now only the
on/off switch and flush trigger for a second presentation layer
(`logging_setup.py::NarrativeFormatter`), which renders the SAME event
stream `JsonLineFormatter` sees as a human-readable execution story — one
file per run (`logs/run-<run_id>.txt`, written only when `--debug`/
`DEBUG_TRACE=true`), with a graph-construction summary, an execution-plan
preview, per-node `INPUT`/`DECISION`/`NEXT` sections, parallel search tasks
serialized into one block per task (correlated via each event's own `task`
field, or query-string matching where the retrieval layer has no task
concept at all), sectioned telemetry, and a final request summary with
elapsed-time markers. This is presentation only — no graph topology,
routing behavior, or D-xx decision changed; see `logging_setup.py`'s own
module docstring for the full design rationale.

---

## 9. Configuration Reference (additions/changes in v3.1)

| Parameter | Default | Rationale |
|---|---|---|
| MCP transport policy | stdio (local) / Streamable HTTP (remote); SSE prohibited | D-30; SSE deprecated by spec 2025-03-26, provider sunsets completed mid-2026 |
| MCP stdio env allowlist | explicit per-server list | D-30; never forward full host environment |
| Decay weights `w_stable`/`w_semi`/`w_volatile` | ~0 / 0.05 / 0.15 (starting points) | D-27.2; keep decay subordinate to RRF-scale similarity scores |
| Decay scales | near-∞ / 90-day / 14-day half-life equivalents | Carried from v3.0, now executed server-side |
| Volatility classes | stable / semi_stable / volatile | D-27.4; keys decay parameters |
| Qdrant payload indexes | `created_at` (datetime), `volatility`, namespace key, `superseded_by` (keyword) — created **before** ingest | D-27.5 |
| State model config | `ConfigDict(extra="forbid")` everywhere | D-29 |
| MAX_REVISIONS | 3 | Hard cap on D-22 critique-revision loops including E4; decremented per E4 trigger |
| Escalation timeout | CLI: blocking stdin; API: indefinite checkpointed | §12 — resolved per-interface; blocking for CLI, persistent for API |

All v3.0 rows (MAX_REVISIONS, anomaly bounds, confidence floor, escalation timeout **unspecified**, collection scope assumption) carry forward unchanged.

**v3.2 additions** (D-47…D-54; full parameter table in `DECISIONS.md` and
`config.py`): `GROUNDED_RECALL_TARGET` (0.5), `RETRIEVAL_FLOOR_WARN_RATIO`
(0.8), `QUALITY_JUDGE_WARN_RATIO` (0.5), `RUN_CALL_BUDGET_WARN` (40),
`LLM_MAX_TOKENS` (4096), and the API's `query` length bound (1–2000
chars) — see §13.

---

## 10. Design Decisions Log (D-27 … D-30)

| ID | Decision | Rationale |
|---|---|---|
| **D-27** | Memory retrieval is one server-side Qdrant query: dense+sparse prefetch → nested RRF fusion → FormulaQuery volatility decay (weighted), with mandatory payload indexes | Fusion and decay execute where the data lives — one round trip, no application-side rerank pass; nested fusion is the only structure in which "$score" is actually an RRF score; unweighted decay would dominate RRF-scale similarity |
| **D-28** | Interrupt idempotency invariant: interrupting nodes re-execute from the beginning on resume, so no non-idempotent side effect may precede `interrupt()`; resume taxonomy mapped (approve/redirect≈edit/abort≈reject); "respond" pattern permitted for read-only E1 correction, excluded for side-effecting paths | Corrects the external review's wrong "resumes from the paused frame" claim; the true semantics impose a code-level obligation this document must state, or implementers will double-fire side effects |
| **D-29** | `model_config = ConfigDict(extra="forbid")` on all state models — schema-level pollution defense complementing the D-15 runtime whitelist | Two independent layers for two distinct failure modes: construction-time pollution vs. worker-return pollution |
| **D-30** | MCP transport policy (stdio/Streamable HTTP; SSE prohibited) + tool-security invariants: join-then-resolve sandboxed paths, minimal subprocess env, AsyncExitStack lifecycle | Aligns with the current MCP spec rather than deprecated transport; the sandbox and env rules close two concrete vulnerabilities present in the external review's own "secure" reference code |

---

## 11. Known Limitations and Deferred Items

Carried forward from v3.0: contradiction *resolution*, threshold calibration (`MIN_EVIDENCE_SCORE`, `CLASSIFIER_CONFIDENCE_FLOOR`), token-usage wiring, escalation timeout policy (now resolved per-interface in §9), Qdrant collection-scope confirmation, volatility-tagging design pass, Tier B out of scope.

New candidates recorded in v3.1, deliberately **not** adopted: (a) a dedup-merging `evidence` reducer (external review's combine-distinct pattern) — correct in principle; deferred pending evidence-identity scheme and associativity proof; (b) sparse-vector generation for S-6 hybrid retrieval — **DEFERRED to v3.2**. Ships dense-only initially with RRF stage as later addition. If the existing corpus is dense-only, hybrid retrieval needs a backfill plan or ships dense-only initially with the RRF stage as a later addition.

---

## 12. Reference Implementation

Status updated at v3.2: the reference implementation's test suite now
collects **348 tests** (see `README.md`/`OPERATIONS.md` for the full
growth history), 47 of which are the guardrails regression coverage
added for D-47…D-54 (§13). The v3.1.2 status line below is otherwise
unchanged and describes the same D-1…D-24/D-28 core. Status updated at v3.1.2: the core-build reference implementation (63 files, 28/32 tests passing [87.5%], covering D-1…D-24, D-28, and HITL. 4 tests skipped pending MCP integration [D-26/D-30].) covers D-1…D-24 together with D-28.t **plus HITL (D-23/D-28)**: a single parametrized escalation node with `interrupt()` as its first effectful statement, all four triggers (E1/E2/E3/E4, off by default via `HITL_ENABLED`), approve/redirect/abort resume under the run's thread_id, CLI stdin loop and API `/resume` endpoint. The timeout policy deferred since v3.0 is resolved per-interface: blocking stdin for the CLI; indefinite checkpointed persistence for the API. MCP (D-26/D-30), typed workers (D-25), and server-side fusion/decay (D-27) remain design-level by explicit scoping decision. The external review's code stubs remain **not** adoptable as-is (see Appendix C and the accompanying conversation).

---

## 13. Guardrails Architecture (v3.2, D-47…D-54)

A new `research_agent/guardrails/` package holds deterministic
post-processing checks — `citations.py` and `fencing.py` predate this
revision; `hedging.py` is new. The package's own module docstring states
the operating rule: check deterministically where possible, ask an LLM
(the critic, §6) only where a mechanical check genuinely cannot judge.
Every guardrail below is either a telemetry addition (a WARNING log line,
never a routing change), a flag set on existing `Evidence`, or a
rejection folded into the existing `producer_rejects` counter (D-13-
adjacent) — none of them are a new LLM call, and none change the graph
topology of §2.

**Detail is intentionally not duplicated here.** As with D-38…D-46
immediately preceding this range, full rationale, live evidence, and file
references live in `DECISIONS.md` (D-47…D-54); this section is the
summary a reader of this document needs, not a second copy of that log.

| Guardrail | Summary | DECISIONS.md |
|---|---|---|
| Grounded convergence | `route_convergence` (§6) will not accept `recall_score` reaching target as full convergence unless a second measure, `grounded_score`, also clears a configured floor — a covered goal must have at least one corpus/mcp evidence item that is both above the score floor AND topically about the goal, reusing the same topical check `corpus_recall` already applies. | D-47 |
| Retrieval-floor telemetry | A run-level WARNING when the fraction of dense retrieval candidates dropped by the relevance floor clears a configured ratio — the aggregate view of the per-query floor-drop signal that previously existed only in raw debug lines. | D-48 |
| False-precision flag | A deterministic (regex, no LLM call) check flags a model-knowledge-tier claim that pairs a specific year with a specific quantity — the shape of claim self-reported confidence does not reliably catch. | D-49 |
| Orphaned-task guard | A gap-generator task naming a goal id absent from the current run's goal set is rejected before retrieval, not after — closes a path where such evidence was retrieved, scored, and merged but could never be marked covered. | D-50 |
| Hedge enforcement | The compiled report is searched for the exact quantity span a false-precision flag (above) identified; if it survived unhedged, a visible marker is appended in place. Detection and enforcement are two separate guardrails because a prompt instruction to hedge is not reliable enough alone. | D-51 |
| API input validation | The API's `query` field gets a length bound, matching the `Field(...)`-constraint convention every other configuration value in this system already uses. | D-52 |
| Quality-judge alerting | A run-level WARNING when the self-scoring quality judge (§6, S-6-adjacent) fails on every attempt in a run — its existing fail-open design is unchanged; this only makes a 100%-failure run visible. | D-53 |
| Generation budget + call-budget observability | A token generation budget is now sent to every LLM provider on every call; a run-level WARNING (carrying revision-cycle and escalation-history counts) fires past a configured total-call threshold. Deliberately NOT an enforcement mechanism — the existing bounds (MAX_REVISIONS, escalation cap, `recursion_limit`, §9) already make an unbounded run structurally impossible. | D-54 |

---

## Appendix B — Execution Model Background (imported, corrected)

*Imported from the external review; the fabricated "DeltaChannel" construct is removed. What remains is an accurate description of LangGraph's Pregel-style superstep loop, useful as onboarding background for this document's concurrency rules.*

```
┌───────────────────────────────────────────────────────────┐
│                   Pregel superstep loop                   │
│                                                           │
│  1. PLAN:      inspect updated channels → select the      │
│                nodes subscribed to them                   │
│                                                           │
│  2. EXECUTE:   run selected nodes in parallel; their      │
│                writes are held privately — invisible to   │
│                other nodes in the same superstep          │
│                                                           │
│  3. UPDATE:    apply all held writes to the shared        │
│                channels — through the registered reducer  │
│                where one exists; a same─superstep write   │
│                collision on a reducerless channel raises  │
│                InvalidUpdateError                         │
└───────────────────────────────────────────────────────────┘	
```

This is the runtime basis for three of this document's rules: the superstep join that makes `EvidenceGraphMerger` a barrier without a barrier node (§2.1), the D-5 reducer requirement, and the D-15 enforcement rationale (the collision only manifests when two writers share a superstep — i.e., under parallel load). Checkpoints serialize channel values per superstep under the run's `thread_id`; resume semantics for interrupted nodes are as specified in D-28.

---

## Appendix C — External Review Disposition (second round, v3.0 → v3.1)

| External claim / artifact | Verdict | Action |
|---|---|---|
| Pregel Plan/Execute/Update superstep model | Correct | Imported as Appendix B |
| "DeltaChannel" state-history construct | **Fabricated** — no such LangGraph construct | Removed from all imported material |
| HITL resume "restores cursor, resumes from paused frame" | **Wrong** — interrupted node re-executes from its beginning; `interrupt()` returns the resume payload on re-execution | Corrected; elevated to invariant D-28 |
| HTTP+SSE as current remote MCP transport | **Outdated** — deprecated by spec 2025-03-26 for Streamable HTTP; provider sunsets completed mid-2026 | Corrected; transport policy D-30 |
| MCP roles, handshake, stdio-vs-remote tradeoffs | Correct | Informed D-30; diagram imported (corrected) |
| Hybrid dense+sparse retrieval with RRF + server-side decay concept | Correct and valuable | Adopted as D-27 |
| Qdrant code: flat two-prefetch + FormulaQuery labeled "RRF" | **Wrong** — no fusion stage exists; RRF never applied | Corrected structure (nested RrfQuery prefetch) in D-27 |
| Qdrant API symbols (`DecayExpression`, `DecayFunction.EXPONENTIAL`, `DateTimeKey`; epoch-float target) | **Invented** — real API: `ExpDecayExpression`/`DecayParamsExpression`/`DatetimeKeyExpression`/`DatetimeExpression` | Real symbols specified in D-27.3 |
| Decay math (λ = −ln(midpoint)/scale; week/0.1 example) | Correct | Retained |
| Payload indexing before formula queries | Correct | Mandatory in D-27.5 |
| Path-traversal helper | **Buggy** — resolves relative input against CWD, not sandbox root | Corrected join-then-resolve invariant in D-30 |
| `extra="forbid"` state hardening | Correct (modulo deprecated `class Config` form) | Adopted as D-29 (ConfigDict form) |
| Custom reducer patterns + associativity requirement | Correct in principle; deferred pending evidence-identity scheme and associativity proof | Dedup-evidence reducer recorded as a scoped candidate (§11), not adopted |
| HITL approve/edit/reject/respond taxonomy; "respond" caveat | Correct | Mapped to our actions in D-28.2; respond permitted for read-only E1, excluded for side-effecting |
| Stdio orphan-process risk / AsyncExitStack | Correct | Lifecycle invariant in D-30.4 |
| MCP client stub forwarding full `os.environ` | **Vulnerability** in "secure" reference code | Prohibited by D-30.3 |
| HITL stub: no revision cap; feedback-ignoring regenerator | **Defective** — unbounded human-reject loop; blind retry | Rejected; our D-22/MAX_REVISIONS pattern stands |
