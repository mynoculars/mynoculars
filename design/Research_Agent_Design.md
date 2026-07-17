# Production Research Agent — Architecture & Design Document

| Field | Value |
|---|---|
| Document version | v3.1.1 |
| Status | v3.1 + diagram corrections (topology trace errors; glyph consistency) |
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

> **Scope note.** v3.1.1 remains inside the **workflow** pattern (fixed graph topology; the LLM does not choose its own control flow). Tier B dynamic supervisor delegation remains out of scope by design — see the companion dynamic-agent piece.

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
            \Present?/─────────►│ ModularCompiler  │
             \      /           │ (error rpt) ►(C) │
              \    /            └──────────────────┘
               \  /
                \/
                │ Yes
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
    │ModularCompiler│   │ typed search_worker(s) │◄────────────────┐
    │(empty rpt)►(C)│   │ via MCP (xN,D-25/D-26) │                 │
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
               │   ┌───────────────────────┐    tasks present      │
               │   │  D-1: backlog check   ├─────────►─────────────┘
               │   └───────────┬───────────┘    (loop continues)
               │               │ backlog empty
               │               ▼
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
      │          │ (revise)     ▼              ▼
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
           ReportCritic, which waves them through (nothing to judge on the
           D-21 path) — so EVERY path reaches TelemetryLogger → [END], as
           the termination proof (§6) requires.
  resume*  HumanEscalation resumes (approve / redirect / abort) at the
           check that raised it: E1 → plan anomaly check, E2/E3 →
           convergence check, E4 → critique check. abort → terminal error
           report → (C). Drawn as a footnote, not arrows: four different
           resume targets in one 2D flowchart produce false merges (D-28).
  ModularCompiler appears 3× (error / empty / main+revise) — the SAME
  node, different entry reasons.
```

### 2.2 Gather phase as Plan–Map–Reduce (imported diagram)

*Imported from the external review; terminology aligned to this document's node names. This diagram required no correction — it independently confirms the D-5 reducer requirement and D-13 producer-side planning.*

```
                +-------------------------------+
                | TaskExpander / GapGenerator   |
                | (produces ranked task list,   |
                |  ≤ MAX_FANOUT, D-13)          |
                +---------------+---------------+
                                |
                                | conditional edge returns Send() list
                                v
                     +----------+----------+
                     |  Dispatch (D-1)     |
                     +----+-----------+----+
                          |           |
         +----------------+           +----------------+
         |                                             |
         v                                             v
+--------+---------+                        +----------+-------+
| Typed worker     |                        | Typed worker     |
| (task 1, isolated|                        | (task 2, isolated|
|  WorkerSubState) |                        |  WorkerSubState) |
+--------+---------+                        +----------+-------+
         |                                             |
         |   writes ONLY reducer-backed keys (D-5/D-15)|
         +----------------+           +----------------+
                          |           |
                          v           v
                +---------+-----------+---------+
                |  Reducer-backed channels      |
                |  (evidence: concat,           |
                |   key sets: union,            |
                |   counters: additive)         |
                +---------------+---------------+
                                |
                                v
                +---------------+---------------+
                |     EvidenceGraphMerger       |
                |  (reduce: reconcile, flag     |
                |   contradictions, D-18/D-24)  |
                +-------------------------------+
```

---

## 3. Node Inventory

Unchanged from v3.0 (see v3.0 §3), with two amendments:

- **Typed workers** now carry the D-30 transport/security policy: MCP connections are stdio (local) or Streamable HTTP (remote) only; filesystem-capable tools enforce the sandbox-path invariant (§5, S-3).
- **HumanEscalation** carries the D-28 idempotency invariant: the node re-executes from its beginning on resume, so it must perform no non-idempotent side effect before `interrupt()`.

---

## 4. State Model

The v2.1/v3.0 concurrency rule stands: every fanned-out-writer field is reducer-backed, enforced at runtime by the D-15 whitelist decorator.

**New in v3.1 (D-29): schema-level defense in depth.** All state models (`ProductionGraphState`, `WorkerSubState`, domain entities) set `model_config = ConfigDict(extra="forbid")` (Pydantic V2 form — the `class Config:` form is deprecated). This blocks construction-time state pollution (unexpected keys silently absorbed into state objects) and complements, without replacing, D-15: `extra="forbid"` guards object construction; the D-15 whitelist guards worker returns. Two independent layers, two distinct failure modes.

Field reference otherwise unchanged from v3.0 §4.1. One recorded non-change: the `evidence` reducer remains plain list-concat; a dedup-merging reducer (external review's `combine_distinct` pattern) is a scoped candidate in §11, not adopted here — it requires an evidence-identity scheme first and changes reducer semantics, which is not a change to make as a side effect of a review pass.

---

## 5. Integration Seams

**S-3 Retrieval (amended by D-30).** MCP-mediated as in v3.0, now with an explicit transport and security policy:

*MCP architecture (imported diagram — corrected: the original presented HTTP+SSE as a current first-class remote transport; SSE was deprecated by MCP spec 2025-03-26 in favor of Streamable HTTP, with provider sunsets landing mid-2026):*

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
            SSE: PROHIBITED — deprecated by spec 2025-03-26 (D-30)
```

D-30 invariants on this seam:
1. **Transport:** stdio for local servers; Streamable HTTP for remote. HTTP+SSE is prohibited for new connections (deprecated; provider endpoints are being sunset through mid-2026).
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
3. Decay uses the real API shape — `ExpDecayExpression(exp_decay=DecayParamsExpression(x=DatetimeKeyExpression(datetime_key="created_at"), target=DatetimeExpression(datetime=<ISO-8601 now>), scale=<seconds>, midpoint=...))` — with `target` as a datetime, not epoch seconds.
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
2. Resume-action mapping to the standard HITL taxonomy: our `approve` = approve; our `redirect` (inject corrective guidance, re-route upstream) ≈ edit; our `abort` ≈ reject-to-terminal-error-report. The **respond** pattern (synthetic answer injected as if a tool succeeded) is explicitly excluded for anything side-effecting — a model treating a fabricated result as a real one is a correctness hazard, and this design has no read-only case that needs it.
3. Timeout policy remains **unspecified by design** (carried from v3.0) — an operational decision, not a silent default.

---

## 6. Control Flow

§6.1–6.9 unchanged from v2.1/v3.0 (dispatch, two-point termination, coverage rule, memory staleness policy, critique loop, escalation triggers, termination proof). The termination proof gains one clause: node re-execution on resume (D-28) does not affect termination — re-execution is bounded by the same escalation count that triggered it, and every resume path still routes to a compiler-reaching edge.

---

## 7. Multi-Agent Specialization — unchanged from v3.0 (Tier A; Tier B declined).

---

## 8. Observability — unchanged from v3.0 §8, plus one counter: `worker_contract_violations` (D-15 prod-mode filtered keys), which should alert at any nonzero value.

---

## 9. Configuration Reference (additions/changes in v3.1)

| Parameter | Default | Rationale |
|---|---|---|
| MCP transport policy | stdio (local) / Streamable HTTP (remote); SSE prohibited | D-30; SSE deprecated by spec 2025-03-26, provider sunsets mid-2026 |
| MCP stdio env allowlist | explicit per-server list | D-30; never forward full host environment |
| Decay weights `w_stable`/`w_semi`/`w_volatile` | ~0 / 0.05 / 0.15 (starting points) | D-27.2; keep decay subordinate to RRF-scale similarity scores |
| Decay scales | near-∞ / 90-day / 14-day half-life equivalents | Carried from v3.0, now executed server-side |
| Qdrant payload indexes | `created_at` (datetime), `volatility`, namespace key, `superseded_by` (keyword) — created **before** ingest | D-27.5 |
| State model config | `ConfigDict(extra="forbid")` everywhere | D-29 |

All v3.0 rows (MAX_REVISIONS, anomaly bounds, confidence floor, escalation timeout **unspecified**, collection scope assumption) carry forward unchanged.

---

## 10. Design Decisions Log (D-27 … D-30)

| ID | Decision | Rationale |
|---|---|---|
| **D-27** | Memory retrieval is one server-side Qdrant query: dense+sparse prefetch → nested RRF fusion → FormulaQuery volatility decay (weighted), with mandatory payload indexes | Fusion and decay execute where the data lives — one round trip, no application-side rerank pass; nested fusion is the only structure in which "$score" is actually an RRF score; unweighted decay would dominate RRF-scale similarity |
| **D-28** | Interrupt idempotency invariant: interrupting nodes re-execute from the top on resume, so no non-idempotent side effect may precede `interrupt()`; resume taxonomy mapped (approve/redirect≈edit/abort≈reject); "respond" pattern excluded | Corrects the external review's wrong "resumes from the paused frame" claim; the true semantics impose a code-level obligation this document must state, or implementers will double-fire side effects |
| **D-29** | `model_config = ConfigDict(extra="forbid")` on all state models — schema-level pollution defense complementing the D-15 runtime whitelist | Two independent layers for two distinct failure modes: construction-time pollution vs. worker-return pollution |
| **D-30** | MCP transport policy (stdio/Streamable HTTP; SSE prohibited) + tool-security invariants: join-then-resolve sandboxed paths, minimal subprocess env, AsyncExitStack lifecycle | Aligns with the current MCP spec rather than deprecated transport; the sandbox and env rules close two concrete vulnerabilities present in the external review's own "secure" reference code |

---

## 11. Known Limitations and Deferred Items

Carried forward from v3.0: contradiction *resolution*, threshold calibration (`MIN_EVIDENCE_SCORE`, `CLASSIFIER_CONFIDENCE_FLOOR`), token-usage wiring, escalation timeout policy, Qdrant collection-scope confirmation, volatility-tagging design pass, Tier B out of scope.

New candidates recorded in v3.1, deliberately **not** adopted: (a) a dedup-merging `evidence` reducer (external review's combine-distinct pattern) — requires an evidence-identity scheme first and an associativity argument before changing reducer semantics; (b) sparse-vector generation for S-6 hybrid retrieval assumes a BM25/sparse embedding pipeline at both write and query time — if the existing corpus is dense-only, hybrid retrieval needs a backfill plan or ships dense-only initially with the RRF stage as a later addition.

---

## 12. Reference Implementation

Status updated since v3.1: a **core-build reference implementation now exists** (`agentic-research-agent-core.zip`, 61 files, 22 passing tests) covering the D-1…D-24 subset — graph mechanics, reducers, D-15 runtime enforcement, semantic memory with Python-side decay, and the D-22 critique loop. MCP (D-26/D-30), HITL interrupts (D-23/D-28), typed workers (D-25), and server-side fusion/decay (D-27) remain design-level, deferred to the next build stage by explicit scoping decision. The external review's code stubs remain **not** adoptable as-is (see Appendix C and the accompanying conversation).

---

## Appendix B — Execution Model Background (imported, corrected)

*Imported from the external review; the fabricated "DeltaChannel" construct is removed. What remains is an accurate description of LangGraph's Pregel-style superstep loop, useful as onboarding background for this document's concurrency rules.*

```
+------------------------------------------------------------+
|                   Pregel superstep loop                    |
|                                                            |
|  1. PLAN:      inspect updated channels → select the       |
|                nodes subscribed to them                    |
|                                                            |
|  2. EXECUTE:   run selected nodes in parallel; their       |
|                writes are held privately — invisible to    |
|                other nodes in the same superstep           |
|                                                            |
|  3. UPDATE:    apply all held writes to the shared         |
|                channels — through the registered reducer   |
|                where one exists; a same-superstep write    |
|                collision on a reducerless channel raises   |
|                InvalidUpdateError                          |
+------------------------------------------------------------+
```

This is the runtime basis for three of this document's rules: the superstep join that makes `EvidenceGraphMerger` a barrier without a barrier node (§2.1), the D-5 reducer requirement, and the D-15 enforcement rationale (the collision only manifests when two writers share a superstep — i.e., under parallel load). Checkpoints serialize channel values per superstep under the run's `thread_id`; resume semantics for interrupted nodes are as specified in D-28.

---

## Appendix C — External Review Disposition (second round, v3.0 → v3.1)

| External claim / artifact | Verdict | Action |
|---|---|---|
| Pregel Plan/Execute/Update superstep model | Correct | Imported as Appendix B |
| "DeltaChannel" state-history construct | **Fabricated** — no such LangGraph construct | Removed from all imported material |
| HITL resume "restores cursor, resumes from paused frame" | **Wrong** — interrupted node re-executes from its beginning; `interrupt()` returns the resume payload on re-execution | Corrected; elevated to invariant D-28 |
| HTTP+SSE as current remote MCP transport | **Outdated** — deprecated by spec 2025-03-26 for Streamable HTTP; provider sunsets mid-2026 | Corrected; transport policy D-30 |
| MCP roles, handshake, stdio-vs-remote tradeoffs | Correct | Informed D-30; diagram imported (corrected) |
| Hybrid dense+sparse retrieval with RRF + server-side decay concept | Correct and valuable | Adopted as D-27 |
| Qdrant code: flat two-prefetch + FormulaQuery labeled "RRF" | **Wrong** — no fusion stage exists; RRF never applied | Corrected structure (nested RrfQuery prefetch) in D-27 |
| Qdrant API symbols (`DecayExpression`, `DecayFunction.EXPONENTIAL`, `DateTimeKey`; epoch-float target) | **Invented** — real API: `ExpDecayExpression`/`DecayParamsExpression`/`DatetimeKeyExpression`/`DatetimeExpression` | Real symbols specified in D-27.3 |
| Decay math (λ = −ln(midpoint)/scale; week/0.1 example) | Correct | Retained |
| Payload indexing before formula queries | Correct | Mandatory in D-27.5 |
| Path-traversal helper | **Buggy** — resolves relative input against CWD, not sandbox root | Corrected join-then-resolve invariant in D-30 |
| `extra="forbid"` state hardening | Correct (modulo deprecated `class Config` form) | Adopted as D-29 (ConfigDict form) |
| Custom reducer patterns + associativity requirement | Correct | Dedup-evidence reducer recorded as a scoped candidate (§11), not adopted |
| HITL approve/edit/reject/respond taxonomy; "respond" caveat | Correct | Mapped to our actions in D-28.2; respond excluded |
| Stdio orphan-process risk / AsyncExitStack | Correct | Lifecycle invariant in D-30.4 |
| MCP client stub forwarding full `os.environ` | **Vulnerability** in "secure" reference code | Prohibited by D-30.3 |
| HITL stub: no revision cap; feedback-ignoring regenerator | **Defective** — unbounded human-reject loop; blind retry | Rejected; our D-22/MAX_REVISIONS pattern stands |