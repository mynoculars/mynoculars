# Design Decisions Log

Consolidated D-1…D-32. Sourced from code comments (`grep -rn "D-[0-9]" src/`) and
`design/Research_Agent_Design.md` §10 (which only ever logged D-27…D-30 as a
table — D-1…D-26 previously existed only scattered across module docstrings,
which is the gap this file closes). Populated as part of **P2-09**; previously
0 bytes despite being referenced by name in `internal/PHASE-2_PLAN.md` and
several source comments.

Numbers with no entry below (D-7, D-9, D-10, D-11) are referenced only as part
of a grouped citation in code (e.g. "D-3/D-4/D-9/D-13/D-17/D-22" for "graph
bounds") with no independent rationale text found in any file in this repo —
consult `design/Research_Agent_Design.md` for those, or treat their absence
here as a documentation gap in its own right.

| ID | Decision | Where it lives |
|---|---|---|
| **D-1** | `dispatch_tasks` never returns an empty `Send` list — an empty backlog always falls through to `compiler` instead of silently halting the graph. | `orchestration/graph.py` |
| **D-2** | Task dedup: `pending_tasks` is replace-on-write (no reducer — one producer per superstep); `completed_task_keys` is the dedup guard, not the backlog itself. | `state.py`, `agents/task_utils.py` |
| **D-3** | `iteration_depth` is the gather loop's only clock; ticked exactly once per cycle, in `progress_checker_node`. | `agents/gathering.py` |
| **D-4** | `recall_target` is the convergence bound the gather loop measures itself against. | `config.py`, `agents/gathering.py` |
| **D-5** | Every field a fanned-out worker writes must carry an `Annotated[...]` reducer; a reducerless field written by two parallel nodes in one superstep raises `InvalidUpdateError`. Made impossible by construction, not by testing. | `state.py` |
| **D-6** | Workers receive `WorkerPayload` (one `SearchTask`), never the full `ResearchState` — a worker cannot see other workers' tasks or unrelated state. | `state.py`, `orchestration/graph.py::dispatch_tasks` |
| **D-8** | Durable checkpointing: `PostgresSaver` when reachable, `MemorySaver` fallback otherwise, so a bare laptop still runs with zero infrastructure. `recursion_limit` is the invoke-time backstop. | `storage/postgres.py`, `config.py` |
| **D-12** | Telemetry only aggregates counters other nodes already recorded — it never invents a figure. (P2-07 extends this to provider-level counters without changing the principle.) | `agents/compilation.py::telemetry_node` |
| **D-13** | Cap-at-production: the task producer (not the dispatcher) ranks by priority and keeps only the top `max_fanout` — `graph.py`'s `dispatch_tasks` always sends everything it's given. | `agents/task_utils.py::cap_and_filter` |
| **D-14** | Convergence routing reads only `recall_score`/`iteration_depth`/`escalation_trigger` — never the backlog, which is stale by the time routing runs. | `orchestration/graph.py::route_convergence` |
| **D-15** | Worker return-key whitelist enforced at runtime (`WORKER_WRITABLE_KEYS`) — turns a rare concurrent `InvalidUpdateError` into a deterministic, first-run failure. | `orchestration/contracts.py` |
| **D-16** | Failed tasks are recorded separately from completed ones; a task that failed at depth *d* may be re-emitted only at depth > *d* — "failure is data," never silently retried in a tight loop or permanently burned. | `state.py`, `agents/gathering.py::search_worker`, `agents/task_utils.py` |
| **D-17** | Quality-gated coverage: a goal counts as covered only if it has evidence scoring **strictly above** `min_evidence_score` (P2-01 changed this from an inert `>= 0.0`). | `agents/gathering.py::progress_checker_node` |
| **D-18** | A goal with an unresolved contradiction is `contested` and cannot be `covered`, regardless of evidence volume — drives the gap generator toward adjudicating evidence. Detection machinery is wired; the semantic detector itself is P2-12 (not yet built — see Phase 2 backlog). | `agents/gathering.py::merger_node` |
| **D-19** | Telemetry counters must be monotonic countables only (call counts, flags) — never durations, which don't merge meaningfully across parallel workers. | `state.py::merge_counters` |
| **D-20** | Run identity is a `thread_id`; reusing an old one resumes/continues that checkpoint. | `cli.py`, `api/server.py` |
| **D-21** | Zero composed goals is a **legal** output, not an exception — routed to an explicit "planning failed" report (optionally via HITL's E1) rather than a `KeyError` or silent empty run. | `agents/planning.py::goal_manager_node` |
| **D-22** | Bounded critique loop: the critic's notes travel into the next compile pass (grounded rewrite, never a blind retry), bounded by `max_revisions`. | `agents/compilation.py::compiler_node`, `critic_node` |
| **D-23** | Human-in-the-loop escalation on four triggers (E1 zero goals, E2 contested non-convergence, E3 exhausted non-convergence, E4 critique exhaustion), off by default (`HITL_ENABLED=false`). P2-09 added disabled-mode `escalation.stub` logging for E2/E3, matching E1/E4's pre-existing parity. | `agents/escalation.py`, `agents/gathering.py`, `agents/planning.py`, `agents/compilation.py` |
| **D-24** | Cross-run semantic memory: retrieval reranked by similarity × volatility-aware decay (never a hard TTL); only evidence from a **passed** critique is written back. | `memory/semantic_memory.py` |
| **D-25** | Typed specialist workers: a `SearchTask.tool_hint` field (default `""`) lets a task-producing node route a task to a NAMED specialist worker instead of the default corpus search — today the only specialist is `"mcp"` (P2-13's tool), gated behind `settings.mcp_enabled`. `task_utils.py::cap_and_filter` is the only place a raw hint is ever validated against what's actually wired into the run; `orchestration/graph.py::dispatch_tasks` trusts it and just routes. Implemented, P2-14. | `state.py`, `agents/task_utils.py`, `orchestration/graph.py`, `cli.py` |
| **D-26** | Retrieval tools are mediated through a plain callable seam today (`ToolFn`) — the original design's MCP stand-in (D-30) is now the SAME seam, not a replacement of it: `tools/corpus_search.py` and `tools/mcp_client.py` are both `ToolFn` implementations, chosen (and, since P2-14, combined) at `cli.py`'s wiring layer. Implemented, P2-13. **Known unresolved issue**: `scripts/mcp_corpus_server.py`'s synchronous tool handler blocks FastMCP's single event loop per call (no thread offload) — concurrent MCP calls fully serialize under real load; see `README.md` Limitations #6 / `OPERATIONS.md` Troubleshooting. Off by default, not a correctness bug. | `tools/corpus_search.py`, `tools/mcp_client.py`, `scripts/mcp_corpus_server.py` |
| **D-27** | Memory retrieval fusion + decay runs server-side in Qdrant (`FormulaQuery` + `ExpDecayExpression`, mandatory payload indexes), gated by `settings.memory_server_side_decay` (off by default). The original Python path (`memory/semantic_memory.py::decay_factor`) is kept permanently as the parity oracle, never removed — see that module's own docstring. Implemented, P2-10. | `storage/qdrant_store.py`, `memory/semantic_memory.py` |
| **D-28** | Interrupt idempotency: an interrupting node **re-executes from its top** on resume — it does not resume mid-function. No non-idempotent effect may precede `interrupt()`; `escalation_history` is appended only in the resume-path update for exactly this reason. Resume taxonomy: approve / redirect / abort. | `agents/escalation.py` |
| **D-29** | `model_config = ConfigDict(extra="forbid")` on every state model — schema-level pollution defense at construction time, complementing D-15's runtime worker-return whitelist. Two layers, two distinct failure modes. | `state.py` |
| **D-30** | MCP transport policy (stdio implemented for local servers; Streamable HTTP for remote servers is a documented future variant, not yet built; SSE prohibited outright) plus tool-security invariants (explicit per-server env allowlist — `tools/mcp_client.py::_build_subprocess_env` never forwards `os.environ`, `AsyncExitStack` lifecycle for the stdio subprocess). Implemented (stdio), P2-13. | `tools/mcp_client.py` |
| **D-31** *(proposed, P2-03)* | Store writes should carry stable, content-derived identity rather than a fresh `uuid4()` per call — re-ingesting unchanged content should overwrite in place, not accumulate duplicates. Implemented for `QdrantStore.upsert_texts` via an optional `id_fn` parameter (default preserves old behaviour). Not yet formally added to the design doc's decision log. | `storage/qdrant_store.py` |
| **D-32** *(proposed, P2-04)* | Provider output normalization happens at the client boundary (`llm/client.py`) — chat-template sentinels are stripped/truncated there, so nodes and the router never see transport or template artefacts. Covers both the JSON path (`_extract_json`) and the free-text path (`_truncate_at_sentinel`). Not yet formally added to the design doc's decision log. | `llm/client.py` |

## Not yet decided / explicitly deferred

See `internal/PHASE-2_PLAN.md`'s "Explicitly not planned" table for the full,
reasoned list (dead-code-safe rewrite of `rrf_fuse`, `decay_factor` retention
as a parity oracle, no internal worker retry, no dynamic/LLM-decided control
flow, production concerns out of scope). Not duplicated here to avoid the two
lists drifting apart — that table is the source of truth for "won't do."
