# Agentic Research Agent — Reference Implementation (Core Build)

<p align="center">
  <img src="mynoculars_logo.png" alt="Mynoculars Logo" width="250">
</p>

A LangGraph research agent that plans goals, retrieves evidence in parallel, 
iteratively deepens coverage, and self-critiques until it produces a cited 
report. If it cannot converge, it halts for human review rather than 
hallucinating a conclusion. State and learnings persist across sessions.

This is a reference implementation of production agent architecture—graph-based 
state machines, provider fallback chains, structured observability, and 
exhaustive regression testing—not a hosted SaaS, but a demonstration of how 
to build agentic systems that degrade gracefully and improve with oversight.

> **Status:** Core build. Implements the workflow graph, hybrid retrieval,
> semantic memory, LLM fallback routing, the self-critique loop, and
> human-in-the-loop escalation. All six phases of a comprehensive review
> (`DECISIONS.md` D-1…D-79) are closed as of this revision -- correctness
> fixes, structural simplification, and operational hardening. Release
> history, prior test counts, and superseded claims from earlier revisions
> live in [CHANGELOG.md](CHANGELOG.md), not here.


## Overview

**Project goals**

1. Learn Agentic AI architecture hands-on.
2. Showcase engineering practice: bounded loops, concurrency-safe state,
   graceful degradation, honest telemetry.
3. Serve as a readable reference — every module explains *why*, not just *what*.

**Learning objectives** — after reading this codebase you should understand:
orchestration with LangGraph (nodes, edges, conditional routing, `Send`
fan-out), planning (goal decomposition, gap-driven iteration), tool calling,
hybrid retrieval (dense + BM25 + RRF), long-term memory with staleness decay,
self-evaluation, provider fallback, and interrupt/resume human-in-the-loop.

**Architecture summary** — a *workflow*, not a free-form agent: the graph
topology is fixed at build time; LLMs fill in content (goals, tasks, reports,
critiques) but never choose the control flow. That distinction is the single
most important concept here — dynamic agency is a different pattern, kept out
of this repo on purpose.

**Where this document sits.** Three companion documents exist, and this one
deliberately does not duplicate them:

| Document | Owns |
|---|---|
| `OPERATIONS.md` | Install, the L1/L2/L3 run ladder, service startup, ingest, manual test recipes |
| `internal/LEARNING_GUIDE.md` | Pedagogy — follow-one-query walkthrough, concept teaching, technical-evaluation framing (note: `internal/` is gitignored, so it ships only in archives) |
| `design/Research_Agent_Design.md` | The full target architecture and D-1…D-30 rationale — a strict superset of this build |
| **`README.md`** (this file) | What exists, how it is wired, what each store actually holds, and what is broken |

## Agent Harness

This system **is** an agent harness, in the sense the term is used across
current agentic-AI reference architectures: a runtime layer that sits
around an LLM and is responsible for state, orchestration, planning,
tool/retrieval access, model selection, verification, policy, memory,
human escalation, and observability. The LLM supplies reasoning and
content; the harness controls execution.

```text
                 ┌────────────────────────┐
                 │      Frontier LLM       │
                 │  local Cogito → Mistral │
                 │      → Gemini Flash     │
                 └───────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │      AGENT HARNESS       │
                │                          │
                │  Planner / Reasoner      │
                │  Memory                  │
                │  Tool Router             │
                │  Model Router            │
                │  Verifier                │
                │  Policy / Guardrails     │
                │  HITL                    │
                │  Observability           │
                └──────┬─────────────┬─────┘
                       │             │
                 ┌─────▼─────┐ ┌─────▼─────┐
                 │    RAG    │ │   Tools   │
                 │ Qdrant +  │ │ MCP (D-76,│
                 │ OpenSearch│ │ standalone│
                 │  + RRF    │ │  servers) │
                 └───────────┘ └───────────┘
```

**The one design principle everything else follows from:**

> **The LLM decides *what to say*. The harness decides *what happens
> next*.** Goals, task queries, report text, and critique verdicts are
> all LLM output — content the harness reads and acts on. Which node
> runs next, whether to loop again, when to escalate to a human, and
> when to stop are ALL deterministic graph routing (`orchestration/
> graph.py`'s conditional edges), never an LLM's own choice of what tool
> to call next. This is a genuinely different shape from a free-form
> autonomous agent loop, and it is deliberate — the fixed graph topology
> is the single load-bearing architectural decision this whole project
> rests on (see `internal/LEARNING_GUIDE.md` for the full argument for
> why, and D-1's own rationale in `DECISIONS.md`).

### System Architecture at a Glance

| Harness role | This project | Where |
|---|---|---|
| **Frontier / Foundation LLM** | Local Cogito (primary) → Mistral → cloud fallback, quality-gated, per-hop timeout isolation. The third slot is **named by `LLM_FALLBACK_NAME`** (D-114) — Gemini by default, Grok by uncommenting a block in `.env`; every "Gemini Flash" elsewhere in this document is that default, not a hardwire | `llm/router.py::FallbackRouter`, `llm/client.py` |
| **Planner / Reasoner** | Query classification → goal decomposition → task expansion → iterative gap-driven gathering → self-critique → compilation | `agents/planning.py`, `agents/gathering.py`, `agents/compilation.py` |
| **Memory** | Cross-run semantic memory, Qdrant-backed, volatility-aware decay, namespaced by goal | `memory/semantic_memory.py` |
| **Tool Router** | 5-tier retrieval ladder (corpus → reformulated → MCP → web → model knowledge); a task's `tool_hint` can route straight to a named specialist | `tools/retrieval_chain.py`, `agents/task_utils.py` |
| **Model Router** | The same `FallbackRouter` above, reused for every LLM call in the graph, not just retrieval | `llm/router.py` |
| **Verifier** | An LLM critic checking faithfulness/coverage against evidence, PLUS deterministic guardrails on top of it (grounded-convergence gate, the zero-citation gate, evidence-score gates) — "don't trust the LLM alone to enforce a hard rule" is a recurring pattern here | `agents/compilation.py::critic_node`, `guardrails/retrieval.py`, `orchestration/graph.py::route_convergence` |
| **Policy / Guardrails** | Depth/revision/escalation budgets, prompt/data fencing for external content, citation cleanup, hedging enforcement, dedup | `guardrails/` package |
| **HITL (Human control)** | `interrupt()`/resume, four trigger types (E1–E4: no goals, low relevance, low grounding, critique budget exhausted) | `agents/escalation.py`, `orchestration/graph.py` |
| **Observability** | Structured JSON logs, a separate human-readable execution narrative, per-run telemetry (dozens of honesty metrics), optional Langfuse tracing, CLI wall-clock/HITL reporting | `logging_setup.py`, `reporting/narrative.py`, `agents/compilation.py::telemetry_node`, `langfuse/` |
| **RAG** | Dense (Qdrant) + keyword (OpenSearch BM25), fused by RRF, with a two-stage relevance floor and a topical-overlap gate | `retrieval/hybrid.py`, `guardrails/retrieval.py` |
| **Tools** | Corpus search (in-process or MCP-mediated), web search (MCP-mediated), model's own recollection as a last-resort tier | `tools/corpus_search.py`, `tools/mcp_client.py`, `tools/model_knowledge.py` |
| **Execution** | Bounded LangGraph node execution, plus external MCP servers reached over Streamable HTTP (D-76: standalone processes you start and stop yourself, not spawned by the agent). **No general-purpose code-execution sandbox exists in this build** — that is a real, stated gap, not an oversight (see Limitations) | `orchestration/graph.py`, `tools/mcp_client.py` |
| **Persistence** | PostgreSQL (workflow checkpoints + run history), Qdrant (dense retrieval + semantic memory), OpenSearch (BM25) — three stores with three distinct jobs, not one interchangeable "database" | `storage/postgres.py`, `storage/qdrant_store.py`, `storage/opensearch_store.py` |

This table is a map, not a substitute for the sections below — each row's
"Where" column is where the real detail (and the honest caveats) live.

**What this project is / is not:**

| This project **is** | This project **is not** |
|---|---|
| An agent harness reference implementation | A hosted product or SaaS |
| A fixed-graph orchestration (deterministic routing) | A free-form autonomous agent that picks its own next action |
| Hybrid RAG (dense + BM25 + RRF) | A vector-search-only demo |
| Multi-provider model routing with quality gating | A single-model application |
| MCP-mediated tool integration | A hard-coded, unpluggable search client |
| Cross-run semantic long-term memory | Chat-turn history dressed up as memory |
| Deterministic guardrails layered on top of LLM judgment | Safety enforced by prompting alone |
| Self-critique with a human escalation path | Blind, unreviewed generation |
| Bounded, budgeted iteration (depth/revision/escalation caps) | An unbounded agent loop |
| Structured, queryable observability | `print()`-based debugging |


All five Tier 1 items and all four Tier 2 items in `internal/PHASE-2_PLAN.md` have
landed, plus one incidental fix outside either tier's original scope.
Listed here first, ahead of the architecture walkthrough, because they
change what several sections below used to say — if you've read an
earlier revision of this README, this is the fastest way to see what
moved. Tier 1's table is unchanged from the prior revision (reproduced
below for continuity); Tier 2's table follows it.

| Item | What changed | Verified how |
|---|---|---|
| **P2-04** — provider output sanitizer | `llm/client.py::_extract_json` now strips known chat-template end-of-turn sentinels (`<\|im_end\|>`, `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|endoftext\|>`, `</s>`) before parsing, and falls back to extracting the outermost `{...}` span if parsing still fails after that | Reproduced the exact failure pattern (`'{"goals": [...]}<\|im_end\|>'`) and confirmed it now parses; confirmed clean JSON and genuine garbage are unaffected. **Then confirmed live**: three separate follow-up runs against a real local Cogito model, previously failing with `JSONDecodeError` on every structured call, now succeed with zero fallback needed |
| **P2-01** — relevance floor + calibrated evidence gate | Two-stage gate, not one: `min_similarity` (new, default `0.35`) filters low-relevance dense hits *before* they enter fusion or become Evidence at all (`retrieval/hybrid.py`); `min_evidence_score` (raised from the inert `0.0` to `0.5`) still filters *after*, at the coverage-check step, as before | Constructed a fake retriever returning hits at similarity 0.9/0.4/0.1; confirmed the floor drops the 0.1 hit and keeps the other two, and that `min_similarity=0.0` reproduces the exact old (no-floor) behavior |
| **P2-02** — namespace memory `goal_id` | `memory/semantic_memory.py::retrieve` now tags every memory-sourced `Evidence` with `goal_id="memory::<original>"` instead of the bare original, so it can never satisfy a *current* run's goal by string collision | Constructed a fake memory hit tagged `goal_id="g3"` (simulating an unrelated earlier run) and confirmed the returned `Evidence.goal_id` (`"memory::g3"`) is provably `!= "g3"` |
| separate, related fix — Cogito timeout | One shared `llm_timeout_seconds` (`60.0`, applied to every provider identically) split into `llm_primary_timeout_seconds` (`120.0`, Cogito only) and `llm_timeout_seconds` (`90.0`, now Mistral/Gemini only) | Confirmed live: two calls that previously timed out at exactly 60.0s now time out at exactly 120.0s (the local model genuinely needed more room for large prompts and, apparently, a cold-start model load on the session's first call) |
| **P2-01 follow-up** — exact-boundary collision | Coverage check changed from `>=` to `>` in `agents/gathering.py::progress_checker_node`. Found live: with `RRF_SQUASH=30.0` and `RRF_K=60`, a rank-0 hit under SINGLE-LEG fusion (i.e. whenever OpenSearch is down) squashes to *exactly* `1/60 × 30 = 0.5` — a deterministic artifact of the fusion math, not a measure of relevance — and a `>=` comparison against `min_evidence_score=0.5` let that exact value through every time | Reconstructed the exact evidence shape a real run produced (`score=0.5` against a `0.5` threshold) and confirmed the goal is no longer marked covered. This closes the collision regardless of the threshold chosen — it does not replace actually fixing why OpenSearch is unreachable |
| **New — free-text runaway-generation truncation** | New `_truncate_at_sentinel()` in `llm/client.py`, applied inside `OpenAICompatibleClient.complete()` (the free-text path `compiler_node` uses). P2-04's sentinel-stripping only ever covered the JSON-mode path (`_extract_json`) — the free-text path had zero cleanup. A real run showed the local model generating 94+ seconds and 3,151+ tokens past its actual answer, hallucinating an entire fake follow-up conversation (a fabricated `system` turn, the prompt regenerated, a duplicate report) that went straight into `final_report` verbatim | Reconstructed the exact runaway pattern from a real trace (clean report → `<\|im_end\|>` → fake turns → duplicate report) and confirmed only the clean report survives. **Confirmed firing live, repeatedly** — a later real run shows `llm.truncated_runaway_generation` on `goal_manager`, `task_expander`, both `gap_generator` calls, both `compiler` calls, and `critic`; the worst case kept 4,742 of 10,901 raw characters |

### Tier 2 — closed this revision

| Item | What changed | Verified how |
|---|---|---|
| **P2-06** — validate LLM producer output | New `RawTask` (`agents/task_utils.py`) and `RawGoal` (`agents/planning.py`) Pydantic models validate every raw goal/task dict *before* any key is indexed. A malformed entry is dropped and counted (`counters["producer_rejects"]`) instead of raising `KeyError` and aborting the whole run. `cap_and_filter`'s signature changed to `(tasks, rejected_count)` | Unit tests for the rejection/count behavior on malformed input; three separate live runs against a real model all show `producer_rejects: 0` — the well-formed path is provably unaffected |
| **P2-07** — boundary-scoped telemetry (router half) | `llm/router.py::FallbackRouter` gained `drain_counters()` — a `threading`-free accumulator (single-threaded by nature; the router isn't shared across parallel workers the way retrieval is) tracking real attempts (`llm_provider_calls`), real fallback hops (`llm_fallback_hops`), and real self-scoring calls (`llm_quality_calls`) — *self-scoring as of P2-07; P2-11 later made the judge the NEXT provider in the chain, and the counter kept its name*. Every LLM-calling node merges these into its own returned counters. `llm_calls` renamed `llm_node_calls` (no alias — an honest rename) | Unit tests on the router directly; three live traces with different fallback/timeout/escalation shapes, every number in the final telemetry traced back by hand to a specific log line each time |
| **P2-07** — boundary-scoped telemetry (retrieval half) | `retrieval/hybrid.py::HybridRetriever` gained `threading.local()`-backed counters (`retrieval_dense_calls`, `retrieval_keyword_calls`, `retrieval_leg_unavailable`), bumped as the *first* statement in `search()` so an attempt that raises partway through (e.g. a Qdrant `NotFoundError` on a missing collection) still counts as attempted. Exposed via an optional `drain_retrieval_counts` attribute on the `corpus_search` tool closure — deliberately not part of `ToolFn`'s return type, so no existing fake-tool test fixture needed to change shape | A dedicated concurrency test (`ThreadPoolExecutor`, 8 concurrent callers, each asserted to see only its own count — not a leaked or lost one); a live trace showing `retrieval_dense_calls: 6, retrieval_keyword_calls: 6` matching 6 real `search_worker` invocations exactly |
| **P2-08** — Postgres lifecycle + API run-history parity | New `close_checkpointer()` in `storage/postgres.py` (reads the real `PostgresSaver.conn` attribute — verified against actual langgraph source, not guessed). `build_app_and_settings` now returns a named `AppBundle(app, settings, durable, checkpointer)` instead of a bare 2-tuple that silently dropped `durable` (this function has since moved from `cli.py` to `assembly.py` — see Architecture; `cli.py` re-exports it). `api/server.py` surfaces `durable` in `/health`, closes the checkpointer on FastAPI shutdown, and calls `record_run` on completed `/research`/`/resume` calls | A live run shows `checkpointer.closed` logged on CLI exit against a real Postgres connection; `/health`'s `durable` field confirmed via a degraded-storage smoke test |
| **P2-09** — config strictness + populated `DECISIONS.md` | `config.py::warn_on_likely_env_typos()` logs a WARNING for a fixed list of plausible env-key typos (`HITL` vs `HITL_ENABLED`, etc.) — chosen over `extra="forbid"` outright, which risked rejecting legitimate stray env vars. E2/E3's trigger condition in `agents/gathering.py` is now evaluated regardless of `hitl_enabled`, so an `escalation.stub` WARNING fires when HITL is off, matching E1/E4's existing parity. `DECISIONS.md` populated: D-1 through D-32, sourced only from code comments and this document's own decision citations — gaps (D-7/9/10/11) flagged as such, not invented | Unit tests for the typo warning firing/not-firing and for the E2/E3 stub-log parity; a live HITL-disabled run confirmed the `escalation.stub` line actually appears |
| **Incidental — opensearch-py 3.x compatibility** | `storage/opensearch_store.py`'s `indices.exists`/`.create`/`.index`/`indices.refresh` calls passed the index/document name **positionally**; the installed `opensearch-py` 3.x client makes this a hard `TypeError` (`index=` must be a keyword). Fixed at all four call sites — `search()` already used the keyword form and was unaffected | Live: `python scripts/ingest_sample_data.py` failed with exactly this `TypeError` before the fix and completed cleanly (`OpenSearch: indexed 10`) after it |
| **P2-03 follow-up — ingest script now actually idempotent** | `scripts/ingest_sample_data.py` was still calling `QdrantStore.upsert_texts(docs)` with no `id_fn` — the mechanism P2-03 added existed but nothing used it, so every re-ingest still duplicated the dense leg. New `content_id()` helper (`uuid.uuid5` of each document's content — deterministic, and a valid Qdrant point-id shape, unlike a raw hash digest) is now passed as `id_fn` | Three new unit tests (determinism, distinctness, valid-UUID shape); **your own Qdrant collection still has the ~20 duplicate points from ingest runs before this fix landed** — this only stops future re-ingests from adding more, it doesn't retroactively clean up what's already there (a `reset_stores.py --yes` + re-ingest gets you back to a clean 10) |

**Full test suite: 157/157 passing.**

**A calibration caveat, stated plainly rather than buried:** `min_similarity`
ships with a code default of `0.35`, which predates any real measurement and
is **too low for the sample corpus and embedding model this repo actually
uses**. Measured against `sample_data/corpus.jsonl` with fastembed's default
`BAAI/bge-small-en-v1.5`, by running one query the corpus genuinely answers
and one it cannot:

```text
  0.40 ───── NOISE ─────► 0.527                    0.737 ◄──── SIGNAL ──── 0.843
                                └──── empty ────┘
                                      ▲
                                     0.60
```

Unrelated text scores **0.40–0.53** with this model, not near zero — so
`0.35` sits *below the floor of pure noise* and cannot filter anything. Set
`MIN_SIMILARITY=0.60` in `.env` for this corpus. `min_evidence_score=0.5`
stays as-is; it is a strict `>` against an RRF rank artefact, not a
similarity, and it still does real work on the single-leg path.

**Re-derive both for your own corpus rather than copying these numbers** —
OPERATIONS.md's [Fine-Tuning the System](OPERATIONS.md#fine-tuning-the-system)
section is the step-by-step procedure, with the exact commands and the
expected output at each step.

**Also new since the last revision, unrelated to the fixes above:**

- **Per-node debug logging.** `--debug` (or `DEBUG_TRACE=true`) now emits a
  `"node.enter"` log line the instant *every* node starts running —
  including `merger` and `progress_checker`, which make no LLM or store
  call and so never appeared anywhere in a trace file before this. See
  [Telemetry — read it honestly](#telemetry--read-it-honestly).
- **`--print-graph`.** Prints the compiled graph's static topology (ASCII
  if `grandalf` is installed, Mermaid text otherwise) via LangGraph's own
  introspection — independent of any run. Usable alone (no query, exits
  after printing) or combined with a query (prints, then runs normally).

## Guardrails

*New since D-46. `Full test suite: 341/341 passing` (up from 294 — the 47
new tests are entirely regression coverage for the items below).*

A dedicated `research_agent/guardrails/` package holds deterministic
post-processing checks — `citations.py` and `fencing.py` predate this work
and were already documented above; `hedging.py` is new in Phase 1. The
package's own module docstring (`guardrails/__init__.py`) states the rule
this work followed: check deterministically where possible, ask an LLM
(the critic) only where a mechanical check genuinely cannot judge — the
same split Part 7 of `internal/LEARNING_GUIDE.md` documents for the four
judges. Every guardrail below is a WARNING-level telemetry addition, a
flag on existing evidence, or a routing check already present in
`orchestration/graph.py` — none of them are a new LLM call, and none
change the graph's topology.

### Phase 1 — grounded convergence, retrieval-floor telemetry, false-precision flag, orphaned-task guard

| Item | What changed | Verified how |
|---|---|---|
| **Grounded convergence** | `state.py::ResearchState.grounded_score` (new field) is computed every gather cycle in `agents/gathering.py::progress_checker_node`, alongside the existing `recall_score`: a goal only counts toward it if at least one covering evidence item is `source in ("corpus", "mcp")` **and** shares distinctive vocabulary with the goal's own description — the same topical-overlap check `telemetry_node` already applied for `corpus_recall`, reused here (`tools/retrieval_chain.py::_distinctive_terms`) so the two numbers can never disagree. `orchestration/graph.py::route_convergence` will not treat `recall_score` reaching target as full convergence unless `grounded_score` also clears `settings.grounded_recall_target` (new, default `0.5`); otherwise it spends remaining depth budget on another gather cycle instead of compiling on ungrounded evidence | Unit tests on `route_convergence` and `progress_checker_node` directly, including the exact live shape that motivated it (a corpus hit topically unrelated to the goal it was credited against, scoring above the floor); confirmed live across six consecutive real runs — `grounded_score` correctly stayed `0.0` while `recall_score` reached `1.0`, and the router correctly kept looping instead of compiling early |
| **Retrieval-floor telemetry** | `retrieval/hybrid.py::HybridRetriever` now counts `retrieval_dense_candidates` / `retrieval_dropped_by_floor` (thread-local, same pattern as the existing P2-07 counters). `agents/compilation.py::telemetry_node` computes the drop ratio and logs a WARNING (`retrieval.floor_starvation`) once it clears `settings.retrieval_floor_warn_ratio` (new, default `0.8`) — purely observational, no routing change | Unit tests for both the counter and the WARNING threshold; confirmed live — a run with `retrieval_floor_drop_ratio: 0.902` correctly logged the WARNING, an earlier run at `0.75` correctly did not |
| **False-precision flag** | `tools/model_knowledge.py::overspecific_span` (renamed from a private `_looks_overspecific` boolean check so `guardrails/hedging.py` — Phase 2 — can reuse the same definition) flags a model-tier claim pairing a specific year with a specific quantity — percentages, energy/mass/area/air-quality units — as `Evidence.hedge_specific=True`. Deliberately narrow: a bare year or a bare rounded figure alone is not flagged, only the pairing that reads as verified fact while being unverifiable model recollection | Unit tests covering every unit class added, plus a regression test for a real `%`-matching bug found while widening the unit list (a trailing `\b` that never matched real text with a space after the sign); confirmed live across every run this session — `hedge_specific_items` reliably nonzero whenever the model tier contributed evidence |
| **Orphaned-task guard** | `agents/task_utils.py::cap_and_filter` now rejects a well-formed task whose `goal_id` isn't one of `state.goals`' actual ids — found live: `gap_generator` emitted a task tagged with a goal id that was never produced by `goal_manager`, and it was retrieved, scored, and merged into evidence anyway, permanently uncoverable. Folds into the existing `producer_rejects` counter; empty `state.goals` skips the check (nothing to validate against) rather than rejecting everything | Unit tests reproducing the exact live shape (a task tagged with a nonexistent goal id) plus the empty-goals safety case |

### Phase 2 — hedge enforcement, input validation, quality-judge alerting

| Item | What changed | Verified how |
|---|---|---|
| **Hedge enforcement** | New `guardrails/hedging.py::enforce_hedging`, called from `compiler_node` right after the existing `clean_citations` call. For every `hedge_specific=True` evidence item, finds its flagged quantity span (`tools/model_knowledge.py::overspecific_span`) and, if that exact text survived into the compiled report unhedged, appends a visible `(unverified figure)` marker after it — every occurrence, unless the surrounding text already carries an honest hedge word (`approximately`, `roughly`, etc.), in which case it's left alone. Closes the gap between "the flag exists on the evidence" and "the compiler actually followed the instruction to hedge it," found live: `hedge_specific_items: 29` on one run with zero visible hedging in the shipped report | Unit tests for tagging, non-double-tagging an already-hedged claim, ignoring non-flagged and corpus/mcp evidence, and a mixed case confirming one figure's hedge word doesn't suppress tagging an unrelated one; confirmed live — `hedge_markers_inserted: 10` (then `12`, `18`, `19` across later runs) matching visible `(unverified figure)` markers in the shipped report |
| **Input validation** | `api/server.py::ResearchRequest.query` gets `Field(min_length=1, max_length=2000)` — previously unconstrained, confirmed absent by reading the class directly. Rejected at the FastAPI/pydantic layer (422) before the graph is ever invoked | Unit tests for empty, over-cap, and at-cap boundary cases |
| **Quality-judge alerting** | *(Extended by D-106 — the ratio below says how often the judge FAILED; `llm_quality_scores_judged`, `llm_quality_score_mean`, `llm_quality_rejections` and `llm_quality_bands` now say what it DECIDED, in the RESULT block, the run narrative and `analyze_runs.py`.)* Same shape as the retrieval-floor WARNING above: `agents/compilation.py::telemetry_node` computes `llm_quality_failure_ratio` and logs a WARNING (`quality.judge_unreliable`) once it clears `settings.quality_judge_warn_ratio` (new, default `0.5`). `evaluation/quality.py::score_answer`'s fail-open design (Part 7 of the learning guide) is correct and unchanged — this only makes a 100%-failure run visible instead of indistinguishable from a 100%-genuine-pass run | Unit tests for the threshold and the 0/0 no-calls-made guard; confirmed live — every run this session showed `llm_quality_calls_failed == llm_quality_calls` (2/2, 3/3), and the WARNING fired every time |

### Phase 3 — LLM call budget observability

| Item | What changed | Verified how |
|---|---|---|
| **Call budget observability** | Same WARNING shape a third time: `settings.run_call_budget_warn` (new, default `40`) and `agents/compilation.py::telemetry_node` logs `run.call_budget_high` if `llm_provider_calls` clears it — carrying `revision_cycles` and `len(state.escalation_history)` as context, so a high call count can be read alongside how many revision/escalation cycles produced it. **Deliberately observational only, not a circuit breaker**: `max_depth`, `max_revisions`, `max_escalations`, and LangGraph's own `recursion_limit` already bound every run's worst case together, and no run to date has come near the threshold (18 provider calls is the highest observed) — enforcing a hard stop here would be acting on a failure mode with no supporting evidence, the same reasoning `min_similarity`'s own calibration caveat elsewhere in this document argues against doing blind | Unit tests for the threshold, the escalation/revision context fields, and the 0-calls no-op case |

### Phase 4 — deterministic web-source attribution (D-57)

| Item | What changed | Verified how |
|---|---|---|
| **Deterministic `## Sources`** | `guardrails/sources.py::append_web_sources` runs LAST in `compiler_node`, after `clean_citations` and `enforce_hedging`, appending a Sources section built from `source="web"` evidence whose `goal_id` the report actually cites — deduplicated by URL, ordered by score. **Prompt-instructed inline attribution was considered and rejected**: D-51 exists precisely because a prompt instruction to hedge was followed unreliably enough that a shipped report reached `hedge_specific_items: 29` with zero visible hedging. Attribution is the same shape of problem and gets the same shape of answer. It also leaves D-40's `[gN]`-only prose rule fully intact — the section sits BELOW the report, so nothing above it changes, and `citations.py` / `hedging.py` / the critic prompt need to know nothing about a new inline form | 15 unit tests, including that a report with no cited web evidence comes back **byte-identical** (the path every run with `WEB_SEARCH_ENABLED=false` takes) |
| **Web evidence cannot forge grounding** | `make_web_search_tool` tags `source="web"`, deliberately NOT `"mcp"`. Both `progress_checker_node` and `telemetry_node` test `source in ("corpus", "mcp")` as a proxy for "a real DOCUMENT backed this", so routing web results through the existing `make_mcp_tool` unchanged would have made every snippet inflate `grounded_score` and `corpus_recall` — silently restoring the exact `recall=1.0 / corpus_recall=0.0` blindness D-43 and D-47 exist to expose. `make_mcp_tool` itself is left byte-identical, so the proven Phase 1–3 corpus path cannot regress from this work | Two dedicated regression tests asserting a web item yields `recall_score 1.0` with `grounded_score 0.0` and `corpus_recall 0.0`; a third asserting `make_mcp_tool`'s output is unchanged |
| **Web evidence never enters durable memory** | `semantic_memory.store_run` excludes `source="web"` alongside `"memory"`/`"model"`. D-42's reason applies unchanged (anything stored returns on a later run as `source="memory"`, indistinguishable from document-backed evidence), plus one more: a snippet is volatile by construction, so a cached copy of today's result is a wrong answer next month with nothing marking it stale | Unit test on a mixed corpus/web batch |
| **Per-domain diversity cap** | `websearch/filtering.py::cap_by_domain` allows at most `WEB_SEARCH_MAX_PER_DOMAIN` (default `2`) hits per registrable domain, applied before scoring so the band interpolates across survivors. Not tidiness: five hits from one site read to the compiler as five independent sources agreeing, and the retrieved-item count looks identical either way | Unit tests for the cap, the `www.` collapse, order preservation, and the disable path (`0`) |

### Phase 5 — deterministic provenance notice (D-85)

| Item | What changed | Verified how |
|---|---|---|
| **Provenance notice** | New `guardrails/grounding.py::annotate_ungrounded_report`, run LAST in `compiler_node` (after `clean_citations`, `enforce_hedging` and `append_web_sources`). When fewer goals are backed by a real corpus/MCP document than `grounded_recall_target` requires, it prepends a short blockquote saying so — in the REPORT, not just in telemetry. Closes the gap run p205.246-check exposed: `grounded_score 0.0`, the retrieval floor dropping 36 of 36 dense candidates, the whole answer carried by the web tier, and the run still finishing `Final status: SUCCESS` with nothing in the deliverable saying the corpus contributed nothing. **Deliberately a notice, not a critique failure** — a rewrite cannot make evidence grounded (D-44's lesson), and D-38/D-57 make web/model answers legitimate-with-attribution rather than failures. Adds no new setting; reuses `grounded_recall_target` and the shared `has_grounded_evidence` predicate (M-1), so the notice is the report-side rendering of `corpus_recall` and cannot disagree with it | 19 tests, including that a well-grounded report comes back **byte-identical**, that the notice adds no `[gN]` marker (which would inflate `evidence_cited` and could slip a report past D-66's zero-citation gate) and no `##` heading (which would move `count_sections`), and that it is idempotent across revision passes |

### Phase 6 — the cited-figure audit (D-91)

| Item | What changed | Verified how |
|---|---|---|
| **Claim-level figure check** | New `guardrails/claims.py::audit_cited_figures`, run from `telemetry_node` against the SHIPPED report. For every sentence stating a figure AND citing a goal, it checks whether any evidence under that goal contains that figure, and reports the ones that do not (`cited_figures_checked` / `cited_figures_unsupported` / `unsupported_figures`, plus a `report.unsupported_cited_figures` WARNING). **This is the first claim-level honesty signal in the harness** — every earlier one judges the whole report or the whole evidence set. Scope is deliberately narrow: figures only, since a number is decidable without meaning; it does NOT check semantic support, which stays the critic's job (D-43/D-46). WARN-only, matching G1/G4/G7 and D-54 — the false-positive surface is real and unmeasured, and failing a critique would burn revision budget on a finding a rewrite often cannot fix (D-85's objection verbatim) | 20 tests covering both halves — that a fabricated figure IS caught, and that list numbering, heading numbers, the `## Sources` block and the D-85 provenance notice are NOT flagged. Demonstrated end-to-end on a p205.246-shaped report: `14.7%` and `2015` flagged, while `2,000,000` and `231` (both genuinely in the evidence) passed |

**Guardrails config summary** (all in `config.py`, all `Field`-validated):

| Setting | Default | Guards |
|---|---|---|
| `grounded_recall_target` | `0.5` | fraction of covered goals that must be topically-grounded corpus/mcp evidence before `route_convergence` accepts full convergence |
| `retrieval_floor_warn_ratio` | `0.8` | dense-candidate drop ratio above which `retrieval.floor_starvation` WARNs |
| `web_search_min_score` / `web_search_max_score` | `0.60` / `0.75` | the band a web result's RANK maps onto. The floor MUST exceed `min_evidence_score` or the whole tier is inert — it retrieves, costs real time, and can never cover a goal; `warn_on_web_search_band` WARNs when it does not. The ceiling must stay well under the ~1.0 a two-leg corpus hit reaches under RRF, or a snippet outranks a real document in the compiler's context |
| `web_search_max_per_domain` | `2` | hits allowed from any one registrable domain before the rest are dropped |
| `quality_judge_warn_ratio` | `0.5` | quality-judge failure ratio above which `quality.judge_unreliable` WARNs |
| `run_call_budget_warn` | `40` | `llm_provider_calls` count above which `run.call_budget_high` WARNs |
| `llm_max_tokens` | `4096` | generation budget sent to every LLM provider on every call (`llm/client.py::OpenAICompatibleClient`) — bounds a runaway generation at the request level, complementing (not replacing) the existing `_truncate_at_sentinel` cleanup on whatever comes back |
| `ResearchRequest.query` length | `1`–`2000` chars | API-boundary input validation |

**What guardrails are still open, stated plainly rather than rounded up:**
the underlying reason `gap_generator` proposes off-topic tasks in the first
place (confirmed recurring across at least three different query topics
this session) is not fixed — the orphaned-task guard and the grounded-
convergence gate both contain the *symptom*, neither fixes the *cause*.
Semantic claim verification (a second LLM judge checking claims against
evidence mechanically) was considered and deliberately not built: the
critic already performs this function correctly on every run observed, and
a second judge duplicating that role would violate this codebase's own
"LLM judgment only where deterministic validation isn't feasible" rule for
no measured benefit.

**Corrected since — D-55, the topical gate's floor was still too weak.**
The grounded-convergence gate above (D-47) contains the *symptom* of this
drift; it doesn't stop off-topic content from being retrieved into
evidence in the first place. Live trace (run p205.141-check):
`retrieval_chain._sufficient`'s scaling formula
(`need = min(3, max(1, len(terms)//4))`, from D-44) still floored `need`
at 1 for any query with ≤7 distinctive terms — which is *every*
`corpus_reformulated` retry by construction, since `_reformulate` caps
its output at 6 words. A reformulated query about Indian Army size
matched a completely unrelated Memcached document on the single
accidental word "size" (from "chunk size classes"), which cleared the
similarity floor at 0.57 and got merged into evidence under a real,
correctly-tagged goal id — no orphaning, no mislabeling, just a
legitimate citation to an irrelevant document, which then sat in
`gap_generator`'s next-cycle prompt tail and nudged it further off-topic.
Fixed by raising the floor from `max(1, ...)` to `max(2, ...)`, with a
`min(..., len(terms), ...)` cap so the new floor can never demand more
shared terms than a short query even has (a real edge case an existing
integration test caught before this shipped). Test suite: 348/348, up
from 341. See D-55 for the full account — and D-56, which corrects this
paragraph's original claim: raising the floor closed the **single**
accidental-word case, but bare years counted as distinctive terms and
travel in pairs, so a date-ranged query like `2020-2023` handed any
off-topic document mentioning those years a free two-term match. D-56
drops bare numbers (and stops discarding short acronyms like GDP, US,
PLA, which the term extractor had been throwing away all along), plus
fixes a substring-boundary bug in the hedge marker found in the same
review.

**What this still doesn't fix, confirmed live on the very next run
(p205.145-check) after the fix shipped**: `gap_generator` can still
*generate* an off-topic query in the first place (two genuine
Redis/Memcached tasks under an army-comparison goal at depth 2) — the
fix stops noise from masquerading as signal via an accidental word
match, it doesn't stop the generator from asking the wrong question.
That run's drift cost real compute but never reached the report, since
the compiler's own citation discipline had no reason to cite Redis
content in an army report. The root cause — why `gap_generator` proposes
this in the first place — remains open, same as stated above.

## Features

**Capabilities**
- Goal-driven research over an ingested corpus (hybrid dense+keyword search),
  with a retrieval-time relevance floor and a post-retrieval evidence-quality
  gate (P2-01 — see Recent Fixes).
- Parallel retrieval fan-out with concurrency-safe state merging.
- Iterative gap-filling bounded by depth and recall targets.
- Bounded self-critique with grounded rewrites.
- Cross-run semantic memory with volatility-aware staleness decay, namespaced
  on retrieval so it cannot impersonate a current run's goal (P2-02).
- **Three-hop** LLM fallback routing (local Cogito → Mistral → Gemini Flash)
  with a self-scored quality threshold, each hop joining the chain only if
  its API key is configured. The local hop and the two cloud fallbacks now
  use **separate timeouts** — the local model gets more room (large prompts,
  possible cold-start delay), the cloud fallbacks stay quick to fail over.
- Human-in-the-loop escalation (`HITL_ENABLED=true`): the graph pauses via
  LangGraph `interrupt()` on four triggers — E1 zero goals, E2 contested
  goals, E3 cannot-converge, E4 critique exhausted — and resumes on
  approve/redirect/abort under the same thread_id. CLI prompts on stdin;
  the API returns `status: interrupted` plus a `/resume` endpoint.
- Per-run debug tracing (`--debug`) dumping the exact prompt, raw response,
  provider, tokens and latency of every LLM call plus every retrieval engine's
  raw hits, **plus a `node.enter` log line for every node that runs**,
  including the ones that touch neither an LLM nor a store.
- Optional **Langfuse observability** (`LANGFUSE_ENABLED=true`, Phase 3):
  traces, spans, generations, events, and scores over every node, LLM call,
  retrieval, memory op, and HITL decision — with per-call cost, fail-open
  behaviour, and zero SDK import/network calls when disabled (the default).
  See [Observability — Langfuse (Phase 3)](#observability--langfuse-phase-3).
- `--print-graph`: the compiled graph's static topology, independent of any
  run.
- **Guardrails** (`research_agent/guardrails/`, new): grounded convergence
  (a corpus/mcp evidence item must also be topically on-topic for the goal
  it covers, not merely score above the floor); deterministic hedge
  enforcement on model-tier claims pairing a specific year with a specific
  quantity; run-level WARNING telemetry for a starved retrieval floor, a
  failing quality judge, and a high LLM call count (observational only, no
  circuit breaker); and API-boundary input length validation. See
  [Guardrails](#guardrails).
- **Web search** (`WEB_SEARCH_ENABLED=true`, Phase 4 / D-57): a real search
  engine as retrieval tier 4, reached over its OWN MCP server subprocess
  (`scripts/mcp_web_search_server.py`) so the agent process never imports a
  search client or makes an outbound request of its own. Results are
  rank-scored onto a bounded band, deduplicated, capped per domain, and
  tagged `source="web"` — which COVERS a goal but never GROUNDS one, so
  `grounded_score`/`corpus_recall` stay honest. Cited pages are appended as
  a deterministic `## Sources` section. Off by default; with it off the
  ladder is byte-identical to every pre-Phase-4 run.
- Fully offline demo mode (`LLM_MODE=stub`) — the entire graph runs with zero
  services and zero API keys.

**Non-goals**
- Production deployment, auth, multi-tenancy, horizontal scaling.
- Dynamic (LLM-decided) control flow / supervisor agents.
- ~~Web search~~ — **no longer a non-goal.** Delivered in Phase 4 (D-57) as
  retrieval tier 4, off by default. See Capabilities above.

## Architecture

### Overall architecture

Everything is assembled in exactly one place —
`assembly.py::build_app_and_settings` — and both the CLI and the API import
that same function. Nodes never construct their own dependencies, which is
why the whole system can be rewired with fakes in a single test fixture.

**The assembly function used to live in `cli.py`.** It moved to its own
module because `api/server.py` — a long-running HTTP service — was importing
its entire startup path from a module named "cli". That was merely odd while
the API was a demonstration of the seam; it becomes actively wrong once the
API is packaged and consumed by a separate project. `cli.py` still re-exports
both `build_app_and_settings` and `AppBundle`, so every existing
`from research_agent.cli import ...` call site keeps working unchanged.

```text
            ┌─────────────────────────────────────────────────────┐
            │  CLI (cli.py)               FastAPI (api/server.py) │
            │  build_app_and_settings()   — one wiring point      │
            └──────────────────────────┬──────────────────────────┘
                                       │  ResearchState(raw_query=…)
                                       │  config={thread_id, recursion_limit}
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     LangGraph workflow  —  13 nodes, fixed                   │
└──────┬──────────────────┬───────────────────┬───────────────────┬────────────┘
       │                  │                   │                   │
       ▼                  ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐    ┌──────────────┐   ┌──────────────────┐
│FallbackRouter│   │Retrieval tool│    │SemanticMemory│   │   Checkpointer   │
│ hop on error │   │ D-38 ladder: │    │ decay rerank │   │   + run history  │
│ OR on low    │   │ 5 tiers, one │    │ in Python    │   │                  │
│ quality      │   └──────┬───────┘    └─────────┬────┘   └───┬──────────┬───┘
└─┬──────┬───┬─┘          │                      │   LangGraph│      app │
  │      │   │            ▼                      │     writes │    writes│
  │      │   │    ┌───────────────────┐          │            │          │
  │      │   │    │ HybridRetriever   │          │            │          │
  │      │   │    │   RRF in Python   │          │            ▼          ▼
  │      │   │    │   + min_similar.  │          │        ┌───────────────────────┐
  │      │   │    │  floor (P2-01)    │          │        │    PostGreSQL         │
  │      │   │    └───────────────────┘          │        │ ┌───────────────────┐ │
  │      │   └──┐         ▲        ▲             │        │ │ checkpoints       │ │
  │      │      │         │        │             │        │ │ checkpoint_blobs  │ │
  ▼      ▼      ▼         │        │             │        │ │ checkpoint_writes │ │
┌─────┐ ┌────┐┌────┐  ┌───┴───┐┌───┴────────┐    │        │ ├───────────────────┤ │
│  C  │ │ M  ││ G  │  │Qdrant ││ OpenSearch │    │        │ │ agent_runs (app)  │ │
│  O  │ │ I  ││ E  │  │agent_ ││   agent_   │    │        │ └───────────────────┘ │
│  G  │ │ S  ││ M  │  │corpus ││     corpus │    │        └───────────────────────┘
│  I  │ │ T  ││ I  │  │       ││            │    │
│  T  │ │ R  ││ N  │  │ dense ││ sparse     │    │         
│  O  │ │ A  ││ I  │  └───────┘└────────────┘    │       ┌──────────────────────────┐  
│     │ │ L  ││    │      ▲          ▲           │       │ Qdrant                   │
│     │ │    ││    │      └─────┬────┘           └──────►│   agent_semantic_memory  │ 
│local│ │    ││    │            │                        │   namespaced ids (P2-02) │ 
│hop 0│ │hop1││hop2│   ┌────────┴──────────────┐         └──────────────────────────┘ 
│     │ │    ││    │   │ ingest_sample_data.py │                      ▲
│120s │ │ 90s││ 90s│   │                       │                      |  
└─────┘ └────┘└────┘   │     (manual, 1×)      │            ┌─────────┴──────────┐
                       └───────────────────────┘            │ memory_writer node │
                                                            │ (after a PASSED    │
                                                            │    critique only)  │
                                                            └────────────────────┘


```

Three things this picture is telling you that the previous one did not.
First, the fallback chain's timeouts are no longer uniform — the local hop
gets 120s, the two cloud hops get 90s each. Second, `HybridRetriever` now has
a relevance floor sitting between the dense leg and fusion. Third, memory's
Qdrant collection now writes namespaced ids on the *read* side (retrieval),
not the write side — the stored payload is unchanged, only what
`SemanticMemory.retrieve` builds an `Evidence` object with has changed.

Unchanged from before: Postgres still has **two independent writers who do
not know about each other** — the LangGraph library owns the four
`checkpoint*` tables, and this repo's own code owns `agent_runs`. That split
is exactly what you need to remember when you reset the database.

### Agent workflow

The topology below is the code in `orchestration/graph.py`, edge for edge. The
routing functions are pure — they *read* state and return a destination; they
can never write. That is why every escalation trigger is set by the **node
whose check fired**, never by the router that reads it.

```text
                     [START]
                        │
                        ▼
              ┌──────────────────┐
              │     classify     │  LLM ─► {intent, confidence}
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ memory_retrieve  │  Qdrant ─► past evidence, decay-reranked,
              └────────┬─────────┘      goal_id namespaced (P2-02)
                       │
                       ▼
              ┌──────────────────┐
              │   goal_manager   │  LLM ─► goals (+ human redirect guidance)
              └────────┬─────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │ goals present? (D-21/E1*)  │
        └───────┬─────────────┬──────┘
                │ NO          │ YES
                ▼             ▼
        (ERROR report)   ┌──────────────────┐
                │        │  task_expander   │  LLM ─► ranked backlog, capped
                │        └────────┬─────────┘        at MAX_FANOUT (D-13)
                │                 │
                │                 ▼
                │     ┌────────────────────────────┐
                │     │    D-1: backlog check      │
                │     └──────┬───────────────┬─────┘
                │            │ empty         │ tasks present
                │            ▼               ▼
                │     (EMPTY report)  ┌───────────────────┐
                │            │        │ search_worker ×N  │◄────────┐
                │            │        │   (Send fan-out,  │         │
                │            │        │  min_similarity   │         │
                │            │        │  floor applies)   │         │
                │            │        └────────┬──────────┘         │
                │            │                 ▼                    │
                │            │        ┌────────────────┐            │
                │◄───────────┘        │     merger     │  contested │
                │                     │  (D-18 flags)  │  goals     │
                │                     └────────┬───────┘            │
                │                              ▼                    │
                │                     ┌──────────────────┐         L│
                │                     │ progress_checker │         O│
                │                     │ recall + depth++ │         O│
                │                     └────────┬─────────┘         P│
                │                              ▼                    │
                │       ┌─────────────────────────────────┐         │
                │       │   convergence (D-14/E2-E3*)     │         ▲
                │       └──────┬──────────────────┬───────┘         │
                │              │ compile          │ expand          │
                │              │                  ▼                 │
                │              │        ┌────────────────┐          │
                │              │        │ gap_generator  │  LLM ─►  │
                │              │        │  (E2-E3* too)  │  new     │
                │              │        └───────┬────────┘  tasks   │
                │              │                ▼                   │
                │              │      ┌──────────────────┐  tasks   │
                │              │      │D-1: backlog chk  ├────►─────┘
                │              │      │     (E2-E3*)     │
                │              │      └────────┬─────────┘
                │              │               │ empty
                │              │               │
                │              └──────────────►┤
                │                              │
                │                              ▼
                │                     ┌────────────────┐
                │       ┌────────────►│    compiler    │  LLM ─► Markdown
                │       │             └────────┬───────┘  (+ critique notes
                │       │                      │           on a rewrite)
                │       │                      ▼
                │       │             ┌────────────────┐
                └───────┼────────────►│     critic     │  LLM ─► {passed,
                        │             └────────┬───────┘          notes}
                        │                      │
                        │                      ▼
                        │      ┌─────────────────────────────────┐
                        └──────┤       D-22: critique check      │
                        FAIL,  └───────────┬───────────────┬─────┘
                         budget            │ FAIL,         │ PASS
                         (revise)          │ exhausted     │
                                           │ (E4*)         │
                                           │               ▼
                                           │     ┌───────────────┐
                                           │     │ memory_writer │──► Qdrant
                                           │     └───────┬───────┘
                                           │             │
                                           ▼             ▼
                                      ┌─────────────────────┐
                                      │      telemetry      │
                                      └───────────┬─────────┘
                                                  │
                                                  ▼
                                                [END]

Legend
  Left rail — ERROR/EMPTY reports join the main flow at critic (waved
  through, D-21): every path reaches telemetry → [END].
  E1*/E2-E3*/E4* — with HITL_ENABLED=true the marked checks INTERRUPT the
  run for human review (approve / redirect with guidance / abort) and
  resume under the same thread_id.
  With HITL disabled: E1–E4 all log `escalation.stub` at WARNING and
  continue (E4 ships the report marked unreviewed — never silently as
  good). P2-09 closed the E2/E3 parity gap that used to exist here
  (their trigger block used to sit inside `if settings.hitl_enabled`,
  so they logged nothing when HITL was off, unlike E1/E4 — now fixed,
  see Recent Fixes and Limitations).
```

**Termination is guaranteed by four independent bounds** (design §6.3, all four
present in code): the depth counter ticked once per cycle by
`progress_checker`, the finite task supply enforced by the dedup key sets in
`cap_and_filter`, the empty-backlog fallthrough in `dispatch_tasks`, and the
invoke-time `recursion_limit` backstop (default 60).

**`human_escalation` has no static edges.** It returns `Command(goto=…)` and
routes *itself* to `goal_manager`, `gap_generator`, `compiler`, or `telemetry`
based on `(trigger, action)`. That is why you will not find it on the
right-hand side of any `add_edge` call.

### Request flow (one worker, and why it no longer always finds something)

The earlier version of this diagram stopped politely at "fused hits (RRF)",
with no relevance floor anywhere. That gap — a dense index always returning
its k nearest neighbours regardless of actual relevance — is now closed on
the dense leg by P2-01's `min_similarity` floor, applied BEFORE fusion.

```text
 dispatch          search_worker       corpus tool      HybridRetriever    stores
    │                    │                  │                  │              │
    │ Send(WorkerPayload)│                  │                  │              │
    ├───────────────────►│                  │                  │              │
    │                    │    tool(task)    │                  │              │
    │                    ├─────────────────►│                  │              │
    │                    │                  │  search(query)   │              │
    │                    │                  ├─────────────────►│              │
    │                    │                  │                  │ dense top-k  │
    │                    │                  │                  ├─────────────►│
    │                    │                  │                  │◄─────────────┤
    │                    │                  │                  │ BM25 top-k   │
    │                    │                  │                  ├─────────────►│
    │                    │                  │                  │◄─────────────┤
    │                    │                  │                  │              │
    │                    │                  │      ┌───────────┴────────────┐ │
    │                    │                  │      │ P2-01: drop dense hits │ │
    │                    │                  │      │ below min_similarity   │ │
    │                    │                  │      │ (default 0.35) BEFORE  │ │
    │                    │                  │      │ fusion — NEW           │ │
    │                    │                  │      └───────────┬────────────┘ │
    │                    │                  │       ┌──────────┴───────────┐  │
    │                    │                  │       │ rrf_fuse()           │  │
    │                    │                  │       │ join key = title, or │  │
    │                    │                  │       │ content[:60] — NOT   │  │
    │                    │                  │       │ any store's id       │  │
    │                    │                  │       └──────────┬───────────┘  │
    │                    │                  │ fused hits (RRF) │              │
    │                    │                  │◄─────────────────┤              │
    │                    │   [Evidence]     │                  │              │
    │                    │   score = min(1, │                  │              │
    │                    │   fused × 30.0)  │                  │              │
    │                    │◄─────────────────┤                  │              │
    │ {evidence, keys,   │                  │                  │              │
    │  counters} ONLY —  │                  │                  │              │
    │  D-15 whitelist    │                  │                  │              │
    │◄───────────────────┤                  │                  │              │
    │                    │                  │                  │              │
```

**A dense index still always returns its k nearest neighbours** — that fact
about how dense retrieval works hasn't changed and can't be changed at this
layer. What changed is that "nearest" no longer automatically becomes
"evidence": anything below `min_similarity` is dropped before it can ever
reach `HybridRetriever`'s output, let alone become an `Evidence` object. The
BM25 leg has no equivalent floor — BM25 scores are corpus-dependent and
unbounded, so there's no principled fixed cutoff the way there is for a 0..1
cosine similarity; this is a documented, deliberate gap, not an oversight.

### Retrieval is now a ladder, not just this one hop (D-38–D-46, D-57)

The diagram above is tier 1 of 5. If corpus search comes back below `min_evidence_score` — or scores high but shares no distinctive vocabulary with the query (D-39) — `tools/retrieval_chain.py` tries, in order: **one reformulated corpus retry** (shorter, stripped-down query), then **MCP**, then **web search** (Phase 4 / D-57, off by default), then the **model’s own knowledge** (`tools/model_knowledge.py`), stopping at the first tier that clears the bar. Model-tier items always carry `source="model"`, are never relabelled as corpus hits, never persist to durable memory (D-42), and the compiler must attribute them as general knowledge rather than a retrieved finding (D-40). Telemetry’s `corpus_recall` and `model_sourced_items` fields exist specifically so a large gap between them and `recall` is visible — it means the answer is recollection, not retrieval. Full rationale and live evidence: `DECISIONS.md` D-38 through D-46, and D-57 for the web tier.

```text
  tier                        runs in                 evidence      counts as
                                                      source        a document?
  ──────────────────────────────────────────────────────────────────────────────
  1  corpus                   in-process              "corpus"      YES
       dense + BM25, RRF-fused
       │ miss
  2  corpus, reformulated     in-process              "corpus"      YES
       │ _reformulate(), ≤6 words
       │ miss
  3  mcp                      standalone ──HTTP────► scripts/mcp_corpus_server.py
       MCPBridge #1                                   "mcp"         YES
       │ ...but that is the SAME ingested corpus, reached a second way
       │ miss
  4  web    (Phase 4)         standalone ──HTTP────► scripts/mcp_web_search_server.py
       MCPBridge #2                                 └──► THE INTERNET
                                                      "web"         no
       │ miss
  5  model                    in-process (LLM)        "model"       no
       the model's own recollection — no retrieval at all
```

**Tiers 1–3 all resolve to the SAME ingested documents.** That is exactly why
"the corpus does not contain it" used to fall straight through to the
model's own memory — there was nowhere else to look. Tier 4 is the first
tier that can return something the corpus never held, which is why it sits
where it does rather than replacing tier 3.

**The right-hand column is the honesty rail.** `recall` counts a goal
answered by any tier; `grounded_score` (G2/D-47) and `corpus_recall` (D-43)
count it answered *by a real document* — and only tiers 1–3 qualify. So a
run answered wholly from the web reads `recall 1.0 / grounded_score 0.0 /
web_sourced_items 12`: visible rather than flattering. Tagging web results
`source="mcp"` instead would have made every snippet inflate both metrics,
silently restoring the exact blindness they exist to expose.


### Storage interactions

Three stores, five distinct data flows. Qdrant's `upsert_texts` gained an
optional idempotent-id mechanism under P2-03; as of this revision
`scripts/ingest_sample_data.py` actually passes it — see the note under
Ingest identity below for exactly what that does and doesn't fix (it's
forward-looking only; it doesn't retroactively deduplicate a collection
that already has stale duplicate points in it from before the fix).

```text
┌───────────────────────────┐         ┌────────────────────────────────────┐
│ scripts/                  │         │ OpenSearch                         │
│ ingest_sample_data.py     ├────────►│ index: agent_corpus (CORPUS_INDEX) │
│                           │         │ _id = str(i)   ── OVERWRITES ──    │
│ run manually, once per    │         │ mapping: content:text title:text   │
│ corpus change             │         │          topic:keyword             │
│                           │         └────────────────────────────────────┘
│                           │         ┌────────────────────────────────────┐
│                           ├────────►│ Qdrant                             │
│                           │         │ collection: agent_corpus           │
└───────────────────────────┘         │ id = uuid.uuid5(content) ── FIXED ─│
                                      │ (P2-03 wired in — deterministic,   │
                                      │  overwrites in place; see Ingest   │
                                      │  identity note below for the id_fn │
                                      │  mechanism this actually uses)     │
                                      │ vector: fastembed(content)         │
                                      │ payload: title, topic, content,    │
                                      │          created_at                │
                                      └────────────────────────────────────┘

┌───────────────────────────┐         ┌────────────────────────────────────┐
│ memory_retrieve node      │◄────────┤ Qdrant                             │
│  similarity × decay,      │         │ collection: agent_semantic_memory  │
│  goal_id NAMESPACED on    │         │ id = uuid4()   ── DUPLICATES ──    │
│  the way out (P2-02)      │         │ vector: fastembed(content)         │
│                           │         │ payload: content, goal_id (RAW,    │
│ memory_writer node        ├────────►│   unnamespaced — see note),        │
│  ONLY after a PASSED      │         │          volatility, source_query, │
│  critique (D-24)          │         │          created_at                │
└───────────────────────────┘         │ NO payload indexes (D-27.5 unmet)  │
                                      └────────────────────────────────────┘

┌───────────────────────────┐         ┌────────────────────────────────────┐
│ LangGraph PostgresSaver   │         │ PostgreSQL                         │
│ .setup() at assembly      ├────────►│ checkpoints                        │
│ writes once per superstep │         │ checkpoint_blobs                   │
│ (D-8/D-20 — this is what  │         │ checkpoint_writes                  │
│  makes interrupt/resume   │         │ checkpoint_migrations              │
│  and --thread-id work)    │         │  + 3 thread_id indexes             │
└───────────────────────────┘         │····································│
┌───────────────────────────┐         │ agent_runs                         │
│ storage/postgres.py       ├────────►│  id, thread_id, query, recall,     │
│ record_run() — CLI ONLY   │         │  telemetry JSONB, created_at       │
│ CREATE TABLE IF NOT EXISTS│         │  written by APP code, one row per  │
│ on every single call      │         │  completed CLI run; nothing ever   │
└───────────────────────────┘         │  reads it back                     │
                                      └────────────────────────────────────┘
```

**On the memory `goal_id` note above:** P2-02's fix happens on the *read*
side only — `SemanticMemory.retrieve` builds the namespaced
`"memory::<original>"` id when constructing the `Evidence` object it returns.
The *stored* Qdrant payload is untouched; `memory_writer` still writes the
raw, unnamespaced `goal_id` exactly as before. This is a deliberate,
minimal-surface fix: the collision only ever mattered at the coverage-check
comparison (`e.goal_id == g.goal_id` in `agents/gathering.py`), which only
ever sees the retrieved `Evidence`, never the raw stored payload directly.

The dotted line inside the Postgres box is the ownership boundary. Above it,
LangGraph. Below it, us. Both halves are recreated automatically after a wipe —
`PostgresSaver.setup()` rebuilds the top, `record_run`'s
`CREATE TABLE IF NOT EXISTS` rebuilds the bottom.

### Ingest identity — fixed this revision, but check your existing collection

```text
                      sample_data/corpus.jsonl  (10 lines)
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
      ┌───────────────────┐                   ┌───────────────────┐
      │ OpenSearchStore   │                   │ QdrantStore       │
      │ .ingest()         │                   │ .upsert_texts()   │
      │ _id = str(i)      │                   │ id = uuid.uuid5(  │
      │                   │                   │   NAMESPACE_URL,  │
      │                   │                   │   content)   FIXED│
      └─────────┬─────────┘                   └─────────┬─────────┘
                │                                       │
   run ingest 1 ▼                          run ingest 1 ▼
      ┌───────────────────┐                   ┌───────────────────┐
      │  10 documents     │                   │  10 points        │
      └─────────┬─────────┘                   └─────────┬─────────┘
                │                                       │      
   run ingest 2 ▼                          run ingest 2 ▼
      ┌───────────────────┐                   ┌───────────────────┐
      │  10 documents     │                   │  10 points        │
      │  same ids ─►      │                   │  same content ─►  │
      │  overwritten      │                   │  SAME uuid5 id ─► │
      └─────────┬─────────┘                   │  overwritten too  │
                |                             └─────────┬─────────┘
   run ingest 3 ▼                                       |
      ┌───────────────────┐               run ingest 3  ▼
      │  10 documents     │                  ┌───────────────────┐
      └─────────┬─────────┘                  │  10 points, still │
                │                            └─────────┬─────────┘
                └──────────────────┬───────────────────┘
                                   |
                                   ▼
                 ┌────────────────────────────────────┐
                 │ rrf_fuse() joins on `title`        │
                 │ ── still fragile for a corpus      │
                 │    with repeated/missing titles,   │
                 │    but no longer masking a growing │
                 │    pile of duplicate points        │
                 └────────────────────────────────────┘
```

**P2-03 added the MECHANISM for idempotent ingest** —
`QdrantStore.upsert_texts` accepts an optional `id_fn` parameter, and
passing a deterministic, content-derived function makes re-ingesting the
same content overwrite in place instead of duplicating. **As of this
revision, `scripts/ingest_sample_data.py` actually passes one** — a new
`content_id(item)` helper computing `uuid.uuid5(uuid.NAMESPACE_URL,
item["content"])`. `uuid5` was chosen deliberately over a raw content hash:
Qdrant point ids must be an unsigned int or a UUID string, so a hex digest
would be rejected outright, while `uuid5` gives a real, valid UUID that is
still fully deterministic on its input.

- **The corpus ingest duplication bug described in earlier revisions is
  now actually fixed in practice**, not just mechanism-available. Verified
  live: `python scripts/ingest_sample_data.py` run twice in a row against
  the same corpus reports `Qdrant: embedded 10` both times, not `10` then
  `20`.
- **⚠ This does not retroactively clean up a collection that already
  accumulated duplicates before this fix landed.** If you ran the old
  ingest script more than once against a real Qdrant instance, your
  `agent_corpus` collection likely still has stale duplicate points sitting
  in it (10 logical documents × however many times you ran ingest before
  this fix). This fix only stops *future* re-ingests from adding more. To
  get back to a clean state: `scripts/reset_stores.py --yes` (drops the
  collection), then re-ingest.
- **Memory's accumulation is unchanged, and deliberately so** —
  `memory_writer` still calls `upsert_texts` with no `id_fn`, and that's
  intentional: it is meant to keep accumulating fresh evidence every passed
  run, not collapse repeats. Deduping memory is a larger, separate piece of
  work tracked as `P2-15`, not something P2-03 touches.

**Do the chunks match between Qdrant and OpenSearch?** Yes, now on both axes.
Neither store chunks anything — one JSONL line becomes exactly one
OpenSearch document and exactly one Qdrant point, with byte-identical
`content`. Only `content` is embedded, and only `content` is BM25-matched.
Both stores now assign the SAME document the SAME id on every re-ingest —
OpenSearch via `str(i)` (unchanged, always was idempotent), Qdrant via
`uuid5(content)` (new this revision).

`scripts/reset_stores.py` remains the way back to a clean state for a
collection with stale duplicates already in it — see below.

### LLM fallback chain

One policy, applied identically at every hop — except, as of this revision,
for **which timeout each hop uses**.

```text
     complete_json(messages)                    complete(messages)
              │                                          │
              ▼                                          ▼
   ┌─────────────────────┐                    ┌─────────────────────┐
   │ hop 0: local Cogito │                    │ hop 0: local Cogito │
   │ timeout: 120s (NEW, │                    │ timeout: 120s (NEW, │
   │  was 60s shared)    │                    │  was 60s shared)    │
   └──────────┬──────────┘                    └──────────┬──────────┘
              │                                          │
     ┌────────┴────────┐                        ┌────────┴────────┐
     │ transport error │                        │ transport error │
     │ OR unparseable  │                        │ OR self-scored  │
     │ JSON            │                        │ quality < 0.6   │
     └────────┬────────┘                        └────────┬────────┘
              ▼                                          ▼
   ┌─────────────────────┐                    ┌─────────────────────┐
   │ hop 1: Mistral      │   joins ONLY if    │ hop 1: Mistral      │
   │ timeout: 90s (was   │   its key is set   │ timeout: 90s (was   │
   │  60s shared)        │                    │  60s shared)        │
   └──────────┬──────────┘                    └──────────┬──────────┘
              ▼                                          ▼
   ┌─────────────────────┐                    ┌─────────────────────┐
   │ hop 2: Gemini Flash │   joins ONLY if    │ hop 2: Gemini Flash │
   │ timeout: 90s        │   its key is set   │ timeout: 90s        │
   └──────────┬──────────┘                    └──────────┬──────────┘
              │                                          │
              ▼                                          ▼
   chain exhausted ─► raise                    chain exhausted ─► return
   the LAST provider's error                   the LAST answer we got
   (because no answer exists)                  (better thin than nothing)

  ┌────────────────────────────────────────────────────────────────────┐
  │ FIXED (P2-04) — was: OBSERVED DEFECT, trace run-0d7d0448906a       │
  │                                                                    │
  │ The local model would answer correctly, then append its chat       │
  │ template's end-of-turn sentinel:                                   │
  │                                                                    │
  │       { "goals": [ … ] } <|im_end|>                                │
  │                                                                    │
  │ _extract_json() now strips known sentinels (see Recent Fixes)      │
  │ before parsing, and falls back to extracting the outermost {...}   │
  │ span if that still isn't enough. Confirmed live: three separate    │
  │ runs against a real local model, previously failing with           │
  │ JSONDecodeError on every structured call, now succeed with zero    │
  │ fallback needed.                                                   │
  │                                                                    │
  │ The genuine timeouts on the same runs (a tiny classify prompt      │
  │ hitting the wall at exactly the configured limit every time, and   │
  │ a large compiler prompt genuinely needing more room) are a         │
  │ SEPARATE issue from the parsing bug — see the timeout split above. │
  │ classify's case remains unexplained even at 120s and is worth      │
  │ investigating the local server itself for, independent of any      │
  │ fix in this codebase.                                              │
  └────────────────────────────────────────────────────────────────────┘
```

Note the asymmetry between the two columns, because it is deliberate. JSON calls
have **no quality gate** — a parsed object either satisfies the caller's schema
or it does not, and the nodes validate their own required keys. Free-text calls
do have one. And that quality score comes from asking the *next provider in the
fallback chain* to rate the answer — never the same provider that produced it
(P2-11): cheap and honest about being weak in `evaluation/quality.py`. A scorer
that itself errors returns `1.0`, so a flaky scoring call can never burn a
working answer path — and that fail-open path is itself counted
(`llm_quality_calls_failed`), so it's visible in telemetry rather than only
in a log line.

### Observability — Langfuse (Phase 3)

A dedicated package is the *only* place the Langfuse SDK is imported. Every
business module talks to it through five thin, always-safe functions —
`start_trace`, `span`, `generation`, `event`, `score` (plus `end_trace`,
`flush`, `shutdown`) — and never sees an SDK object, a trace handle, or a
span handle:

```text
research_agent/langfuse/
    __init__.py   thin public functions only (the whole import surface)
    client.py     the ONLY file that imports the langfuse SDK; build_client()
                   returns a real client or None, never raises
    observer.py   Observer — trace lifecycle, spans, generations, events,
                   scores, cost, shutdown, thread safety (threading.Lock)
    pricing.py     cost = f(Settings-configured $/1M rates, token usage)
    helpers.py     thread_id_from_config, traced_node (generic node wrapper)
```

```text
 cli.py / graph.py / llm / retrieval / memory / agents
              │  from research_agent import langfuse as lf
              ▼
   research_agent/langfuse/__init__.py  (thin functions, always safe)
              │
              ▼
        observer.py::Observer  ──fail-open, never raises──►  caller
              │  (enabled?)
              ▼
        client.py::build_client()  →  langfuse SDK v4 (OTel-based)
              │
              ▼
     Langfuse Cloud  /  self-hosted  /  enterprise  (LANGFUSE_HOST)
```

```text
  Langfuse Observability — Instrumentation Points (Phase 3)

┌─────────────────────────────────────────────────────┐
│  CLI (cli.py)               FastAPI (api/server.py) │
│  build_app_and_settings()   — one wiring point      │◄──┐
└──────────────────────────┬──────────────────────────┘   │ start_trace()/
                           │                              │ end_trace()
                           ▼                              │ (root, per thread_id)
┌─────────────────────────────────────────────────────┐   │
│       LangGraph workflow — 13 nodes, fixed          │◄──┤ traced_node() wraps
└───────┬──────────────────┬────────────────────┬─────┘   │ EVERY add_node() call
        │                  │                    │         │ (span per node)
        ▼                  ▼                    ▼         │
┌──────────────┐   ┌────────────────┐   ┌──────────────┐  │
│FallbackRouter│◄──┼─ event()       │   │SemanticMemory│◄─┤ event()
│ hop on error │   │  "llm.fallback"│   │ decay rerank │  │  memory.retrieved/
└─┬─────┬────┬─┘   └──────┬─────────┘   └──────────────┘  │  memory.stored
  │     │    │            │                               │
  ▼     ▼    ▼            ▼                               │
┌───┐ ┌───┐ ┌───┐     ┌─────────────────┐                 │
│ C │ │ M │ │ G │     │ HybridRetriever │◄────────────────┤ span()
│ o │ │ i │ │ e │     │  RRF + floor    │                 │  retrieval.hybrid_search
│ g │ │ s │ │ m │     │                 │                 │
│ i │ │ t │ │ i │     │  RRF + floor    │                 │
│ t │ │ r │ │ n │     └─────────────────┘                 │
│ o │ │ a │ │ i │                                         │
└───┘ │ l │ └───┘                                         │
  ▲   └───┘   ▲                                           │
  │      ▲    │                                           │
  │      │    │                                           │
  └──────┴────┴───────────────────────────────────────────┤
     generation()                                         │
        — tokens, latency, cost                           │
        — temperature, per real provider call             │
                                                          │
                                                          │
                    ┌─────────────────────────────────────┴────────────┐
                    │         research_agent/langfuse/                 │
                    │  start_trace / end_trace / span / generation /   │
                    │  event / score / flush / shutdown                │
                    │                                                  │
                    │  client.py → Langfuse SDK v4 (OTel) → Langfuse   │
                    │  Cloud / self-hosted / enterprise (LANGFUSE_HOST)│
                    │                                                  │
                    │  Off by default (LANGFUSE_ENABLED=false):        │
                    │  zero import, zero network calls                 │
                    └──────────────────────────────────────────────────┘
```

Disabled by default (`LANGFUSE_ENABLED=false`): the package never imports the
`langfuse` SDK and makes zero network calls. A client that raises on every
call (bad credentials, unreachable host) never propagates an exception to a
caller — the agent runs identically either way. `traced_node()` wraps each of
the 13 `add_node(...)` call sites in `orchestration/graph.py` in one place,
so every node is spanned (real `input`/`output`, timing, error state) without
touching any of the 13 node function bodies.

**Instrumented:** one root trace per query keyed by `thread_id`
(`cli.py::_run`); every graph node (`traced_node`); every real LLM provider
call — tokens, latency, cost, temperature (`llm/client.py`); every fallback
hop (`llm/router.py`); hybrid retrieval hit counts and degraded-leg state
(`retrieval/hybrid.py`); memory retrieve/store counts
(`memory/semantic_memory.py`); recall/coverage per gather cycle
(`agents/gathering.py`); critique pass/fail and self-score
(`agents/compilation.py`); and the HITL trigger, human decision, and resume
latency as both an event and a `human_review` score (`cli.py`'s HITL loop).
Session grouping (`propagate_attributes(session_id=thread_id, ...)`) survives
LangGraph's own parallel `Send` dispatch, so `search_worker`'s fan-out
inherits the correct session with no special-casing.

**Cost** (`langfuse/pricing.py`) maps `FallbackRouter`'s own provider names
(`"primary"`, `"mistral"`, and whichever of `"gemini"`/`"grok"`
`LLM_FALLBACK_NAME` selects — D-114) to Settings-configured `$/1M`-token
rates — an unconfigured provider costs `$0` (correct for a free local model,
honest "unknown" for cloud), and a misconfigured negative rate clamps to
zero rather than reporting negative dollars.

**Traces are nested, and the API path is traced too — both were listed here
as deferred limitations in earlier revisions and are now closed.**

`TraceContext` accepts an optional `parent_span_id`, and `Observer` already
held the run's open root span for `end_trace`'s sake — so nesting needed no
new state and no call-site changes. Node spans additionally open *before*
the node runs (`Observer.span_ctx`, wrapping the SDK's
`start_as_current_observation`), which makes their timestamps and durations
real rather than ~0, and makes everything a node produces nest underneath
it. The UI now renders `research_run → node:compiler → llm` rather than a
flat list.

`api/server.py` opens and closes a root trace per request, emits the same
four scores the CLI does, and calls `lf.shutdown()` in its lifespan hook.
The pairing is deliberately per-REQUEST rather than per-run: `Observer`'s
`propagate_attributes` context is sync-only and attaches to the calling
thread's OTel context, and FastAPI runs these `def` endpoints in a REUSED
threadpool — so a context entered serving `/research` and exited serving
`/resume` would leak one caller's session onto the next request that worker
picks up. A HITL run therefore produces two root spans on one trace, which
is an honest record of two HTTP requests and is the version that cannot
leak.

**What is still a limitation:** `span()` (the post-hoc form, still used by
`retrieval/hybrid.py`) carries its duration as a `duration_ms` metadata
field rather than as real span timestamps — the observation is created after
the work finished, so a true end time would produce a negative duration.
`span_ctx()` is the fix where a block exists to wrap. `flush()` and
`is_enabled()` remain exported with no caller.

## Storage Contracts

The diagrams above give you the shape. This is the reference you come back to
when something is in the wrong place.

### PostgreSQL

| Table | Created by | Written by | Purpose |
|---|---|---|---|
| `checkpoint_migrations` | `PostgresSaver.setup()`, called from `storage/postgres.py::get_checkpointer` | LangGraph library | Checkpointer schema versioning. Never touched by our code. |
| `checkpoints` | same | LangGraph, once per superstep | Serialized `ResearchState` channel values keyed by `thread_id`. This is what makes `interrupt()`/resume and `--thread-id` replay work (D-8/D-20). |
| `checkpoint_blobs` | same | LangGraph | Out-of-line channel values too large to inline. |
| `checkpoint_writes` | same | LangGraph | Pending per-task writes for a superstep — including the interrupt resume payload. |
| `agent_runs` | **our code**, via `CREATE TABLE IF NOT EXISTS` on *every* `record_run`/`record_failed_run` call | **our code**, one row per completed run from **either** interface — `cli.py` and `api/server.py::_respond` alike (P2-08) — plus, since D-103, one per FAILED run. Failed runs are **CLI-only**: nothing in the API calls `record_failed_run` (D-121). An earlier revision of this row said the API never recorded anything; that was wrong | Post-hoc run history: `id BIGSERIAL PK`, `thread_id TEXT`, `query TEXT`, `recall REAL`, `telemetry JSONB`, `created_at TIMESTAMPTZ DEFAULT now()`. A failed run's row carries `recall` NULL and `telemetry` `{"run_outcome": "failed", "failure": {...}}`; a completed run's row has no `run_outcome` key at all, so "absent means completed" classifies the whole history including every pre-D-103 row. Read back by `scripts/analyze_runs.py` (D-92) — and by you and DBeaver. |

Plus three LangGraph indexes: `checkpoints_thread_id_idx`,
`checkpoint_blobs_thread_id_idx`, `checkpoint_writes_thread_id_idx`.

Three behaviours worth knowing before you debug something:

- **Both CLI and API write run history (P2-08).** `record_run` is called
  from both `cli.py::main` and `api/server.py` on completion of
  `/research` and `/resume` — a run through either path produces an
  `agent_runs` row. **Corrected this pass:** `api/server.py` previously
  couldn't even import (`AppBundle` unpack crash) — if you were on an
  un-patched checkout, this path wasn't reachable at all until this
  session's fix.
- **Degradation is surfaced, not silent (P2-08).** `get_checkpointer`
  still catches *any* exception and returns `MemorySaver()` with
  `durable=False`, but `build_app_and_settings` no longer discards that
  flag — it's carried on the returned `AppBundle` and surfaced in the
  API's `/health` response. The stderr log line
  (`checkpointer.postgres_active` / `checkpointer.memory_fallback`) still
  fires too, so there are now two independent ways to see it, not one.
  **Corrected this pass:** you may now also see `checkpointer.pool_active`
  logged (Postgres checkpointer pooling, new this session) — a healthy
  sign, not an error.
- **The checkpointer connection is closed on exit (P2-08).**
  `close_checkpointer()` is called from both the CLI's `finally` block
  and the FastAPI shutdown handler.
- **⚠ Reusing the same `--thread-id` across unrelated runs silently
  accumulates state.** Every `Annotated[..., reducer]` field on
  `ResearchState` (`counters`, `evidence`, `completed_task_keys`,
  `critique_notes`, `escalation_history`) merges the new invoke's fresh
  input with whatever was already checkpointed for that thread_id, rather
  than replacing it — confirmed across four consecutive live runs under one
  reused thread-id, where `evidence_items`, `memory_writes`, and
  `revision_cycles` all grew linearly, run over run. Harmless if you're
  deliberately resuming a paused HITL run; a real correctness risk if you
  reuse a thread-id for a second, unrelated question — the second run's
  compiled report can silently draw on leftover evidence from the first.
  Not yet fixed; the practical workaround is simply not reusing thread-ids
  for unrelated queries. **Corrected this pass:** `escalation_history`
  from this same reducer set is, as of this session, ALSO surfaced in
  every run's final telemetry (`telemetry["escalations"]`) — previously
  written and never read anywhere; the accumulation risk described here
  is unchanged, but the escalation history itself is now visible in the
  output, not only in Postgres.

### Qdrant

Two collections, both bootstrapped lazily by `ensure_collection` — which runs
on *every* search as well as every upsert, so there is an extra round trip per
query hiding in there.

| Collection | Setting (default) | Written by | Read by |
|---|---|---|---|
| Corpus, dense leg | `CORPUS_INDEX` (`agent_corpus`) | `scripts/ingest_sample_data.py` **only** | `HybridRetriever`, from the `search_worker` nodes |
| Semantic memory | `MEMORY_COLLECTION` (`agent_semantic_memory`) | `memory_writer` node → `SemanticMemory.store_run`, **only after a passed critique** | `memory_retrieve` node |

> Watch the name reuse: `CORPUS_INDEX` is one setting serving as *both* the
> Qdrant collection name and the OpenSearch index name.

**Embeddings** come from `fastembed.TextEmbedding()` called with **no model
argument** — fastembed's own default small English model, downloaded on first
use (~100 MB). The dimension is never hardcoded: at collection creation the
store embeds the literal string `"probe"` and takes `len(vector)`. Distance is
`COSINE`. **Corrected this pass:** the lazy build of this embedder is now
guarded by a lock (`threading.Lock`) — one `QdrantStore` is shared across
every parallel `search_worker`, and without the lock a fan-out could
previously trigger several concurrent embedder builds at once.

**Per-vector payload**

*Corpus collection* — the entire source JSONL object plus one field added at
upsert time:

| Key | Origin |
|---|---|
| `title` | corpus.jsonl |
| `topic` | corpus.jsonl |
| `content` | corpus.jsonl — **this is the embedded text** |
| `created_at` | `time.time()` at upsert |

*Memory collection* — assembled in `SemanticMemory.store_run`:

| Key | Origin | Note |
|---|---|---|
| `content` | `Evidence.content` — **the embedded text** | |
| `goal_id` | `Evidence.goal_id` | The RAW goal id of the run that stored it — unchanged by P2-02, which namespaces only on the *read* side (see the Storage interactions note above) |
| `volatility` | `Evidence.volatility.value` | Always `semi_stable` in practice; nothing classifies volatility |
| `source_query` | the storing run's `raw_query` | Provenance only; never filtered on |
| `created_at` | `time.time()` at upsert | Drives the decay rerank |

At **search** time `QdrantStore.search` adds two keys to each returned dict that
are **not stored**: `similarity` (raw Qdrant score) and `age_days` (derived from
`created_at`). Decay is applied on top of those, in Python, in
`memory/semantic_memory.py`. `SemanticMemory.retrieve` then rebuilds the
`goal_id` as `"memory::<stored value>"` before constructing the `Evidence`
object it hands back (P2-02) — that transformation happens in this file, not
in storage.

### OpenSearch

One index, `CORPUS_INDEX` (`agent_corpus`). Memory never touches OpenSearch at
all.

- **Created by** `OpenSearchStore.ensure_index`, called from `ingest`.
- **Written by** `scripts/ingest_sample_data.py` **only**.
- **Mapping:** `content: text`, `title: text`, `topic: keyword`.
- **Document `_id` = `str(i)`** — the zero-based line index in `corpus.jsonl`.
  Deterministic, therefore idempotent. If you *shrink* the corpus, ids at or
  beyond the new length survive as orphans; reset rather than re-ingest.
- **Query:** a single `match` on `content`. `title` and `topic` are mapped but
  never queried.
- `ingest` calls `indices.refresh`, so documents are searchable immediately.

### Resetting all three stores

`scripts/reset_stores.py` (with `reset.bat` for Windows) is the supported way
back to a pristine, re-ingestable state. Because corpus ingest is still not
idempotent by default (see Ingest identity above), this is not an optional
convenience — it is the only way to reload a corpus without silently
multiplying the dense index.

```bash
# preview — touches nothing
PYTHONPATH=src python scripts/reset_stores.py --dry-run

# full reset, then reload
PYTHONPATH=src python scripts/reset_stores.py --yes
PYTHONPATH=src python scripts/ingest_sample_data.py

# keep everything the agent has learned, reset only the corpus
PYTHONPATH=src python scripts/reset_stores.py --yes --keep-memory

# one store at a time
PYTHONPATH=src python scripts/reset_stores.py --yes --qdrant
```

Windows: `reset.bat` previews; `reset.bat --yes` resets **and** re-ingests.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ reset_stores.py --yes                                                    │
├──────────────────────────────────────────────────────────────────────────┤
│ Qdrant      ─► drop collection CORPUS_INDEX                              │
│             ─► drop collection MEMORY_COLLECTION (unless --keep-memory)  │
│ OpenSearch  ─► delete index    CORPUS_INDEX                              │
│ Postgres    ─► drop agent_runs, checkpoint_writes, checkpoint_blobs,     │
│                checkpoints, checkpoint_migrations                        │
├──────────────────────────────────────────────────────────────────────────┤
│ Everything is recreated on next use:                                     │
│   PostgresSaver.setup()   rebuilds the checkpoint* tables                │
│   record_run()            rebuilds agent_runs                            │
│   ensure_collection()     rebuilds the Qdrant collections                │
│   ensure_index()          rebuilds the OpenSearch index                  │
├──────────────────────────────────────────────────────────────────────────┤
│ WARNING: dropping the checkpoint* tables destroys every resumable        │
│ thread — including any run currently PAUSED at an interrupt(). Check     │
│ for paused threads before you run this.                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

Each store is independent: an unreachable one is reported and skipped, never
fatal — the same graceful-degradation posture as the rest of the codebase.
Exit code 1 means at least one requested store could not be reached, and that
applies to `--dry-run` too, deliberately: an unreachable store is exactly what
you want a preview to tell you about.

## Debugging a live run

Two independent presentation layers, both fed by the SAME `log_event()`
call at every site — not two separate recorders (see
`logging_setup.py`'s own module docstring for the full "one
instrumentation path" design) — both behind `--debug` (or
`DEBUG_TRACE=true`):

| Output | Where | What it answers |
|---|---|---|
| JSON lines | **stderr** — visible in a normal terminal by default | "What ran, in what order?" Machine-parseable, unchanged in shape whether `--debug` is on or not. |
| Human-readable execution narrative | `logs/run-<run_id>.txt` **only** — never printed to console, written once at the end of the run | The same event stream, rendered as a story: graph construction, an execution-plan preview, one section per node (`INPUT`/`DECISION`/`NEXT`), parallel search tasks serialized into one block per task, sectioned telemetry, and a final request summary with elapsed-time markers. |

```bash
# one run, both streams captured separately
python -m research_agent.cli "your question" --debug --thread-id demo 2> run.log 1> report.txt

# every node that fired, in order
grep '"msg": "node.enter"' run.log        # bash
Select-String '"msg": "node.enter"' run.log   # PowerShell

# the graph's static wiring, no run at all
python -m research_agent.cli --print-graph
```

This is also how to calibrate `min_evidence_score`/`min_similarity` for your
own corpus rather than trusting the defaults blindly — run one on-topic and
one off-topic query with `--debug`, and compare the `similarity` values that
show up for each (either in `run.log`'s JSON lines, or the corresponding
`SEARCH TASK`/`SEARCH RESULTS` block in `logs/run-<run_id>.txt`).

## The HITL Investigation

`OPERATIONS.md` tells you to switch `HITL_ENABLED=true`, ask
*"Compare Redis vs Cassandra vs DynamoDB at petabyte scale"*, and watch the CLI
pause with `action [approve/redirect/abort]:`. As of two revisions ago, it
didn't. This section is the story of why, what was fixed, and what a real
run confirmed since.

> **Status check, read this before the rest of the section:** the escalation
> machinery has now fired live — a real run reached `human_escalation` with
> trigger `E3`, paused, and resumed correctly on `approve`. But be precise
> about what that run actually proved: retrieval failed outright
> (`NotFoundError` from Qdrant — the corpus collection was empty), so recall
> hit `0.0` via the pre-existing D-16 failed-task path, not via the P2-01
> exact-boundary fix described below. That fix is separately confirmed by
> reconstructing the precise `score=0.5` artifact a different real run
> produced (see [Recent Fixes](#recent-fixes)), but the specific scenario
> this section walks through — retrieval SUCCEEDING with low-relevance
> evidence, not failing — has not yet been the thing that triggered a live
> escalation on its own. Both paths are real and both are fixed; treat the
> walkthrough below as accurate for what it diagnoses, with that one
> distinction kept in mind.

### The machinery is real

This needs saying first, because the conclusion is easy to misread as "HITL is
a stub". It is not. `agents/escalation.py` is a fully wired, tested,
interrupt-based human station:

```text
   node raises a trigger            routing reads it          escalation node
  ┌─────────────────────────┐   ┌──────────────────────┐   ┌───────────────────┐
  │ goal_manager   ─► E1    │   │ route_after_goals    │   │ 1. build payload  │
  │ progress_check ─► E2/E3 ├──►│ dispatch_tasks       ├──►│    (PURE READ)    │
  │ gap_generator  ─► E2/E3 │   │ route_convergence    │   │ 2. interrupt()    │
  │ critic         ─► E4    │   │ route_after_critique │   │    FIRST effectful│
  └─────────────────────────┘   └──────────────────────┘   │    statement      │
    checks WRITE state            routers only READ        │ 3. map (trigger,  │
    (routers cannot)                                       │    action) ─►     │
                                                           │    Command(goto=) │
                                                           └─────────┬─────────┘
                                                                     │
                 ┌───────────────────────┬───────────────────────────┤
                 ▼                       ▼                           ▼
             approve                  redirect                    abort
        ┌──────────────┐       ┌───────────────────┐      ┌─────────────────┐
   E1   │ ─► compiler  │       │ ─► goal_manager   │      │ ─► compiler     │
        │  (error rpt) │       │  + human_guidance │      │  + abort_reason │
   E2/3 │ ─► compiler  │       │ ─► gap_generator  │      │ ─► compiler     │
        │  (ship thin) │       │  + human_guidance │      │  + abort_reason │
   E4   │ ─► telemetry │       │ ─► compiler       │      │ ─► telemetry    │
        │  (unreviewed,│       │  + note "HUMAN    │      │  (no memory)    │
        │   no memory) │       │    REVIEWER: …"   │      │                 │
        └──────────────┘       └───────────────────┘      └─────────────────┘

  D-28 made concrete: the whole node RE-EXECUTES from its top on resume.
  Everything above interrupt() is a pure read, and escalation_history is
  appended in the RESUME update — never before — so re-execution cannot
  double-append. Six tests in tests/integration/test_hitl_escalation.py
  assert exactly that from
  the outside: one history entry per escalation, despite two executions.
```

The CLI loops on `"__interrupt__" in result` and resumes with
`Command(resume={...})` under the same `thread_id`; the API returns
`status: "interrupted"` and exposes `POST /resume`. All six HITL tests still
pass, unaffected by the fixes below — they test the escalation node's own
logic, not the coverage calculation that used to prevent it from ever being
reached for this specific query.

### So why didn't the documented query escalate — and what changed

Because **recall was 1.0 before anything could go wrong.** The trace of
`run-0d7d0448906a` ran straight through — `classify → memory_retrieve →
goal_manager → task_expander → 6 workers → compiler → critic → END` — at
`iteration_depth = 1`, with no `gap_generator` and no interrupt.

```text
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. MIN_EVIDENCE_SCORE defaulted to 0.0            [FIXED — now 0.5]  │
  │    the coverage predicate WAS  e.score >= min_evidence_score         │
  │    at 0.0 that was TRUE for every item — even one scored exactly 0.0 │
  │    (the predicate is now the STRICT  e.score > min_evidence_score,   │
  │     which additionally closes a separate exact-boundary collision    │
  │     — see the P2-01 follow-up row in Recent Fixes)                   │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 2. No relevance floor existed ANYWHERE       [FIXED — min_similarity]│
  │      QdrantStore.search      ─► top-k neighbours, unconditionally    │
  │      HybridRetriever.search  ─► NOW drops dense hits below the floor │
  │                                   BEFORE fusion (P2-01)              │
  │      corpus_search           ─► converts every SURVIVING hit         │
  │    an out-of-domain query can now, in principle, produce zero        │
  │    evidence — not yet confirmed end-to-end against a live run (P2-05)│
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 3. task_expander emitted one task per goal (g1…g5)                   │
  │    every goal received 3 off-topic Redis documents, 0.48–0.50        │
  │    — these would now be DROPPED before reaching Evidence, since      │
  │    0.48-0.50 sits below the new 0.5 min_evidence_score floor too     │
  │    (belt and braces: two independent gates, either alone sufficient) │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 4. route_convergence: recall 1.0 >= RECALL_TARGET 0.85 ─► compiler   │
  │    (this is what USED TO happen; with 1-2 above fixed, recall on     │
  │    this exact query should now come back lower — not yet verified    │
  │    against a live run with both fixes active together)               │
  └──────────────────────────────────────────────────────────────────────┘
```

E4 did not fire either: both critics returned `passed: true` on the **first**
pass — including on a report whose Cassandra and DynamoDB sections were
supported by no retrieved evidence whatsoever. Self-critique by the same
model family is optimistic; that is what it costs. **P2-11 (judge-model
quality scoring, Tier 3) directly addressed this** by making the quality
judge always the next provider in the fallback chain rather than the
answering one — it closes the same-model-optimism half of the problem;
there is still no programmatic, claim-by-claim grounding check, which
remains a separate, genuinely open gap (see Limitations).

Reproduced deterministically before the fix, no services required:
`hitl_enabled=True`, stub LLM, and a tool returning one `score=0.0` item per
task → `recall: 1.0, iterations: 1`, no interrupt. **Re-running that same
reproduction with the new defaults is the first thing P2-05 should do** —
if `min_evidence_score=0.5` alone is enough to reject a `score=0.0` item (it
should be, trivially), that specific stub reproduction should now correctly
raise E3, and the interesting remaining question is whether it does the same
against real, live retrieval.

### The contributing defect: memory `goal_id` collision — now namespaced

`SemanticMemory.retrieve` used to rebuild `Evidence` with
`goal_id=h.get("goal_id", "memory")` — the goal id of **whichever earlier run
stored that fact**. Goal ids are `g1, g2, g3…` in every single run, so:

```text
  run 1: "Compare Redis and Memcached for session caching"
         goals g1…g5  ─►  memory_writer stores evidence tagged goal_id=g3
                                        │
                                        ▼
  run 2: "Compare Redis vs Cassandra vs DynamoDB at petabyte scale"
         goals g1…g5  ◄── memory_retrieve returns it
         BEFORE (bug):  still tagged bare "g3" ─► FALSELY matches current g3
         AFTER (P2-02): tagged "memory::g3" ─► can NEVER match current "g3"

  Worse, unaffected by this fix: memory items still typically score ~0.75,
  corpus hits ~0.50 — memory still OUTRANKS fresh retrieval in the
  compiler's evidence listing. Namespacing stops memory from falsely
  satisfying COVERAGE; it doesn't change memory's relative ranking in the
  final report's context construction.
```

The original trace showed five such items — all the same Memcached
throughput sentence, tagged `g2`/`g3`/`g5` — entering a petabyte-scale
comparison as covering evidence. With P2-02 in place, an item like that would
now retrieve as `goal_id="memory::g2"` etc. — still usable as informational
context in the compiled report (it's still real, relevant-or-not evidence,
still shown to the reader), but structurally incapable of marking a *current*
goal "covered" by id collision alone.

### One more trap: the environment variable

The variable is `HITL_ENABLED` (field `hitl_enabled`), and that is what
`.env.example` and this README use. But `Settings` is configured with
`extra="ignore"`, so a line reading `HITL=true` in your `.env` is **silently
accepted and does nothing** — no warning, no error, HITL stays off:

```text
  HITL=true          ─►  settings.hitl_enabled = False    ← silent no-op
  HITL_ENABLED=true  ─►  settings.hitl_enabled = True
```

If HITL "isn't working", check the variable name before you check the code.

### What it takes to make the documented test real — updated status

| # | Change | Status |
|---|---|---|
| 1 | Raise `MIN_EVIDENCE_SCORE` above the measured off-topic floor | **Done** — now `0.5`, was `0.0` |
| 2 | Namespace memory goal ids on retrieval | **Done** — `memory/semantic_memory.py::retrieve` |
| 3 | Apply a similarity floor before hits become Evidence | **Done** — `retrieval/hybrid.py`, `min_similarity` |
| 4 | Require at least one `source == "corpus"` item for coverage, or weight memory below fresh retrieval | Not done — still open |
| 5 | Add an integration test asserting an interrupt for an out-of-corpus query | **Done** — two tests in `tests/integration/test_hitl_escalation.py`, both passing (**157/157** total, current suite). **Live confirmation exists too**, though via the retrieval-failure path (D-16) rather than the low-relevance-evidence path these tests specifically target — see the status note at the top of this section |

Items 1–3 were each individually sufficient to make E3 reachable for the
documented query; all three are now in place together. Item 5 is what turns
"should work now" into "verified to work, and guaranteed to keep working."

## Telemetry — read it honestly

`telemetry_node` aggregates counters that nodes recorded. It invents nothing,
exactly as D-12 requires. **As of P2-07, the counters are boundary-scoped,
not just node-scoped** — the single biggest gap this section used to
describe is now closed, on both the LLM side and the retrieval side:

| Field | Counts | Boundary |
|---|---|---|
| `llm_node_calls` | one per LLM-using **node execution** — renamed from `llm_calls` for honesty; a node that fell through two fallback hops still counts as one | node |
| `llm_provider_calls` | one per **real provider attempt**, win or lose — fallback hops now visible | `llm/router.py::FallbackRouter` |
| `llm_fallback_hops` | one per actual hop to the next provider (error or low-quality) | same |
| `llm_quality_calls` | one per **judge-scoring call** (`compiler_node`'s free-text path only — the only path with a quality gate). Not "self-scoring": P2-11 made the judge the NEXT provider in the chain, never the model being judged. Counts ATTEMPTS, so it includes calls that failed open; `llm_quality_calls_failed` counts those separately | same |
| `llm_quality_scores_judged` *(D-106)* | one per **real** judgement — a scoring call that failed open produced a fabricated `1.0` and is deliberately NOT counted here. This, not `llm_quality_calls`, is the only safe denominator for a mean: counting fail-opens would make a dead judge read as a generous one | `llm/router.py::_record_quality_score` |
| `llm_quality_score_mean` *(D-106)* | mean of the real judgements, 3dp. **`None`, never `0.0`,** when nothing was judged — `0.0` is a score, and no run should report one it never received | `agents/compilation.py::telemetry_node` |
| `llm_quality_rejections` *(D-106)* | real judgements that fell below `LLM_QUALITY_THRESHOLD`. Read against `llm_quality_scores_judged` for the rejection rate | `llm/router.py::_record_quality_score` |
| `llm_quality_bands` *(D-106)* | the score distribution as `{band: count}` over fixed bands (`<0.2 / <0.4 / <0.6 / <0.8 / rest`), emitted only for bands that occurred. Bands rather than a min/max pair because counters merge by ADDITION across parallel nodes (`state.py::merge_counters`) — a running minimum would be silently wrong the first time two nodes judged in one superstep, while a histogram sums correctly by construction. **Fixed and independent of the threshold**, which is what lets them show whether `LLM_QUALITY_THRESHOLD` sits in the right place | same |
| `retrieval_dense_calls` / `retrieval_keyword_calls` | one pair per real `HybridRetriever.search()` attempt, bumped before either leg is even queried — so an attempt that raises partway through still counts | `retrieval/hybrid.py::HybridRetriever` |
| `retrieval_leg_unavailable` | counts a store being unreachable **at the moment of the call** — now includes a leg going down MID-RUN, not only one down at boot (per-leg failure isolation added this session), so this field is a more complete signal than in earlier revisions | same |
| `producer_rejects` | malformed goal/task dicts the LLM returned, dropped by P2-06's validation instead of crashing the run | `agents/task_utils.py`, `agents/planning.py::goal_manager_node` |
| `search_calls` | one per **successful worker** | node |
| `search_failures` | one per worker that raised | node |
| `memory_hits` / `memory_writes` | items in / points out | node |
| `revision_cycles` | critic passes | node |
| `escalations` *(new this session)* | `[{"trigger":..., "action":...}]` — every entry from `state.escalation_history`, previously written and never read anywhere | `agents/compilation.py::telemetry_node` |
| `goals_without_evidence` | the goal ids that reached the compiler with **zero** evidence attached — counted from `state.evidence`'s own `goal_id` field, never parsed out of the report | same |
| `grounding_ratio` | `(goals − goals_without_evidence) / goals`, rounded to 3dp; `0.0` when a run produced no goals at all. **Deliberately distinct from `recall`**: recall asks "did enough evidence clear the coverage threshold" and is score-derived, so threshold tuning moves it; `grounding_ratio` asks the cruder prior question "did this goal get ANY evidence", which no threshold can affect. `recall: 1.0` with `grounding_ratio: 0.5` means the coverage rule is passing goals the retriever never fed | same |
| `corpus_recall` *(D-38/D-39)* | same shape as `recall`, but counts a goal covered only if a `corpus`/`mcp` item both cleared `min_evidence_score` AND passed the D-39 topical gate (shared distinctive terms with the goal description). Exists because `recall` alone can no longer tell you whether the CORPUS answered a goal or the model tier did — the two are frequently different, and a large gap between `recall` and `corpus_recall` means the answer is recollection, not retrieval | `agents/compilation.py::telemetry_node` |
| `web_sourced_items` *(D-57)* | count of `state.evidence` entries with `source == "web"` — results from the Phase 4 search tier. Read together with `corpus_recall` and `grounded_score`: a run showing `recall: 1.0, grounded_score: 0.0, web_sourced_items: 12` is telling you precisely and honestly that the web answered it and the corpus did not. Web evidence deliberately does NOT count toward `corpus_recall` or `grounded_score` — a snippet is retrieval, not curation | `agents/compilation.py::telemetry_node` |
| `web_source_domains` *(D-57)* | how many DISTINCT domains that evidence came from. Twelve items from one domain is one source repeated, not twelve agreeing — `websearch/filtering.py::cap_by_domain` limits this at retrieval time (`WEB_SEARCH_MAX_PER_DOMAIN`, default 2) and this field makes it visible after the fact | same |
| `web_sources_listed` / `web_sources_suppressed` *(D-57)* | from `compiler_node`’s `append_web_sources` pass: how many web pages the report actually attributed in its `## Sources` section, versus how many were retrieved but belonged to goals the compiler never cited. A high suppressed count against a low listed count is the signature of the web tier doing work that never reached the report | `guardrails/sources.py` |
| `cited_figures_checked` / `cited_figures_unsupported` / `unsupported_figures` *(D-91)* | the claim-level check. `checked` is how many cited figures the shipped report stated at all; `unsupported` how many appear in no evidence under the goal the sentence cites; `unsupported_figures` is a capped sample naming WHICH, so the number is actionable without re-running anything. **`0 / 0` means the report stated no cited figures — not that it passed.** Read against `corpus_recall` and `grounding_notice_shipped`: a run answering from the web with unsupported figures in its prose is a very different artifact from one answering from the corpus with none | `guardrails/claims.py` |
| `grounding_notice_shipped` *(D-85)* | whether the SHIPPED report carries the deterministic provenance notice. Derived from `state.final_report`, never from a counter — D-59's rule, because `compiler_node` runs once per revision and its counters merge additively. Read together with `corpus_recall`: `corpus_recall 0.0` with this `true` is a run that answered from the web or from recollection **and told its reader so**; the same pair reading `false` is a deliverable that claimed nothing about its own provenance, which `report.shipped_ungrounded` now WARNs about | `guardrails/grounding.py` |
| `llm_context_skips` *(D-93)* | provider hops SKIPPED because the prompt clearly could not fit that provider's configured context window. `0` unless `LLM_PRIMARY_CONTEXT_TOKENS` is set. A nonzero value is the count of guaranteed-failed provider calls this run did **not** make — read against `llm_provider_calls`, which no longer includes them | `llm/router.py::_skips_for_context` |
| `llm_prompt_tokens` / `llm_completion_tokens` / `llm_total_tokens` *(D-86)* | what the run actually COST, as opposed to how many requests it made. `llm_provider_calls` cannot distinguish three cheap `classify` calls from three 7,000-token `compiler` calls; these can. Counted at the router boundary from the usage each provider reports, judge calls included. Tokens rather than dollars deliberately: every `LANGFUSE_PRICE_*` defaults to `0.0`, so a spend figure built on them would be structurally zero, while tokens are real whether or not a rate was ever configured | `llm/router.py::_bump_usage` |
| `tier_answers` / `chain_tier_failures` / `chain_exhausted` *(D-87)* | WHICH tier of the D-38 ladder actually answered, as `{tier: count}` — previously readable only by grepping `chain.answered` out of a debug trace. Read against `corpus_recall`: `{"corpus": 6}` at `corpus_recall 1.0` is a healthy corpus run; `{"web": 6}` at `corpus_recall 0.0` is the p205.246-check shape, now one field instead of three inferred ones. A tier that answered nothing is omitted rather than reported as `0`. Note `chain_tier_failures` counts TIER attempts, not tasks — tiers 1 and 2 are the same tool, so one dead corpus fails both | `tools/retrieval_chain.py` |
| `model_sourced_items` *(D-38)* | count of `state.evidence` entries with `source == "model"` — the LLM's own knowledge, retrieved deliberately because no document served that goal. Read together with `corpus_recall`: `corpus_recall: 0.0, model_sourced_items: 24` means the whole report rests on recollection, attributed as such in the prose (D-40) | same |
| `last_compile_guardrails` *(D-45, D-88)* | every deterministic repair the SHIPPED report needed, as one nested dict: `citations_pasted_evidence_removed` (verbatim evidence text the compiler glued onto a claim with no delimiter, e.g. `"...the whole session blobRedis is an in-memory data store..."`), `citations_to_unevidenced_goals` (`[gN]` markers removed because goal N retrieved no evidence at all), `hedge_markers_inserted`, `evidence_deduplicated`, `grounding_notice_inserted`. **Corrected in D-88:** earlier revisions of this table documented the first two as top-level telemetry fields; they were never emitted there at all, and reading them out of `counters` would have been wrong anyway — `compiler_node` runs once per REVISION and `counters` merges additively, so that view sums every compile ATTEMPT. This field is replace-on-write and describes the artifact the reader received. A key absent means that repair wasn't needed on the final pass | `state.py::last_compile_guardrails` |

A real live trace showed this working correctly under genuinely messy
conditions — two provider timeouts, a low-quality rejection, and a 429 —
in one run: `llm_node_calls: 8, llm_provider_calls: 11, llm_fallback_hops: 3,
llm_quality_calls: 1`. Every one of those four numbers was verified by hand
against the corresponding `llm.call`/`llm.fallback`/`llm.quality_reject` log
lines in that same trace and matched exactly. A separate run with real
corpus retrieval showed `retrieval_dense_calls: 6, retrieval_keyword_calls:
6, retrieval_leg_unavailable: 0`, matching 6 real `search_worker`
invocations, each with both legs answering (`"dense": 3, "keyword": 3` in
every `retrieval.hybrid` log line that cycle).

**What this does NOT fix, and never claimed to:** `llm_node_calls` still
counts node executions, not provider requests — that's the *point* of the
rename, not a residual bug. If you need "how many nodes touched an LLM at
all" that's still the right field; if you need real provider traffic or
spend, use `llm_provider_calls`.

**The narrative log is still the honest view for exact prompt/response
detail, but it's not the only signal for volume.** `--debug` (or
`DEBUG_TRACE=true`) records every LLM call and every retrieval call at the
boundary it actually crosses, to `logs/run-<run_id>.txt` — and every node
now gets its own section there too, including `merger` and
`progress_checker`, which touch neither an LLM nor a store but still get a
plain `NODE:` heading with timing, not just a `"node.enter"` line to
stderr. See [Debugging a live run](#debugging-a-live-run) for exactly how
to use both together. In one traced run, this combination revealed
something telemetry alone never would: **OpenSearch never appeared at
all**, because the keyword leg was down — the "hybrid" retriever was
running single-legged, and nothing in the report or the (then
node-scoped-only) telemetry said so. `retrieval_leg_unavailable` now
surfaces that same fact directly in the telemetry block itself, without
needing the narrative log.

## Design

- **Assembly** (`assembly.py`): the whole dependency graph, in one place —
  `AppBundle` and `build_app_and_settings`. Deliberately a neutral module
  rather than part of `cli.py`, so the HTTP surface does not import its
  startup path from a command-line module. `cli.py` re-exports both names
  for backward compatibility.
- **Orchestration** (`orchestration/`): the graph topology and routing live in
  `graph.py`; the worker return contract (`contracts.py`) turns a
  non-deterministic concurrency bug into a deterministic unit-test failure.
- **State** (`state.py`): every field parallel workers write carries a reducer.
  Read the reducer docstrings first — they are the concurrency model. Note the
  standing rule in `merge_counters`: **monotonic countables only, never
  durations** — two workers writing 150ms and 200ms would "merge" to 350ms of
  nothing.
- **LLM routing** (`llm/`): one OpenAI-compatible client serves all three
  providers; fallback policy lives in exactly one place (`router.py`), which
  now also owns the per-hop timeout split (local vs. cloud). Both the router
  and the underlying client now expose `close()` (this session), called on
  CLI exit, so the httpx connections per provider are no longer leaked.
- **Retrieval** (`retrieval/`, `storage/`, `tools/`): storage modules are
  policy-free wrappers; fusion math is a pure function; the tool translates
  hits into domain Evidence. `HybridRetriever` now also owns the
  `min_similarity` relevance floor, and (this session) per-leg failure
  isolation at runtime, not just at boot. `tools/corpus_search.py` is the MCP
  seam — the calling pattern is already MCP-shaped, so the upgrade touches
  one module.
- **Memory** (`memory/`): similarity × volatility-decay reranking; memory items
  re-enter the graph as ordinary evidence, so every downstream rule (coverage,
  contradiction) treats memory and fresh sources identically — **except for
  goal-id equality specifically**, which is now deliberately asymmetric after
  P2-02, so memory can inform but not falsely satisfy coverage.
- **Evaluation** (`evaluation/`): the cross-provider quality signal behind fallback,
  now judged by the next provider in the chain, never the answering one.
- **Escalation** (`agents/escalation.py`): one parametrized node for all four
  triggers, carrying the D-28 idempotency obligation. `escalation_history`
  now surfaces in telemetry (this session), rather than only in Postgres.

## Project Structure

```text
research-agent-dmp/
├── src/research_agent/
│   ├── assembly.py          # AppBundle + build_app_and_settings — the whole
│   │                        # dependency graph; imported by BOTH cli.py and
│   │                        # api/server.py (cli.py re-exports it)
│   ├── config.py            # all tunables, validated, from .env
│   ├── state.py             # entities, graph state, reducers (read first)
│   ├── logging_setup.py     # JSON-lines structured logging + run_id
│   ├── tracing.py           # Tracer / NullTracer behind --debug
│   ├── llm/                 # client (real + stub) and 3-hop fallback router
│   ├── prompts/             # every prompt, one place; TASK=<tag> drives stub
│   ├── agents/              # planning / gathering / compilation / escalation
│   ├── orchestration/       # graph wiring + worker contract enforcement
│   ├── retrieval/           # hybrid dense+BM25 with RRF + relevance floor
│   ├── memory/              # semantic memory with decay + namespaced ids
│   ├── storage/             # Postgres / Qdrant / OpenSearch wrappers
│   ├── langfuse/             # Phase 3: optional Langfuse tracing, the only
│   │                        package that imports the langfuse SDK — see
│   │                        Observability — Langfuse (Phase 3) above
│   ├── tools/                # the retrieval tools workers invoke (D-38 ladder):
│   │                        retrieval_chain.py (the 5-tier ladder itself),
│   │                        corpus_search.py, model_knowledge.py, and
│   │                        mcp_client.py — MCPBridge, shared by BOTH MCP
│   │                        bridges: make_mcp_tool (corpus, P2-13,
│   │                        MCP_ENABLED) and make_web_search_tool (Phase 4,
│   │                        WEB_SEARCH_ENABLED). Streamable HTTP only
│   │                        (D-76): a pure network client connecting to an
│   │                        independent, standalone server you start and
│   │                        stop yourself — this process spawns nothing
│   ├── websearch/            # Phase 4 (D-57): SearchProvider protocol, DDGS
│   │                        provider, rank→score band, dedupe/domain-cap.
│   │                        Imported ONLY by scripts/mcp_web_search_server.py
│   │                        (a separate process) — the agent never imports a
│   │                        search client, which is what keeps the rest of
│   │                        the codebase uncoupled from the chosen engine
│   ├── guardrails/           # grounded convergence, hedging, citation repair,
│   │                        fencing, and sources.py (D-57: the deterministic
│   │                        `## Sources` pass for cited web evidence)
│   ├── evaluation/          # answer quality, judged cross-provider
│   ├── api/server.py        # FastAPI: /health, /research, /resume
│   └── cli.py               # CLI entry + dependency assembly + HITL loop
├── tests/                   # offline. Organized by module,
│                              mirroring src/research_agent/'s own layout:
│                              tests/unit/<module>.py (one file per source
│                              module) + tests/integration/<scenario>.py
│                              (full graph.invoke() runs). See
│                              OPERATIONS.md "Running and Interpreting the
│                              Test Suite" for the full file-by-file map
│                              and how to read the current count (M-4: a
│                              literal count here goes stale on the next
│                              added test — run `pytest -q` for the real
│                              number).
├── scripts/ingest_sample_data.py
├── scripts/reset_stores.py  # wipe all three stores to pristine (see above)
├── scripts/mcp_corpus_server.py  # real MCP server wrapping the corpus tool 
│                                    (P2-13, off by default via MCP_ENABLED=false)
├── scripts/mcp_web_search_server.py  # Phase 4 MCP server exposing web search
│                                    (D-57, off by default via WEB_SEARCH_ENABLED=false)
├── scripts/analyze_runs.py       # read-only: cross-run analysis over agent_runs
│                                 (D-92) — which tier answers, how often the
│                                 corpus grounds anything, what a run costs.
│                                 The reader that made a separate "strategy
│                                 memory" store unnecessary
├── scripts/inspect_memory.py     # read-only: what long-term memory holds, and
│                                 what it would recall for a given question (D-90).
│                                 The only read path into memory that does not
│                                 require running a full research query
├── scripts/check_services.py     # health check: Qdrant/OpenSearch/Postgres/LLM/
│                                 MCP/web-search/API (D-33). The web-search row is
│                                 the ONLY live verification of that path — the
│                                 test suite is entirely offline by design
├── sample_data/corpus.jsonl # 10 docs, Redis-vs-Memcached theme
├── design/Research_Agent_Design.md
├── OPERATIONS.md   internal/LEARNING_GUIDE.md   internal/PHASE-2_PLAN.md
├── docker-compose.yml       # optional: Postgres + Qdrant + OpenSearch
├── pyproject.toml           # packaging: extras, console script, public API
│                              and versioning policy — see Packaging below
├── requirements.txt          # core; does NOT install the web-search client
├── requirements-websearch.txt  # Phase 4 only: ddgs + mcp. Install into the SAME
│                              venv that runs the agent — see Setup
├── .env.example  run.bat  reset.bat
└── DECISIONS.md             # populated: D-1..D-79, sourced from code comments
```

## Setup

`OPERATIONS.md` is the real manual — it owns the L1/L2/L3 ladder, native
Windows service startup, DBeaver setup, and the llama-server invocations. 
The 30-second version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults run fully offline (LLM_MODE=stub)
export PYTHONPATH=src

python -m research_agent.cli "Compare Redis and Memcached for session caching"
python -m pytest tests/ -q    # expect: all green — see the summary line for the count (M-4)
```

**Or install it as a package** (`pyproject.toml`, new) — which is what
another project consuming this over HTTP would do, and which removes the
need for `PYTHONPATH=src` and gives you a `research-agent` console script:

```bash
pip install -e .            # core only: no FastAPI, no MCP, no Langfuse
pip install -e ".[api]"     # + the HTTP surface
pip install -e ".[websearch]"  # + Phase 4 web search (mcp + ddgs)
pip install -e ".[all]"     # everything requirements.txt installs

research-agent "Compare Redis and Memcached for session caching"
```

See [Packaging](#packaging) for the extras, the public API, and the
versioning policy.

Windows: `run.bat` does the venv, install, and a stub run in one command.

**Web search (Phase 4) is opt-in and does not affect the steps above.**
`WEB_SEARCH_ENABLED=false` is the default, the `ddgs` client is not in
`requirements.txt` at all, and the agent process never imports it unless
you turn this on. D-76: it is also a standalone server you run yourself,
separately from the agent — there is nothing for the agent process to
spawn.

```bash
pip install -r requirements-websearch.txt   # into a venv for the SERVER

# In its own terminal, left running:
python scripts/mcp_web_search_server.py --port 8766

# .env — these two lines are the whole minimum
WEB_SEARCH_ENABLED=true
WEB_MCP_SERVER_URL=http://127.0.0.1:8766/mcp

python scripts/check_services.py            # look for the 'Web search (MCP)' row
```

Full walkthrough, including the standalone-server terminal setup:
`OPERATIONS.md` → *Enabling Web Search (Phase 4)* and *Running the MCP
servers standalone*.

**MCP servers always run standalone (D-76), independent of any single
run.** Both the corpus and web-search MCP servers are ordinary,
independent processes you start yourself, in their own terminals, and
leave running — the agent process never spawns either one, and stopping
a `research_agent` run never stops them. Point the agent at an
already-running server with `MCP_SERVER_URL` / `WEB_MCP_SERVER_URL`.
Full start/stop workflow: `OPERATIONS.md` → *Running the MCP servers
standalone*.

Langfuse tracing is opt-in and does not affect the steps above: `langfuse` is
an installed dependency (`requirements.txt`), but `LANGFUSE_ENABLED=false`
(the `.env.example` default) means it is never imported and makes no network
calls. Flip it on later per [Observability — Langfuse (Phase 3)](#observability--langfuse-phase-3).

Defaults are `LLM_MODE=stub` with every store unreachable, so the first run
reports `evidence_items: 0`. **That is success for L1** — the graph is proven,
there is simply nothing to search yet. `OPERATIONS.md` walks you up from there.

## Packaging

`pyproject.toml` makes this repo an installable artifact. Until it existed,
the only way to run this code was `PYTHONPATH=src` from inside a checkout,
which gives a *separate* project nothing to depend on.

**Extras.** Each is optional because its code path is off by default *and*
its import is lazy — checked against the source, not assumed:

| Install | Adds | Why it can be optional |
|---|---|---|
| `pip install research-agent` | core: langgraph, pydantic, httpx, qdrant-client, fastembed, opensearch-py, psycopg | `assembly.py` imports every storage client at module level, so a core install must have them; they degrade at *runtime* when a server is unreachable, not at import |
| `[api]` | fastapi, uvicorn | nothing outside `api/server.py` imports either |
| `[mcp]` | mcp | `MCP_ENABLED=false` by default; `tools/mcp_client.py` has no module-level `import mcp` |
| `[langfuse]` | langfuse | `LANGFUSE_ENABLED=false` by default; `langfuse/client.py` returns `None` before importing the SDK |
| `[viz]` | grandalf | `--print-graph` falls back to Mermaid text without it |
| `[dev]` | pytest | — |
| `[all]` | everything above | matches what `requirements.txt` installs today |

**Console script.** `research-agent = research_agent.cli:main`, so an
installed package exposes the CLI without `python -m`.

**Public API — what a MAJOR version bump is owed for.** Stated explicitly
because "it's all importable" stops being an answer once another project
depends on you:

```text
  research_agent.assembly     build_app_and_settings(), AppBundle
  research_agent.api.server   /research, /resume, /health request+response shapes
  the `research-agent` console script's arguments
  the .env setting NAMES in config.py
```

Everything else — `agents/`, `orchestration/`, `retrieval/`, `prompts/`, and
the internals of `langfuse/` — is internal and may change in a MINOR release.

**Versioning.** Manual SemVer in `pyproject.toml`'s `version` field, bumped
in the same commit as the change it describes, and tagged. Deliberately not
derived from git: that adds a build-time dependency and hides the version
from anyone reading the file. PATCH = bug fix, MINOR = new capability with
existing callers unaffected, MAJOR = a consumer must change code.

**Releasing / consuming.** Until this is published to an index, a consuming
project pins a tag:

```bash
pip install "research-agent[api] @ git+ssh://...@v0.3.0"
```

Pin the tag, never a branch.

**`requirements.txt` still works and is unchanged** — it is now the
development pin-set, while `pyproject.toml` is what a consumer resolves
against. Add a dependency to both, or they drift.

`requires-python = ">=3.11"` because that is what this project is actually
tested on. The source needs 3.10 at minimum (`api/server.py` uses PEP 604
`str | None` with no `from __future__ import annotations`), but claiming
3.10 support without running the suite on 3.10 would be an untested
assertion.

## Walkthrough

1. **A request arrives** (CLI or API) → `build_app_and_settings()`
   (`assembly.py`) wires every dependency, each storage module probing its
   service and degrading if absent.
2. **Plan**: `classify` labels the intent → `memory_retrieve` recalls related
   past evidence (decay-reranked, goal_id namespaced) → `goal_manager`
   composes goals (memory hints included) → `task_expander` emits a ranked
   backlog capped at `MAX_FANOUT` — overflow is the *producer's* decision
   (D-13), so dispatch is always total.
3. **Gather (cyclic)**: `dispatch_tasks` fans one `Send` per task to
   `search_worker` instances that run in the same superstep; each returns only
   reducer-backed keys (enforced), recording success or failure-with-depth.
   Dense hits below `min_similarity` are dropped before they ever become
   Evidence. `merger` flags contradictions; `progress_checker` computes
   quality-gated, contradiction-aware recall (now against a meaningful
   `min_evidence_score`) and ticks the depth counter. Below target and depth:
   `gap_generator` produces new tasks (dedup + failed-key rules applied) and
   the cycle repeats.
4. **Decisions**: the graph decides *where to go*; models decide *content*.
   Termination is guaranteed by four independent bounds: recall target, depth
   counter, empty-backlog fallthrough, recursion-limit backstop.
5. **Compile & critique**: the report is drafted, judged for faithfulness and
   completeness only, and rewritten against explicit notes up to
   `MAX_REVISIONS` (**code default 2**, not the 3 in design §9); exhaustion
   either interrupts for human review or logs the E4 stub and ships the report
   marked unreviewed — never silently. **Corrected this pass:** the compiler's
   free-text output now passes through `strip_code_fence()` before being
   stored — a fallback provider that wraps its answer in a code fence (or
   echoes the evidence-fencing tag back literally) no longer leaks that into
   the final report.
6. **Persist & learn**: a *passed* report's fresh evidence enters semantic
   memory (with its raw, unnamespaced goal_id, per the storage note above);
   telemetry aggregates node-recorded counters (including, this session,
   `escalations`); a run-history row lands in Postgres when available —
   **from the CLI or the API, both** (see P2-08 above).

## Design Decisions

Decision IDs reference the full architecture document that precedes this build.
This table lists only decisions with code behind them here.

| ID | Decision | Why |
|---|---|---|
| D-1 | Empty backlog routes to the compiler | An empty `Send` list would silently halt the graph with no report |
| D-2 | Dedup key sets + replace-on-write backlog | Idempotent dispatch; finite task supply per depth |
| D-3/D-4 | Depth counter + configurable recall target (0.85) | Exact recall ≥ 1.0 degenerates to always-max-depth |
| D-5/D-15 | Reducer-backed worker fields + runtime whitelist | The collision only appears under parallel load — must be impossible, not tested-for |
| D-6 | Workers receive `WorkerPayload`, not full state | Least privilege; keeps the return contract checkable |
| D-8/D-20 | Postgres checkpointer keyed by `thread_id` + `recursion_limit` backstop | Every run resumable and inspectable; a bounded loop still needs a floor |
| D-12/D-19 | Additive counters aggregated at `telemetry` | The logger aggregates, it never invents |
| D-13 | Fan-out capped at the producers | Overflow is a ranking decision, not a dispatcher accident |
| D-14 | Two termination points: convergence (recall/depth) and dispatch (backlog) | The backlog is stale at convergence time; judge it where it's fresh |
| D-16 | Failed ≠ completed; retry at strictly greater depth | Transient backend errors must not permanently burn a query formulation |
| D-17/D-18 | Quality-gated, contradiction-aware coverage | **Now active** — `min_evidence_score=0.5` and the new `min_similarity` floor (P2-01) replace the previously inert `0.0` default; contested goals still drive adjudication automatically, though the detector itself remains marker-only (unchanged, see Limitations) |
| D-21 | Zero goals → explicit error report | Diagnosable beats silent |
| D-22 | Bounded critique, grounded rewrites, scoped to faithfulness | One judge per question; no blind retries |
| D-23/D-28 | Escalation via `interrupt()`; nothing non-idempotent precedes the interrupt | The node re-executes on resume — history is appended in the resume update only |
| D-24 | Memory decay = rerank by volatility class, never an age filter | One TTL is wrong for both stable and volatile facts. Coverage-matching by goal_id is now namespaced away from this rerank (P2-02) — the two are independent axes |
| D-29 | `ConfigDict(extra="forbid")` on all state models | Construction-time pollution and worker-return pollution are two failure modes; two layers |
| D-31 | Store writes carry stable, content-derived identity, not a fresh random id per call | Re-ingesting unchanged content should overwrite in place, not accumulate duplicates — now implemented for the corpus ingest script AND memory writes (P2-15). Evidence.task_key for memory items is the one identity-related thing NOT yet fixed this way (still `hash()`-based) |
| D-32 | Provider output normalization happens at the client boundary (`llm/client.py`), never inside a node or the router | Chat-template sentinels and runaway free-text generation are transport/template artefacts, not content — nodes should never have to know a specific model's quirks. This session extended the same principle to the compiler's free-text output (`strip_code_fence`) |
| D-35 *(Phase 3)* | All Langfuse SDK usage isolated in one package (`langfuse/`); every business module imports only thin, always-safe functions and never sees an SDK/trace/span object | Non-invasive observability must not leak a third-party SDK's shape into business logic, and must be safe to call even when disabled or misconfigured |
| D-36 | External MCP tool inputs travel to the LLM inside an `<external_data>` fence, treated as data never instructions | Prompt-injection surface from third-party tool output must not be trusted at the same level as system/user turns |
| D-37 | The repo is an installable artifact (`pyproject.toml`, optional dependency groups mirroring lazy-import code paths) | A core install genuinely omits FastAPI/MCP/Langfuse; `requirements.txt` stays the dev pin-set |
| D-38 | Retrieval is a LADDER — corpus → reformulated retry → MCP → model’s own knowledge, stopping at the first tier clearing `min_evidence_score` | A corpus that doesn’t contain the subject was reporting a retrieval limitation as an absence of knowledge; the model tier is always last so a real document still wins |
| D-57 | **Phase 4 — web search.** A real search engine as tier 4, reached over its OWN MCP server (`scripts/mcp_web_search_server.py`). `source="web"` evidence COVERS a goal but never GROUNDS one | Tiers 1–3 all resolve to the same ingested documents, so “the corpus does not contain it” fell straight through to recollection. Tagging web results `"mcp"` would have made every snippet inflate `grounded_score` and `corpus_recall` — the exact blindness D-43/D-47 exist to expose |
| D-39 | A tier only counts as "answered" if its evidence shares distinctive terms with the query, not merely a high RRF score | Fixed-k retrieval over a small corpus always returns k results — score alone can never signal "nothing relevant here" |
| D-40 | Citations are goal ids only (`[g1]`) — never pasted evidence text, never internal scores | Live output was gluing source sentences onto claims with no delimiter and leaking `score=0.60`-style bookkeeping into the report |
| D-41 | The model-knowledge tier has hard anti-fabrication limits (no invented named entities, confidence reflects the weakest part of a compound claim) | Self-reported confidence alone does not catch confabrication; this reduces but does not eliminate the failure rate, which is why `corpus_recall` stays in telemetry |
| D-42 | Model recollection never enters durable memory; recalled memory can never re-frame a query | A prior run’s unverified recollection was being stored, then read back indistinguishable from retrieved evidence and silently re-framing an unrelated later query |
| D-43 | Citation correctness enforced deterministically where possible (`_clean_citations`), and asked of the critic where it can’t be (evidence-support judgment) | Code can detect a pasted sentence or an uncited goal; it cannot judge whether cited evidence actually supports a claim — that needs the critic |
| D-44 | Topical gate strictness scales with query specificity; E4 redirect routes to `gap_generator`, not straight back to the compiler | A long, specific query matching only its one broad subject word isn’t a topical match; a redirect asking for new evidence can’t be served by recompiling the same evidence block |
| D-45 | Deterministic citation stripping only removes GLUED pastes (no delimiter before the match), never legitimately-repeated evidence text; prompts carry no concrete worked examples | Stripping every verbatim occurrence deleted whole report sections on in-corpus queries where restating corpus sentences is correct; worked examples in prompts were echoed back as if they were live findings |
| D-46 | The critic is shown the evidence (`state.evidence`) it’s instructed to verify claims against | Asking it to check citations without showing it the evidence made the check unanswerable — it was failing correct, evidence-backed reports on every off-corpus run |
| — | Graceful degradation everywhere | First run must succeed on a bare laptop |
| — | Stub LLM mode | Deterministic offline demo + honest tests using real prompts/schemas |

`DECISIONS.md` (populated as of P2-09, extended in every pass since) is the
authoritative consolidated log, currently D-1 through D-79 — this table is a
curated subset for readability, not a replacement.

## Limitations

Split into what was *deferred by design* and what is simply *broken*, because
conflating the two is how a reference build stops being trustworthy. Items
below that were fixed or added across revisions are marked as such and left
visible rather than deleted, so the history stays auditable.

**Deferred by design**

- **Contradiction detection is minimal**: the machinery (contested goals block
  coverage) is fully wired; the detector only honors explicit markers, which no
  tool sets. Consequence: **E2 has never fired in a real run** — every observed
  escalation would be E3.

**Fixed since the last revision** (kept here, not deleted, for auditability)

1. ~~`MIN_EVIDENCE_SCORE=0.0` makes the coverage gate inert~~ — **P2-01**, now
   `0.5`.
2. ~~No relevance floor anywhere in retrieval~~ — **P2-01**, `min_similarity`
   now filters the dense leg before fusion.
3. ~~Memory `goal_id` collides across runs~~ — **P2-02**, namespaced on
   retrieval.
4. ~~`<\|im_end\|>` breaks JSON parsing from the local primary~~ — **P2-04**,
   sentinel stripping + balanced-brace fallback in `_extract_json`.
5. (separate from the numbered list, but related) ~~one shared 60s timeout for
   every provider~~ — split into a 120s local timeout and a 90s cloud
   timeout.
6. ~~LLM producer output is unvalidated~~ — **P2-06**, `RawTask`/`RawGoal`
   validation; malformed entries dropped and counted, never `KeyError`.
7. ~~Qdrant ingest is not idempotent by default~~ — **P2-03 follow-up**,
   `scripts/ingest_sample_data.py` now passes a deterministic `id_fn`. Does
   not retroactively clean an already-duplicated collection — see Ingest
   identity above.
8. ~~The checkpointer connection is never closed~~ — **P2-08**,
   `close_checkpointer()`, called from both `cli.py`'s `finally` block and
   `api/server.py`'s shutdown handler.
9. ~~The API writes no run history~~ — **P2-08**, `record_run` now called
   from `/research` and `/resume` on completion.
10. ~~`llm_calls` and `search_calls` under-report actual traffic~~ —
    **P2-07**, `llm_provider_calls`/`llm_fallback_hops`/`llm_quality_calls`
    (router boundary) and `retrieval_dense_calls`/`retrieval_keyword_calls`/
    `retrieval_leg_unavailable` (retrieval boundary) now report the real
    boundary crossings; the renamed `llm_node_calls` still reports node
    executions, deliberately, alongside them.
11. ~~E2/E3 emit no log line when HITL is off~~ — **P2-09**, `escalation.stub`
    now fires for E2/E3 too, matching E1/E4.
12. ~~`durable` is discarded by `build_app_and_settings`~~ — **P2-08**,
    surfaced via `AppBundle` and `/health`'s `durable` field.
13. ~~`HITL=true` (typo for `HITL_ENABLED`) is silently discarded~~ —
    **P2-09**, `warn_on_likely_env_typos()` now logs a WARNING for this and
    a fixed list of other plausible env-key typos. (The variable itself
    still requires the exact name `HITL_ENABLED` — this only adds
    visibility when you get it wrong, it doesn't relax the requirement.)
14. ~~MCP deferred~~ — **P2-13**, `tools/mcp_client.py` + `scripts/mcp_corpus_server.py`
    (real server wrapping the existing corpus tool). Off by default
    (`MCP_ENABLED=false`). Originally stdio (spawned by the agent, D-30);
    **D-76** removed that entirely in favor of D-30's other documented
    transport, Streamable HTTP — a standalone server you start and stop
    yourself, independent of any run, reached at `MCP_SERVER_URL`.
15. ~~Single tool, single worker type~~ — **P2-14**, `SearchTask.tool_hint`
    (D-25) routes a task to a named specialist (`"mcp"` today, the only
    one this build has) instead of the default corpus worker.
    `cap_and_filter` validates the hint against what's actually wired in;
    inert with `MCP_ENABLED=false`. **Confirmed usable under real
    concurrency** since the fixes in item 6 below (6 concurrent calls: 13.5s
    wall time vs 79.2s summed -- ratio 1.02, genuinely parallel).
16. ~~Server-side hybrid fusion deferred~~ — **P2-10**,
    `storage/qdrant_store.py::search_with_decay` (Qdrant `FormulaQuery` +
    payload indexes), gated by `MEMORY_SERVER_SIDE_DECAY` (off by
    default). Python path kept permanently as parity oracle.
17. ~~Memory simplifications (no supersession, no GC)~~ — **P2-15**,
    content-hash dedup (`content_id`, exact-match upsert-in-place) +
    `scripts/gc_memory.py` (decay-threshold GC, `--dry-run`/`--yes`
    gated). No per-item volatility classification still applies
    (unchanged).
18. ~~Self-evaluated quality is optimistic~~ — **P2-11**,
    `evaluation/quality.py::score_answer` — the NEXT provider in the
    fallback chain judges, never the answering provider itself.
19. ~~`Connection pool is full, discarding connection` warning under
    concurrent corpus search~~ — `storage/opensearch_store.py` now sets
    `pool_maxsize=20` (was relying on the client library's small
    default). Harmless before this fix (each discarded connection still
    worked, just paid a fresh TCP+TLS handshake); now avoided.
20. ~~No way to check whether Qdrant/OpenSearch/Postgres/the LLM engine
    (and, opt-in, MCP/the FastAPI server) are actually reachable without
    either running a full query or checking each one by hand~~ — **D-33**,
    `scripts/check_services.py`: one command, clear PASS/FAIL/SKIP per
    service, non-zero exit code if anything's down. Alongside this, the
    test suite itself was reorganized (**D-34**) into
    `tests/unit/<module>.py` + `tests/integration/<scenario>.py`,
    mirroring `src/research_agent/`'s own layout.
21. ~~`api/server.py` couldn't even import~~ — **fixed, post-Tier-3
    session.** `AppBundle` had grown a 5th field (`mcp_bridge`, P2-13) but
    `_graph, _settings, _durable, _checkpointer = _bundle` still unpacked
    four — a `ValueError` at import that made the entire API unreachable.
    Fixed to named-field access; `tests/unit/test_api_server.py` (new)
    covers this so the class of regression can't ship silently again.
22. ~~MCP evidence hardcoded `score=1.0`~~ — **fixed, post-Tier-3 session.**
    `tools/mcp_client.py` stamped every MCP-sourced Evidence item at
    `score=1.0`, which cleared `MIN_EVIDENCE_SCORE` unconditionally — the
    exact defect P2-01 fixed on the corpus path, reintroduced on the MCP
    path. Now scored at `settings.min_evidence_score` (never higher), so
    MCP evidence can't single-handedly satisfy coverage.
23. ~~Retrieval only degraded gracefully at BOOT, not mid-run~~ — **fixed,
    post-Tier-3 session.** A store dying after startup used to raise
    straight through `HybridRetriever`, killing the whole task and
    discarding the healthy leg's hits too. Now genuinely per-leg
    fail-safe at runtime — matches what the docstring always claimed.
24. ~~Postgres checkpointer was a single, unpooled connection~~ — **fixed,
    post-Tier-3 session.** Every parallel `search_worker`'s checkpoint
    write serialized behind it. Now a `psycopg_pool.ConnectionPool`
    (falls back to a single connection if `psycopg[pool]` isn't
    installed).
25. ~~No defense against prompt injection via retrieved evidence~~ —
    **fixed, post-Tier-3 session.** Corpus/MCP content is now wrapped in
    `<evidence>...</evidence>` in every prompt that inlines it, with an
    explicit system-prompt clause marking that span as untrusted data,
    never instructions, and forbidding the model from echoing the tag
    itself back literally (added after a live trace showed a fallback
    provider doing exactly that in a citation).
26. ~~`escalation_history` was written and never read~~ — **fixed,
    post-Tier-3 session.** Now surfaced in `telemetry["escalations"]`.
27. ~~CLI exit code was always 0~~ — **fixed, post-Tier-3 session.**
    `main()` now returns 2 on `GraphRecursionError` and 1 when a run ends
    with no telemetry, instead of unconditionally 0. Extended since: 3
    when the `--thread-id` already holds a run (D-20), and 4 on
    provider-chain exhaustion (D-101). Full table under **CLI Exit
    Codes** in `OPERATIONS.md`.
28. ~~Compiler free-text output could leak a wrapping code fence, or echo
    the evidence-fencing tag literally~~ — **fixed, post-Tier-3 session.**
    `strip_code_fence()` (tested against 15 edge cases, including
    punctuated language tags like `c++` that an earlier, buggier version
    of this same fix mishandled) plus the system-prompt clause in item 25.

**MCP corpus server concurrency — fixed, kept visible for auditability**

29. ~~MCP corpus server serialized under concurrent load~~ — **P2-13,
    Tier 3, fixed.** `scripts/mcp_corpus_server.py`'s tool handler used
    to be synchronous; FastMCP called it directly on its single event
    loop with no thread offload (confirmed by reading
    `func_metadata.py::call_fn_with_arg_validation` — `fn(**args)`
    called inline, not via `asyncio.to_thread`). Result: one real MCP
    request blocked the whole server for its full duration (~13s+ for a
    real Qdrant/OpenSearch round trip), so `MAX_FANOUT` concurrent
    requests fully serialized instead of running in parallel — confirmed
    live: 6 concurrent calls that take 14.4s total called DIRECTLY (no
    MCP) took 100+ seconds through MCP, with two additional real
    concurrency bugs (a `MCPBridge.start()` race and a
    `_get_corpus_tool()` thundering-herd race, both since fixed) found
    and fixed along the way but NOT the cause of this specific slowdown.
    Never a correctness bug — every request always completed correctly,
    just slowly. Fixed by making `mcp_corpus_server.py::search` `async
    def` and offloading the blocking `hits_for_query` call to a
    dedicated `ThreadPoolExecutor` sized by the new `mcp_max_workers`
    setting (default 6, matching `MAX_FANOUT`) via
    `loop.run_in_executor(...)` — functionally equivalent to the
    `asyncio.to_thread(...)` approach originally proposed here, with an
    explicitly bounded pool instead of the default executor.
    `MCP_ENABLED=false` (default) is unaffected either way; P2-14's
    tool_hint routing is now usable at real concurrency.

    **A second, separate stall found and fixed after the above:** even
    with the async/executor fix in place, the FIRST ever `search()` call
    in a real deployment still stalled for ~120s before any network call
    started. Root cause: `qdrant_client` was imported lazily, for the
    first time in that process, on a `_search_executor` worker thread,
    while the main thread's asyncio Proactor loop was already doing real
    overlapped I/O on the stdio pipes -- that specific combination (live
    Proactor I/O + a first-time native-extension import on another
    thread) stalled reproducibly on at least one Windows deployment
    machine, independent of any configured timeout (isolated,
    reproduced, and fixed live; two other plausible causes --
    antivirus file-hash scanning, and `CREATE_NO_WINDOW` plus a stripped
    subprocess environment -- were tested directly and ruled out). Fixed
    by importing `qdrant_client` and `opensearchpy` eagerly, on the main
    thread, at module load time in `scripts/mcp_corpus_server.py`,
    before `mcp.run()` starts the event loop -- see that file's module
    docstring ("First-import gotcha") for the full account. **Do not
    remove those two imports as "dead code"**; they look redundant with
    `QdrantStore`/`OpenSearchStore`'s own lazy imports but are
    load-order-critical. This also loosened
    `tests/unit/test_mcp_corpus_server.py::test_mcp_corpus_server_imports_instantly_without_a_live_backend`'s
    import-speed guard from 2s to 30s (an intentional trade-off,
    documented on that test).

**Still broken, in rough order of consequence**

1. Self-critique can pass a report whose claims appear in no retrieved
   evidence. Several things have chipped at this and none of them fully
   closes it: P2-11 made the judge a different provider than the writer;
   the compiler prompt states per-goal evidence coverage explicitly
   (count, best score, and a `WEAK` flag when the best score sits at or
   below the single-leg RRF ceiling of 0.5); and telemetry reports
   `grounding_ratio`.
   **Superseded (D-38/D-40):** the line above used to read "forbids
   filling gaps from model knowledge" — that GROUNDING RULE was the
   direct cause of a corpus miss being reported as an absence of
   knowledge rather than a retrieval limitation, and D-38 replaced it
   with an ATTRIBUTION RULE: model-tier claims are now permitted, but
   must be marked as general knowledge, never presented as a retrieved
   finding. D-43/D-46 also added a check the critic is now explicitly
   asked to perform — failing any named entity, figure or date absent
   from every evidence item, with the evidence block finally shown to it
   (D-46; before that the check was unanswerable by construction) — plus
   a deterministic pass (D-45) that strips citations glued directly onto
   evidence text with no delimiter, and drops `[gN]` markers naming a
   goal with zero evidence.
   **What remains open, precisely:** none of this is a programmatic,
   claim-by-claim relevance check. D-43/D-46's critic check is still an
   LLM judging another LLM's output against evidence — it helps and it
   is not a guarantee (D-41 says the same of the model tier's own
   anti-fabrication limits: "reduces but does not eliminate"). And
   `grounding_ratio` still measures evidence **presence, not
   relevance** by design — a run can report `grounding_ratio: 1.0` while
   every attached item is topically irrelevant. D-39's topical gate
   closes this specific failure mode for `corpus_recall` (a topically
   irrelevant document can no longer stop the retrieval ladder), but
   does not touch `grounding_ratio` itself, which stays a coarser,
   deliberately threshold-independent signal — see the Telemetry table.
   *A report-parsing check was considered and rejected: across four live
   runs the model cited goal ids four different ways and in one run used no
   bracket citations at all, so a regex would have been silent on the least
   grounded report of the set. That is why the measurement reads
   `state.evidence` instead.*
2. ~~RRF joins the two legs on `title`, not on any store id~~ — **fixed,
   D-61(b).** `retrieval/hybrid.py::_join_key` now fuses on
   `content_id(content)`, the same UUID5-of-content function already used
   for Qdrant point ids and memory dedup. Closed three failures at once:
   two different documents sharing a title (or a 60-char boilerplate
   prefix) used to fuse into one and silently lose a document; and the
   same document titled in only one store never fused at all, scoring a
   genuine two-leg agreement as two single-leg hits — which lands exactly
   on the `min_evidence_score` boundary collision P2-01's follow-up fixed.
3. ~~`Evidence.task_key` for memory items uses `hash()`~~ — **fixed.**
   `memory/semantic_memory.py::retrieve` builds `memory-<content_id>`;
   `hash()` appears nowhere in `src/` any more. This entry was already
   stale when D-60/D-61 landed and is corrected rather than deleted.
4. Reusing the same `--thread-id` across unrelated runs silently accumulates
   reducer-backed state (`evidence`, `counters`, etc.) — see the Postgres
   section above for the full explanation and a live example. Not addressed
   by any Tier; no P2-xx item currently scoped to it.
5. Contradiction detection remains marker-only — `E2` has never fired in a
   real run (`P2-12`, Tier 3, depends on P2-01 which is done). Wiring the
   detector to something more than explicit markers is the only remaining
   step to make `E2` reachable in practice, not just in principle.

6. **Nothing retries a failed provider call.** `llm/client.py` sets a
   per-request timeout but has no retry, and `FallbackRouter` does not add
   one either — the client's docstring says "the router owns retry/fallback
   policy", and the router's policy is to hop, not to re-attempt. A single
   transient 429 or connection reset therefore consumes a whole provider
   position in the chain.
   **This is a stated choice, not an oversight.** Falling through to a
   different provider is usually a better answer than retrying one that
   just failed, and run p205.212 shows the design working: Gemini returned
   an `HTTPStatusError`, the chain absorbed it, and D-60(c) now makes that
   path return the best retained answer rather than depending on luck.
   Adding retry means choosing backoff and idempotency semantics, and no
   live run has yet shown a failure it would have caught — D-54's reasoning
   for not building against an unobserved failure mode applies directly.
   Revisit if provider errors ever become a *majority* of
   `llm_fallback_hops` rather than the occasional one.
7. **The HTTP API is unauthenticated by design.** `api/server.py` exposes
   `POST /research`, `POST /resume` and `GET /health` with no auth, no CORS
   policy and no rate limiting. Resolved deliberately in
   `internal/PHASE-2_PLAN.md`: *"Production concerns — auth, multi-tenancy,
   scaling | Out of scope for a reference implementation."* That remains the
   right call and this entry does not reopen it.
   It is recorded here only because of a tension worth naming: the module
   docstring calls the endpoint a demonstration, but D-37 packages this repo
   as an installable artifact whose declared public surface includes those
   three HTTP shapes — so somebody will eventually deploy it. **If you run
   this anywhere reachable, put it behind a gateway that terminates auth and
   rate-limits.** The graph has no notion of a caller identity, so there is
   nothing inside the application to fall back on.

## Documentation Corrections and Roadmap History

Moved to [CHANGELOG.md](CHANGELOG.md) (N-1) -- this README stays a
current-state entry point rather than an archive of superseded claims.
