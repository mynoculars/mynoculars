# CHANGELOG

Release-note history for this project, moved out of README.md (N-1) so the
README can stay a current-state entry point instead of an archive. Entries
below are kept exactly as originally written, in the order they were
written — each one was accurate AT THE TIME. For the current state of any
design decision, `DECISIONS.md` (not this file) is the authoritative,
continuously-updated log; this file is a historical record, not a second
source of truth to keep in sync with the code.

---

## Release banners (formerly README.md's opening section)

> **The four banners below are HISTORY, kept for auditability.** For the
> current state of any decision, `DECISIONS.md` (D-1…D-64) is the
> authoritative log; to just run something, jump to [Setup](README.md#setup).

> **Guardrails, Phases 1–3 — new since D-46.** A dedicated
> `research_agent/guardrails/` package now exists (`citations.py`,
> `fencing.py`, `hedging.py`) — deterministic post-processing checks
> applied at fixed points in the graph, documented in full under
> [Guardrails](README.md#guardrails) below. Phase 1 closed the false-convergence
> gap `grounded_score`/`grounded_recall_target` were built for (a topical
> gate on top of the existing corpus-recall check), added run-level
> WARNING telemetry for a starved retrieval floor, flagged model-tier
> claims pairing a specific year with a specific quantity
> (`hedge_specific`), and rejected `gap_generator` tasks naming a goal
> that doesn't exist in the current run. Phase 2 added deterministic
> enforcement for the Phase 1 flag (`(unverified figure)` markers
> inserted into the compiled report, not just noted in evidence), a
> length cap on `ResearchRequest.query` at the API boundary, and a
> WARNING when the quality judge fails on every attempt in a run
> (`llm_quality_calls_failed == llm_quality_calls`, observed on every
> live run to date). Phase 3 added `run_call_budget_warn` — observability
> only, no circuit breaker; see below. **Test suite: 348/348**, up from
> 294 — every count below is corrected to match. Nothing else changed:
> the architecture, D-xx decisions, and Tier 1/2/3 narrative are all
> still accurate.

> **CORRECTED THIS PASS** (post-Tier-3 session, 4 further live-tested
> patches applied on top of everything below): `api/server.py`'s
> `AppBundle` unpack crashed the whole API at import — fixed. MCP
> evidence's hardcoded `score=1.0` bypassed the coverage gate — fixed.
> Mid-run retrieval-leg failures are now isolated per-leg (previously
> only *boot-time* unavailability degraded gracefully; a store dying
> mid-run used to kill the whole task, discarding the healthy leg's hits
> too). Postgres checkpointer is now pooled (`psycopg_pool`, with a
> fallback to a single connection). A `qdrant_client` embedder race under
> parallel fan-out is now locked. Retrieved evidence is now fenced against
> prompt injection (`<evidence>...</evidence>` + an explicit system-prompt
> clause) in every prompt that inlines it. `escalation_history` now
> surfaces in telemetry (`telemetry["escalations"]`) — previously written
> by `human_escalation`, read by nothing. **The CLI's exit code is no
> longer always 0** (previously Limitations item 27, now fixed). The compiler's
> free-text output now survives a fallback provider wrapping it in a code
> fence or echoing the `<evidence>` tag back literally (`strip_code_fence`,
> tested against 15 edge cases including punctuated language tags like
> `c++`). **Test suite: 157/157**, up from 135 — every count below is
> corrected to match. Nothing else changed: the architecture, D-xx
> decisions, and Tier 1/2/3 narrative below are all still accurate.

> **Phase 3 — Langfuse observability, now documented.** A dedicated
> `research_agent/langfuse/` module adds optional, non-invasive tracing —
> traces, spans, generations, events, and scores — over every node in the
> graph, every LLM provider call, retrieval, memory, and HITL action.
> Disabled by default (`LANGFUSE_ENABLED=false`, zero SDK import, zero
> network calls); enabling it changes no business logic, prompts, or graph
> topology. See [Observability — Langfuse (Phase 3)](README.md#observability--langfuse-phase-3)
> below. **Test suite: 294/294** (grown across D-38–D-46's regression coverage, fully
> offline). D-35 logs the module-boundary decision in `DECISIONS.md`.

> **Phase 4 — web search (D-57/D-58).** The retrieval ladder gains a real
> search engine as tier 4, between the MCP tier and the model’s own
> knowledge. It runs in its OWN MCP server subprocess
> (`scripts/mcp_web_search_server.py`, `research_agent/websearch/`), so the
> agent process never imports a search client — swapping DDGS for a keyed
> API is one module plus one setting. Web evidence is tagged `source="web"`
> and deliberately does NOT count toward `grounded_score` or
> `corpus_recall`: a snippet is retrieval, not curation. Off by default
> (`WEB_SEARCH_ENABLED=false`), and with it off the ladder is byte-identical
> to every prior run. D-58 additionally fixes a latent path bug affecting
> BOTH MCP servers: relative paths resolved against the launch directory,
> not the repo. **Test suite: 476/476** (348 before Phase 4).

> **Post-Phase-4 correctness pass (D-60/D-61).** Diagnosed from a diff of
> two live runs — p205.211 and p205.212, same code, same query, one shipping
> a 652-character report with zero citations and the other a correct 11,121-
> character one. The only difference was that Gemini errored in one run and
> succeeded in the other. Three defects in one chain: the quality judge was
> handed `answer[:4000]` and so scored a mid-word truncation, i.e. the gate
> measured LENGTH not quality; `finish_reason` was never read, so a
> provider-reported truncated generation shipped as a finished report; and
> `FallbackRouter.complete` returned the LAST answer rather than the BEST,
> with the final provider exempt from scoring entirely. D-61 adds
> `guardrails/dedup.py` (one corpus sentence appeared **26 times** in a
> single 10,626-token compiler prompt — the corpus/MCP counterpart to the
> per-domain cap D-57 already applies to web results) and moves RRF's join
> key from `title` to `content_id`, closing Limitations 2 and 3.
> **Test suite: 518/518**, up from 492. `DECISIONS.md` D-60/D-61 carry the
> full account with the live counts.

> **Post-Phase-3 work (D-38–D-46, no separate phase number assigned):** retrieval was rebuilt as a multi-tier ladder ending in the model’s own knowledge (4 tiers then; 5 since Phase 4 added web search), with anti-fabrication limits, deterministic citation repair, and a critic that now sees the evidence it verifies against. Not a phase/tier bump in name — `DECISIONS.md` is the source of truth for this range; see D-38 through D-46.

> **Status:** Core build. Implements the workflow graph, hybrid retrieval,
> semantic memory, LLM fallback routing, the self-critique loop, and
> human-in-the-loop escalation from the accompanying design document
> (decisions D-1…D-24, D-28, D-29 + proposed D-31/D-32). Tier 1 AND Tier 2
> of `internal/PHASE-2_PLAN.md` are both closed as of this revision. MCP tool
> mediation is deliberately deferred — see [Limitations](README.md#limitations).
>
> **This revision is code-truth, updated after a round of live fixes.**
> Every claim below was verified against the source and against real
> debug traces captured across several live runs while diagnosing and
> closing Tier 2. Where the design document or an earlier README said
> something the code does not do, the delta is recorded in
> [Documentation Corrections](#documentation-corrections) rather than
> quietly repeated.
>
> **The headline change since the last revision: Tier 2 is closed, and
> every item in it was verified against real, live traces — not just the
> offline test suite.** `internal/PHASE-2_PLAN.md`'s four Tier 2 items (P2-06
> producer validation, P2-07 boundary-scoped telemetry, P2-08 Postgres
> lifecycle + API parity, P2-09 config strictness + `DECISIONS.md`) are
> all done. P2-03's idempotent-ingest mechanism is now actually wired into
> `scripts/ingest_sample_data.py` (previously the mechanism existed but no
> caller used it — see [Recent Fixes](#recent-fixes)). One incidental fix
> landed alongside this work, outside its original scope: an
> `opensearch-py` 3.x compatibility break (`indices.exists`/`.create`/
> `.index`/`indices.refresh` now require `index=` as a keyword argument,
> not positional) surfaced during live ingest testing and is fixed in
> `storage/opensearch_store.py`. Full suite: **157/157 tests passing**.
>
> The escalation path remains confirmed live, end-to-end, from the prior
> revision — a real run with `HITL_ENABLED=true` reached
> `human_escalation` with trigger `E3`, paused, and resumed correctly on
> `approve`. **One honest caveat on HOW it fired, unchanged this
> revision:** the run that proved this hit an outright retrieval failure
> (`NotFoundError` — the Qdrant corpus collection was empty), so `recall`
> hit `0.0` via the pre-existing D-16 failed-task path, not via the
> specific P2-01 boundary-collision fix. That fix is separately
> unit-proven; see [Recent Fixes](#recent-fixes).


## Fix and guardrail tables (formerly README.md's Recent Fixes and Guardrails phases)

> **Moved here verbatim, nothing rewritten.** These tables record WHAT
> CHANGED IN WHICH REVISION and what verified it -- changelog work that had
> accumulated inside an architecture document. README.md keeps the
> current-state material (the guardrails config surface, the telemetry
> contract, what is still open) and points here for the history.
> `DECISIONS.md` remains the authoritative log for any single decision.

### Recent Fixes

*This heading was deleted while the table below stayed, leaving it
orphaned under an unrelated section and breaking every reference to it:
three `[Recent Fixes](#recent-fixes)` links (two in `CHANGELOG.md`, one
here) pointed at a slug that existed in neither file, and five more
mentions in prose named a section a reader could not find. Restored, which
also gives `### Tier 2` below the sibling it was written to have.*

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

*(An earlier revision stated a literal test count here. See the Status
note above for why this file no longer carries one.)*

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
OPERATIONS.md's [Fine-Tuning the System](OPERATIONS.md#step-3--calibrate-the-retrieval-floor-required-before-trusting-any-result)
section is the step-by-step procedure, with the exact commands and the
expected output at each step.

**Also new since the last revision, unrelated to the fixes above:**

- **Per-node debug logging.** `--debug` (or `DEBUG_TRACE=true`) now emits a
  `"node.enter"` log line the instant *every* node starts running —
  including `merger` and `progress_checker`, which make no LLM or store
  call and so never appeared anywhere in a trace file before this. See
  [Telemetry — read it honestly](README.md#telemetry--read-it-honestly).
- **`--print-graph`.** Prints the compiled graph's static topology (ASCII
  if `grandalf` is installed, Mermaid text otherwise) via LangGraph's own
  introspection — independent of any run. Usable alone (no query, exits
  after printing) or combined with a query (prints, then runs normally).

### Guardrails — Phases 1-8

### Phase 1 — grounded convergence, retrieval-floor telemetry, false-precision flag, orphaned-task guard

| Item | What changed | Verified how |
|---|---|---|
| **Grounded convergence** | `state.py::ResearchState.grounded_score` (new field) is computed every gather cycle in `agents/gathering.py::progress_checker_node`, alongside the existing `recall_score`: a goal only counts toward it if at least one covering evidence item is `source in ("corpus", "mcp")` **and** shares distinctive vocabulary with the goal's own description — the same topical-overlap check `telemetry_node` already applied for `corpus_recall`, reused here (`retrieval/terms.py::distinctive_terms` — S-7 moved it out of `tools/retrieval_chain.py`, where it was underscore-prefixed while four other modules imported it) so the two numbers can never disagree. `orchestration/graph.py::route_convergence` will not treat `recall_score` reaching target as full convergence unless `grounded_score` also clears `settings.grounded_recall_target` (new, default `0.5`); otherwise it spends remaining depth budget on another gather cycle instead of compiling on ungrounded evidence | Unit tests on `route_convergence` and `progress_checker_node` directly, including the exact live shape that motivated it (a corpus hit topically unrelated to the goal it was credited against, scoring above the floor); confirmed live across six consecutive real runs — `grounded_score` correctly stayed `0.0` while `recall_score` reached `1.0`, and the router correctly kept looping instead of compiling early |
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

### Phase 8 — attribution repair, the memory floor, and a composed verdict (D-140…D-146)

| Item | What changed | Verified how |
|---|---|---|
| **Deterministic citation attachment (D-144)** | New `guardrails/attribution.py::attach_missing_citations`, run between `normalise_citation_form` and `clean_citations`. When the report's PROSE cites nothing at all, it attaches `[gN]` markers by distinctive-term overlap between each sentence and each goal's own evidence. Four runs had shipped zero-citation reports (`p205.276`, `p205.277`, `p205.280`) against 35–100 evidence items, and every prior fix was a prompt instruction (D-40, D-73) or a form-normaliser (D-99) — none of which help when the model emits no goal id in any form. Deliberately narrow: all-or-nothing, unambiguous only (a tie attributes nothing), a two-term floor, and **add-only**, so every marker it writes then passes through `clean_citations` like the model's own. Counted as `citations_attached` so a rescue is never invisible | 18 tests, including a replay of `p205.280-check`'s shipped report (0 markers → 5), idempotence across revision passes, and that a partially-cited report comes back **byte-identical** |
| **Sources decoupled from citations (D-144)** | `append_web_sources` gains `list_when_uncited`. The old `listed` was a strict subset of what the prose cited, so a report with no markers got no Sources section either — one formatting failure took out attribution twice. Live: `web_sources_listed: 0` against 58 web items across **33 distinct domains**, under a provenance notice telling the reader to trust figures *"unless a listed source confirms them"*. The pages are now listed under a note saying they were retrieved rather than cited. D-59's rule is kept BY the note, and its topical gate still applies in full | 7 tests, including that D-59's nine-Redis-URL failure is still dropped on the fallback path, that the fallback is off by default, and that `count_listed_sources` still parses the block |
| **`cited_goal_ids_in_prose` (D-144)** | `cited_goal_ids` matches `[gN]` anywhere, and every Sources entry begins `1. [g1] ` by construction — so a report whose prose cited nothing but which carried a Sources block read back as fully cited. That would have made `evidence_cited` wrong, the D-66 gate silent, and telemetry's backstop agree with both. **The defect predates Phase 8** and was unreachable only because the block was itself gated on prose citations — the exact coupling above removes | 3 tests pinning the old whole-report read against the new prose-scoped one |
| **Memory relevance floor (D-142)** | New `MEMORY_MIN_SIMILARITY` (default `0.60`). `SemanticMemory.retrieve` had **no floor at all** — `memory_write_min_score` gates what goes IN, nothing gated what came OUT. Live: five Redis-vs-Memcached items recalled at similarity 0.45–0.47 into a China-vs-India query, **leading** the compile prompt, while the corpus floor at 0.55 dropped 72 of 72 dense candidates. Tested against RAW similarity before decay: relevance and freshness are different questions. Memory pseudo-goals also collapse to one prompt-budget bucket, and the evidence block is now ordered by provenance then score | 7 memory tests + 4 budget tests + 4 prompt-ordering tests, including that a stale-but-relevant item is de-ranked rather than deleted, and that `EVIDENCE_ORDER == _SOURCE_RANK` |
| **Composed confidence verdict (D-145)** | New `reporting/confidence.py`. Caps, not a weighted mean — a mean scores `p205.280-check` near 0.5 because `recall` and `grounding_ratio` were both 1.0. Surfaced in the RESULT block, the run narrative, telemetry (and so the `agent_runs` row) and `analyze_runs.py` | 27 tests, including that a naive average really is that generous, each cap individually, and a parametrised calibration against `golden_queries.jsonl` |
| **Config self-contradiction check (D-143)** | `warn_on_primary_context_below_prompt_budget`. `PROMPT_EVIDENCE_MAX_CHARS=12000` (~3,000 tokens) and `LLM_PRIMARY_CONTEXT_TOKENS=1536` are arithmetically incompatible, so D-93 skipped the primary on **every** compile and critique and the chain became cloud-only for the two nodes that write the report | 5 tests, including that both named remedies actually silence it |
| **The report pipeline is a list (D-146)** | `reporting/pipeline.py::REPORT_PASSES`. Twelve straight-line steps with a paragraph between each became a named ordered tuple whose ordering constraints are **data** (`ReportPass.after`) and are asserted by a test. `reporting/telemetry.py` extracts the counter-only half of the 531-line `telemetry_node`. `compiler_node` 231 → 171 lines; `agents/compilation.py` 1,149 → 1,049 | 23 tests, and — the claim that matters — **not one existing test changed**, plus a direct key-set comparison proving the telemetry contract lost nothing |
| **The test suite stopped opening sockets (D-140/D-141)** | `probe=False` on both stores, session-scoped `settings`/`off_memory`, and `-n auto --dist loadfile` in a `pytest.ini` that now has more than one setting in it. See OPERATIONS.md's "Running and Interpreting the Test Suite" | Measured, not assumed: **24.39 s → 17.14 s** serial and **57 → 1** warnings on Linux; 1019/1019 verified at `-n0`, `-n4` and `-n8` before xdist was adopted |
| **Postgres was the one store that bounded nothing (D-149)** | `storage/postgres.py` set `connect_timeout` at no call site, so `record_run` against an unreachable DSN waited out the OS TCP timeout. Five `POST /research` calls in one test file cost **650.61s of a 662.25s suite** — 98% of it, against ~12s for everything else. Not only a test problem: `record_run` runs at the END of every CLI run, so an unreachable Postgres hangs the process with the answer already on screen. `CONNECT_TIMEOUT_SECONDS = 5` now applies to `record_run`, the single connection and the pool, matching `QdrantStore`, `OpenSearchStore` and `check_services.py` | Measured from a real `--durations=25`. D-140 fixed the two store CONSTRUCTORS and missed this one; five tests in that file already stubbed `record_run` by hand and four did not, and **those four were exactly the four slowest tests in the suite** — now stubbed in the shared harness so it cannot recur |
| **The relevance floor had a second door (D-150)** | `MIN_SIMILARITY` gated only the dense leg; OpenSearch hits went straight into fusion. Live: BM25 matched a Redis document at `0.92` for *"organizational structures command hierarchies Chinese People's"* — on `command` and `structures` — putting **42 corpus + 36 mcp** items into a China-vs-India run. The floor's verdict now binds both legs: if every dense candidate fell below it, the corpus does not cover that query and the keyword hits go too | The obvious fix was **measured and rejected**: a `>=2` term-overlap gate drops the Redis hits correctly but also drops `in-corpus-operational` from this repo's own golden set, a query two documents genuinely answer in different words. 5 tests, including a replay of the live shape and one pinning single-leg degradation |
| **The context setting no longer has to be right (D-151)** | `LLM_PRIMARY_CONTEXT_TOKENS` describes the SERVER and had drifted from it — `.env` said 8876, llama-server reported `n_ctx` 1536 — so D-93 stopped skipping and spent two guaranteed-failed calls where it used to spend two free skips. The real window is now read out of the provider's own 400 and used for the rest of the process. **D-143's note in `.env.example` caused this** by saying "the model's real window"; it is corrected to the `-c` the server was started with | 11 tests keyed off the exact body the server returned. A third defect surfaced while fixing it: `looks_like_context_overflow` never matched llama.cpp's phrasing at all — `"exceeds context"` does not occur in `"exceeds the available context size"` — so a textbook context rejection read as a generic bad request |
| **One ratio, two meanings (D-152)** | `retrieval_floor_drop_ratio >= 0.8` capped confidence with *"retrieval was starved"*. For a query the corpus covers that names a real defect; for one it does not, the identical ratio is the floor working and the message sends someone to fix a correct threshold. `tier_answers` separates them | The **cap is unchanged** either way — an answer with no corpus behind it is LOW whichever the cause. Only the remedy differs, so only the wording does |
| **Lint config and a test workflow (D-154)** | `[tool.ruff]` in `pyproject.toml`, selecting the rules that find bugs rather than express taste (`E4, E7, E9, F, W, B, C4`) — the repository passes them with **zero violations**. `.github/workflows/tests.yml` runs the suite and nothing else, so the badge means one legible thing. `scripts/sanity.py` (D-158) is the same three checks run LOCALLY, before a demo, since CI tells you about a push and not about the machine you are standing at | Reaching zero found 35 real defects, several from this phase: 11 unused imports, 3 dead locals, a `zip()` that would silently truncate an upsert, an `assert False` that `python -O` deletes, and 4 tests asserting a blind `Exception` that would pass for a typo in the test. What is *not* selected is recorded with its count and its reason |
| **A context window per provider (D-153)** | D-93 shipped `LLM_PRIMARY_CONTEXT_TOKENS` alone, reasoning that cloud fallbacks never refuse on size. True of the providers it was written for, not of the **slot** — D-114 lets the fallback point at any OpenAI-compatible endpoint, including a second local `llama-server`. New `LLM_MISTRAL_CONTEXT_TOKENS` and `LLM_FALLBACK_CONTEXT_TOKENS`, both defaulting to `0` so an existing `.env` routes byte-identically. Named for the slot, not the vendor | The architecture was already per-provider — `context_tokens` has always been per client, `_skips_for_context` has always read it with `getattr`, and D-151 already learns it per client. 14 tests, including one that drives a real three-provider chain of too-small windows and asserts the last is still attempted |
| **The critic's verdict had no counterweight (D-155)** | `passed = bool(result.get("passed", False))` was the one LLM judgement here taken unconditionally. Live, the critic failed a correct report on four notes objecting to **faithful rounding** — *"approximately 2 million"* against evidence reading *"approximately 2 million to 2.1 million"* — while D-91's deterministic audit reported `cited_figures_unsupported: 0` on the same report. The prompt now states that restating an evidence figure in different words is not unfaithful (`critic` → `v2`), and `resolve_verdict` lets a failing verdict stand **unless every note disputes a figure the evidence the critic was shown actually contains** | It resolves a disagreement, it does not overrule: a note naming no figure is a coverage finding an LLM critic is *for*, it survives, and **one survivor stops the flip**. Never fires when D-91 flagged anything. `critic.failure_not_corroborated` at WARNING plus `critique_notes_dismissed` in `run_metrics`, so a resolved verdict is never mistaken for a clean pass. 20 tests, including a replay of the live notes |

### Fixed since the last revision

*Kept struck-through rather than deleted, so the history stays auditable.*

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
    Tier 3, fixed.** *(Implementation now lives in
    `src/research_agent/servers/corpus.py`; D-157 reduced
    `scripts/mcp_corpus_server.py` to a 28-line launcher, so read the
    package module for every code detail named below — the command you
    RUN is still `python scripts/mcp_corpus_server.py`.)* The tool
    handler used to be synchronous; FastMCP called it directly on its single event
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
    just slowly. Fixed by making `servers/corpus.py::search` `async
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
    thread, at module load time in `src/research_agent/servers/corpus.py`
    (lines 137-138), before `mcp.run()` starts the event loop -- see that
    file's module docstring ("First-import gotcha") for the full account. **Do not
    remove those two imports as "dead code"**; they look redundant with
    `QdrantStore`/`OpenSearchStore`'s own lazy imports but are
    load-order-critical. This also loosened
    `tests/unit/test_mcp_corpus_server.py::test_mcp_corpus_server_imports_instantly_without_a_live_backend`'s
    import-speed guard from 2s to 30s (an intentional trade-off,
    documented on that test).

## Documentation deltas and roadmap history (formerly README.md's closing sections)

## Documentation Corrections

Applied above; listed here so the deltas against the older documents are
auditable rather than invisible.

| Claim in older docs | Reality in code |
|---|---|
| README: fallback is "local Qwen Cogito → Gemini Flash" | Three hops: primary → Mistral → Gemini, each fallback gated on its API key, **and each hop tier uses a different timeout** |
| README / OPERATIONS: "28 tests" | **344** tests collected and passing (grew across Tier 2/3 to 135, then 157 post-Tier-3, 190 with Phase 3's 33 `test_langfuse.py` tests, 294 across D-38–D-46's retrieval-ladder and citation-repair regression coverage, 341 across the Guardrails Phase 1–3 regression coverage, then to **344** with D-55's topical-gate floor correction) |
| design §12: "63 files, 28/32 tests passing, 4 skipped" | ~100 files in this distribution; **344** tests. Skip count is environment-dependent (langfuse extras) — 0 skipped when installed, 9 when not (5 langfuse-extras, 4 MCP live-server) |
| README legend: with HITL off the checks "log and continue" | True for E1/E4 only; E2/E3 log nothing when HITL is off |
| OPERATIONS §"Writing Your Own Test Corpus": "re-run ingest (it upserts by id, so re-running overwrites)" | True for both stores — OpenSearch always was idempotent (`str(i)`); Qdrant's `id_fn` mechanism is wired into `scripts/ingest_sample_data.py` via a deterministic `uuid5(content)` id. Does not retroactively clean up a collection that already accumulated duplicates before this fix — see Ingest identity above |
| OPERATIONS §"Test HITL": that query escalates | Previously converged at `recall 1.0` at depth 1 and never interrupted. Root causes fixed (P2-01, P2-02) and re-verified end-to-end against real live runs — both a genuine E3 escalation (via the D-16 failed-task path) and a clean convergence at `recall 1.0` with real evidence once the corpus was properly ingested. See The HITL Investigation |
| design §9: `MAX_REVISIONS` default 3 | Code default is **2** (`config.py`) |
| README structure tree: root `agentic-research-agent/` | Distributed directory is `research-agent-dmp/` |
| Storage diagram implied one Qdrant use | Two collections; `CORPUS_INDEX` names **both** a Qdrant collection and an OpenSearch index |
| `DECISIONS.md` referenced as the decision log | Populated, currently D-1 through D-64 (sourced from code comments and this document's own citations — D-7/9/10/11 are flagged as ungrounded rather than invented; see `DECISIONS.md`'s own header for the up-to-date range and gap note rather than repeating the count here) |
| `internal/LEARNING_GUIDE.md` cited as a companion doc | `internal/` is in `.gitignore`, so it ships only in archives like this one |
| OPERATIONS §L1: "add two `logging.getLogger(...)` lines" | Already present in `logging_setup.py::configure_logging` |
| This README's own citations of "`PHASE2_PLAN.md`" | The actual tracked file is `internal/PHASE-2_PLAN.md` (hyphenated, under `internal/`) |
| **Limitations, "Exit code is always 0"** *(this pass)* | **Fixed.** `main()` now returns 2 on `GraphRecursionError`, 1 when telemetry never populated. See Limitations item 27 above. |
| Any doc citing `cli.py::build_app_and_settings` as the wiring point *(this pass)* | It moved to `assembly.py`. `api/server.py` imports it from there; `cli.py` re-exports both it and `AppBundle`, so older call sites still work. |
| "the repo is run via `PYTHONPATH=src`" as the only option *(this pass)* | `pyproject.toml` now exists — `pip install -e .` (plus extras) and a `research-agent` console script. See [Packaging](README.md#packaging). |

## P Improvements

`internal/PHASE-2_PLAN.md` has the full 15-item plan — every item scoped to an existing
seam, with complexity, dependencies, and the D-xx tag it extends or replaces.
The shape of it, updated with this revision's progress:

```text
  TIER 1 ── correctness ──────► make the documented behaviour true    ✅ CLOSED
     P2-01  relevance floor + calibrated evidence gate       ✅ DONE
              + follow-up: exact-boundary collision (`>` fix)  ✅ DONE
     P2-02  namespace memory goal_id on retrieval             ✅ DONE
     P2-03  deterministic Qdrant point ids                    ✅ DONE — mechanism
                                                                 AND wired into
                                                                 ingest script
                                                                 (see Recent Fixes)
     P2-04  provider output sanitizer  (<|im_end|>)           ✅ DONE
              + free-text runaway-generation truncation        ✅ DONE
     P2-05  escalation reachability tests                     ✅ DONE — tests
                                                                 passing AND
                                                                 live-fired
                     │
  TIER 1 STATUS: CLOSED.
                     │
  TIER 2 ── robustness ───────► make it observable and hard to crash  ✅ CLOSED
     P2-06  validate LLM producer output                      ✅ DONE
     P2-07  boundary-scoped telemetry                         ✅ DONE — router
                                                                 AND retrieval
                                                                 halves both
     P2-08  postgres lifecycle + API run-history parity       ✅ DONE
     P2-09  config strictness + populate DECISIONS.md         ✅ DONE
                     │
  TIER 2 STATUS: CLOSED.
                     │
  TIER 3 ── design catch-up ──► close the D-25/26/27/30 gap        ✅ CLOSED
     P2-10  Qdrant payload indexes + server-side decay      (D-27) ✅ DONE
     P2-11  judge-model quality scoring                             ✅ DONE
     P2-12  semantic contradiction detector — E2 wiring reachable,  ⚠ PARTIAL
              detector itself still marker-only (see Limitations)
     P2-13  MCP tool seam                                   (D-26/D-30) ✅ DONE
     P2-14  typed specialist workers                        (D-25) ✅ DONE
     P2-15  memory supersession + GC                                ✅ DONE
                     │
  TIER 3 STATUS: CLOSED (P2-12 wiring done; detector quality tracked separately).
                     │
  POST-TIER-3 SESSION ── a further round of live-tested fixes    ✅ CLOSED
     api/server.py AppBundle unpack crash                    ✅ FIXED
     MCP evidence hardcoded score=1.0                         ✅ FIXED
     Mid-run retrieval-leg failure isolation                  ✅ FIXED
     Postgres checkpointer pooling                            ✅ FIXED
     qdrant_client embedder race under fan-out                ✅ FIXED
     Prompt-injection fencing on retrieved evidence            ✅ FIXED
     escalation_history surfaced in telemetry                  ✅ FIXED
     CLI exit code no longer always 0                          ✅ FIXED
     Compiler free-text fence/tag-echo leakage                 ✅ FIXED
     Test suite: 135 → 157
                     │
  PHASE 3 ── observability ───► optional Langfuse tracing, off by default ✅ CLOSED
     research_agent/langfuse/ package (D-35)                     ✅ DONE
     Every node, LLM call, retrieval, memory op traced            ✅ DONE
     Cost tracking from Settings-configured $/1M rates             ✅ DONE
     propagate_attributes session/environment grouping             ✅ DONE
     Crash-safe end_trace (try/except in cli.py::_run)              ✅ DONE
     Test suite: 157 → 190 → 294 → 341 → 344 → 348 → 476 (Phase 4)
```

**Status, updated**: Tiers 1, 2, and 3 are all closed, and a further,
post-Tier-3 session of live-tested fixes has landed on top (see the roadmap
above). `P2-12` (semantic contradiction detector) made `E2` reachable in
principle — the interrupt wiring is correct and tested — though the detector
itself remains marker-only in practice, so no real run has triggered `E2`
yet; see Limitations for the current honest status of that gap. `P2-11`
(judge-model quality scoring) directly addressed a report whose
Cassandra/DynamoDB sections cited no retrieved evidence, passed anyway by
same-model self-critique — the judge is now always the next provider in the
fallback chain, never the one being judged; a programmatic, claim-by-claim
grounding check remains open and is the top item on this document's honest
list of what's still missing.
