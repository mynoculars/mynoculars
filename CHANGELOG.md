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
> authoritative log; to just run something, jump to [Setup](#setup).

> **Guardrails, Phases 1–3 — new since D-46.** A dedicated
> `research_agent/guardrails/` package now exists (`citations.py`,
> `fencing.py`, `hedging.py`) — deterministic post-processing checks
> applied at fixed points in the graph, documented in full under
> [Guardrails](#guardrails) below. Phase 1 closed the false-convergence
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
> topology. See [Observability — Langfuse (Phase 3)](#observability--langfuse-phase-3)
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
> mediation is deliberately deferred — see [Limitations](#limitations).
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
| "the repo is run via `PYTHONPATH=src`" as the only option *(this pass)* | `pyproject.toml` now exists — `pip install -e .` (plus extras) and a `research-agent` console script. See [Packaging](#packaging). |

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
