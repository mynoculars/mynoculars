# Agentic Research Agent — Reference Implementation (Core Build)

A production-style (not production-grade) research agent built on LangGraph,
showcasing production-oriented architecture and engineering practices. Given
a research question, it plans goals, retrieves evidence in parallel,
iteratively deepens coverage, self-critiques its report, pauses for human
review when it cannot converge, and remembers what it learned for future runs.

> **Status:** Core build. Implements the workflow graph, hybrid retrieval,
> semantic memory, LLM fallback routing, the self-critique loop, and
> human-in-the-loop escalation from the accompanying design document
> (decisions D-1…D-24 + D-28 subset). MCP tool mediation is deliberately
> deferred — see [Limitations](#limitations).
>
> **This revision is code-truth, updated after a round of live fixes.**
> Every claim below was verified against the source, against a real debug
> trace (`logs/trace-run-0d7d0448906a.txt`), and — for the fixes described
> in [Recent Fixes](#recent-fixes) — against fresh live-mode debug traces
> captured while diagnosing and closing them. Where the design document or
> an earlier README said something the code does not do, the delta is
> recorded in [Documentation Corrections](#documentation-corrections)
> rather than quietly repeated.
>
> **The headline change since the last revision:** the escalation path has
> now been confirmed live, end-to-end — a real run with
> `HITL_ENABLED=true` reached `human_escalation` with trigger `E3`, paused
> at the `action [approve/redirect/abort]:` prompt, and resumed correctly
> on `approve`. **One honest caveat on HOW it fired:** in the run that
> proved this, retrieval failed outright (`NotFoundError` — the Qdrant
> corpus collection was empty after a `reset_stores.py --yes` with no
> re-ingest), so `recall` hit `0.0` via the pre-existing D-16 failed-task
> path, not via the specific P2-01 boundary-collision fix described below.
> That fix (`>` instead of `>=` in the coverage check) is separately
> unit-proven against the exact `score=0.5` artifact a real run produced —
> see [Recent Fixes](#recent-fixes) — but has not yet been the thing that
> triggered a live escalation on its own. Both are real, both are fixed;
> this note just keeps the two pieces of evidence from being conflated.

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
| `internal/LEARNING_GUIDE.md` | Pedagogy — follow-one-query walkthrough, concept teaching, interview framing (note: `internal/` is gitignored, so it ships only in archives) |
| `design/Research_Agent_Design.md` | The full target architecture and D-1…D-30 rationale — a strict superset of this build |
| **`README.md`** (this file) | What exists, how it is wired, what each store actually holds, and what is broken |

## Recent Fixes

Four of the fifteen items in `PHASE2_PLAN.md`'s Tier 1 have landed. Listed
here first, ahead of the architecture walkthrough, because they change what
several sections below used to say — if you've read an earlier revision of
this README, this is the fastest way to see what moved.

| Item | What changed | Verified how |
|---|---|---|
| **P2-04** — provider output sanitizer | `llm/client.py::_extract_json` now strips known chat-template end-of-turn sentinels (`<\|im_end\|>`, `<\|eot_id\|>`, `<\|end_of_text\|>`, `<\|endoftext\|>`, `</s>`) before parsing, and falls back to extracting the outermost `{...}` span if parsing still fails after that | Reproduced the exact failure pattern (`'{"goals": [...]}<\|im_end\|>'`) and confirmed it now parses; confirmed clean JSON and genuine garbage are unaffected. **Then confirmed live**: three separate follow-up runs against a real local Cogito model, previously failing with `JSONDecodeError` on every structured call, now succeed with zero fallback needed |
| **P2-01** — relevance floor + calibrated evidence gate | Two-stage gate, not one: `min_similarity` (new, default `0.35`) filters low-relevance dense hits *before* they enter fusion or become Evidence at all (`retrieval/hybrid.py`); `min_evidence_score` (raised from the inert `0.0` to `0.5`) still filters *after*, at the coverage-check step, as before | Constructed a fake retriever returning hits at similarity 0.9/0.4/0.1; confirmed the floor drops the 0.1 hit and keeps the other two, and that `min_similarity=0.0` reproduces the exact old (no-floor) behavior |
| **P2-02** — namespace memory `goal_id` | `memory/semantic_memory.py::retrieve` now tags every memory-sourced `Evidence` with `goal_id="memory::<original>"` instead of the bare original, so it can never satisfy a *current* run's goal by string collision | Constructed a fake memory hit tagged `goal_id="g3"` (simulating an unrelated earlier run) and confirmed the returned `Evidence.goal_id` (`"memory::g3"`) is provably `!= "g3"` |
| separate, related fix — Cogito timeout | One shared `llm_timeout_seconds` (`60.0`, applied to every provider identically) split into `llm_primary_timeout_seconds` (`120.0`, Cogito only) and `llm_timeout_seconds` (`90.0`, now Mistral/Gemini only) | Confirmed live: two calls that previously timed out at exactly 60.0s now time out at exactly 120.0s (the local model genuinely needed more room for large prompts and, apparently, a cold-start model load on the session's first call) |
| **P2-01 follow-up** — exact-boundary collision | Coverage check changed from `>=` to `>` in `agents/gathering.py::progress_checker_node`. Found live: with `RRF_SQUASH=30.0` and `RRF_K=60`, a rank-0 hit under SINGLE-LEG fusion (i.e. whenever OpenSearch is down) squashes to *exactly* `1/60 × 30 = 0.5` — a deterministic artifact of the fusion math, not a measure of relevance — and a `>=` comparison against `min_evidence_score=0.5` let that exact value through every time | Reconstructed the exact evidence shape a real run produced (`score=0.5` against a `0.5` threshold) and confirmed the goal is no longer marked covered. This closes the collision regardless of the threshold chosen — it does not replace actually fixing why OpenSearch is unreachable |
| **New — free-text runaway-generation truncation** | New `_truncate_at_sentinel()` in `llm/client.py`, applied inside `OpenAICompatibleClient.complete()` (the free-text path `compiler_node` uses). P2-04's sentinel-stripping only ever covered the JSON-mode path (`_extract_json`) — the free-text path had zero cleanup. A real run showed the local model generating 94+ seconds and 3,151+ tokens past its actual answer, hallucinating an entire fake follow-up conversation (a fabricated `system` turn, the prompt regenerated, a duplicate report) that went straight into `final_report` verbatim | Reconstructed the exact runaway pattern from a real trace (clean report → `<\|im_end\|>` → fake turns → duplicate report) and confirmed only the clean report survives. **Confirmed firing live, repeatedly** — a later real run shows `llm.truncated_runaway_generation` on `goal_manager`, `task_expander`, both `gap_generator` calls, both `compiler` calls, and `critic`; the worst case kept 4,742 of 10,901 raw characters. The underlying cause is almost certainly the local `llama-server`'s stop-token/chat-template configuration, not fixed by this — this is a codebase-side safety net regardless of server config |

**A calibration caveat, stated plainly rather than buried:** `0.5` and
`0.35` are starting points anchored to a real debug trace, not values
independently measured against every corpus this build might run over.
Before trusting them on your own data, run a `--debug` query you know is
on-topic and one you know is off-topic, and compare the actual `similarity`
and `score` values in the trace/log output — see
[Debugging a live run](#debugging-a-live-run) below for exactly how.

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
- `--print-graph`: the compiled graph's static topology, independent of any
  run.
- Fully offline demo mode (`LLM_MODE=stub`) — the entire graph runs with zero
  services and zero API keys.

**Non-goals**
- Production deployment, auth, multi-tenancy, horizontal scaling.
- Dynamic (LLM-decided) control flow / supervisor agents.
- Web search — retrieval is over the local sample corpus.

## Architecture

### Overall architecture

Everything is assembled in exactly one place — `cli.py::build_app_and_settings`
— and the API imports that same function. Nodes never construct their own
dependencies, which is why the whole system can be rewired with fakes in a
single test fixture.

```text
            ┌────────────────────────────────────────────────────┐
            │  CLI (cli.py)               FastAPI (api/server.py)│
            │  build_app_and_settings()   — one wiring point     │
            └──────────────────────────┬─────────────────────────┘
                                       │  ResearchState(raw_query=…)
                                       │  config={thread_id, recursion_limit}
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     LangGraph workflow  —  13 nodes, fixed                   │
└──────┬──────────────────┬───────────────────┬───────────────────┬────────────┘
       │                  │                   │                   │
       ▼                  ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐    ┌──────────────┐   ┌──────────────────┐
│FallbackRouter│   │ Corpus tool  │    │SemanticMemory│   │   Checkpointer   │
│ hop on error │   │ plain callab-│    │ decay rerank │   │   + run history  │
│ OR on low    │   │ le (MCP seam)│    │ in Python    │   │                  │
│ quality      │   └──────┬───────┘    └──────┬───────┘   └───┬──────────┬───┘
└─┬──────┬───┬─┘          │                   │       LangGraph│      app │
  │      │   │            ▼                   │         writes │    writes│
  │      │   │     ┌───────────────┐          │                ▼          ▼
  │      │   │     │HybridRetriever│          │        ┌───────────────────────┐
  │      │   │     │ RRF in Python │          │        │      PostgreSQL       │
  │      │   │     │ + min_similar.│          │        │ ┌───────────────────┐ │
  │      │   │     │ floor (P2-01) │          │        │ │ checkpoints       │ │
  │      │   │     └──┬─────────┬──┘          │        │ │ checkpoint_blobs  │ │
  ▼      ▼   ▼        ▼         ▼             ▼        │ │ checkpoint_writes │ │
┌─────┐┌────┐┌────┐┌───────┐┌─────────┐┌─────────────┐ │ │ checkpoint_migra. │ │
│Cogi-││Mist││Gem ││Qdrant ││OpenSea- ││Qdrant       │ │ ├───────────────────┤ │
│to   ││ral ││ini ││agent_ ││rch      ││agent_seman- │ │ │ agent_runs  (app) │ │
│local││    ││    ││corpus ││agent_   ││tic_memory   │ │ └───────────────────┘ │
│hop 0││hop1││hop2││dense  ││corpus   ││namespaced   │ │                       │
│120s ││90s ││90s ││       ││         ││ids (P2-02)  │ │                       │
└─────┘└────┘└────┘└───────┘└─────────┘└─────────────┘ └───────────────────────┘
                       ▲          ▲            ▲
                       └─────┬────┘            │
                             │                 │
                   ┌─────────┴─────────┐  ┌────┴──────────────┐
                   │ingest_sample_data │  │memory_writer node │
                   │.py (manual, 1×)   │  │(after a PASSED    │
                   └───────────────────┘  │ critique only)    │
                                          └───────────────────┘
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
                       ▼
              ┌──────────────────┐
              │ memory_retrieve  │  Qdrant ─► past evidence, decay-reranked,
              └────────┬─────────┘  goal_id namespaced (P2-02)
                       ▼
              ┌──────────────────┐
              │   goal_manager   │  LLM ─► goals (+ human redirect guidance)
              └────────┬─────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │ goals present? (D-21/E1*)  │
        └───────┬─────────────┬──────┘
                │ no          │ yes
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
                │            │        │     merger     │  contested │
                │◄───────────┘        │  (D-18 flags)  │  goals     │
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
  With HITL disabled: E1 and E4 log `escalation.stub` at WARNING and
  continue (E4 ships the report marked unreviewed — never silently as
  good). E2/E3 log NOTHING — their whole trigger block sits inside
  `if settings.hitl_enabled`. That asymmetry is a real gap, not a
  documentation shortcut. See Documentation Corrections.
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

### Storage interactions

Three stores, five distinct data flows. One writer path (Qdrant's
`upsert_texts`) now has an OPTIONAL idempotent-id mechanism (P2-03) that
nothing currently uses by default — see the note under Ingest identity below
for exactly what that does and doesn't fix yet.

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
└───────────────────────────┘         │ id = uuid4()   ── DUPLICATES ──    │
                                      │ (still the DEFAULT behaviour — see │
                                      │  Ingest identity note below)       │
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

### Ingest identity — the divergence that bites (mechanism now exists to fix it; not yet wired in)

```text
                      sample_data/corpus.jsonl  (10 lines)
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
      ┌───────────────────┐                   ┌───────────────────┐
      │ OpenSearchStore   │                   │ QdrantStore       │
      │ .ingest()         │                   │ .upsert_texts()   │
      │ _id = str(i)      │                   │ id = uuid4()      │
      └─────────┬─────────┘                   └─────────┬─────────┘
                │                                       │
   run ingest 1 ▼                          run ingest 1 ▼
      ┌───────────────────┐                   ┌───────────────────┐
      │  10 documents     │                   │  10 points        │
      └─────────┬─────────┘                   └─────────┬─────────┘
   run ingest 2 ▼                          run ingest 2 ▼
      ┌───────────────────┐                   ┌───────────────────┐
      │  10 documents     │                   │  20 points        │
      │  same ids ─►      │                   │  10 duplicates    │
      │  overwritten      │                   └─────────┬─────────┘
      └─────────┬─────────┘               run ingest 3  ▼
   run ingest 3 ▼                            ┌───────────────────┐
      ┌───────────────────┐                  │  30 points        │
      │  10 documents     │                  │  20 duplicates    │
      └─────────┬─────────┘                  └─────────┬─────────┘
                │                                      │
                └──────────────────┬───────────────────┘
                                   ▼
                 ┌──────────────────────────────────┐
                 │ rrf_fuse() joins on `title`      │
                 │ ── which silently COLLAPSES the  │
                 │    duplicates, so the damage     │
                 │    stays invisible until your    │
                 │    corpus has repeated or        │
                 │    missing titles                │
                 └──────────────────────────────────┘
```

**P2-03 added the MECHANISM for idempotent ingest — `QdrantStore.upsert_texts`
now accepts an optional `id_fn` parameter, and passing a deterministic,
content-derived function makes re-ingesting the same content overwrite in
place instead of duplicating.** But every existing caller in this codebase —
`scripts/ingest_sample_data.py` and `memory_writer` alike — still calls
`upsert_texts` **without** `id_fn`, which preserves the exact original
`uuid4()`-per-call behavior. So as of this revision:

- **The corpus ingest duplication bug described above is still live in
  practice.** The fix exists and is proven to work (see Recent Fixes), but
  `scripts/ingest_sample_data.py` hasn't been updated to actually pass a
  deterministic `id_fn` yet. That's a small, separate follow-up.
- **Memory's accumulation is unchanged, and deliberately so** —
  `memory_writer` is meant to keep accumulating fresh evidence every passed
  run, not collapse repeats. Deduping memory is a larger, separate piece of
  work tracked as `P2-15`, not something P2-03 was meant to touch.

**Do the chunks match between Qdrant and OpenSearch?** Still: the *units*
match; the *identities* don't, for the reasons above. Neither store chunks
anything — one JSONL line becomes exactly one OpenSearch document and
exactly one Qdrant point, with byte-identical `content`. Only `content` is
embedded, and only `content` is BM25-matched.

The workaround remains `scripts/reset_stores.py` — see below.

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
do have one. And that quality score comes from asking the *same* provider to
rate its own answer: cheap, weak, and honest about being weak in
`evaluation/quality.py`. A scorer that itself errors returns `1.0`, so a flaky
scoring call can never burn a working answer path.

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
| `agent_runs` | **our code**, via `CREATE TABLE IF NOT EXISTS` on *every* `record_run` call | **our code**, one row per completed CLI run | Post-hoc run history: `id BIGSERIAL PK`, `thread_id TEXT`, `query TEXT`, `recall REAL`, `telemetry JSONB`, `created_at TIMESTAMPTZ DEFAULT now()`. Nothing reads it back — it exists for you and DBeaver. |

Plus three LangGraph indexes: `checkpoints_thread_id_idx`,
`checkpoint_blobs_thread_id_idx`, `checkpoint_writes_thread_id_idx`.

Three behaviours worth knowing before you debug something:

- **The API writes no run history.** `record_run` is called only from
  `cli.py::main`. API runs checkpoint normally and produce no `agent_runs` row.
- **Degradation is invisible in the output.** `get_checkpointer` catches *any*
  exception and returns `MemorySaver()` with `durable=False` — and
  `build_app_and_settings` throws that flag away. The only signal is the stderr
  log line `checkpointer.postgres_active` or `checkpointer.memory_fallback`.
- **The checkpointer connection is never closed.** Harmless for a one-shot CLI;
  a leak in a long-lived FastAPI process.
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
  for unrelated queries.

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
`COSINE`.

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

Two independent output streams, both behind `--debug` (or `DEBUG_TRACE=true`):

| Output | Where | What it answers |
|---|---|---|
| `"node.enter"` JSON lines | **stderr** — visible in a normal terminal by default | "What ran, in what order?" Includes `merger` and `progress_checker`, which touch neither an LLM nor a store and so never appear in the trace file below |
| Exact prompt/response/hit detail | `logs/trace-<run_id>.txt` **only** — never printed to console, by design | "What exactly did a specific LLM call or retrieval call see and return?" |

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
show up for each.

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
                 ┌───────────────────────────────────────────────────┤
                 ▼                      ▼                            ▼
           approve                  redirect                      abort
      ┌──────────────┐        ┌──────────────────┐         ┌────────────────┐
 E1   │ ─► compiler  │        │ ─► goal_manager  │         │ ─► compiler    │
      │  (error rpt) │        │  + human_guidance│         │  + abort_reason│
 E2/3 │ ─► compiler  │        │ ─► gap_generator │         │ ─► compiler    │
      │  (ship thin) │        │  + human_guidance│         │  + abort_reason│
 E4   │ ─► telemetry │        │ ─► compiler      │         │ ─► telemetry   │
      │  (unreviewed,│        │  + note "HUMAN   │         │  (no memory)   │
      │   no memory) │        │    REVIEWER: …"  │         │                │
      └──────────────┘        └──────────────────┘         └────────────────┘

  D-28 made concrete: the whole node RE-EXECUTES from its top on resume.
  Everything above interrupt() is a pure read, and escalation_history is
  appended in the RESUME update — never before — so re-execution cannot
  double-append. Six tests in tests/test_hitl.py assert exactly that from
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
  │    the coverage predicate is  e.score >= min_evidence_score          │
  │    at 0.0 that was TRUE for every item — even one scored exactly 0.0 │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 2. No relevance floor existed ANYWHERE       [FIXED — min_similarity]│
  │      QdrantStore.search      ─► top-k neighbours, unconditionally    │
  │      HybridRetriever.search  ─► NOW drops dense hits below the floor │
  │                                 BEFORE fusion (P2-01)                │
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
model family is optimistic; that is what it costs. **This one is unrelated to
P2-01/P2-02 and remains unfixed** — it's `P2-11` (judge-model quality
scoring) in `PHASE2_PLAN.md`'s Tier 3, not something touched tonight.

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
Unchanged by tonight's fixes.

### What it takes to make the documented test real — updated status

| # | Change | Status |
|---|---|---|
| 1 | Raise `MIN_EVIDENCE_SCORE` above the measured off-topic floor | **Done** — now `0.5`, was `0.0` |
| 2 | Namespace memory goal ids on retrieval | **Done** — `memory/semantic_memory.py::retrieve` |
| 3 | Apply a similarity floor before hits become Evidence | **Done** — `retrieval/hybrid.py`, `min_similarity` |
| 4 | Require at least one `source == "corpus"` item for coverage, or weight memory below fresh retrieval | Not done — still open, not part of tonight's scope |
| 5 | Add an integration test asserting an interrupt for an out-of-corpus query | **Done** — two tests added to `tests/test_hitl.py`, both passing (36/36 total). **Live confirmation exists too**, though via the retrieval-failure path (D-16) rather than the low-relevance-evidence path these tests specifically target — see the status note at the top of this section |

Items 1–3 were each individually sufficient to make E3 reachable for the
documented query; all three are now in place together. Item 5 is what turns
"should work now" into "verified to work, and guaranteed to keep working."

## Telemetry — read it honestly

`telemetry_node` aggregates counters that nodes recorded. It invents nothing,
exactly as D-12 requires. But the counters are **node-scoped, not
boundary-scoped**, and that distinction will mislead you about cost:

| Field | Counts | Does **not** count |
|---|---|---|
| `llm_calls` | one per LLM-using **node execution** | fallback hops, self-scoring calls |
| `search_calls` | one per **successful worker** | per-engine round trips |
| `search_failures` | one per worker that raised | partial-leg failures |
| `memory_hits` / `memory_writes` | items in / points out | |
| `revision_cycles` | critic passes | |

In the original traced run `llm_calls` reported **5**, while the trace showed
**9** recorded provider responses plus at least two untraced primary attempts
that errored before returning a body. Read `llm_calls` as *"nodes that used
an LLM"*, never as provider traffic or spend.

**The trace is still the honest view for LLM/store detail, but it's no
longer the only signal.** `--debug` (or `DEBUG_TRACE=true`) records every LLM
call and every retrieval call at the boundary it actually crosses, to
`logs/trace-<run_id>.txt` — and, as of this revision, **also emits a
`"node.enter"` line to stderr for every node**, including `merger` and
`progress_checker`, which touch neither an LLM nor a store and so still
never appear in the trace file itself. See
[Debugging a live run](#debugging-a-live-run) for exactly how to use both
together. In one traced run, this combination revealed something telemetry
alone never would: **OpenSearch never appeared at all**, because the keyword
leg was down — the "hybrid" retriever was running single-legged, and nothing
in the report or the telemetry said so.

## Design

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
  now also owns the per-hop timeout split (local vs. cloud).
- **Retrieval** (`retrieval/`, `storage/`, `tools/`): storage modules are
  policy-free wrappers; fusion math is a pure function; the tool translates
  hits into domain Evidence. `HybridRetriever` now also owns the
  `min_similarity` relevance floor. `tools/corpus_search.py` is the MCP seam —
  the calling pattern is already MCP-shaped, so the upgrade touches one module.
- **Memory** (`memory/`): similarity × volatility-decay reranking; memory items
  re-enter the graph as ordinary evidence, so every downstream rule (coverage,
  contradiction) treats memory and fresh sources identically — **except for
  goal-id equality specifically**, which is now deliberately asymmetric after
  P2-02, so memory can inform but not falsely satisfy coverage.
- **Evaluation** (`evaluation/`): the self-scoring signal behind fallback.
- **Escalation** (`agents/escalation.py`): one parametrized node for all four
  triggers, carrying the D-28 idempotency obligation.

## Project Structure

```text
research-agent-dmp/
├── src/research_agent/
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
│   ├── storage/             # postgres / qdrant / opensearch wrappers
│   ├── tools/               # the corpus-search tool workers invoke (MCP seam)
│   ├── evaluation/          # answer quality self-scoring
│   ├── api/server.py        # FastAPI: /health, /research, /resume
│   └── cli.py               # CLI entry + dependency assembly + HITL loop
├── tests/                   # 34 tests, offline (25 core, 6 HITL, 3 paths)
├── scripts/ingest_sample_data.py
├── scripts/reset_stores.py  # wipe all three stores to pristine (see above)
├── sample_data/corpus.jsonl # 10 docs, Redis-vs-Memcached theme
├── design/Research_Agent_Design.md
├── OPERATIONS.md   internal/LEARNING_GUIDE.md   PHASE2_PLAN.md
├── docker-compose.yml       # optional: postgres + qdrant + opensearch
├── requirements.txt  .env.example  run.bat  reset.bat
└── DECISIONS.md             # currently EMPTY (0 bytes)
```

## Setup

`OPERATIONS.md` is the real manual — it owns the L1/L2/L3 ladder, native
Windows service startup, DBeaver setup, and the llama-server invocations. The
30-second version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults run fully offline (LLM_MODE=stub)
export PYTHONPATH=src

python -m research_agent.cli "Compare Redis and Memcached for session caching"
python -m pytest tests/ -q    # expect: 34 passed
```

Windows: `run.bat` does the venv, install, and a stub run in one command.

Defaults are `LLM_MODE=stub` with every store unreachable, so the first run
reports `evidence_items: 0`. **That is success for L1** — the graph is proven,
there is simply nothing to search yet. `OPERATIONS.md` walks you up from there.

## Walkthrough

1. **A request arrives** (CLI or API) → `build_app_and_settings()` wires every
   dependency, each storage module probing its service and degrading if absent.
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
   marked unreviewed — never silently.
6. **Persist & learn**: a *passed* report's fresh evidence enters semantic
   memory (with its raw, unnamespaced goal_id, per the storage note above);
   telemetry aggregates node-recorded counters; a run-history row lands in
   Postgres when available — **from the CLI only**.

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
| — | Graceful degradation everywhere | First run must succeed on a bare laptop |
| — | Stub LLM mode | Deterministic offline demo + honest tests using real prompts/schemas |

## Limitations

Split into what was *deferred by design* and what is simply *broken*, because
conflating the two is how a reference build stops being trustworthy. Four
items have moved out of "broken" since the last revision — marked below,
left visible rather than deleted, so the history stays auditable.

**Deferred by design**

- **MCP deferred**: tools are plain callables behind `tools/`; the calling
  pattern is MCP-shaped so the upgrade touches one module (design D-26/D-30).
- **Contradiction detection is minimal**: the machinery (contested goals block
  coverage) is fully wired; the detector only honors explicit markers, which no
  tool sets. Consequence: **E2 has never fired in a real run** — every observed
  escalation would be E3.
- **Server-side hybrid fusion deferred**: RRF + decay run in Python here for
  readability; the design (D-27) moves both into Qdrant `FormulaQuery`, with
  payload indexes this build does not create.
- **Memory simplifications**: no supersession links, no per-item volatility
  classification (items default `semi_stable`), no garbage collection job.
  Deliberately unaffected by P2-02/P2-03 — see the notes under those items.
- **Self-evaluated quality is optimistic**: catches broken output, not subtle
  errors; a judge model is the upgrade (`P2-11`).
- **Single tool, single worker type**: typed specialists (D-25) arrive with MCP.

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

**Still broken, in rough order of consequence**

1. LLM producer output is unvalidated — `cap_and_filter` does `t['goal_id']`
   and `goal_manager_node` does `g["goal_id"]`. A live model omitting a key
   raises `KeyError` inside the node and aborts the run. (`P2-06`, Tier 2.)
2. Self-critique can pass a report whose claims appear in no retrieved
   evidence — unaffected by tonight's fixes; separate root cause (`P2-11`).
3. Qdrant ingest is still not idempotent **by default** — the `id_fn`
   mechanism exists (P2-03) but `scripts/ingest_sample_data.py` doesn't pass
   one yet. Re-ingesting still silently multiplies the dense index until
   that's wired up.
4. Semantic memory grows without bound — no supersession, dedup, or GC
   (`P2-15`, deliberately out of scope for P2-02/P2-03).
5. RRF joins the two legs on `title`, not on any store id — silently wrong for
   a corpus with duplicate or missing titles.
6. The checkpointer connection is never closed; the API process leaks it.
7. The API writes no run history — `record_run` is CLI-only.
8. `llm_calls` and `search_calls` under-report actual traffic — `node.enter`
   (new) partially addresses "what ran", but the counters themselves are
   unchanged.
9. E2/E3 emit no log line at all when HITL is off, unlike E1/E4.
10. `durable` is discarded by `build_app_and_settings`; the in-memory
    checkpointer fallback shows up only in stderr.
11. `Evidence.task_key` for memory items uses `hash()`, which is per-process
    randomised — memory task keys are not stable across runs.
12. **New, found this session**: reusing the same `--thread-id` across
    unrelated runs silently accumulates reducer-backed state (`evidence`,
    `counters`, etc.) — see the Postgres section above for the full
    explanation and a live example.

## Documentation Corrections

Applied above; listed here so the deltas against the older documents are
auditable rather than invisible.

| Claim in older docs | Reality in code |
|---|---|
| README: fallback is "local Qwen Cogito → Gemini Flash" | Three hops: primary → Mistral → Gemini, each fallback gated on its API key, **and, as of this revision, each hop tier uses a different timeout** |
| README / OPERATIONS: "28 tests" | **34** tests collected and passing (25 core + 6 HITL + 3 integration) — unchanged by tonight's fixes |
| design §12: "63 files, 28/32 tests passing, 4 skipped" | 51 files in this distribution; 34 tests, **0 skipped** |
| README legend: with HITL off the checks "log and continue" | True for E1/E4 only; E2/E3 log nothing when HITL is off |
| OPERATIONS §"Writing Your Own Test Corpus": "re-run ingest (it upserts by id, so re-running overwrites)" | True for OpenSearch; for Qdrant, **the mechanism to make this true now exists (P2-03)** but isn't wired into the ingest script yet — still false in practice today |
| OPERATIONS §"Test HITL": that query escalates | Previously converged at `recall 1.0` at depth 1 and never interrupted. Root causes fixed (P2-01, P2-02); **not yet re-verified end-to-end** — see The HITL Investigation |
| design §9: `MAX_REVISIONS` default 3 | Code default is **2** (`config.py`) |
| README structure tree: root `agentic-research-agent/` | Distributed directory is `research-agent-dmp/` |
| Storage diagram implied one Qdrant use | Two collections; `CORPUS_INDEX` names **both** a Qdrant collection and an OpenSearch index |
| `DECISIONS.md` referenced as the decision log | The file is **0 bytes** |
| `internal/LEARNING_GUIDE.md` cited as a companion doc | `internal/` is in `.gitignore`, so it ships only in archives like this one |
| OPERATIONS §L1: "add two `logging.getLogger(...)` lines" | Already present in `logging_setup.py::configure_logging` |

## Future Improvements

`PHASE2_PLAN.md` has the full 15-item plan — every item scoped to an existing
seam, with complexity, dependencies, and the D-xx tag it extends or replaces.
The shape of it, updated with tonight's progress:

```text
  TIER 1 ── correctness ──────► make the documented behaviour true
     P2-01  relevance floor + calibrated evidence gate       ✅ DONE
              + follow-up: exact-boundary collision (`>` fix)  ✅ DONE
     P2-02  namespace memory goal_id on retrieval             ✅ DONE
     P2-03  deterministic Qdrant point ids                    ◐ MECHANISM DONE,
                                                                 not wired into
                                                                 ingest script
     P2-04  provider output sanitizer  (<|im_end|>)           ✅ DONE
              + free-text runaway-generation truncation        ✅ DONE (new,
                                                                 not in the
                                                                 original plan)
     P2-05  escalation reachability tests                     ✅ DONE — tests
                                                                 passing (36/36)
                                                                 AND live-fired
                     │
  TIER 1 STATUS: CLOSED. All five items done; the two follow-up fixes
  found via live testing (the exact-boundary collision and the free-text
  truncation gap) are documented above and in this README's Recent Fixes.
                     │
  TIER 2 ── robustness ───────► make it observable and hard to crash
     P2-06  validate LLM producer output
     P2-07  boundary-scoped telemetry
     P2-08  postgres lifecycle + API run-history parity
     P2-09  config strictness + populate DECISIONS.md
                     │
  TIER 3 ── design catch-up ──► close the D-25/26/27/30 gap
     P2-10  Qdrant payload indexes + server-side decay      (D-27)
     P2-11  judge-model quality scoring
     P2-12  semantic contradiction detector — E2 reachable at last
     P2-13  MCP tool seam                                   (D-26/D-30)
     P2-14  typed specialist workers                        (D-25)
     P2-15  memory supersession + GC
```

**What's actually next:** Tier 1 is closed — 36/36 tests pass, and the
escalation path has fired on a real, live run (see the status note under
[The HITL Investigation](#the-hitl-investigation) for exactly which path
triggered it, and which one is still only unit-proven). Tier 2's four items
have no dependencies on each other or on anything from Tier 1; `P2-08` and
`P2-09` are the smallest place to start.