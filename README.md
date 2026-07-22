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
> **This revision is code-truth.** Every claim below was verified against the
> source and against a real debug trace (`logs/trace-run-0d7d0448906a.txt`).
> Where the design document or an earlier README said something the code does
> not do, the delta is recorded in [Documentation Corrections](#documentation-corrections)
> rather than quietly repeated. The most consequential one: **HITL is fully
> implemented and, for the query `OPERATIONS.md` tells you to test it with,
> unreachable.** That story is in [The HITL Investigation](#the-hitl-investigation).

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

## Features

**Capabilities**
- Goal-driven research over an ingested corpus (hybrid dense+keyword search).
- Parallel retrieval fan-out with concurrency-safe state merging.
- Iterative gap-filling bounded by depth and recall targets.
- Bounded self-critique with grounded rewrites.
- Cross-run semantic memory with volatility-aware staleness decay.
- **Three-hop** LLM fallback routing (local Qwen Cogito → Mistral → Gemini
  Flash) with a self-scored quality threshold, each hop joining the chain only
  if its API key is configured.
- Human-in-the-loop escalation (`HITL_ENABLED=true`): the graph pauses via
  LangGraph `interrupt()` on four triggers — E1 zero goals, E2 contested
  goals, E3 cannot-converge, E4 critique exhausted — and resumes on
  approve/redirect/abort under the same thread_id. CLI prompts on stdin;
  the API returns `status: interrupted` plus a `/resume` endpoint.
- Per-run debug tracing (`--debug`) dumping the exact prompt, raw response,
  provider, tokens and latency of every LLM call plus every retrieval engine's
  raw hits — the honest view of what the run really did.
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
  │      │   │     └──┬─────────┬──┘          │        │ ┌───────────────────┐ │
  │      │   │        │         │             │        │ │ checkpoints       │ │
  ▼      ▼   ▼        ▼         ▼             ▼        │ │ checkpoint_blobs  │ │
┌─────┐┌────┐┌────┐┌───────┐┌─────────┐┌─────────────┐ │ │ checkpoint_writes │ │
│Qwen ││Mist││Gem ││Qdrant ││OpenSea- ││Qdrant       │ │ │ checkpoint_migra. │ │
│local││ral ││ini ││agent_ ││rch      ││agent_seman- │ │ ├───────────────────┤ │
│hop 0││hop1││hop2││corpus ││agent_   ││tic_memory   │ │ │ agent_runs  (app) │ │
└─────┘└────┘└────┘│dense  ││corpus   ││             │ │ └───────────────────┘ │
                   └───────┘└─────────┘└─────────────┘ └───────────────────────┘
                       ▲          ▲            ▲
                       └─────┬────┘            │
                             │                 │
                   ┌─────────┴─────────┐  ┌────┴──────────────┐
                   │ingest_sample_data │  │memory_writer node │
                   │.py (manual, 1×)   │  │(after a PASSED    │
                   └───────────────────┘  │ critique only)    │
                                          └───────────────────┘
```

Two things this picture is telling you that the old one did not. First, the
fallback chain is **three** hops, not two — Mistral sits between the local
model and Gemini, and in the traced run it served nearly every call. Second,
Postgres has **two independent writers who do not know about each other**: the
LangGraph library owns the four `checkpoint*` tables, and this repo's own code
owns `agent_runs`. That split is exactly what you need to remember when you
reset the database.

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
              │ memory_retrieve  │  Qdrant ─► past evidence, decay-reranked
              └────────┬─────────┘
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
                │            │        │   (Send fan-out)  │         │
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

### Request flow (one worker, and why it always finds something)

The earlier version of this diagram stopped politely at "fused hits (RRF)". The
interesting part is what happens *inside* the fusion box, because it is the
root cause of the HITL defect further down.

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
    │                    │                  │       ┌──────────┴───────────┐  │
    │                    │                  │       │ rrf_fuse()           │  │
    │                    │                  │       │ join key = title, or │  │
    │                    │                  │       │ content[:60] — NOT   │  │
    │                    │                  │       │ any store's id       │  │
    │                    │                  │       │ NO relevance floor   │  │
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

Read that fusion box carefully. **A dense index always returns its k nearest
neighbours.** "Nearest" is not "relevant". Ask the Redis-vs-Memcached corpus
about Cassandra at petabyte scale and it will cheerfully hand back three Redis
documents, scored 0.48–0.50 after the `RRF_SQUASH` squash. Nothing downstream
rejects them, because `MIN_EVIDENCE_SCORE` defaults to `0.0`. Hold that thought.

### Storage interactions

Three stores, five distinct data flows, and exactly one of them is idempotent.

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
                                      │ vector: fastembed(content)         │
                                      │ payload: title, topic, content,    │
                                      │          created_at                │
                                      └────────────────────────────────────┘

┌───────────────────────────┐         ┌────────────────────────────────────┐
│ memory_retrieve node      │◄────────┤ Qdrant                             │
│  similarity × decay       │         │ collection: agent_semantic_memory  │
│  over-fetch 2×k, cut to k │         │ id = uuid4()   ── DUPLICATES ──    │
│                           │         │ vector: fastembed(content)         │
│ memory_writer node        ├────────►│ payload: content, goal_id,         │
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

The dotted line inside the Postgres box is the ownership boundary. Above it,
LangGraph. Below it, us. Both halves are recreated automatically after a wipe —
`PostgresSaver.setup()` rebuilds the top, `record_run`'s
`CREATE TABLE IF NOT EXISTS` rebuilds the bottom.

### Ingest identity — the divergence that bites

This is the diagram I most wish had existed before I read the trace.

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

**Do the chunks match between Qdrant and OpenSearch?** The *units* match; the
*identities* do not. Neither store chunks anything — one JSONL line becomes
exactly one OpenSearch document and exactly one Qdrant point, with
byte-identical `content`. Only `content` is embedded (title and topic ride along
in the payload but contribute nothing to the vector), and only `content` is
BM25-matched (title and topic are mapped but never queried). The divergence is
entirely in the id scheme, and it compounds every time you re-ingest.

Semantic memory has the same defect on the write path, and the supplied trace
shows it plainly: **ten memory hits, ten distinct point ids, one identical
sentence.**

The fix for today is a reset script; the fix for tomorrow is P2-03 in
`PHASE2_PLAN.md`.

### LLM fallback chain

One policy, applied identically at every hop. It lives in exactly one place —
`llm/router.py` — and nothing else in the codebase decides when to fall back.

```text
     complete_json(messages)                    complete(messages)
              │                                          │
              ▼                                          ▼
   ┌─────────────────────┐                    ┌─────────────────────┐
   │ hop 0: local Qwen   │                    │ hop 0: local Qwen   │
   │         Cogito      │                    │         Cogito      │
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
   │ (mistral-small)     │   its key is set   │ (mistral-small)     │
   └──────────┬──────────┘                    └──────────┬──────────┘
              ▼                                          ▼
   ┌─────────────────────┐                    ┌─────────────────────┐
   │ hop 2: Gemini Flash │   joins ONLY if    │ hop 2: Gemini Flash │
   └──────────┬──────────┘   its key is set   └──────────┬──────────┘
              │                                          │
              ▼                                          ▼
   chain exhausted ─► raise                    chain exhausted ─► return
   the LAST provider's error                   the LAST answer we got
   (because no answer exists)                  (better thin than nothing)

  ┌────────────────────────────────────────────────────────────────────┐
  │ OBSERVED DEFECT — trace run-0d7d0448906a                           │
  │                                                                    │
  │ The local model answers correctly, then appends its chat           │
  │ template's end-of-turn sentinel:                                   │
  │                                                                    │
  │       { "goals": [ … ] } <|im_end|>                                │
  │                                                                    │
  │ _extract_json() strips markdown fences and nothing else, so        │
  │ json.loads raises — and EVERY structured call falls through to     │
  │ Mistral. In that run the local model served ZERO JSON calls        │
  │ despite responding successfully at goal_manager and critic.        │
  │                                                                    │
  │ Fix either side: strip trailing special tokens in the client, or   │
  │ set llama-server's stop tokens / chat template.  See P2-04.        │
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
| `goal_id` | `Evidence.goal_id` | The goal id **of the run that stored it** — this is the collision defect described below |
| `volatility` | `Evidence.volatility.value` | Always `semi_stable` in practice; nothing classifies volatility |
| `source_query` | the storing run's `raw_query` | Provenance only; never filtered on |
| `created_at` | `time.time()` at upsert | Drives the decay rerank |

At **search** time `QdrantStore.search` adds two keys to each returned dict that
are **not stored**: `similarity` (raw Qdrant score) and `age_days` (derived from
`created_at`). Decay is applied on top of those, in Python, in
`memory/semantic_memory.py`.

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
back to a pristine, re-ingestable state. Because there is no idempotent ingest,
this is not an optional convenience — it is the only way to reload a corpus
without silently multiplying the dense index.

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

## The HITL Investigation

`OPERATIONS.md` tells you to switch `HITL_ENABLED=true`, ask
*"Compare Redis vs Cassandra vs DynamoDB at petabyte scale"*, and watch the CLI
pause with `action [approve/redirect/abort]:`. It does not pause. Here is why,
and what it costs to fix.

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
`status: "interrupted"` and exposes `POST /resume`. All six HITL tests pass.

### So why doesn't the documented query escalate?

Because **recall is 1.0 before anything can go wrong.** The trace of
`run-0d7d0448906a` runs straight through — `classify → memory_retrieve →
goal_manager → task_expander → 6 workers → compiler → critic → END` — at
`iteration_depth = 1`, with no `gap_generator` and no interrupt.

```text
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. MIN_EVIDENCE_SCORE defaults to 0.0                                │
  │    the coverage predicate is  e.score >= min_evidence_score          │
  │    at 0.0 that is TRUE for every item — even one scored exactly 0.0  │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 2. No relevance floor exists ANYWHERE                                │
  │      QdrantStore.search      ─► top-k neighbours, unconditionally    │
  │      HybridRetriever.search  ─► fuses whatever it is handed          │
  │      corpus_search           ─► converts every hit into Evidence     │
  │    an out-of-domain query therefore CANNOT produce zero evidence     │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 3. task_expander emitted one task per goal (g1…g5)                   │
  │    every goal received 3 off-topic Redis documents, 0.48–0.50        │
  │    every goal therefore "covered"  ─►  recall = 1.0                  │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 4. route_convergence: recall 1.0 >= RECALL_TARGET 0.85 ─► compiler   │
  │                                                                      │
  │    The E2/E3 checks in progress_checker AND in gap_generator are     │
  │    BOTH guarded by `recall < recall_target`. Neither is evaluated.   │
  │    HITL_ENABLED=true is irrelevant — no trigger can be raised.       │
  └──────────────────────────────────────────────────────────────────────┘
```

E4 did not fire either: both critics returned `passed: true` on the **first**
pass — including on a report whose Cassandra and DynamoDB sections are supported
by no retrieved evidence whatsoever. Self-critique by the same model family is
optimistic; that is what it costs.

Reproduced deterministically, no services required: `hitl_enabled=True`, stub
LLM, and a tool returning one `score=0.0` item per task →
`recall: 1.0, iterations: 1`, no interrupt.

### The contributing defect: memory `goal_id` collision

`SemanticMemory.retrieve` rebuilds `Evidence` with
`goal_id=h.get("goal_id", "memory")` — the goal id of **whichever earlier run
stored that fact**. Goal ids are `g1, g2, g3…` in every single run.

```text
  run 1: "Compare Redis and Memcached for session caching"
         goals g1…g5  ─►  memory_writer stores evidence tagged goal_id=g3
                                                                    │
                                                                    ▼
  run 2: "Compare Redis vs Cassandra vs DynamoDB at petabyte scale"
         goals g1…g5  ◄── memory_retrieve returns it, still tagged g3
                          progress_checker: "g3 has evidence" ─► COVERED
                          …by a sentence about Memcached thread counts.

  Worse: memory items score ~0.75, corpus hits ~0.50 — so memory OUTRANKS
  fresh retrieval in the compiler's evidence listing.
```

The trace shows five such items — all the same Memcached throughput sentence,
tagged `g2`/`g3`/`g5` — entering a petabyte-scale comparison as covering
evidence.

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

### What it takes to make the documented test real

Smallest change first. Items 1–3 are each individually sufficient to make E3
reachable for that query; all five together make the escalation path
trustworthy rather than incidental.

| # | Change | Seam | Effort |
|---|---|---|---|
| 1 | Raise `MIN_EVIDENCE_SCORE` above the measured off-topic floor (trace: ~0.50) | `.env` only, zero code | config |
| 2 | Namespace memory goal ids on retrieval so memory can never impersonate a current-run goal | `memory/semantic_memory.py::retrieve`, one line | small |
| 3 | Apply a similarity floor before hits become Evidence, so a query with nothing close returns `[]` | `retrieval/hybrid.py` or `tools/corpus_search.py` | small |
| 4 | Require at least one `source == "corpus"` item for coverage, or weight memory below fresh retrieval | `agents/gathering.py::progress_checker_node` | small |
| 5 | Add an integration test asserting an interrupt for an out-of-corpus query, so this cannot regress | `tests/` | small |

These are Tier 1 of `PHASE2_PLAN.md` (items P2-01, P2-02, P2-05).

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

In the traced run `llm_calls` would report **5**, while the trace shows **9**
recorded provider responses plus at least two untraced primary attempts that
errored before returning a body. Read `llm_calls` as *"nodes that used an LLM"*,
never as provider traffic or spend.

**The trace is the honest view.** `--debug` (or `DEBUG_TRACE=true`) records
every LLM call and every retrieval call at the boundary it actually crosses, to
`logs/trace-<run_id>.txt`. In that same run the trace reveals something
telemetry never would: **OpenSearch never appears at all**, because the keyword
leg was down. The "hybrid" retriever was running single-legged, and nothing in
the report or the telemetry said so.

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
  providers; fallback policy lives in exactly one place (`router.py`).
- **Retrieval** (`retrieval/`, `storage/`, `tools/`): storage modules are
  policy-free wrappers; fusion math is a pure function; the tool translates
  hits into domain Evidence. `tools/corpus_search.py` is the MCP seam — the
  calling pattern is already MCP-shaped, so the upgrade touches one module.
- **Memory** (`memory/`): similarity × volatility-decay reranking; memory items
  re-enter the graph as ordinary evidence, so every downstream rule (coverage,
  contradiction) treats memory and fresh sources identically. That symmetry is
  elegant and, as the collision defect above shows, currently too symmetric.
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
│   ├── retrieval/           # hybrid dense+BM25 with RRF (pure fusion fn)
│   ├── memory/              # semantic memory with decay
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
   past evidence (decay-reranked) → `goal_manager` composes goals (memory hints
   included) → `task_expander` emits a ranked backlog capped at `MAX_FANOUT` —
   overflow is the *producer's* decision (D-13), so dispatch is always total.
3. **Gather (cyclic)**: `dispatch_tasks` fans one `Send` per task to
   `search_worker` instances that run in the same superstep; each returns only
   reducer-backed keys (enforced), recording success or failure-with-depth.
   `merger` flags contradictions; `progress_checker` computes quality-gated,
   contradiction-aware recall and ticks the depth counter. Below target and
   depth: `gap_generator` produces new tasks (dedup + failed-key rules applied)
   and the cycle repeats.
4. **Decisions**: the graph decides *where to go*; models decide *content*.
   Termination is guaranteed by four independent bounds: recall target, depth
   counter, empty-backlog fallthrough, recursion-limit backstop.
5. **Compile & critique**: the report is drafted, judged for faithfulness and
   completeness only, and rewritten against explicit notes up to
   `MAX_REVISIONS` (**code default 2**, not the 3 in design §9); exhaustion
   either interrupts for human review or logs the E4 stub and ships the report
   marked unreviewed — never silently.
6. **Persist & learn**: a *passed* report's fresh evidence enters semantic
   memory; telemetry aggregates node-recorded counters; a run-history row lands
   in Postgres when available — **from the CLI only**.

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
| D-17/D-18 | Quality-gated, contradiction-aware coverage | No recall=1.0 on junk — **inert today at `MIN_EVIDENCE_SCORE=0.0`**; contested goals drive adjudication automatically |
| D-21 | Zero goals → explicit error report | Diagnosable beats silent |
| D-22 | Bounded critique, grounded rewrites, scoped to faithfulness | One judge per question; no blind retries |
| D-23/D-28 | Escalation via `interrupt()`; nothing non-idempotent precedes the interrupt | The node re-executes on resume — history is appended in the resume update only |
| D-24 | Memory decay = rerank by volatility class, never an age filter | One TTL is wrong for both stable and volatile facts |
| D-29 | `ConfigDict(extra="forbid")` on all state models | Construction-time pollution and worker-return pollution are two failure modes; two layers |
| — | Graceful degradation everywhere | First run must succeed on a bare laptop |
| — | Stub LLM mode | Deterministic offline demo + honest tests using real prompts/schemas |

## Limitations

Split into what was *deferred by design* and what is simply *broken*, because
conflating the two is how a reference build stops being trustworthy.

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
- **Self-evaluated quality is optimistic**: catches broken output, not subtle
  errors; a judge model is the upgrade.
- **Single tool, single worker type**: typed specialists (D-25) arrive with MCP.

**Broken, in rough order of consequence**

1. `MIN_EVIDENCE_SCORE=0.0` makes the coverage gate inert, so `recall` measures
   "did anything come back", not "is this goal answered".
2. No relevance floor anywhere in retrieval — an out-of-domain query cannot
   produce zero evidence.
3. Memory `goal_id` collides across runs, letting an old run's evidence satisfy
   an unrelated goal with the same id.
4. `<|im_end|>` breaks JSON parsing from the local primary, forcing every
   structured call onto a paid fallback.
5. LLM producer output is unvalidated — `cap_and_filter` does `t['goal_id']`
   and `goal_manager_node` does `g["goal_id"]`. A live model omitting a key
   raises `KeyError` inside the node and aborts the run.
6. Self-critique passed a report whose Cassandra/DynamoDB claims appear in no
   retrieved evidence.
7. Qdrant ingest is not idempotent (`uuid4` ids); OpenSearch is. Re-ingesting
   silently multiplies the dense index.
8. Semantic memory grows without bound — no supersession, dedup, or GC.
9. RRF joins the two legs on `title`, not on any store id — silently wrong for
   a corpus with duplicate or missing titles.
10. The checkpointer connection is never closed; the API process leaks it.
11. The API writes no run history — `record_run` is CLI-only.
12. `llm_calls` and `search_calls` under-report actual traffic.
13. E2/E3 emit no log line at all when HITL is off, unlike E1/E4.
14. `durable` is discarded by `build_app_and_settings`; the in-memory
    checkpointer fallback shows up only in stderr.
15. `Evidence.task_key` for memory items uses `hash()`, which is per-process
    randomised — memory task keys are not stable across runs.

## Documentation Corrections

Applied above; listed here so the deltas against the older documents are
auditable rather than invisible.

| Claim in older docs | Reality in code |
|---|---|
| README: fallback is "local Qwen Cogito → Gemini Flash" | Three hops: primary → Mistral → Gemini, each fallback gated on its API key |
| README / OPERATIONS: "28 tests" | **34** tests collected and passing (25 core + 6 HITL + 3 integration) |
| design §12: "63 files, 28/32 tests passing, 4 skipped" | 51 files in this distribution; 34 tests, **0 skipped** |
| README legend: with HITL off the checks "log and continue" | True for E1/E4 only; E2/E3 log nothing when HITL is off |
| OPERATIONS §"Writing Your Own Test Corpus": "re-run ingest (it upserts by id, so re-running overwrites)" | True for OpenSearch, **false for Qdrant** — `uuid4` ids append duplicates |
| OPERATIONS §"Test HITL": that query escalates | It converges at `recall 1.0` at depth 1 and never interrupts |
| design §9: `MAX_REVISIONS` default 3 | Code default is **2** (`config.py`) |
| README structure tree: root `agentic-research-agent/` | Distributed directory is `research-agent-dmp/` |
| Storage diagram implied one Qdrant use | Two collections; `CORPUS_INDEX` names **both** a Qdrant collection and an OpenSearch index |
| `DECISIONS.md` referenced as the decision log | The file is **0 bytes** |
| `internal/LEARNING_GUIDE.md` cited as a companion doc | `internal/` is in `.gitignore`, so it ships only in archives like this one |
| OPERATIONS §L1: "add two `logging.getLogger(...)` lines" | Already present in `logging_setup.py::configure_logging` |

## Future Improvements

`PHASE2_PLAN.md` has the full 15-item plan — every item scoped to an existing
seam, with complexity, dependencies, and the D-xx tag it extends or replaces.
The shape of it:

```text
  TIER 1 ── correctness ──────► make the documented behaviour true
     P2-01  relevance floor + calibrated evidence gate
     P2-02  namespace memory goal_id on retrieval
     P2-03  deterministic Qdrant point ids
     P2-04  provider output sanitizer  (<|im_end|>)
     P2-05  escalation reachability tests
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

**Suggested first cut:** P2-01 → P2-02 → P2-04 → P2-03 → P2-05. Five small
items, no new dependencies, and at the end of them the HITL test case in
`OPERATIONS.md` behaves exactly as documented — with a regression test holding
it there.