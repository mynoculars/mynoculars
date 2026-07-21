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
self-evaluation, and provider fallback.

**Architecture summary** — a *workflow*, not a free-form agent: the graph
topology is fixed at build time; LLMs fill in content (goals, tasks, reports,
critiques) but never choose the control flow. That distinction is the single
most important concept here — dynamic agency is a different pattern, kept out
of this repo on purpose.

## Features

**Capabilities**
- Goal-driven research over an ingested corpus (hybrid dense+keyword search).
- Parallel retrieval fan-out with concurrency-safe state merging.
- Iterative gap-filling bounded by depth and recall targets.
- Bounded self-critique with grounded rewrites.
- Cross-run semantic memory with volatility-aware staleness decay.
- Primary→fallback LLM routing (local Qwen Cogito → Gemini Flash) with a
  quality threshold.
- Human-in-the-loop escalation (`HITL_ENABLED=true`): the graph pauses via
  LangGraph `interrupt()` on four triggers — E1 zero goals, E2 contested
  goals, E3 cannot-converge, E4 critique exhausted — and resumes on
  approve/redirect/abort under the same thread_id. CLI prompts on stdin;
  the API returns `status: interrupted` plus a `/resume` endpoint.
- Fully offline demo mode (`LLM_MODE=stub`) — the entire graph runs with zero
  services and zero API keys.

**Non-goals**
- Production deployment, auth, multi-tenancy, horizontal scaling.
- Dynamic (LLM-decided) control flow / supervisor agents.
- Web search — retrieval is over the local sample corpus.

## Architecture

### Overall architecture

```text
            ┌────────────────────────────────────────────┐
            │              CLI  /  FastAPI               │
            └──────────────────────┬─────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                               LangGraph workflow                             │
└──────┬────────────────────┬────────────────────┬────────────────────┬────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐    ┌──────────────┐     ┌────────────────┐
│FallbackRouter│     │ Corpus tool  │    │SemanticMemory│     │ Checkpointer   │
└──┬────────┬──┘     └──────┬───────┘    └──────┬───────┘     │ + run history  │
   │        │               ▼                   │             └──────┬─────────┘
   ▼        ▼        ┌────────────────┐         ▼                    ▼
┌───────┐ ┌──────┐   │HybridRetriever │    ┌──────────────┐    ┌──────────────┐
│ Qwen  │ │Gemini│   └──┬──────────┬──┘    │Qdrant memory │    │   Postgres   │
│Cogito │ │Flash │      │          │       │  collection  │    └──────────────┘
│(local)│ │(fbk) │      ▼          ▼       └──────────────┘
└───────┘ └──────┘  ┌───────┐ ┌──────────┐
                    │Qdrant │ │OpenSearch│
                    │dense  │ │ BM25     │
                    └───────┘ └──────────┘

```

### Agent workflow

```text
                     [START]
                        │
                        ▼
              ┌──────────────────┐
              │     classify     │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ memory_retrieve  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │   goal_manager   │
              └────────┬─────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │ goals present? (D-21/E1*)  │
        └───────┬─────────────┬──────┘
                │ no          │ yes
                ▼             ▼
        (ERROR report)   ┌──────────────────┐
                │        │  task_expander   │
                │        └────────┬─────────┘
                │                 │
                │                 ▼
                │     ┌────────────────────────────┐
                │     │    D-1: backlog check      │
                │     └──────┬───────────────┬─────┘
                │            │ empty         │ tasks present
                │            ▼               ▼
                │     (EMPTY report)  ┌───────────────────┐
                │            │        │ search_worker ×N  │◄────────┐
                │            │        └────────┬──────────┘         │
                │            │                 ▼                    │
                │            │        ┌────────────────┐            │
                │            │        │     merger     │            │
                │            │        └────────┬───────┘            │
                │◄───────────┘                 │                    │
                │                              ▼                    │
                │                     ┌──────────────────┐         L│
                │                     │ progress_checker │         O│
                │                     └────────┬─────────┘         O│
                │                              ▼                   P│
                │       ┌─────────────────────────────────┐         │
                │       │   convergence (D-14/E2-E3*)     │         ▲
                │       └──────┬──────────────────┬───────┘         │
                │              │ compile          │ expand          │
                │              │                  ▼                 │
                │              │        ┌────────────────┐          │
                │              │        │ gap_generator  │          │
                │              │        └───────┬────────┘          │
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
                │       ┌────────────►│    compiler    │
                │       │             └────────┬───────┘
                │       │                      │
                │       │                      ▼
                │       │             ┌────────────────┐
                └───────┼────────────►│     critic     │
                        │             └────────┬───────┘
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
                                           │     │ memory_writer │
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
  resume under the same thread_id; disabled, they log and continue (E4
  ships the report marked unreviewed — never silently as good).
```

### Request flow (one worker, simplified)

```text
 dispatch           search_worker        corpus tool      HybridRetriever
    │                     │                   │                  │
    │ Send(WorkerPayload) │                   │                  │
    ├────────────────────►│                   │                  │
    │                     │    tool(task)     │                  │
    │                     ├──────────────────►│                  │
    │                     │                   │  search(query)   │
    │                     │                   ├─────────────────►│
    │                     │                   │ fused hits (RRF) │
    │                     │                   │◄─────────────────┤
    │                     │    [Evidence]     │                  │
    │                     │◄──────────────────┤                  │
    │ {evidence, keys,    │                   │                  │
    │  counters} ONLY —   │                   │                  │
    │  D-15 whitelist     │                   │                  │
    │◄────────────────────┤                   │                  │
    │                     │                   │                  │
```

### Storage interactions

```text
┌──────────────────────────┐        ┌──────────────────────────┐
│ ingest_sample_data.py    ├───────►│ OpenSearch: corpus BM25  │
│                          │        └──────────────────────────┘
│                          │        ┌──────────────────────────┐
│                          ├───────►│ Qdrant: corpus dense     │
└──────────────────────────┘        └──────────────────────────┘

┌──────────────────────────┐        ┌──────────────────────────┐
│ memory_retrieve node     │◄───────┤ Qdrant: semantic memory  │
│ memory_writer node       ├───────►│         collection       │
└──────────────────────────┘        └──────────────────────────┘

┌──────────────────────────┐        ┌──────────────────────────┐
│ graph checkpointer (D-8) ├───────►│         Postgres         │
│ run-history rows         ├───────►│                          │
└──────────────────────────┘        └──────────────────────────┘
```

## Design

- **Orchestration** (`orchestration/`): the graph topology and routing live in
  `graph.py`; the worker return contract (`contracts.py`) turns a
  non-deterministic concurrency bug into a deterministic unit-test failure.
- **State** (`state.py`): every field parallel workers write carries a reducer.
  Read the reducer docstrings first — they are the concurrency model.
- **LLM routing** (`llm/`): one OpenAI-compatible client serves both providers;
  fallback policy lives in exactly one place (`router.py`).
- **Retrieval** (`retrieval/`, `storage/`, `tools/`): storage modules are
  policy-free wrappers; fusion math is a pure function; the tool translates
  hits into domain Evidence.
- **Memory** (`memory/`): similarity × volatility-decay reranking; memory items
  re-enter the graph as ordinary evidence, so every downstream rule (coverage,
  contradiction) treats memory and fresh sources identically.
- **Evaluation** (`evaluation/`): the self-scoring signal behind fallback.

## Project Structure

```
agentic-research-agent/
├── src/research_agent/
│   ├── config.py            # all tunables, validated, from .env
│   ├── state.py             # entities, graph state, reducers (read first)
│   ├── logging_setup.py     # JSON-lines structured logging
│   ├── llm/                 # client (real + stub) and fallback router
│   ├── prompts/             # every prompt, one place
│   ├── agents/              # node functions by phase: planning/gathering/compilation
│   ├── orchestration/       # graph wiring + worker contract enforcement
│   ├── retrieval/           # hybrid dense+BM25 with RRF (pure fusion fn)
│   ├── memory/              # semantic memory with decay
│   ├── storage/             # postgres / qdrant / opensearch wrappers
│   ├── tools/               # the corpus-search tool workers invoke
│   ├── evaluation/          # answer quality self-scoring
│   ├── api/server.py        # FastAPI wrapper
│   └── cli.py               # CLI entry + dependency assembly
├── tests/                   # offline: reducers, contracts, routing, e2e
├── scripts/ingest_sample_data.py
├── sample_data/corpus.jsonl # 10 docs, Redis-vs-Memcached theme
├── docker-compose.yml       # optional: postgres + qdrant + opensearch
├── requirements.txt  .env.example  run.bat
```

## Setup

Windows (PowerShell):

```powershell
# 1. Install
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

#hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q5_k_m*.gguf" --local-dir .

# Download Cogito
hf download bartowski/deepcogito_cogito-v1-preview-llama-8B-GGUF `
  --include "deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf" `
  --local-dir .

#Bring up th LLM
PS D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\llama-precompiled> .\llama-server.exe `
>>   -m "D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\models\qwen\cogito\deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf" `  

#Test its Chat window on Browser by asking a general question
http://127.0.0.1:8080/
# 2. Configure
copy .env.example .env        # defaults run fully offline (LLM_MODE=stub)

# 3. First run — no services, no keys, deterministic
$env:PYTHONPATH = "src"
python -m research_agent.cli "Compare Redis and Memcached for session caching"

# Or simply:  run.bat
```

Linux/macOS:

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # defaults run fully offline (LLM_MODE=stub)

# 3. First run — no services, no keys, deterministic
export PYTHONPATH=src
python -m research_agent.cli "Compare Redis and Memcached for session caching"

# 4. Tests
python -m pytest tests/ -q

# 5. Optional: real infrastructure + real retrieval
docker compose up -d
python scripts/ingest_sample_data.py

# 6. Optional: live LLMs — set in .env:
#    LLM_MODE=live, your llama-server URL/model, and a Gemini API key

# 7. Optional: HTTP interface
uvicorn research_agent.api.server:app --reload
```

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
   `MAX_REVISIONS`; exhaustion either interrupts for human review
   (`HITL_ENABLED=true`: approve / redirect with guidance / abort) or logs
   the E4 stub and ships the report marked unreviewed — never silently.
6. **Persist & learn**: a *passed* report's fresh evidence enters semantic
   memory; telemetry aggregates node-recorded counters; a run-history row lands
   in Postgres when available.

## Design Decisions

Decision IDs reference the full architecture document that precedes this build.

| ID | Decision | Why |
|---|---|---|
| D-1 | Empty backlog routes to the compiler | An empty `Send` list would silently halt the graph with no report |
| D-2 | Dedup key sets + replace-on-write backlog | Idempotent dispatch; finite task supply per depth |
| D-3/D-4 | Depth counter + configurable recall target (0.85) | Exact recall ≥ 1.0 degenerates to always-max-depth |
| D-5/D-15 | Reducer-backed worker fields + runtime whitelist | The collision only appears under parallel load — must be impossible, not tested-for |
| D-13 | Fan-out capped at the producers | Overflow is a ranking decision, not a dispatcher accident |
| D-14 | Two termination points: convergence (recall/depth) and dispatch (backlog) | The backlog is stale at convergence time; judge it where it's fresh |
| D-16 | Failed ≠ completed; retry at strictly greater depth | Transient backend errors must not permanently burn a query |
| D-17/D-18 | Quality-gated, contradiction-aware coverage | No recall=1.0 on junk; contested goals drive adjudication automatically |
| D-21 | Zero goals → explicit error report | Diagnosable beats silent |
| D-22 | Bounded critique, grounded rewrites, scoped to faithfulness | One judge per question; no blind retries |
| D-24 | Memory decay = rerank by volatility class, never an age filter | One TTL is wrong for both stable and volatile facts |
| D-23/D-28 | Escalation via `interrupt()`; nothing non-idempotent precedes the interrupt | The node re-executes on resume — history is appended in the resume update only |
| — | Graceful degradation everywhere | First run must succeed on a bare laptop |
| — | Stub LLM mode | Deterministic offline demo + honest tests using real prompts/schemas |

## Limitations

- **MCP deferred**: tools are plain callables behind `tools/`; the calling
  pattern is MCP-shaped so the upgrade touches one module (design D-26).
- **Contradiction detection is minimal**: the machinery (contested goals block
  coverage) is fully wired; the detector only honors explicit markers. A
  semantic detector slots into `merger` without wiring changes.
- **Server-side hybrid fusion deferred**: RRF + decay run in Python here for
  readability; the design (D-27) moves both into Qdrant `FormulaQuery`.
- **Memory simplifications**: no supersession links, no per-item volatility
  classification (items default `semi_stable`), no garbage collection job.
- **Self-evaluated quality is optimistic**: catches broken output, not subtle
  errors; a judge model is the upgrade.
- **Single tool, single worker type**: typed specialists (D-25) arrive with MCP.

## Future Improvements

In rough priority order: MCP tool mediation with typed workers → server-side Qdrant
fusion/decay with payload indexes → semantic contradiction detection → memory
supersession + GC job → judge-model quality scoring → token-usage telemetry
from provider usage metadata.