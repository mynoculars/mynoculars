# OPERATIONS — How To Actually Run This Thing

The operations manual. No diagrams, no theory—just the exact sequence to go 
from zero to running. Copy-paste each command; compare your output to what's 
shown. 

**Mismatch** means stop and debug.

> ## Latest version notes — read these five, then start
>
> 1. **`.env.example` ships `MIN_SIMILARITY=0.35`, which filters nothing on
>    this corpus.** Set it to `0.60` before your first L2 query — Step 2e
>    below makes this a required step, and *Calibrate the Retrieval Floor*
>    explains how to measure it for your own corpus. Skip it and your first
>    run will report `recall: 1.0` on a question the corpus cannot answer.
> 2. **Use a fresh `--thread-id` per QUESTION.** Re-running the same question
>    on one id is safe and useful; reusing that id for a *different* question
>    silently merges the old run's evidence into the new one.
> 3. **`MCP_SERVER_COMMAND=` (blank) is now the recommended value**, not a
>    misconfiguration — it resolves to the agent's own interpreter (D-58).
>    On a checkout *without* D-58 a blank value fails silently as
>    `Connection closed`. **Step 2a — 5. MCP corpus server** covers both.
> 4. **`pip install "mcp>=1.9"` resolves to 2.x today and breaks the MCP
>    servers** (`mcp.server.fastmcp` moved). `requirements.txt` now pins
>    `mcp>=1.9,<2`; if you installed before that pin landed, reinstall.
> 5. **Web search (Phase 4) is off by default** and needs no action. Turning
>    it on is one `.env` line plus one `pip install` — see *Enabling Web
>    Search (Phase 4)*.
>
> Test suite: fully offline, all green — see **Running and Interpreting the
> Test Suite** for how to run it and read the result; a literal count is
> deliberately not repeated here, since it drifts every time a test is
> added and a stale one costs more trust than no number at all (M-4). Every
> historical release note that used to sit here has moved to *Appendix C —
> Version History* at the end of this document; nothing there is needed to
> run the system.

---


## Contents

  - [How to use this document](#how-to-use-this-document)
- **[Part 0 — Before you start](#part-0-before-you-start)**
  - [The One Thing Nobody Told You: There Are THREE Run Levels](#the-one-thing-nobody-told-you-there-are-three-run-levels)
- **[Part 1 — Get it running](#part-1-get-it-running)**
  - [Step 1 — Skeleton (L1): run this first, it needs nothing](#step-1-skeleton-l1-run-this-first-it-needs-nothing)
  - [Step 2 — Real retrieval (L2): the level you actually want to see](#step-2-real-retrieval-l2-the-level-you-actually-want-to-see)
  - [Step 3 — Calibrate the retrieval floor (required before trusting any result)](#step-3-calibrate-the-retrieval-floor-required-before-trusting-any-result)
  - [Step 4 — Full (L3): real report text from a real model](#step-4-full-l3-real-report-text-from-a-real-model)
  - [Step 5 — Verify: service health checklist](#step-5-verify-service-health-checklist)
  - [Step 5b — Verify: the three senses of "test"](#step-5b-verify-the-three-senses-of-test)
- **[Part 2 — Optional capabilities](#part-2-optional-capabilities)**
  - [Running the HTTP API (optional)](#running-the-http-api-optional)
  - [Enabling Web Search (Phase 4, optional)](#enabling-web-search-phase-4-optional)
  - [Enabling Langfuse Observability (Phase 3, optional)](#enabling-langfuse-observability-phase-3-optional)
- **[Part 3 — Tuning `.env`](#part-3-tuning-env)**
- **[Part 4 — Reference & debugging](#part-4-reference-debugging)**
  - [Which Software Runs, And Why (the whole inventory)](#which-software-runs-and-why-the-whole-inventory)
  - [Running and Interpreting the Test Suite](#running-and-interpreting-the-test-suite)
  - [Using Debug Mode](#using-debug-mode)
  - [Understanding and Interpreting the Debug Logs — Node by Node](#understanding-and-interpreting-the-debug-logs-node-by-node)
  - [Debugging a Workflow Execution](#debugging-a-workflow-execution)
  - [Printing the LangGraph Topology](#printing-the-langgraph-topology)
  - [Performing a Dry Run](#performing-a-dry-run)
  - [Guardrails — What To Expect In The Logs](#guardrails-what-to-expect-in-the-logs)
  - [Thread IDs — Usage, Lifecycle, and Reuse Considerations](#thread-ids-usage-lifecycle-and-reuse-considerations)
  - [Writing Your Own Test Corpus](#writing-your-own-test-corpus)
  - [Troubleshooting Common Errors](#troubleshooting-common-errors)
  - [Appendix A — Terms and Acronyms](#appendix-a-terms-and-acronyms)
  - [Appendix B — DBeaver Setup (optional GUI database access)](#appendix-b-dbeaver-setup-optional-gui-database-access)
  - [Appendix C — Version History](#appendix-c-version-history)

## How to use this document

Read Parts 0 and 1 in order, once, executing as you go. After that this is a
reference — jump straight to the section you need.

| Part | What it covers | Read it |
|---|---|---|
| **0 — Before you start** | Prerequisites, conventions, the three run levels | Once, first |
| **1 — Get it running** | L1 → L2 → calibrate → L3 → verify. The linear path | Once, in order |
| **2 — Optional capabilities** | HTTP API, web search, Langfuse | When you want one |
| **3 — Tuning `.env`** | Which settings are safe to change, and in what order | Before your first real workload |
| **4 — Reference & debugging** | Test suite, debug logs, guardrails, thread IDs, troubleshooting | When something surprises you |
| **Appendices** | A: acronyms · B: DBeaver setup · C: version history | As needed |

**The one rule that saves the most time: change ONE thing at a time.** Every
level below builds on the previous one. If L2 misbehaves, the question is
always "what did I change since L1 worked?"

# Part 0 — Before you start

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | 3.11 or 3.12. Check with `python --version` |
| **A virtualenv** | Never install into system Python |
| **~2 GB disk** | Mostly the embedding model (~100 MB) and the local LLM (L3 only, several GB) |
| **Free ports** | `5432` PostgreSQL · `6333` Qdrant · `9200` OpenSearch · `8080` llama-server · `8000` FastAPI (optional) |
| **Outbound internet** | Only for: first-run embedding-model download, Gemini fallback (L3, optional), web search (Phase 4, optional) |

Nothing above is needed for **L1** except Python and a venv. L1 runs with zero
services, zero API keys and zero network.

### Conventions used throughout

**Every command assumes these two things are already true in your shell:**

```powershell
.venv\Scripts\Activate.ps1     # virtualenv active
$env:PYTHONPATH = "src"        # so `research_agent` is importable
```

Both are repeated in the code blocks below rather than assumed, deliberately —
they are cheap to re-read and expensive to forget, and every block should be
copy-pasteable on its own.

**Shell.** Part 4 (Reference & debugging) is PowerShell-only, matching how
this environment is actually driven day to day. Step 1b in Part 1 is the one
exception: it shows both a PowerShell and a Linux/macOS bash block side by
side, since that is the one command sequence someone might genuinely run from
either platform on a fresh clone. Everywhere else, PowerShell is what is
shown and what is meant.

**Terminals.** Some things are long-running and OWN their terminal until you
stop them. This document marks each one. You will need up to four at once at
L3:

| Terminal | Runs | When | Blocks the window? |
|---|---|---|---|
| **T1 — your working shell** | venv + `PYTHONPATH`, `pg_ctl start` (Postgres), the CLI, `pytest`, ingest, `check_services.py` | Always | No — each command returns |
| **T2 — Qdrant** | `qdrant.exe` | L2+ | **Yes**, until Ctrl-C |
| **T3 — OpenSearch** | `opensearch-windows-install.bat` | L2+ | **Yes**, until Ctrl-C |
| **T4 — llama-server** | The local LLM | L3 only | **Yes**, until Ctrl-C |
| **T5 — uvicorn** | The FastAPI server | Only if using the HTTP API | **Yes**, until Ctrl-C |
| **T6 — logs/psql/DBeaver** | Tailing a log, ad-hoc `psql`, or the DBeaver GUI | As needed | No |

**Postgres does not get its own terminal.** `pg_ctl start` launches it as a
detached background process and returns your prompt immediately — run it from
T1, alongside everything else. Every OTHER service below prints "own window"
because the command itself runs in the foreground and never returns until you
stop it.

Every section below that starts a foreground service is marked with a numbered
banner — `Terminal N — Name` — in the order you would actually open them.

**The MCP corpus server (Phase 1––D-30) gets NO terminal at all.** It is not a
standing process — the agent spawns and reaps it per call. See Step 2a §5 for
why running it by hand accomplishes nothing.

**The example query.** This document reuses one question throughout so you can
compare your output to what is shown — it is a genuinely on-topic question
against the sample corpus (`sample_data/corpus.jsonl`, a Redis-vs-Memcached
theme):

```powershell
python -m research_agent.cli "Compare Redis and Memcached for session caching" --debug --thread-id p205.201-check
```

**A second query appears from Step 3 onward:** *"Compare Indian and Chinese
army on battlefield."* That one is deliberately OFF-topic — the corpus has
nothing to say about it — and calibration needs both an off-topic and an
on-topic probe to find the gap between noise and signal. Nowhere before
Step 3 should you see that query; if you do, something has drifted.

Read the flags:

| Flag | Effect | When to use it |
|---|---|---|
| *(none)* | Normal run. Telemetry JSON to stderr at the end | Everyday |
| `--debug` | Adds `node.enter` for every node, plus the full prompt, raw response, provider, tokens, latency of every LLM call and every retrieval engine's raw hits | Whenever something is surprising. Verbose |
| `--thread-id NAME` | Names this run's checkpoint thread | Whenever you want to find the run again, or resume a HITL pause |

**Naming thread IDs.** `p205.201-check` is the convention used here:
`<workstream>.<sequence>-<purpose>`. Any string works, but a naming scheme
matters more than it looks — thread ids accumulate state, and
`test`/`test2`/`asdf` become impossible to attribute a week later.

> ⚠ **Use a fresh `--thread-id` per QUESTION.** Re-running the *same* question
> under one id is safe and useful. Reusing it for a *different* question
> silently merges the old run's evidence into the new one. See **Thread IDs**
> in Part 4.

## The One Thing Nobody Told You: There Are THREE Run Levels

The biggest source of confusion is that two independent switches control what
actually runs. They are NOT the same switch:

| Switch | Controls | Values |
|---|---|---|
| `LLM_MODE` (in `.env`) | Whether the **language model** is real or faked | `stub` / `live` |
| Whether Qdrant + OpenSearch are **running** | Whether **retrieval** works | up / down |

Cross them and you get three meaningful levels:

| Level | LLM_MODE | Services | What you get | Use it to |
|---|---|---|---|---|
| **L1 — Skeleton** | `stub` | down | Graph runs end-to-end, but `evidence_items: 0`, `recall: 0.0`. Empty report. | Prove the plumbing works. First thing you run. |
| **L2 — Real retrieval** | `stub` | **up + ingested** | Graph runs, workers actually retrieve from the corpus, `evidence_items > 0`, `recall > 0`. Report is stub text but telemetry is real. | See the RESEARCH loop do work. This is the one you're missing. |
| **L3 — Full** | `live` | up + ingested | Real Qwen/Gemini writes the report from real retrieved evidence. | Demo / real use. Needs a running model. |

**When you ran it and got `evidence_items: 0` — that was L1.** The machinery
worked; there was just nothing to search. Everything below is about getting you
to L2 (the honest "it researches" level) and then L3.

---

# Part 1 — Get it running

Four steps, in order. Do not skip step 3.

| Step | You get | Needs |
|---|---|---|
| **1** | The graph runs end to end (L1) | Python + venv |
| **2** | Real retrieval over a real corpus (L2) | Qdrant + OpenSearch |
| **3** | Retrieval you can actually trust | 10 minutes |
| **4** | A real model writing the report (L3) | llama-server or a Gemini key |
| **5** | Confidence that all of it works | — |

## Step 1 — Skeleton (L1): run this first, it needs nothing

### Step 1a — Run the unit tests BEFORE anything else

Do this first, every time, before any live run — and especially before you
start blaming a service for a problem.

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/ -q

# expect: all green (see the summary line pytest prints, e.g. "N passed")
```

**All green means the graph, the reducers, the guardrails and every tool seam
are correct.** The suite is fully offline: no services, no API keys, no
network, a few seconds to run. So a failure here is *never* an environment
problem — it is a real regression in the code, and no amount of restarting
Qdrant will fix it.

That is what makes this a useful gate rather than a ritual. If the tests pass
and a live run then misbehaves, you have already eliminated half the search
space: the problem is configuration, data, or a service — not logic.

| Result | What it means | Do next |
|---|---|---|
| All passed, 0 failed | Code is sound | Continue to Step 1b |
| Some failures | A real regression | Stop. Read the failure. Do not proceed to L2 |
| `ModuleNotFoundError: research_agent` | `PYTHONPATH` not set | `$env:PYTHONPATH = "src"` |
| `ModuleNotFoundError: mcp` (9 failures) | You have `mcp` 2.x | `pip install "mcp>=1.9,<2"` — 2.0 moved `mcp.server.fastmcp` |
| Everything errors on import | Wrong venv, or deps not installed | `pip install -r requirements.txt` |

Full detail on reading the suite: **Running and Interpreting the Test Suite**
in Part 4.

### Step 1b — Run the graph with nothing else running

Zero services, zero API keys, zero network. If this works, your Python
environment is correct and the whole graph is wired.

**Nothing to configure first.** L1 talks to no service, so the OpenSearch
and Qdrant settings do not matter yet — those belong to Step 2c, once the
services are actually up. Library log noise is already suppressed in code
(`logging_setup.py::configure_logging` sets both the `opensearch` and
`qdrant_client` loggers to ERROR); earlier revisions of this document asked
you to add those two lines by hand, and that instruction is obsolete.

**Windows (PowerShell):**
```powershell
cd research-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
$env:PYTHONPATH = "src"
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

**Linux/macOS:**
```bash
cd research-agent-dmp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=src
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

> **Alternative, since `pyproject.toml` landed:** `pip install -e ".[all]"`
> installs the package itself instead of just its dependencies — which drops
> the `PYTHONPATH=src` step entirely and gives you a `research-agent`
> console script, so `research-agent "your question"` works from anywhere in
> the venv. `pip install -e .` (no extras) deliberately omits FastAPI,
> uvicorn, MCP and Langfuse; use `[api]`, `[mcp]`, `[langfuse]` or `[all]`
> to add them back. Everything below works identically either way — the
> `PYTHONPATH=src` form is kept throughout this manual because it is what a
> checkout without an install still needs. See README.md's Packaging
> section for the full extras table and the versioning policy.

**What you should see** (this is L1 — note the zeros, they are EXPECTED here):
```json
{
  "intent": "Comparison",
  "goals": 2,
  "iterations": 1,
  "evidence_items": 0,      <-- zero because no corpus is loaded yet
  "recall": 0.0,            <-- zero for the same reason
  "llm_node_calls": 6,
  "llm_provider_calls": 6,  <-- stub mode: 1 attempt per node call, no fallbacks
  "llm_fallback_hops": 0,
  "llm_quality_calls": 0,
  "retrieval_dense_calls": 2,     <-- workers RAN (2 tasks), retrieval attempted
  "retrieval_keyword_calls": 2,
  "retrieval_leg_unavailable": 4, <-- both legs down, both counted, both tasks
  "producer_rejects": 0,
  "search_calls": 2,        <-- workers RAN, they just found nothing
  "search_failures": 0,
  "memory_hits": 0,
  "memory_writes": 0,
  "revision_cycles": 1,
  "critique_passed": true, 
  "planning_error": null,
  "escalations": []

}
```

`search_calls: 2` with `evidence_items: 0` is the signature of L1: the workers
executed, retrieval degraded to empty because the stores are down.
`retrieval_leg_unavailable: 4` is the newer, more direct way to see the same
thing (P2-07) — 2 search calls × 2 unavailable legs each = 4, without having
to infer it from zero evidence. **This is success for L1.**

If Step 1a already passed (and you ran it, right?), you do not need to run the
suite again here — the graph itself is proven sound; the problem, if any,
is now configuration or data, not logic. Everything from here is about
feeding it data.

## Step 2 — Real retrieval (L2): the level you actually want to see

Now we bring up the two search engines and load the sample corpus so the
workers have something to find. **Still `LLM_MODE=stub`** — we are NOT touching
the language model yet. One new variable at a time. That is the whole trick to
not being confused: change ONE thing, observe, then change the next.

### Step 2a — Start the services, in this order

Five things can run alongside the agent. **Only the first three are needed for
L2.** Start them in the order below — each check must pass before you start the
next, because a failure at step 2 is much easier to diagnose than a failure at
step 5 caused by step 2.

| # | Service | Port | Needed for | Terminal |
|---|---|---|---|---|
| 1 | **PostgreSQL** | 5432 | Durable checkpoints + run history | T1 (detaches, no dedicated window) |
| 2 | **Qdrant** | 6333 | Dense (meaning) retrieval + semantic memory | **T2** — own window |
| 3 | **OpenSearch** | 9200 | Keyword (BM25) retrieval | **T3** — own window |
| 4 | **llama-server** | 8080 | L3 only — the local LLM | **T4** — own window |
| 5 | **MCP corpus server** | *(none)* | Optional alternative corpus tool | **None — do not start it yourself** |
| 6 | **uvicorn** | 8000 | Optional — the HTTP API | **T5** — own window |

**Every one of these is optional and degrades gracefully.** The agent never
crashes because a store is down; it logs `qdrant.unavailable` /
`opensearch.unavailable` and carries on with whatever is up. That is the design
(L1 proves it), and it is also the trap: *a run with everything down still
succeeds*, it just finds nothing. Which is why you verify each service rather
than assuming.

#### The one command that checks everything

```powershell
$env:PYTHONPATH = "src"
python scripts/check_services.py
```

Run this after every startup step. It reports `PASS` / `FAIL` / `SKIPPED` per
service and exits non-zero if anything is genuinely down. `SKIPPED` means "you
turned this off in `.env`" — that is a pass, not a problem.

Everything below is what to do when a row says `FAIL`.

---

#### 1. PostgreSQL — durable checkpoints and run history

===============================================================================
**Terminal 1 — PostgreSQL (runs from your working shell, T1)**
===============================================================================

`pg_ctl start` detaches and returns your prompt immediately — this is the one
service that does NOT need its own window. Run it from T1, right before
everything else.

**Start (native Windows):**

```powershell
cd D:\work\softwares\PostgreSQL\pgsql\bin
.\pg_ctl -D D:\work\softwares\PostgreSQL\postgre-data `
         -l D:\work\softwares\PostgreSQL\logs\postgres.log start
```

**Stop:**

```powershell
cd D:\work\softwares\PostgreSQL\pgsql\bin
.\pg_ctl -D D:\work\softwares\PostgreSQL\postgre-data `
         -l D:\work\softwares\PostgreSQL\logs\postgres.log stop
```

**Or via Docker** (the repo ships `docker-compose.yml`; it starts Postgres,
Qdrant and OpenSearch together):

```powershell
docker compose up -d
docker compose ps        # all three should say "running"
```

**`.env`:**

```ini
POSTGRES_DSN=postgresql://agent:agent@localhost:5432/research_agent
```

**Verify:**

```powershell
python scripts/check_services.py          # look for the PostgreSQL row
psql -h localhost -U agent -d research_agent -c "SELECT 1"
```

**Healthy signal in a run's logs:** `checkpointer.postgres_active` or
`checkpointer.pool_active`. Both are good.

**If it fails:**

| Symptom | Cause | Fix |
|---|---|---|
| `checkpointer.memory_fallback` in the logs | Postgres unreachable | Agent still runs, but HITL resume across restarts is lost. Start Postgres |
| `connection refused` | Not started, or wrong port | Check `pg_ctl status`; check the log file named in your start command |
| `database "research_agent" does not exist` | DSN names a DB you never created | Create it **with UTF-8 encoding** — see Appendix B |
| `Deserializing unregistered type ... from checkpoint` | Old checkpoints from an incompatible schema | See Troubleshooting in Part 4 |

The agent works without Postgres. You lose durability, not function.

---

#### 2. Qdrant — dense retrieval and semantic memory

===============================================================================
**Terminal 2 — Qdrant**
===============================================================================

Runs in the foreground and holds this window until you Ctrl-C it. Open a new
PowerShell window for it and leave it running.

```powershell
cd D:\work\softwares\Qdrant
.\qdrant.exe
```

**`.env`:**

```ini
QDRANT_URL=http://localhost:6333
```

**Verify:**

```powershell
python scripts/check_services.py                  # look for the Qdrant row
curl http://localhost:6333/collections            # JSON response = up
```

After ingest you should see the `agent_corpus` collection, and after a run
that stored memory, `agent_semantic_memory`.

**If it fails:**

| Symptom | Cause | Fix |
|---|---|---|
| `qdrant.unavailable` in the logs | Not running, or wrong `QDRANT_URL` | Start it; confirm with the `curl` above |
| Ingest says `Qdrant: SKIPPED (unreachable)` | Same | Start Qdrant, re-run ingest |
| First ingest hangs ~1 min | Downloading the embedding model (~100 MB, one time) | Wait. Subsequent runs are fast |
| `dense: 0` on every query, Qdrant clearly up | Not a Qdrant fault — this is `MIN_SIMILARITY` filtering everything | See Step 3 |

With Qdrant down, retrieval runs **keyword-only** and semantic memory is off.
You will see results, just worse ones — and nothing will shout about it.

---

#### 3. OpenSearch — keyword (BM25) retrieval

===============================================================================
**Terminal 3 — OpenSearch**
===============================================================================

Own window, same as Qdrant. Takes 30–60 s to become ready — do not assume it
is down just because `curl` fails in the first half-minute.

```powershell
cd D:\work\softwares\opensearch-3.6.0
.\opensearch-windows-install.bat
```

**`.env` — this is the fiddly one.** OpenSearch 2.x/3.x ships with the
security plugin ON, so the URL scheme, the SSL flags and the credentials must
agree with how your node is actually running:

```ini

# If your node runs WITH the security plugin (default for a native install):
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=false
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=<your admin password>

# If you disabled the security plugin (what docker-compose.yml does):
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_USE_SSL=false
OPENSEARCH_VERIFY_CERTS=false
```

`http` with `USE_SSL=true`, or `https` with `USE_SSL=false`, both fail. The
scheme in the URL and the flag must match.

**Verify:**

```powershell
python scripts/check_services.py                             # OpenSearch row
curl -k -u admin:<password> https://localhost:9200           # secured node
curl http://localhost:9200                                   # unsecured node
```

A healthy response is JSON naming the cluster and version.

**If it fails:**

| Symptom | Cause | Fix |
|---|---|---|
| `opensearch.unavailable` + `NotSslRecordException` in the OpenSearch log | You sent plain HTTP to a TLS port | Set `OPENSEARCH_URL=https://...` and `OPENSEARCH_USE_SSL=true` |
| `opensearch.unavailable` + `AuthenticationException` | TLS is right, credentials are not | Set `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`. A **different** problem from the one above — see Troubleshooting in Part 4 |
| `worker.failed` with `"reason": "NotFoundError"` on every search | The index does not exist | Run the ingest (Step 2d) |
| Nothing on 9200 for the first minute | Still starting | Wait. It is genuinely slow to boot |

With OpenSearch down, retrieval runs **dense-only**. Same caveat as Qdrant: it
works, quietly worse.

---

#### 4. llama-server — the local LLM (L3 only)

Not needed for L1 or L2. Full detail, including the VRAM tuning that actually
matters, is in **Step 4a**. In short:

===============================================================================
**Terminal 4 — llama-server**
===============================================================================

Runs in the foreground and holds this window — own PowerShell window, exactly
like Qdrant and OpenSearch above.

```powershell
cd D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\llama-precompiled
.\llama-server.exe -m ..\models\qwen\cogito\deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf `
    -ngl 28 -c 1536 --chat-template chatml --port 8080 `
    > ..\logs\llama-server_cogito.log 2>&1
```

**`.env`:**

```ini
LLM_MODE=live
LLM_PRIMARY_BASE_URL=http://127.0.0.1:8080/v1
LLM_PRIMARY_MODEL=deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf
```

**Verify:**

```powershell
python scripts/check_services.py                  # LLM engine row
curl http://127.0.0.1:8080/v1/models              # JSON listing the model = up
```

**If it fails:** see Step 4a for OOM tuning (`-ngl` / `-c`), and
**Tuning the LLM Timeouts** in Part 4. With `LLM_MODE=live` and no engine, the
router falls through to Mistral/Gemini if those keys are set, and fails the run
if they are not.

---

#### 5. MCP corpus server — **do not start this yourself**

This is the one people get wrong, so read the whole subsection.

**There is no MCP service to start.** Unlike every other row in the table, the
MCP corpus server is **not** a standing process. The agent spawns
`scripts/mcp_corpus_server.py` as a child process over stdio when it needs it,
and tears it down when the run ends. Running it manually in a terminal
accomplishes nothing — it will sit waiting for JSON-RPC on a stdin nobody is
writing to.

**`.env`:**

```ini
MCP_ENABLED=true
MCP_SERVER_COMMAND=
MCP_SERVER_ARGS=scripts\mcp_corpus_server.py
MCP_SERVER_ENV_ALLOWLIST=
MCP_TOOL_NAME=search
MCP_QUERY_ARG_NAME=query
MCP_MAX_WORKERS=6
MCP_CALL_TIMEOUT_SECONDS=120
```

**About `MCP_SERVER_COMMAND` being blank — this changed, and the change
matters:**

> **Before D-58, a blank `MCP_SERVER_COMMAND` was a silent failure.** The value
> went straight to the subprocess spawn, an empty command could not be
> executed, and — because the spawn dies before the MCP handshake completes —
> the error surfaced as a generic `Connection closed` rather than anything
> naming the real cause. Commenting the line out to "use the default" produced
> exactly this. There was no default.
>
> **Since D-58, blank is valid and is the RECOMMENDED value.** `config.py::
> resolve_server_command` turns an empty command into `sys.executable`: the
> interpreter already running the agent. That is better than the absolute path
> for three concrete reasons — it needs no configuration, it survives being
> cloned to another machine or drive, and it *guarantees* the server runs in
> the same virtualenv as the agent, which is what makes its imports work.
>
> **If you are on a checkout without D-58, blank still fails.** Check with
> `python scripts/check_services.py`: if the MCP row reports
> `MCP_SERVER_COMMAND is empty -- misconfigured`, your `check_services.py`
> predates the fix, and so does your `assembly.py`. Either apply D-58 or set an
> absolute path.

Both of these are correct configurations:

```ini

# Recommended — portable, always the agent's own interpreter
MCP_SERVER_COMMAND=

# Also fine — explicit, but machine-specific; do not commit it
MCP_SERVER_COMMAND=D:\work\...\research_agent\.venv\Scripts\python.exe
```

`MCP_SERVER_ARGS` is resolved **relative to the repository root**, not your
working directory — so `scripts\mcp_corpus_server.py` works from anywhere.
(Before D-58 it resolved against the launch directory, so it worked only if
you happened to `cd` into the repo first.)

**Verify MCP actually starts** — this is the only way to know, and it spawns a
real server to find out:

```powershell
python scripts/check_services.py
```

A healthy row looks like:

```text
PASS  MCP server   <python.exe> ['...\scripts\mcp_corpus_server.py'] --
                   tool 'search' responded, N content item(s)
                   (spawned fresh for this check -- MCP has no persistent server)
```

**Failure symptoms and what each one means:**

| Row / log | Meaning | Fix |
|---|---|---|
| `SKIPPED -- MCP_ENABLED=false` | Off in `.env`. Correct and working | Nothing. This is the repo default |
| `MCP_SERVER_COMMAND is empty -- misconfigured` | Your checkout predates D-58 | Apply D-58, or set an absolute path to your venv python |
| `McpError: Connection closed` | The subprocess died before the handshake | Wrong interpreter, a `.venv` missing dependencies, or a bad `MCP_SERVER_ARGS` path |
| `FileNotFoundError` | `MCP_SERVER_COMMAND` points at an interpreter that is not there | Blank it out, or fix the path |
| `TimeoutError` after `MCP_CALL_TIMEOUT_SECONDS` | Server started but never answered | Usually a cold embedding-model load on first call. `120` is generous for this reason |
| Nothing at all in the logs | `MCP_ENABLED=false` | Set it `true` if you actually want MCP |

**What MCP does and does not buy you here.** `MCP_ENABLED=true` routes corpus
search through the MCP server *instead of* the in-process tool. It reaches the
**same ingested documents** by a different path — it is a transport
demonstration, not extra recall. Leaving it `false` costs you nothing in answer
quality. (Web search, Phase 4, is a genuinely *different* MCP server that does
add reach — see Part 2.)

---

#### 6. uvicorn — the HTTP API (optional)

Not needed for the CLI at all. Full setup, including the separate-terminal
requirement, is in **Running the HTTP API** in Part 2.

---

#### Putting it together

A normal L2 startup, from cold, using the SAME terminal numbers as the table
above and every banner in this section:

===============================================================================
**Terminal 1 — Working shell**
===============================================================================
```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

cd D:\work\softwares\PostgreSQL\pgsql\bin
.\pg_ctl -D D:\work\softwares\PostgreSQL\postgre-data `
         -l D:\work\softwares\PostgreSQL\logs\postgres.log start
cd D:\work\research_agent          # back to the repo
```

===============================================================================
**Terminal 2 — Qdrant (new window)**
===============================================================================
```powershell
cd D:\work\softwares\Qdrant
.\qdrant.exe
```

===============================================================================
**Terminal 3 — OpenSearch (new window)**
===============================================================================
```powershell
cd D:\work\softwares\opensearch-3.6.0
.\opensearch-windows-install.bat
```

**Back in T1** — verify before you go any further:

```powershell
python scripts/check_services.py
```

T4 (llama-server) and T5 (uvicorn) are not part of L2 — they join at Step 4a
and in **Running the HTTP API**, respectively, each with its own numbered
banner when you get there.

Do not continue to the ingest until `check_services.py` shows PASS or SKIPPED
on every row. Everything after this point assumes the stores are actually up,
and the failure modes get much harder to read if they are not.

### Step 2b — Understand what "ingest" means (30-second concept)

The agent does not search your question against raw documents. It searches
against a **pre-loaded index**. Ingestion is the one-time step that reads
`sample_data/corpus.jsonl` (10 short docs about Redis vs Memcached) and loads
each doc into BOTH engines:

```text
sample_data/corpus.jsonl   (10 documents)
        │
        ├──────────────► OpenSearch   (keyword / BM25 index)
        │                 "find docs containing these WORDS"
        │
        └──────────────► Qdrant        (dense vector index)
                          "find docs with similar MEANING"
                          (embeds each doc with a local model first)
```

At query time the agent hits both, then fuses the two ranked lists (RRF). That's
why it's called *hybrid* retrieval. You must ingest before retrieval returns
anything — an empty index returns empty results, which is your L1 zeros.

### Step 2c — Point `.env` at your services

Open `.env` and confirm these match where your engines actually run. Defaults
assume localhost on standard ports:

```ini
LLM_MODE=stub                                 # still stub — one thing at a time
QDRANT_URL=http://localhost:6333
OPENSEARCH_URL=http://localhost:9200
POSTGRES_DSN=postgresql://agent:agent@localhost:5432/research_agent
CORPUS_INDEX=agent_corpus
MEMORY_COLLECTION=agent_semantic_memory
```

If your native installs use different ports/credentials, change them HERE, not
in code.

> **Fix, not a TODO anymore:** a native OpenSearch install on Windows with the
> security plugin enabled runs its HTTP layer over **TLS by default** — even
> on plain `http://localhost:9200`, the server is doing a TLS handshake and
> will reject a plaintext request outright. If ingest (or any run) logs
> `opensearch.unavailable` with `"reason": "ConnectionError"`, and the
> OpenSearch server log itself shows `NotSslRecordException: not an SSL/TLS
> record`, that's this. The support already exists in the codebase — set:
> ```ini
> OPENSEARCH_USE_SSL=true
> OPENSEARCH_VERIFY_CERTS=false
> ```
> (`VERIFY_CERTS=false` because a default install uses OpenSearch's demo
> self-signed certificate — the client already suppresses the resulting
> warning). Confirm it worked by checking a `--debug` run: `retrieval.hybrid`
> log lines should show `"keyword": 3` (or similar, non-zero) instead of
> `"keyword": 0` — see **Understanding the Debug Logs** below for exactly
> what that field means and where to find it.

### Step 2d — Run the ingest

```bash

# PYTHONPATH still needs to be set (src) in this shell
python scripts/ingest_sample_data.py
```

**What you should see:**
```text
Loaded 10 sample documents
OpenSearch: indexed 10
Qdrant:     embedded 10
```

The Qdrant line downloads a small embedding model on first run (~100 MB, one
time). If either line says `SKIPPED (unreachable)`, that engine isn't running or
`.env` points at the wrong place — fix that before continuing. You can run one
engine only; retrieval will use whichever leg is up.

### Step 2e — REQUIRED: raise the retrieval floor before you trust a result

> ⚠ **Do this now, before your first L2 query. It is not optional and it is
> not tuning-for-later.**
>
> `.env.example` ships `MIN_SIMILARITY=0.35`. **That value filters nothing on
> this repo's own sample corpus** — measured cosine similarity for deliberate
> nonsense queries clusters at 0.40–0.53, i.e. *above* the shipped floor. Left
> at 0.35 the dense leg admits pure noise, and your very first run will report
> `recall: 1.0` on a query the corpus cannot answer.
>
> ```ini
> # in .env
> MIN_SIMILARITY=0.60
> ```
>
> 0.60 is the calibrated starting point for THIS corpus, not a universal
> constant. **Calibrate the Retrieval Floor** later in this document is the
> full procedure and explains how to measure it for your own corpus — but do
> not run L2 on 0.35 and form an opinion about the system's recall first.
>
> Why does the repo ship a value it documents as wrong? Because the right
> value is corpus-dependent and nobody can pick it for you; 0.35 is a
> deliberately inert default that changes no behaviour. That is defensible as
> a library default and indefensible as an operating value — hence this box.

### Step 2f — Run the SAME query, now with data

```bash
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

**What changed — this is L2, and this is the moment it "works".** Same schema
as Step 1b's L1 telemetry (see that step for the full field-by-field
glossary); only the fields that actually moved are shown here:

```json
{
  "evidence_items": 6,             // was 0 at L1 — workers found real evidence
  "recall": 1.0,                   // was 0.0 — goals covered by retrieved facts
  "retrieval_dense_calls": 2,      // both legs actually answered now
  "retrieval_keyword_calls": 2,
  "retrieval_leg_unavailable": 0,  // was 4 — neither leg was down this time
  "memory_writes": 6               // the passed report fed evidence into memory
}
```

`evidence_items` jumped from 0 to a real number. **That is the research loop
doing its job.** The report text is still the stub placeholder (because
`LLM_MODE=stub`), but the *retrieval, coverage, and memory* are all genuinely
working now. Run it a second time and watch `memory_hits` become non-zero — the
agent now remembers the first run.

> ⚠ **Re-running the SAME question under the same `--thread-id` is safe —
> that is what you just did, and it is the point of the exercise. Reusing
> that ID for a DIFFERENT question is not.**
>
> State is accumulated by reducers under the thread id, so a second,
> unrelated query on the same id merges the first run's evidence into the new
> run — silently. You get coverage numbers built partly from evidence about a
> question you are no longer asking, and nothing in the telemetry says so.
>
> **Use a fresh `--thread-id` per question, or omit the flag entirely.**
> Full mechanics: **Thread IDs — Usage, Lifecycle, and Reuse
> Considerations**, below.

## Step 3 — Calibrate the retrieval floor (required before trusting any result)

*Formerly "Fine-Tuning the System". Renamed because the old name undersold
it: this is not optional polish. `.env.example`'s shipped
`MIN_SIMILARITY=0.35` admits pure noise on this corpus, so until you have
worked through this section your `recall` numbers do not mean what they
appear to mean. Step 2e is the one-line version; this is the measurement
procedure behind it.*

Everything above gets the agent *running*. This section is about making it
*correct* for your corpus — which is a different problem, and one the rest
of this manual was silent on.

**Why this needs its own section.** Every retrieval threshold in this
codebase ships with a default anchored to one historical debug trace, not
measured against your data. Two of them (`MIN_SIMILARITY`,
`MIN_EVIDENCE_SCORE`) decide what counts as evidence at all, and a wrong
value does not produce an error — it produces a confident report. The agent
will tell you `recall: 1.0` while answering from documents that have nothing
to do with the question. That failure is invisible unless you go looking, so
this section is the going-looking procedure.

### The one measurement everything else depends on

`MIN_SIMILARITY` is the only threshold you can measure directly, and it is
the one that matters most. The method is two queries and a comparison.

**Step 1 — a query your corpus genuinely answers.** This is your SIGNAL.

```powershell
python -m research_agent.cli "Compare Redis and Memcached for session caching" --debug --thread-id sig-01
findstr /C:"similarity" logs\run-sig-01.txt
```

**Step 2 — a query your corpus cannot possibly answer.** This is your NOISE.
Pick something with zero genuine overlap; the point is that every hit it
returns is by definition irrelevant.

```powershell
python -m research_agent.cli "Compare Indian and Chinese army on battlefield" --debug --thread-id noise-01
findstr /C:"similarity" logs\run-noise-01.txt
```

**Step 3 — compare the two populations.**

```powershell
function Get-Sims($path) {
  (Select-String -Path $path -Pattern '"similarity":\s*([0-9.]+)' -AllMatches).Matches |
    ForEach-Object { [double]$_.Groups[1].Value }
}
"SIGNAL:"; Get-Sims logs\run-sig-01.txt   | Measure-Object -Minimum -Maximum -Average
"NOISE :"; Get-Sims logs\run-noise-01.txt | Measure-Object -Minimum -Maximum -Average
```

**Step 4 — set the floor between them.** Run against this repo's own
`sample_data/corpus.jsonl` with fastembed's default
`BAAI/bge-small-en-v1.5`, the two populations came out cleanly separated:

```text

0 ◄───────────────────────────── MIN_SIMILARITY (0.60) ─────────────────────► 1
├──────────────────────────────────────────────┼──────────────────────────────┤


0.40                              0.527    0.60     0.737          0.843
 │──────────────────────────────────│────────▲──────────│────────────│
              NOISE                        EMPTY            SIGNAL                                         

```

46 off-topic hits spanned **0.402–0.527**. 22 on-topic hits spanned
**0.737 to 0.843**. Nothing landed in the 0.21-wide band between them.

```ini
MIN_SIMILARITY=0.60
```

Two things about that number are worth internalising rather than copying:

- **Unrelated text does not score near zero.** With this embedding model it
  clusters at **0.40–0.53**. Whatever "0.35 similarity" means in your head
  from a different model, it does not mean that here — and `0.35`, the
  shipped code default, sits *below the floor of pure noise*, which is why
  it filters nothing. If you leave it there, every off-topic query still
  returns three confident, irrelevant documents.
- **Do not pick the midpoint.** The midpoint here is 0.632. `0.60` is
  better, because the two errors are not symmetric. Too low lets noise
  through — visible in the report, recoverable. Too high silently drops real
  evidence and is indistinguishable from "the corpus doesn't cover this."
  Leave the wider margin below the signal floor, not above the noise
  ceiling.

**If the two populations OVERLAP**, stop tuning. That is a real finding, not
a threshold problem: your embedding model cannot separate on-topic from
off-topic for this corpus, and no floor will fix it. The answer is a bigger
or better-written corpus, or a different embedding model.

**Watch your lowest signal hit.** In the measurement above it was `Redis
data structures` at **0.737**, noticeably below its neighbours. If a future
on-topic query drops under ~0.65, back the floor off to 0.55. That single
hit is your early-warning marker.

### Verifying the floor took effect

```powershell
python -m research_agent.cli "Compare Indian and Chinese army on battlefield" --debug --thread-id floor-check
```

| What to look for | Expected after the fix | What it means if you don't see it |
|---|---|---|
| `retrieval.below_floor` log lines with `"floor": 0.60` | present, several per run | the setting didn't load — check `.env` parses, and that no shell variable overrides it (`echo $env:MIN_SIMILARITY`) |
| `"dense": 0` on most `retrieval.hybrid` lines | yes, for a genuinely off-topic query | your "off-topic" query overlaps the corpus more than you thought |
| `retrieval.no_backends` WARNING on some queries | yes — both legs returned nothing | — |
| `recall` at depth 1 | **below** 1.0 | see "Why recall still reads 1.0" below |
| `escalations` | may contain `E3` | correct, not a failure — the corpus genuinely can't answer |
| the report itself | says "no evidence retrieved" per goal, rather than answering from model knowledge | the compiler's grounding rule isn't firing; check the prompt reached it |

Then re-run the SIGNAL query and confirm you have not cut into it:

```powershell
python -m research_agent.cli "Compare Redis and Memcached for session caching" --debug --thread-id sig-02
```

Expect `evidence_items` roughly unchanged from your `sig-01` run and
`recall: 1.0`. If this run collapses too, the floor is too high — drop to
0.55 and repeat.

### Why `recall` can still read 1.0 — and which knob actually moves it

This trips people up, so it is worth stating directly. `MIN_EVIDENCE_SCORE`
does **not** gate relevance, and cannot. It is applied to `Evidence.score`,
which is `min(1.0, fused_score × RRF_SQUASH)` — an RRF *rank* artefact, not
a similarity. With `RRF_K=60` and `RRF_SQUASH=30`:

```text
  BOTH legs answered              ONE leg only
    rank 0: 1.000                   rank 0: 0.500   ← exactly, always
    rank 1: 0.984                   rank 1: 0.492
    rank 2: 0.968                   rank 2: 0.484
```

When both legs are healthy, **every hit scores 0.968 or above**, no matter
how irrelevant. `MIN_EVIDENCE_SCORE=0.5` cannot touch it. The rank-0
single-leg value of exactly `0.500` is why the coverage comparison in
`progress_checker_node` is a strict `>` and not `>=`.

So: **`MIN_SIMILARITY` is the relevance gate. `MIN_EVIDENCE_SCORE` is the
single-leg-fallback gate.** They are not two dials on the same thing, and
raising the second will not fix a relevance problem. (Quick-reference version
of this distinction, without the RRF derivation: **[Part 3 — The two filters
people confuse](#the-two-filters-people-confuse)**.)

### The other knobs, and what each one actually moves

Change one at a time and re-run the same query — that is the whole
discipline. Every value below lives in `.env`.

| Setting | Default | Raise it when | Lower it when | Watch |
|---|---|---|---|---|
| `MIN_SIMILARITY` | `0.35` (too low — measure it) | off-topic queries still return evidence | genuinely on-topic goals come back uncovered | `retrieval.below_floor`, `dense` counts |
| `MIN_EVIDENCE_SCORE` | `0.5` | rarely — it is pinned to the RRF single-leg ceiling | never below 0.5 without reading the RRF table above | `recall` when one leg is down |
| `RECALL_TARGET` | `0.85` | you want the loop to work harder before compiling | runs escalate too eagerly | `iterations`, `escalations` |
| `MAX_DEPTH` | `3` | gap-filling is converging but running out of cycles | runs are slow and later cycles add nothing | `iterations` vs `evidence_items` |
| `MAX_FANOUT` | `6` | goals are under-served by too few queries | you are rate-limited or the pool is saturated | `search_calls`, checkpointer pool warnings |
| `MAX_REVISIONS` | `2` | the critic keeps finding real problems | rewrites are cosmetic and cost money | `revision_cycles`, `critique_passed` |
| `LLM_PRIMARY_TIMEOUT_SECONDS` | `120` | the local model times out on large prompts | — | `llm.fallback` with `ReadTimeout` |
| `MEMORY_TOP_K` | see `.env.example` | memory rarely contributes | memory outranks fresh retrieval | `memory_hits` vs `evidence_by_source` |
| `MODEL_KNOWLEDGE_ENABLED` | `true` | never, unless reproducing pre-D-38 corpus-only behavior for comparison | you want retrieval to stop at MCP and escalate to a human instead of the model tier | `model_sourced_items`, `corpus_recall` vs `recall` |
| `MODEL_KNOWLEDGE_SCORE` | `0.6` | rarely — must stay above `MIN_EVIDENCE_SCORE` (so the model tier can cover a goal) and below the ~1.0 a corpus/MCP hit with cross-leg agreement scores (so a real document always wins) | you want the model tier to win coverage races against weak corpus hits, which is rarely what you want | `model_sourced_items`, whether the ladder stops at `corpus` or falls through to `model` in `chain.answered` logs |
| `QUERY_REFORMULATION_ENABLED` | `true` | rarely — the reformulated retry is a single, cheap, deterministic string simplification, not an LLM call | you need to isolate whether a miss is coming from the original query or the reformulated one, while debugging | `chain.answered` log lines with `tier: corpus_reformulated` |
| `MAX_ESCALATIONS` | `2` | HITL runs are hitting the escalation budget and falling through to the compiler before you've had a chance to redirect | escalations are looping without making progress and you want the run to give up sooner | `escalations` list length in telemetry, `escalation.suppressed` log lines |

### Reading the telemetry block as a tuning instrument

Every run prints one. These are the fields that tell you a threshold is
wrong, and what they mean together:

| Pattern | Diagnosis |
|---|---|
| `recall: 1.0` on **every** query, including nonsense ones | `MIN_SIMILARITY` too low — the retriever is feeding irrelevant documents and the coverage rule is accepting them |
| `grounding_ratio: 1.0` but the report is visibly ungrounded | the audit counts evidence **presence**, not relevance. Same root cause; same fix |
| `grounding_ratio` < 1.0 | some goal reached the compiler with nothing at all. The `goals_without_evidence` list names which |
| `iterations: 1` always | the loop never runs — recall is clearing `RECALL_TARGET` on the first pass. Usually the same too-low floor |
| `llm_fallback_hops` high every run | the primary is failing, not the thresholds — check its context length before touching retrieval |
| `llm_quality_calls_failed` == `llm_quality_calls` | the cross-provider judge is erroring and failing open at 1.0; quality gating is effectively off |
| `retrieval_leg_unavailable` > 0 | a store is down; every score you measure this run is single-leg and will read ~0.5 |
| `memory_writes` growing run over run under one `--thread-id` | reducer accumulation, not a threshold — see **Thread IDs** above |

### A worked tuning session, start to finish

```powershell

# 0. Baseline. Note evidence_items, recall, iterations.
python -m research_agent.cli "<a query your corpus answers>" --debug --thread-id tune-00

# 1. Measure. Two populations, as above.
python -m research_agent.cli "<on-topic>"  --debug --thread-id tune-sig
python -m research_agent.cli "<off-topic>" --debug --thread-id tune-noise
findstr /C:"similarity" logs\run-tune-sig.txt
findstr /C:"similarity" logs\run-tune-noise.txt

# 2. Set MIN_SIMILARITY between the populations, closer to the noise side.

# 3. Confirm the off-topic query now fails honestly.
python -m research_agent.cli "<off-topic>" --debug --thread-id tune-01

#    expect: dense 0, recall < 1.0, possibly E3, report says no evidence

# 4. Confirm the on-topic query did NOT collapse.
python -m research_agent.cli "<on-topic>" --debug --thread-id tune-02

#    expect: evidence_items ~unchanged from step 0, recall 1.0

# 5. Only now touch RECALL_TARGET / MAX_DEPTH, one at a time.
```

**Always use a fresh `--thread-id` for each tuning run.** Reusing one merges
the previous run's evidence into the next (see **Thread IDs** above), which
will make a threshold change look like it did nothing.

### One thing to fix before you tune anything

If `llm_fallback_hops` is non-zero on every run and the fallbacks are
`HTTPStatusError` on the `compiler`/`critic` nodes but not on `classify`,
that is a context-window ceiling on the local model, not a retrieval
problem. Raise the context length in your model server (see **Step 3a**
above) before you spend time on thresholds — otherwise you are tuning
retrieval while a different subsystem is quietly failing over every run,
and the two effects are hard to separate in the telemetry.

## Step 4 — Full (L3): real report text from a real model

Only now do we touch the language model. Two providers, primary + fallback.

### Step 4a — Bring up the primary model (local Qwen Cogito)

===============================================================================
**Terminal 4 — llama-server**
===============================================================================

Runs in the foreground and holds this window until you Ctrl-C it. Open a
fourth PowerShell window for it and leave it running; go back to your
working shell (T1) for everything else. Closing this window stops the model,
and every L3 run will then fail over to Gemini or fail outright.

This project expects a local **llama-server** exposing an OpenAI-compatible
endpoint. You already run this for your other work. Start it on the model of
your choice; note the URL and port. Example shape:

```powershell
cd D:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\llama-precompiled

# START TEST: -c 8192
.\llama-server.exe -m ..\models\qwen\cogito\deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf -ngl 999 -c 8192 --chat-template chatml --port 8080 > ..\logs\llama-server_cogito.log 2>&1

# USE THIS — stable on 8 GB VRAM: -c 1536, typically ~28 GPU layers (~4 layers remain in RAM)
.\llama-server.exe -m ..\models\qwen\cogito\deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf -ngl 28 -c 1536 --chat-template chatml --port 8080 > ..\logs\llama-server_cogito.log 2>&1

# DIAGNOSTICS
curl http://127.0.0.1:8080/v1/models

# THIS WILL MOST PROBABLY OOM
.\llama-server.exe -m ..\models\qwen\cogito\deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf -ngl 999 -c 32768 --chat-template chatml --port 8080 > ..\logs\llama-server_cogito.log 2>&1


```

#### Verify loaded model

```powershell
curl http://127.0.0.1:8080/v1/models
```

**Sample output *(formatted)***

```json
{
  "models": [
    {
      "name": "deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf",
      "model": "deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf",
      "modified_at": "", "size": "", "digest": "", "type": "model",
      "description": "", "tags": [""], "capabilities": ["completion"],
      "parameters": "",
      "details": {
        "parent_model": "","format": "gguf","family": "",
        "families": [""], "parameter_size": "","quantization_level": ""
      }
    }
  ],
  "object": "list",
  "data": [
    {
      "id": "deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf",
      "aliases": [], "tags": [], "object": "model",   
      "created": 1784637215, "owned_by": "llamacpp",
      "meta": {
        "vocab_type": 2, "n_vocab": 128256, "n_ctx_train": 131072,
        "n_embd": 4096, "n_params": 8030261312, "size": 5725151488
      }
    }
  ]
}
```

#### Interpreting the output

| Field | Meaning |
|-------|---------|
| `n_params` | **8,030,261,312** parameters (approximately **8.03B**, commonly referred to as an **8B model**). |
| `size` | **5,725,151,488 bytes**, approximately **5.33 GiB** (≈ **5.73 GB** decimal). This is the size of the GGUF model file on disk. |
| `n_ctx_train` | **131,072 tokens**. The maximum context length the model was trained to support. The runtime context (`-c` option) may be configured lower. |
| `n_vocab` | **128,256** vocabulary tokens used by the tokenizer. |
| `n_embd` | **4096** embedding dimensions (hidden size of the transformer). |
| `format` | **GGUF**, the model file format used by `llama.cpp`. |
| `owned_by` | Indicates the model is currently being served by **llama.cpp**. |

> **Note:** The values shown above are specific to the loaded **DeepCogito v1 Preview Llama 8B Q5_K_M** model and will differ for other models.

> **New, see also:** if this local model times out on a tiny prompt like the
> `classify` step, or on a large one like the final report, `LLM_MODE=live`'s
> primary and fallback hops now use **separate, tunable timeouts** rather
> than one shared value — see **Tuning the LLM Timeouts** under
> **Troubleshooting Common Errors** below.





### Step 4b — (optional) a Gemini key for fallback

Get a Google AI Studio API key for `gemini-2.0-flash`. If you skip this, the
agent still runs on the primary alone; it just won't have a fallback when the
local model errors or scores low.

### Step 4c — Flip `.env` to live

```ini
LLM_MODE=live
LLM_PRIMARY_BASE_URL=http://127.0.0.1:8080/v1       # your llama-server
LLM_PRIMARY_MODEL=deepcogito_cogito-v1-preview-llama-8B-Q5_K_M.gguf                       # whatever it reports
LLM_FALLBACK_API_KEY=your-gemini-key-here            # optional
LLM_PRIMARY_TIMEOUT_SECONDS=120                      # local model — raise if it's timing out
LLM_TIMEOUT_SECONDS=90                               # Mistral + Gemini fallback
```

### Step 4d — Run

```bash
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

Now the report body is a real, evidence-grounded answer the model wrote from the
6 retrieved facts. Watch the logs for `llm.fallback` lines — those tell you when
the primary failed or scored below threshold and Gemini took over.

## Step 5 — Verify: service health checklist

Work through this when something isn't behaving. Steps 1–2 are automated and
take seconds; 3–5 are manual and are where the actual diagnosis happens — it
is a diagnostic procedure, not a glance.

### Automated

```bash

# 1. Is my venv active and PYTHONPATH set?
echo $PYTHONPATH          # must print: src

# 2. Are all the live services reachable? (only needed for L2/L3)

#    One command, checks Qdrant + OpenSearch + Postgres + the LLM engine,

#    plus the MCP and web-search subprocesses when those are enabled.

#    Clear PASS/FAIL per service, nonzero exit code if anything's down.

#    Both subprocess rows report SKIPPED when disabled -- that is a pass,

#    not a failure:
python scripts/check_services.py
```

### Manual fallbacks

Only needed when you want to isolate one service, or when
`check_services.py` itself will not start.

```bash

# 3. Check any one service directly
curl http://localhost:6333/collections     # Qdrant: JSON response = up
curl http://localhost:9200                 # OpenSearch: JSON response = up
curl http://127.0.0.1:8080/v1/models       # LLM engine: JSON response = up
psql -h localhost -U agent -d agent -c "SELECT 1"   # Postgres: "1" back = up

# 3. Is the corpus loaded?

#    Re-run ingest; "indexed 10 / embedded 10" = yes, "SKIPPED" = engine down

# 4. Run L1 (stub, no services) — does the graph itself work?
python -m research_agent.cli "test"        # telemetry prints = graph OK

# 5. Read the logs. Every run prints JSON log lines to stderr. The truth is

#    there: qdrant.unavailable, opensearch.unavailable, llm.fallback,

#    checkpointer.memory_fallback, checkpointer.pool_active. Grep for

#    ".unavailable" to see what's down.
```

The logs are the diagnostic. Degradation is silent by design in the *output*,
but every degradation writes a log line. When confused: read stderr.

---

## Step 5b — Verify: the three senses of "test"

**1. Run the unit/integration test suite (proves the logic):**
```bash
export PYTHONPATH=src        # or $env:PYTHONPATH="src" on Windows
python -m pytest tests/ -q   # fully offline, a few seconds — see summary line for count
```
This needs NO services and NO model — it uses the stub and fakes. If these pass,
the graph logic is correct. Run this after any code change. **See "Running and
Interpreting the Test Suite" below for what each test file actually verifies,
and how to run just one of them.**

**2. Test one query by hand (proves retrieval + flow):**
Do L2 above. Change the question to something your corpus can answer:
```bash
python -m research_agent.cli "Which cache persists data to disk?"
python -m research_agent.cli "What handles high concurrency better?"
```
Watch `evidence_items` and `recall` in the telemetry. The 10-doc corpus is all
about Redis vs Memcached, so ask about that. Ask about something absent (e.g.
"MongoDB sharding") and watch `recall` drop — that's the agent honestly
reporting thin coverage, not failing.

**3. Test HITL (proves the human-in-the-loop path):**
```ini

# in .env
HITL_ENABLED=true
```
Then ask something the corpus can't cover, so the agent can't converge and
escalates to you:
```bash
python -m research_agent.cli "Compare Redis vs Cassandra vs DynamoDB at petabyte scale"
```
The CLI will PAUSE and print a review payload, then prompt:
`action [approve/redirect/abort]:`. Type `approve` to ship the partial report,
`redirect` then give guidance to send it back for another pass, or `abort` to
stop with an explicit aborted-report. This is the interrupt/resume machinery
running live.

**If you type anything other than those three words** — including typing your redirect guidance directly at this prompt, which the payload's own `hint` field can make look natural — the CLI now prints `'<input>' is not one of the three actions. Type 'redirect' first -- you will be asked for your guidance text on the NEXT line.` and re-prompts, rather than silently re-prompting with no explanation. Type `redirect`, press enter, and you'll be asked for the guidance text on the following line.

**If it converges instead of escalating**, check `recall` in the printed
telemetry — if it's `≥ 0.85`, the agent believes it found enough evidence
and has no reason to pause. Use `--debug` (see below) to inspect whether
memory items or off-topic corpus hits are being counted as coverage; a
corpus that happens to contain something relevant-sounding to this exact
query can make it converge cleanly instead of escalating. For the full
story of why this specific documented query didn't escalate on an earlier
revision, what was fixed, and one honest caveat about which trigger a real
escalation has actually been confirmed on, see README.md's
[The HITL Investigation](../README.md#the-hitl-investigation) section.

**Note on which trigger you'll actually see:** in practice, real
escalations observed so far have been **E3** (cannot-converge) or **E1**
(zero goals). **E2** (contested goals) is wired correctly end-to-end — the
interrupt/resume machinery works for it exactly like the others — but its
underlying contradiction detector is marker-only and no tool in this build
sets that marker, so E2 has not yet fired in a real run. Don't read a
missing E2 escalation as something broken in your setup.

---


# Part 2 — Optional capabilities

Everything in Part 1 works without any of these. Add them one at a time, and
re-run the example query after each so you can attribute any change.

## Running the HTTP API (optional)

> **TERMINAL:** **this needs its own terminal (T5).** `uvicorn` runs in the
> foreground and holds the terminal until you Ctrl-C it — it does not return
> to a prompt, and that is not a hang. Start it in a dedicated PowerShell
> window, leave it running, and issue your `curl` / `Invoke-RestMethod` calls
> from your working shell (T1).
>
> That terminal also needs the venv active and `$env:PYTHONPATH = "src"` set,
> exactly like your working shell — a fresh window inherits neither.
>
> **`.env` is read once, at startup**, and cached for the life of the process.
> Edit a setting and uvicorn will not notice; Ctrl-C and restart it, or use
> `--reload` during development.

Everything above uses the CLI (`python -m research_agent.cli`). This
codebase ALSO ships a FastAPI app (`api/server.py`) with `/health`,
`/research`, and `/resume` — a genuinely separate, optional way to run
this codebase, not required for L1/L2/L3 or for anything else in this
manual. Skip this section entirely if you only ever use the CLI.

> If `uvicorn` raises `ValueError: too many values to unpack` at startup,
> your checkout predates `AppBundle`'s current shape (`api/server.py` is
> unpacking it as a 4-tuple against a struct that now has more fields) —
> update the code, not your config.

===============================================================================
**Terminal 5 — uvicorn / FastAPI**
===============================================================================

```powershell
d:\work\CONFIDENTAIL\KREUPASANAM\digital-evaluation_ai\research_agent\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = "src"
uvicorn research_agent.api.server:app --reload
```

The command is unchanged, but what it imports is not: `api/server.py` now
takes `build_app_and_settings` from `research_agent.assembly`, not from
`research_agent.cli`. Nothing you type differs; it matters only if you are
reading the startup path or patching it in a test (`tests/unit/
test_api_server.py` patches `research_agent.assembly.build_app_and_settings`
accordingly). If you installed with `pip install -e ".[api]"`, drop the
`PYTHONPATH` line.

**Health-check it** (also covered by `scripts/check_services.py`'s
`--api-url`/`--skip-api` flags — see "Service Health Checklist"
Check" below):
```powershell
curl http://127.0.0.1:8000/health
```
`durable` in the response tells you whether the checkpointer is really
backed by Postgres (`true`) or silently degraded to `MemorySaver()`
(`false`) — same signal the CLI logs as `checkpointer.postgres_active`/
`checkpointer.memory_fallback`, surfaced here for a caller that doesn't
read logs (P2-08).

**Run a query through it:**
```powershell
$body = @{ query = "Compare Redis and Memcached for session caching"; thread_id = "api-test-1" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/research -Method Post -ContentType "application/json" -Body $body
```

A run through the API writes an `agent_runs` row exactly like a CLI run
does (P2-08 — before this, only the CLI did), and closes its checkpointer
connection on FastAPI shutdown, not per-request.

## Enabling Web Search (Phase 4, optional)

Off by default. With `WEB_SEARCH_ENABLED=false` the retrieval ladder is
byte-identical to every pre-Phase-4 run — no second subprocess, no outbound
requests, nothing to configure.

### 1. Install the dependency

```bash
pip install -r requirements-websearch.txt
```

**Into the same virtualenv that runs the agent.** The search server
subprocess is launched with `sys.executable` (see step 2), so if `ddgs` is
missing from *that* environment the subprocess dies before the MCP handshake
and you get an opaque `Connection closed`, never a readable `ImportError`.

### 2. Configure `.env`

```ini
WEB_SEARCH_ENABLED=true
```

That is genuinely the whole minimum. **Leave `WEB_MCP_SERVER_COMMAND`
empty** — that is the recommended setting, not an unset one. Empty means
`sys.executable`, the interpreter already running the agent (D-58), which:

- is correct on every machine with zero configuration — nothing
  machine-specific in the committed `.env`, nothing to change when a
  colleague clones to a different drive, no separate CI override, no
  Windows/POSIX split (`.venv\Scripts\python.exe` vs `.venv/bin/python`);
- guarantees the server runs in the **same virtualenv** as the agent, which
  is what makes step 1 sufficient;
- cannot drift from however you launched the agent.

Set it only if the server must run under a genuinely *different* interpreter.
All three forms work:

| Form | Example | Notes |
|---|---|---|
| *(empty)* | | **Recommended.** `sys.executable`. |
| Repo-relative | `.venv/Scripts/python.exe` | Portable across clones, if the venv lives inside the repo. |
| Absolute | `C:\projects\research_agent\.venv\Scripts\python.exe` | Correct but machine-specific — do not commit it. |
| Bare PATH name | `python3` | Least reliable: whatever PATH resolves to, which may not have `ddgs`. |

**Relative paths resolve against the repository root, not your working
directory.** Worth stating because the MCP SDK does the opposite —
`MCPBridge` does not set `StdioServerParameters.cwd`, so without D-58’s
resolution a relative path would break the moment you ran the CLI from
anywhere but the repo root, or from a service manager, scheduled task, or
IDE that sets its own working directory. Backslashes and forward slashes
both work on both platforms.

### 3. Behind a corporate proxy

```ini
WEB_MCP_SERVER_ENV_ALLOWLIST=HTTPS_PROXY,HTTP_PROXY,NO_PROXY
```

This is the shipped default, and it is the one setting most likely to bite
you. The search server is the only subprocess making outbound internet
calls, and `_build_subprocess_env` forwards **nothing** from the parent
environment (D-30). With an empty allowlist behind a proxy, every search
fails as a timeout with nothing in the log explaining why. Naming a variable
here does not leak it unless it is actually set — on a machine with no proxy
this forwards nothing.

### 4. Verify

```bash
python scripts/check_services.py
```

Look for the `Web search (MCP)` row. This is the **only** live verification
of the search path in this repo — the unit suite is entirely offline by
design and stops at a fake provider. A PASS means a real query reached a
real engine and came back scored:

```
PASS  Web search (MCP)  ... tool 'web_search' returned 5 scored result(s)
                           across 4 domain(s) via WEB_SEARCH_PROVIDER=ddgs
```

| Symptom | Cause |
|---|---|
| `Connection closed` | Almost always `ddgs` missing from the interpreter running the server, or a bad `WEB_MCP_SERVER_ARGS` path. The subprocess died before the handshake. |
| `responded but returned NO results` | Engine throttling this host, or the subprocess has no network route — check the proxy allowlist above. |
| `FileNotFoundError` | `WEB_MCP_SERVER_COMMAND` points at an interpreter that is not there. Try emptying it. |

### What to expect in a run

The tier fires only when corpus, the reformulated retry, and MCP have all
missed. In the logs:

- `chain.answered tier=web` — web satisfied the task.
- `chain.web_gate_rejected` — web returned results that missed the score or
  topical gate, so the ladder continued to the model tier. Watch this one:
  web snippets are short (~150–250 chars) and can fail a two-distinctive-term
  overlap even when genuinely relevant. It is logged rather than tuned so
  there is evidence before anyone changes the threshold.
- `web_search.dropped_unscored_items` — the server returned an item with no
  usable score. No default is safe, so it was dropped and counted.

In telemetry: `web_sourced_items`, `web_source_domains`, `web_sources_listed`,
`web_sources_suppressed`. **Web evidence covers goals but never grounds
them** — a run answered wholly from the web reads `recall 1.0 /
grounded_score 0.0 / corpus_recall 0.0`. That is correct and deliberate, not
a bug: a snippet is retrieval, not curation.

Cited web pages are appended as a deterministic `## Sources` section below
the report (`guardrails/sources.py`). Report prose still carries `[gN]`
markers only — D-40 is unchanged.

### Reading a web-enabled run honestly

The one thing to internalise: **web evidence COVERS a goal but never GROUNDS
one.** `recall` counts a goal answered; `grounded_score` and `corpus_recall`
count it answered *by a real document*. Web is deliberately absent from both
of the latter, because a snippet is retrieval, not curation.

So these three telemetry shapes mean three different things:

| Telemetry | Reading |
|---|---|
| `recall 1.0`, `corpus_recall 1.0`, `web_sourced_items 0` | The corpus answered it. Web never fired. Best case. |
| `recall 1.0`, `corpus_recall 0.0`, `grounded_score 0.0`, `web_sourced_items 12` | The **web** answered it, the corpus could not. Legitimate and attributed — but this is recollection-adjacent, not document-backed. If you expected the corpus to cover this, the corpus is the problem. |
| `recall 1.0`, `corpus_recall 0.0`, `web_sourced_items 0`, `model_sourced_items 24` | The **model** answered it from memory. Web either was off, or fired and missed the gate — check for `chain.web_gate_rejected`. |

`web_source_domains` is the honesty check on the second row: `web_sourced_items 12`
with `web_source_domains 1` is one source repeated twelve times, not twelve
sources agreeing. `WEB_SEARCH_MAX_PER_DOMAIN` (default 2) limits this at
retrieval time; the field lets you confirm it worked.

### Tuning, and what not to touch

| Setting | Default | When to change it |
|---|---|---|
| `WEB_SEARCH_MAX_RESULTS` | 5 | Raise cautiously. More results means more untrusted third-party text in one compile prompt, and more aggressive querying is what triggers throttling. Hard ceiling 25. |
| `WEB_SEARCH_MAX_PER_DOMAIN` | 2 | Lower to 1 for contentious topics where one site dominating the results would read as consensus. `0` disables the cap. |
| `WEB_SEARCH_REGION` | `wt-wt` | Set to `in-en` for India-weighted results, `de-de` for Germany, etc. `wt-wt` is DDGS’s own no-region default. |
| `WEB_MCP_CALL_TIMEOUT_SECONDS` | 45 | Raise if you see timeouts on a slow link. Distinct from `WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS` (20), which bounds a single HTTP request inside the server — both are needed and neither substitutes for the other. |
| `WEB_SEARCH_MIN_SCORE` / `WEB_SEARCH_MAX_SCORE` | 0.60 / 0.75 | **Leave these alone unless you have measured something.** The floor must exceed `MIN_EVIDENCE_SCORE` (0.5) or the whole tier goes inert — it will retrieve, cost real time, and never cover a goal. The ceiling must stay well under the ~1.0 a two-leg corpus hit reaches, or a snippet outranks a real document in the compiler’s context. Startup WARNs `config.web_search_tier_inert` if you get the floor wrong; nothing warns you about the ceiling. |

### Swapping the search provider

DDGS needs no API key, which makes it the right default for a reference
implementation. It is **not** a production dependency: unofficial client,
no SLA, throttling expected. Replacing it is one new module in
`research_agent/websearch/` implementing `SearchProvider`, one entry in
`build_provider`, and one `WEB_SEARCH_PROVIDER` value — nothing in
`agents/`, `orchestration/` or `tools/` changes, because the agent process
never imports the search implementation at all.

## Enabling Langfuse Observability (Phase 3, optional)

`--debug`/`DEBUG_TRACE` above is local and file-based. Langfuse is the
alternative for a hosted, queryable trace UI (Langfuse Cloud, self-hosted, or
enterprise) — the two are independent and can both be on at once.

```ini

# .env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com    # no stray quotes -- see below
LANGFUSE_ENVIRONMENT=development
```

```powershell
pip install langfuse
python -m pytest -q                          # all green, fully offline, unaffected
python -m research_agent.cli "your question" --debug
```

**Confirming it's actually working:** watch the startup log for
`langfuse.client_active`, and confirm there's no
`opentelemetry.exporter...` `WARNING`/`ERROR` line during the run — that
combination means both the host and the credentials are correct. If neither
appears (no error, but no `client_active` either), `LANGFUSE_ENABLED` is
still `false` — check `.env` was actually loaded, not a leftover shell
variable overriding it (same class of trap as the `HITL_ENABLED` one
described in **Running and Interpreting the Test Suite** above).

**Cost tracking is opt-in per provider.** `LANGFUSE_PRICE_PRIMARY_IN_PER_1M`,
`..._OUT_PER_1M`, and the `MISTRAL`/`GEMINI` equivalents default to `0.0` —
leave a provider unpriced and its generations show `$0` (correct for a free
local model, an honest "unknown" for an unpriced cloud one) rather than a
guessed figure.

**Turning it off** is just `LANGFUSE_ENABLED=false` (or deleting the four
credential lines) — nothing else in this document changes; the graph, the
CLI, and the test suite behave identically either way.

---


# Part 3 — Tuning `.env`

`.env` is the only place runtime behaviour is configured. `config.py` validates
every field and refuses to start on an out-of-range value, so a typo is loud —
but a *plausible wrong value* is silent, and that is what this part is about.

### The rules

1. **Change ONE setting at a time, and re-run the example query after each.**
   Two changes at once means you cannot attribute the difference. This is the
   single most common way people waste an afternoon here.
2. **Write it to `.env`, not to your shell.** A shell variable
   (`$env:MIN_SIMILARITY = "0.6"`) overrides `.env` and survives until you close
   the terminal — including into `pytest`, where it silently changes what the
   tests assert. If you set one for an experiment, clear it:
   `Remove-Item Env:\MIN_SIMILARITY`.
3. **Restart whatever is running.** Settings are read once per process and
   cached (`@lru_cache`). A running uvicorn will not notice your edit.
4. **Read the WARNINGs at startup.** `config.py` emits `config.likely_typo`
   for near-miss variable names, `config.inert_coverage_gate` and
   `config.web_search_tier_inert` for values that make a feature run while
   doing nothing. These fire once per process and are easy to scroll past.

### Which settings are safe to touch

| Tier | Settings | Guidance |
|---|---|---|
| **Change freely** | `LLM_MODE`, `DEBUG_TRACE`, `LOG_LEVEL`, `*_ENABLED` flags, `LANGFUSE_*`, timeouts | Behavioural switches. Worst case you turn a feature on or off |
| **Change deliberately, after measuring** | `MIN_SIMILARITY`, `MIN_EVIDENCE_SCORE`, `RECALL_TARGET`, `MAX_DEPTH`, `MAX_FANOUT` | These decide what counts as evidence and when to stop. Wrong values do not error — they quietly change what "recall 1.0" means. See **Calibrate the Retrieval Floor** |
| **Leave alone unless you have data** | `RRF_K`, `RRF_SQUASH`, `WEB_SEARCH_MIN_SCORE`, `WEB_SEARCH_MAX_SCORE`, `GROUNDED_RECALL_TARGET`, `MEMORY_DECAY_*` | These have calibrated relationships with each other. `WEB_SEARCH_MIN_SCORE` must exceed `MIN_EVIDENCE_SCORE` or the whole web tier goes inert; `RRF_SQUASH` maps fused scores onto the same 0–1 scale `MIN_EVIDENCE_SCORE` is compared against |
| **Connection details** | `POSTGRES_DSN`, `QDRANT_URL`, `OPENSEARCH_*`, `*_API_KEY`, `*_SERVER_COMMAND` | Environment-specific. Wrong values fail loudly at `check_services.py` |

### The two filters people confuse

They sound alike and do different jobs at different stages:

| | `MIN_SIMILARITY` | `MIN_EVIDENCE_SCORE` |
|---|---|---|
| **Applies** | To dense candidates, BEFORE fusion | To fused evidence, AFTER retrieval |
| **Units** | Raw cosine similarity from the embedding model | Post-RRF score, squashed to 0–1 |
| **Raise it to** | Stop the dense leg admitting noise | Stop weak evidence counting toward coverage |
| **Symptom of too low** | `recall: 1.0` on nonsense queries | Goals marked covered by near-irrelevant hits |
| **Symptom of too high** | `dense: 0` on genuinely on-topic queries | Real evidence retrieved but `recall` stays 0 |

Both must be right. Raising one to compensate for the other produces a system
that looks calibrated and is not. **For the RRF math behind why
`MIN_EVIDENCE_SCORE` cannot act as a relevance gate** — every hit from two
healthy retrieval legs scores 0.968 or above, regardless of relevance — see
**[Step 3 — Why `recall` can still read 1.0](#why-recall-can-still-read-10-and-which-knob-actually-moves-it)**.

### A safe tuning loop

```powershell

# 0. Baseline — record evidence_items, recall, grounded_score, iterations
python -m research_agent.cli "Compare Indian and Chinese army on battlefield" --thread-id tune.00-baseline

# 1. Change exactly ONE value in .env

# 2. Re-run with a NEW thread id, so no state carries over
python -m research_agent.cli "Compare Indian and Chinese army on battlefield" --thread-id tune.01-minsim

# 3. Compare the telemetry blocks. Different in the way you expected?

#    Yes -> keep it, go to 1 for the next setting.

#    No  -> revert it. A change you cannot explain is a change you do not want.

# 4. Confirm you have not broken the GOOD case: re-run a query the corpus

#    genuinely answers and check recall did not collapse.
python -m research_agent.cli "Compare Redis and Memcached for session caching" --thread-id tune.02-signal
```

Note the fresh `--thread-id` at every step. Reusing one across a tuning session
merges previous runs' evidence into the current one and makes the comparison
meaningless.

**Full measurement procedure**, including how to find the right
`MIN_SIMILARITY` for your own corpus rather than copying this one's:
**Calibrate the Retrieval Floor** in Part 1.

---


# Part 4 — Reference & debugging

Everything above gets the agent running. Everything below is about watching it
run, understanding what a specific execution actually did, and knowing which
knobs are safe to turn without surprising yourself. All commands here are
**PowerShell**, matching how this environment is actually driven day to day.

Nothing in this part needs to be read in order.

## Which Software Runs, And Why (the whole inventory)

| Software | Needed for | Port | If it's down |
|---|---|---|---|
| **Python 3.11+ venv** | Everything | — | Nothing runs |
| **Qdrant** | Dense (meaning) retrieval + semantic memory | 6333 | Dense leg off; memory off; agent still runs |
| **OpenSearch** | Keyword (BM25) retrieval | 9200 | Keyword leg off; agent still runs on dense only |
| **Postgres** | Durable checkpointer + run history | 5432 | Falls back to in-memory checkpointer; HITL still works within a run |
| **llama-server (Qwen)** | Primary LLM (L3 only) | 8080 | L3 fails unless Gemini fallback is set |
| **Gemini API** | Fallback LLM (L3 only) | cloud | No fallback; primary must work |
| **Web search server** (Phase 4, optional) | Retrieval tier 4, when corpus + MCP both miss | — (stdio subprocess, no port) | Ladder falls straight through to the model's own knowledge, exactly as before Phase 4 — `WEB_SEARCH_ENABLED=false` is the default. NOT a standing service: a fresh subprocess is spawned per CLI run and torn down with it |
| **Outbound internet** (Phase 4, optional) | The search server's HTTP calls | 443 | Every web search fails. Behind a proxy this needs `WEB_MCP_SERVER_ENV_ALLOWLIST` — see "Enabling Web Search" |
| **MCP corpus server** (optional) | Corpus search over MCP instead of in-process | — (stdio subprocess) | Nothing — `MCP_ENABLED=false` is the default and retrieval uses the in-process tool. **Never started by hand**: the agent spawns and reaps it per run |
| **uvicorn / FastAPI** (optional) | The HTTP API (`/health`, `/research`, `/resume`) | 8000 | Nothing — the CLI is unaffected. Needs its own terminal when you do want it |
| **Langfuse** (Phase 3, optional) | Hosted trace/cost UI | cloud/self-hosted | Traces just don't appear; agent runs identically — `LANGFUSE_ENABLED=false` is the default |

**Key mental model:** every one of these is *optional* and degrades gracefully.
The agent is designed to run on a bare laptop (L1) and light up more capability
as you add services (L2, L3). You never have to bring up everything at once —
that's the whole point, and it's also how you debug: add one service, observe
the telemetry change, move on.

## Running and Interpreting the Test Suite

The suite is fully offline — no services, no API keys, no
network. It's organized into `tests/unit/` and `tests/integration/`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/ -q
```

```text
tests/unit/                   one file per src/research_agent/ module
tests/integration/            full graph.invoke() runs, offline
                              --------
                              see the summary line pytest -q prints
                              for the exact, current count (M-4: a
                              literal number here goes stale the next
                              time a test is added or removed)
```

The suite is organized by MODULE, mirroring `src/research_agent/`'s own
layout, not by the development phase each test was written in (an earlier
revision of this suite used five files named `test_core.py`/`test_tier2.py`/
`test_tier3.py`/etc. -- `test_tier3.py` alone had grown to 74 tests spanning
at least six unrelated areas by the time it was split up). Two top-level
directories:

- **`tests/unit/`** — one file per source module, named to match:
  `test_state.py`, `test_config.py`, `test_llm_router.py`, `test_llm_client.py`,
  `test_memory_semantic.py`, `test_orchestration_contracts.py`,
  `test_orchestration_graph.py`, `test_agents_task_utils.py`,
  `test_agents_gathering.py`, `test_evaluation_quality.py`,
  `test_storage_qdrant_store.py`, `test_storage_postgres.py`,
  `test_retrieval_hybrid.py`, `test_tools_corpus_search.py`,
  `test_tools_mcp_client.py`, `test_mcp_corpus_server.py`, `test_gc_memory.py`,
  `test_tracing.py`, `test_prompts.py`. **Two new files this session:**
  `test_api_server.py` (this file had ZERO coverage before, which is exactly
  how the `AppBundle` unpack crash shipped unnoticed) and
  `test_agents_compilation.py` (covers `strip_code_fence` and the compiler's
  fence-stripping wiring). **Phase 3 adds `test_langfuse.py`** (33 tests):
  disabled-by-default is genuinely zero-cost, SDK-not-installed degrades
  cleanly, a client that raises on every call never propagates, `config`
  forwarding in `traced_node`, cost calculation (zero-rate, configured-rate,
  unrecognized provider, negative-rate clamping), and the
  `propagate_attributes` session lifecycle including the `shutdown()`
  backstop — all fully offline, same convention as the rest of the suite.
  If you change a function in
  `src/research_agent/foo/bar.py`, `tests/unit/test_foo_bar.py` (or
  `tests/unit/test_bar.py` for a top-level module) is the file that will
  catch a regression first -- that's the whole point of the module-mirrored
  layout.
- **`tests/integration/`** — full `graph.invoke()` runs, offline, on
  StubClient + fake tools: `test_graph_end_to_end.py` (the base e2e run,
  telemetry counters, `evidence_by_source`), `test_hitl_escalation.py`
  (all four HITL triggers E1-E4, D-28's idempotency invariant),
  `test_failure_paths.py` (critique exhaustion, worker failure recording,
  D-16), `test_mcp_routing_end_to_end.py` (P2-14's mixed corpus/MCP
  backlog, real graph run). A unit-level test proves one function's logic
  in isolation; these prove the WHOLE chain still works when everything is
  wired together -- a telemetry field can be correctly bumped in
  `state.counters` and still never reach `result["telemetry"]`, for
  instance, which only a real `graph.invoke()` would catch (see
  `test_graph_end_to_end.py::test_telemetry_surfaces_llm_quality_calls_failed_end_to_end`
  for exactly that shape of regression).

`tests/conftest.py` holds every fixture and stub shared across more than
one file (`settings`, `stub_router`, `fake_tool`, `off_memory`, `graph`,
and `RejectingCriticStub` -- shared between `test_failure_paths.py` and
`test_hitl_escalation.py`, since E4 in the latter is triggered BY the same
critique-exhaustion behavior the former tests directly). It applies
automatically to both `tests/unit/` and `tests/integration/` -- that's
standard pytest conftest.py scoping, not anything special to this repo.

**Run just one file, or just one test:**

```powershell
python -m pytest tests/integration/test_hitl_escalation.py -q
python -m pytest tests/integration/test_hitl_escalation.py::test_e3_interrupts_then_approve_ships_partial -q
python -m pytest tests/ -k "critique" -q       # anything with "critique" in its name
```

**Interpreting a failure.** Every test here runs entirely offline against
`StubClient` and fake tools (see `tests/conftest.py`) — if a test fails, it is
almost always a real regression in the graph's logic, not an environment
problem, precisely *because* nothing here depends on a live service. Treat a
failing test in this suite as a stop-and-look signal, not something to
re-run and hope passes.

**When to run it:** after any change to `src/research_agent/`, before you
touch a live service. It takes under a second; there's no reason to skip it.

**Before running it, if you've been doing manual live testing in the same
shell: clear any HITL env var you set.** `Settings(_env_file=None, ...)`
only skips reading a `.env` FILE — it does nothing to insulate against a
real OS environment variable still sitting in your session. A leftover
`$env:HITL_ENABLED = "true"` from an earlier manual test silently flips
HITL on inside tests that specifically expect it off, and instead of a
clear assertion failure you get a confusing `KeyError` on `state.telemetry`
(an interrupted run never reaches `telemetry_node`, so it's stuck at its
empty default). `tests/conftest.py`'s `settings` fixture now passes
`hitl_enabled=False` explicitly as a hardening measure, so this specific
failure can no longer happen — but clearing the variable first is still
good practice, since other env vars (timeouts, corpus paths) aren't guarded
the same way:

```powershell
$env:PYTHONPATH = "src"
Remove-Item Env:\HITL_ENABLED -ErrorAction SilentlyContinue
$env:PYTHONPATH = "src"
```

## Using Debug Mode

`--debug` (or setting `DEBUG_TRACE` for every run without the flag) turns on
**two independent output streams** at once — knowing which is which is the
whole trick to using this well.

```powershell
$env:DEBUG_TRACE = "true"          # turns it on for every run in this shell

# or, per-run, without setting anything:
python -m research_agent.cli "your question" --debug --thread-id demo1
```

| Stream | Where it goes | What it answers |
|---|---|---|
| `"node.enter"` + other JSON log lines | **stderr** — visible on screen by default | "What ran, in what order, and what happened at each step?" |
| Human-readable execution narrative | `logs\run-<run_id>.txt` **only** — never printed to the console | The full story of the run: graph construction, an execution-plan preview, one section per node (`INPUT`/`DECISION`/`NEXT`), parallel search tasks grouped one-block-per-task, sectioned telemetry, and a final request summary — including the exact prompt/response/hit detail that used to be all this file contained. |

**Capture both streams separately, so you can search each on its own:**

```powershell
python -m research_agent.cli "your question" --debug --thread-id demo1 `
    2> run.log 1> report.txt
```

That splits into: `report.txt` (the final report + telemetry — what you'd
normally see on screen), `run.log` (every structured log line, including
`node.enter`), and `logs\run-demo1.txt` (the narrative, written once at
the very end of the run — see **Understanding the Debug Logs** below for what
to actually look for in each).

**One correction worth internalizing up front:** setting `DEBUG_TRACE=true`
does **not** print prompts and raw responses to your screen. That detail only
ever goes into the narrative file. What *does* print live to your screen (via
stderr) is the shorter JSON breadcrumb trail — provider names, node names,
fallback decisions, timings. If you want the full prompt/response detail,
open the narrative file; it is never going to appear in your terminal directly.

## Understanding and Interpreting the Debug Logs — Node by Node

Every log line is one JSON object. The `"msg"` field names the event;
`"node"` (where present) names which of the 13 nodes it came from. Here is
what a *healthy* line looks like for each node, and what to actually check.

| Node | What to look for | A healthy example |
|---|---|---|
| `classify` | one `llm.call`, `node=classify`; watch `latency_s` | fast (well under a second to a few seconds) |
| `memory_retrieve` | `memory.retrieved`, `count` — 0 on a fresh install, non-zero once you've run a passing query before | `"count": 5` |
| `goal_manager` | one `llm.call`, `node=goal_manager`; if it falls to a fallback, `llm.fallback` fires first with a `reason` | `reason: "JSONDecodeError"` was Cogito's trailing chat-template token before P2-04 — should be rare now |
| `task_expander` | `node.expand`, `"produced": N` — the number of search tasks actually dispatched | `"produced": 6` |
| `search_worker` (×N) | one `node.enter` **per task**, all within milliseconds of each other — that's the parallel fan-out, not a bug | 6 entries within ~10ms |
| — | `retrieval.hybrid` per search, `dense`/`keyword`/`fused` counts | `"dense": 3, "keyword": 3, "fused": 3` once OpenSearch is actually reachable |
| — | `worker.done`, `"items": N` — confirms the task succeeded, not failed | `"items": 3` |
| `merger` | `node.enter` only — no LLM, no store; this is the node that was invisible before node-level logging existed | — |
| `progress_checker` | `node.progress`, `"recall"` and `"depth"` — this is the number that decides whether the loop continues | `"recall": 1.0, "depth": 1` means it converged on the first pass |
| `gap_generator` | only reached if recall was below target; one `llm.call`, `node=gap_generator` | (absent entirely on a converged run — that's normal) |
| `compiler` | one `llm.call` with `mode=text` — the only free-text call in the system; large `prompt_tokens` here is normal (it inlines all gathered evidence). **Watch for `llm.truncated_runaway_generation`** — a WARNING that fires if the model kept generating past its own answer (a fake follow-up conversation, a repeated turn); if you see it often, check your `llama-server`'s stop-token/chat-template config. The compiled report passes through `strip_code_fence()` before being stored, so a fallback provider wrapping its answer in a code fence (or echoing the `<evidence>` fencing tag back literally in a citation) does not leak that into the final report — you should see clean, unfenced Markdown here | `prompt_tokens` in the thousands is expected, not a problem; `llm.truncated_runaway_generation` appearing occasionally is handled gracefully — appearing on nearly every call across every node is a sign of a real server-config issue worth fixing at the source |
| `critic` | `node.critique`, `"passed"` and `"revision"` | `"passed": true, "revision": 1` |
| `memory_writer` | only reached if critique passed; `memory.stored`, `"count"` | `"count"` roughly matching this run's own fresh evidence |
| `telemetry` | `run.telemetry` — the final summary; `llm_node_calls`/`search_calls` are still **node-scoped counts**, but `llm_provider_calls`/`llm_fallback_hops`/`llm_quality_calls`/`retrieval_dense_calls`/`retrieval_keyword_calls`/`retrieval_leg_unavailable` (P2-07) are real boundary-crossing counts now — see below. **New this session:** `escalations`, an array of every `{trigger, action}` pair from `state.escalation_history` — previously written by `human_escalation` and never read anywhere | — |
| `human_escalation` | only with `HITL_ENABLED=true` and a trigger fired; **fires twice** on one escalation (once pausing, once resuming) — expected, not a duplicate | one `escalation_history` entry recorded despite the two log lines, now also visible in the final `telemetry["escalations"]` |

**One thing to expect, not investigate, if you inspect a raw prompt in the
narrative file:** every prompt that inlines retrieved content wraps it in
`<evidence>...</evidence>` tags, with a system-prompt clause telling the
model that span is untrusted data, never instructions. This is deliberate
prompt-injection fencing, not a formatting bug — if you see literal
`<evidence>` tags around corpus/MCP text in `logs\run-<run_id>.txt`,
that's the defense working as intended.

**This used to be the single biggest gotcha in the telemetry block; as of
P2-07 it mostly isn't anymore.** `llm_node_calls` and `search_calls` still
count **node executions**, not actual provider traffic — a node that fell
through the primary to Mistral still counts as one `llm_node_calls`, even
though two real network calls happened. But you no longer have to count log
lines by hand to see the real number: `llm_provider_calls` in the same
telemetry block now IS the real provider-attempt count, and
`llm_fallback_hops` is the real hop count. If you still want to
cross-check against the raw log (or you're debugging something P2-07
doesn't cover, like exact latencies):

```powershell
(Select-String '"msg": "llm.call"' run.log).Count
```

`llm_provider_calls` in telemetry and this log-line count should now agree —
if they don't, that's worth investigating as a real discrepancy, not an
expected gap the way it used to be.


## Debugging a Workflow Execution

The fastest way to actually understand what one run did is `node.enter`,
combined with the trace file. Here is the full recipe:

```powershell
$env:PYTHONPATH = "src"
python -m research_agent.cli "your question" --debug --thread-id debug-run-1 `
    2> run.log 1> report.txt
```

**See every node that fired, in the exact order it ran:**

```powershell
Select-String '"msg": "node.enter"' run.log
```

This includes **`merger`** and **`progress_checker`**, which touch neither an
LLM nor a store — before `--debug` gained per-node logging, these two nodes
were invisible in any trace, and you could only *infer* they ran from the
node before or after them. Now every one of the 13 nodes shows up here,
including those two.

**See exactly what one node's LLM call sent and received:**

```powershell
Get-Content logs\run-debug-run-1.txt
```

Search that file for the node name you care about (e.g. `node=goal_manager`)
to jump straight to its prompt and raw response.

**A practical debugging loop, in order:**

1. `Select-String '"msg": "node.enter"' run.log` — confirm the run reached
   the node you're investigating at all, and see what ran immediately before
   and after it.
2. If that node calls an LLM or a store, open `logs\run-<run_id>.txt` and
   find its entry — see the exact prompt it sent and what came back, now
   inside that node's own `NODE:` section (with `INPUT`/`DECISION`/`NEXT`)
   rather than a flat, separately-numbered entry.
3. `Select-String '"msg": "llm.fallback"' run.log` — see whether the primary
   model failed for that call, and why (`ReadTimeout`, `JSONDecodeError`,
   etc. appear directly in the `reason` field).
4. `Select-String '"msg": "retrieval.hybrid"' run.log` — for retrieval-heavy
   debugging, see the `dense`/`keyword`/`fused` hit counts per query; a
   `"keyword": 0` on every line usually means OpenSearch is down or
   misconfigured (see **Troubleshooting Common Errors** below), not that
   nothing matched.

## Printing the LangGraph Topology

`--print-graph` prints the compiled graph's **static wiring** — the 13 node
names and how they're connected — completely independent of running any
query. This is not telemetry (a summary of what *happened*); it's the shape
of the graph itself, unchanged from run to run.

```powershell

# topology only, no query, exits after printing
python -m research_agent.cli --print-graph

# topology first, then a normal run
python -m research_agent.cli "your question" --print-graph
```

It prints ASCII box-and-line art if the optional `grandalf` package is
installed, or falls back automatically to Mermaid text (no extra install
needed) if it isn't:

```powershell
pip install grandalf     # optional, for nicer terminal output
```

Solid arrows in the Mermaid output are the graph's fixed edges
(`add_edge` in `orchestration/graph.py`); dotted arrows are the conditional
ones (`add_conditional_edges`) — the four decision forks (after goal
composition, after task dispatch, after convergence checking, after
critique) are visible directly in the output as the dotted lines.

## Performing a Dry Run

Two different things in this codebase are legitimately called a "dry run,"
and they answer different questions.

### 1 — A dry run of the whole pipeline (Level 1, stub mode, no services)

This is the safest possible way to check the graph itself is sound before you
touch a live model or a live database — no cost, no network, no risk of
mutating anything:

```powershell
$env:LLM_MODE = "stub"
python -m research_agent.cli "test"
```

If this prints a telemetry block at all (even with `evidence_items: 0`), the
graph, the config, and your Python environment are all fine. See **Step 1 —
Skeleton** above for exactly what a healthy result looks like.

### 2 — A dry run before resetting your stores

`scripts/reset_stores.py` is destructive — it drops Qdrant collections, the
OpenSearch index, and five Postgres tables. Before ever running it for real,
preview exactly what it would touch:

```powershell
$env:PYTHONPATH = "src"
python scripts/reset_stores.py --dry-run
```

This connects to each store (reporting which ones are actually reachable
right now) and prints its plan, but **changes nothing** — confirmed by its own
exit code convention: exit code `1` from `--dry-run` means "at least one
store you asked about is unreachable," which is exactly what you want a
preview to tell you, not an error to panic over.

```powershell

# once you've reviewed the plan and are ready:
python scripts/reset_stores.py --yes

# keep everything the agent has learned, reset only the corpus:
python scripts/reset_stores.py --yes --keep-memory

# one store at a time:
python scripts/reset_stores.py --yes --qdrant
```

> **Before running this for real:** if you have a paused HITL run sitting on
> an `action [approve/redirect/abort]:` prompt in another window, resolve it
> first. Dropping the Postgres checkpoint tables destroys every resumable
> thread, including that one — there is no way to get a paused run back once
> its checkpoint is gone.

## Guardrails — What To Expect In The Logs

*New since D-46. All three log lines below are WARNING-level, purely
observational, and never change routing or abort a run — if you see one,
nothing broke; it is telling you a threshold was crossed.*

**`retrieval.floor_starvation`** — `agents/compilation.py::telemetry_node`,
once per run, only if `retrieval_dropped_by_floor / retrieval_dense_candidates
>= retrieval_floor_warn_ratio` (default `0.8`):

```json
{"msg": "retrieval.floor_starvation", "dropped": 119, "candidates": 132,
 "ratio": 0.902, "floor": 0.55, "warn_ratio": 0.8}
```

This is the run-level aggregate of the per-query `retrieval.below_floor`
lines you already know from **Calibrate the Retrieval Floor** above — it fires
when nearly every dense candidate across the whole run got dropped by
`min_similarity`, not just an individual query. If you see this on most
runs, re-check your `MIN_SIMILARITY` calibration against the procedure
above before assuming the corpus is simply thin.

**`quality.judge_unreliable`** — same node, once per run, only if
`llm_quality_calls_failed / llm_quality_calls >= quality_judge_warn_ratio`
(default `0.5`):

```json
{"msg": "quality.judge_unreliable", "failed": 2, "attempted": 2,
 "ratio": 1.0, "warn_ratio": 0.5}
```

`evaluation/quality.py::score_answer` fails **open** by design (a broken
judge must never reject a good answer) — this WARNING doesn't change that,
it only tells you the judge never actually scored anything on this run
(a `2/2` or `3/3` failure ratio has been the norm rather than the
exception across live testing), so a report reaching `memory_writer`
wasn't necessarily quality-checked on the way there.

**`run.call_budget_high`** — same node, once per run, only if
`llm_provider_calls >= run_call_budget_warn` (default `40`):

```json
{"msg": "run.call_budget_high", "llm_provider_calls": 44,
 "warn_threshold": 40, "revision_cycles": 3, "escalations": 2}
```

Purely observational — see [Guardrails](../README.md#guardrails) in
`README.md` for why this is a WARNING and not a circuit breaker. No run
observed to date has come close to the default threshold (18 calls is the
highest seen); if this fires routinely for you, `revision_cycles` and
`escalations` in the same log line tell you whether the cost is coming
from repeated critique failures or from repeated human redirects, which
is the first thing worth checking before raising the threshold.

**D-59 adds two more.** Both are new this revision, and unlike every
WARNING above, both describe an action actually taken — they are not
observational.

**`node.gaps_skipped_nothing_to_target`** — `agents/gathering.py::
gap_generator_node`, at most once per gather cycle, when the run has no
uncovered goal AND no ungrounded goal left to work on:

```json
{"msg": "node.gaps_skipped_nothing_to_target", "depth": 2}
```

The node returns an empty backlog without calling the LLM, and D-1's
empty-backlog exit routes to the compiler. Seeing this is normal on a run
that genuinely converged. Seeing it at `depth: 1` on a query you expected
to need several cycles usually means your goals were marked covered by
evidence you would not have accepted — check `grounded_score` against
`recall` in the telemetry block before changing anything here.

**`sources.off_topic_dropped`** — `guardrails/sources.py`, once per
compile, when a web page was filed under a goal the report cites but is
not topically about that goal:

```json
{"msg": "sources.off_topic_dropped", "dropped": 9, "kept": 25}
```

A page reaching this check was retrieved, scored, tagged with a real goal
id, and belongs to a goal the compiler cited — everything except being
about the right subject. A high `dropped` count means a gather cycle
drifted off-topic and the drift got as far as the report's attribution
block; it does not mean the prose is wrong. Read it alongside
`corpus_recall` and `grounded_score`.

**Phase 4 (D-57) adds two more, both `chain`/`web_search` scoped.** Neither
changes routing either.

**`chain.web_gate_rejected`** — `tools/retrieval_chain.py`, per task, when
the web tier returned results that missed the score or topical gate:

```json
{"msg": "chain.web_gate_rejected", "task": "g3::pla equipment", "items": 5,
 "reason": "web returned results that missed the score or topical gate"}
```

**This is the one to watch.** Web snippets are short (~150–250 chars), and
`_sufficient` requires a two-distinctive-term overlap with the query — so a
genuinely relevant result can fail the gate and drop the ladder to the model
tier unnecessarily. The gate is applied to web anyway, for consistency (one
rule for what "a tier answered" means) and because a search engine judges
relevance against the query *string* while this gate judges it against the
query's *distinctive terms* — which is what catches a plausible-looking
result set about the wrong subject. It is logged rather than tuned so there
is evidence before anyone changes a threshold, the same measure-first
posture D-54 took. If you see this on most runs alongside
`model_sourced_items` climbing, that is the signal to look at the gate.

**`web_search.dropped_unscored_items`** — `tools/mcp_client.py`, per task,
when the server returned a result with no usable `score` field:

```json
{"msg": "web_search.dropped_unscored_items", "count": 2, "tool": "web_search",
 "reason": "server returned an item with no usable 'score'; no default is safe, so the item was dropped"}
```

Should be rare with the shipped server, which always scores. If it fires
routinely you are almost certainly pointing `WEB_MCP_*` at a different
server whose schema does not match — check `WEB_MCP_TOOL_NAME`. The item is
dropped rather than defaulted because no default is safe: too high silently
defeats the `MIN_EVIDENCE_SCORE` coverage gate (the hardcoded `1.0` this
codebase already shipped once in `make_mcp_tool` and had to fix), too low
burns a compile-prompt slot on evidence that cannot cover anything.

**Also worth knowing about, not a WARNING**: a compiled report may now end
with a `## Sources` section listing web pages — e.g.
`1. [g3] PLA modernization (rand.org) — https://...`. That is
`guardrails/sources.py`, appended deterministically after the citation and
hedging passes, listing only web evidence attached to goals the report
actually cited. The prose above it is untouched and still carries `[gN]`
markers only — D-40 is unchanged. If `web_sourced_items` is high but
`web_sources_listed` is 0, the web tier retrieved material the compiler
never cited; `web_sources_suppressed` counts exactly that.

**Also worth knowing about, not a WARNING**: a compiled report may now
contain a visible `(unverified figure)` marker after a specific number —
e.g. `9.1% (unverified figure)`. This is `guardrails/hedging.py` doing its
job: the number came from the model's own recollection (not retrieved
evidence), paired a specific year with a specific quantity, and the
compiler didn't hedge it itself despite the prompt asking it to. The
figure isn't removed or corrected — only flagged — because guessing at a
*different*, "corrected" number would be worse than being honest that this
one is unverified.

**Guardrails config knobs** (all in `.env`/`config.py`, same tuning
philosophy as `MIN_SIMILARITY`/`MIN_EVIDENCE_SCORE` above — defaults are
starting points, not universal constants):

| Setting | Default | Guards |
|---|---|---|
| `GROUNDED_RECALL_TARGET` | `0.5` | fraction of covered goals that must be topically-grounded before convergence is accepted |
| `RETRIEVAL_FLOOR_WARN_RATIO` | `0.8` | see `retrieval.floor_starvation` above |
| `QUALITY_JUDGE_WARN_RATIO` | `0.5` | see `quality.judge_unreliable` above |
| `RUN_CALL_BUDGET_WARN` | `40` | see `run.call_budget_high` above |
| `LLM_MAX_TOKENS` | `4096` | generation budget sent to every provider on every call — set well above this session's observed legitimate compile completions (1400–1800 tokens); tune down only after checking your own report sizes |

## Thread IDs — Usage, Lifecycle, and Reuse Considerations

`--thread-id` is the identity a run's checkpointed state lives under in
Postgres (or in-memory, if Postgres is unreachable). Understanding its
lifecycle matters more than it looks.

### The default, if you don't set one

```powershell
python -m research_agent.cli "your question"
```

Generates a fresh id automatically (`run-<12 random hex characters>`) — you
never need to think about this for a normal, one-shot run.

### When you SHOULD reuse a thread-id

**Exactly one case:** resuming a paused HITL escalation, in a **separate CLI
invocation**, after the process that started it has already exited. Within
one CLI invocation, the pause/resume loop happens automatically in the same
process — you don't need to do anything with `--thread-id` yourself for
that. Reusing it manually only matters if you closed the terminal and are
coming back later:

```powershell
python -m research_agent.cli "your question" --thread-id my-paused-run

# ... it pauses, you close the terminal ...

# ... later, in a NEW terminal ...
$env:PYTHONPATH = "src"
python -m research_agent.cli "" --thread-id my-paused-run   # resumes where it left off
```

### When you should NOT reuse a thread-id — confirmed with real data

Reusing the same `--thread-id` for a **second, unrelated question** does not
give you a clean slate. Several of `ResearchState`'s fields (`evidence`,
`counters`, `completed_task_keys`, `critique_notes`, `escalation_history`)
are built to *merge* across invocations under the same thread-id, rather than
reset — that's what makes HITL resume possible, but it has a side effect you
need to know about: it applies to *every* invocation under that id, paused
run or not.

**This was confirmed directly, across four consecutive real runs under one
reused thread-id**, all asking the same question:

| Run | `evidence_items` | `memory_writes` | `revision_cycles` |
|---|---|---|---|
| 1 | (baseline) | 54 | 2 |
| 2 | 46 | 54 | 2 |
| 3 | 69 | 108 | 3 |
| 4 | 92 | 180 | 4 |

Every number climbs, run over run — each new run's real work is being added
on top of everything the previous runs under that same id already
accumulated, not replacing it. Harmless here because all four runs asked the
same question. **It would not be harmless** if run 2 had asked something
unrelated — its compiled report could silently include leftover evidence
from run 1's completely different topic, with nothing in the output telling
you that happened. `escalation_history` IS surfaced in telemetry
(`escalations`), so at least check that field after any run on a reused
thread-id — but the accumulation risk itself is not fixed by that
visibility, only made checkable.

**The practical rule:** use a fresh thread-id (or none at all — let it
auto-generate) for every new, independent question. Only ever reuse one
when you are deliberately resuming a run that is genuinely still paused.

**Corrected since the table above was captured (D-38 batch):** the CLI
no longer lets this happen silently. Before invoking, it checks whether
the thread already holds a run (`app.get_state(config)` has a
`raw_query`) and, if so, refuses outright — prints
`[thread-id '<id>' already holds a run for "<query>". Re-invoking it with
a new query ACCUMULATES the old run's evidence and counters instead of
replacing them (D-20). Use a fresh --thread-id, or omit the flag to get a
generated one.]` to stderr, and exits with status `3` without touching
the graph. The four-run table above is still an accurate demonstration of
the underlying reducer-accumulation mechanism, and it is still LIVE for
anyone driving the graph directly — the guard lives in `cli.py`'s `main()`
only, not in `assembly.py` or the graph itself, so the API's `/research`
endpoint has no equivalent check today.

## Writing Your Own Test Corpus

The sample corpus is 10 docs on one topic. To test other questions, replace or
extend `sample_data/corpus.jsonl`. Each line is one JSON document:

```json
{"title": "short title", "topic": "a_tag", "content": "the actual text the agent will retrieve and cite"}
```

Rules that matter:
- One JSON object per line (JSONL — no commas between lines, no wrapping array).
- `content` is what gets embedded and searched — make it a real, self-contained
  fact or paragraph.
- After editing, **re-run the ingest** (Step 2d) so the indexes rebuild. The
  agent never reads the file directly at query time — it reads the indexes.
- Then ask questions your new docs can answer and watch `recall` climb.

To wipe and reload cleanly (Docker): `docker compose down -v && docker compose up -d`
then re-ingest. Native: `scripts/reset_stores.py` (see **Performing a Dry
Run** below) is still the recommended path when you're changing the corpus's
*shape* (adding/removing documents changes which ids exist). For an
unchanged corpus, re-running `ingest_sample_data.py` directly is now safe on
both legs — Qdrant's ingest was fixed to be idempotent (a deterministic
`uuid5(content)` id, same as OpenSearch's long-standing `str(i)` behavior).
**One caveat:** this fix is forward-looking only. If your Qdrant collection
already has duplicate points from ingest runs before this fix landed,
re-running ingest again won't clean those up — you'll need
`reset_stores.py --yes` followed by one fresh ingest to get back to a clean
count.

## Troubleshooting Common Errors

### `opensearch.unavailable` with `NotSslRecordException` in the OpenSearch log

Covered in full under Step 2c above. Short version: your OpenSearch server
is running its HTTP layer over TLS (the default for a security-plugin-enabled
install) while the client is configured for plain HTTP. Fix:

```ini
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=false
```

### `opensearch.unavailable` with `AuthenticationException` (a DIFFERENT problem than the one above)

If you've already fixed the SSL issue above and now see `AuthenticationException`
instead of `ConnectionError`, that's progress — TLS is negotiating correctly
now, but the request is being rejected on credentials. Check the OpenSearch
server's own log for the specific line:

```text
No 'Authorization' header, send 401 and 'WWW-Authenticate Basic'
```

This means the client sent **no credentials at all**, not wrong ones. Check
`.env` for `OPENSEARCH_USERNAME` — the client code only attaches Basic Auth
if that field is non-empty (`if username: kwargs["http_auth"] = (username,
password)` in `storage/opensearch_store.py`); leave it blank and no
`Authorization` header goes out, regardless of what `OPENSEARCH_PASSWORD` is
set to. Fix:

```ini
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=<your actual OpenSearch admin password>
```

**One thing worth knowing if you don't remember setting a password
deliberately:** OpenSearch 2.12+ removed the old default `admin:admin`
credential for security reasons. A native install requires an initial admin
password to have been set explicitly — typically via an
`OPENSEARCH_INITIAL_ADMIN_PASSWORD` environment variable at first startup,
or configured directly in `config/opensearch-security/internal_users.yml`
under your OpenSearch install directory. Check there if `admin`/`admin`
doesn't work.

### `TypeError: IndicesClient.exists() takes 1 positional argument but 2 positional arguments ... were given` during ingest

This is a client-library version issue, not a config problem. `opensearch-py`
3.x made `indices.exists`/`.create`, `.index`, and `indices.refresh` require
the index/document name as a **keyword** argument (`index=...`), where older
versions of the library also accepted it positionally. If your installed
`opensearch-py` is 3.x and you see this `TypeError` from
`scripts/ingest_sample_data.py`, this has already been fixed in
`storage/opensearch_store.py` (all four call sites now use `index=` — the
`search()` call already did, which is why searches never showed this
error, only ingest did). If you're on an older checkout without that fix,
either update `storage/opensearch_store.py` to match, or pin
`opensearch-py<3` in `requirements.txt` as a stopgap. Confirm the fix
worked: `python scripts/ingest_sample_data.py` should print
`OpenSearch: indexed 10` with no traceback.

### `worker.failed` with `"reason": "NotFoundError"` on every single search

This means the Qdrant **collection itself doesn't exist** right now —
Qdrant is reachable, but `agent_corpus` isn't there to query. Almost always
caused by running `scripts/reset_stores.py --yes` (or its `.bat` equivalent)
and not following it with a re-ingest:

```powershell
$env:PYTHONPATH = "src"
python scripts/reset_stores.py --yes      # this DROPS the collection
python scripts/ingest_sample_data.py      # this must run again afterward
```

Symptom in the logs is unambiguous: every `search_worker` fails with the
same `NotFoundError`, `search_calls` in the final telemetry is `0`, and
`recall` is `0.0` — not because retrieval found nothing relevant, but
because it found nothing to search at all. This is easy to mistake for
retrieval genuinely returning empty results (the L1 "skeleton" zeros); the
tell is `search_failures` being non-zero rather than `evidence_items` simply
being `0` with no failures recorded.

### `"Deserializing unregistered type research_agent.state.Goal from checkpoint"` (WARNING, not an error)

You'll see one of these per custom type (`Goal`, `Volatility`, `Evidence`,
`SearchTask`) the first time each is checkpointed to Postgres in a given
process. This is LangGraph's serializer telling you it reconstructed one of
this project's own Pydantic models from checkpoint data without that type
being on an explicit allowlist — currently harmless (it still works), but
the message is a forward-looking one: a future LangGraph version (or setting
`LANGGRAPH_STRICT_MSGPACK=true` yourself right now) would **block** this
entirely, which would break `--thread-id` resume and HITL pause/resume
outright, since the checkpointer would no longer be able to reconstruct your
own state. Not yet fixed in this codebase; safe to ignore for now, but not
indefinitely — do not set `LANGGRAPH_STRICT_MSGPACK=true` until this project
explicitly allowlists its own four types.

### HITL not pausing even with `HITL_ENABLED=true` set

Check the exact variable name first. The setting is `HITL_ENABLED`, not
`HITL` — and because config parsing silently ignores unrecognized keys, a
typo here produces **no error, no warning, nothing** — HITL simply stays off:

```powershell
$env:HITL_ENABLED = "true"     # correct
$env:HITL = "true"             # silently does NOTHING — common typo
```

### `recall: 1.0` on a query the corpus should not be able to answer

Your retrieval floor is too low — `MIN_SIMILARITY=0.35` (the shipped default)
admits noise on this corpus. This is not a troubleshooting-time fix; it is a
setup-time one. See
**[Step 3 — Calibrate the retrieval floor](#step-3-calibrate-the-retrieval-floor-required-before-trusting-any-result)**
for the full measurement procedure, and
**[Part 3 — Tuning `.env`](#part-3-tuning-env)** for the two-filter
distinction (`MIN_SIMILARITY` vs `MIN_EVIDENCE_SCORE`) that trips people up
here — raising `MIN_EVIDENCE_SCORE` will NOT fix this; it filters a
different stage entirely.

### Tuning the LLM Timeouts

The local primary model and the two cloud fallbacks (Mistral, Gemini) use
**separate** timeout settings, not one shared value:

```ini
LLM_PRIMARY_TIMEOUT_SECONDS=120     # local Cogito — default 120s
LLM_TIMEOUT_SECONDS=90              # Mistral + Gemini — default 90s
```

Raise `LLM_PRIMARY_TIMEOUT_SECONDS` if the local model is timing out on
genuinely large prompts (the `compiler` node's prompt, which inlines all
gathered evidence, is usually the biggest one — check `prompt_tokens` in a
`--debug` trace for that node). **If a small, early prompt (e.g. `classify`)
times out too, at the exact configured limit, every single run** — that is
not a "needs more time" problem; it's a sign the local server itself may be
spending that time on a one-time model load rather than on your actual
question. Worth checking the model server's own startup behavior directly,
independent of this setting.

```powershell
$env:LLM_PRIMARY_TIMEOUT_SECONDS = "150"
python -m research_agent.cli "your question" --debug
```

### Local model generates fake follow-up turns after its answer

Watch for `llm.truncated_runaway_generation` in the log (see the `compiler`
row of **Understanding and Interpreting the Debug Logs** above — this is
the free-text path's equivalent of the JSON path's sentinel-stripping,
`_truncate_at_sentinel()` in `llm/client.py`). It fires when the local
model keeps generating past its actual answer — a fabricated `system`
turn, the prompt echoed back, a duplicate report — and the extra content is
cut before it can reach `final_report`. Occasional occurrences are handled
gracefully and not worth chasing; if you see it on nearly every call across
every node, that's a real `llama-server` configuration issue, not something
this codebase should keep silently absorbing — check the stop-token and
`--chat-template` settings on your model server first.

### `Connection pool is full, discarding connection: localhost` (WARNING, fixed)

Seen under real concurrent corpus search (`MAX_FANOUT` search_workers all
hitting OpenSearch at once). Harmless before the fix below — the request
still succeeded, urllib3 just couldn't cache the connection for reuse
(pool size 1), forcing a fresh TCP+TLS handshake instead. Fixed:
`storage/opensearch_store.py` now passes `pool_maxsize=20` to the
`OpenSearch(...)` client, covering `MAX_FANOUT`'s default (6) with
headroom. If you still see this warning after updating, you've likely
raised `MAX_FANOUT` above 20 — bump `pool_maxsize` to match.

### MCP tool (`MCP_ENABLED=true`) was slow / timed out under concurrent load — FIXED

Root cause (confirmed by reading `mcp/server/fastmcp/utilities/
func_metadata.py::call_fn_with_arg_validation`): a synchronous tool
handler is called directly on FastMCP's single event loop, with no
thread offload (`fn(**args)`, not `asyncio.to_thread(fn, **args)`). A
synchronous `scripts/mcp_corpus_server.py::search()` doing real, blocking
Qdrant/OpenSearch I/O (~13s+ per call) therefore blocked the ENTIRE
server for one in-flight request — `MAX_FANOUT` concurrent
search_worker calls fully serialized instead of running in parallel.
Confirmed live: 6 concurrent searches that complete in 14.4s total
called directly (no MCP) took 100+ seconds through MCP, all still
eventually succeeding (never a correctness bug — see the real
end-to-end trace evidence in `README.md`'s Limitations section, item 29 —
filed there as a fixed item, not an open one, despite the historical
"still broken" framing of that part of the list).

**Fixed**: `mcp_corpus_server.py::search` is now `async def`, and the
blocking `hits_for_query` call is offloaded to a dedicated
`ThreadPoolExecutor` (sized by the new `MCP_MAX_WORKERS` setting,
default 6) via `loop.run_in_executor(...)`, so FastMCP's event loop stays
free to service other requests while one call is mid-flight.
`MCP_ENABLED=false` (the default) was always unaffected. If you still
see slow first calls: that's the one-time ~13s cold start building the
real `QdrantStore`/`OpenSearchStore`/embedding model on the server's
first request (see `mcp_corpus_server.py::_get_corpus_tool`), not the
concurrency issue above — raise `MCP_CALL_TIMEOUT_SECONDS` (e.g. 120+)
for that instead.

**A second, separate stall, found after the above fix was already in
place**: the FIRST `search()` call in a real deployment still stalled
for ~120s, before any network call even started — nowhere near the
~13s cold-start figure above. Root cause: `qdrant_client` was being
imported lazily, for the first time in that process, on a
`_search_executor` worker thread, while the main thread's asyncio
Proactor loop was already running real overlapped I/O on the stdio
pipes — that specific combination (a live Proactor loop doing real I/O,
plus a first-time native-extension import happening on another thread)
reproduced reliably on at least one Windows deployment machine,
independent of any configured timeout. Two other plausible explanations
(antivirus/EDR file-hash scanning of the DLLs; `CREATE_NO_WINDOW` plus a
stripped-down subprocess environment) were tested directly, in
isolation, and ruled out — it's specifically the thread/event-loop
combination. **Fixed** by importing `qdrant_client` and `opensearchpy`
eagerly, on the main thread, at module load time in
`mcp_corpus_server.py`, before `mcp.run()` starts the event loop — see
that file's own module docstring ("First-import gotcha") for the full
account. If you ever see a mysterious ~120s stall on the very first
`search()` call (and nowhere else), check that those two eager imports
are still present before looking anywhere else.

### `AppBundle` unpack crash on API startup — FIXED (post-Tier-3 session)

If `uvicorn research_agent.api.server:app` raises `ValueError` at import
(something like `too many values to unpack (expected 4)`), you're on a
checkout from before this fix. `AppBundle` grew a 5th field (`mcp_bridge`,
P2-13) while `api/server.py` still destructured it as a 4-tuple. Fixed to
named-field access (`bundle.app`, `bundle.settings`, etc.) — update the
code rather than trying to work around it in config.

### MCP evidence always satisfies coverage — FIXED (post-Tier-3 session)

If MCP-routed tasks (`MCP_ENABLED=true`, a task with `tool_hint="mcp"`)
always mark their goal as covered regardless of actual relevance, you're on
a checkout from before this fix — `tools/mcp_client.py` used to hardcode
`score=1.0` on every MCP-sourced Evidence item, which cleared
`MIN_EVIDENCE_SCORE` unconditionally. Fixed: MCP evidence is now scored at
`settings.min_evidence_score` (never higher), so it behaves like corpus
evidence with respect to the coverage gate.

### DNS failure or a request going to an unexpected host with Langfuse enabled

If `LANGFUSE_HOST` resolves to something that doesn't look like what you
typed — check for a **stray trailing quote** in `.env`, e.g.
`LANGFUSE_HOST=https://cloud.langfuse.com"` (trailing `"`, no leading one).
Standard `.env` parsing only strips *matched* leading+trailing quote pairs,
so an unmatched one becomes a literal character in the value, gets
percent-encoded into the request URL, and produces a DNS failure that
doesn't obviously point back to the `.env` file. Confirmed live:

```python
Settings(_env_file='.env').langfuse_host == 'https://cloud.langfuse.com"'  # broken
```

Not yet patched with a defensive validator — check every `LANGFUSE_*` value
for balanced quotes by hand if you hit this. Every other quoted value in the
file with properly matched quotes parses cleanly, so this is specific to
values with an odd number of `"` characters.

## Appendix A — Terms and Acronyms

| Term | Expansion | What it means here |
|---|---|---|
| **L1 / L2 / L3** | Run levels | Skeleton (no services) / real retrieval / real LLM. See *There Are THREE Run Levels* |
| **HITL** | Human-In-The-Loop | The graph pauses for human approve/redirect/abort at four escalation points |
| **E1–E4** | Escalation triggers | E1 zero goals, E2 contested goals, E3 cannot-converge, E4 critique exhausted — the four points HITL can interrupt |
| **MCP** | Model Context Protocol | The tool protocol used here for two stdio subprocess servers: the corpus server (P2-13) and the web-search server (Phase 4) |
| **RRF** | Reciprocal Rank Fusion | How the dense (Qdrant) and keyword (OpenSearch) result lists are merged into one ranking |
| **`RRF_K`** | RRF constant | Damping term in the RRF formula; higher values flatten the difference between ranks |
| **`RRF_SQUASH`** | RRF score scaling | Maps a fused RRF score into 0–1 so it is comparable with `MIN_EVIDENCE_SCORE` |
| **`D-nn`** | Decision record | A numbered entry in `DECISIONS.md` — the *why*, not the *how*. `D-57` = web search, `D-58` = server path resolution |
| **`P2-nn`, `P205`** | Patch series | Historical fix batches. Meaningful for archaeology only — see Appendix C |
| **`G1–G7`** | Guardrails | Deterministic post-processing checks in `research_agent/guardrails/`. See README's *Guardrails* |
| **Tier 1–5** | Retrieval ladder | corpus → reformulated corpus → MCP → web → model recollection. Stops at the first tier clearing the bar |
| **Floor / gate** | Two different filters | `MIN_SIMILARITY` filters dense candidates BEFORE fusion; `MIN_EVIDENCE_SCORE` filters fused evidence AFTER. Both must be right |

## Appendix B — DBeaver Setup (optional GUI database access)

*Moved out of the Step 2 flow: this is a one-time GUI convenience, not part
of getting the agent running. Nothing in Part 1 requires it — the agent talks
to PostgreSQL directly, and `scripts/check_services.py` verifies the
connection without a GUI. Set it up if you like inspecting checkpoints and
`agent_runs` rows by hand.*

> **Important:** Always create the `research_agent` database with **UTF-8** encoding. PostgreSQL on Windows may default to **WIN1252**, which cannot store emoji generated by the LLM.

1. **Connect as `postgres`** (or another admin user).

2. **Create the application user:**
```sql
CREATE USER agent WITH PASSWORD 'agent';
```

3. **Create the database (UTF-8):**
```sql
CREATE DATABASE research_agent
WITH OWNER agent
ENCODING 'UTF8'
TEMPLATE template0;
```

> If `research_agent` already exists with the wrong encoding:

```sql
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'research_agent'
  AND pid <> pg_backend_pid();

DROP DATABASE IF EXISTS research_agent;

CREATE DATABASE research_agent
WITH OWNER agent
ENCODING 'UTF8'
TEMPLATE template0;
```

4. **Grant privileges:**
```sql
GRANT ALL PRIVILEGES ON DATABASE research_agent TO agent;
GRANT ALL ON SCHEMA public TO agent;
```

5. **Connect to `research_agent` as `agent`.**

6. **Verify:**
```sql
SELECT current_database(), current_user;
```

Expected:

```text
current_database | research_agent
current_user     | agent
```

Your `.env` is now ready:

```text
POSTGRES_DSN=postgresql://agent:agent@localhost:5432/research_agent
```

**One table worth knowing about once you're in DBeaver**: alongside
LangGraph's own `checkpoints`/`checkpoint_blobs`/`checkpoint_writes`
tables, this codebase's own `record_run()` creates and writes an
`agent_runs` table — one row per completed run (CLI or API), with
`thread_id`, `query`, `recall`, `telemetry` (JSONB), and a timestamp.
Nothing in this codebase reads it back; it exists purely for you and
DBeaver to inspect post-hoc run history. See README.md's Storage
Contracts section for the full column list. You may see either
`checkpointer.postgres_active` or `checkpointer.pool_active` at startup —
both are healthy signs.

*(The native Qdrant and OpenSearch startup commands that used to sit here now live in **Step 2a — Start the services, in this order**, alongside their verification steps and failure modes.)*

## Appendix C — Version History

Release notes accumulated during development, newest concern last. **None of
this is needed to operate the system** — it is kept for archaeology: to explain
why a setting has the value it does, and what a given fix was responding to.
Historical test counts (57 → 135 → 157 → 294 → 341 → 344 → 348 → 476 → 492) appear
below as they were written, at the time each note was made. As of this pass
none of them is current — see **Running and Interpreting the Test Suite**
above, which reports the count by running the suite rather than stating a
literal (M-4: six different hardcoded counts across this document and the
README had drifted out of sync with each other and with the actual suite).

**Every note below refers to the document structure AT THE TIME it was
written**, including section names that have since been renamed or moved
(e.g. "Development & Debugging Workflows" is now Part 4; "Fine-Tuning the
System" is now Step 3). Do not use a section name in these notes to navigate
the current document — use the Contents at the top. Claims a note makes about
"the rest of this document" being unchanged describe that PAST state, not the
current one.


> **CORRECTED THIS PASS** (post-Tier-3 session, 4 further live-tested
> patches applied on top of everything below): test suite reached
> **157** at this point (was 57, then 135 across Tier 2/3 — see the
> **Phase 3 note** below for the current, higher count). CLI exit code is
> no longer always 0 — `main()` now returns 2 on `GraphRecursionError`, 1
> when a run ends with no telemetry. Postgres checkpointer is now pooled
> — you may see a new `checkpointer.pool_active` log line alongside or
> instead of `checkpointer.postgres_active`; both are healthy signs. MCP
> evidence no longer defaults to a fabricated `score=1.0` — expect real
> evidence-gate behavior on MCP-routed tasks now, same as corpus-routed
> ones. `api/server.py` previously could not even import (`AppBundle`
> unpack crash) — if you were following "Running the API" on an
> un-patched checkout, `uvicorn` would have raised `ValueError` at
> startup; fixed. Telemetry examples throughout this document use the
> current field names (`llm_node_calls`/`llm_provider_calls`/etc. — see
> **Understanding and Interpreting the Debug Logs** below), plus an
> `escalations` field surfacing `escalation_history` in every run's
> telemetry, previously written and never read anywhere. Qdrant ingest is
> now genuinely idempotent by default — see **Writing Your Own Test
> Corpus** below for the one caveat that still applies to a collection
> that already has stale duplicates in it from before this fix.
> Everything else in this document — the L1/L2/L3 ladder, native Windows
> service startup, DBeaver setup, ingest steps, Human-In-The-Loop (HITL)
> testing, every
> troubleshooting entry, all of **Development & Debugging Workflows** —
> is unchanged and still accurate as written. All PowerShell-targeted
> command examples remain PowerShell, per how this environment actually
> runs.

> **Phase 3/D-38–D-46 note:** test suite is now **294** (157 + 33 Phase 3 + 104 new
> `test_langfuse.py` tests, still fully offline). Optional Langfuse
> tracing is documented in **Observability — Langfuse (Phase 3)** below;
> it is off by default (`LANGFUSE_ENABLED=false`) and every step above and
> below this note works identically whether it's on or off.

> **Guardrails, Phases 1–3 note:** test suite is now **341** (294 + 47 new
> guardrails regression tests, still fully offline). A new
> `research_agent/guardrails/` package (`citations.py`, `fencing.py`,
> `hedging.py`) plus new checks in `agents/gathering.py`,
> `agents/task_utils.py`, and `agents/compilation.py::telemetry_node` add:
> a grounded-convergence check (route_convergence won't accept full
> convergence on evidence that scores well but isn't topically about the
> goal it's credited against); a deterministic `(unverified figure)`
> marker inserted into the compiled report for a model-tier claim that
> pairs a specific year with a specific quantity and wasn't hedged by the
> compiler; a rejection for `gap_generator` tasks naming a goal id that
> doesn't exist in the current run; and three new run-level WARNING log
> lines — `retrieval.floor_starvation`, `quality.judge_unreliable`, and
> `run.call_budget_high` — all purely observational, none change routing
> or abort a run. Full details, including the new config knobs and what
> to expect in a debug trace, are under **Guardrails — What To Expect In
> The Logs** below.

> **D-55 note (test suite now 344):** the grounded-convergence gate above
> catches ungrounded evidence at CONVERGENCE time; it never stopped
> off-topic content from entering `state.evidence` at RETRIEVAL time in
> the first place. Live trace (run p205.141-check) found the actual entry
> point: `retrieval_chain._sufficient`'s topical-overlap requirement
> floored at a single shared word for any query with ≤7 distinctive
> terms — which is every `corpus_reformulated` retry by construction. A
> reformulated army query matched an unrelated Memcached document on one
> accidental shared word ("size"), and that legitimately-cited but
> topically-wrong evidence went on to prime `gap_generator`'s next cycle
> toward more off-topic queries. Fixed by raising the floor to 2 shared
> terms (capped so it never exceeds the query's own term count). This
> closes the entry point for one specific failure shape — an accidental
> single-word match — not `gap_generator`'s tendency to propose off-topic
> queries at all, which remains open; see **Guardrails** in `README.md`
> for the follow-up run that confirmed both halves of that distinction.

> **This is a hands-on manual, not the honesty audit.** `README.md`'s
> [Limitations](../README.md#limitations) section is the authoritative,
> itemized account of what's fixed vs. still broken (28 fixed items, plus
> a short list of genuine open gaps — most notably that self-critique can
> still pass a report whose claims aren't backed by any retrieved
> evidence; there is no programmatic, claim-by-claim grounding check yet).
> For the rationale behind a specific design decision, see `DECISIONS.md`
> (D-1 onward; see that file for the current range). If a step below "works" but the result looks thin
> or oddly confident, check those two documents before assuming your
> setup is broken — it may be a known, documented gap instead.