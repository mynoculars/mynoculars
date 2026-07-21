# OPERATIONS — How To Actually Run This Thing

This is the missing manual. No architecture theory. Just: what to install, how to
load data, how to run, in the exact order, with copy-paste commands and what you
should see. If a step's output doesn't match, that's the diagnostic.

---

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

## Level 1 — Skeleton (run this first, needs nothing)

Zero services, zero API keys, zero network. If this works, your Python
environment is correct and the whole graph is wired.

**Windows (PowerShell):**
```powershell
cd agentic-research-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
$env:PYTHONPATH = "src"
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

**Linux/macOS:**
```bash
cd agentic-research-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=src
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

**What you should see** (this is L1 — note the zeros, they are EXPECTED here):
```json
{
  "intent": "Comparison",
  "goals": 2,
  "iterations": 1,
  "evidence_items": 0,      <-- zero because no corpus is loaded yet
  "recall": 0.0,            <-- zero for the same reason
  "llm_calls": 6,
  "search_calls": 2,        <-- workers RAN, they just found nothing
  "critique_passed": true
}
```

`search_calls: 2` with `evidence_items: 0` is the signature of L1: the workers
executed, retrieval degraded to empty because the stores are down. **This is
success for L1.** Now run the tests to confirm the logic:

```bash
python -m pytest tests/ -q
# expect: 28 passed
```

If L1 runs and 28 tests pass, your code is fine. Everything from here is about
feeding it data.

---

## Level 2 — Real Retrieval (the level you actually want to see)

Now we bring up the two search engines and load the sample corpus so the
workers have something to find. **Still `LLM_MODE=stub`** — we are NOT touching
the language model yet. One new variable at a time. That is the whole trick to
not being confused: change ONE thing, observe, then change the next.

### Step 2a — Install the two search engines

You already have these installed natively, per our earlier conversation
(Qdrant, OpenSearch, Postgres native on Windows). If so, **skip Docker** — just
make sure they're running and point `.env` at them (Step 2c). If you do NOT have
them, the repo ships a Docker option:

```bash
docker compose up -d
# starts postgres (5432), qdrant (6333), opensearch (9200)
docker compose ps       # all three should say "running"
```

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
POSTGRES_DSN=postgresql://agent:agent@localhost:5432/agent
CORPUS_INDEX=agent_corpus
MEMORY_COLLECTION=agent_semantic_memory
```

If your native installs use different ports/credentials, change them HERE, not
in code. (Native OpenSearch on Windows often ships with security ON → it needs
HTTPS + a password. If ingest says `opensearch.unavailable`, that's why — tell
me and I'll add auth support, ~10 lines. For now Docker OpenSearch runs with
security off and Just Works.)

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

### Step 2e — Run the SAME query, now with data

```bash
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

**What changed — this is L2, and this is the moment it "works":**
```json
{
  "evidence_items": 6,      <-- NOW the workers found real evidence
  "recall": 1.0,            <-- goals covered by retrieved facts
  "search_calls": 2,
  "memory_writes": 6        <-- passed report fed evidence into memory
}
```

`evidence_items` jumped from 0 to a real number. **That is the research loop
doing its job.** The report text is still the stub placeholder (because
`LLM_MODE=stub`), but the *retrieval, coverage, and memory* are all genuinely
working now. Run it a second time and watch `memory_hits` become non-zero — the
agent now remembers the first run.

---

## Level 3 — Full (real report text from a real model)

Only now do we touch the language model. Two providers, primary + fallback.

### Step 3a — Bring up the primary model (local Qwen Cogito)

This project expects a local **llama-server** exposing an OpenAI-compatible
endpoint. You already run this for your other work. Start it on the model of
your choice; note the URL and port. Example shape:

```powershell
.\llama-server.exe -m <your-qwen-model>.gguf -ngl 999 -c 32768 --port 8080
```

### Step 3b — (optional) a Gemini key for fallback

Get a Google AI Studio API key for `gemini-2.0-flash`. If you skip this, the
agent still runs on the primary alone; it just won't have a fallback when the
local model errors or scores low.

### Step 3c — Flip `.env` to live

```ini
LLM_MODE=live
LLM_PRIMARY_BASE_URL=http://127.0.0.1:8080/v1       # your llama-server
LLM_PRIMARY_MODEL=qwen-cogito                         # whatever it reports
LLM_FALLBACK_API_KEY=your-gemini-key-here            # optional
```

### Step 3d — Run

```bash
python -m research_agent.cli "Compare Redis and Memcached for session caching"
```

Now the report body is a real, evidence-grounded answer the model wrote from the
6 retrieved facts. Watch the logs for `llm.fallback` lines — those tell you when
the primary failed or scored below threshold and Gemini took over.

---

## Which Software Runs, And Why (the whole inventory)

| Software | Needed for | Port | If it's down |
|---|---|---|---|
| **Python 3.11+ venv** | Everything | — | Nothing runs |
| **Qdrant** | Dense (meaning) retrieval + semantic memory | 6333 | Dense leg off; memory off; agent still runs |
| **OpenSearch** | Keyword (BM25) retrieval | 9200 | Keyword leg off; agent still runs on dense only |
| **Postgres** | Durable checkpointer + run history | 5432 | Falls back to in-memory checkpointer; HITL still works within a run |
| **llama-server (Qwen)** | Primary LLM (L3 only) | 8080 | L3 fails unless Gemini fallback is set |
| **Gemini API** | Fallback LLM (L3 only) | cloud | No fallback; primary must work |

**Key mental model:** every one of these is *optional* and degrades gracefully.
The agent is designed to run on a bare laptop (L1) and light up more capability
as you add services (L2, L3). You never have to bring up everything at once —
that's the whole point, and it's also how you debug: add one service, observe
the telemetry change, move on.

---

## How To Actually Test (three senses of "test")

**1. Run the unit/integration test suite (proves the logic):**
```bash
export PYTHONPATH=src        # or $env:PYTHONPATH="src" on Windows
python -m pytest tests/ -q   # 28 tests, all offline, ~0.3s
```
This needs NO services and NO model — it uses the stub and fakes. If these pass,
the graph logic is correct. Run this after any code change.

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

---

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
then re-ingest. Native: delete the Qdrant collection and OpenSearch index, or
just re-run ingest (it upserts by id, so re-running overwrites).

---

## The 60-Second "Is Everything Up?" Check

Run this mental checklist when something's not working:

```bash
# 1. Is my venv active and PYTHONPATH set?
echo $PYTHONPATH          # must print: src

# 2. Are the engines reachable? (only needed for L2/L3)
curl http://localhost:6333/collections     # Qdrant: JSON response = up
curl http://localhost:9200                 # OpenSearch: JSON response = up

# 3. Is the corpus loaded?
#    Re-run ingest; "indexed 10 / embedded 10" = yes, "SKIPPED" = engine down

# 4. Run L1 (stub, no services) — does the graph itself work?
python -m research_agent.cli "test"        # telemetry prints = graph OK

# 5. Read the logs. Every run prints JSON log lines to stderr. The truth is
#    there: qdrant.unavailable, opensearch.unavailable, llm.fallback,
#    checkpointer.memory_fallback. Grep for ".unavailable" to see what's down.
```

The logs are the diagnostic. Degradation is silent by design in the *output*,
but every degradation writes a log line. When confused: read stderr.