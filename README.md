# AgenticOS — Multi-Agent Operations Intelligence Platform

AgenticOS investigates building-operations issues (HVAC alarms, energy
spikes, maintenance questions) using a **Planner → Specialist Data Fetchers →
Supervisor** architecture, built with [LangGraph](https://github.com/langchain-ai/langgraph)
and [LangChain](https://github.com/langchain-ai/langchain), running on a
local, open-source LLM via [Ollama](https://ollama.com/) — no external API
keys required.

This is a refactor of an earlier prototype. The behavior is the same from
the user's point of view; the internals were rebuilt around one goal:
**cut every LLM call that wasn't doing planning or summarization.**

## What changed, and why

The original prototype made **up to 10 LLM calls per request**: one for the
planner, then *two* per specialist agent (one call just to decide to invoke
the one tool it was already bound to, one to turn a small JSON blob into a
paragraph), then one for the supervisor. Each of those calls is a full
model round-trip — on a local 3B model, that's most of the request's
latency, for calls that add no information.

**This version makes exactly 2 LLM calls per request, always:**

```
                ┌─────────┐
   request ──▶  │ Planner │  1 structured-output call:
                └────┬────┘  picks agents AND extracts their parameters
                     │       (asset_id / topic) in one shot
                     ▼
          ┌─────────────────────┐
          │   run_specialists    │  0 LLM calls — direct, concurrent
          │ (deterministic I/O)  │  DB queries / vector-store lookups
          └──────────┬───────────┘
                      ▼
               ┌────────────┐
               │ Supervisor │  1 call: synthesizes the raw structured
               └────────────┘  results into the final answer
```

- **Planner** (1 call) — a structured-output call (`PlannerDecision`, a
  Pydantic schema) that both selects which specialists are relevant *and*
  extracts the parameter each one needs (`asset_id`, `topic`) directly from
  the request text. This is what removes the old "decide how to call the
  tool" call per specialist: the planner already knows.
- **Specialists** (0 calls) — no longer LLM agents at all. Each is a plain
  function (`AssetRepository.get_asset`, `AlarmRepository.get_alarm_history`,
  `EnergyRepository.get_energy`, `DocumentationRepository.get_sop`) invoked
  directly with the planner's extracted parameter. Independent specialists
  run **concurrently** via a thread pool (`AGENTICOS_SPECIALIST_CONCURRENCY`,
  default on), since they're blocking I/O (SQLite, vector search) with no
  dependencies between them.
- **Supervisor** (1 call) — synthesizes the specialists' raw structured
  results (still just data, never LLM-generated prose) into one final
  answer: summary, probable causes, next steps.
- **`no_data` fallback** — if the planner selects nothing (irrelevant
  request, or a planner error after retries), the graph routes to a
  dedicated node instead of dead-ending, same as before.

Net effect: response time scales with the number of *specialist I/O calls*
running in parallel, not with the number of *sequential LLM calls* — and the
LLM is only ever invoked twice, full stop.

## Other production hardening

- **Retries with backoff.** The original wrapped every LLM call in a
  try/except that gave up on the first error. `llm/client.py` now retries
  transient failures (`AGENTICOS_LLM_MAX_RETRIES`, default 2) with
  exponential backoff before falling back to the graceful-degradation path.
- **Validated config.** `os.getenv` calls scattered around are now one
  `pydantic-settings` `Settings` object (`settings.py`), type-checked at
  startup, all overridable via `AGENTICOS_*` env vars or `.env`.
- **Repository pattern for data access.** `AssetRepository`,
  `AlarmRepository`, `EnergyRepository`, `DocumentationRepository` each own
  one data source, take an injectable path (for tests), and raise a single
  `ToolExecutionError` instead of leaking raw `sqlite3.Error`/Chroma
  exceptions into the graph.
- **A tool registry, not four copy-pasted node functions.** The prototype
  had near-identical `alarm_node`/`asset_node`/`documentation_node`/
  `energy_node` functions. `tools/registry.py` is the single place that
  defines what a specialist is (a label, a required parameter, and a fetch
  function); the planner's prompt and the specialist executor both read
  from it. Adding a fifth data source is one registry entry, not four files.
- **Structured logging**, separate from the UI-facing reasoning log
  (`logs: list[str]` in state, still returned for the Streamlit reasoning
  panel) — `logging_config.py` gives JSON logs in production
  (`AGENTICOS_LOG_JSON=true`) or readable text in development.
- **Lazy-loaded RAG stack.** `langchain-community`, HF embeddings, and
  Chroma (which pull in `torch`) are only imported inside
  `DocumentationRepository.__init__`, not at module load time, so anything
  that doesn't touch documentation lookups — including most of the test
  suite — doesn't pay to import them.
- **A gap in the seed script fixed.** The original `data/seed_db.py` never
  created the `energy` table `EnergyRepository` reads from, even though the
  shipped `.db` file had one — a fresh clone couldn't reproduce it. Fixed in
  `scripts/seed_db.py` (which now also supports `--if-missing` for idempotent
  container startup).
- **Chunked RAG retrieval.** The documentation agent used to embed one
  vector per raw PDF *page*; a manual page mixing several procedures made
  for noisy matches. `rag/loader.py` now splits pages into overlapping
  chunks (`AGENTICOS_RAG_CHUNK_SIZE` / `_CHUNK_OVERLAP`) before embedding,
  and retrieval defaults to MMR (`AGENTICOS_RAG_SEARCH_TYPE=mmr`) instead of
  plain similarity search, so the top-k results aren't several near-duplicate
  chunks from the same page. Results also carry `source`/`page` metadata
  instead of being opaque text blobs.
- **Conversation memory.** Turns (request + final answer + which agents ran)
  are persisted per `conversation_id` in `memory/store.py` (a SQLite table,
  best-effort — a memory write failure never fails the request). The
  Planner and Supervisor prompts get the last `AGENTICOS_MEMORY_MAX_TURNS`
  turns as context, and the Planner's asset-ID regex fallback also checks
  history, so "what about its alarms?" resolves the asset named in an
  earlier turn. Single-shot requests (CLI without `--conversation-id`, or
  any caller that doesn't pass a `conversation_id`) never touch the memory
  DB at all.
- **A FastAPI service for containerized deployment.** `api/app.py` exposes
  `POST /investigate`, `GET /conversations/{id}/history`, and split
  `/health` (liveness) / `/ready` (readiness) probes — the deployment target
  for the `Dockerfile` / `docker-compose.yml` / `deploy/aws/` assets. The
  Streamlit UI is unchanged as the interactive front end and now also
  supports multi-turn conversations via the same memory store.

## Project structure

```
agenticos/
├── src/agenticos/
│   ├── settings.py              # pydantic-settings config (AGENTICOS_* env vars)
│   ├── logging_config.py        # structured logging (JSON in prod, text in dev)
│   ├── exceptions.py            # LLMUnavailableError, ToolExecutionError, PlannerParseError, ConversationMemoryError
│   ├── db/
│   │   ├── connection.py        # SQLite session context manager
│   │   └── repositories.py      # AssetRepository, AlarmRepository, EnergyRepository
│   ├── rag/
│   │   ├── loader.py            # PDF loading + chunking (RecursiveCharacterTextSplitter)
│   │   ├── vector_store.py      # Chroma construction/loading + similarity/MMR search
│   │   └── documentation_repository.py
│   ├── memory/
│   │   └── store.py             # ConversationStore: per-conversation turn history (SQLite)
│   ├── llm/
│   │   └── client.py            # model construction + retrying safe_invoke
│   ├── tools/
│   │   └── registry.py          # single source of truth: agent -> (param, fetch fn)
│   ├── agents/
│   │   ├── state.py             # AgentState, PlannerDecision/AgentAssignment schemas
│   │   ├── planner.py           # the 1st LLM call (history-aware)
│   │   ├── specialists.py       # 0 LLM calls, concurrent data fetch
│   │   ├── supervisor.py        # the 2nd LLM call + no_data fallback (history-aware)
│   │   └── graph.py             # LangGraph wiring, incl. memory persistence node
│   ├── api/
│   │   └── app.py               # FastAPI service for Docker/AWS deployment
│   └── ui/
│       └── streamlit_app.py     # interactive UI, multi-turn via the same memory store
├── scripts/seed_db.py           # seeds data/agenticos.db (assets, alarms, energy)
├── tests/                       # 35 tests, fully offline (FakeChatModel, temp SQLite)
├── data/agenticos.db
├── docs/                        # source PDFs for the documentation agent
├── deploy/aws/                  # ECS Fargate task definition + deployment guide
├── Dockerfile / docker-compose.yml / .dockerignore
├── main.py                      # CLI entry point (one-shot or --interactive REPL)
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## Setup

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally

### 2. Install Ollama and pull a model
```bash
ollama pull qwen2.5:3b-instruct
```

### 3. Install
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure (optional)
```bash
cp .env.example .env
```

### 5. Seed the database
```bash
python scripts/seed_db.py
```

### 6. Run
```bash
streamlit run src/agenticos/ui/streamlit_app.py
```
or from the command line (one-shot):
```bash
python main.py "Investigate why AHU-01 is consuming more energy today."
```
or interactively, with conversation memory across turns:
```bash
python main.py --interactive
```
or as an HTTP API (see [Deployment](#deployment) below for Docker/AWS):
```bash
uvicorn agenticos.api.app:app --reload
```

## Deployment

### Docker (local / any container host)
```bash
docker compose up --build
```
This brings up Ollama, a one-shot model pull, the FastAPI service
(`http://localhost:8000`, docs at `/docs`), and the Streamlit UI
(`http://localhost:8501`) — see `docker-compose.yml`. Data (SQLite DB +
Chroma vector store) persists in named volumes across restarts.

To build just the app image:
```bash
docker build -t agenticos .
docker run -p 8000:8000 -e AGENTICOS_OLLAMA_BASE_URL=http://host.docker.internal:11434 agenticos
```

### AWS
See `deploy/aws/README.md` for an ECS Fargate + ALB + EFS deployment guide
(ECR push, task definition, IAM roles, LLM-backend options, and a lower-ops
App Runner alternative).

## Testing

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest -v
```

All 35 tests run fully offline via `FakeChatModel` (a scripted stand-in
implementing `invoke` and `with_structured_output(...).invoke`) and a
throwaway SQLite file per test — no Ollama server or vector store required.
`tests/test_graph.py::test_full_graph_happy_path_makes_exactly_two_llm_calls`
is a regression test for the core guarantee: it asserts every scripted LLM
response is consumed exactly once, proving the graph never exceeds 2 calls
regardless of how many specialists run. `tests/test_memory.py` covers the
conversation store and history-aware planning; `tests/test_api.py` covers
the FastAPI service (health/readiness probes, `/investigate`, history).

## Known limitations

- Documentation lookups depend on `sentence-transformers` embeddings being
  downloaded on first run (no network access at query time after that).
- Single global model instance per process; horizontal scaling for
  multi-user production traffic would mean running multiple app instances
  behind a queue/load balancer rather than one in-process Streamlit app.
  The FastAPI service is stateless aside from the SQLite-backed data/memory
  stores, so scaling it is mostly a question of those stores (see below).
- SQLite (both operational data and conversation memory) is fine for a
  single instance or an EFS-mounted single-writer ECS task, but isn't a good
  fit for many concurrent writers over NFS. Scaling the API horizontally
  would mean migrating `db/connection.py` and `memory/store.py` to
  RDS/Aurora Postgres.
- Conversation memory is a fixed recent-turn window (`AGENTICOS_MEMORY_MAX_TURNS`),
  not summarized — fine for the few-turn follow-ups this is built for, but
  a very long conversation will eventually stop fitting useful context in
  that window. Add a summarization step on top of `memory/store.py` if that
  becomes a real usage pattern; the store itself doesn't need to change.
- The regex-based asset-ID fallback in the planner is a safety net for a
  small local model occasionally forgetting to extract a parameter it
  already selected an agent for — it's not a substitute for the planner
  extracting it correctly in the first place, and won't catch every asset
  ID format.
