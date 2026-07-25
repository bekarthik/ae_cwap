# AI Cognitive Workflow Platform

A builder for multi-step AI agent workflows. Someone who has never written code
describes a goal in plain language; the system asks only what it cannot work out,
then decides the steps itself and staffs each one with an agent that has its own
skills and its own memory. The user reviews that design on a canvas, adjusts it,
and runs it — watching every step, every skill invocation and every decision
stream back in real time.

Each run makes the next one better. What an agent learns about its role stays
with the agent, what a capability learns about being used well stays with the
skill, and what a workflow establishes stays with the workflow.

The platform runs end to end with **no external infrastructure and no API keys**:
SQLite, an in-memory broker and a deterministic stub model are the defaults. Each
of those swaps to PostgreSQL, Redis and a real model through configuration alone
— and "a real model" means any of them, from Llama on your laptop to a hosted
Claude.

```bash
make install
make api    # http://localhost:8000  (gateway + inline worker)
make web    # http://localhost:3000  (canvas)
```

Register an account at `localhost:3000`. You land on three ways in:

* **Describe what you need** — say it in plain language; the system asks what it
  cannot infer, decides the steps, and staffs each with an agent.
* **Start from a template** — five that run as given. Copy one and change it.
* **Build it yourself** — an empty canvas, if you already know the steps.

Press **Run** and watch each agent think, reach for skills, and hand over.

---

## What it does

| Epic | What the user sees | Where it lives |
| --- | --- | --- |
| **1 · Requirement diagnosis** | States a goal, answers a couple of questions, reviews the agents the system decided on — or copies a template | `services/design`, `services/templates` |
| **2 · Visual builder** | Adjusts the design on a canvas, or assembles one by hand | `web/` |
| **3 · Memory** | Reads what each agent and skill has learned, corrects it, uploads documents agents can search | `services/memory`, `services/knowledge` |
| **4 · Execution & monitoring** | Presses Run, watches each agent think and reach for skills, reads a step-by-step report of *why* the answer came out that way | `services/orchestrator`, `services/agents` |

Four properties are load-bearing throughout:

**The system designs the workflow; the user reviews it.** Someone who already
knows which steps they need does not need this product, and someone who does not
cannot draw them. So the platform asks what it cannot infer — each question
carrying *why* it is being asked — then decides the steps itself.

**Every working step is an agent, not a prompt.** It is given an objective and a
set of skills, and it decides which to use and in what order, seeing each result
before choosing again. Agents are stored rather than embedded in a graph, so
improving one improves every workflow that uses it, and its memory has a stable
identity to accumulate against.

**Missing capabilities are built, within a boundary.** When a design needs a
skill the tenant does not have, the platform creates it — as a prompt, a search
over documents the tenant already owns, or a text transform. Never code, and
never an outbound call: those need a credential and an allow-listed host, which
are a human's decision, so they are reported as a gap instead.

**Any model, any tool, chosen from the product.** The model backend is picked in
the browser — the endpoint is asked what it actually serves, the configuration is
tested with a real completion, and saving takes effect on the next run with no
restart. Systems the platform has no connector for are reached through **MCP**: a
tenant connects a server and its tools become skills agents can be given.

**A workflow that cannot execute cannot be saved.** Cycles, dangling edges,
half-wired decisions, retrieval steps with no corpus and agent steps with no
agent are rejected by the contract during request parsing, so structural problems
surface at design time rather than mid-run in front of the user.

---

## Architecture

```
Browser (Next.js + React Flow)
    │  REST + WebSocket
    ▼
API Gateway ─────── auth · routing · uniform error translation
    │
    │  ProducerWrapper: schema validation → authorisation pre-check → enqueue
    ▼
Queue (in-memory | Redis)                    ──► Dead Letter Queue
    │                                              (contract + authz failures,
    │  ConsumerWrapper: schema validation →         never retried, always visible)
    │  authorisation runtime re-check
    ▼
Worker ── state machine ── node executors ── PostgreSQL / SQLite
                                │
                                ├─ Agent runtime  ── skills ── LLM Proxy
                                │      │                         (any model,
                                │      └─ memory: agent · skill · workflow
                                │                          local or hosted)
                                ├─ RAG retrieval  (tenant-scoped)
                                └─ HTTP connector (default-deny egress, 2PC gated)
```

An agent step is one node to the state machine — the loop inside it runs within
that node's single transaction, so a redelivered job re-runs the whole agent
rather than resuming half of one.

Every message between those boxes is an instance of a model in
`packages/cwap_contracts`, the versioned schema registry. Services import those
models; nothing hand-rolls a dict for an inter-service payload.

Full detail: [`docs/architecture.md`](docs/architecture.md).

---

## The resilience layer

The platform's operating discipline is enforced in code, not convention.
[`docs/contracts.md`](docs/contracts.md) maps each rule to its implementation and
the test that proves it. In summary:

- **Contracts are immutable and versioned.** `extra="forbid"` and `frozen=True`
  on every model; a JSON-Schema fingerprint lock means changing a contract's
  shape without a version bump fails CI.
- **Determinism is structural.** A job payload carries `current_state` and a
  single legal `next_step_definition`, validated against a transition table.
  An illegal path is unrepresentable rather than merely discouraged.
- **Validation and authorisation happen twice** — once when a job is enqueued,
  once when it is consumed. The second check is what catches a permission
  revoked while the job sat in the queue.
- **Failures are parked, not retried.** A contract violation is deterministic and
  an authorisation revocation is intentional; retrying either only burns worker
  capacity. Both go straight to an observable DLQ and the main queue keeps
  flowing.
- **Every side effect is gated.** External calls run a PRE-CHECK / EXECUTE &
  COMMIT sequence keyed on `(run_id, step_execution_id)`, so a redelivered job
  cannot double-charge anything.
- **Every step is one transaction.** State, logs and progress commit together or
  roll back together.

---

## Repository layout

```
packages/
  cwap_contracts/     Versioned schema registry + fingerprint lock
  cwap_common/        Transactions, idempotency, broker, dual gateway, log bus
services/
  api_gateway/        Auth, routing, run reports, WebSocket log stream
  design/             The design conversation: questions, agents, the graph
  templates/          Ready-made workflows, instantiated per tenant
  mcp_connect/        MCP servers: policy, sessions, tools-as-skills
  agents/             The agent loop, and agent storage
  skills/             Built-in skills, synthesis of missing ones, execution
  memory/             Remember, recall, reinforce, prune — at every scope
  knowledge/          Chunking, embeddings, tenant-scoped RAG
  orchestrator/       State machine, executors, worker, Celery entry point
  llm_proxy/          The only module that talks to a model, any vendor
web/                  Next.js canvas (React Flow)
tests/                507 tests, no external services required
                      (and the same suite runs against PostgreSQL)
deploy/               Container images
docs/                 Architecture, contracts, model backends, epics
```

---

## Configuration

Everything has a working default; see [`.env.example`](.env.example) for the full
list. The ones that matter:

| Variable | Default | Change it to… |
| --- | --- | --- |
| `CWAP_DATABASE_URL` | `sqlite:///./cwap.sqlite3` | a PostgreSQL DSN |
| `CWAP_BROKER` | `memory` | `redis` (then run the worker separately) |
| `CWAP_LLM_PROVIDER` | `stub` | `ollama`, `vllm`, `together`, `anthropic`, … |
| `CWAP_LLM_TOOL_MODE` | `auto` | `native` or `prompted` to force one tool path |
| `CWAP_JWT_SECRET` | a known dev string | **required** in any deployment |
| `CWAP_HTTP_ALLOWLIST` | empty — all outbound calls blocked | hosts an HTTP node or HTTP MCP server may reach |
| `CWAP_MCP_ALLOWED_COMMANDS` | empty — stdio MCP servers disabled | commands a stdio MCP server may launch |
| `CWAP_SECRET_KEY` | falls back to the JWT secret | encrypts stored provider keys and MCP credentials |

A production-shaped stack (Postgres, Redis, gateway, two workers, canvas):

```bash
CWAP_JWT_SECRET=$(openssl rand -hex 32) docker compose up --build
```

Two worker replicas are not decoration — idempotent state writes are what make
workers fungible, and running more than one is the cheapest way to keep that
honest.

---

## Development

```bash
make test        # backend suite, on SQLite
make typecheck   # frontend
make check       # both
make contracts   # re-approve the contract lock after a deliberate version bump

# The same suite against the database production actually uses. Worth running
# before a release: SQLite and PostgreSQL differ in exactly the places this
# platform leans on — ON CONFLICT reporting, JSONB, locking — and a bug that
# only appears on Postgres is the worst kind. Each test gets its own schema.
make test-postgres CWAP_TEST_DATABASE_URL=postgresql+psycopg://cwap@localhost/cwap
```

### Using a real model

Any model, including open-weight ones running on your own hardware. Full table
and rationale: [`docs/models.md`](docs/models.md).

```bash
# Open-weight, local, no credentials
ollama pull llama3.1
export CWAP_LLM_PROVIDER=ollama CWAP_LLM_MODEL=llama3.1
```

```bash
# Hosted open models, or any OpenAI-compatible endpoint
export CWAP_LLM_PROVIDER=together CWAP_LLM_API_KEY=...
export CWAP_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

```bash
# Anthropic
export CWAP_LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...
```

RAG embeddings are configured the same way and independently, since a local
`nomic-embed-text` alongside a hosted chat model is a normal setup:

```bash
export CWAP_EMBEDDING_PROVIDER=ollama CWAP_EMBEDDING_MODEL=nomic-embed-text
```

In the canvas, click the model chip in the header. Pick a provider, press
**Detect models** to ask that endpoint what it actually serves, **Test** the
configuration with one real completion, and save. It applies to your next run —
nothing restarts, and the environment variables below remain the deployment
default for anyone who has not chosen.

`services/llm_proxy` is the only module in the codebase that talks to a model.
Two implementations cover everything: the Anthropic SDK, and one HTTP client for
every backend speaking the OpenAI chat-completions format.

Backends genuinely differ — current Claude models *reject* `temperature` with a
400, and open models have no notion of `effort`. Rather than send a parameter
that errors or silently drop one the user set, each provider declares its
capabilities, the canvas renders only the controls that backend honours, and
anything ignored at run time is logged and recorded on the step. A saved workflow
keeps knobs for other backends, so it stays portable across models.

Agents need one capability specifically: tool calling. Many capable open-weight
models never learned it, and tool support is not reliably discoverable up front —
a hosted API advertises it, a local llama.cpp build may or may not have it. So the
first agent turn sends `tools`; a rejection that names them downgrades that
provider permanently to a **prompted JSON protocol** and retries immediately,
rather than failing the turn. Agents therefore work on every backend, and the
canvas says which path a step is on rather than implying they are equivalent.

---

## Connecting to other systems (MCP)

An agent reaches a system nobody here wrote a connector for through the **Model
Context Protocol**. Connect a server under *Connected systems*, and each tool it
advertises becomes a skill an agent can be given — so an agent that can read a
repository is an agent holding a skill, exactly like one that can summarise text.

The two transports have very different blast radii and are gated separately:

| Transport | What it is | Gate |
| --- | --- | --- |
| `http` | Streamable HTTP to a URL | `CWAP_HTTP_ALLOWLIST`, the same list an HTTP node uses |
| `stdio` | The platform launches a process on the worker | `CWAP_MCP_ALLOWED_COMMANDS` — **empty by default** |

stdio is off until an operator turns it on, because without that gate "connect an
MCP server" is a remote shell with the worker's privileges. An allow-list entry
matches either a bare command name resolved through `PATH`, or an exact absolute
path — a bare entry deliberately does not authorise `/tmp/uploaded/npx`.

```bash
# Let this deployment run npx-based MCP servers
export CWAP_MCP_ALLOWED_COMMANDS=npx
export CWAP_HTTP_ALLOWLIST=api.github.com
```

A tool that can change something needs the same `WRITE_EXTERNAL` scope an HTTP
node does and passes the same two-phase gate, so a redelivered step cannot open
two pull requests. Tools a server marks read-only are exempt — reading a
repository is not a side effect.

---

## Known limitations

Stated plainly, because each is a deliberate boundary rather than an oversight:

- **The default embedder is lexical, not semantic.** `HashingEmbedder` is a
  hashed bag-of-words with crude stemming. It is deterministic and needs no
  credentials, which makes retrieval testable — but it matches wording, not
  meaning. Set `CWAP_EMBEDDING_PROVIDER` to a real embedding model for anything
  that cares about retrieval quality.
- **Streaming token output is not plumbed through.** Model calls are
  request/response, so a node's answer appears when the step finishes rather
  than token by token. The log stream is live; the model output within a step is
  not.
- **Uploads are plain text only.** PDF and DOCX extraction belongs in its own
  service. Binary uploads are refused with an explanation rather than indexed as
  mojibake that would quietly poison every retrieval.
- **Workflow design is a deterministic blueprint, reworded by the model.** The
  model sharpens each agent's role and objective against the actual goal but does
  not choose how many agents there are or what they hand to each other. That is
  deliberate: it makes the design reviewable (the same goal and answers give the
  same shape) and portable (it works on a small local model, or the offline
  stub). It also means a goal shaped unlike anything in
  `services/design/blueprints.py` becomes one capable generalist agent rather
  than a guessed pipeline. The blueprints are data, so adding a shape is an
  entry, not a rewrite.
- **Reflection is mechanical, not introspective.** After a run an agent records
  which skills worked, which failed, and whether its budget was enough — things
  the runtime knows for certain. It does not ask the model to write its own
  lessons, which reliably fills memory with plausible platitudes. The cost is
  that a genuinely subtle insight goes unrecorded unless a user types it in.
- **Skill synthesis cannot produce code or network calls.** A synthesised skill
  is a prompt, a retrieval, a transform, or an ordered composition of those. Any
  capability that genuinely needs to reach outside the platform is reported as a
  gap for a human to wire up with a credential and an allow-list entry.
- **Parallel fan-out is not supported.** Only decision nodes branch, and the two
  paths do not rejoin. Concurrent step execution is a real feature, not a
  configuration flag, and the contract would need a join primitive.
- **Migrations are not wired up.** `init_db()` creates tables; a deployment that
  outlives its first schema change needs Alembic. Concurrent workers racing to
  create the schema is handled (a Postgres advisory lock), but that is
  bootstrapping, not migration.
