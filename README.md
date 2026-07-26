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

Above those three, the workspace itself is drawn as a brain — every agent, skill,
workflow, connected server and model backend, sized by how much of a thing each
is and lit by how much it has learned. `/brain` opens it full screen; a workspace
with nothing in it yet shows a plainly labelled example rather than an empty
skull.

---

## What it does

| Epic | What the user sees | Where it lives |
| --- | --- | --- |
| **1 · Requirement diagnosis** | States a goal, answers a couple of questions, reviews the agents the system decided on — or copies a template | `services/design`, `services/templates` |
| **2 · Visual builder** | Adjusts the design on a canvas, or assembles one by hand | `web/` |
| **3 · Memory** | Reads what each agent and skill has learned, corrects it, uploads documents agents can search | `services/memory`, `services/knowledge` |
| **4 · Execution & monitoring** | Presses Run, watches each agent think and reach for skills, reads a step-by-step report of *why* the answer came out that way | `services/orchestrator`, `services/agents` |

These properties are load-bearing throughout:

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
restart. That is the workspace default; **any agent can name its own backend,
model and thinking depth**, because a triage step and the synthesis step that
someone actually reads do not want the same model. Systems the platform has no
connector for are reached through **MCP**: a tenant connects a server and its
tools become skills agents can be given.

**Work can be checked before it is handed on.** Any step may be given a reviewer
— another agent, with its own instructions — that either approves the output or
sends it back with what to fix, up to five rounds. The graph contract rejects
cycles, so the loop lives inside the step rather than as an edge that loops back.

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
tests/                760 tests, no external services required
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
| `CWAP_LLM_THINKING_MODE` | `auto` | `off` to stop engaging reasoning models' thinking |
| `CWAP_LLM_STREAM` | `auto` | `off` only for a proxy that mangles server-sent events |
| `CWAP_JWT_SECRET` | a known dev string | **required** in any deployment |
| `CWAP_HTTP_ALLOWLIST` | empty — all outbound calls blocked | hosts an HTTP node or an unlisted MCP server may reach |
| `CWAP_WRITE_EXTERNAL` | `owner` — a workspace's first account may run outward-acting workflows | `everyone`, or `nobody` to grant it by hand |
| `CWAP_MCP_DIRECTORY` | `true` — the built-in server list is connectable | `off` to require the allow-list for every host |
| `CWAP_LLM_TIMEOUT` | `600` seconds | lower it on hosted models, or raise it in Models per workspace |
| `CWAP_MCP_ALLOWED_COMMANDS` | empty — stdio MCP servers disabled | commands a stdio MCP server may launch |
| `CWAP_MCP_TIMEOUT` | `60` seconds | raise it for an MCP server behind a slow proxy (`CWAP_MCP_CONNECT_TIMEOUT`, default `10`, bounds *reaching* the host separately) |
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
400, and a reasoning model's depth is asked for four different ways depending on
whose server is serving it. Rather than send a parameter
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

Answers are **streamed**: text appears in the run panel as the model writes it,
and — the part that matters more — the timeout becomes a silence detector rather
than a ceiling on how long an answer may take, so a slow local model runs to
completion while a wedged one fails in seconds.

A reasoning model is asked to reason, in whichever dialect its server speaks, and
a server that refuses the parameter costs one retried call. What it thought is
kept and shown in the run report — on a thinking model that is usually where the
work is. Agents are also told to *check rather than recall*: where a fact could be
established with a tool they hold, using it beats answering from memory.

A model that can see is sent the picture. An image URL in a step's input becomes
an image block rather than twenty-eight characters describing a chart — but only
when the model reports vision, since image blocks to a text model are a 400 on
most servers and silently dropped content on the rest.

**The model is chosen at three levels, each inheriting from the next.**

```
step  →  agent  →  workspace
```

Open an agent and set its provider, model and thinking depth, and every workflow
using it follows. Select a step on the canvas and set *Model for this step*, and
only that step changes — which is the grain most decisions actually have ("the
synthesis step in this one workflow needs the big model"). Before this existed
the only way to vary one step was to clone the agent, splitting its memory in
two and making both halves worse.

Blank means inherit, at every level, so a workflow nobody has touched behaves
exactly as it did. The credential comes from what the workspace saved *for that
provider*, so a key entered for one backend is never sent to another's endpoint,
and both editors say up front which backends this workspace can actually reach.

---

## Connecting to other systems (MCP)

An agent reaches a system nobody here wrote a connector for through the **Model
Context Protocol**. Connect a server under *Connected systems*, and each tool it
advertises becomes a skill an agent can be given — so an agent that can read a
repository is an agent holding a skill, exactly like one that can summarise text.

**Start from the list.** *Connect a system* opens on the servers the platform
already ships the details for — GitHub, DeepWiki, Context7, Hugging Face,
Sentry, Stripe, Cloudflare's docs, and the reference local ones — each with its
endpoint, what it is for and which credential it will ask for. Their hosts need
no allow-listing, because a fixed, code-reviewed list *is* an answer to "which
hosts may this deployment reach". Everything in the form stays editable, and a
server nobody listed is still one URL away.

The two transports have very different blast radii and are gated separately:

| Transport | What it is | Gate |
| --- | --- | --- |
| `http` | Streamable HTTP to a URL | The built-in list, or `CWAP_HTTP_ALLOWLIST` for any other host |
| `stdio` | The platform launches a process on the worker | `CWAP_MCP_ALLOWED_COMMANDS` — **empty by default** |

stdio is off until an operator turns it on, because without that gate "connect an
MCP server" is a remote shell with the worker's privileges. An allow-list entry
matches either a bare command name resolved through `PATH`, or an exact absolute
path — a bare entry deliberately does not authorise `/tmp/uploaded/npx`.

```bash
# Let this deployment run npx-based MCP servers, and reach a host of its own
export CWAP_MCP_ALLOWED_COMMANDS=npx
export CWAP_HTTP_ALLOWLIST=mcp.internal.example.com

# Or require the allow-list for everything, listed servers included
export CWAP_MCP_DIRECTORY=off
```

A tool that can change something needs the same `WRITE_EXTERNAL` scope an HTTP
node does and passes the same two-phase gate, so a redelivered step cannot open
two pull requests. Tools a server marks read-only are exempt — reading a
repository is not a side effect.

**A workspace's first account holds that scope**, because on a self-hosted
install the person who registered is the operator, and a permission nothing can
grant is a wall rather than a permission. Later accounts do not, which is what
keeps it meaningful once a workspace is shared; `CWAP_WRITE_EXTERNAL` changes
who gets it. Holding the scope is permission to *ask* — what may actually be
reached is still `CWAP_HTTP_ALLOWLIST` and the MCP directory, both unchanged.

**When a connection fails, the error says which kind of failure it was.** One
timeout covering everything reported four different problems with one sentence,
and none of the four remedies is the same:

| What happened | What it says | What to do |
| --- | --- | --- |
| The host never answers a SYN | *could not reach the host* (10s) | Fix the URL, or the deployment's egress |
| The server refuses us | *refused the connection: HTTP 400. It said: …* | Whatever the server said — it is quoted verbatim |
| It answers with a page, not MCP | *that URL answered, but not with MCP* | It is a login page, a proxy, or the wrong URL |
| The handshake stalls | *stopped responding while completing the MCP handshake* | Probably not an MCP endpoint; or raise `CWAP_MCP_TIMEOUT` |
| The handshake works, listing stalls | *stopped responding while listing its tools* | Your setup is fine — see below |

The second row exists because the MCP client library cannot report it. A
server's `4xx` is raised inside a background task where nothing delivers it to
the waiting request, so an outright refusal arrives as a hang. The platform
therefore sends its own `initialize` first, with plain httpx, and quotes the
answer.

**MCP needs its library.** `mcp` is an optional extra, like Redis and
PostgreSQL — the container image installs it, and a local `make install` gets it
through the `dev` extra. If it is missing, the connect dialog says so up front
and the gateway logs `MCP connector unavailable` at boot, because every server
in the catalogue would otherwise fail identically for a reason that has nothing
to do with any of them.

**Each attempt is narrated in the gateway's log** — the pre-flight's verdict
(`MCP <url>: pre-flight 400 application/json`) and each phase with its elapsed
time (`finished completing the MCP handshake in 0.4s`). At startup the gateway
prints `MCP connector ready — pre-flight on; …`; **if that line is missing from
`docker logs`, the container is running an older image** — `docker compose up
--build` rebuilds it, and merely restarting after a `git pull` does not. The
older builds also identify themselves by wording: only they say *"did not
respond within Ns"*, which the current code never emits.

That last row is a known defect in the MCP client library rather than anything
about your server ([python-sdk#1941](https://github.com/modelcontextprotocol/python-sdk/issues/1941)):
a server that declines the *optional* server-to-client stream — GitHub's does,
with a `405` — can stall the client on the request after the handshake. Raising
`CWAP_MCP_TIMEOUT` is the workaround, and the error says so rather than leaving
someone re-checking a URL that was never wrong.

An endpoint that replies `200 OK` with an HTML page is worth calling out
separately: the client library reports that by posting an error into a stream
nobody reads, so the request waits forever without ever failing. That is why a
proxy's login page used to look exactly like a server that had gone silent.

---

## Known limitations

Stated plainly, because each is a deliberate boundary rather than an oversight:

- **The default embedder is lexical, not semantic.** `HashingEmbedder` is a
  hashed bag-of-words with crude stemming. It is deterministic and needs no
  credentials, which makes retrieval testable — but it matches wording, not
  meaning. Set `CWAP_EMBEDDING_PROVIDER` to a real embedding model for anything
  that cares about retrieval quality.
- **Uploads are plain text only.** PDF and DOCX extraction belongs in its own
  service. Binary uploads are refused with an explanation rather than indexed as
  mojibake that would quietly poison every retrieval.
- **A design is not repeatable between runs.** The model decides how many agents
  a goal needs and what they hand to each other, so two identical requests can
  come back with different teams. That is the cost of the count being a property
  of the goal rather than of a blueprint, and the response says which way a given
  design was produced. `services/design/blueprints.py` survives as a hint when
  classification is confident and as a fallback when there is no usable model, so
  the offline stub and small local models still get a coherent workflow.
- **Reflection is mechanical, not introspective.** After a run an agent records
  which skills worked, which failed, and whether its budget was enough — things
  the runtime knows for certain. It does not ask the model to write its own
  lessons, which reliably fills memory with plausible platitudes. The cost is
  that a genuinely subtle insight goes unrecorded unless a user types it in.
- **Skill synthesis cannot produce code or network calls.** A synthesised skill
  is a prompt, a retrieval, a transform, or an ordered composition of those. Any
  capability that genuinely needs to reach outside the platform is reported as a
  gap for a human to wire up with a credential and an allow-list entry.
- **Fan-out is concurrent in the graph, not in wall-clock time.** Two branches
  off one node are dispatched as independent jobs and rejoin at a barrier, so the
  work is genuinely parallel across workers — but a single inline worker still
  runs them one after another. More throughput is more workers.
- **Migrations are not wired up.** `init_db()` creates tables; a deployment that
  outlives its first schema change needs Alembic. Concurrent workers racing to
  create the schema is handled (a Postgres advisory lock), but that is
  bootstrapping, not migration.
