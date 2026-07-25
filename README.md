# AI Cognitive Workflow Platform

A visual builder for multi-step AI agent workflows. Someone who has never written
code describes a goal in plain language, reviews the workflow the system proposes,
adjusts it on a canvas, and runs it — watching every step, every model call and
every decision stream back in real time.

The platform runs end to end with **no external infrastructure and no API keys**:
SQLite, an in-memory broker and a deterministic stub model are the defaults. Each
of those swaps to PostgreSQL, Redis and the Claude API through configuration
alone.

```bash
make install
make api    # http://localhost:8000  (gateway + inline worker)
make web    # http://localhost:3000  (canvas)
```

Register an account at `localhost:3000`, click **Start from a goal**, type
*"Plan my weekend trip to Denver"*, and press **Run**.

---

## What it does

| Epic | What the user sees | Where it lives |
| --- | --- | --- |
| **1 · Requirement diagnosis** | Types a goal, gets a reasoned proposal and a draft workflow | `services/nlp` |
| **2 · Visual builder** | Drags steps onto a canvas, draws arrows, maps values between them | `web/` |
| **3 · Memory** | Uploads documents, links a Knowledge Context to a step, previews what it retrieves | `services/knowledge` |
| **4 · Execution & monitoring** | Presses Run, watches a live log, reads a step-by-step report of *why* the answer came out that way | `services/orchestrator` |

Two properties are load-bearing throughout:

**Diagnosis reports gaps instead of guessing.** If a goal needs documents that
have not been uploaded, or an API credential the platform does not have, that is
returned as a `gap` — not scaffolded as a node that would fail at run time.

**A workflow that cannot execute cannot be saved.** Cycles, dangling edges,
half-wired decisions and retrieval steps with no corpus are rejected by the
contract during request parsing, so structural problems surface at design time
rather than mid-run in front of the user.

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
                                ├─ LLM Proxy      (the only vendor SDK caller)
                                ├─ RAG retrieval  (tenant-scoped)
                                └─ HTTP connector (default-deny egress, 2PC gated)
```

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
  nlp/                Goal diagnosis and workflow scaffolding
  knowledge/          Chunking, embeddings, tenant-scoped RAG
  orchestrator/       State machine, executors, worker, Celery entry point
  llm_proxy/          The only module that talks to a model vendor
web/                  Next.js canvas (React Flow)
tests/                160 tests, no external services required
deploy/               Container images
docs/                 Architecture, contract mandate, epic traceability
```

---

## Configuration

Everything has a working default; see [`.env.example`](.env.example) for the full
list. The ones that matter:

| Variable | Default | Change it to… |
| --- | --- | --- |
| `CWAP_DATABASE_URL` | `sqlite:///./cwap.sqlite3` | a PostgreSQL DSN |
| `CWAP_BROKER` | `memory` | `redis` (then run the worker separately) |
| `CWAP_LLM_PROVIDER` | `stub` | `anthropic` (with `ANTHROPIC_API_KEY`) |
| `CWAP_JWT_SECRET` | a known dev string | **required** in any deployment |
| `CWAP_HTTP_ALLOWLIST` | empty — all outbound calls blocked | hosts an HTTP node may reach |

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
make test        # backend suite
make typecheck   # frontend
make check       # both
make contracts   # re-approve the contract lock after a deliberate version bump
```

### Using a real model

```bash
export CWAP_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...      # or run `ant auth login`
```

`services/llm_proxy` is the only place that imports a vendor SDK. It handles the
three things that are easy to get wrong on current Claude models: no sampling
parameters (they are rejected with a 400), safety refusals arrive as successful
responses and must be checked before reading content, and server-side fallback
is opted into so a declined request is re-run rather than failing the workflow.

---

## Known limitations

Stated plainly, because each is a deliberate boundary rather than an oversight:

- **The default embedder is lexical, not semantic.** `HashingEmbedder` is a
  hashed bag-of-words with crude stemming. It is deterministic and needs no
  credentials, which makes retrieval testable — but it matches wording, not
  meaning. Production should swap in a real embedding model; the `Embedder`
  protocol is the only thing that has to change.
- **Uploads are plain text only.** PDF and DOCX extraction belongs in its own
  service. Binary uploads are refused with an explanation rather than indexed as
  mojibake that would quietly poison every retrieval.
- **Goal diagnosis is a deterministic keyword planner.** That is a defensible
  default — it is predictable, testable, free, and reports what it cannot do —
  but it will not infer intent the catalogue has never seen. The catalogue in
  `services/nlp/skills.py` is data, so extending it is an entry, not a rewrite.
- **Parallel fan-out is not supported.** Only decision nodes branch, and the two
  paths do not rejoin. Concurrent step execution is a real feature, not a
  configuration flag, and the contract would need a join primitive.
- **Migrations are not wired up.** `init_db()` creates tables; a deployment that
  outlives its first schema change needs Alembic.
