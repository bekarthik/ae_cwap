# Architecture

## Why this shape

The brief called for a microservices topology with an API gateway. This
repository implements that as a **monorepo of independently deployable
services** that share one contract package. Each service under `services/` owns
a bounded context, imports only `cwap_contracts` and `cwap_common` from its
siblings, and could be lifted into its own process or container without touching
its logic. `docker-compose.yml` already runs the gateway and the worker as
separate services against shared Postgres and Redis.

What is deliberately *not* split yet is the HTTP surface: all routers live behind
one FastAPI app. Splitting them is a routing change — each router maps 1:1 to a
target service — and doing it before there is load to justify it would buy
deployment complexity and network hops with nothing in return.

## Request paths

### Designing a workflow

```
Browser ──PUT /api/workflows/{id}──► Gateway ──► WorkflowGraph (contract)
                                                      │ validation
                                                      ▼
                                              workflows table (JSONB)
```

Validation happens during request parsing. `WorkflowGraph` rejects duplicate ids,
dangling edges, self-loops, multiple entry points, cycles, over-wide fan-out,
outbound edges on a result node, and retrieval steps with no corpus. A graph that
would fail at run time therefore returns a 422 on save.

### Diagnosing a goal

```
Browser ──POST /api/diagnose──► Gateway ──► nlp.scaffold
                                              │
                                   keyword match over the skill catalogue
                                              │
                                    ┌─────────┴─────────┐
                              can wire it up?      cannot wire it up
                                    │                   │
                              graph node             gap (shown to the user)
```

The scaffolder composes each generated prompt **from the bindings the node will
actually receive**, so a draft cannot reference a variable that will not exist.
That invariant is asserted directly in `tests/test_diagnosis.py`, and the
scaffolded graphs are executed end to end in `tests/test_execution.py`.

### Running a workflow

```
POST /api/workflows/{id}/runs
   │
   ├── snapshot the graph onto the run     (an edit mid-run cannot change it)
   ├── build JobContext from the principal (tenant, user, required scopes)
   ├── ProducerWrapper.publish
   │      ├── schema validation
   │      ├── authorisation pre-check
   │      └── enqueue
   └── 202 Accepted + run_id            ← the request never waits for execution

Worker loop
   │
   ├── ConsumerWrapper.next
   │      ├── schema validation           ─┐ failure → DLQ, no retry,
   │      └── authorisation runtime check ─┘          main queue keeps flowing
   │
   ├── rebuild run context from committed state rows   (restartable)
   ├── execute exactly the node the payload names
   ├── one transaction: state row + progress counter
   ├── plan the successor from the immutable graph
   └── enqueue it, or finalise the run
```

## Key design decisions

### Branch nodes are resolved at plan time

A job payload must declare a single, concrete `next_step_definition` before it is
enqueued. A decision node's successor is not knowable in advance, which appears
to conflict with that requirement.

It does not, because a decision performs no external work — it only compares
values already present in run context. So the planner evaluates it inline, writes
its own state row (with `source_service = BRANCH_EVALUATOR`, so the run report
explains the path taken), and continues to the chosen successor. Every enqueued
job therefore names one deterministic next step, and pathing is derived from the
saved graph rather than chosen by a worker.

### Run context is rebuilt from the database, never carried in memory

`Worker._rebuild_context` reads committed `workflow_execution_state` rows in
insertion order. A job redelivered to a different process reconstructs exactly
the context the original would have had. This is what makes workers fungible and
lets `docker-compose.yml` run two of them without coordination.

### The entry node's defaults become the run's inputs

The Start node merges its declared defaults with whatever the caller supplied,
and that merged set is written back to the run. Downstream `$run.input.*`
bindings therefore mean "the values this run effectively started with", which is
what both the scaffolder and a user editing bindings expect. (This was found by
driving the real UI: every scaffolded workflow failed on its second step until
the merged set was persisted. `tests/test_run_inputs.py` is the guard.)

### One module talks to the model vendor

`services/llm_proxy` is the only place that imports an LLM SDK. That keeps API
keys in one process, gives model calls one place to be logged and redacted, and
means the current-model API details — no sampling parameters, refusals arriving
as successful responses, server-side fallback — are handled once instead of in
every node executor.

### Egress is default-deny

An HTTP node is user-authored content executed server-side. Without an allow-list
it is a server-side request forgery primitive pointed at the cloud metadata
endpoint. `CWAP_HTTP_ALLOWLIST` is empty by default, and a workflow containing an
HTTP node additionally requires the `WRITE_EXTERNAL` scope, which is not granted
on registration.

## Data model

| Table | Purpose |
| --- | --- |
| `users` | Auth Service. PBKDF2 password digests, per-tenant scopes. |
| `workflows` | Saved graphs as JSONB, versioned on each save. |
| `runs` | One execution instance, with the graph snapshot and effective inputs. |
| `workflow_execution_state` | Per-step output. `UNIQUE(run_id, step_execution_id, source_service)` — first write wins. |
| `idempotency_ledger` | The external-call gate. `UNIQUE(run_id, step_execution_id, operation)`. |
| `run_logs` | Durable copy of the streamed feed. `UNIQUE(run_id, seq)`. |
| `dead_letters` | Parked messages with their precise failure reason. |
| `documents` / `chunks` | Indexed corpora and their embeddings. |

Every JSON column is declared with a `JSONB` variant, so the same models run on
SQLite and PostgreSQL without a second schema definition.

## Scaling notes

- **The gateway is stateless.** Scale horizontally behind any load balancer.
- **Workers are fungible** because state writes are idempotent and context is
  reconstructed from the database. Add replicas freely.
- **The log bus fans out in-process.** With multiple gateway replicas, a browser
  watching a run must reach the replica whose worker is emitting — either pin
  with sticky sessions, or move the bus to Redis pub/sub. The durable copy in
  `run_logs` means nothing is lost either way; only the live tail is affected.
- **Vector search is a table scan** in the reference implementation. That is
  correct for hundreds of documents and wrong for millions; the `Embedder` and
  the retrieval query are the two places to change.
