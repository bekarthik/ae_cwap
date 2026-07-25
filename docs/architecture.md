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
outbound edges on a result node, retrieval steps with no corpus, and agent steps
with no agent. A graph that would fail at run time therefore returns a 422 on
save.

### Designing a workflow from a goal

```
Browser ──POST /api/design──► Gateway ──► design.design
                                             │
                                    classify → blueprint
                                             │
                              ┌──────────────┴──────────────┐
                     unanswered questions?            everything known
                              │                              │
                     CLARIFYING: what it                     ▼
                     could not infer, each          plan agents from the
                     with a stated reason           blueprint (structure)
                              │                              │
                          answers ─────────────►   model rewords them (wording)
                                                             │
                                                    resolve each agent's skills
                                                             │
                                              ┌──────────────┴──────────────┐
                                        exists / can be built        needs a human
                                              │                              │
                                    attached to the agent         gap (shown to the user)
                                              │
                                    input → agent → … → output,
                                    sharing one memory scope
```

Two boundaries hold this together.

**Structure is deterministic; only wording is model-generated.** A design that
varied between identical requests could not be reviewed, and one that only
appeared on a frontier model would make the platform's central promise
conditional on which model you configured. So `design/blueprints.py` decides how
many agents there are and what they hand to each other, and the model is asked
only to sharpen each role and objective against the actual goal. Any failure —
no model, bad JSON, the wrong number of agents — keeps the deterministic design,
so the model call can only improve the result.

**Synthesis produces data, never code.** A capability the tenant lacks is built
as a prompt, a retrieval over a corpus they already own, a text transform, or an
ordered composition of those — everything a user could have assembled by hand on
the canvas. If synthesis emitted Python, a model that had read a hostile document
could write arbitrary code into a privileged worker, and the egress allow-list
and scope checks would all become bypassable. HTTP skills are refused outright by
`_validate_proposal` and reported as a gap instead.

Each generated objective is composed **from the bindings the node will actually
receive**, so a design cannot reference a variable that will not exist. That
invariant is asserted in `tests/test_design.py`, and the designed graphs are
executed end to end in `tests/test_execution.py`.

### Running an agent step

```
node executor ──► agents.runtime.run_agent
                       │
                       ├─ recall: agent memory + workflow memory ──► system prompt
                       │
                       ├─ loop, up to the agent's iteration budget:
                       │     model decides ──► tool call? ──► skill executes
                       │           ▲                              │
                       │           └──── result fed back ─────────┘
                       │     no tool call → that is the answer
                       │
                       ├─ budget spent → ask once for its best answer so far
                       │
                       └─ reflect: skill failures → skill memory
                                   budget/skill sequence → agent memory
                                   recalled entries reinforced or weakened
```

The whole loop runs inside the node's single transaction. A redelivered job
re-runs the entire agent rather than resuming half of one, which keeps the
mandate's transactional boundary intact without the agent needing its own
resumption protocol.

A skill that fails is reported **to the agent**, not raised: it can try different
arguments or a different capability, which is the entire point of giving it a
loop. Only a failure that stops the agent running at all — an unreachable model,
an unusable definition — fails the step.

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

### Memory lives at the scope that owns the lesson

One `memories` table, four scopes. The split is not filing — it decides who
inherits a lesson:

* A lesson about **using a capability** ("this needs an explicit focus on long
  inputs") belongs to the skill, so every agent that reaches for it benefits.
* A lesson about **performing a role** ("this shape of objective needs more than
  six iterations") belongs to the agent, so it applies in every workflow the
  agent staffs.
* A **fact this job established** belongs to the workflow, so the next agent —
  and the next run — has it.

Reflection is deliberately mechanical rather than introspective. Asking a model
to write its own lesson after every run reliably fills memory with plausible
platitudes; the observations actually worth keeping — which skills failed,
whether the budget was enough, which sequence worked — are things the runtime
already knows for certain.

Three failure modes get explicit handling, because each turns memory from an
asset into a liability:

| Failure mode | Handling |
| --- | --- |
| The same lesson accumulating every run | A near-identical write reinforces the existing entry instead of adding another |
| The embedder unavailable or since swapped | Relevance falls back to lexical overlap, topped up with the scope's most-proven entries, rather than returning nothing |
| A plausible but wrong lesson recalled forever | Recalled entries are weakened after a bad run; below a floor they stop being recalled, and a user can delete them outright |

### The entry node's defaults become the run's inputs

The Start node merges its declared defaults with whatever the caller supplied,
and that merged set is written back to the run. Downstream `$run.input.*`
bindings therefore mean "the values this run effectively started with", which is
what both the scaffolder and a user editing bindings expect. (This was found by
driving the real UI: every scaffolded workflow failed on its second step until
the merged set was persisted. `tests/test_run_inputs.py` is the guard.)

### One module talks to any model

`services/llm_proxy` is the only place that talks to a model. That keeps
credentials in one process, gives model calls one place to be logged and
redacted, and — more importantly — makes the platform model-agnostic: nothing in
a saved workflow names a vendor, so the same graph runs on a local Llama, a
self-hosted Qwen, or a hosted Claude.

Two client implementations cover everything, because almost every model server in
use speaks the OpenAI chat-completions wire format. Adding a backend is an entry
in `presets.py`, not a new class, and `catalogue.py` records what each common
model can do — tool calling, vision, a reasoning mode — so the picker offers real
choices instead of two free-text boxes. An unlisted model still runs: its
capabilities are inferred from its name, and anything the backend then rejects is
handled at run time.

The hard part is not transport, it is that backends accept genuinely different
knobs: current Claude models *reject* `temperature` with a 400, and open models
have no notion of `effort`. Each provider therefore declares
`ProviderCapabilities`; `GET /api/runtime` reports them so the canvas renders
only the controls that backend honours; a node keeps knobs meant for other
backends so a workflow stays portable; and anything ignored at run time is logged
and recorded on the step, so a run report never implies a setting took effect
when it did not. Full rationale in [`models.md`](models.md).

Agents need one capability specifically, and it is not reliably discoverable: a
hosted API advertises tool calling, a local llama.cpp build may or may not have
it, and asking is not possible. So the first agent turn simply tries. A 4xx that
is *about* tools and expresses "unsupported" downgrades that provider permanently
to a prompted JSON protocol and retries within the same call. The match is
subject-plus-negation rather than a list of exact sentences, because every
backend phrases it differently — and it is deliberately narrow on the subject,
since mistaking a rejected API key for a missing capability would silently
degrade every agent and hide the real problem.

### An MCP tool is a skill, not a node type

Connecting an MCP server could have been a new node type, a new executor and a
new place for failures to be handled. Making it a *skill* means none of that:
an agent is given it the same way, the model sees it in the same tool list,
failures come back to the agent the same way, the per-skill memory accrues the
same lessons, and the write-scope check already knows to ask.

The cost is one contract version bump — `SkillKind` could not gain a value
without one — and that is the versioning discipline working rather than a
workaround for it. v3 redefines only the skill contracts and re-exports the rest.

Two things real servers forced rather than being chosen:

* Parameters may be objects and arrays. A `create_pull_request` taking a list of
  labels is unusable if arguments flatten to strings.
* The server's own JSON schema is handed to the model verbatim. A paraphrase
  drifts from what the server validates against, and the model is told a shape
  that then gets rejected.

Sessions are held open on one background event loop and reused; a process per
tool call would make a three-tool agent start three subprocesses. Each connection
is owned by a single task that both opens and closes it, because anyio task
groups may only be exited from the task that entered them — closing from
elsewhere left the subprocess running.

### The model backend is chosen per tenant, at call time

A workflow author comparing a local Llama against a hosted Claude wants to try
one, look at the result, and try the other. So a tenant's choice is stored and
`get_provider()` resolves it per call.

Reaching the call was the interesting part. A skill deep inside an agent loop
asks for a provider and has no tenant argument; threading one through every
signature would touch every executor for a concern none of them own. A context
variable fits: the worker sets it once from the job's `job_context.tenant_id`,
and everything under that step resolves the right backend without knowing it
happened. It is set and reset around **one step**, so a fungible worker moving to
another tenant's job cannot inherit the previous tenant's endpoint or credential.
Environment configuration remains the deployment default and the fallback, and
the store is failure-tolerant — a model call must still work before the schema
exists and in a process with no database.

### A run must never look stuck

Three separate things conspired to make a finished run display as PENDING, and
each hid the others:

* The worker caught two exception types and let everything else propagate, which
  killed the standalone worker's loop and left a run nothing would ever finish.
  Any unanticipated failure now fails the *run*, with the reason, and the loop
  survives to take the next job.
* The log bus fanned out in-process, and `docker-compose` runs the worker as its
  own container — so the live feed reached nobody in the topology we ship. A
  Redis deployment now relays events over pub/sub, using the broker it already
  has. The relay never re-persists: the emitting process already did.
* The canvas derived status from the stream alone. It now polls the run report
  as well, so a delivery problem costs the live tail and nothing else.

The last is the important one. The stream is how a run *feels* live; the run
report is how the canvas *knows* what happened, and it is always reachable.

### The goal decides the structure

How many agents a workflow needs, what each hands to the next, and how much each
may iterate are properties of the goal. They cannot be settled before reading it.

This was originally the other way round: a blueprint fixed the team and the model
was asked only to reword it, under a prompt that said *"return exactly the same
number of agents, in the same order"*. That is how a request for a whole software
development organisation came back as three writers — the count was decided
before anything read the request.

Now the model plans, and the blueprint has two smaller jobs:

* **A hint.** When classification is confident, the shape that usually works for
  that kind of request is offered as context — explicitly as something the model
  may ignore, use partially, or replace.
* **A fallback.** No usable model, or a plan that does not validate, and the
  blueprint runs. A deployment on the offline stub, or a small local model that
  cannot return clean JSON, still gets a coherent workflow.

A plan is validated against the same contracts as everything else: within the
agent ceiling, no duplicate names, objectives that reference the bindings they
will actually receive, and skills that are either found or synthesised. One bad
agent rejects the whole plan, because half a design is not a design.

`MAX_AGENTS` is a bound on review effort and cost, not an opinion about
structure. Every agent is a stored object with its own memory and its own chain
of model calls.

**Iteration lives inside agents**, not in the graph. A review-and-rework cycle is
a cycle and the graph contract rejects those, so an agent that must produce,
check and correct is given a larger budget — the model sets it per agent, since
only the plan knows which agents those are. The value is clamped: structural
decisions are the model's, somebody's bill is not.

What this costs is repeatability. A design is no longer identical between runs,
so the response says when it was planned rather than taken from a blueprint.

### A token is a claim; the database is the truth

`current_principal` confirms the identity a token names still exists in the
tenant it claims. They can disagree without anything being wrong — a container's
database is recreated while `CWAP_JWT_SECRET` stays put, an account is deleted, a
user moves tenant — and in every case the token still verifies.

Checking at the *authentication* boundary rather than only at the authorisation
one is what stops a phantom identity from creating agents, skills and workflows
it owns, then being refused the moment someone presses Run. And the status is
401, not 403: 403 means "we know you and you may not", while the remedy here is
to sign in again, which a browser can act on by dropping the stored session.

A job whose user is revoked *while it sits in the queue* is a different case and
stays a 403 into the dead letter queue — the job is already in flight and there
is nobody to re-authenticate. Mandate §2's consumer re-check is unchanged.

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
| `agents` | Stored agents: role, skills, iteration budget, memory settings, version. |
| `skills` | Declarative capabilities, with their origin and reliability counters. |
| `memories` | Every scope's learned entries, with accumulated usefulness. |

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
