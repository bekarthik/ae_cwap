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
dangling edges, self-loops, multiple entry points, cycles, two edges between the
same pair of nodes, a branch that is not exactly one true and one false edge,
outbound edges on a result node, retrieval steps with no corpus, agent steps with
no agent, a review on a step that produces no answer, and a step that nominates
itself as its own reviewer. A graph that would fail at run time therefore returns
a 422 on save.

A plain node *may* have several outgoing edges — that is parallel work, and each
edge becomes its own job. It is a branch's two edges that are a choice.

### Designing a workflow from a goal

```
Browser ──POST /api/design──► Gateway ──► design.design
                                             │
                                   classify (for a hint)
                                             │
                              ┌──────────────┴──────────────┐
                     unanswered questions?            everything known
                              │                              │
                     CLARIFYING: what it                     ▼
                     could not infer, each          model plans the team:
                     with a stated reason           how many agents, what
                              │                     each hands to the next
                          answers ─────────────►            │
                                                    match each against the
                                                    agents this tenant has
                                                             │
                                              ┌──────────────┴──────────────┐
                                        an existing agent            a new one
                                              │                              │
                                     reused, marked as such         planned fresh
                                              └──────────────┬──────────────┘
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

**The plan is the model's; the validation is not.** How many agents a goal needs
is a property of the goal (see *The goal decides the structure* below), so the
model decides it and every plan is then checked against the same contracts as a
hand-drawn graph. One bad agent rejects the whole plan — half a design is not a
design — and the blueprint runs as the fallback, so a deployment on the offline
stub still gets a coherent workflow.

**An agent that already exists is reused, not duplicated.** A workspace that
plans a fresh "Researcher" for every goal ends up with six of them, each with a
sixth of the memory, and none getting better. `_agent_for` matches a planned
agent to a stored one by name; the stored agent keeps its role and its memory and
gains any skill the new plan needs, and the design response marks it `reused` so
the user can see what they are getting rather than assuming it is new. Reuse
never rewrites a role: an agent the user has tuned is not something a later
design gets to overwrite.

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

Fan-out does not weaken this. A step with three outgoing edges enqueues three
jobs, each naming one concrete node; what a payload never contains is a *choice*
for the worker to make.

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
knobs: current Claude models *reject* `temperature` with a 400, and a reasoning
model's depth is asked for four different ways depending on whose server is in
front of it. Each provider therefore declares
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

Every answer is read as it is produced. `streaming.py` reassembles the stream
into the body a blocking call would have returned, so nothing above the transport
knows the difference; what changes is that the deadline now measures the gap
between fragments rather than the length of the answer, and that the text reaches
the run panel while it is being written. Fragments are coalesced by `live.py` and
published with `LogBus.transient` — fanned out and relayed, never persisted,
because the finished text is already on the step and a row per fragment would
bury the run report in its own tokens.

Reasoning works the same way, and for the same reason: `reasoning_effort`,
`reasoning: {effort}`, `think: true` and `chat_template_kwargs` are four spellings
of one idea, and sending the wrong one is a 400. `reasoning.py` holds the table,
`_send` drops the parameter and retries once if the server refuses it, and what
the model thought is kept — an `agent.reasoned` line in the run's log and the full
text on the step — rather than discarded, since on a thinking model that is where
most of the work is visible.

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

### Fan-out, and the barrier that makes it safe

v2 allowed exactly one outgoing edge from anything but a branch. That was the
right first constraint — one successor per job is what makes a payload
deterministic — and it also meant two unrelated pieces of work were done one
after the other for no reason.

v4 lets any node have several outgoing edges. **Each enqueued job still names
exactly one node**, so the mandate is untouched; what changed is how many jobs a
completed step may produce.

```
      ┌──► research ──┐
start ┤               ├──► synthesise ──► output
      └──► survey  ───┘
              ▲                ▲
        two jobs, two      a join: dispatched once,
        workers            when every inbound path has landed
```

Three things had to hold before the contract could allow it:

* **A join waits for all of its inbound paths.** `_join_ready` checks the
  execution state for every inbound edge's source; a branch that arrives first
  dispatches nothing.
* **A join is dispatched exactly once.** Two branches finishing at the same
  moment on two workers would both see the other's output committed.
  `claim_dispatch` is an insert against a unique key, so precisely one wins and
  the other returns without enqueuing.
* **A join sees every branch's output, not just the one that woke it.**
  `_joined_bindings` unions the bindings of all inbound edges and the context is
  rebuilt from the run's committed state. Getting this wrong is not subtle — the
  join fails with *template referenced undefined variable*, which is how it was
  found.

The cycle rule is unchanged and load-bearing. The graph is still a DAG, so a run
still terminates.

### Reviewing a step's work without a cycle

A produce-check-correct loop is the single most reliable way to raise output
quality, and drawn on a canvas it is a cycle — which the graph contract rejects,
because termination would become a property of the model's judgement rather than
of the structure.

So the loop lives *inside* the step:

```
node executor ──► agent produces a draft
                       │
                       ▼
                 reviewer agent reads it against the objective
                       │
              ┌────────┴────────┐
        says APPROVED      says what is wrong
              │                  │
              ▼                  ▼
        that is the answer   author revises  ──┐
                                   ▲            │
                                   └────────────┘
                              up to max_rounds (1–5)
```

`ReviewConfig` names the reviewing agent, the round limit and the approval
phrase. The reviewer is an agent rather than a prompt on purpose: it has its own
skills and its own memory, so a reviewer gets better at reviewing the same way
everything else here gets better. A step cannot review itself — an agent that
wrote the draft approves it — and only a step that produces an answer can be
reviewed at all.

The round limit is a bound on cost, not an opinion about quality. A reviewer that
never approves would otherwise spend a run's entire budget, and when the limit is
reached the last draft stands with the review history recorded on the step, so
the report says the work went out unapproved rather than implying it passed.

### Which model a step runs on

Three levels, resolved outward until one of them says something:

```
node.params  →  AgentDefinition  →  the tenant's stored choice  →  the environment
(this step)     (this agent)        (this workspace)               (this deployment)
```

`executors._with_node_model` applies the node's override by rebuilding the agent
through its contract — not `model_copy`, which skips validation. `params` is a
free-form dict that no schema checks, so a thinking effort of "enormous" typed
into a node would otherwise travel to the provider and fail there, naming
neither the node nor the field.

A node override is a copy, never a write: the stored agent keeps its own
settings, so the override is scoped to the node and disappears with it.

#### The bug this had

`get_provider()` began `if _provider is not None: return _provider`, and
`_provider` was doing two jobs — a deliberate pin from `reset_provider_cache`,
*and* a lazy cache of the deployment default assigned whenever anyone asked for
a provider with no tenant in scope. The second poisons the first. `/health`
calls `describe_provider()` → `get_provider()` with no tenant, so a container
healthcheck pinned the whole process to the deployment default seconds after
boot, and from then on the tenant's stored choice was never looked up. A
workspace could store LM Studio, display LM Studio, and run every step on the
stub.

It is worth recording why no test caught it: the suite always pins a provider,
and with a pin that early return is the correct behaviour. The bug only exists
in the unpinned case — which is every real deployment and was no test.
`tests/test_provider_selection.py` is that case.

The two roles are now `_pinned` and `_default`, and only `_pinned` short-circuits.

### One model per agent

`AgentDefinition` carries `model_provider`, `model_override` and
`thinking_effort`. `runtime._provider_for` reads them:

```
agent names a backend?  ──yes──►  provider_for_choice(provider, model)
        │                              │
        no                    credential = what this tenant saved
        │                              for *that* provider
        ▼
the workspace's provider
```

The fields existed on the contract, in the database and in the API response for
two versions while nothing read them, which is worse than not offering them: a
stored field that is ignored is a promise the product does not keep, and "mix
models freely within one workflow" was that promise.

A pinned provider — what the tests and the dev runner set — still wins over an
agent's preference, so nothing routes around a deliberate override.

### What a workflow remembers about itself

Agent memory and skill memory were written after every run; workflow memory was
read and never written. So the scope that should have accumulated *this
workflow's* hard-won knowledge — which sources are worth checking, what the
output is supposed to look like — stayed empty for the life of a workspace.

A run that **succeeds** now writes its answer to the workflow's memory scope,
condensed to `OUTCOME_EXCERPT` characters. Two limits do the work:

* **Only the answer, and only on success.** A failed run's output is a symptom,
  not a lesson; storing one teaches the next run to repeat it. Failures already
  reach the agent and skill layers, which is where the actionable part of a
  failure lives.
* **An excerpt, not a transcript.** A scope that stores every output in full is a
  transcript, and a transcript recalled into a later system prompt is a
  context-window problem wearing a memory's clothes.

Recording a lesson is never worth failing a run over, so the write is wrapped and
a failure is logged rather than raised.

### The workspace as one picture

`GET /api/workspace/map` returns the tenant's agents, skills, workflows,
connected servers and model backends as nodes, and the relationships between them
as links — `holds`, `from`, `staffs`, `reviews`, `runs_on`. A node's size is how
much of a thing it is (skills held, times invoked, steps in the graph, tools on
the server); its glow is how much it has learned.

Links pointing at anything the caller cannot see are dropped rather than rendered
as a dangling edge, which matters more than it sounds: a workflow referencing an
agent from another tenant must not become a visible node labelled with somebody
else's name.

The rendering (`web/components/Brain.tsx`, `web/lib/cortex.ts`) is a generated
shell rather than a downloaded anatomical mesh — layered value noise over a
sphere, a midline groove for the fissure, a cerebellum and a stem. A real mesh
would look better and cost several megabytes plus a loading state on the first
screen a new user sees.

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

There is exactly one exemption, and it applies only to MCP: the hosts named by
`mcp_connect/directory.py`. Empty-by-default was making the intended use as
impossible as the unintended one — nobody could connect GitHub without an
environment variable on a machine they may not administer — and a fixed,
code-reviewed list of published endpoints answers "which hosts may this
deployment reach" as well as an allow-list does, in a reviewed file rather than
in a text box. An HTTP *node* gets no exemption at all, an unlisted MCP host is
gated exactly as before, and `CWAP_MCP_DIRECTORY=off` removes the exemption.

### An MCP connection fails in four distinguishable ways

`mcp_connect/client.py` bounds a connection with more than one deadline, which
is worth explaining because a single one looks simpler and was actively
misleading.

```
reach the host        REACH_TIMEOUT (10s)   → could not reach the host
  │                                            (URL, DNS, or egress)
  ▼
answer with MCP       (no deadline — it       → that URL answered, but not
  │                    either does or does       with MCP (login page? proxy?)
  │                    not)
  ▼
handshake             ┐                      → stopped responding while
  │                   ├ CONNECT_TIMEOUT (60s)   completing the handshake
  ▼                   │  shared between them
list tools            ┘                      → stopped responding while
                                                listing its tools
```

Every deadline sits strictly inside the next one out. When the outer wall was
equal to the SDK's own 30s default, the two fired together and the vaguer
message won — so a host that could not be reached and a server that had gone
quiet produced the same sentence after the same thirty seconds.

Ordering the deadlines was necessary and not sufficient, and the two further
faults are worth keeping written down because both make a deadline *look* set
while doing nothing:

* **`asyncio.wait_for` awaits the cancellation it requests.** It honours its own
  deadline only when the task can be cancelled promptly, and a request wedged
  inside the SDK's anyio task group cannot be — the group is suspended at the
  generator yield we are still inside, so the cancellation never completes and
  `wait_for` never returns. `_phase` uses `asyncio.wait`, which returns when the
  timeout elapses whatever the task is doing, and treats the cancel as a request
  rather than something to wait on.
* **Reporting after teardown is reporting too late.** `ready.set_exception` used
  to sit outside the `async with`, so it ran only once the transport had closed
  — and closing waits on that same task group. The phase error was raised on
  time and queued behind a teardown that outlasted the caller.

Together those two are why a 60s phase deadline produced no error at all and the
70s outer wall reported the generic one.

Before any of that, the platform sends **its own `initialize`** with plain httpx
and reports whatever comes back. That is not belt-and-braces; it is the only way
to see a refusal at all. `_handle_post_request` calls `raise_for_status()` inside
a task started with `tg.start_soon`, so a `400` lands in a background task while
`initialize()` goes on waiting for a reply on a memory stream nothing will ever
write to — the transport dies and the caller is never told
([python-sdk#1941](https://github.com/modelcontextprotocol/python-sdk/issues/1941),
[adk-python#4901](https://github.com/google/adk-python/issues/4901)). GitHub's
remote server answers some clients with exactly that `400`
([github-mcp-server#598](https://github.com/github/github-mcp-server/issues/598)),
which is how an immediate, articulate refusal reached a user as "did not respond
within 70s".

The pre-flight only speaks up about answers that are unambiguously a refusal — a
4xx/5xx, or a body that is not MCP. A transport fault falls through to the SDK,
which reports those well, so this probe can never be the reason a working server
is turned away.

The two content faults are the interesting ones, because neither is a timeout:

* **Not MCP at all.** The SDK reports an unusable content type by *sending a
  `ValueError` into the read stream*, and `ClientSession`'s default message
  handler discards it. The pending request is never answered and never fails —
  it waits. Passing a `message_handler` that fails the connection is what turns
  that permanent wait into an immediate, accurate error.
* **A stall after the handshake.** A server may decline the optional
  server-to-client GET stream; GitHub's answers `405`. That is spec-legal, and
  it can stall the client library on the *next* request
  ([python-sdk#1941](https://github.com/modelcontextprotocol/python-sdk/issues/1941)).
  The platform cannot fix the library, but it can tell the difference between
  "your URL is wrong" and "your setup is fine and this is a known upstream
  defect" — which are opposite instructions to give somebody.

`tests/test_mcp_http.py` runs against two real servers for this: one that offers
the GET stream and one that refuses it. Until it existed, every MCP test used
stdio, so the transport every hosted server actually uses had no coverage at all.

### Schema growth

`create_all` only creates whole tables, so a release that adds a column to an
existing one breaks every database created before it. `add_missing_columns()`
runs after creation and applies the one migration that is safe unattended:
adding a column that has a default or is nullable. Renames, retypes and drops
are left to a migration tool and a human, and a column that cannot be added
safely is logged by name rather than failing every boot with a database error
that does not say which column it meant.

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
| `agents` | Stored agents: role, skills, iteration budget, memory settings, the model backend and thinking depth this agent prefers, version. |
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
