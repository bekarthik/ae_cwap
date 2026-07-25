# Epics and user stories — traceability

Each epic and user story from the brief, what was built, and where to verify it.

---

## Epic 1 · Intelligent Requirement Diagnosis & Setup

**Brief:** NLP intake, skill gap analysis, agent recommendation, automatic
scaffolding. Outcome: a proposed workflow structure presented before the user
touches the canvas.

| Component (brief) | Built as | Output contract |
| --- | --- | --- |
| Goal Intake Endpoint | `POST /api/design` → `design.design` | `DesignResponse` — either clarifying questions, or the understanding, agents, **skill gaps** and graph |
| Skill Scaffolding Logic | `design._build_graph` | `WorkflowGraph` of agent nodes, validated on construction |
| Skill gap analysis | `skills.registry.ensure_capability` | `SkillDefinition` + whether it had to be created |
| Agent recommendation | `design.blueprints` + `design._plan_agents` | `PlannedAgent` — name, role, objective, skills, **rationale** |

### Three ways in

An empty canvas answers none of the questions a new user has, and "describe a
goal" assumes you can articulate one. So the launcher offers three routes,
matched to how much certainty someone arrives with:

| Route | Best when | What it does |
| --- | --- | --- |
| Describe what you need | You know the outcome, not the steps | The design conversation below |
| Start from a template | You do not yet know what this can do | Copies a working workflow into your workspace |
| Build it yourself | You already know the steps | An empty canvas |

Templates are data, instantiated per tenant — agents created, skills resolved or
built, a fresh memory scope — so copying one gives you your own agents to edit
rather than a shared object that changes under other people. Each declares its
prerequisites, checked against your workspace and shown before you copy: a
template that quietly produces a workflow failing on its third step is worse than
one that says "upload something first".

**Verify:** `tests/test_templates.py`, including a parametrised test that every
template in the catalogue runs end to end on the offline stub — the claim a
template makes is "this works".

### User Story 2 — goal input

> As a beginner workflow creator, I want to input a simple, high-level goal, so
> that the system automatically researches and presents a recommended initial
> sequence of agents and skills.

Designing is a short conversation rather than a single shot. A goal like "sort
out our onboarding" does not contain enough to design anything, and guessing from
it produces a confident, useless workflow. So the system says back what it
understood, asks only what the goal did not already answer — each question
carrying **why** it is being asked and a default so none is compulsory — and then
shows the design: the agents in order, what each is for, the skills each was
given, and which of those had to be built. Only then does anything reach the
canvas.

Four properties beyond the brief, each a deliberate choice:

- **The goal decides the structure.** How many agents, what each does, and how
  much each may iterate are decided from the request, not fixed in advance. A
  blueprint that matches is offered as a hint the model may ignore, and is the
  fallback when there is no usable model — which is what keeps the offline stub
  and small local models working. The cost is repeatability, and the design
  review says when a design was planned rather than taken from a blueprint.
- **Honest about gaps.** A capability that would need to reach outside the
  platform is reported as a gap with what a human must supply, rather than
  invented. A goal mentioning documents that have not been uploaded gets agents
  without a search skill, not a step that fails at run time.
- **Missing skills are created.** Anything the design needs and the tenant lacks
  is built as a prompt, a document search or a text transform, attached to the
  agent that needs it, and named in the review.
- **A question with no default is still answerable.** Leaving one blank counts as
  an answer, so clicking through never re-asks the same thing forever.

**Verify:** `tests/test_design.py` (51 tests), and `TestDesignedWorkflowsRun` in
`tests/test_execution.py`, which executes six designed workflows end to end and
asserts every working node is an agent.

---

## Epic 2 · Visual Workflow Builder Canvas

**Brief:** drag-and-drop nodes, directional edges with conditional and sequential
flow, parameter mapping between connected components.

| Component (brief) | Built as |
| --- | --- |
| Workflow Editor UI | `web/components/Builder.tsx` on React Flow |
| Node Interaction Handler | `web/lib/graph.ts` — canvas ⇄ contract mapping, connection defaults, pre-flight |

### User Story 1 — drag and connect

> As a non-technical user, I want to drag agents onto a canvas and draw arrows
> between them, so that I can visually design multi-step workflows without
> writing code or understanding API calls.

Eight step types in the palette, each with a plain-language description — an
**Agent** among them, which is what a designed workflow is built from. Drawing
a connection pre-fills a sensible parameter mapping for the two node types being
joined, so a new arrow is useful immediately. Decision nodes expose two labelled
handles (green "yes", red "no"). Selecting a connection opens its parameter
mapping for editing.

Conditional flow is real, not cosmetic: the canvas writes `condition: true/false`
onto the edge, the contract validates that a decision node has exactly one of
each, and the orchestrator resolves the decision at plan time and records why.

Selecting an agent step shows which agent staffs it, what that agent can do, and
what it has learned — read-only for the parts that belong to the agent rather
than to this step, because the same agent may staff a step in another workflow.
The step's *objective* is editable there; the agent's role and skills are edited
in the roster, where the consequence is visible.

**Verify:** `tests/test_graph_validation.py`, and the run report of a branching
workflow, which contains a `decide` step showing the comparison that was made.

---

## Epic 3 · Memory Management System

**Brief:** integration point for vector storage, document upload, retrieval
method selection. Outcome: a "Knowledge Context" node linkable into any step.

| Component (brief) | Built as | Output contract |
| --- | --- | --- |
| Knowledge Indexer Service | `knowledge.ingest` — chunk, embed, store | `KnowledgeHandle` |
| RAG Node Integration | `rag_retrieve` node + `knowledge.retrieve` | `RetrievalResult` → context payload |
| Agent-facing retrieval | a `retrieval` **skill** over one corpus | tool result, fetched when the agent decides it is worth it |
| Learned memory | `memory.remember` / `recall` at four scopes | `MemoryEntry`, `RecallResult` |

### Memory at four scopes

Documents are what the platform was *told*. Memory is what it *worked out*, and
it accumulates at the level that owns the lesson:

| Scope | Holds | So that |
| --- | --- | --- |
| **Skill** | how this capability is used well, what it does badly | every agent that reaches for it inherits the lesson |
| **Agent** | what this role has learned across every workflow it staffs | improving an agent once improves it everywhere |
| **Workflow** | facts this job established | one agent's finding is available to the next, and to the next run |
| **Run** | working state within a single execution | it does not leak into the next run |

Three problems this has to solve, none of which is storing text:

- **Duplicates.** An agent that learns the same lesson every run would, after
  fifty runs, recall fifty copies of it and nothing else. A near-identical write
  reinforces the existing entry instead of adding another.
- **Recall without embeddings.** The embedder can be down, or can have been
  swapped since a memory was written. Recall degrades to lexical overlap rather
  than returning nothing, and tops up with the scope's most-proven lessons so an
  agent is never left with an empty prompt on a query that happens to share no
  words.
- **Wrong lessons.** A memory that preceded a bad run loses trust; below a floor
  it stops being recalled. Users can also read every scope in the UI and delete
  anything they disagree with — memory you cannot inspect is memory you cannot
  trust.

### User Story 3 — contextual memory

> As an advanced user needing reliability, I want to link a specific knowledge
> bank to a workflow node, so that the agent's responses are grounded in my
> private corporate data.

Upload in the Knowledge panel, select the corpus on a Knowledge step, and — this
is the part that makes it trustworthy — **preview what the step will actually
retrieve** before wiring it in. Grounding you cannot inspect is grounding you
cannot trust.

Two safety properties:

- **A handle is not an authorisation.** Retrieval is scoped to the calling tenant
  on every read, so copying a graph between tenants cannot leak a corpus.
- **A retrieval step with no corpus cannot be saved.** The contract rejects it, so
  the user sees the problem at design time.

The default `HashingEmbedder` is lexical rather than semantic — see the
limitations section in the README. Swapping it is a change to one file.

**Verify:** `tests/test_knowledge.py` (17 tests, including tenant isolation),
`tests/test_memory.py` (25 tests covering dedupe, reinforcement, pruning, scope
and tenant isolation, and recall with a broken embedder), and the memory
assertions in `tests/test_agents.py` and `tests/test_skills.py`.

---

## Epic 4 · Execution & Monitoring Engine

**Brief:** a Run button with clear command execution, real-time log streaming of
actions, state changes, errors and output at every node. Outcome: a transparent
report letting the user trace *why* the AI arrived at an answer.

| Component (brief) | Built as | Output contract |
| --- | --- | --- |
| Execution Orchestrator | `orchestrator.Worker` + `state_machine` | `StepOutputContext` per node |
| Agent execution | `orchestrator.executors.execute_agent` → `agents.runtime.run_agent` | `AgentTurn` per iteration, in the step's derived context |
| Real-time Log Streamer | `cwap_common.logbus` + `WS /api/runs/{id}/logs` | `LogEvent` — timestamp, node, level, event, data |

Pressing **Run** returns in milliseconds with a run id; execution happens on the
worker. The canvas then shows each node pulsing as it runs and settling green or
red, while the log streams underneath.

The report at `GET /api/runs/{id}` is the trace: every step with its output, the
state it reached, how long it took, and the full ordered log. A branching run
includes the decision node's own entry, so the answer to "why did it go that
way?" is in the record rather than inferred.

An agent step is not opaque within that trace. It emits `agent.started`,
`agent.memory_recalled`, `agent.thinking` per iteration, `agent.skill_used` with
the arguments and whether it failed, and `agent.finished` or `agent.incomplete`.
Every turn is kept, so "why did it answer that?" reaches down to which capability
it reached for and what came back.

Streaming details that matter in practice:

- Sequence numbers are monotonic per run, so a browser reconnecting mid-run knows
  exactly where it left off — no gaps, no duplicates.
- History is replayed on connect, so opening the panel late still shows
  everything.
- Credentials are redacted at the single point every event passes through.
- Every event is also persisted, so a run report survives a browser refresh and
  an auditor asking three weeks later.

**Verify:** `tests/test_execution.py`, `TestLogStream` and `TestRuns` in
`tests/test_api.py`.

---

## Coverage summary

| Deliverable | Status |
| --- | --- |
| Goal → clarifying questions → designed graph | Built, deterministic in structure, gap-reporting |
| Every working step staffed by an agent with skills and memory | Built |
| Missing skills created and attached automatically | Built, declarative only |
| Per-agent, per-skill and per-workflow memory that improves across runs | Built |
| Drag-and-drop canvas with conditional edges | Built |
| Parameter mapping between steps | Built, with type-aware defaults |
| Document upload → embeddings → RAG node | Built, tenant-scoped, previewable |
| Run + real-time logs + step report | Built |
| Versioned contract registry with a change gate | Built |
| Dual validation/authorisation gateway + DLQ | Built |
| Idempotency, 2PC external calls, transactions | Built |
| Redis/Celery transport | Built (abstraction + Celery entry point; in-memory is the default) |
| Any model backend — local open-weight, self-hosted, or hosted | Built, capability-aware |
| Choosing provider and model from the browser, no restart | Built, per tenant, credentials encrypted |
| Auto-detecting what an endpoint actually serves | Built, with a real test call before saving |
| MCP connectors — any server, tools become skills | Built, stdio and streamable HTTP |
| Three canvas entry points, with a template catalogue | Built |
| Agents on models with no tool calling | Built — automatic downgrade to a prompted protocol |
| Model catalogue and picker (tools / vision / thinking per model) | Built |
| Embeddings from any OpenAI-compatible endpoint | Built; hashing is the offline default |
| PostgreSQL | Supported (JSONB variants); SQLite is the default |
| Parallel step execution | **Not built** — only decision branching |
| Token-by-token streaming from a model | **Not built** — steps are request/response |
| Schema migrations | **Not built** — `init_db()` only |
| Binary document extraction (PDF/DOCX) | **Not built** — refused with an explanation |
