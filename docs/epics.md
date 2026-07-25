# Epics and user stories — traceability

Each epic and user story from the brief, what was built, and where to verify it.

---

## Epic 1 · Intelligent Requirement Diagnosis & Setup

**Brief:** NLP intake, skill gap analysis, agent recommendation, automatic
scaffolding. Outcome: a proposed workflow structure presented before the user
touches the canvas.

| Component (brief) | Built as | Output contract |
| --- | --- | --- |
| Goal Intake Endpoint | `POST /api/diagnose` → `nlp.diagnose` | `DiagnosisResult` — intent, required steps, suggestions, confidence, **gaps** |
| Skill Scaffolding Logic | `nlp.build_graph` | `WorkflowGraph`, validated on construction |

### User Story 2 — goal input

> As a beginner workflow creator, I want to input a simple, high-level goal, so
> that the system automatically researches and presents a recommended initial
> sequence of agents and skills.

Type a goal in **Start from a goal**. The modal shows the intent, the ordered
skills with a rationale for each, a confidence score, anything it could not wire
up, and suggestions — *then* offers to place the draft on the canvas. The user
reviews the reasoning before the graph appears, not after.

Three properties beyond the brief, each a deliberate choice:

- **Deterministic.** The same goal always produces the same plan, so onboarding
  is testable and a user retyping their goal does not get a different workflow.
- **Honest about gaps.** A goal mentioning private documents that have not been
  uploaded, or an external API, produces a `gap` rather than a node that would
  fail at run time.
- **Generated prompts match their wiring.** Each prompt is composed from the
  bindings the node will actually receive, so a scaffold cannot emit
  `{{context}}` when nothing supplies it.

**Verify:** `tests/test_diagnosis.py` (17 tests), and
`TestScaffoldedWorkflowsRun` in `tests/test_execution.py`, which executes six
scaffolded workflows end to end.

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

Seven step types in the palette, each with a plain-language description. Drawing
a connection pre-fills a sensible parameter mapping for the two node types being
joined, so a new arrow is useful immediately. Decision nodes expose two labelled
handles (green "yes", red "no"). Selecting a connection opens its parameter
mapping for editing.

Conditional flow is real, not cosmetic: the canvas writes `condition: true/false`
onto the edge, the contract validates that a decision node has exactly one of
each, and the orchestrator resolves the decision at plan time and records why.

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

**Verify:** `tests/test_knowledge.py` (17 tests, including tenant isolation).

---

## Epic 4 · Execution & Monitoring Engine

**Brief:** a Run button with clear command execution, real-time log streaming of
actions, state changes, errors and output at every node. Outcome: a transparent
report letting the user trace *why* the AI arrived at an answer.

| Component (brief) | Built as | Output contract |
| --- | --- | --- |
| Execution Orchestrator | `orchestrator.Worker` + `state_machine` | `StepOutputContext` per node |
| Real-time Log Streamer | `cwap_common.logbus` + `WS /api/runs/{id}/logs` | `LogEvent` — timestamp, node, level, event, data |

Pressing **Run** returns in milliseconds with a run id; execution happens on the
worker. The canvas then shows each node pulsing as it runs and settling green or
red, while the log streams underneath.

The report at `GET /api/runs/{id}` is the trace: every step with its output, the
state it reached, how long it took, and the full ordered log. A branching run
includes the decision node's own entry, so the answer to "why did it go that
way?" is in the record rather than inferred.

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
| Goal → diagnosis → scaffolded graph | Built, deterministic, gap-reporting |
| Drag-and-drop canvas with conditional edges | Built |
| Parameter mapping between steps | Built, with type-aware defaults |
| Document upload → embeddings → RAG node | Built, tenant-scoped, previewable |
| Run + real-time logs + step report | Built |
| Versioned contract registry with a change gate | Built |
| Dual validation/authorisation gateway + DLQ | Built |
| Idempotency, 2PC external calls, transactions | Built |
| Redis/Celery transport | Built (abstraction + Celery entry point; in-memory is the default) |
| Any model backend — local open-weight, self-hosted, or hosted | Built, capability-aware |
| Embeddings from any OpenAI-compatible endpoint | Built; hashing is the offline default |
| PostgreSQL | Supported (JSONB variants); SQLite is the default |
| Parallel step execution | **Not built** — only decision branching |
| Token-by-token streaming from a model | **Not built** — steps are request/response |
| Schema migrations | **Not built** — `init_db()` only |
| Binary document extraction (PDF/DOCX) | **Not built** — refused with an explanation |
