# The Workflow Contract & Resilience Layer

Each clause of the architectural mandate, where it is implemented, and the test
that proves it. If a row has no test, treat it as unimplemented.

---

## 1 · The Execution Contract

> The canonical payload must be an operational contract dictating not only what
> the inputs are, but how and when the workflow should proceed.

| Requirement | Implementation | Proof |
| --- | --- | --- |
| Centrally versioned registry | `cwap_contracts.registry` — `Name@version` → model | `test_contracts.py::TestRegistryLock` |
| Change requires a version bump | JSON-Schema fingerprints in `contracts.lock.json` | `test_every_registered_contract_matches_the_approved_lock` |
| Versions coexist during a rollout | `v1` and `v2` both registered and resolvable | `test_both_versions_stay_registered` |
| `run_id` | `WorkflowJobPayload.run_id` | `test_structurally_mandatory_fields` |
| `step_execution_id` | `WorkflowJobPayload.step_execution_id` | ↑ |
| `job_context` with tenant, user, permission scope | `JobContext` + `PermissionRequirement` | `TestJobContext` |
| `current_state` | `ExecutionState`, the state left by the previous step | `TestTransitionTable` |
| `next_step_definition` | `NextStepDefinition` — target service, node, bindings | `TestNextStepDefinition` |
| `current_state → next_step_definition` consistency | `LEGAL_TRANSITIONS` + a model validator | `test_terminal_state_may_not_dispatch_more_work` |
| Pydantic models are the mandatory type at every boundary | `ProducerWrapper` / `ConsumerWrapper` accept and return models only | `test_contract_gateway.py` |

The transition guard makes an illegal path **unrepresentable** rather than
merely discouraged: a payload whose `current_state` is terminal cannot be
constructed with a dispatchable `target_service`, so it fails at
construction — before it is serialised, before it is enqueued, before a worker
ever sees it.

Two supporting choices:

- `extra="forbid"` — an unrecognised field is a contract violation, not something
  silently dropped. That is the whole point of a closed schema.
- `frozen=True` — a payload cannot be mutated after it crosses a boundary, so a
  worker can never "fix up" a message in place and hide a schema drift.

### Adding agents was a version bump, not an edit

Agent nodes, skills, memory and the design conversation arrived as **v2**, added
alongside v1 rather than modifying it. That is the gate working as intended: the
fingerprint lock made changing `WorkflowGraph` in place impossible without an
approved bump, so the change had to be a new version — which is also what lets a
deployment mid-rollout hold workers on both.

Two contracts are deliberately narrower than they could be:

- **`SkillProposal` cannot carry an id, a tenant, an origin or usage counters.**
  A synthesised skill is therefore structurally incapable of claiming to be a
  built-in or of overwriting another skill's history — the registry assigns all
  of those. The narrowing *is* the security property.
- **`WorkflowNode` of type `agent` must carry an `agent_id`.** An agent step with
  nobody to run it is rejected at construction rather than failing mid-run.

---

## 2 · The Validation & Security Enforcement Layer

> All message flow must pass two mandatory checks at both the Producer and
> Consumer boundaries: schema validation and authorisation context enforcement.

### 2.A Producer wrapper

`cwap_common.contract_gateway.ProducerWrapper.publish` performs, in order:

1. **Schema validation** — re-validated even when handed a model instance, so a
   caller cannot smuggle through a subclass or a stale version.
2. **Authorisation pre-check** — the calling identity is checked against
   `job_context` for the operation named in `next_step_definition`.
3. **Only then** does the payload become a message.

A rejected job leaves no trace on the queue at all.

| Proof | |
| --- | --- |
| `test_invalid_payload_never_becomes_a_message` | queue depth stays 0 |
| `test_unauthorised_payload_never_becomes_a_message` | queue depth stays 0 |
| `test_unknown_principal_is_rejected` | non-member of the tenant |

### 2.B Consumer wrapper

`ConsumerWrapper.next` re-runs both checks at the moment of execution. The
runtime authorisation check is the one that matters: a job can sit in a queue for
minutes, and permissions may have been revoked in the meantime.

| Proof | |
| --- | --- |
| `test_revoked_permission_halts_the_job_at_consumption` | authorised at enqueue, revoked before consumption, dead-lettered |
| `test_valid_message_is_returned_as_a_model` | workers never see an unvalidated dict |

### 2.C Dead Letter Queue

Neither failure is retried — a contract violation is deterministic, and an
authorisation revocation is intentional. Both park the message with its precise
reason and let the main queue keep flowing.

`DeadLetterSink.park` writes to two places on purpose: the DLQ topic (so operator
tooling can watch it) and the `dead_letters` table (so triage and resubmission
have a durable record). `/api/admin/dead-letters` lists them and resubmits;
resubmitted messages go back through the consumer wrapper, so a message that is
still invalid is simply parked again.

| Proof | |
| --- | --- |
| `test_malformed_message_is_dead_lettered_not_raised` | reason code recorded |
| `test_main_queue_keeps_flowing_past_a_poisoned_message` | no worker starvation |
| `test_dead_letter_preserves_the_original_payload_for_resubmission` | triage is possible |

---

## 3 · Execution Resilience Core

### 3.A External APIs — two-phase commit check

`cwap_common.idempotency.IdempotencyGate`, used by the HTTP executor:

1. **PRE-CHECK** — claim `(run_id, step_execution_id, operation)` atomically. If
   a prior attempt already succeeded, return its recorded response and **do not
   call outward**.
2. **EXECUTE & COMMIT** — perform the call, then write the outcome before the job
   is acknowledged.

A failed call calls `abandon()`, releasing the claim so a genuine retry may
proceed. A concurrent worker holding an in-flight claim is told to stand down
rather than racing.

| Proof | |
| --- | --- |
| `test_completed_operation_replays_instead_of_calling_again` | no double side effect |
| `test_second_caller_while_in_flight_is_told_to_stand_down` | no concurrent double-call |
| `test_commit_is_guarded_and_cannot_clobber_a_resolved_claim` | `WHERE status = IN_FLIGHT` |
| `test_abandoned_claim_lets_a_genuine_retry_proceed` | transient failures recover |

### 3.B Database state — atomic, non-destructive writes

`workflow_execution_state` enforces
`UNIQUE(run_id, step_execution_id, source_service)`, and `commit_step_output`
writes with `ON CONFLICT DO NOTHING`. However many times a worker retries step X
of run Y, the first successful write commits and the original output stands.

Run status transitions use a guarded update
(`WHERE status NOT IN (SUCCEEDED, FAILED)`), so a late duplicate cannot re-open
or overwrite a finished run.

| Proof | |
| --- | --- |
| `test_retry_is_a_no_op_and_the_original_output_stands` | first write wins |
| `test_a_redelivered_job_does_not_duplicate_state` | at the worker level |
| `test_a_job_for_a_finished_run_is_ignored` | terminal runs stay terminal |

### 3.C Transactional boundaries

`cwap_common.db.unit_of_work` is the only sanctioned way to open a write
transaction, so "did this path remember to be atomic?" is answerable by grep. One
node execution commits its state row, its log entries and the run's progress
counter together — or rolls all of them back.

| Proof | |
| --- | --- |
| `test_a_failure_mid_step_rolls_back_every_write` | no partial state, no orphaned logs |

**An agent is one node, so its whole loop is inside one transaction.** A
redelivered job re-runs the entire agent rather than resuming half of one. That
keeps this boundary intact without inventing a resumption protocol for the loop,
and the cost is bounded: any external call a skill makes still passes through
§3.A's gate, so a re-run cannot fire a side effect twice.

---

## What the mandate does not cover, and is still true

- **Least privilege.** A workflow containing a step that can act on the outside
  world requires the `WRITE_EXTERNAL` scope, which is not granted at
  registration. `SERVICE_REQUIRES_WRITE` enforces it at both gateway boundaries.
- **Tenant isolation on knowledge.** A `knowledge_handle` is an identifier, never
  an authorisation. Retrieval is scoped to the calling tenant on every read.
- **Tenant and scope isolation on memory.** Every read and write is filtered by
  tenant, scope and scope id, so one agent's lessons cannot reach another's
  prompt and one tenant's memory is unreachable from another's. Proved in
  `test_memory.py::TestRememberAndRecall`.
- **Agents get no privilege their user lacks.** A skill that calls out requires
  the run to carry a write scope *and* the host to be allow-listed, exactly as an
  HTTP node does. A denial is reported to the agent as a failed skill so it can
  finish another way, rather than killing the run. Proved in
  `test_skills.py::TestHttpSkillsAreGuarded`.
- **Synthesis cannot widen what the platform can do.** `_validate_proposal`
  refuses an HTTP skill outright and refuses a retrieval over a corpus the tenant
  does not own. If synthesis emitted code, a model that had read a hostile
  document could write arbitrary logic into a privileged worker and every check
  above would become bypassable. Proved in
  `test_skills.py::test_synthesis_may_never_produce_a_network_call`.
- **Credential redaction.** The log feed deliberately records external call
  payloads, so `logbus._scrub` redacts credential-shaped keys at the single point
  every event passes through.
- **Cross-tenant reads return 404, not 403.** A 403 would confirm the resource
  exists, leaking the id space.

## Changing a contract

```bash
# 1. Add the new version alongside the old one in cwap_contracts/vN/
# 2. Register it
# 3. Re-approve the lock — in the same reviewed commit as the change
make contracts
```

`tests/test_contracts.py` fails if the lock and the live schemas disagree, so the
only way to change a published contract's shape is deliberately and in review.
