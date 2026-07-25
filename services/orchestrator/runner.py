"""Execution & Monitoring Engine (Epic 4).

`start_run` accepts a run and returns immediately — the API never blocks on
execution. A `Worker` consumes jobs off the queue, executes exactly the node its
payload names, commits state idempotently, and enqueues the successor the state
machine chose.

Transaction discipline (mandate §3.C): the step's state row, its log entries and
the run's progress counter are written in one `unit_of_work`. If anything in the
node's processing fails, all of it rolls back together and the run stays at its
last consistent state.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from cwap_common.contract_gateway import ConsumerWrapper, ProducerWrapper
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.idempotency import commit_step_output
from cwap_common.logbus import log_bus
from cwap_common.models import Run, WorkflowExecutionState
from cwap_common.settings import get_settings
from cwap_contracts.v3 import (
    ExecutionState,
    JobContext,
    LogLevel,
    NodeType,
    StepOutputContext,
    WorkflowGraph,
    WorkflowJobPayload,
)
from llm_proxy.store import acting_for

from orchestrator.executors import ExecutionRequest, NodeExecutionError, executor_for
from orchestrator.state_machine import (
    build_payload,
    initial_step,
    plan_next,
    result_state_for,
)
from orchestrator.variables import BindingError, RunContext, resolve_bindings

logger = logging.getLogger("cwap.worker")

RUN_PENDING = "PENDING"
RUN_RUNNING = "RUNNING"
RUN_SUCCEEDED = "SUCCEEDED"
RUN_FAILED = "FAILED"
TERMINAL_RUN_STATUSES = frozenset({RUN_SUCCEEDED, RUN_FAILED})


@dataclass
class RunHandle:
    run_id: str
    workflow_id: str
    status: str


def start_run(
    *,
    graph: WorkflowGraph,
    job_context: JobContext,
    inputs: dict[str, Any] | None = None,
    producer: ProducerWrapper | None = None,
) -> RunHandle:
    """Accept a run and enqueue its first step.

    The graph is snapshotted into the run row. A workflow edited mid-run
    therefore cannot change what that run executes — replaying the run report
    later shows the graph that actually ran.
    """
    producer = producer or ProducerWrapper()
    run_id = f"run_{uuid.uuid4().hex[:20]}"
    resolved_inputs = dict(inputs or {})

    with unit_of_work() as session:
        session.add(
            Run(
                id=run_id,
                workflow_id=graph.id,
                tenant_id=job_context.tenant_id,
                initiating_user_id=job_context.initiating_user_id,
                trace_id=job_context.trace_id,
                status=RUN_PENDING,
                graph_snapshot=graph.model_dump(mode="json"),
                inputs=resolved_inputs,
            )
        )

    log_bus.emit(
        run_id,
        "run.accepted",
        message=f"run accepted for workflow '{graph.name}'",
        data={"workflow_id": graph.id, "node_count": len(graph.nodes)},
    )

    first = initial_step(graph)
    payload = build_payload(
        graph=graph,
        run_id=run_id,
        job_context=job_context,
        current_state=ExecutionState.RUN_ACCEPTED,
        next_step=first,
        inputs=resolved_inputs,
    )

    # Producer wrapper validates the contract and authorises the transition
    # before anything reaches the broker. A rejected run never becomes a message.
    producer.publish(payload)
    return RunHandle(run_id=run_id, workflow_id=graph.id, status=RUN_PENDING)


class Worker:
    """Consumes validated jobs and drives one step of a run per message."""

    def __init__(
        self,
        *,
        consumer: ConsumerWrapper | None = None,
        producer: ProducerWrapper | None = None,
    ) -> None:
        self.consumer = consumer or ConsumerWrapper()
        self.producer = producer or ProducerWrapper()

    # ---- loop ----------------------------------------------------------

    def run_forever(self, *, poll_timeout: float = 1.0) -> None:  # pragma: no cover - daemon
        while True:
            try:
                self.poll(timeout=poll_timeout)
            except Exception:
                # A worker must not die. `process` already turns a failing step
                # into a failed run, so reaching here means something outside a
                # step went wrong — a broker hiccup, a database blip. Losing the
                # loop over it would take every queued run down with it, and the
                # runs would sit at PENDING with nothing to explain why.
                logger.exception("worker loop iteration failed; continuing")

    def poll(self, *, timeout: float = 0.0) -> bool:
        """Handle at most one message. Returns True if a job was executed."""
        payload = self.consumer.next(timeout=timeout)
        if payload is None:
            return False
        self.process(payload)
        return True

    def drain(self, *, max_messages: int = 1000) -> int:
        """Process until the queue is empty. Used by dev mode and the tests."""
        handled = 0
        while handled < max_messages and self.poll():
            handled += 1
        return handled

    # ---- one step ------------------------------------------------------

    def process(self, payload: WorkflowJobPayload) -> None:
        # Every model call under this step resolves the *job owner's* chosen
        # backend, not the worker process's. Scoped to one step so a fungible
        # worker moving to another tenant's job cannot inherit the previous
        # tenant's endpoint or credential.
        with acting_for(payload.job_context.tenant_id):
            self._process(payload)

    def _process(self, payload: WorkflowJobPayload) -> None:
        run = self._load_run(payload.run_id)
        if run is None:
            log_bus.emit(
                payload.run_id,
                "run.unknown",
                level=LogLevel.ERROR,
                message="job referenced a run that does not exist; dropping",
            )
            return
        if run["status"] in TERMINAL_RUN_STATUSES:
            log_bus.emit(
                payload.run_id,
                "step.skipped",
                level=LogLevel.WARN,
                node=payload.node_id,
                message=f"run already {run['status']}; ignoring redelivered job",
            )
            return

        graph = WorkflowGraph.model_validate(run["graph_snapshot"])
        context = self._rebuild_context(payload.run_id, run["inputs"])
        node = graph.node(payload.node_id)

        settings = get_settings()
        if run["steps_executed"] >= settings.max_steps_per_run:
            self._fail(
                payload,
                f"run exceeded the {settings.max_steps_per_run}-step limit",
            )
            return

        def emit(event: str, *, message: str = "", level: LogLevel = LogLevel.INFO, data=None):
            log_bus.emit(
                payload.run_id,
                event,
                node=payload.node_id,
                level=level,
                message=message,
                data=data or {},
            )

        def stream(event: str, *, message: str = "", data=None):
            """Live-only: watched while the run happens, never written down."""
            log_bus.transient(
                payload.run_id,
                event,
                node=payload.node_id,
                message=message,
                data=data or {},
            )

        emit(
            "step.started",
            message=f"executing '{node.label or node.id}' ({node.type.value})",
            data={"step_execution_id": payload.step_execution_id, "attempt": payload.attempt},
        )

        started = time.perf_counter()
        try:
            inputs = dict(payload.inputs)
            outcome = executor_for(node.type)(
                ExecutionRequest(
                    node=node,
                    inputs=inputs,
                    context=context,
                    run_id=payload.run_id,
                    step_execution_id=payload.step_execution_id,
                    tenant_id=payload.job_context.tenant_id,
                    emit=emit,
                    stream=stream,
                    workflow_memory_scope=graph.memory_scope,
                    # An agent's skills get no privilege the run does not have:
                    # a write scope on the job is what permits reaching outward.
                    allow_side_effects=bool(payload.job_context.permissions.required_write),
                )
            )
        except (NodeExecutionError, BindingError) as exc:
            self._fail(payload, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - deliberately the last resort
            # Anything a node executor did not anticipate: a provider client
            # raising something new, a shape nobody expected. It used to
            # propagate, kill the standalone worker's loop, and leave the run at
            # PENDING forever — the one outcome a user cannot act on. A run that
            # will never finish has to say so, and name what happened.
            logger.exception("unexpected failure executing %s", node.id)
            self._fail(payload, f"{type(exc).__name__}: {exc}")
            return

        duration_ms = int((time.perf_counter() - started) * 1000)
        step_output = StepOutputContext(
            run_id=payload.run_id,
            step_execution_id=payload.step_execution_id,
            source_service=payload.next_step_definition.target_service,
            node_id=node.id,
            resulting_state=result_state_for(node.type),
            output=outcome.output,
            derived_context=outcome.derived_context,
            duration_ms=duration_ms,
        )

        # One transaction for the whole step: state row + progress counter, plus
        # the run's effective inputs when this was the entry node.
        with unit_of_work() as session:
            first_write = commit_step_output(session, step_output)
            if first_write:
                changes: dict[Any, Any] = {
                    Run.steps_executed: Run.steps_executed + 1,
                    Run.status: RUN_RUNNING,
                }
                if node.type is NodeType.INPUT:
                    # The entry node merges declared defaults with whatever the
                    # caller supplied; that merged set is what `$run.input.*`
                    # must mean for the rest of the run. Persisting it also
                    # means a worker that picks up a later step on another
                    # process reconstructs the same inputs.
                    changes[Run.inputs] = outcome.output
                    context.inputs = dict(outcome.output)
                session.query(Run).filter(Run.id == payload.run_id).update(changes)

        if not first_write:
            emit(
                "step.deduplicated",
                level=LogLevel.WARN,
                message="this step was already committed; keeping the original output",
            )

        emit(
            "step.completed",
            message=f"'{node.id}' finished in {duration_ms}ms",
            data={"state": step_output.resulting_state.value, "output": outcome.output},
        )
        context.record(node.id, outcome.output)

        self._dispatch_next(graph, payload, context, node_type=node.type, emit=emit)

    # ---- transitions ---------------------------------------------------

    def _dispatch_next(
        self,
        graph: WorkflowGraph,
        payload: WorkflowJobPayload,
        context: RunContext,
        *,
        node_type: NodeType,
        emit,
    ) -> None:
        try:
            planned = plan_next(
                graph,
                from_node_id=payload.node_id,
                context=context,
                run_id=payload.run_id,
                tenant_id=payload.job_context.tenant_id,
                emit=emit,
            )
        except (NodeExecutionError, BindingError) as exc:
            self._fail(payload, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - deliberately the last resort
            # Planning has the same obligation as execution: a run that cannot
            # continue must be told so rather than left at PENDING.
            logger.exception("unexpected failure planning after %s", payload.node_id)
            self._fail(payload, f"could not plan the next step — {type(exc).__name__}: {exc}")
            return

        # Persist any decision nodes resolved during planning, so the run report
        # explains the path taken.
        if planned.branch_outputs:
            with unit_of_work() as session:
                for branch_output in planned.branch_outputs:
                    commit_step_output(session, branch_output)

        if planned.next_step.terminal:
            self._succeed(payload, context, node_type=node_type)
            return

        try:
            inputs = resolve_bindings(planned.next_step.input_bindings, context)
        except BindingError as exc:
            self._fail(payload, str(exc))
            return

        next_payload = build_payload(
            graph=graph,
            run_id=payload.run_id,
            job_context=payload.job_context,
            current_state=result_state_for(node_type),
            next_step=planned.next_step,
            inputs=inputs,
        )
        self.producer.publish(next_payload)

    def _succeed(
        self, payload: WorkflowJobPayload, context: RunContext, *, node_type: NodeType
    ) -> None:
        result = context.previous_output
        with unit_of_work() as session:
            # Guarded update: a late duplicate cannot re-open or overwrite a
            # run that already reached a terminal status.
            session.query(Run).filter(
                Run.id == payload.run_id, Run.status.notin_(list(TERMINAL_RUN_STATUSES))
            ).update(
                {
                    Run.status: RUN_SUCCEEDED,
                    Run.result: result,
                    Run.finished_at: _now(),
                },
                synchronize_session=False,
            )

        log_bus.emit(
            payload.run_id,
            "run.succeeded",
            message="workflow completed",
            data={"result": result, "final_node": payload.node_id},
        )
        log_bus.forget(payload.run_id)

    def _fail(self, payload: WorkflowJobPayload, reason: str) -> None:
        with unit_of_work() as session:
            session.query(Run).filter(
                Run.id == payload.run_id, Run.status.notin_(list(TERMINAL_RUN_STATUSES))
            ).update(
                {Run.status: RUN_FAILED, Run.error: reason, Run.finished_at: _now()},
                synchronize_session=False,
            )

        log_bus.emit(
            payload.run_id,
            "run.failed",
            level=LogLevel.ERROR,
            node=payload.node_id,
            message=reason,
            data={"step_execution_id": payload.step_execution_id},
        )
        log_bus.forget(payload.run_id)

    # ---- state reconstruction -------------------------------------------

    @staticmethod
    def _load_run(run_id: str) -> dict[str, Any] | None:
        with read_only_session() as session:
            run = session.get(Run, run_id)
            if run is None:
                return None
            return {
                "status": run.status,
                "graph_snapshot": run.graph_snapshot,
                "inputs": run.inputs or {},
                "steps_executed": run.steps_executed,
            }

    @staticmethod
    def _rebuild_context(run_id: str, inputs: dict[str, Any]) -> RunContext:
        """Rebuild run state from committed step rows, not from worker memory.

        This is what makes the worker restartable: a job redelivered to a
        different process sees exactly the same context the original would have.
        """
        context = RunContext(run_id=run_id, inputs=dict(inputs))
        with read_only_session() as session:
            # Materialise inside the session: the session closes on exit, and
            # detached ORM instances cannot lazily refresh their columns.
            rows = [
                (row.node_id, row.output or {})
                for row in session.query(WorkflowExecutionState)
                .filter(WorkflowExecutionState.run_id == run_id)
                .order_by(WorkflowExecutionState.id)
                .all()
            ]
        for node_id, output in rows:
            context.record(node_id, output)
        return context


def _now():
    from datetime import datetime, timezone  # noqa: PLC0415 - narrow use

    return datetime.now(timezone.utc)


def run_to_completion(
    *,
    graph: WorkflowGraph,
    job_context: JobContext,
    inputs: dict[str, Any] | None = None,
    max_steps: int = 100,
) -> str:
    """Start a run and drive it synchronously to a terminal state.

    Used by dev mode and by the execution tests. Production runs the same
    `Worker` as a separate process against Redis — this just pumps the loop
    in-process so there is no infrastructure to stand up.
    """
    handle = start_run(graph=graph, job_context=job_context, inputs=inputs)
    Worker().drain(max_messages=max_steps)
    return handle.run_id


__all__ = [
    "RUN_FAILED",
    "RUN_PENDING",
    "RUN_RUNNING",
    "RUN_SUCCEEDED",
    "TERMINAL_RUN_STATUSES",
    "RunHandle",
    "Worker",
    "run_to_completion",
    "start_run",
]
