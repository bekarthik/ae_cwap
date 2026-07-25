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
from cwap_common.idempotency import claim_dispatch, commit_step_output
from cwap_common.logbus import log_bus
from cwap_common.models import Run, WorkflowExecutionState
from cwap_common.settings import get_settings
from cwap_contracts.v4 import (
    ExecutionState,
    JobContext,
    LogLevel,
    NodeType,
    StepOutputContext,
    WorkflowGraph,
    WorkflowJobPayload,
)
from llm_proxy import store
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

#: A workflow's memory is a few sentences of context, not a transcript. A whole
#: report pasted into every future run's prompt would crowd out the actual task.
OUTCOME_EXCERPT = 600

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

        dispatched = 0
        for step in planned.next_steps:
            if step.terminal:
                continue
            if self._dispatch(graph, payload, context, step, node_type=node_type):
                dispatched += 1

        # Nothing left to start *from here*. That is not the same as the run
        # being over: a sibling branch may still be working, and a join this
        # step could not open will be opened by whoever finishes last.
        if dispatched == 0 and not self._work_remains(graph, payload.run_id):
            self._succeed(payload, context, node_type=node_type)

    def _dispatch(
        self,
        graph: WorkflowGraph,
        payload: WorkflowJobPayload,
        context: RunContext,
        step,
        *,
        node_type: NodeType,
    ) -> bool:
        """Enqueue one successor, if it is this worker's to enqueue.

        Two gates, both of which only matter once a workflow can fan out. A node
        with several inbound edges is a join and must not start until every one
        of them has arrived — otherwise it runs on half its inputs. And two
        branches finishing at the same moment both see a join ready, so exactly
        one of them wins the right to enqueue it; without that claim the join
        would run twice, under two step ids that the state row's own uniqueness
        cannot collapse.
        """
        target_id = step.target_node_id
        joining = bool(target_id) and len(graph.incoming(target_id)) > 1
        if joining and not self._join_ready(graph, payload.run_id, target_id):
            return False

        if joining:
            # A join is fed by every path into it, not just the one this worker
            # happens to be finishing. Its bindings are the union of all of them,
            # resolved against freshly committed state — the sibling's output may
            # have landed after this worker started, and the whole point of the
            # step is that it sees both.
            step = step.model_copy(
                update={"input_bindings": _joined_bindings(graph, target_id)}
            )
            context = self._rebuild_context(payload.run_id, self._inputs_for(payload.run_id))

        try:
            inputs = resolve_bindings(step.input_bindings, context)
        except BindingError as exc:
            self._fail(payload, str(exc))
            return False

        if joining:
            with unit_of_work() as session:
                if not claim_dispatch(session, payload.run_id, target_id):
                    return False

        next_payload = build_payload(
            graph=graph,
            run_id=payload.run_id,
            job_context=payload.job_context,
            current_state=result_state_for(node_type),
            next_step=step,
            inputs=inputs,
        )
        self.producer.publish(next_payload)
        return True

    @staticmethod
    def _inputs_for(run_id: str) -> dict[str, Any]:
        with read_only_session() as session:
            run = session.get(Run, run_id)
            return dict(run.inputs or {}) if run is not None else {}

    def _join_ready(self, graph: WorkflowGraph, run_id: str, node_id: str) -> bool:
        """Whether every path into this node has arrived.

        A branch's untaken side never arrives, so an edge whose source resolved
        the other way is treated as closed rather than pending — waiting for it
        would hang the run on a path the workflow deliberately did not take.
        """
        incoming = graph.incoming(node_id)
        if len(incoming) <= 1:
            return True

        completed, decisions = self._run_state(run_id)
        for edge in incoming:
            if edge.source in completed:
                continue
            if _is_closed(graph, edge.source, completed, decisions):
                continue
            return False
        return True

    def _work_remains(self, graph: WorkflowGraph, run_id: str) -> bool:
        """Whether any step of this run is still going or still to come.

        Asked when a path ends, to tell "this branch is done" from "the run is
        done". Read from committed state rather than from a counter, for the
        same reason run context is: a worker that restarts mid-run must reach
        the same answer as one that did not.
        """
        completed, decisions = self._run_state(run_id)
        for node in graph.nodes:
            if node.id in completed or node.type is NodeType.BRANCH:
                continue
            if _is_closed(graph, node.id, completed, decisions):
                continue
            # Every path into it has arrived and it has not run: it is either in
            # flight or about to be.
            if all(
                edge.source in completed or _is_closed(graph, edge.source, completed, decisions)
                for edge in graph.incoming(node.id)
            ):
                return True
        return False

    @staticmethod
    def _run_state(run_id: str) -> tuple[set[str], dict[str, bool]]:
        """Which nodes have finished, and which way each decision went.

        Both read from committed state rather than held in memory, for the same
        reason run context is: a worker that picks up a run someone else started
        has to reach the same answer as the one that started it.
        """
        completed: set[str] = set()
        decisions: dict[str, bool] = {}
        with read_only_session() as session:
            rows = (
                session.query(WorkflowExecutionState)
                .filter(WorkflowExecutionState.run_id == run_id)
                .all()
            )
            for row in rows:
                completed.add(row.node_id)
                verdict = (row.output or {}).get("decision")
                if isinstance(verdict, bool):
                    decisions[row.node_id] = verdict
        return completed, decisions

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

        self._remember_outcome(payload, result)

        log_bus.emit(
            payload.run_id,
            "run.succeeded",
            message="workflow completed",
            data={"result": result, "final_node": payload.node_id},
        )
        log_bus.forget(payload.run_id)

    def _remember_outcome(self, payload: WorkflowJobPayload, result: Any) -> None:
        """Write what this run produced into the workflow's own memory.

        The third memory layer was half-built: every agent step *read* workflow
        memory into its prompt, and nothing ever wrote to it, so the layer could
        only ever hold what a person typed by hand. A workflow that remembers
        its own outcomes is the difference between "recent runs are context" as
        a description and as a feature.

        Only on success, and only the answer: a failed run's output is a symptom
        rather than a lesson, and storing one would teach the next run to repeat
        it. Failures already reach the agent and skill layers, which is where
        the actionable part of a failure lives.
        """
        # Imported here rather than at module scope: the agent runtime reaches
        # back into the orchestrator's executors, and a top-level import would
        # close that circle.
        from agents.runtime import remember_for_workflow  # noqa: PLC0415

        graph = self._graph_for(payload.run_id)
        if graph is None:
            return

        text = _condense_result(result)
        if not text:
            return

        try:
            with store.acting_for(payload.job_context.tenant_id):
                remember_for_workflow(
                    payload.job_context.tenant_id,
                    graph.memory_scope,
                    text,
                    run_id=payload.run_id,
                )
        except Exception:  # noqa: BLE001 - a lesson is never worth a failed run
            logger.warning("could not record the run outcome", exc_info=True)

    @staticmethod
    def _graph_for(run_id: str) -> WorkflowGraph | None:
        with read_only_session() as session:
            run = session.get(Run, run_id)
            if run is None:
                return None
            try:
                return WorkflowGraph.model_validate(run.graph_snapshot)
            except Exception:  # noqa: BLE001 - a stored graph that will not parse
                return None

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


def _joined_bindings(graph: WorkflowGraph, node_id: str) -> dict[str, str]:
    """Every value a join is fed, from every path into it.

    Later edges win a name collision, which is arbitrary and also unambiguous —
    two paths binding the same name to different values is a workflow that has
    not decided what it means, and the canvas shows both edges plainly.
    """
    bindings: dict[str, str] = {}
    for edge in graph.incoming(node_id):
        bindings.update(edge.bindings)
    return bindings


def _is_closed(
    graph: WorkflowGraph,
    node_id: str,
    completed: set[str],
    decisions: dict[str, bool],
) -> bool:
    """Whether this node can never run, because the path to it was not taken.

    Only a decision closes a path: its untaken side leads to nodes that will
    never arrive, and a join that waited for one of them would hang the run on a
    branch the workflow deliberately did not choose. Everything else has either
    run, is running, or is waiting on something that will come.
    """
    incoming = graph.incoming(node_id)
    if not incoming:
        return False

    for edge in incoming:
        source = graph.node(edge.source)
        if source.type is NodeType.BRANCH and edge.condition is not None:
            taken = decisions.get(source.id)
            if taken is not None and taken is not edge.condition:
                continue  # this path was closed by the decision
        if edge.source in completed:
            return False
        if not _is_closed(graph, edge.source, completed, decisions):
            return False
    return True


def _condense_result(result: Any) -> str:
    """The run's answer, short enough to be context rather than a transcript."""
    if isinstance(result, dict):
        result = result.get("result", result)
    text = result if isinstance(result, str) else str(result or "")
    condensed = " ".join(text.split())
    if not condensed:
        return ""
    if len(condensed) > OUTCOME_EXCERPT:
        condensed = condensed[:OUTCOME_EXCERPT].rstrip() + "…"
    return f"A previous run produced: {condensed}"
