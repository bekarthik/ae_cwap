"""The state machine that turns a saved JSON graph into a directed run.

The orchestrator never lets a worker choose where to go next. Pathing is derived
here, from the immutable saved graph plus the run's own recorded state, and is
written into `next_step_definition` before the job is ever enqueued.

Branch nodes are resolved *here*, during planning, rather than dispatched as
their own queued jobs. A branch performs no external work — it only compares
values already in run context — so evaluating it at plan time is what keeps
`next_step_definition` a single, concrete, deterministic successor. That is the
property mandate §1.5 is asking for.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from cwap_contracts.v2 import (
    NODE_TYPE_RESULT_STATE,
    NODE_TYPE_SERVICE,
    ExecutionState,
    JobContext,
    NextStepDefinition,
    NodeType,
    ServiceName,
    StepOutputContext,
    WorkflowGraph,
    WorkflowJobPayload,
)

from orchestrator.executors import (
    ExecutionOutcome,
    ExecutionRequest,
    NodeExecutionError,
    evaluate_branch,
)
from orchestrator.variables import BindingError, RunContext, resolve_bindings

#: Guards a pathological chain of branch-to-branch edges. The graph is a DAG so
#: this cannot loop forever, but a wide fan of decisions still deserves a bound.
MAX_INLINE_BRANCHES = 16


@dataclass
class PlannedStep:
    """The next dispatchable step, plus any branches resolved to reach it."""

    next_step: NextStepDefinition
    #: Branch decisions taken while planning — each becomes its own state row so
    #: the run report shows *why* the run took the path it did.
    branch_outputs: list[StepOutputContext] = field(default_factory=list)


def new_step_execution_id() -> str:
    return f"se_{uuid.uuid4().hex[:20]}"


def initial_step(graph: WorkflowGraph) -> NextStepDefinition:
    """Where every run of this workflow begins."""
    entry = graph.entry_node()
    return NextStepDefinition(
        target_service=NODE_TYPE_SERVICE[entry.type],
        target_node_id=entry.id,
        input_bindings={},
        terminal=False,
    )


def plan_next(
    graph: WorkflowGraph,
    *,
    from_node_id: str,
    context: RunContext,
    run_id: str,
    tenant_id: str,
    emit: Callable[..., None],
) -> PlannedStep:
    """Decide what runs after `from_node_id` completes."""
    branch_outputs: list[StepOutputContext] = []

    outgoing = graph.outgoing(from_node_id)
    if not outgoing:
        return PlannedStep(
            next_step=NextStepDefinition(
                target_service=ServiceName.TERMINAL, target_node_id=None, terminal=True
            )
        )

    # A non-branch node has at most one outgoing edge (enforced by WorkflowGraph),
    # so following edge zero is the whole of deterministic pathing.
    edge = outgoing[0]

    for hop in range(MAX_INLINE_BRANCHES + 1):
        target = graph.node(edge.target)

        if target.type is not NodeType.BRANCH:
            return PlannedStep(
                next_step=NextStepDefinition(
                    target_service=NODE_TYPE_SERVICE[target.type],
                    target_node_id=target.id,
                    input_bindings=dict(edge.bindings),
                    terminal=False,
                ),
                branch_outputs=branch_outputs,
            )

        if hop == MAX_INLINE_BRANCHES:
            break

        outcome, output_context = _resolve_branch(
            graph,
            node_id=target.id,
            edge_bindings=edge.bindings,
            context=context,
            run_id=run_id,
            tenant_id=tenant_id,
            emit=emit,
        )
        branch_outputs.append(output_context)
        context.record(target.id, outcome.output)
        edge = _edge_for_decision(graph, target.id, bool(outcome.branch))

    raise NodeExecutionError(
        f"more than {MAX_INLINE_BRANCHES} chained decision nodes after '{from_node_id}'; "
        "simplify the workflow"
    )


def _resolve_branch(
    graph: WorkflowGraph,
    *,
    node_id: str,
    edge_bindings: dict[str, str],
    context: RunContext,
    run_id: str,
    tenant_id: str,
    emit: Callable[..., None],
) -> tuple[ExecutionOutcome, StepOutputContext]:
    node = graph.node(node_id)
    step_execution_id = new_step_execution_id()

    try:
        inputs = resolve_bindings(edge_bindings, context)
    except BindingError as exc:
        raise NodeExecutionError(f"decision node '{node_id}': {exc}") from exc

    outcome = evaluate_branch(
        ExecutionRequest(
            node=node,
            inputs=inputs,
            context=context,
            run_id=run_id,
            step_execution_id=step_execution_id,
            tenant_id=tenant_id,
            emit=emit,
        )
    )

    return outcome, StepOutputContext(
        run_id=run_id,
        step_execution_id=step_execution_id,
        source_service=ServiceName.BRANCH_EVALUATOR,
        node_id=node_id,
        resulting_state=NODE_TYPE_RESULT_STATE[NodeType.BRANCH],
        output=outcome.output,
        derived_context=outcome.derived_context,
    )


def _edge_for_decision(graph: WorkflowGraph, node_id: str, decision: bool):
    for edge in graph.outgoing(node_id):
        if edge.condition is decision:
            return edge
    # Unreachable for a validated graph: WorkflowGraph rejects a branch that
    # lacks exactly one true and one false edge.
    raise NodeExecutionError(  # pragma: no cover
        f"decision node '{node_id}' has no edge for the {decision} path"
    )


def build_payload(
    *,
    graph: WorkflowGraph,
    run_id: str,
    job_context: JobContext,
    current_state: ExecutionState,
    next_step: NextStepDefinition,
    inputs: dict,
    attempt: int = 1,
) -> WorkflowJobPayload:
    """Assemble the queue message for the next step.

    `node_id` is taken from `next_step.target_node_id` so the executed node and
    the authorised transition can never disagree.
    """
    if next_step.terminal or next_step.target_node_id is None:
        raise NodeExecutionError("cannot build a job payload for a terminal step")

    return WorkflowJobPayload(
        run_id=run_id,
        step_execution_id=new_step_execution_id(),
        workflow_id=graph.id,
        node_id=next_step.target_node_id,
        job_context=job_context,
        current_state=current_state,
        next_step_definition=next_step,
        inputs=inputs,
        attempt=attempt,
    )


def result_state_for(node_type: NodeType) -> ExecutionState:
    return NODE_TYPE_RESULT_STATE[node_type]
