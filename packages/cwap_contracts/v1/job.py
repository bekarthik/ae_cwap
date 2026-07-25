"""The Execution Contract — `WorkflowJobPayload`.

Mandate §1: this is not merely a data structure, it is an *operational
contract*. It states what the inputs are, what state the run reached, and —
critically — what the single legal next action is. A worker cannot invent a
path; it can only follow `next_step_definition`, and that field is itself
validated against `current_state` before the message is ever enqueued.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from cwap_contracts.v1.base import CONTRACT_VERSION, ContractModel, utcnow
from cwap_contracts.v1.graph import NodeType
from cwap_contracts.v1.identity import JobContext


class ServiceName(str, Enum):
    """Every destination a job batch may be routed to."""

    ORCHESTRATOR = "ORCHESTRATOR"
    LLM_PROXY = "LLM_PROXY"
    RAG_SEARCH = "RAG_SEARCH"
    HTTP_CONNECTOR = "HTTP_CONNECTOR"
    TRANSFORM_ENGINE = "TRANSFORM_ENGINE"
    BRANCH_EVALUATOR = "BRANCH_EVALUATOR"
    TERMINAL = "TERMINAL"


class ExecutionState(str, Enum):
    """The derived state achieved by the *previously completed* step."""

    RUN_ACCEPTED = "RUN_ACCEPTED"
    INPUT_RESOLVED = "INPUT_RESOLVED"
    LLM_INFERENCE_COMPLETE = "LLM_INFERENCE_COMPLETE"
    RAG_CONTEXT_LOADED = "RAG_CONTEXT_LOADED"
    HTTP_CALL_COMPLETE = "HTTP_CALL_COMPLETE"
    TRANSFORM_COMPLETE = "TRANSFORM_COMPLETE"
    BRANCH_EVALUATED = "BRANCH_EVALUATED"
    DATA_VALIDATED = "DATA_VALIDATED"
    WORKFLOW_COMPLETE = "WORKFLOW_COMPLETE"
    FAILED = "FAILED"


#: States from which no further work may be dispatched.
TERMINAL_STATES: frozenset[ExecutionState] = frozenset(
    {ExecutionState.WORKFLOW_COMPLETE, ExecutionState.FAILED}
)

#: Services that actually perform node work (as opposed to ending the run).
DISPATCHABLE_SERVICES: frozenset[ServiceName] = frozenset(
    {
        ServiceName.ORCHESTRATOR,
        ServiceName.LLM_PROXY,
        ServiceName.RAG_SEARCH,
        ServiceName.HTTP_CONNECTOR,
        ServiceName.TRANSFORM_ENGINE,
        ServiceName.BRANCH_EVALUATOR,
    }
)

#: current_state -> the services it is legal to route to next.
#: A terminal state may only be followed by TERMINAL. This is the determinism
#: guard the mandate calls for: it makes an illegal path unrepresentable.
LEGAL_TRANSITIONS: dict[ExecutionState, frozenset[ServiceName]] = {
    state: (
        frozenset({ServiceName.TERMINAL})
        if state in TERMINAL_STATES
        else DISPATCHABLE_SERVICES | {ServiceName.TERMINAL}
    )
    for state in ExecutionState
}

#: Which service owns each node type. The orchestrator uses this to fill in
#: `next_step_definition.target_service` — it is never chosen by hand.
NODE_TYPE_SERVICE: dict[NodeType, ServiceName] = {
    NodeType.INPUT: ServiceName.ORCHESTRATOR,
    NodeType.LLM: ServiceName.LLM_PROXY,
    NodeType.RAG_RETRIEVE: ServiceName.RAG_SEARCH,
    NodeType.HTTP_REQUEST: ServiceName.HTTP_CONNECTOR,
    NodeType.TRANSFORM: ServiceName.TRANSFORM_ENGINE,
    NodeType.BRANCH: ServiceName.BRANCH_EVALUATOR,
    NodeType.OUTPUT: ServiceName.ORCHESTRATOR,
}

#: The state a node type leaves behind once it completes successfully.
NODE_TYPE_RESULT_STATE: dict[NodeType, ExecutionState] = {
    NodeType.INPUT: ExecutionState.INPUT_RESOLVED,
    NodeType.LLM: ExecutionState.LLM_INFERENCE_COMPLETE,
    NodeType.RAG_RETRIEVE: ExecutionState.RAG_CONTEXT_LOADED,
    NodeType.HTTP_REQUEST: ExecutionState.HTTP_CALL_COMPLETE,
    NodeType.TRANSFORM: ExecutionState.TRANSFORM_COMPLETE,
    NodeType.BRANCH: ExecutionState.BRANCH_EVALUATED,
    NodeType.OUTPUT: ExecutionState.WORKFLOW_COMPLETE,
}


class NextStepDefinition(ContractModel):
    """The explicitly declared next action. Prevents arbitrary pathing."""

    target_service: ServiceName
    target_node_id: str | None = Field(
        default=None, description="Node to execute next. None only when terminal."
    )
    input_bindings: dict[str, str] = Field(
        default_factory=dict,
        description="Variable expressions resolved against run state, e.g. {'q': '$output.topic'}.",
    )
    terminal: bool = Field(
        default=False, description="True when this job batch is the last one in the run."
    )

    @model_validator(mode="after")
    def _terminal_consistency(self) -> NextStepDefinition:
        if self.terminal:
            if self.target_service is not ServiceName.TERMINAL:
                raise ValueError("a terminal next_step_definition must target TERMINAL")
            if self.target_node_id is not None:
                raise ValueError("a terminal next_step_definition cannot name a target node")
        else:
            if self.target_service is ServiceName.TERMINAL:
                raise ValueError("TERMINAL target requires terminal=true")
            if not self.target_node_id:
                raise ValueError("a non-terminal next_step_definition must name a target node")
        return self


class WorkflowJobPayload(ContractModel):
    """The canonical queue message. Structurally mandatory fields per mandate §1."""

    schema_version: str = Field(default=CONTRACT_VERSION, pattern=r"^v\d+$")

    run_id: str = Field(..., min_length=1, description="Globally unique id for the run instance.")
    step_execution_id: str = Field(
        ..., min_length=1, description="Unique id for this specific node execution."
    )
    workflow_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1, description="The node this job batch executes.")

    job_context: JobContext
    current_state: ExecutionState
    next_step_definition: NextStepDefinition

    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Already-resolved inputs for this node."
    )
    attempt: int = Field(default=1, ge=1, description="Incremented by the broker on redelivery.")
    enqueued_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _validate_transition(self) -> WorkflowJobPayload:
        allowed = LEGAL_TRANSITIONS[self.current_state]
        target = self.next_step_definition.target_service
        if target not in allowed:
            raise ValueError(
                f"illegal transition: state {self.current_state.value} may not be followed by "
                f"{target.value} (legal: {sorted(s.value for s in allowed)})"
            )
        return self

    @property
    def idempotency_key(self) -> tuple[str, str]:
        """Composite key used for every idempotency gate and state write."""
        return (self.run_id, self.step_execution_id)


class StepOutputContext(ContractModel):
    """What a worker commits after executing one node. Written under the
    composite key so a retry can never double-write."""

    schema_version: str = Field(default=CONTRACT_VERSION, pattern=r"^v\d+$")

    run_id: str = Field(..., min_length=1)
    step_execution_id: str = Field(..., min_length=1)
    source_service: ServiceName
    node_id: str = Field(..., min_length=1)

    resulting_state: ExecutionState
    output: dict[str, Any] = Field(default_factory=dict)
    derived_context: dict[str, Any] = Field(
        default_factory=dict, description="Non-output metadata worth showing in the run report."
    )
    duration_ms: int = Field(default=0, ge=0)
    completed_at: datetime = Field(default_factory=utcnow)
