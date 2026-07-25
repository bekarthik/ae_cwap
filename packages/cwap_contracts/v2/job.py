"""The v2 Execution Contract.

Identical discipline to v1 — immutable, versioned, with a legal-transition guard
that makes an illegal path unrepresentable. The only changes are the new agent
states and the `AGENT_RUNTIME` service that owns them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from cwap_contracts.v1.base import ContractModel, utcnow
from cwap_contracts.v1.identity import JobContext
from cwap_contracts.v2.graph import NodeType

CONTRACT_VERSION = "v2"


class ServiceName(str, Enum):
    ORCHESTRATOR = "ORCHESTRATOR"
    #: Owns the agent loop: recall memory, call the model, invoke skills, repeat.
    AGENT_RUNTIME = "AGENT_RUNTIME"
    LLM_PROXY = "LLM_PROXY"
    RAG_SEARCH = "RAG_SEARCH"
    HTTP_CONNECTOR = "HTTP_CONNECTOR"
    TRANSFORM_ENGINE = "TRANSFORM_ENGINE"
    BRANCH_EVALUATOR = "BRANCH_EVALUATOR"
    TERMINAL = "TERMINAL"


class ExecutionState(str, Enum):
    RUN_ACCEPTED = "RUN_ACCEPTED"
    INPUT_RESOLVED = "INPUT_RESOLVED"
    LLM_INFERENCE_COMPLETE = "LLM_INFERENCE_COMPLETE"
    #: An agent finished its loop having met its objective.
    AGENT_OBJECTIVE_MET = "AGENT_OBJECTIVE_MET"
    #: An agent stopped because it exhausted its iteration budget. Distinct from
    #: success on purpose: the answer may be partial, and the run report should
    #: not present it as complete.
    AGENT_BUDGET_EXHAUSTED = "AGENT_BUDGET_EXHAUSTED"
    RAG_CONTEXT_LOADED = "RAG_CONTEXT_LOADED"
    HTTP_CALL_COMPLETE = "HTTP_CALL_COMPLETE"
    TRANSFORM_COMPLETE = "TRANSFORM_COMPLETE"
    BRANCH_EVALUATED = "BRANCH_EVALUATED"
    DATA_VALIDATED = "DATA_VALIDATED"
    WORKFLOW_COMPLETE = "WORKFLOW_COMPLETE"
    FAILED = "FAILED"


TERMINAL_STATES: frozenset[ExecutionState] = frozenset(
    {ExecutionState.WORKFLOW_COMPLETE, ExecutionState.FAILED}
)

DISPATCHABLE_SERVICES: frozenset[ServiceName] = frozenset(
    {
        ServiceName.ORCHESTRATOR,
        ServiceName.AGENT_RUNTIME,
        ServiceName.LLM_PROXY,
        ServiceName.RAG_SEARCH,
        ServiceName.HTTP_CONNECTOR,
        ServiceName.TRANSFORM_ENGINE,
        ServiceName.BRANCH_EVALUATOR,
    }
)

LEGAL_TRANSITIONS: dict[ExecutionState, frozenset[ServiceName]] = {
    state: (
        frozenset({ServiceName.TERMINAL})
        if state in TERMINAL_STATES
        else DISPATCHABLE_SERVICES | {ServiceName.TERMINAL}
    )
    for state in ExecutionState
}

NODE_TYPE_SERVICE: dict[NodeType, ServiceName] = {
    NodeType.INPUT: ServiceName.ORCHESTRATOR,
    NodeType.AGENT: ServiceName.AGENT_RUNTIME,
    NodeType.LLM: ServiceName.LLM_PROXY,
    NodeType.RAG_RETRIEVE: ServiceName.RAG_SEARCH,
    NodeType.HTTP_REQUEST: ServiceName.HTTP_CONNECTOR,
    NodeType.TRANSFORM: ServiceName.TRANSFORM_ENGINE,
    NodeType.BRANCH: ServiceName.BRANCH_EVALUATOR,
    NodeType.OUTPUT: ServiceName.ORCHESTRATOR,
}

NODE_TYPE_RESULT_STATE: dict[NodeType, ExecutionState] = {
    NodeType.INPUT: ExecutionState.INPUT_RESOLVED,
    NodeType.AGENT: ExecutionState.AGENT_OBJECTIVE_MET,
    NodeType.LLM: ExecutionState.LLM_INFERENCE_COMPLETE,
    NodeType.RAG_RETRIEVE: ExecutionState.RAG_CONTEXT_LOADED,
    NodeType.HTTP_REQUEST: ExecutionState.HTTP_CALL_COMPLETE,
    NodeType.TRANSFORM: ExecutionState.TRANSFORM_COMPLETE,
    NodeType.BRANCH: ExecutionState.BRANCH_EVALUATED,
    NodeType.OUTPUT: ExecutionState.WORKFLOW_COMPLETE,
}


class NextStepDefinition(ContractModel):
    target_service: ServiceName
    target_node_id: str | None = None
    input_bindings: dict[str, str] = Field(default_factory=dict)
    terminal: bool = False

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
    schema_version: str = Field(default=CONTRACT_VERSION, pattern=r"^v\d+$")

    run_id: str = Field(..., min_length=1)
    step_execution_id: str = Field(..., min_length=1)
    workflow_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)

    job_context: JobContext
    current_state: ExecutionState
    next_step_definition: NextStepDefinition

    inputs: dict[str, Any] = Field(default_factory=dict)
    attempt: int = Field(default=1, ge=1)
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
        return (self.run_id, self.step_execution_id)


class StepOutputContext(ContractModel):
    schema_version: str = Field(default=CONTRACT_VERSION, pattern=r"^v\d+$")

    run_id: str = Field(..., min_length=1)
    step_execution_id: str = Field(..., min_length=1)
    source_service: ServiceName
    node_id: str = Field(..., min_length=1)

    resulting_state: ExecutionState
    output: dict[str, Any] = Field(default_factory=dict)
    derived_context: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = Field(default=0, ge=0)
    completed_at: datetime = Field(default_factory=utcnow)
