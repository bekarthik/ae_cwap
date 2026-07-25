"""Schema Registry — the canonical, versioned contract definitions for the platform.

Every payload that crosses a service boundary (API Gateway -> Orchestrator,
Orchestrator -> Queue, Worker -> DB) is defined here and nowhere else. Services
import these models; they never hand-roll a dict for an inter-service message.

The registry is versioned. Changing the *shape* of a published contract without
bumping its version fails `tests/test_contract_registry.py`, which compares the
live JSON Schema fingerprint against `contracts.lock.json`.
"""

from cwap_contracts.errors import (
    AuthorizationFailure,
    ContractViolation,
    CwapContractError,
)
from cwap_contracts.registry import (
    CONTRACT_REGISTRY,
    fingerprint,
    latest_version,
    lock_snapshot,
    register,
    resolve,
)
from cwap_contracts.v1 import (
    BRANCHING_NODE_TYPES,
    DISPATCHABLE_SERVICES,
    LEGAL_TRANSITIONS,
    NODE_TYPE_RESULT_STATE,
    NODE_TYPE_SERVICE,
    TERMINAL_NODE_TYPES,
    TERMINAL_STATES,
    ContractModel,
    DiagnosisResult,
    ExecutionState,
    GoalIntakeRequest,
    IngestRequest,
    JobContext,
    KnowledgeHandle,
    LogEvent,
    LogLevel,
    NextStepDefinition,
    NodeType,
    PermissionRequirement,
    Position,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
    ScaffoldResponse,
    ServiceName,
    SkillRequirement,
    StepOutputContext,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowJobPayload,
    WorkflowNode,
)

SCHEMA_VERSION = "v1"

__all__ = [
    "SCHEMA_VERSION",
    "BRANCHING_NODE_TYPES",
    "CONTRACT_REGISTRY",
    "DISPATCHABLE_SERVICES",
    "LEGAL_TRANSITIONS",
    "NODE_TYPE_RESULT_STATE",
    "NODE_TYPE_SERVICE",
    "TERMINAL_NODE_TYPES",
    "TERMINAL_STATES",
    "ContractModel",
    "AuthorizationFailure",
    "ContractViolation",
    "CwapContractError",
    "DiagnosisResult",
    "ExecutionState",
    "GoalIntakeRequest",
    "IngestRequest",
    "JobContext",
    "KnowledgeHandle",
    "LogEvent",
    "LogLevel",
    "NextStepDefinition",
    "NodeType",
    "PermissionRequirement",
    "Position",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievedChunk",
    "ScaffoldResponse",
    "ServiceName",
    "SkillRequirement",
    "StepOutputContext",
    "WorkflowEdge",
    "WorkflowGraph",
    "WorkflowJobPayload",
    "WorkflowNode",
    "fingerprint",
    "latest_version",
    "lock_snapshot",
    "register",
    "resolve",
]
