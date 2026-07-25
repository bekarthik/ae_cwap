"""Version 2 of the platform contract set — agents, skills and memory.

v1 modelled a workflow as typed *steps*: an LLM step was a single model call.
v2 makes each step an **agent** — a role with skills it can invoke, an iterative
loop, and memory it reads before working and writes after.

Everything unchanged is imported from v1 rather than copied, so there is exactly
one definition of `JobContext`, `LogEvent` and the knowledge contracts. Only the
models whose *shape* changed are redefined here:

* `NodeType` gains `AGENT`, so `WorkflowNode` and `WorkflowGraph` change with it.
* `ExecutionState` gains agent-loop states, so `WorkflowJobPayload` changes.

v1 stays registered and valid. A deployment mid-upgrade can hold both.
"""

from cwap_contracts.v1.base import CONTRACT_VERSION as V1_VERSION
from cwap_contracts.v1.base import ContractModel, utcnow
from cwap_contracts.v1.identity import JobContext, PermissionRequirement
from cwap_contracts.v1.knowledge import (
    IngestRequest,
    KnowledgeHandle,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from cwap_contracts.v1.logs import LogEvent, LogLevel
from cwap_contracts.v2.agents import (
    AgentDefinition,
    AgentMemoryConfig,
    AgentTurn,
    ToolCall,
    ToolResult,
)
from cwap_contracts.v2.design import (
    ClarifyingQuestion,
    DesignRequest,
    DesignResponse,
    DesignStage,
    PlannedAgent,
    SkillGap,
)
from cwap_contracts.v2.graph import (
    BRANCHING_NODE_TYPES,
    TERMINAL_NODE_TYPES,
    NodeType,
    Position,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from cwap_contracts.v2.job import (
    DISPATCHABLE_SERVICES,
    LEGAL_TRANSITIONS,
    NODE_TYPE_RESULT_STATE,
    NODE_TYPE_SERVICE,
    TERMINAL_STATES,
    ExecutionState,
    NextStepDefinition,
    ServiceName,
    StepOutputContext,
    WorkflowJobPayload,
)
from cwap_contracts.v2.memory import (
    MemoryEntry,
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    RecallRequest,
    RecallResult,
)
from cwap_contracts.v2.skills import (
    SKILL_KINDS,
    SkillDefinition,
    SkillKind,
    SkillOrigin,
    SkillParameter,
    SkillProposal,
)

CONTRACT_VERSION = "v2"

__all__ = [
    "BRANCHING_NODE_TYPES",
    "CONTRACT_VERSION",
    "DISPATCHABLE_SERVICES",
    "LEGAL_TRANSITIONS",
    "NODE_TYPE_RESULT_STATE",
    "NODE_TYPE_SERVICE",
    "SKILL_KINDS",
    "TERMINAL_NODE_TYPES",
    "TERMINAL_STATES",
    "V1_VERSION",
    "AgentDefinition",
    "AgentMemoryConfig",
    "AgentTurn",
    "ClarifyingQuestion",
    "ContractModel",
    "DesignRequest",
    "DesignResponse",
    "DesignStage",
    "ExecutionState",
    "IngestRequest",
    "JobContext",
    "KnowledgeHandle",
    "LogEvent",
    "LogLevel",
    "MemoryEntry",
    "MemoryKind",
    "MemoryScope",
    "MemoryWriteRequest",
    "NextStepDefinition",
    "NodeType",
    "PermissionRequirement",
    "PlannedAgent",
    "Position",
    "RecallRequest",
    "RecallResult",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievedChunk",
    "ServiceName",
    "SkillDefinition",
    "SkillGap",
    "SkillKind",
    "SkillOrigin",
    "SkillParameter",
    "SkillProposal",
    "StepOutputContext",
    "ToolCall",
    "ToolResult",
    "WorkflowEdge",
    "WorkflowGraph",
    "WorkflowJobPayload",
    "WorkflowNode",
    "utcnow",
]
