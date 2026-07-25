"""Version 4 of the platform contract set — agents pick their model, workflows
fan out, and a step can be reviewed.

Three changes, each of which was a promise the previous shape could not keep:

* `AgentDefinition` gains `model_provider`, `model_override` (now actually read)
  and `thinking_effort`. v2 declared `model_override` and ignored it, so "mix
  providers within one workflow" was a field, not a feature.
* `WorkflowNode` gains `review`: another agent critiques the answer and the
  author revises, bounded by a round limit. Repetition lives *inside* a step, so
  the graph stays acyclic and a run still terminates by construction.
* `WorkflowGraph` allows fan-out from any node. Independent work runs at the
  same time; a node with several inbound edges is a join that waits for all of
  them. Every enqueued job still names exactly one node.

Everything else is imported from v3, v2 and v1, so there is exactly one
definition of `JobContext`, `SkillDefinition` and the rest. Earlier versions
stay registered and valid.
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
    AgentMemoryConfig,
    AgentTurn,
    ToolCall,
    ToolResult,
)
from cwap_contracts.v2.design import (
    ClarifyingQuestion,
    DesignRequest,
    DesignStage,
    SkillGap,
)
from cwap_contracts.v2.graph import (
    BRANCHING_NODE_TYPES,
    REASONING_NODE_TYPES,
    TERMINAL_NODE_TYPES,
    NodeType,
    Position,
    WorkflowEdge,
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
from cwap_contracts.v3.mcp import (
    MCPConnectionResult,
    MCPServerConfig,
    MCPServerRecord,
    MCPTool,
    MCPTransport,
)
from cwap_contracts.v3.skills import (
    SIDE_EFFECTING_KINDS,
    SKILL_KINDS,
    SkillDefinition,
    SkillKind,
    SkillOrigin,
    SkillParameter,
    SkillProposal,
)
from cwap_contracts.v4.agents import EFFORT_LEVELS, AgentDefinition
from cwap_contracts.v4.design import DesignResponse, PlannedAgent
from cwap_contracts.v4.graph import (
    DEFAULT_APPROVAL,
    MAX_REVIEW_ROUNDS,
    ReviewConfig,
    WorkflowGraph,
    WorkflowNode,
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
    "DEFAULT_APPROVAL",
    "EFFORT_LEVELS",
    "MAX_REVIEW_ROUNDS",
    "REASONING_NODE_TYPES",
    "ReviewConfig",
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
    "MCPConnectionResult",
    "MCPServerConfig",
    "MCPServerRecord",
    "MCPTool",
    "MCPTransport",
    "SIDE_EFFECTING_KINDS",
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
