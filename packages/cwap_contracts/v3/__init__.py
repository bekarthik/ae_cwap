"""Version 3 of the platform contract set — MCP connectors.

v2 made every working step an agent with skills. v3 lets a skill point at a tool
on a **connected MCP server**, so an agent can reach a system nobody here wrote a
connector for — a repository, a filesystem, an internal service.

Only the skill contracts changed shape, so only they are redefined:

* `SkillKind` gains `MCP`, and `SkillOrigin` gains `MCP` for tools imported from
  a server rather than authored here.
* `SkillParameter.type` admits `object` and `array`, because real MCP tools take
  structured arguments and flattening them would make the tool unusable.
* `SkillProposal` refuses `MCP` outright: synthesis may not invent a connection
  to something outside the platform.

`MCPServerRecord` and friends are new rather than changed. Everything else is
imported from v2 and v1, so there is exactly one definition of `JobContext`,
`WorkflowGraph` and the rest. v1 and v2 stay registered and valid.
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
