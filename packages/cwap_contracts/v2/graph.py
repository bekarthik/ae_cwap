"""The v2 workflow blueprint.

Same structural guarantees as v1 — one entry point, no cycles, well-formed
decisions — with one addition: `AGENT`, a node that carries a role, a set of
skills, and its own loop.

The v1 node types are kept rather than replaced. Not everything wants to be an
agent: a template render or a comparison is deterministic, cheaper and easier to
reason about as a plain step, and forcing a model into that position would be
worse in every dimension.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from cwap_contracts.v1.base import ContractModel
from cwap_contracts.v1.graph import Position


class NodeType(str, Enum):
    INPUT = "input"
    #: A role with skills and a loop. The v2 addition.
    AGENT = "agent"
    #: A single model call. Retained: sometimes one call is the right answer.
    LLM = "llm"
    RAG_RETRIEVE = "rag_retrieve"
    HTTP_REQUEST = "http_request"
    TRANSFORM = "transform"
    BRANCH = "branch"
    OUTPUT = "output"


TERMINAL_NODE_TYPES = frozenset({NodeType.OUTPUT})
BRANCHING_NODE_TYPES = frozenset({NodeType.BRANCH})

#: Node types that reason. Only these read and write memory.
REASONING_NODE_TYPES = frozenset({NodeType.AGENT, NodeType.LLM})


class WorkflowNode(ContractModel):
    """A single step. `params` is type-specific and validated by its executor."""

    id: str = Field(..., min_length=1, max_length=128)
    type: NodeType
    label: str = Field(default="", max_length=200)
    params: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)
    knowledge_handle: str | None = Field(
        default=None, description="Set on RAG_RETRIEVE nodes: which corpus to search."
    )
    #: Set on AGENT nodes: which stored agent definition to run.
    agent_id: str | None = Field(
        default=None, description="Set on AGENT nodes: the agent definition to run."
    )

    @model_validator(mode="after")
    def _type_specific_requirements(self) -> WorkflowNode:
        if self.type is NodeType.RAG_RETRIEVE and not self.knowledge_handle:
            raise ValueError(
                f"node '{self.id}' is a rag_retrieve node but has no knowledge_handle; "
                "link a Knowledge Context before saving"
            )
        if self.type is NodeType.AGENT and not self.agent_id:
            raise ValueError(
                f"node '{self.id}' is an agent node but names no agent; "
                "an agent node without a role and skills has nothing to run"
            )
        return self


class WorkflowEdge(ContractModel):
    id: str = Field(..., min_length=1, max_length=128)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    condition: bool | None = None
    bindings: dict[str, str] = Field(default_factory=dict)


class WorkflowGraph(ContractModel):
    """A complete, executable blueprint."""

    id: str = Field(..., min_length=1)
    name: str = Field(default="Untitled workflow", max_length=200)
    version: int = Field(default=1, ge=1)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    #: Memory shared by every reasoning node in this workflow. Defaults to the
    #: workflow's own id, so a workflow accumulates knowledge about its job
    #: without anyone configuring anything.
    memory_scope_id: str | None = None

    # ---- lookups -------------------------------------------------------

    def node(self, node_id: str) -> WorkflowNode:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(f"no node '{node_id}' in workflow '{self.id}'")

    def outgoing(self, node_id: str) -> list[WorkflowEdge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def incoming(self, node_id: str) -> list[WorkflowEdge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def entry_node(self) -> WorkflowNode:
        targets = {edge.target for edge in self.edges}
        return next(node for node in self.nodes if node.id not in targets)

    def agent_ids(self) -> list[str]:
        return [node.agent_id for node in self.nodes if node.agent_id]

    @property
    def memory_scope(self) -> str:
        return self.memory_scope_id or self.id

    # ---- validation ----------------------------------------------------

    @model_validator(mode="after")
    def _validate_structure(self) -> WorkflowGraph:
        if not self.nodes:
            raise ValueError("a workflow must contain at least one node")

        node_ids = [node.id for node in self.nodes]
        duplicates = {nid for nid in node_ids if node_ids.count(nid) > 1}
        if duplicates:
            raise ValueError(f"duplicate node ids: {sorted(duplicates)}")

        edge_ids = [edge.id for edge in self.edges]
        dup_edges = {eid for eid in edge_ids if edge_ids.count(eid) > 1}
        if dup_edges:
            raise ValueError(f"duplicate edge ids: {sorted(dup_edges)}")

        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known:
                raise ValueError(f"edge '{edge.id}' has unknown source '{edge.source}'")
            if edge.target not in known:
                raise ValueError(f"edge '{edge.id}' has unknown target '{edge.target}'")
            if edge.source == edge.target:
                raise ValueError(f"edge '{edge.id}' is a self-loop on '{edge.source}'")

        targets = {edge.target for edge in self.edges}
        roots = [nid for nid in node_ids if nid not in targets]
        if len(roots) != 1:
            raise ValueError(
                "a workflow needs exactly one entry node (a node with no inbound edge); "
                f"found {len(roots)}: {sorted(roots)}"
            )

        self._reject_cycles(node_ids)
        self._validate_branches()
        return self

    def _reject_cycles(self, node_ids: list[str]) -> None:
        """Kahn's algorithm. A cycle in the canvas means a run that never ends."""
        indegree = dict.fromkeys(node_ids, 0)
        for edge in self.edges:
            indegree[edge.target] += 1

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for edge in self.outgoing(current):
                indegree[edge.target] -= 1
                if indegree[edge.target] == 0:
                    queue.append(edge.target)

        if visited != len(node_ids):
            raise ValueError("workflow graph contains a cycle; execution would never terminate")

    def _validate_branches(self) -> None:
        for node in self.nodes:
            outgoing = self.outgoing(node.id)
            if node.type in BRANCHING_NODE_TYPES:
                conditions = {edge.condition for edge in outgoing}
                if conditions != {True, False}:
                    raise ValueError(
                        f"branch node '{node.id}' must have exactly one edge with "
                        "condition=true and one with condition=false"
                    )
            elif node.type in TERMINAL_NODE_TYPES:
                if outgoing:
                    raise ValueError(f"output node '{node.id}' cannot have outgoing edges")
            elif len(outgoing) > 1:
                raise ValueError(
                    f"node '{node.id}' has {len(outgoing)} outgoing edges; only a branch "
                    "node may fan out"
                )
