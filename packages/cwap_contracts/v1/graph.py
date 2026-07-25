"""The workflow blueprint: the JSON graph the canvas produces and the
orchestrator consumes.

This is the single artefact shared by Epic 2 (the canvas draws it) and Epic 4
(the state machine executes it), so its validation rules are deliberately
strict: a graph that cannot be executed must fail to *save*, not fail at run
time in front of the user.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from cwap_contracts.v1.base import ContractModel


class NodeType(str, Enum):
    """Every kind of block a user can drop onto the canvas."""

    INPUT = "input"
    LLM = "llm"
    RAG_RETRIEVE = "rag_retrieve"
    HTTP_REQUEST = "http_request"
    TRANSFORM = "transform"
    BRANCH = "branch"
    OUTPUT = "output"


#: Node types that terminate a path. A run finishes when it reaches one.
TERMINAL_NODE_TYPES = frozenset({NodeType.OUTPUT})

#: Node types that fan out on a boolean decision rather than a single successor.
BRANCHING_NODE_TYPES = frozenset({NodeType.BRANCH})


class Position(ContractModel):
    """Canvas coordinates. Purely presentational, but part of the saved graph so
    a reopened workflow looks exactly as the user left it."""

    x: float = 0.0
    y: float = 0.0


class WorkflowNode(ContractModel):
    """A single step. `params` is type-specific and validated by the executor
    that owns that node type (see `orchestrator.executors`)."""

    id: str = Field(..., min_length=1, max_length=128)
    type: NodeType
    label: str = Field(default="", max_length=200)
    params: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)
    knowledge_handle: str | None = Field(
        default=None,
        description="Set on RAG_RETRIEVE nodes: which indexed corpus to search (User Story 3).",
    )

    @model_validator(mode="after")
    def _rag_needs_a_corpus(self) -> WorkflowNode:
        if self.type is NodeType.RAG_RETRIEVE and not self.knowledge_handle:
            raise ValueError(
                f"node '{self.id}' is a rag_retrieve node but has no knowledge_handle; "
                "link a Knowledge Context before saving"
            )
        return self


class WorkflowEdge(ContractModel):
    """A directed connection. `condition` is only meaningful on edges leaving a
    BRANCH node, where it selects which outgoing path the truth value takes."""

    id: str = Field(..., min_length=1, max_length=128)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    condition: bool | None = Field(
        default=None,
        description="On a BRANCH node's outgoing edges: which branch this edge represents.",
    )
    bindings: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Parameter mapping from upstream output into the downstream node's inputs, "
            "e.g. {'search_term': '$output.topic'}."
        ),
    )


class WorkflowGraph(ContractModel):
    """A complete, executable blueprint.

    Invariants enforced here (all of them are things that would otherwise blow up
    mid-run): unique ids, edges that reference real nodes, exactly one entry
    point, no cycles, and branch nodes that actually branch.
    """

    id: str = Field(..., min_length=1)
    name: str = Field(default="Untitled workflow", max_length=200)
    version: int = Field(default=1, ge=1)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)

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
        roots = [node for node in self.nodes if node.id not in targets]
        return roots[0]

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
        indegree = {nid: 0 for nid in node_ids}
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
