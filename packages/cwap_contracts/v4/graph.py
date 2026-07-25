"""Workflows that fan out, and steps that get reviewed. Version 4.

Two shape changes, both of which v2 deliberately excluded and both of which the
product needs.

**Fan-out.** v2 allowed exactly one outgoing edge from anything but a branch,
which made every workflow a line. That was the right first constraint — one
successor per step is what makes the job payload deterministic — but it also
means two independent pieces of work are done one after the other for no reason.
v4 lets any node have several outgoing edges: each is dispatched as its own job,
and a node with several *inbound* edges is a join that waits for all of them.
Each enqueued job still names exactly one node, so the mandate's determinism is
untouched; what changed is how many jobs a completed step may produce.

The cycle rule is unchanged and load-bearing: the graph is still a DAG, so a
run still terminates.

**Review.** A step may nominate another agent as its reviewer. The reviewer
critiques the draft, the author revises, and that repeats until the reviewer
approves or the round limit is reached. This is bounded repetition *inside* one
step rather than an edge back into the graph, which is what keeps the DAG a DAG
— an edge that loops would make termination a property of the model's judgement
rather than of the structure.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, model_validator

from cwap_contracts.v1.base import ContractModel
from cwap_contracts.v2.graph import (
    BRANCHING_NODE_TYPES,
    REASONING_NODE_TYPES,
    TERMINAL_NODE_TYPES,
    NodeType,
    Position,
    WorkflowEdge,
)

#: A reviewer that never approves would otherwise cost a run's whole budget, and
#: the point of a loop is to converge, not to grind.
MAX_REVIEW_ROUNDS = 5

#: What a reviewer says when the work is good enough. Matched case-insensitively
#: as a prefix, so a reviewer that adds "APPROVED — nice work" still approves.
DEFAULT_APPROVAL = "APPROVED"


class ReviewConfig(ContractModel):
    """A second agent that has to be satisfied before a step's answer is used."""

    #: The reviewing agent. Deliberately an agent, not a prompt: a reviewer with
    #: its own skills and its own memory gets better at reviewing.
    agent_id: str = Field(..., min_length=1, max_length=64)
    #: How many revise-and-recheck rounds are permitted before the draft stands
    #: as it is. Bounded so a run's cost is knowable before it starts.
    max_rounds: int = Field(default=2, ge=1, le=MAX_REVIEW_ROUNDS)
    #: The word that ends the loop.
    approval_phrase: str = Field(default=DEFAULT_APPROVAL, min_length=1, max_length=64)


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
    #: Optional quality loop: another agent reviews this step's answer.
    review: ReviewConfig | None = None

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
        if self.review is not None and self.type not in REASONING_NODE_TYPES:
            raise ValueError(
                f"node '{self.id}' is a {self.type.value} node and cannot be reviewed; "
                "only a step that produces an answer has something to review"
            )
        if self.review is not None and self.review.agent_id == self.agent_id:
            raise ValueError(
                f"node '{self.id}' names itself as its own reviewer; a reviewer that "
                "wrote the draft approves it"
            )
        return self


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

        pairs = [(edge.source, edge.target) for edge in self.edges]
        repeated = {pair for pair in pairs if pairs.count(pair) > 1}
        if repeated:
            # Two edges between the same pair would dispatch the same step twice
            # and make a join wait for an arrival that has already happened.
            raise ValueError(f"duplicate edges between the same nodes: {sorted(repeated)}")

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

    #: Whether a plain node may have several outgoing edges. Held as a flag
    #: rather than simply deleted because a permission the runtime cannot honour
    #: is worse than a restriction: until the worker dispatches every successor,
    #: a second edge would be accepted at save time and silently never run.
    #: `orchestrator` flips this on when it can fan out.
    ALLOW_FAN_OUT: ClassVar[bool] = False

    def _validate_branches(self) -> None:
        """A *branch* is exactly two paths; plain fan-out is parallel work.

        The distinction is what each means at run time. Several plain edges are
        independent work — all of them run. A branch's two edges are a choice —
        one of them runs — so a branch with three edges, or with two "true"
        edges, has no defined behaviour rather than an inconvenient one.
        """
        for node in self.nodes:
            outgoing = self.outgoing(node.id)
            if node.type in BRANCHING_NODE_TYPES:
                conditions = sorted(
                    (edge.condition for edge in outgoing), key=lambda value: str(value)
                )
                if conditions != [False, True]:
                    raise ValueError(
                        f"branch node '{node.id}' must have exactly one edge with "
                        "condition=true and one with condition=false"
                    )
            elif node.type in TERMINAL_NODE_TYPES:
                if outgoing:
                    raise ValueError(f"output node '{node.id}' cannot have outgoing edges")
            elif len(outgoing) > 1 and not self.ALLOW_FAN_OUT:
                raise ValueError(
                    f"node '{node.id}' has {len(outgoing)} outgoing edges; only a branch "
                    "node may fan out"
                )
