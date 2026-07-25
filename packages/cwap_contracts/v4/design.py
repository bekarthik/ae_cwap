"""The design conversation. Version 4.

Two changes, both about telling the user what the planner actually did.

`PlannedAgent` gains `reused`. v2's planner created every agent it named, so a
workspace grew a new "Researcher" on every design and the roster filled with
near-duplicates nobody asked for. Reuse-first matching fixes that, and the field
is how the review screen can say *which* agents are yours already — a plan that
silently reuses is as opaque as one that silently duplicates.

`DesignResponse.graph` is the v4 graph, because a designed workflow may now
carry a review loop on a step.
"""

from __future__ import annotations

from pydantic import Field

from cwap_contracts.v1.base import ContractModel
from cwap_contracts.v2.design import ClarifyingQuestion, DesignStage, SkillGap
from cwap_contracts.v4.graph import WorkflowGraph


class PlannedAgent(ContractModel):
    """One agent in a proposed team."""

    name: str = Field(..., min_length=1, max_length=120)
    role: str = Field(..., min_length=1, max_length=500)
    objective: str = Field(..., min_length=1, max_length=2000)
    #: Skill names — existing ones by name, new ones matching a `SkillGap`.
    skills: list[str] = Field(default_factory=list, max_length=24)
    rationale: str = Field(default="", max_length=1000)
    #: True when this is an agent the workspace already had, matched by role
    #: rather than created. Shown on the review screen so "you already have
    #: these" is visible before anything is stored.
    reused: bool = False
    #: The agent nominated to review this one's work, if the plan asked for a
    #: quality loop. A name, resolved to an id when the graph is built.
    reviewer: str = Field(default="", max_length=120)


class DesignResponse(ContractModel):
    stage: DesignStage
    #: Restated back to the user, so a misunderstanding is caught before build.
    understanding: str = Field(default="", max_length=2000)
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    agents: list[PlannedAgent] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    #: Only on the DESIGNED stage.
    graph: WorkflowGraph | None = None
    notes: list[str] = Field(default_factory=list)
