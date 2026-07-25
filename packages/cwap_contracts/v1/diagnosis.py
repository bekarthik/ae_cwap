"""Epic 1 contracts — natural-language goal in, proposed workflow out.

Output contract from the brief:
    { "intent": ..., "required_steps": [...], "suggestions": [...] }
plus the scaffolded graph the user lands on when the canvas opens.
"""

from __future__ import annotations

from pydantic import Field

from cwap_contracts.v1.base import ContractModel
from cwap_contracts.v1.graph import NodeType, WorkflowGraph


class GoalIntakeRequest(ContractModel):
    """User Story 2: "Plan my weekend trip to Denver"."""

    goal: str = Field(..., min_length=3, max_length=2000)
    knowledge_handles: list[str] = Field(
        default_factory=list,
        description="Corpora the user already uploaded, offered to the scaffolder.",
    )


class SkillRequirement(ContractModel):
    """One capability the diagnosis decided the goal needs."""

    skill: str = Field(..., min_length=1)
    node_type: NodeType
    rationale: str = Field(default="", max_length=500)


class DiagnosisResult(ContractModel):
    """Structured read of an unstructured goal."""

    intent: str = Field(..., min_length=1)
    required_steps: list[SkillRequirement] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    gaps: list[str] = Field(
        default_factory=list,
        description="Things the system could not scaffold, e.g. a missing API credential.",
    )


class ScaffoldResponse(ContractModel):
    """What the onboarding layer hands the canvas: the reasoning *and* the draft."""

    diagnosis: DiagnosisResult
    graph: WorkflowGraph
