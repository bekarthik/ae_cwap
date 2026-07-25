"""The design conversation.

v1 took a goal and produced a graph in one shot. That is fine for a goal that is
already specific and wrong for everything else: "sort out our onboarding" does
not contain enough information to design anything, and guessing produces a
confident, useless workflow.

So designing is a short conversation. The system asks only what it genuinely
cannot infer, then designs the agents, the skills each one needs, and the memory
wiring — and says which skills it will have to create.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from cwap_contracts.v1.base import ContractModel
from cwap_contracts.v2.graph import WorkflowGraph
from cwap_contracts.v2.skills import SkillProposal


class DesignStage(str, Enum):
    #: The system needs answers before it can design anything useful.
    CLARIFYING = "clarifying"
    #: A complete design is attached.
    DESIGNED = "designed"


class ClarifyingQuestion(ContractModel):
    """One thing the system could not reasonably infer.

    `why` is required: a question the user cannot see the purpose of feels like
    an interrogation, and they answer it badly or not at all.
    """

    id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=500)
    why: str = Field(..., min_length=1, max_length=300)
    #: Offered answers. Free text is always allowed too.
    options: list[str] = Field(default_factory=list, max_length=6)
    #: Used when the user skips it, so an unanswered question never blocks design.
    default: str = Field(default="", max_length=500)


class SkillGap(ContractModel):
    """A capability an agent needs that the tenant does not have yet."""

    needed_by: str = Field(..., min_length=1, description="Agent name that needs it.")
    capability: str = Field(..., min_length=1, max_length=300)
    #: Present when the platform can build it; absent when it needs a human
    #: (a credential, a system it cannot reach).
    proposal: SkillProposal | None = None
    blocked_reason: str = Field(default="", max_length=500)


class PlannedAgent(ContractModel):
    """An agent the design intends to create, before it exists."""

    name: str = Field(..., min_length=1, max_length=120)
    role: str = Field(..., min_length=1, max_length=500)
    objective: str = Field(..., min_length=1, max_length=2000)
    #: Skill names — existing ones by name, new ones matching a `SkillGap`.
    skills: list[str] = Field(default_factory=list, max_length=24)
    rationale: str = Field(default="", max_length=1000)


class DesignRequest(ContractModel):
    goal: str = Field(..., min_length=3, max_length=4000)
    #: question id -> the user's answer. Absent on the first turn.
    answers: dict[str, str] = Field(default_factory=dict)
    knowledge_handles: list[str] = Field(default_factory=list)
    #: Proceed with defaults rather than asking anything.
    skip_questions: bool = False


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
