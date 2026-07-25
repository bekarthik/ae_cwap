"""HTTP request/response bodies.

Distinct from `cwap_contracts` on purpose: those are the *inter-service* schema
registry, versioned and locked. These are the browser-facing shapes, which are
free to change with the UI without triggering a contract version bump.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cwap_contracts.v4 import (
    ClarifyingQuestion,
    DesignStage,
    PlannedAgent,
    SkillGap,
    WorkflowGraph,
)
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- auth ----------------------------------------------------------------


class RegisterRequest(Strict):
    email: EmailStr
    password: str = Field(min_length=12, max_length=200)
    tenant_id: str = Field(default="default", min_length=1, max_length=64)


class LoginRequest(Strict):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(Strict):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    tenant_id: str
    scopes: list[str]


# ---- workflows -----------------------------------------------------------


class SaveWorkflowRequest(Strict):
    graph: WorkflowGraph


class WorkflowSummary(Strict):
    id: str
    name: str
    version: int
    node_count: int
    updated_at: datetime


class WorkflowDetail(Strict):
    id: str
    name: str
    version: int
    graph: WorkflowGraph
    updated_at: datetime


# ---- design --------------------------------------------------------------


class DesignGoalRequest(Strict):
    goal: str = Field(min_length=3, max_length=4000)
    #: question id -> answer. Absent on the first turn of the conversation.
    answers: dict[str, str] = Field(default_factory=dict)
    knowledge_handles: list[str] = Field(default_factory=list)
    #: Skip the questions and design from defaults.
    skip_questions: bool = False


class DesignGoalResponse(Strict):
    stage: DesignStage
    understanding: str
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    agents: list[PlannedAgent] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    graph: WorkflowGraph | None = None
    notes: list[str] = Field(default_factory=list)


# ---- knowledge -----------------------------------------------------------


class IngestTextRequest(Strict):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1)
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)


class KnowledgeSummary(Strict):
    handle: str
    title: str
    chunk_count: int
    created_at: datetime


class RetrievePreviewRequest(Strict):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


# ---- runs ----------------------------------------------------------------


class StartRunRequest(Strict):
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: Run the graph in the body instead of the saved one — used by the canvas
    #: "Run" button so a user can test unsaved edits.
    graph: WorkflowGraph | None = None


class RunSummary(Strict):
    run_id: str
    workflow_id: str
    status: str
    steps_executed: int
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class StepReport(Strict):
    node_id: str
    source_service: str
    resulting_state: str
    output: dict[str, Any]
    derived_context: dict[str, Any]
    duration_ms: int
    completed_at: datetime


class RunReport(Strict):
    """The transparent execution report — why the AI answered as it did."""

    run: RunSummary
    result: dict[str, Any] | None
    steps: list[StepReport]
    logs: list[dict[str, Any]]


# ---- admin ---------------------------------------------------------------


class DeadLetterRecord(Strict):
    id: int
    queue: str
    reason_code: str
    reason: str
    run_id: str | None
    detail: dict[str, Any] | None
    created_at: datetime
