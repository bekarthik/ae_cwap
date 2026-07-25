"""Agents — a role, the skills it can use, a loop, and memory.

The difference from a v1 step is the loop. A step calls a model once and returns
the text. An agent is given an objective and a set of skills, decides which to
invoke, sees the results, and decides again — until it judges the objective met
or its iteration budget runs out.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from cwap_contracts.v1.base import ContractModel


class AgentMemoryConfig(ContractModel):
    """How an agent uses memory.

    Both directions are optional and separately controllable: an agent can read
    accumulated learnings without writing new ones (useful for a deterministic
    reporting step), or write without reading.
    """

    recall: bool = True
    #: How many prior learnings to inject. Small on purpose — a long memory block
    #: crowds out the actual task, and recall is ranked by relevance anyway.
    recall_limit: int = Field(default=5, ge=0, le=20)
    #: Whether the agent reflects after a run and records what it learned.
    write_learnings: bool = True
    #: Also read the workflow's own memory, not just the agent's.
    use_workflow_memory: bool = True


class AgentDefinition(ContractModel):
    """A reusable agent. Referenced by canvas nodes, not embedded in them, so
    improving an agent improves every workflow that uses it."""

    id: str = Field(..., min_length=1, max_length=64)
    tenant_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=120)
    #: What this agent *is* — "a meticulous travel planner".
    role: str = Field(..., min_length=1, max_length=500)
    #: What it is for — used when the node does not override it.
    objective: str = Field(default="", max_length=2000)
    #: Extra standing instructions appended to the generated system prompt.
    instructions: str = Field(default="", max_length=4000)
    skill_ids: list[str] = Field(default_factory=list, max_length=24)
    #: Ceiling on think-act cycles. The loop is the expensive part, so this is
    #: the main cost control.
    max_iterations: int = Field(default=6, ge=1, le=30)
    memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)
    #: Run this agent on a different model than the platform default — a cheap
    #: model for triage, a strong one for synthesis.
    model_override: str | None = None
    version: int = Field(default=1, ge=1)


class ToolCall(ContractModel):
    """A model's request to invoke a skill."""

    id: str = Field(..., min_length=1)
    skill_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(ContractModel):
    """What came back, fed to the model on the next iteration."""

    call_id: str = Field(..., min_length=1)
    skill_name: str = Field(..., min_length=1)
    output: str = Field(default="", max_length=100_000)
    is_error: bool = False
    duration_ms: int = Field(default=0, ge=0)


class AgentTurn(ContractModel):
    """One think-act cycle, recorded so the run report shows the agent's
    reasoning path rather than only its final answer."""

    iteration: int = Field(..., ge=1)
    text: str = Field(default="", max_length=100_000)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
