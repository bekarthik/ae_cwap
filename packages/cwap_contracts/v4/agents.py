"""Agents that choose their own model. Version 4.

v2 gave an agent a role, skills, a loop and memory, and left the model to the
deployment: every agent in every workflow ran on whatever the workspace was
configured with. `model_override` was declared for exactly this and never read,
which is worse than not offering it — a field that is stored, returned by the
API and ignored at run time is a promise the product does not keep.

So v4 makes the model part of the agent, and says so in three fields rather than
one, because "which model" is not answerable without "whose server" and "how
hard should it think":

* `model_provider` — which backend serves it. Without this, `model_override`
  cannot be resolved: `claude-opus-5` and `qwen3` are not reachable at the same
  endpoint, and guessing the provider from a model name is the kind of inference
  this codebase refuses to make elsewhere.
* `model_override` — the model id on that backend.
* `thinking_effort` — how hard it may think, for the models that have a
  reasoning mode. Per agent because that is where the decision belongs: a
  triage step wants `low`, a synthesis step wants `high`, and one deployment
  setting cannot be both.

All three are blank by default, meaning "use the workspace's model". A workflow
that names nothing runs exactly as it did before.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from cwap_contracts.v1.base import ContractModel
from cwap_contracts.v2.agents import AgentMemoryConfig

#: Reasoning depths, cheapest first. The same vocabulary the proxy uses, so a
#: value set here means one thing on every backend that honours it.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


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
    #: Which backend serves this agent's model. Blank means the workspace's.
    model_provider: str = Field(default="", max_length=64)
    #: Run this agent on a different model than the workspace default — a cheap
    #: model for triage, a strong one for synthesis.
    model_override: str | None = Field(default=None, max_length=256)
    #: How hard this agent may think, where the backend has a reasoning mode.
    thinking_effort: str = Field(default="", max_length=16)
    version: int = Field(default=1, ge=1)

    @field_validator("thinking_effort")
    @classmethod
    def _known_effort(cls, value: str) -> str:
        if value and value not in EFFORT_LEVELS:
            raise ValueError(
                f"unknown thinking effort '{value}'; expected one of {list(EFFORT_LEVELS)}"
            )
        return value
