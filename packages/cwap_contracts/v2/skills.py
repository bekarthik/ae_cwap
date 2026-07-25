"""Skills — the capabilities an agent can invoke.

A skill is **declarative, never code**. That is a deliberate security boundary,
because the platform synthesises skills it does not already have: if synthesis
emitted Python, a model that had read a malicious document could write arbitrary
code into a privileged worker.

Instead a synthesised skill is a composition of primitives the platform already
trusts — a parameterised model call, a retrieval over a corpus the tenant owns, a
text transform, or an HTTP call that still passes the egress allow-list and still
requires the caller to hold a write scope. Everything a synthesised skill can do,
a user could already have built by hand on the canvas.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from cwap_contracts.v1.base import ContractModel


class SkillKind(str, Enum):
    """The primitives a skill can be built from."""

    #: A parameterised model call. The workhorse.
    PROMPT = "prompt"
    #: Semantic search over a knowledge corpus the tenant owns.
    RETRIEVAL = "retrieval"
    #: An HTTP call. Still allow-listed, still needs a write scope.
    HTTP = "http"
    #: Pure text shaping — no model, no network.
    TRANSFORM = "transform"
    #: An ordered sequence of other skills.
    COMPOSITE = "composite"


SKILL_KINDS = frozenset(SkillKind)


class SkillOrigin(str, Enum):
    BUILTIN = "builtin"
    USER = "user"
    #: Created by the platform because an agent needed a capability that did not
    #: exist. Tracked separately so it can be reviewed, and so a run report can
    #: say where a capability came from.
    SYNTHESIZED = "synthesized"


class SkillParameter(ContractModel):
    """One input a skill accepts. Kept deliberately small — this becomes a tool
    schema handed to a model, and models follow simple schemas more reliably."""

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    type: str = Field(default="string", pattern="^(string|number|integer|boolean)$")
    description: str = Field(default="", max_length=400)
    required: bool = True


class SkillDefinition(ContractModel):
    """A capability, as stored and as offered to a model."""

    id: str = Field(..., min_length=1, max_length=64)
    tenant_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(..., min_length=1, max_length=1000)
    kind: SkillKind
    parameters: list[SkillParameter] = Field(default_factory=list)
    #: Kind-specific and closed: `prompt_template`, `knowledge_handle`, `url`, …
    definition: dict[str, Any] = Field(default_factory=dict)
    origin: SkillOrigin = SkillOrigin.USER
    version: int = Field(default=1, ge=1)
    #: How many times this skill has been invoked, and how often it succeeded.
    #: Surfaced so a synthesised skill that never works is visible rather than
    #: quietly re-selected forever.
    invocations: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _definition_matches_kind(self) -> SkillDefinition:
        required_keys = {
            SkillKind.PROMPT: ("prompt_template",),
            SkillKind.RETRIEVAL: ("knowledge_handle",),
            SkillKind.HTTP: ("url", "method"),
            SkillKind.TRANSFORM: ("template",),
            SkillKind.COMPOSITE: ("steps",),
        }[self.kind]

        missing = [key for key in required_keys if not self.definition.get(key)]
        if missing:
            raise ValueError(
                f"skill '{self.name}' is a {self.kind.value} skill and needs "
                f"{list(required_keys)}; missing {missing}"
            )
        return self

    def tool_schema(self) -> dict[str, Any]:
        """The JSON-schema tool definition handed to a model."""
        properties = {
            parameter.name: {
                "type": parameter.type,
                "description": parameter.description or parameter.name,
            }
            for parameter in self.parameters
        }
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": [p.name for p in self.parameters if p.required],
            },
        }

    @property
    def reliability(self) -> float:
        """Fraction of invocations that did not fail. 1.0 when never used."""
        if not self.invocations:
            return 1.0
        return round(1.0 - (self.failures / self.invocations), 3)


class SkillProposal(ContractModel):
    """What the synthesiser is allowed to return.

    Deliberately narrower than `SkillDefinition`: no id, no tenant, no origin, no
    counters. The platform assigns those, so a proposal cannot claim to be a
    builtin or overwrite an existing skill's history.
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(..., min_length=1, max_length=1000)
    kind: SkillKind
    parameters: list[SkillParameter] = Field(default_factory=list, max_length=8)
    definition: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=1000)
