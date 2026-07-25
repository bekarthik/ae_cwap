"""Skills — the capabilities an agent can invoke. Version 3.

v2's boundary was "a skill is declarative, never code", because the platform
*synthesises* skills and a model that had read a hostile document must not be
able to write Python into a privileged worker.

MCP does not weaken that boundary; it widens what "declarative" can point at. An
MCP skill is a reference — a server the tenant deliberately connected, plus a
tool name that server advertises. The platform never authors it: an operator or
a user connects a server, the tools are read from it, and each becomes a skill.
Synthesis still cannot produce one, for exactly the reason it cannot produce an
HTTP skill — a connection to something outside the platform is a decision a
human makes, with a credential behind it.

Two other changes come with it, both forced by real MCP servers:

* `SkillParameter.type` now admits `object` and `array`. A GitHub server's
  `create_pull_request` takes structured arguments, and flattening them to
  strings would make the tool unusable.
* An MCP skill carries the server's own `input_schema` and passes it to the
  model verbatim. A schema the platform paraphrased would drift from the one the
  server validates against, and the model would be told the wrong shape.
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
    #: One tool on a connected MCP server. The platform holds a reference, never
    #: an implementation — what the tool does is the server's business.
    MCP = "mcp"


SKILL_KINDS = frozenset(SkillKind)

#: Kinds that reach outside the platform, and therefore need a write scope on the
#: run and an allow-listed destination. Named here so the orchestrator, the skill
#: executor and the run authoriser cannot drift apart on the question.
SIDE_EFFECTING_KINDS = frozenset({SkillKind.HTTP, SkillKind.MCP})


class SkillOrigin(str, Enum):
    BUILTIN = "builtin"
    USER = "user"
    #: Created by the platform because an agent needed a capability that did not
    #: exist. Tracked separately so it can be reviewed, and so a run report can
    #: say where a capability came from.
    SYNTHESIZED = "synthesized"
    #: Imported from a connected MCP server. Distinct from USER because it is not
    #: hand-authored and is re-synced when the server's tool list changes.
    MCP = "mcp"


class SkillParameter(ContractModel):
    """One input a skill accepts.

    Kept small for skills the platform authors — this becomes a tool schema
    handed to a model, and models follow simple schemas more reliably. `object`
    and `array` exist for MCP tools, whose shape the platform does not choose.
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    type: str = Field(
        default="string", pattern="^(string|number|integer|boolean|object|array)$"
    )
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
    #: Kind-specific and closed: `prompt_template`, `knowledge_handle`, `url`,
    #: `server_id` + `tool_name`, …
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
            SkillKind.MCP: ("server_id", "tool_name"),
        }[self.kind]

        missing = [key for key in required_keys if not self.definition.get(key)]
        if missing:
            raise ValueError(
                f"skill '{self.name}' is a {self.kind.value} skill and needs "
                f"{list(required_keys)}; missing {missing}"
            )
        return self

    @property
    def is_side_effecting(self) -> bool:
        """Whether invoking this can change something outside the platform."""
        return self.kind in SIDE_EFFECTING_KINDS

    def tool_schema(self) -> dict[str, Any]:
        """The JSON-schema tool definition handed to a model.

        An MCP tool's schema is passed through verbatim. Paraphrasing it would
        drift from the schema the server actually validates against, and the
        model would be told a shape that then gets rejected.
        """
        supplied = self.definition.get("input_schema")
        if isinstance(supplied, dict) and supplied:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": supplied,
            }

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

    It also cannot name an MCP server. Connecting to something outside the
    platform is a decision a human makes with a credential behind it, so a
    proposal that reaches for one is rejected at the contract — the same reason
    an HTTP proposal is refused in the synthesiser.
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(..., min_length=1, max_length=1000)
    kind: SkillKind
    parameters: list[SkillParameter] = Field(default_factory=list, max_length=8)
    definition: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _may_not_claim_a_connection(self) -> SkillProposal:
        if self.kind is SkillKind.MCP:
            raise ValueError(
                "a proposed skill may not be an MCP tool — connect the server "
                "first, and its tools become skills"
            )
        return self
