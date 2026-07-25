"""Memory — what the system carries forward.

Three scopes, because they answer different questions and have different
lifetimes:

* **Workflow** — "what do I know about this job?" Facts and preferences that
  apply every time this workflow runs.
* **Agent** — "what have I learned about doing my role?" Accumulates across every
  workflow the agent appears in.
* **Skill** — "what have I learned about using this capability?" Accumulates
  across every agent that invokes it.

Separating them is what makes the learning transferable. A lesson about how to
phrase a retrieval query belongs to the *skill*, so every agent that uses it
benefits — putting it on the workflow would relearn it once per workflow.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from cwap_contracts.v1.base import ContractModel, utcnow


class MemoryScope(str, Enum):
    WORKFLOW = "workflow"
    AGENT = "agent"
    SKILL = "skill"
    #: Working memory for one execution. Not carried forward.
    RUN = "run"


class MemoryKind(str, Enum):
    #: Something that worked, or a rule derived from experience.
    LEARNING = "learning"
    #: Something durable about the domain.
    FACT = "fact"
    #: Something that went wrong, kept so it is not repeated.
    FAILURE = "failure"
    #: A stated preference from the user.
    PREFERENCE = "preference"


class MemoryEntry(ContractModel):
    id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    scope: MemoryScope
    scope_id: str = Field(..., min_length=1)
    kind: MemoryKind = MemoryKind.LEARNING
    text: str = Field(..., min_length=1, max_length=4000)
    #: The run that produced this, so a bad lesson can be traced to its source.
    source_run_id: str | None = None
    #: Raised when recalling an entry preceded a good outcome, lowered when it
    #: preceded a failure. Low-value memories fall out of recall rather than
    #: accumulating forever.
    usefulness: float = Field(default=1.0, ge=0.0, le=10.0)
    created_at: datetime = Field(default_factory=utcnow)


class MemoryWriteRequest(ContractModel):
    tenant_id: str = Field(..., min_length=1)
    scope: MemoryScope
    scope_id: str = Field(..., min_length=1)
    kind: MemoryKind = MemoryKind.LEARNING
    text: str = Field(..., min_length=1, max_length=4000)
    source_run_id: str | None = None


class RecallRequest(ContractModel):
    tenant_id: str = Field(..., min_length=1)
    scope: MemoryScope
    scope_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)


class RecallResult(ContractModel):
    scope: MemoryScope
    scope_id: str
    entries: list[MemoryEntry] = Field(default_factory=list)

    def as_prompt_block(self, heading: str = "What you have learned before") -> str:
        """Render for injection into a system prompt. Empty when nothing recalled,
        so an agent with no history gets no confusing empty section."""
        if not self.entries:
            return ""
        lines = "\n".join(f"- ({entry.kind.value}) {entry.text}" for entry in self.entries)
        return f"{heading}:\n{lines}"
