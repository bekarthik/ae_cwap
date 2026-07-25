"""The security anchor carried by every job: `JobContext`.

Mandate §1.3 — immutable metadata (`tenant_id`, `initiating_user_id`) plus the
scope of permissions the job requires. Both the Producer pre-check and the
Consumer runtime check authorise against this object, never against ambient
request state.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from cwap_contracts.v1.base import ContractModel


class PermissionRequirement(ContractModel):
    """The permissions a job batch needs in order to be allowed to run.

    Modelled as an explicit read scope plus an optional write scope so that an
    authorisation decision is a set-containment check, not string parsing.
    """

    required_scope: str = Field(
        ..., min_length=1, description="Read scope needed, e.g. 'READ_HR'."
    )
    required_write: str | None = Field(
        default=None, description="Write scope needed, e.g. 'WRITE_DB'. None = read-only job."
    )

    @field_validator("required_scope", "required_write")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value

    def as_set(self) -> frozenset[str]:
        scopes = {self.required_scope}
        if self.required_write:
            scopes.add(self.required_write)
        return frozenset(scopes)


class JobContext(ContractModel):
    """Immutable identity + permission envelope. Travels with every job batch."""

    tenant_id: str = Field(..., min_length=1)
    initiating_user_id: str = Field(..., min_length=1)
    permissions: PermissionRequirement
    trace_id: str = Field(..., min_length=1, description="Correlates logs across all services.")
    principal_kind: str = Field(
        default="user",
        pattern="^(user|service)$",
        description="Whether the initiating principal is an end user or a service account.",
    )
