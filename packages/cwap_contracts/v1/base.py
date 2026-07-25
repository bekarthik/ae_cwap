"""Base class shared by every v1 contract model."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

CONTRACT_VERSION = "v1"


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp. Contracts never carry naive datetimes."""
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    """Immutable, closed-world base model.

    `extra="forbid"` means an unrecognised field is a contract violation rather
    than something silently dropped — that is the whole point of the mandate.
    `frozen=True` means a payload cannot be mutated after it crosses a boundary,
    so a worker can never "fix up" a message in place and hide a schema drift.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )
