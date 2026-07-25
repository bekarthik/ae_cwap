"""Typed failures raised by the contract layer.

These exist so that a boundary violation surfaces as a *predictable, early*
error instead of an arbitrary runtime exception deep inside a worker.
"""

from __future__ import annotations

from typing import Any


class CwapContractError(Exception):
    """Base class for every failure produced by the contract/enforcement layer."""

    reason_code: str = "CONTRACT_ERROR"
    http_status: int = 400

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "detail": self.detail,
        }


class ContractViolation(CwapContractError):
    """Payload failed validation against its registered Pydantic schema.

    Maps to `400 Bad Request: Contract Violation` at an HTTP boundary and routes
    straight to the Dead Letter Queue at a queue boundary — never retried,
    because a malformed payload will stay malformed.
    """

    reason_code = "CONTRACT_VIOLATION"
    http_status = 400


class AuthorizationFailure(CwapContractError):
    """The identity context is missing, expired, or lacks the required scope.

    Raised by both the Producer pre-check and the Consumer runtime re-check. A
    runtime failure means permissions were revoked mid-flight; the job halts
    immediately and is dead-lettered rather than retried.
    """

    reason_code = "AUTHORIZATION_FAILURE"
    http_status = 403


class StateTransitionError(ContractViolation):
    """`current_state` is not a legal predecessor for `next_step_definition`.

    This is the determinism guard: it prevents a job from pathing to an
    arbitrary service that the state machine never authorised.
    """

    reason_code = "ILLEGAL_STATE_TRANSITION"


class SchemaNotRegistered(CwapContractError):
    """A payload referenced a contract name/version that the registry does not host."""

    reason_code = "SCHEMA_NOT_REGISTERED"
    http_status = 500
