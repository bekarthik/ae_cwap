"""The Authorization Service.

Mandate §2: authorisation is checked *twice* — once by the Producer before a job
is enqueued, and again by the Consumer at the moment of execution. The second
check is the one that matters: a job can sit in a queue for minutes, and
permissions may have been revoked in the meantime.

Both checks go through this module, against `JobContext` only. No service is
permitted to authorise off ambient request state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from cwap_common.models import User
from cwap_contracts import JobContext, ServiceName

#: Minimum permission shape a target service demands of any job routed to it.
#: HTTP_CONNECTOR is the sharp one: a node that can cause an external side
#: effect may only run if the job explicitly declared a write scope.
SERVICE_REQUIRES_WRITE: frozenset[ServiceName] = frozenset({ServiceName.HTTP_CONNECTOR})


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str = ""
    principal_scopes: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return self.allowed


class AuthorizationClient(Protocol):
    """The interface both gateway wrappers depend on."""

    def authorize(
        self, context: JobContext, target_service: ServiceName
    ) -> AuthorizationDecision: ...


class LocalAuthorizationService:
    """Database-backed implementation.

    In a split deployment this becomes an HTTP client against a standalone
    Authorization Service; the interface is identical, which is why the gateway
    wrappers never need to change.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def _scopes_for(self, context: JobContext) -> frozenset[str] | None:
        """None means "no such principal in this tenant"."""
        if context.principal_kind == "service":
            # Service accounts are provisioned out of band and carry the scopes
            # they declare; the schema validation already pinned the shape.
            return context.permissions.as_set()

        user = (
            self.session.query(User)
            .filter_by(id=context.initiating_user_id, tenant_id=context.tenant_id)
            .one_or_none()
        )
        if user is None:
            return None
        return frozenset(scope.upper() for scope in (user.scopes or []))

    def authorize(
        self, context: JobContext, target_service: ServiceName
    ) -> AuthorizationDecision:
        required = context.permissions.as_set()

        if target_service in SERVICE_REQUIRES_WRITE and not context.permissions.required_write:
            return AuthorizationDecision(
                allowed=False,
                reason=(
                    f"{target_service.value} performs external side effects and requires the job "
                    "to declare a write scope, but job_context.permissions.required_write is unset"
                ),
            )

        held = self._scopes_for(context)
        if held is None:
            return AuthorizationDecision(
                allowed=False,
                reason=(
                    f"principal '{context.initiating_user_id}' is not a member of tenant "
                    f"'{context.tenant_id}' (revoked, deleted, or never existed)"
                ),
            )

        missing = required - held
        if missing:
            return AuthorizationDecision(
                allowed=False,
                reason=(
                    f"principal '{context.initiating_user_id}' is missing required scope(s): "
                    f"{sorted(missing)}"
                ),
                principal_scopes=held,
            )

        return AuthorizationDecision(allowed=True, principal_scopes=held)


class AlwaysAllow:
    """Escape hatch for isolated unit tests of non-security behaviour.

    Deliberately not wired into any service factory — a deployment cannot select
    it by configuration.
    """

    def authorize(
        self, context: JobContext, target_service: ServiceName
    ) -> AuthorizationDecision:
        return AuthorizationDecision(allowed=True, principal_scopes=context.permissions.as_set())
