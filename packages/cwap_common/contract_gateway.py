"""The Dual Gateway — mandate §2.

Nothing enters or leaves the queue except through these two wrappers. That is
the entire point: validation and authorisation cannot be implemented ad hoc
inside individual workers, because individual workers never touch the broker.

    Producer  ──> schema validate ──> authorisation pre-check ──> enqueue
    Consumer  ──> schema validate ──> authorisation runtime check ──> execute
                        │                       │
                        └───────────┬───────────┘
                                    ▼
                          Dead Letter Queue (no retries)

Producer-side failures are synchronous: the caller gets a `400 Contract
Violation` / `403 Authorization Failure` and the message is never published.
Consumer-side failures are asynchronous: the payload is parked in the DLQ with
its precise reason and the main queue keeps flowing, so one malformed job cannot
starve the worker pool.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from cwap_contracts import (
    AuthorizationFailure,
    ContractViolation,
    CwapContractError,
    JobContext,
    ServiceName,
    WorkflowJobPayload,
)
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from cwap_common.authz import AuthorizationClient, LocalAuthorizationService
from cwap_common.broker import Broker, get_broker
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import DeadLetter
from cwap_common.settings import get_settings

TModel = TypeVar("TModel", bound=BaseModel)

AuthorizerFactory = Callable[[Session], AuthorizationClient]


def _default_authorizer(session: Session) -> AuthorizationClient:
    return LocalAuthorizationService(session)


def validate_payload(raw: str | bytes | dict[str, Any], contract: type[TModel]) -> TModel:
    """Parse and validate against a registered contract.

    Any failure — malformed JSON, missing field, unknown field, illegal state
    transition — surfaces as a single predictable `ContractViolation` carrying
    the structured Pydantic error detail.
    """
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractViolation(
            f"payload is not valid JSON: {exc}", detail={"contract": contract.__name__}
        ) from exc

    if not isinstance(data, dict):
        raise ContractViolation(
            f"payload must be a JSON object, got {type(data).__name__}",
            detail={"contract": contract.__name__},
        )

    try:
        return contract.model_validate(data)
    except ValidationError as exc:
        raise ContractViolation(
            f"payload does not satisfy contract {contract.__name__}",
            detail={"contract": contract.__name__, "errors": exc.errors(include_url=False)},
        ) from exc


class DeadLetterSink:
    """Observable parking for messages that must never be retried.

    Writes to two places on purpose: the DLQ topic (so an operator's tooling can
    watch it) and the `dead_letters` table (so triage and resubmission have a
    durable record with the exact reason).
    """

    def __init__(self, broker: Broker | None = None, queue: str | None = None) -> None:
        settings = get_settings()
        self.broker = broker or get_broker()
        self.queue = queue or settings.dead_letter_queue

    def park(self, raw: str, error: CwapContractError, *, source_queue: str) -> int:
        run_id = _peek_run_id(raw)
        envelope = json.dumps(
            {
                "source_queue": source_queue,
                "reason_code": error.reason_code,
                "reason": error.message,
                "detail": error.detail,
                "payload": raw,
            }
        )
        self.broker.publish(self.queue, envelope)

        with unit_of_work() as session:
            record = DeadLetter(
                queue=source_queue,
                reason_code=error.reason_code,
                reason=error.message,
                run_id=run_id,
                payload=raw,
                detail=_jsonable(error.detail),
            )
            session.add(record)
            session.flush()
            return record.id


def _peek_run_id(raw: str) -> str | None:
    """Best-effort run correlation for a payload we could not validate."""
    try:
        data = json.loads(raw)
        value = data.get("run_id")
        return value if isinstance(value, str) else None
    except Exception:
        return None


def _jsonable(detail: Any) -> dict[str, Any] | None:
    if detail is None:
        return None
    if isinstance(detail, dict):
        try:
            json.dumps(detail, default=str)
            return json.loads(json.dumps(detail, default=str))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {"detail": str(detail)}
    return {"detail": str(detail)}


class ProducerWrapper:
    """Mandate §2.A — the only sanctioned way to enqueue a job.

    Both checks run before anything is written to the broker, so a rejected job
    leaves no trace on the queue at all.
    """

    def __init__(
        self,
        *,
        broker: Broker | None = None,
        queue: str | None = None,
        authorizer_factory: AuthorizerFactory = _default_authorizer,
        contract: type[BaseModel] = WorkflowJobPayload,
    ) -> None:
        settings = get_settings()
        self.broker = broker or get_broker()
        self.queue = queue or settings.work_queue
        self.authorizer_factory = authorizer_factory
        self.contract = contract

    def publish(self, payload: WorkflowJobPayload | dict[str, Any]) -> WorkflowJobPayload:
        # 1. Schema validation. Re-validated even when handed a model instance,
        #    so a caller cannot smuggle through a subclass or a stale version.
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        validated = validate_payload(body, self.contract)
        assert isinstance(validated, WorkflowJobPayload)

        # 2. Authorisation pre-check against the action this job will perform.
        self.assert_authorized(
            validated.job_context, validated.next_step_definition.target_service
        )

        # 3. Only now does it become a message.
        self.broker.publish(self.queue, validated.model_dump_json())
        return validated

    def assert_authorized(self, context: JobContext, target_service: ServiceName) -> None:
        with read_only_session() as session:
            decision = self.authorizer_factory(session).authorize(context, target_service)
        if not decision.allowed:
            raise AuthorizationFailure(
                decision.reason,
                detail={
                    "tenant_id": context.tenant_id,
                    "initiating_user_id": context.initiating_user_id,
                    "target_service": target_service.value,
                    "required": sorted(context.permissions.as_set()),
                },
            )


class ConsumerWrapper:
    """Mandate §2.B — the only sanctioned way to take a job off the queue.

    `next()` returns a payload that has *already* passed schema validation and a
    fresh authorisation check, or `None`. A `None` means either "queue empty" or
    "that message was dead-lettered"; in neither case does the worker see an
    unvalidated dict.
    """

    def __init__(
        self,
        *,
        broker: Broker | None = None,
        queue: str | None = None,
        dead_letters: DeadLetterSink | None = None,
        authorizer_factory: AuthorizerFactory = _default_authorizer,
        contract: type[BaseModel] = WorkflowJobPayload,
    ) -> None:
        settings = get_settings()
        self.broker = broker or get_broker()
        self.queue = queue or settings.work_queue
        self.dead_letters = dead_letters or DeadLetterSink(self.broker)
        self.authorizer_factory = authorizer_factory
        self.contract = contract
        #: Incremented whenever a message is parked; surfaced on /admin/dlq.
        self.dead_lettered = 0

    def next(self, timeout: float = 0.0) -> WorkflowJobPayload | None:
        raw = self.broker.consume(self.queue, timeout=timeout)
        if raw is None:
            return None
        return self.accept(raw)

    def accept(self, raw: str) -> WorkflowJobPayload | None:
        """Run both consumer-side checks on one raw message.

        No internal retries on failure — a contract violation is deterministic
        and an authorisation revocation is intentional. Retrying either one just
        burns worker capacity.
        """
        try:
            validated = validate_payload(raw, self.contract)
            assert isinstance(validated, WorkflowJobPayload)

            with read_only_session() as session:
                decision = self.authorizer_factory(session).authorize(
                    validated.job_context, validated.next_step_definition.target_service
                )
            if not decision.allowed:
                raise AuthorizationFailure(
                    decision.reason,
                    detail={
                        "run_id": validated.run_id,
                        "step_execution_id": validated.step_execution_id,
                        "target_service": validated.next_step_definition.target_service.value,
                    },
                )
            return validated
        except CwapContractError as error:
            self.dead_letters.park(raw, error, source_queue=self.queue)
            self.dead_lettered += 1
            return None
