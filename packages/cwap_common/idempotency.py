"""Resilience Core — idempotent writes and the external-call gate.

Mandate §3 in code:

* §3.A  `IdempotencyGate` implements the two-phase PRE-CHECK / EXECUTE & COMMIT
        sequence for third-party calls, keyed by (run_id, step_execution_id).
* §3.B  `commit_step_output` writes node state with `ON CONFLICT DO NOTHING`, so
        first write wins no matter how many times Celery redelivers.
* §3.C  Callers run all of it inside `db.unit_of_work()`.

All statements are dialect-aware: PostgreSQL in production, SQLite for dev/test.
Both support `ON CONFLICT`, so the semantics are identical in either.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cwap_contracts.v2 import LogEvent, StepOutputContext
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from cwap_common.models import IdempotencyLedger, RunLog, WorkflowExecutionState

CLAIM_IN_FLIGHT = "IN_FLIGHT"
CLAIM_SUCCEEDED = "SUCCEEDED"
CLAIM_FAILED = "FAILED"


def _insert_for(session: Session):
    """Pick the dialect-specific INSERT that exposes `on_conflict_*`."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return pg_insert
    if dialect == "sqlite":
        return sqlite_insert
    raise RuntimeError(
        f"dialect '{dialect}' has no supported ON CONFLICT form; "
        "idempotent writes require PostgreSQL or SQLite"
    )


def _inserted(session: Session, stmt) -> bool:
    """Whether an `ON CONFLICT DO NOTHING` statement actually wrote a row.

    Via `RETURNING`, not `rowcount`. `rowcount` looks like the obvious answer and
    is wrong on PostgreSQL: psycopg reports `-1` for an INSERT without RETURNING,
    so `rowcount > 0` is *always false* there. Every caller below then reads
    "this was a duplicate" on a row it had just successfully written — which on
    the production database meant runs never recorded their inputs, and the
    external-call gate treated every first attempt as already in flight and
    skipped the call. SQLite reports rowcount correctly, so the whole class of
    failure is invisible to a SQLite-only test suite.

    RETURNING is the exact question being asked — "did this statement produce a
    row?" — and both dialects support it (SQLite since 3.35).
    """
    return session.execute(stmt).first() is not None


def commit_step_output(session: Session, output: StepOutputContext) -> bool:
    """Persist one node's result. Returns True if this call wrote the row.

    A False return is the normal, healthy outcome of a retry: the composite key
    already had a row, so the original output stands untouched.
    """
    insert = _insert_for(session)
    stmt = (
        insert(WorkflowExecutionState)
        .values(
            run_id=output.run_id,
            step_execution_id=output.step_execution_id,
            source_service=output.source_service.value,
            node_id=output.node_id,
            resulting_state=output.resulting_state.value,
            output=output.output,
            derived_context=output.derived_context,
            duration_ms=output.duration_ms,
            completed_at=output.completed_at,
        )
        .on_conflict_do_nothing(
            index_elements=["run_id", "step_execution_id", "source_service"]
        )
        .returning(WorkflowExecutionState.id)
    )
    return _inserted(session, stmt)


def append_log(session: Session, event: LogEvent) -> bool:
    """Persist a log line. Deduplicated on (run_id, seq) so a replayed step does
    not double-log into the run report."""
    insert = _insert_for(session)
    stmt = (
        insert(RunLog)
        .values(
            run_id=event.run_id,
            seq=event.seq,
            timestamp=event.timestamp,
            node=event.node,
            level=event.level.value,
            event=event.event,
            message=event.message,
            data=event.data,
        )
        .on_conflict_do_nothing(index_elements=["run_id", "seq"])
        .returning(RunLog.id)
    )
    return _inserted(session, stmt)


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of the PRE-CHECK phase."""

    #: True when this worker won the claim and must perform the side effect.
    should_execute: bool
    #: Populated when a previous attempt already succeeded — replay this instead.
    prior_response: dict[str, Any] | None = None
    #: True when another worker holds an in-flight claim for the same key.
    in_flight_elsewhere: bool = False


class IdempotencyGate:
    """Two-phase commit check around a single external side effect.

    Usage inside a worker::

        gate = IdempotencyGate(session, run_id, step_execution_id, "http_request")
        claim = gate.pre_check()
        if not claim.should_execute:
            return claim.prior_response          # already done; do not call again
        response = call_the_third_party(...)
        gate.commit(response, external_ref=response.get("id"))
    """

    def __init__(
        self, session: Session, run_id: str, step_execution_id: str, operation: str
    ) -> None:
        self.session = session
        self.run_id = run_id
        self.step_execution_id = step_execution_id
        self.operation = operation

    def _existing(self) -> IdempotencyLedger | None:
        return (
            self.session.query(IdempotencyLedger)
            .filter_by(
                run_id=self.run_id,
                step_execution_id=self.step_execution_id,
                operation=self.operation,
            )
            .one_or_none()
        )

    def pre_check(self) -> ClaimResult:
        """PRE-CHECK: has this exact (run, step, operation) already succeeded?

        Claims the key atomically when it has not, so two concurrent workers
        racing on the same redelivered message cannot both call outward.
        """
        insert = _insert_for(self.session)
        stmt = (
            insert(IdempotencyLedger)
            .values(
                run_id=self.run_id,
                step_execution_id=self.step_execution_id,
                operation=self.operation,
                status=CLAIM_IN_FLIGHT,
            )
            .on_conflict_do_nothing(
                index_elements=["run_id", "step_execution_id", "operation"]
            )
            .returning(IdempotencyLedger.id)
        )
        won_claim = _inserted(self.session, stmt)
        if won_claim:
            return ClaimResult(should_execute=True)

        existing = self._existing()
        if existing is None:  # pragma: no cover - only under concurrent delete
            return ClaimResult(should_execute=True)

        if existing.status == CLAIM_SUCCEEDED:
            return ClaimResult(should_execute=False, prior_response=existing.response or {})
        if existing.status == CLAIM_FAILED:
            # A previously failed attempt may be retried: re-open the claim.
            self.session.execute(
                update(IdempotencyLedger)
                .where(IdempotencyLedger.id == existing.id)
                .where(IdempotencyLedger.status == CLAIM_FAILED)
                .values(status=CLAIM_IN_FLIGHT, updated_at=datetime.now(timezone.utc))
            )
            return ClaimResult(should_execute=True)

        return ClaimResult(should_execute=False, in_flight_elsewhere=True)

    def commit(
        self, response: dict[str, Any], *, external_ref: str | None = None
    ) -> None:
        """EXECUTE & COMMIT phase: record success before the job is acked.

        The guarded `WHERE status = IN_FLIGHT` makes this non-destructive — it
        cannot clobber a claim another worker already resolved.
        """
        self.session.execute(
            update(IdempotencyLedger)
            .where(IdempotencyLedger.run_id == self.run_id)
            .where(IdempotencyLedger.step_execution_id == self.step_execution_id)
            .where(IdempotencyLedger.operation == self.operation)
            .where(IdempotencyLedger.status == CLAIM_IN_FLIGHT)
            .values(
                status=CLAIM_SUCCEEDED,
                response=response,
                external_ref=external_ref,
                updated_at=datetime.now(timezone.utc),
            )
        )

    def abandon(self, reason: str) -> None:
        """Release the claim after a failed side effect so a retry may proceed."""
        self.session.execute(
            update(IdempotencyLedger)
            .where(IdempotencyLedger.run_id == self.run_id)
            .where(IdempotencyLedger.step_execution_id == self.step_execution_id)
            .where(IdempotencyLedger.operation == self.operation)
            .where(IdempotencyLedger.status == CLAIM_IN_FLIGHT)
            .values(
                status=CLAIM_FAILED,
                response={"error": reason},
                updated_at=datetime.now(timezone.utc),
            )
        )
