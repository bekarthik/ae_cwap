"""Resilience Core (mandate §3): idempotent writes, the 2PC gate, atomicity."""

from __future__ import annotations

import pytest
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.idempotency import (
    CLAIM_SUCCEEDED,
    IdempotencyGate,
    append_log,
    commit_step_output,
)
from cwap_common.models import IdempotencyLedger, RunLog, WorkflowExecutionState
from cwap_contracts.v2 import ExecutionState, LogEvent, ServiceName, StepOutputContext


def step_output(**overrides) -> StepOutputContext:
    base = dict(
        run_id="run_1",
        step_execution_id="se_1",
        source_service=ServiceName.LLM_PROXY,
        node_id="think",
        resulting_state=ExecutionState.LLM_INFERENCE_COMPLETE,
        output={"text": "first"},
    )
    base.update(overrides)
    return StepOutputContext(**base)


def log_event(**overrides) -> LogEvent:
    base = dict(run_id="run_1", seq=1, event="step.started")
    base.update(overrides)
    return LogEvent(**base)


class TestTheWriteSignalIsDialectNeutral:
    """How "did this write?" is answered, not just what it answered here.

    Found by running this suite against a real PostgreSQL. `rowcount` is the
    obvious way to ask, works on SQLite, and is *always false* on PostgreSQL —
    psycopg reports `-1` for an INSERT without RETURNING. Every caller then read
    "duplicate" on a row it had just written: runs never recorded their inputs,
    and the external-call gate declared every first attempt already in flight and
    skipped the call. None of it was visible from SQLite.

    So this asserts on the mechanism. `CWAP_TEST_DATABASE_URL` runs the whole
    suite against PostgreSQL and catches the behaviour directly; these two keep
    the fix from being undone in a SQLite-only run.
    """

    @pytest.mark.parametrize(
        ("name", "invoke"),
        [
            ("commit_step_output", lambda session: commit_step_output(session, step_output())),
            ("append_log", lambda session: append_log(session, log_event())),
            (
                "IdempotencyGate.pre_check",
                lambda session: IdempotencyGate(session, "run_1", "se_1", "op").pre_check(),
            ),
        ],
    )
    def test_every_conflict_write_asks_via_returning(self, name, invoke):
        """Inspects the statement the production code actually executes."""
        recorder = _StatementRecorder()
        invoke(recorder)

        assert recorder.statements, f"{name} executed no statement"
        compiled = str(recorder.statements[0]).upper()
        assert "ON CONFLICT" in compiled
        assert "RETURNING" in compiled, (
            f"{name} decides its return value from rowcount, which is always -1 "
            "on PostgreSQL — use RETURNING"
        )

    def test_the_write_signal_ignores_a_misleading_rowcount(self):
        """A statement that returned no row must read as "did not write", even
        when the driver reports a healthy-looking rowcount."""
        from cwap_common.idempotency import _inserted

        assert _inserted(_StatementRecorder(returns=None), object()) is False
        assert _inserted(_StatementRecorder(returns=(1,)), object()) is True


class _StatementRecorder:
    """A session stand-in that captures statements instead of running them."""

    def __init__(self, returns: object = (1,)) -> None:
        self.statements: list[object] = []
        self._returns = returns

    def execute(self, statement, *args, **kwargs):
        self.statements.append(statement)
        return self

    # Consumed by `_inserted`; `rowcount` is deliberately plausible-looking, so a
    # reversion to reading it would pass here and fail on PostgreSQL — which is
    # exactly the trap this class exists to close.
    rowcount = 99

    def first(self):
        return self._returns

    def get_bind(self):
        from cwap_common.db import get_engine

        return get_engine()

    def query(self, *args, **kwargs):  # pre_check falls through to this on a miss
        raise AssertionError("pre_check should not query after winning the claim")


class TestStepOutputWrites:
    def test_first_write_commits(self):
        with unit_of_work() as session:
            assert commit_step_output(session, step_output()) is True

    def test_retry_is_a_no_op_and_the_original_output_stands(self):
        """§3.B — first write wins, however many times Celery redelivers."""
        with unit_of_work() as session:
            commit_step_output(session, step_output(output={"text": "first"}))
        with unit_of_work() as session:
            assert commit_step_output(session, step_output(output={"text": "second"})) is False

        with read_only_session() as session:
            rows = session.query(WorkflowExecutionState).all()
        assert len(rows) == 1
        assert rows[0].output == {"text": "first"}

    def test_a_different_step_of_the_same_run_is_a_separate_row(self):
        with unit_of_work() as session:
            commit_step_output(session, step_output())
            commit_step_output(session, step_output(step_execution_id="se_2", node_id="draft"))
        with read_only_session() as session:
            assert session.query(WorkflowExecutionState).count() == 2

    def test_same_step_from_a_different_service_is_a_separate_row(self):
        """The composite key includes source_service, per §3.B."""
        with unit_of_work() as session:
            commit_step_output(session, step_output())
            assert (
                commit_step_output(
                    session, step_output(source_service=ServiceName.TRANSFORM_ENGINE)
                )
                is True
            )


class TestLogWrites:
    def test_replayed_log_line_is_deduplicated(self):
        event = LogEvent(run_id="run_1", seq=1, event="step.started")
        with unit_of_work() as session:
            assert append_log(session, event) is True
        with unit_of_work() as session:
            assert append_log(session, event) is False
        with read_only_session() as session:
            assert session.query(RunLog).count() == 1


class TestIdempotencyGate:
    def test_first_caller_wins_the_claim(self):
        with unit_of_work() as session:
            claim = IdempotencyGate(session, "run_1", "se_1", "http_request").pre_check()
        assert claim.should_execute

    def test_second_caller_while_in_flight_is_told_to_stand_down(self):
        with unit_of_work() as session:
            IdempotencyGate(session, "run_1", "se_1", "http_request").pre_check()
        with unit_of_work() as session:
            claim = IdempotencyGate(session, "run_1", "se_1", "http_request").pre_check()
        assert not claim.should_execute
        assert claim.in_flight_elsewhere

    def test_completed_operation_replays_instead_of_calling_again(self):
        """§3.A PRE-CHECK: a redelivered job must not double-charge a payment."""
        with unit_of_work() as session:
            gate = IdempotencyGate(session, "run_1", "se_1", "http_request")
            gate.pre_check()
            gate.commit({"status": 201, "id": "charge_1"}, external_ref="charge_1")

        with unit_of_work() as session:
            claim = IdempotencyGate(session, "run_1", "se_1", "http_request").pre_check()

        assert not claim.should_execute
        assert claim.prior_response == {"status": 201, "id": "charge_1"}

    def test_commit_is_guarded_and_cannot_clobber_a_resolved_claim(self):
        with unit_of_work() as session:
            gate = IdempotencyGate(session, "run_1", "se_1", "http_request")
            gate.pre_check()
            gate.commit({"status": 200, "attempt": 1})
        with unit_of_work() as session:
            IdempotencyGate(session, "run_1", "se_1", "http_request").commit(
                {"status": 500, "attempt": 2}
            )
        with read_only_session() as session:
            record = session.query(IdempotencyLedger).one()
        assert record.status == CLAIM_SUCCEEDED
        assert record.response == {"status": 200, "attempt": 1}

    def test_abandoned_claim_lets_a_genuine_retry_proceed(self):
        with unit_of_work() as session:
            gate = IdempotencyGate(session, "run_1", "se_1", "http_request")
            gate.pre_check()
            gate.abandon("connection reset")
        with unit_of_work() as session:
            claim = IdempotencyGate(session, "run_1", "se_1", "http_request").pre_check()
        assert claim.should_execute

    def test_distinct_operations_on_one_step_are_independent(self):
        with unit_of_work() as session:
            assert IdempotencyGate(session, "run_1", "se_1", "http_request").pre_check()
            assert IdempotencyGate(session, "run_1", "se_1", "send_email").pre_check()


class TestTransactionalBoundary:
    def test_a_failure_mid_step_rolls_back_every_write(self):
        """§3.C — no partial state, no orphaned log rows."""
        with pytest.raises(RuntimeError, match="boom"), unit_of_work() as session:
            commit_step_output(session, step_output())
            append_log(session, LogEvent(run_id="run_1", seq=1, event="step.started"))
            raise RuntimeError("boom")

        with read_only_session() as session:
            assert session.query(WorkflowExecutionState).count() == 0
            assert session.query(RunLog).count() == 0
