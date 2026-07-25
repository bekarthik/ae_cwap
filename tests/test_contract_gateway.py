"""The Dual Gateway (mandate §2) and the Dead Letter Queue (§2.C)."""

from __future__ import annotations

import json

import pytest
from cwap_common.authz import LocalAuthorizationService
from cwap_common.contract_gateway import (
    ConsumerWrapper,
    DeadLetterSink,
    ProducerWrapper,
    validate_payload,
)
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import DeadLetter, User
from cwap_common.settings import get_settings
from cwap_contracts import (
    AuthorizationFailure,
    ContractViolation,
    ExecutionState,
    JobContext,
    NextStepDefinition,
    PermissionRequirement,
    ServiceName,
    WorkflowJobPayload,
)


def make_payload(context: JobContext, *, service=ServiceName.LLM_PROXY) -> WorkflowJobPayload:
    return WorkflowJobPayload(
        run_id="run_1",
        step_execution_id="se_1",
        workflow_id="wf_1",
        node_id="think",
        job_context=context,
        current_state=ExecutionState.INPUT_RESOLVED,
        next_step_definition=NextStepDefinition(
            target_service=service, target_node_id="think"
        ),
    )


@pytest.fixture
def user_context() -> JobContext:
    with unit_of_work() as session:
        session.add(
            User(
                id="usr_1",
                email="a@example.com",
                tenant_id="tenant-a",
                password_hash="unused",
                scopes=["READ_WORKFLOWS"],
            )
        )
    return JobContext(
        tenant_id="tenant-a",
        initiating_user_id="usr_1",
        permissions=PermissionRequirement(required_scope="READ_WORKFLOWS"),
        trace_id="tr_1",
    )


class TestValidation:
    def test_malformed_json_is_a_contract_violation(self):
        with pytest.raises(ContractViolation, match="not valid JSON"):
            validate_payload("{not json", WorkflowJobPayload)

    def test_json_array_is_a_contract_violation(self):
        with pytest.raises(ContractViolation, match="must be a JSON object"):
            validate_payload("[1,2,3]", WorkflowJobPayload)

    def test_schema_mismatch_carries_structured_detail(self, user_context):
        body = make_payload(user_context).model_dump(mode="json")
        del body["node_id"]
        with pytest.raises(ContractViolation) as excinfo:
            validate_payload(body, WorkflowJobPayload)
        assert excinfo.value.reason_code == "CONTRACT_VIOLATION"
        assert excinfo.value.detail["errors"]


class TestProducerWrapper:
    def test_valid_payload_reaches_the_broker(self, user_context, broker):
        ProducerWrapper().publish(make_payload(user_context))
        assert broker.depth(get_settings().work_queue) == 1

    def test_invalid_payload_never_becomes_a_message(self, user_context, broker):
        body = make_payload(user_context).model_dump(mode="json")
        body["current_state"] = "NOT_A_STATE"
        with pytest.raises(ContractViolation):
            ProducerWrapper().publish(body)
        assert broker.depth(get_settings().work_queue) == 0

    def test_unauthorised_payload_never_becomes_a_message(self, user_context, broker):
        """The pre-check: HTTP_CONNECTOR demands a declared write scope."""
        with pytest.raises(AuthorizationFailure, match="write scope"):
            ProducerWrapper().publish(
                make_payload(user_context, service=ServiceName.HTTP_CONNECTOR)
            )
        assert broker.depth(get_settings().work_queue) == 0

    def test_unknown_principal_is_rejected(self, broker):
        context = JobContext(
            tenant_id="tenant-a",
            initiating_user_id="ghost",
            permissions=PermissionRequirement(required_scope="READ_WORKFLOWS"),
            trace_id="tr_1",
        )
        with pytest.raises(AuthorizationFailure, match="not a member of tenant"):
            ProducerWrapper().publish(make_payload(context))
        assert broker.depth(get_settings().work_queue) == 0


class TestConsumerWrapper:
    def test_valid_message_is_returned_as_a_model(self, user_context):
        consumer = ConsumerWrapper()
        ProducerWrapper().publish(make_payload(user_context))
        payload = consumer.next()
        assert isinstance(payload, WorkflowJobPayload)
        assert payload.node_id == "think"

    def test_empty_queue_returns_none(self):
        assert ConsumerWrapper().next() is None

    def test_malformed_message_is_dead_lettered_not_raised(self, broker):
        settings = get_settings()
        broker.publish(settings.work_queue, "{ not a payload }")
        consumer = ConsumerWrapper()

        assert consumer.next() is None
        assert consumer.dead_lettered == 1

        with read_only_session() as session:
            record = session.query(DeadLetter).one()
        assert record.reason_code == "CONTRACT_VIOLATION"
        assert broker.depth(settings.dead_letter_queue) == 1

    def test_revoked_permission_halts_the_job_at_consumption(self, user_context, broker):
        """The runtime re-check. The message was authorised when enqueued; by
        the time a worker picks it up the principal has lost its scopes."""
        ProducerWrapper().publish(make_payload(user_context))
        assert broker.depth(get_settings().work_queue) == 1

        with unit_of_work() as session:
            session.query(User).filter_by(id="usr_1").update({User.scopes: []})

        consumer = ConsumerWrapper()
        assert consumer.next() is None
        assert consumer.dead_lettered == 1

        with read_only_session() as session:
            record = session.query(DeadLetter).one()
        assert record.reason_code == "AUTHORIZATION_FAILURE"
        assert record.run_id == "run_1"

    def test_main_queue_keeps_flowing_past_a_poisoned_message(self, user_context, broker):
        """§2.C's whole point: one bad job must not starve the workers."""
        settings = get_settings()
        broker.publish(settings.work_queue, "garbage")
        ProducerWrapper().publish(make_payload(user_context))

        consumer = ConsumerWrapper()
        assert consumer.next() is None  # poisoned message parked
        good = consumer.next()
        assert good is not None and good.node_id == "think"

    def test_dead_letter_preserves_the_original_payload_for_resubmission(self, broker):
        broker.publish(get_settings().work_queue, '{"run_id": "run_9"}')
        ConsumerWrapper().next()

        with read_only_session() as session:
            record = session.query(DeadLetter).one()
        assert json.loads(record.payload)["run_id"] == "run_9"
        assert record.run_id == "run_9"


class TestAuthorizationService:
    def test_missing_scope_is_reported_specifically(self, user_context):
        with unit_of_work() as session:
            authorizer = LocalAuthorizationService(session)
            context = user_context.model_copy(
                update={
                    "permissions": PermissionRequirement(
                        required_scope="READ_HR", required_write="WRITE_DB"
                    )
                }
            )
            decision = authorizer.authorize(context, ServiceName.LLM_PROXY)
        assert not decision
        assert "READ_HR" in decision.reason

    def test_granted_scopes_pass(self, user_context):
        with unit_of_work() as session:
            decision = LocalAuthorizationService(session).authorize(
                user_context, ServiceName.LLM_PROXY
            )
        assert decision.allowed


class TestDeadLetterSink:
    def test_park_records_reason_and_detail(self, broker):
        sink = DeadLetterSink()
        record_id = sink.park(
            '{"run_id": "run_5"}',
            ContractViolation("bad shape", detail={"contract": "WorkflowJobPayload"}),
            source_queue="cwap.jobs",
        )
        with read_only_session() as session:
            record = session.get(DeadLetter, record_id)
        assert record.reason == "bad shape"
        assert record.detail == {"contract": "WorkflowJobPayload"}
