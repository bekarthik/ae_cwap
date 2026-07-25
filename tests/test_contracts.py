"""Contract registry and payload-schema tests.

Mandate §1: "schema adherence [is] a primary and mandatory unit test target."
"""

from __future__ import annotations

import json

import pytest
from cwap_contracts.errors import SchemaNotRegistered
from cwap_contracts.registry import (
    CONTRACT_REGISTRY,
    fingerprint,
    latest_version,
    lock_snapshot,
    read_lock,
    resolve,
)
from cwap_contracts.v2 import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    ExecutionState,
    JobContext,
    NextStepDefinition,
    PermissionRequirement,
    ServiceName,
    WorkflowJobPayload,
)
from pydantic import ValidationError


def _context(**overrides) -> JobContext:
    base = dict(
        tenant_id="tenant-a",
        initiating_user_id="usr_1",
        permissions=PermissionRequirement(required_scope="READ_HR"),
        trace_id="tr_1",
    )
    base.update(overrides)
    return JobContext(**base)


def _payload(**overrides) -> WorkflowJobPayload:
    base = dict(
        run_id="run_1",
        step_execution_id="se_1",
        workflow_id="wf_1",
        node_id="think",
        job_context=_context(),
        current_state=ExecutionState.INPUT_RESOLVED,
        next_step_definition=NextStepDefinition(
            target_service=ServiceName.LLM_PROXY, target_node_id="think"
        ),
    )
    base.update(overrides)
    return WorkflowJobPayload(**base)


class TestRegistryLock:
    def test_every_registered_contract_matches_the_approved_lock(self):
        """The version-bump gate.

        If this fails, a contract's shape changed without an approved version
        bump. The fix is to bump the version and re-run
        `python -m cwap_contracts.registry`, in a reviewed commit.
        """
        assert read_lock(), "contracts.lock.json is missing — run the registry module"
        assert lock_snapshot() == read_lock()

    def test_fingerprint_is_stable_across_calls(self):
        model = CONTRACT_REGISTRY["WorkflowJobPayload@v1"]
        assert fingerprint(model) == fingerprint(model)

    def test_resolve_returns_the_registered_model(self):
        assert resolve("WorkflowJobPayload", "v2") is WorkflowJobPayload

    def test_both_versions_stay_registered(self):
        """A deployment mid-upgrade holds both, so v1 must not disappear when
        v2 arrives."""
        assert resolve("WorkflowGraph", "v1") is not resolve("WorkflowGraph", "v2")
        assert latest_version("WorkflowGraph") == "v2"

    def test_resolving_an_unknown_contract_is_a_typed_error(self):
        with pytest.raises(SchemaNotRegistered):
            resolve("NoSuchContract", "v1")


class TestJobPayloadContract:
    def test_round_trips_through_json(self):
        payload = _payload()
        restored = WorkflowJobPayload.model_validate(json.loads(payload.model_dump_json()))
        assert restored == payload

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowJobPayload.model_validate(
                {**_payload().model_dump(mode="json"), "sneaky": True}
            )

    def test_payload_is_immutable(self):
        payload = _payload()
        with pytest.raises(ValidationError):
            payload.run_id = "run_2"

    @pytest.mark.parametrize("missing", ["run_id", "step_execution_id", "job_context"])
    def test_structurally_mandatory_fields(self, missing):
        data = _payload().model_dump(mode="json")
        data.pop(missing)
        with pytest.raises(ValidationError):
            WorkflowJobPayload.model_validate(data)

    def test_terminal_state_may_not_dispatch_more_work(self):
        """The determinism guard from mandate §1."""
        with pytest.raises(ValidationError, match="illegal transition"):
            _payload(
                current_state=ExecutionState.WORKFLOW_COMPLETE,
                next_step_definition=NextStepDefinition(
                    target_service=ServiceName.LLM_PROXY, target_node_id="think"
                ),
            )

    def test_terminal_state_may_transition_to_terminal(self):
        payload = _payload(
            current_state=ExecutionState.WORKFLOW_COMPLETE,
            next_step_definition=NextStepDefinition(
                target_service=ServiceName.TERMINAL, terminal=True
            ),
        )
        assert payload.next_step_definition.terminal

    def test_idempotency_key_is_the_composite_key(self):
        assert _payload().idempotency_key == ("run_1", "se_1")


class TestNextStepDefinition:
    def test_terminal_step_cannot_name_a_target_node(self):
        with pytest.raises(ValidationError):
            NextStepDefinition(
                target_service=ServiceName.TERMINAL, target_node_id="x", terminal=True
            )

    def test_non_terminal_step_must_name_a_target_node(self):
        with pytest.raises(ValidationError):
            NextStepDefinition(target_service=ServiceName.LLM_PROXY)

    def test_terminal_service_requires_the_terminal_flag(self):
        with pytest.raises(ValidationError):
            NextStepDefinition(target_service=ServiceName.TERMINAL, terminal=False)


class TestTransitionTable:
    def test_terminal_states_only_lead_to_terminal(self):
        for state in TERMINAL_STATES:
            assert LEGAL_TRANSITIONS[state] == frozenset({ServiceName.TERMINAL})

    def test_every_state_has_a_transition_rule(self):
        assert set(LEGAL_TRANSITIONS) == set(ExecutionState)


class TestJobContext:
    def test_scopes_are_normalised_and_collected(self):
        context = _context(
            permissions=PermissionRequirement(
                required_scope="read_hr", required_write="write_db"
            )
        )
        assert context.permissions.as_set() == frozenset({"READ_HR", "WRITE_DB"})

    def test_read_only_job_has_no_write_scope(self):
        assert _context().permissions.as_set() == frozenset({"READ_HR"})

    def test_identity_fields_are_required(self):
        with pytest.raises(ValidationError):
            JobContext(
                tenant_id="",
                initiating_user_id="usr_1",
                permissions=PermissionRequirement(required_scope="READ_HR"),
                trace_id="tr_1",
            )
