"""Execution engine: state machine, executors, run reports."""

from __future__ import annotations

import pytest
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.logbus import log_bus
from cwap_common.models import Run, User, WorkflowExecutionState
from cwap_common.settings import get_settings, reset_settings_cache
from cwap_contracts.v4 import (
    NodeType,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from orchestrator.runner import RUN_FAILED, RUN_SUCCEEDED, Worker, run_to_completion, start_run
from orchestrator.variables import BindingError, RunContext, render_template, resolve

from tests.conftest import make_branching_graph, make_linear_graph


def load_run(run_id: str) -> Run:
    with read_only_session() as session:
        return session.get(Run, run_id)


def steps_for(run_id: str) -> list[WorkflowExecutionState]:
    with read_only_session() as session:
        return (
            session.query(WorkflowExecutionState)
            .filter_by(run_id=run_id)
            .order_by(WorkflowExecutionState.id)
            .all()
        )


class TestVariableResolution:
    def test_run_input_reference(self):
        context = RunContext(run_id="run_1", inputs={"goal": "Denver"})
        assert resolve("$run.input.goal", context) == "Denver"

    def test_previous_output_reference(self):
        context = RunContext(run_id="run_1")
        context.record("think", {"text": "hello"})
        assert resolve("$output.text", context) == "hello"

    def test_named_step_reference(self):
        context = RunContext(run_id="run_1")
        context.record("think", {"text": "hello"})
        context.record("draft", {"text": "world"})
        assert resolve("$steps.think.text", context) == "hello"

    def test_literal_passes_through(self):
        assert resolve("just a string", RunContext(run_id="run_1")) == "just a string"

    def test_missing_key_is_a_loud_error(self):
        """A silently-empty prompt variable produces a plausible wrong answer,
        which is worse than a failed run."""
        context = RunContext(run_id="run_1", inputs={"goal": "x"})
        with pytest.raises(BindingError, match="no value at 'missing'"):
            resolve("$run.input.missing", context)

    def test_output_reference_before_any_step_is_an_error(self):
        with pytest.raises(BindingError, match="first step"):
            resolve("$output.text", RunContext(run_id="run_1"))

    def test_unknown_namespace_is_rejected(self):
        with pytest.raises(BindingError, match="unknown namespace"):
            resolve("$nope.value", RunContext(run_id="run_1"))


class TestTemplates:
    def test_named_substitution(self):
        assert render_template("Hi {{name}}", {"name": "Ada"}) == "Hi Ada"

    def test_nested_path(self):
        assert render_template("{{a.b}}", {"a": {"b": "deep"}}) == "deep"

    def test_undefined_variable_is_reported(self):
        with pytest.raises(BindingError, match="undefined variable"):
            render_template("{{missing}}", {"present": 1})

    def test_lists_render_as_lines(self):
        assert render_template("{{items}}", {"items": ["a", "b"]}) == "a\nb"


class TestLinearRun:
    def test_run_reaches_a_result(self, authorized_user):
        run_id = run_to_completion(
            graph=make_linear_graph(),
            job_context=authorized_user,
            inputs={"goal": "Plan a weekend in Denver"},
        )
        run = load_run(run_id)
        assert run.status == RUN_SUCCEEDED
        assert "Plan a weekend in Denver" in run.result["result"]

    def test_every_node_leaves_a_state_row(self, authorized_user):
        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "hi"}
        )
        assert [step.node_id for step in steps_for(run_id)] == ["input", "think", "output"]

    def test_the_log_is_a_complete_trace(self, authorized_user):
        """Epic 4's outcome: the user can see exactly why the answer appeared."""
        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "hi"}
        )
        events = [event.event for event in log_bus.history(run_id)]
        assert events[0] == "run.accepted"
        assert events[-1] == "run.succeeded"
        assert "llm.request" in events and "llm.response" in events

    def test_log_sequence_numbers_are_monotonic(self, authorized_user):
        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "hi"}
        )
        sequences = [event.seq for event in log_bus.history(run_id)]
        assert sequences == sorted(sequences) == list(range(1, len(sequences) + 1))

    def test_graph_is_snapshotted_at_run_start(self, authorized_user):
        """Editing a workflow mid-run cannot change what that run executes."""
        graph = make_linear_graph()
        run_id = run_to_completion(
            graph=graph, job_context=authorized_user, inputs={"goal": "hi"}
        )
        assert load_run(run_id).graph_snapshot["id"] == graph.id


class TestBranching:
    def test_true_path_is_taken_and_recorded(self, authorized_user):
        run_id = run_to_completion(
            graph=make_branching_graph(),
            job_context=authorized_user,
            inputs={"goal": "trip to Denver"},
        )
        run = load_run(run_id)
        assert run.status == RUN_SUCCEEDED
        assert run.result["result"] == "matched"

        decision = [step for step in steps_for(run_id) if step.node_id == "decide"]
        assert decision and decision[0].output["decision"] is True

    def test_false_path_is_taken_when_the_condition_fails(self, authorized_user):
        run_id = run_to_completion(
            graph=make_branching_graph(),
            job_context=authorized_user,
            inputs={"goal": "trip to Lisbon"},
        )
        assert load_run(run_id).result["result"] == "not matched"

    def test_the_decision_is_explained_in_the_log(self, authorized_user):
        run_id = run_to_completion(
            graph=make_branching_graph(),
            job_context=authorized_user,
            inputs={"goal": "trip to Denver"},
        )
        branch_events = [
            event for event in log_bus.history(run_id) if event.event == "branch.evaluated"
        ]
        assert branch_events and branch_events[0].data["operator"] == "contains"


class TestFailureHandling:
    def test_a_missing_required_input_fails_the_run_with_a_reason(self, authorized_user):
        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={}
        )
        run = load_run(run_id)
        assert run.status == RUN_FAILED
        assert "missing required field" in run.error

    def test_a_failed_run_is_reported_on_the_log_stream(self, authorized_user):
        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={}
        )
        assert log_bus.history(run_id)[-1].event == "run.failed"

    def test_an_unresolvable_binding_fails_cleanly(self, authorized_user):
        graph = WorkflowGraph(
            id="wf_bad",
            name="bad binding",
            nodes=[
                WorkflowNode(id="input", type=NodeType.INPUT, params={"fields": []}),
                WorkflowNode(id="out", type=NodeType.OUTPUT),
            ],
            edges=[
                WorkflowEdge(
                    id="e1",
                    source="input",
                    target="out",
                    bindings={"previous": "$run.input.nope"},
                )
            ],
        )
        run_id = run_to_completion(graph=graph, job_context=authorized_user, inputs={})
        assert load_run(run_id).status == RUN_FAILED

    def test_step_limit_stops_a_runaway_run(self, authorized_user, monkeypatch):
        monkeypatch.setenv("CWAP_MAX_STEPS", "1")
        reset_settings_cache()
        assert get_settings().max_steps_per_run == 1

        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "hi"}
        )
        run = load_run(run_id)
        assert run.status == RUN_FAILED
        assert "step limit" in run.error


class TestRedelivery:
    def test_a_redelivered_job_does_not_duplicate_state(self, authorized_user, broker):
        """The worker is restartable: the same message twice is one step."""
        settings = get_settings()
        handle = start_run(
            graph=make_linear_graph(),
            job_context=authorized_user,
            inputs={"goal": "hi"},
        )
        raw = broker.drain(settings.work_queue)[0]

        worker = Worker()
        broker.publish(settings.work_queue, raw)
        worker.poll()
        broker.publish(settings.work_queue, raw)  # duplicate delivery
        worker.poll()

        input_rows = [
            step for step in steps_for(handle.run_id) if step.node_id == "input"
        ]
        assert len(input_rows) == 1

    def test_a_job_for_a_finished_run_is_ignored(self, authorized_user, broker):
        settings = get_settings()
        handle = start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "hi"}
        )
        # Keep a copy of the first message, then let the run finish normally.
        first = broker.drain(settings.work_queue)[0]
        broker.publish(settings.work_queue, first)
        Worker().drain()
        assert load_run(handle.run_id).status == RUN_SUCCEEDED

        broker.publish(settings.work_queue, first)
        Worker().poll()

        assert load_run(handle.run_id).status == RUN_SUCCEEDED
        events = [event.event for event in log_bus.history(handle.run_id)]
        assert "step.skipped" in events


class TestHttpNodeSafety:
    def test_outbound_calls_are_denied_by_default(self, authorized_user):
        """A workflow is user-authored content executed server-side, so an
        unrestricted HTTP node would be an SSRF primitive."""
        graph = WorkflowGraph(
            id="wf_http",
            name="http",
            nodes=[
                WorkflowNode(id="input", type=NodeType.INPUT, params={"fields": []}),
                WorkflowNode(
                    id="call",
                    type=NodeType.HTTP_REQUEST,
                    params={"method": "GET", "url": "http://169.254.169.254/latest/meta-data/"},
                ),
                WorkflowNode(id="out", type=NodeType.OUTPUT),
            ],
            edges=[
                WorkflowEdge(id="e1", source="input", target="call"),
                WorkflowEdge(id="e2", source="call", target="out"),
            ],
        )
        with unit_of_work() as session:
            session.query(User).filter_by(id=authorized_user.initiating_user_id).update(
                {User.scopes: ["READ_WORKFLOWS", "WRITE_EXTERNAL"]}
            )
        context = authorized_user.model_copy(
            update={
                "permissions": authorized_user.permissions.model_copy(
                    update={"required_write": "WRITE_EXTERNAL"}
                )
            }
        )

        run_id = run_to_completion(graph=graph, job_context=context, inputs={})
        run = load_run(run_id)
        assert run.status == RUN_FAILED
        assert "allow-list" in run.error


class TestDesignedWorkflowsRun:
    """The strongest guard on Epic 1: a goal the user types must produce a
    workflow that actually executes, not merely one that validates.

    A design can be structurally valid and still fail on its first run — that is
    how both of the earlier scaffolder bugs got through.
    """

    @pytest.mark.parametrize(
        "goal",
        [
            "Plan my weekend trip to Denver",
            "Research our competitors and write a short comparison",
            "Summarise our internal handbook",
            "Compare two hosting providers and recommend one",
            "Extract the action items and format them as a table",
            "zxqv wibble frobnicate",
        ],
    )
    def test_a_designed_workflow_runs_end_to_end(self, authorized_user, goal):
        from cwap_contracts.v4 import DesignRequest, IngestRequest
        from design.service import design
        from knowledge.service import ingest

        corpus = ingest(
            IngestRequest(
                tenant_id=authorized_user.tenant_id,
                title="Handbook",
                content="Rail travel is preferred for journeys within the United Kingdom.",
            )
        )
        response = design(
            DesignRequest(
                goal=goal, knowledge_handles=[corpus.handle], skip_questions=True
            ),
            authorized_user.tenant_id,
        )
        assert response.graph is not None

        run_id = run_to_completion(
            graph=response.graph, job_context=authorized_user, inputs={}
        )
        run = load_run(run_id)
        assert run.status == RUN_SUCCEEDED, run.error

    def test_every_designed_workflow_is_built_from_agents(self, authorized_user):
        """Item 3: a step in a designed workflow is an agent with a role, skills
        and memory — not a bare model call."""
        from cwap_contracts.v4 import DesignRequest, NodeType
        from design.service import design

        response = design(
            DesignRequest(goal="Plan my weekend trip to Denver", skip_questions=True),
            authorized_user.tenant_id,
        )
        working = [
            node
            for node in response.graph.nodes
            if node.type not in (NodeType.INPUT, NodeType.OUTPUT)
        ]
        assert working, "the design produced no working steps"
        assert all(node.type is NodeType.AGENT for node in working)
        assert all(node.agent_id for node in working)
