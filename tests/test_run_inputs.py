"""The entry node's declared defaults become the run's effective inputs.

Regression guard for a bug found while driving the real UI: a workflow
scaffolded from a goal carries the goal as a *default* on its Start node, but
downstream edges bind `$run.input.goal`. If the merged set is not written back
to the run, every scaffolded workflow fails on its second step with "no value at
'goal'" — even though the Start node itself resolved fine.
"""

from __future__ import annotations

from cwap_common.db import read_only_session
from cwap_common.models import Run
from cwap_contracts.v4 import NodeType, WorkflowEdge, WorkflowGraph, WorkflowNode
from orchestrator.runner import RUN_SUCCEEDED, run_to_completion


def graph_with_default(goal: str) -> WorkflowGraph:
    return WorkflowGraph(
        id="wf_defaults",
        name="defaults",
        nodes=[
            WorkflowNode(
                id="input",
                type=NodeType.INPUT,
                params={"fields": ["goal"], "defaults": {"goal": goal}},
            ),
            WorkflowNode(
                id="think", type=NodeType.LLM, params={"prompt_template": "{{goal}}"}
            ),
            WorkflowNode(
                id="output", type=NodeType.OUTPUT, params={"result_template": "{{previous}}"}
            ),
        ],
        edges=[
            # Deliberately binds through $run.input, the way the scaffolder does.
            WorkflowEdge(
                id="e1", source="input", target="think", bindings={"goal": "$run.input.goal"}
            ),
            WorkflowEdge(
                id="e2", source="think", target="output", bindings={"previous": "$output.text"}
            ),
        ],
    )


def test_declared_defaults_satisfy_downstream_run_input_bindings(authorized_user):
    run_id = run_to_completion(
        graph=graph_with_default("Plan my weekend trip to Denver"),
        job_context=authorized_user,
        inputs={},
    )
    with read_only_session() as session:
        run = session.get(Run, run_id)

    assert run.status == RUN_SUCCEEDED, run.error
    assert "Denver" in run.result["result"]


def test_supplied_inputs_win_over_declared_defaults(authorized_user):
    run_id = run_to_completion(
        graph=graph_with_default("the default goal"),
        job_context=authorized_user,
        inputs={"goal": "the supplied goal"},
    )
    with read_only_session() as session:
        run = session.get(Run, run_id)

    assert "the supplied goal" in run.result["result"]


def test_the_effective_inputs_are_persisted_on_the_run(authorized_user):
    """So a restarted worker rebuilds the same context, and the run report shows
    what the workflow actually ran with."""
    run_id = run_to_completion(
        graph=graph_with_default("Denver"), job_context=authorized_user, inputs={}
    )
    with read_only_session() as session:
        run = session.get(Run, run_id)

    assert run.inputs == {"goal": "Denver"}
