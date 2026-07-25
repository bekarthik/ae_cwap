"""Independent work runs at the same time.

v2 allowed exactly one outgoing edge from anything but a branch, so every
workflow was a line: two pieces of work that had nothing to do with each other
still waited for each other. That was the right first constraint — one successor
per step is what makes a job payload deterministic — and it stopped being the
right one as soon as workflows had more than three steps.

What had to be true before the restriction could be lifted:

* a completed step dispatches **every** successor, not the first;
* a node with several inbound edges is a **join** and waits for all of them,
  because a step that runs on half its inputs produces a confident wrong answer;
* two branches finishing at once must not both start the join — their state rows
  have different step ids, so uniqueness there cannot collapse the duplicate;
* the run finishes when the *last* path finishes, not the first.
"""

from __future__ import annotations

import pytest
from cwap_common.db import read_only_session
from cwap_common.models import Run, WorkflowExecutionState
from cwap_contracts.v4 import (
    NodeType,
    Position,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from orchestrator.runner import RUN_SUCCEEDED, Worker, start_run


def step(node_id: str, template: str) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        type=NodeType.LLM,
        label=node_id,
        params={"prompt_template": template},
        position=Position(x=0, y=0),
    )


def diamond() -> WorkflowGraph:
    """input ─┬─► left ──┬─► merge ─► output
              └─► right ─┘"""
    return WorkflowGraph(
        id="wf_diamond",
        name="Diamond",
        nodes=[
            WorkflowNode(
                id="input", type=NodeType.INPUT, params={"fields": ["goal"]}
            ),
            step("left", "left: {{goal}}"),
            step("right", "right: {{goal}}"),
            step("merge", "merge {{a}} and {{b}}"),
            WorkflowNode(
                id="output", type=NodeType.OUTPUT, params={"result_template": "{{previous}}"}
            ),
        ],
        edges=[
            WorkflowEdge(id="e1", source="input", target="left", bindings={"goal": "$run.input.goal"}),
            WorkflowEdge(id="e2", source="input", target="right", bindings={"goal": "$run.input.goal"}),
            # A join names its sources explicitly. `$output` means "the step
            # before this one", which is exactly the thing a join does not have.
            WorkflowEdge(
                id="e3", source="left", target="merge", bindings={"a": "$steps.left.text"}
            ),
            WorkflowEdge(
                id="e4", source="right", target="merge", bindings={"b": "$steps.right.text"}
            ),
            WorkflowEdge(
                id="e5", source="merge", target="output", bindings={"previous": "text"}
            ),
        ],
    )


def forked() -> WorkflowGraph:
    """Two independent paths that never rejoin, each ending in its own result."""
    return WorkflowGraph(
        id="wf_forked",
        name="Forked",
        nodes=[
            WorkflowNode(id="input", type=NodeType.INPUT, params={"fields": ["goal"]}),
            step("left", "left: {{goal}}"),
            step("right", "right: {{goal}}"),
            WorkflowNode(
                id="out_left", type=NodeType.OUTPUT, params={"result_template": "{{previous}}"}
            ),
            WorkflowNode(
                id="out_right", type=NodeType.OUTPUT, params={"result_template": "{{previous}}"}
            ),
        ],
        edges=[
            WorkflowEdge(id="e1", source="input", target="left", bindings={"goal": "$run.input.goal"}),
            WorkflowEdge(id="e2", source="input", target="right", bindings={"goal": "$run.input.goal"}),
            WorkflowEdge(
                id="e3", source="left", target="out_left", bindings={"previous": "text"}
            ),
            WorkflowEdge(
                id="e4", source="right", target="out_right", bindings={"previous": "text"}
            ),
        ],
    )


class TestBothBranchesRun:
    def test_a_fan_out_dispatches_every_successor(self, authorized_user):
        handle = start_run(
            graph=diamond(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        Worker().drain()

        assert {"left", "right"} <= _completed(handle.run_id)

    def test_the_run_reaches_its_result(self, authorized_user):
        handle = start_run(
            graph=diamond(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        Worker().drain()

        assert _status(handle.run_id) == RUN_SUCCEEDED

    def test_the_join_sees_both_branches(self, authorized_user):
        """A step that ran on half its inputs would be confidently wrong."""
        handle = start_run(
            graph=diamond(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        Worker().drain()

        merged = _output(handle.run_id, "merge")["text"]
        assert "left: otters" in merged
        assert "right: otters" in merged
        assert " and " in merged

    def test_the_join_runs_exactly_once(self, authorized_user):
        """Two branches both finding it ready would otherwise start it twice."""
        handle = start_run(
            graph=diamond(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        Worker().drain()

        assert _times_run(handle.run_id, "merge") == 1

    def test_two_paths_that_never_rejoin_both_finish(self, authorized_user):
        """The run ends when the last path ends, not the first."""
        handle = start_run(
            graph=forked(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        Worker().drain()

        completed = _completed(handle.run_id)
        assert {"out_left", "out_right"} <= completed
        assert _status(handle.run_id) == RUN_SUCCEEDED


class TestAJoinWaits:
    def test_it_does_not_start_before_every_path_arrives(self, authorized_user):
        """Drive one message at a time: after the first branch, the join must
        still be unstarted."""
        handle = start_run(
            graph=diamond(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        worker = Worker()
        worker.poll()  # input
        worker.poll()  # left (or right)

        assert "merge" not in _completed(handle.run_id)

        worker.drain()
        assert "merge" in _completed(handle.run_id)

    def test_a_run_with_a_pending_branch_is_not_finished(self, authorized_user):
        handle = start_run(
            graph=diamond(), job_context=authorized_user, inputs={"goal": "otters"}
        )
        worker = Worker()
        worker.poll()
        worker.poll()

        assert _status(handle.run_id) != RUN_SUCCEEDED


class TestABranchStillChoosesOne:
    def test_a_decision_takes_exactly_one_path(self, authorized_user, branching_graph):
        """Fan-out is parallel work; a branch is a choice. Relaxing one must not
        have relaxed the other."""
        handle = start_run(
            graph=branching_graph,
            job_context=authorized_user,
            inputs={"goal": "somewhere in denver"},
        )
        Worker().drain()

        completed = _completed(handle.run_id)
        assert ("output" in completed) != ("output_alt" in completed)
        assert _status(handle.run_id) == RUN_SUCCEEDED


@pytest.fixture
def branching_graph():
    from conftest import make_branching_graph

    return make_branching_graph()


def _status(run_id: str) -> str:
    with read_only_session() as session:
        return session.get(Run, run_id).status


def _completed(run_id: str) -> set[str]:
    with read_only_session() as session:
        return {
            row.node_id
            for row in session.query(WorkflowExecutionState)
            .filter_by(run_id=run_id)
            .all()
        }


def _times_run(run_id: str, node_id: str) -> int:
    with read_only_session() as session:
        return (
            session.query(WorkflowExecutionState)
            .filter_by(run_id=run_id, node_id=node_id)
            .count()
        )


def _output(run_id: str, node_id: str) -> dict:
    with read_only_session() as session:
        return (
            session.query(WorkflowExecutionState)
            .filter_by(run_id=run_id, node_id=node_id)
            .first()
            .output
        )
