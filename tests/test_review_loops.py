"""A second agent has to be satisfied before a step's answer is used.

An agent checking its own work is a weak check: it stopped because it already
decided the answer was good. A different agent — its own role, its own skills,
its own memory of reviewing — is a real one.

The design constraint that shapes all of this: the repetition happens *inside*
one step. An edge looping back to an earlier node would make the canvas cyclic,
and then whether a run terminates would depend on a model's judgement rather than
on the structure. Bounded rounds inside a step keep the graph acyclic, keep every
run finite, and keep the cost knowable before anyone presses Run.
"""

from __future__ import annotations

import pytest
from agents import registry as agent_registry
from cwap_common.db import read_only_session
from cwap_common.models import Run, WorkflowExecutionState
from cwap_contracts.v4 import (
    MAX_REVIEW_ROUNDS,
    NodeType,
    Position,
    ReviewConfig,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from orchestrator import review as review_loop
from orchestrator.runner import RUN_SUCCEEDED, Worker, start_run


class Scripted:
    """An agent runtime that answers from a script, so a loop is observable."""

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, agent, objective, context):
        from agents.runtime import AgentResult

        self.calls.append((agent.name, objective))
        if agent.name == "Reviewer":
            text = self.verdicts.pop(0) if self.verdicts else "APPROVED"
        else:
            text = f"draft {len([c for c in self.calls if c[0] != 'Reviewer'])}"
        return AgentResult(text=text, status="objective_met")


def reviewed_graph(author_id: str, reviewer_id: str, rounds: int = 2) -> WorkflowGraph:
    return WorkflowGraph(
        id="wf_reviewed",
        name="Reviewed",
        nodes=[
            WorkflowNode(
                id="input",
                type=NodeType.INPUT,
                params={"fields": ["goal"]},
                position=Position(x=0, y=0),
            ),
            WorkflowNode(
                id="author",
                type=NodeType.AGENT,
                label="Author",
                agent_id=author_id,
                params={"objective_template": "Write about {{goal}}"},
                position=Position(x=250, y=0),
                review=ReviewConfig(agent_id=reviewer_id, max_rounds=rounds),
            ),
            WorkflowNode(
                id="output",
                type=NodeType.OUTPUT,
                params={"result_template": "{{previous}}"},
                position=Position(x=500, y=0),
            ),
        ],
        edges=[
            WorkflowEdge(id="e1", source="input", target="author", bindings={"goal": "goal"}),
            WorkflowEdge(
                id="e2", source="author", target="output", bindings={"previous": "text"}
            ),
        ],
    )


@pytest.fixture
def team(isolated_platform):
    author = agent_registry.create("tenant-a", name="Author", role="You write.")
    reviewer = agent_registry.create("tenant-a", name="Reviewer", role="You review.")
    return author, reviewer


class TestTheLoopRuns:
    def test_an_approved_draft_stops_after_one_round(self, team, authorized_user, monkeypatch):
        author, reviewer = team
        scripted = Scripted(["APPROVED"])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        handle = start_run(
            graph=reviewed_graph(author.id, reviewer.id),
            job_context=authorized_user,
            inputs={"goal": "otters"},
        )
        Worker().drain()

        assert _status(handle.run_id) == RUN_SUCCEEDED
        # author, reviewer. No revision, because nothing was asked for.
        assert [name for name, _ in scripted.calls] == ["Author", "Reviewer"]

    def test_changes_send_the_draft_back_to_its_author(self, team, authorized_user, monkeypatch):
        author, reviewer = team
        scripted = Scripted(["Too vague. Name the species.", "APPROVED"])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        start_run(
            graph=reviewed_graph(author.id, reviewer.id),
            job_context=authorized_user,
            inputs={"goal": "otters"},
        )
        Worker().drain()

        assert [name for name, _ in scripted.calls] == [
            "Author",
            "Reviewer",
            "Author",
            "Reviewer",
        ]

    def test_the_revision_sees_the_feedback(self, team, authorized_user, monkeypatch):
        author, reviewer = team
        scripted = Scripted(["Name the species.", "APPROVED"])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        start_run(
            graph=reviewed_graph(author.id, reviewer.id),
            job_context=authorized_user,
            inputs={"goal": "otters"},
        )
        Worker().drain()

        revision = scripted.calls[2][1]
        assert "Name the species." in revision
        assert "draft 1" in revision

    def test_the_reviewed_answer_is_what_the_next_step_receives(
        self, team, authorized_user, monkeypatch
    ):
        author, reviewer = team
        scripted = Scripted(["Fix it.", "APPROVED"])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        handle = start_run(
            graph=reviewed_graph(author.id, reviewer.id),
            job_context=authorized_user,
            inputs={"goal": "otters"},
        )
        Worker().drain()

        assert _output(handle.run_id, "author")["text"] == "draft 2"

    def test_the_round_limit_ends_the_loop_without_failing_the_run(
        self, team, authorized_user, monkeypatch
    ):
        """A reviewer with notes left has still improved the draft several
        times; throwing that away would be worse than shipping it."""
        author, reviewer = team
        scripted = Scripted(["No.", "Still no.", "No.", "No.", "No.", "No."])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        handle = start_run(
            graph=reviewed_graph(author.id, reviewer.id, rounds=2),
            job_context=authorized_user,
            inputs={"goal": "otters"},
        )
        Worker().drain()

        assert _status(handle.run_id) == RUN_SUCCEEDED
        assert [name for name, _ in scripted.calls].count("Reviewer") == 2

    def test_the_report_records_what_the_reviewer_said(
        self, team, authorized_user, monkeypatch
    ):
        author, reviewer = team
        scripted = Scripted(["Too vague.", "APPROVED"])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        handle = start_run(
            graph=reviewed_graph(author.id, reviewer.id),
            job_context=authorized_user,
            inputs={"goal": "otters"},
        )
        Worker().drain()

        review = _context(handle.run_id, "author")["review"]
        assert review["approved"] is True
        assert review["rounds"] == 2
        assert any(note.get("feedback") == "Too vague." for note in review["notes"])

    def test_a_step_with_no_reviewer_costs_nothing(self, team, authorized_user, monkeypatch):
        author, _unused = team
        scripted = Scripted([])
        monkeypatch.setattr("agents.runtime.run_agent", scripted)

        reviewer = agent_registry.create("tenant-a", name="Spare", role="You review.")
        graph = reviewed_graph(author.id, reviewer.id)
        plain = graph.model_copy(
            update={
                "nodes": [
                    node.model_copy(update={"review": None}) if node.id == "author" else node
                    for node in graph.nodes
                ]
            }
        )
        handle = start_run(
            graph=plain, job_context=authorized_user, inputs={"goal": "otters"}
        )
        Worker().drain()

        assert [name for name, _ in scripted.calls] == ["Author"]
        assert _context(handle.run_id, "author").get("review") is None


class TestWhatCountsAsApproval:
    def test_the_word_on_the_first_line_approves(self):
        assert review_loop.approves("APPROVED\nnice work", "APPROVED")
        assert review_loop.approves("approved — ship it", "APPROVED")

    def test_a_refusal_that_contains_the_word_does_not(self):
        """"this is not approved" must not end the loop."""
        assert not review_loop.approves("This is not approved yet.", "APPROVED")
        assert not review_loop.approves("Fix the intro.\nAPPROVED once you do.", "APPROVED")

    def test_silence_is_not_approval(self):
        assert not review_loop.approves("", "APPROVED")


class TestTheGraphStaysSafe:
    def test_a_node_cannot_review_itself(self, team):
        author, _ = team
        with pytest.raises(ValueError, match="its own reviewer"):
            WorkflowNode(
                id="author",
                type=NodeType.AGENT,
                agent_id=author.id,
                review=ReviewConfig(agent_id=author.id),
            )

    def test_rounds_are_bounded(self, team):
        _author, reviewer = team
        with pytest.raises(ValueError):
            ReviewConfig(agent_id=reviewer.id, max_rounds=MAX_REVIEW_ROUNDS + 1)

    def test_only_a_step_that_produces_an_answer_can_be_reviewed(self, team):
        _author, reviewer = team
        with pytest.raises(ValueError, match="cannot be reviewed"):
            WorkflowNode(
                id="fetch",
                type=NodeType.HTTP_REQUEST,
                review=ReviewConfig(agent_id=reviewer.id),
            )

    def test_a_reviewed_graph_is_still_acyclic(self, team):
        """The whole point of bounding repetition inside a step."""
        author, reviewer = team
        graph = reviewed_graph(author.id, reviewer.id)

        assert len(graph.edges) == 2  # no edge loops back


def _status(run_id: str) -> str:
    with read_only_session() as session:
        return session.get(Run, run_id).status


def _state(run_id: str, node_id: str):
    with read_only_session() as session:
        return (
            session.query(WorkflowExecutionState)
            .filter_by(run_id=run_id, node_id=node_id)
            .first()
        )


def _output(run_id: str, node_id: str) -> dict:
    return _state(run_id, node_id).output


def _context(run_id: str, node_id: str) -> dict:
    return _state(run_id, node_id).derived_context
