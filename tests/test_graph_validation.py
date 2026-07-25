"""A graph that cannot execute must fail to save, not fail mid-run."""

from __future__ import annotations

import pytest
from cwap_contracts.v2 import NodeType, WorkflowEdge, WorkflowGraph, WorkflowNode
from pydantic import ValidationError


def graph(nodes, edges, **kwargs) -> WorkflowGraph:
    return WorkflowGraph(id="wf_1", name="test", nodes=nodes, edges=edges, **kwargs)


def node(node_id: str, node_type: NodeType = NodeType.LLM, **kwargs) -> WorkflowNode:
    return WorkflowNode(id=node_id, type=node_type, **kwargs)


class TestStructure:
    def test_empty_graph_is_rejected(self):
        with pytest.raises(ValidationError, match="at least one node"):
            graph([], [])

    def test_duplicate_node_ids_are_rejected(self):
        with pytest.raises(ValidationError, match="duplicate node ids"):
            graph([node("a"), node("a")], [])

    def test_edge_to_unknown_node_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown target"):
            graph([node("a")], [WorkflowEdge(id="e", source="a", target="ghost")])

    def test_self_loop_is_rejected(self):
        with pytest.raises(ValidationError, match="self-loop"):
            graph([node("a")], [WorkflowEdge(id="e", source="a", target="a")])

    def test_exactly_one_entry_point_is_required(self):
        with pytest.raises(ValidationError, match="exactly one entry node"):
            graph([node("a"), node("b")], [])

    def test_cycle_is_rejected(self):
        """A cycle on the canvas is a run that never terminates."""
        with pytest.raises(ValidationError, match="cycle"):
            graph(
                [node("a"), node("b"), node("c")],
                [
                    WorkflowEdge(id="e1", source="a", target="b"),
                    WorkflowEdge(id="e2", source="b", target="c"),
                    WorkflowEdge(id="e3", source="c", target="b"),
                ],
            )

    def test_valid_linear_graph_is_accepted(self):
        result = graph(
            [node("a", NodeType.INPUT), node("b"), node("c", NodeType.OUTPUT)],
            [
                WorkflowEdge(id="e1", source="a", target="b"),
                WorkflowEdge(id="e2", source="b", target="c"),
            ],
        )
        assert result.entry_node().id == "a"


class TestFanOut:
    def test_only_a_branch_node_may_fan_out(self):
        with pytest.raises(ValidationError, match="only a branch"):
            graph(
                [node("a", NodeType.INPUT), node("b"), node("c")],
                [
                    WorkflowEdge(id="e1", source="a", target="b"),
                    WorkflowEdge(id="e2", source="a", target="c"),
                ],
            )

    def test_branch_needs_both_a_true_and_a_false_edge(self):
        with pytest.raises(ValidationError, match="condition=true"):
            graph(
                [node("a", NodeType.BRANCH), node("b", NodeType.OUTPUT)],
                [WorkflowEdge(id="e1", source="a", target="b", condition=True)],
            )

    def test_well_formed_branch_is_accepted(self):
        result = graph(
            [
                node("a", NodeType.INPUT),
                node("d", NodeType.BRANCH),
                node("yes", NodeType.OUTPUT),
                node("no", NodeType.OUTPUT),
            ],
            [
                WorkflowEdge(id="e0", source="a", target="d"),
                WorkflowEdge(id="e1", source="d", target="yes", condition=True),
                WorkflowEdge(id="e2", source="d", target="no", condition=False),
            ],
        )
        assert len(result.outgoing("d")) == 2

    def test_output_node_cannot_have_outgoing_edges(self):
        with pytest.raises(ValidationError, match="cannot have outgoing"):
            graph(
                [node("a", NodeType.INPUT), node("b", NodeType.OUTPUT), node("c")],
                [
                    WorkflowEdge(id="e1", source="a", target="b"),
                    WorkflowEdge(id="e2", source="b", target="c"),
                ],
            )


class TestNodeRules:
    def test_rag_node_without_a_corpus_is_rejected(self):
        """User Story 3: a retrieval step with nothing to retrieve from is a bug
        the user should see at design time, not at run time."""
        with pytest.raises(ValidationError, match="knowledge_handle"):
            node("r", NodeType.RAG_RETRIEVE)

    def test_rag_node_with_a_corpus_is_accepted(self):
        assert node("r", NodeType.RAG_RETRIEVE, knowledge_handle="kb_1").knowledge_handle
