"""Intelligent Requirement Diagnosis & Setup (Epic 1, User Story 2)."""

from __future__ import annotations

from cwap_contracts import GoalIntakeRequest, NodeType
from nlp.service import diagnose, scaffold

DENVER = "Plan my weekend trip to Denver"


def diagnose_goal(goal: str, handles=None):
    return diagnose(GoalIntakeRequest(goal=goal, knowledge_handles=handles or []))


def scaffold_goal(goal: str, handles=None):
    return scaffold(GoalIntakeRequest(goal=goal, knowledge_handles=handles or []))


class TestDiagnosis:
    def test_a_planning_goal_is_recognised_as_planning(self):
        result = diagnose_goal(DENVER)
        assert "plan" in result.intent
        assert any(step.skill == "plan" for step in result.required_steps)

    def test_every_goal_gets_at_least_one_reasoning_step(self):
        """Even a goal with no keyword hit must produce a runnable draft."""
        result = diagnose_goal("zxqv wibble frobnicate")
        assert any(step.node_type is NodeType.LLM for step in result.required_steps)

    def test_diagnosis_is_deterministic(self):
        assert diagnose_goal(DENVER) == diagnose_goal(DENVER)

    def test_confidence_rises_with_evidence(self):
        vague = diagnose_goal("zxqv wibble")
        specific = diagnose_goal("Research and write a report about our internal policy")
        assert specific.confidence > vague.confidence

    def test_confidence_never_claims_certainty(self):
        assert diagnose_goal("research plan write format if api").confidence <= 0.95

    def test_suggestions_are_always_offered(self):
        assert diagnose_goal(DENVER).suggestions


class TestGaps:
    def test_document_grounding_without_a_corpus_is_a_gap_not_a_node(self):
        result = diagnose_goal("Summarise our internal company handbook")
        assert any("Knowledge Context" in gap for gap in result.gaps)
        assert not any(
            step.node_type is NodeType.RAG_RETRIEVE for step in result.required_steps
        )

    def test_document_grounding_with_a_corpus_becomes_a_step(self):
        result = diagnose_goal("Summarise our internal company handbook", handles=["kb_1"])
        assert any(step.node_type is NodeType.RAG_RETRIEVE for step in result.required_steps)
        assert not result.gaps

    def test_an_external_api_is_always_a_gap(self):
        """We cannot invent a credential, so we say so rather than scaffolding a
        node that would fail at run time."""
        result = diagnose_goal("Sync the results to our CRM api endpoint")
        assert any("credential" in gap for gap in result.gaps)


class TestScaffolding:
    def test_the_draft_graph_is_structurally_valid(self):
        """`WorkflowGraph` construction is the validation — if the scaffolder
        emitted something unexecutable this would raise."""
        graph = scaffold_goal(DENVER).graph
        assert graph.entry_node().type is NodeType.INPUT

    def test_the_goal_is_pre_filled_for_review(self):
        graph = scaffold_goal(DENVER).graph
        assert graph.node("input").params["defaults"]["goal"] == DENVER

    def test_a_grounded_goal_links_the_corpus(self):
        graph = scaffold_goal("Summarise our internal handbook", handles=["kb_42"]).graph
        assert graph.node("knowledge").knowledge_handle == "kb_42"

    def test_a_conditional_goal_produces_a_decision_and_two_outcomes(self):
        graph = scaffold_goal(
            "Research the topic and if it mentions Denver write a summary, otherwise stop"
        ).graph
        assert graph.node("decide").type is NodeType.BRANCH
        assert {edge.condition for edge in graph.outgoing("decide")} == {True, False}

    def test_scaffolds_are_capped_so_the_draft_stays_reviewable(self):
        graph = scaffold_goal("Research, plan, write, draft, compose and explain").graph
        llm_nodes = [node for node in graph.nodes if node.type is NodeType.LLM]
        assert len(llm_nodes) <= 2

    def test_nodes_are_laid_out_left_to_right(self):
        graph = scaffold_goal(DENVER).graph
        ordered = sorted(graph.nodes, key=lambda node: node.position.x)
        assert ordered[0].type is NodeType.INPUT

    def test_scaffolding_is_deterministic_apart_from_the_generated_id(self):
        first = scaffold_goal(DENVER).graph
        second = scaffold_goal(DENVER).graph
        assert first.model_dump(exclude={"id"}) == second.model_dump(exclude={"id"})
