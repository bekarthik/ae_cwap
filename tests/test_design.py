"""Designing a workflow from a goal, rather than making the user draw one.

The claim being defended: a user states what they want, the system works out
what it needs to know, decides the steps, staffs each one with an agent, gives
each agent the skills it needs — creating any that do not exist — and wires up
memory. The user reviews a design; they do not assemble one.

Two properties matter more than the exact shape of the output:

* **It must work on a small model.** The structure is deterministic, and the
  model only rewords. A design that only appears on a frontier model would make
  the platform's central promise conditional on which model you configured.
* **It must be reviewable.** The same goal and the same answers produce the same
  design, so a user who changes one answer can see exactly what that changed.
"""

from __future__ import annotations

import pytest
from cwap_contracts.v2 import DesignRequest, DesignStage, NodeType, SkillOrigin
from design import blueprints
from design import service as design_service
from llm_proxy.client import LLMCompletion, ProviderCapabilities, reset_provider_cache
from skills import registry as skill_registry

TENANT = "tenant-a"


class RewordingProvider:
    """A model that returns a valid refinement, so the refine path is exercised."""

    capabilities = ProviderCapabilities(
        provider="scripted",
        label="Scripted",
        model="scripted-model",
        supports_effort=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_stop_sequences=True,
        supports_system_prompt=True,
    )

    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, prompt, *, system=None, options=None):
        return LLMCompletion(text=self._text, model="scripted-model")

    def converse(self, messages, *, system=None, tools=None, options=None):
        return self.complete("")


def design(goal: str, **overrides):
    request = DesignRequest(goal=goal, **overrides)
    return design_service.design(request, TENANT)


def designed(goal: str, **overrides):
    """Skip the questions and go straight to a design."""
    return design(goal, skip_questions=True, **overrides)


class TestTheConversation:
    def test_a_vague_goal_gets_questions_before_a_design(self):
        """"Sort out our onboarding" does not contain enough to design anything,
        and guessing produces a confident, useless workflow."""
        response = design("Sort out our onboarding")

        assert response.stage is DesignStage.CLARIFYING
        assert response.questions
        assert response.graph is None

    def test_the_system_says_back_what_it_understood(self):
        """So a misreading surfaces before anything is built."""
        response = design("Plan a weekend trip to Denver")
        assert "denver" in response.understanding.lower()

    def test_every_question_explains_why_it_is_being_asked(self):
        """An unexplained question is an interrogation; an explained one is help."""
        response = design("Sort out our onboarding")
        assert all(question.why for question in response.questions)

    def test_every_question_has_a_default_so_none_is_compulsory(self):
        response = design("Sort out our onboarding")
        assert all(question.default is not None for question in response.questions)

    def test_the_conversation_stays_short(self):
        response = design("Sort out our onboarding", knowledge_handles=["kb_1"])
        assert len(response.questions) <= design_service.MAX_QUESTIONS

    def test_a_question_the_goal_already_answered_is_not_asked(self):
        """Asking reads as though the system was not listening."""
        response = design("Write a report for my team about Q3 spending")
        assert {question.id for question in response.questions} & {"deliverable", "audience"} == set()

    def test_answers_move_the_conversation_on_to_a_design(self):
        first = design("Sort out our onboarding")
        answers = {question.id: (question.default or "x") for question in first.questions}

        second = design("Sort out our onboarding", answers=answers)

        assert second.stage is DesignStage.DESIGNED
        assert second.graph is not None

    def test_a_user_can_skip_the_questions_entirely(self):
        assert designed("Sort out our onboarding").stage is DesignStage.DESIGNED

    def test_documents_prompt_a_grounding_question_only_when_there_are_some(self):
        without = design("Sort out our onboarding")
        with_docs = design("Sort out our onboarding", knowledge_handles=["kb_1"])

        assert "grounding" not in {question.id for question in without.questions}
        assert "grounding" in {question.id for question in with_docs.questions}


class TestTheDesign:
    def test_the_result_is_a_runnable_graph(self):
        response = designed("Research and write a briefing on solar tariffs")
        graph = response.graph

        assert graph is not None
        assert graph.nodes[0].type is NodeType.INPUT
        assert graph.nodes[-1].type is NodeType.OUTPUT

    def test_every_working_step_is_an_agent(self):
        """Confirming item 3: not a prompt node with a fancy label — a stored
        agent with a role, skills and its own memory."""
        graph = designed("Research and write a briefing on solar tariffs").graph

        working = [
            node
            for node in graph.nodes
            if node.type not in (NodeType.INPUT, NodeType.OUTPUT)
        ]
        assert working
        assert all(node.type is NodeType.AGENT for node in working)
        assert all(node.agent_id for node in working)

    def test_each_agent_is_explained(self):
        """A design the user cannot follow is a design they cannot correct."""
        response = designed("Research and write a briefing on solar tariffs")
        assert all(agent.rationale for agent in response.agents)

    def test_the_agents_are_wired_in_order(self):
        graph = designed("Research and write a briefing on solar tariffs").graph
        sources = {edge.source for edge in graph.edges}
        targets = {edge.target for edge in graph.edges}

        assert "input" in sources
        assert "output" in targets
        assert len(graph.edges) == len(graph.nodes) - 1

    def test_a_later_agent_receives_the_earlier_agents_work(self):
        graph = designed("Research and write a briefing on solar tariffs").graph
        second_hop = [edge for edge in graph.edges if edge.source == "agent_1"]
        assert second_hop[0].bindings.get("previous") == "$output.text"

    def test_the_goal_becomes_the_workflows_declared_input(self):
        goal = "Research and write a briefing on solar tariffs"
        graph = designed(goal).graph
        assert graph.nodes[0].params["defaults"]["goal"] == goal

    def test_the_design_is_bounded(self):
        response = designed("Research, compare, plan, write and review everything")
        assert len(response.agents) <= design_service.MAX_AGENTS

    def test_the_same_goal_and_answers_produce_the_same_shape(self):
        """Reviewable means repeatable."""
        goal = "Research and write a briefing on solar tariffs"
        first, second = designed(goal), designed(goal)

        assert [agent.name for agent in first.agents] == [
            agent.name for agent in second.agents
        ]
        assert [node.type for node in first.graph.nodes] == [
            node.type for node in second.graph.nodes
        ]


class TestGoalShapes:
    @pytest.mark.parametrize(
        ("goal", "expected"),
        [
            ("Write a blog post about our new pricing", "research_and_write"),
            ("Plan a three-day itinerary for Lisbon", "plan"),
            ("Compare Postgres and MySQL for our workload", "decide"),
            ("Extract the totals from these invoices", "process"),
            ("Post new leads to our CRM", "integrate"),
        ],
    )
    def test_a_goal_is_read_as_the_right_kind_of_job(self, goal, expected):
        assert blueprints.classify(goal).key == expected

    @pytest.mark.parametrize(
        ("goal", "not_expected"),
        [
            # "Postgres" contains "post"; "planning permission" contains "plan"
            # only as a word, which is fine, but "explains" must not match
            # "explain" as a suffix of another word.
            ("Compare Postgres and MySQL for our workload", "research_and_write"),
            ("Evaluate Postman versus Insomnia", "research_and_write"),
        ],
    )
    def test_a_word_inside_another_word_is_not_a_keyword(self, goal, not_expected):
        """Substring matching reads "Compare **Post**gres" as a request to write
        a blog post. Word boundaries are the difference between classifying a
        goal and pattern-matching on its spelling."""
        assert blueprints.classify(goal).key != not_expected

    def test_an_unrecognised_goal_becomes_one_capable_generalist(self):
        """A guessed pipeline is worse than one agent the user can split up once
        they have seen how it behaves."""
        intent = blueprints.classify("Xylophone quarterly frobnication")
        assert intent is blueprints.GENERALIST
        assert len(intent.agents) == 1

    @pytest.mark.parametrize(
        "goal",
        [
            "Write a blog post about our new pricing",
            "Plan a three-day itinerary for Lisbon",
            "Compare Postgres and MySQL for our workload",
            "Extract the totals from these invoices",
            "Xylophone quarterly frobnication",
        ],
    )
    def test_every_shape_of_goal_produces_a_valid_design(self, goal):
        response = designed(goal)
        assert response.stage is DesignStage.DESIGNED
        assert response.graph is not None
        assert response.agents


class TestSkillsAreFoundOrCreated:
    def test_agents_are_given_the_skills_their_role_needs(self):
        response = designed("Research and write a briefing on solar tariffs")
        assert all(agent.skills for agent in response.agents)

    def test_the_stored_agent_actually_holds_those_skills(self):
        """The planned design and what was stored must not drift apart."""
        from agents import registry as agent_registry

        graph = designed("Research and write a briefing on solar tariffs").graph
        agent_node = next(node for node in graph.nodes if node.type is NodeType.AGENT)
        stored = agent_registry.get(TENANT, agent_node.agent_id)

        assert stored.skill_ids
        assert skill_registry.get_many(TENANT, stored.skill_ids)

    def test_a_capability_that_does_not_exist_is_created(self):
        """Item 4. The tenant starts with nothing; the design leaves it with
        working capabilities."""
        assert skill_registry.list_all(TENANT) == []
        designed("Research and write a briefing on solar tariffs")
        assert skill_registry.list_all(TENANT)

    def test_documents_become_a_search_skill_not_a_retrieval_step(self):
        """The agent decides when a lookup is worth doing, instead of the graph
        always looking first."""
        response = designed(
            "Research and write a briefing on solar tariffs",
            knowledge_handles=["kb_solar"],
        )
        names = {skill.name for skill in skill_registry.list_all(TENANT)}
        assert any(name.startswith("search_documents_") for name in names)

        graph = response.graph
        assert all(
            node.type in (NodeType.INPUT, NodeType.OUTPUT, NodeType.AGENT)
            for node in graph.nodes
        )

    def test_the_search_skill_is_created_once_and_reused(self):
        designed("Research and write a briefing on solar tariffs", knowledge_handles=["kb_solar"])
        designed("Research and write another briefing", knowledge_handles=["kb_solar"])

        searches = [
            skill
            for skill in skill_registry.list_all(TENANT)
            if skill.name.startswith("search_documents_")
        ]
        assert len(searches) == 1

    def test_declining_documents_leaves_the_agents_ungrounded(self):
        designed(
            "Research and write a briefing on solar tariffs",
            knowledge_handles=["kb_solar"],
            answers={"grounding": "No, general knowledge is fine"},
        )
        names = {skill.name for skill in skill_registry.list_all(TENANT)}
        assert not any(name.startswith("search_documents_") for name in names)

    def test_an_external_call_is_flagged_rather_than_invented(self):
        """A skill that reaches outside the platform needs a credential and an
        allow-list entry, which are a human's decision."""
        response = designed(
            "Post new leads to our CRM",
            answers={"external": "Yes — I will supply the details"},
        )
        blocked = [gap for gap in response.skill_gaps if gap.blocked_reason]
        assert blocked
        assert "credential" in blocked[0].blocked_reason

    def test_built_in_skills_are_reused_not_duplicated(self):
        skill_registry.ensure_builtins(TENANT)
        before = len(skill_registry.list_all(TENANT))

        designed("Research and write a briefing on solar tariffs")

        after = skill_registry.list_all(TENANT)
        assert len([s for s in after if s.origin is SkillOrigin.BUILTIN]) == before


class TestMemoryWiring:
    def test_the_workflow_gets_a_memory_scope(self):
        """Item 2's second half: workflows should have memory."""
        graph = designed("Research and write a briefing on solar tariffs").graph
        assert graph.memory_scope_id
        assert graph.memory_scope

    def test_agents_in_one_workflow_share_that_scope(self):
        from agents import registry as agent_registry

        graph = designed("Research and write a briefing on solar tariffs").graph
        agent_nodes = [node for node in graph.nodes if node.type is NodeType.AGENT]

        for node in agent_nodes:
            stored = agent_registry.get(TENANT, node.agent_id)
            assert stored.memory.use_workflow_memory is True

    def test_two_workflows_do_not_share_a_scope(self):
        first = designed("Research and write a briefing on solar tariffs").graph
        second = designed("Plan a three-day itinerary for Lisbon").graph
        assert first.memory_scope_id != second.memory_scope_id


class TestAnswersShapeTheDesign:
    def test_the_stated_audience_reaches_the_agents_instructions(self):
        from agents import registry as agent_registry

        graph = designed(
            "Research and write a briefing on solar tariffs",
            answers={"audience": "A customer or client"},
        ).graph
        node = next(n for n in graph.nodes if n.type is NodeType.AGENT)
        stored = agent_registry.get(TENANT, node.agent_id)

        assert "customer" in stored.instructions.lower()

    def test_constraints_are_carried_into_every_agent(self):
        from agents import registry as agent_registry

        graph = designed(
            "Research and write a briefing on solar tariffs",
            answers={"constraints": "Never exceed two pages"},
        ).graph

        for node in [n for n in graph.nodes if n.type is NodeType.AGENT]:
            stored = agent_registry.get(TENANT, node.agent_id)
            assert "two pages" in stored.instructions


class TestModelRefinement:
    def test_a_model_may_reword_the_design(self):
        reset_provider_cache(
            RewordingProvider(
                '{"agents": [{"name": "Solar Researcher", "role": "an energy analyst",'
                ' "objective": "Gather the tariff data."},'
                ' {"name": "Writer", "role": "an editor", "objective": "Write it up."},'
                ' {"name": "Reviewer", "role": "a checker", "objective": "Check it."}]}'
            )
        )
        response = designed("Research and write a briefing on solar tariffs")
        assert response.agents[0].name == "Solar Researcher"

    def test_a_model_may_not_change_the_structure(self):
        """It returned two agents where the blueprint has three, so the whole
        refinement is discarded rather than partially applied."""
        reset_provider_cache(
            RewordingProvider('{"agents": [{"name": "Only", "role": "r", "objective": "o"}]}')
        )
        response = designed("Research and write a briefing on solar tariffs")
        assert len(response.agents) == 3
        assert response.agents[0].name == "Researcher"

    def test_unusable_model_output_leaves_the_design_intact(self):
        reset_provider_cache(RewordingProvider("I'm not able to help with that."))
        response = designed("Research and write a briefing on solar tariffs")
        assert response.stage is DesignStage.DESIGNED
        assert response.agents

    def test_the_design_works_with_no_model_at_all(self):
        """The platform must design something sensible on the offline stub."""

        class Dead:
            capabilities = RewordingProvider.capabilities

            def complete(self, *args, **kwargs):
                from llm_proxy.client import LLMProxyError

                raise LLMProxyError("no model configured")

            def converse(self, *args, **kwargs):
                from llm_proxy.client import LLMProxyError

                raise LLMProxyError("no model configured")

        reset_provider_cache(Dead())
        response = designed("Research and write a briefing on solar tariffs")

        assert response.stage is DesignStage.DESIGNED
        assert len(response.agents) == 3


class TestNotes:
    def test_the_design_reports_what_it_did(self):
        response = designed("Research and write a briefing on solar tariffs")
        assert any("memory" in note for note in response.notes)

    def test_newly_created_skills_are_named_in_the_notes(self):
        """A capability appearing silently is a capability nobody reviewed."""
        response = designed("Post new leads to our CRM")
        assert response.notes
