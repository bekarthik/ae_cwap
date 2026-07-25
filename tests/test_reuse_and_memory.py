"""Two halves of "your workspace gets better", both of which were missing.

**The planner created every agent it named.** Design the same kind of goal twice
and the workspace held two Researchers, each with its own empty memory. That is
the opposite of the promise: an agent is supposed to improve across runs, and it
cannot if every plan hands the job to a brand-new one.

**Workflow memory was write-only in the wrong direction.** Every agent step read
the workflow's memory into its prompt, and nothing ever wrote to it, so the layer
could only hold what somebody typed by hand. "Recent run outcomes are injected as
context" was true of the reading half alone.
"""

from __future__ import annotations

from agents import registry as agent_registry
from conftest import make_linear_graph
from cwap_contracts.v4 import DesignRequest, MemoryScope, NodeType, RecallRequest
from design.service import design
from memory import service as memory_service
from orchestrator.runner import Worker, start_run

TENANT = "tenant-reuse"


def a_design(goal: str = "Research vector databases and write a briefing"):
    return design(DesignRequest(goal=goal, skip_questions=True), TENANT)


class TestThePlannerReusesWhatIsThere:
    def test_designing_twice_does_not_double_the_roster(self, isolated_platform):
        first = a_design()
        after_first = len(agent_registry.list_all(TENANT))

        a_design()
        after_second = len(agent_registry.list_all(TENANT))

        assert after_first == len(first.agents)
        assert after_second == after_first

    def test_the_same_agent_object_is_wired_into_the_second_workflow(
        self, isolated_platform
    ):
        """Reuse has to reach the *graph*, or the design merely avoids a row."""
        first = a_design()
        second = a_design()

        def agent_ids(response):
            return [
                node.agent_id
                for node in response.graph.nodes
                if node.type is NodeType.AGENT
            ]

        assert agent_ids(first) == agent_ids(second)

    def test_a_reused_agent_keeps_what_it_learned(self, isolated_platform):
        """The whole reason reuse beats a fresh copy."""
        first = a_design()
        agent_id = next(
            node.agent_id for node in first.graph.nodes if node.type is NodeType.AGENT
        )
        memory_service.remember(
            _lesson(agent_id, "Check the vendor's own docs before the blog posts.")
        )

        a_design()

        recalled = memory_service.recall(
            RecallRequest(
                tenant_id=TENANT, scope=MemoryScope.AGENT, scope_id=agent_id, query="docs"
            )
        )
        assert any("vendor" in entry.text for entry in recalled.entries)

    def test_the_plan_says_which_agents_are_already_yours(self, isolated_platform):
        a_design()
        second = a_design()

        assert all(agent.reused for agent in second.agents)
        assert any("Reusing" in note for note in second.notes)

    def test_a_first_design_claims_no_reuse(self, isolated_platform):
        first = a_design()

        assert not any(agent.reused for agent in first.agents)

    def test_a_reused_agent_gains_the_skills_the_new_plan_needs(self, isolated_platform):
        first = a_design()
        agent_id = next(
            node.agent_id for node in first.graph.nodes if node.type is NodeType.AGENT
        )
        before = set(agent_registry.get(TENANT, agent_id).skill_ids)

        a_design("Research vector databases and write a briefing, with sources")
        after = set(agent_registry.get(TENANT, agent_id).skill_ids)

        assert before <= after

    def test_a_reused_agent_keeps_the_role_a_person_may_have_edited(
        self, isolated_platform
    ):
        """A planner silently rewriting an edited description would undo the
        edit without saying so."""
        first = a_design()
        agent_id = next(
            node.agent_id for node in first.graph.nodes if node.type is NodeType.AGENT
        )
        agent_registry.update(TENANT, agent_id, role="You are deliberately terse.")

        a_design()

        assert agent_registry.get(TENANT, agent_id).role == "You are deliberately terse."


class TestAWorkflowRemembersItsRuns:
    def test_a_successful_run_writes_its_outcome_back(self, authorized_user):
        graph = make_linear_graph()
        start_run(graph=graph, job_context=authorized_user, inputs={"goal": "say hello"})
        Worker().drain()

        recalled = memory_service.recall(
            RecallRequest(
                tenant_id=authorized_user.tenant_id,
                scope=MemoryScope.WORKFLOW,
                scope_id=graph.memory_scope,
                query="hello",
            )
        )
        assert recalled.entries
        assert "previous run" in recalled.entries[0].text.lower()

    def test_a_failed_run_writes_nothing(self, authorized_user, monkeypatch):
        """A failure's output is a symptom, not a lesson; storing one would
        teach the next run to repeat it."""
        from orchestrator import executors

        def explode(request):
            raise executors.NodeExecutionError("nope")

        monkeypatch.setitem(executors.EXECUTORS, NodeType.LLM, explode)
        graph = make_linear_graph()
        start_run(graph=graph, job_context=authorized_user, inputs={"goal": "say hello"})
        Worker().drain()

        recalled = memory_service.recall(
            RecallRequest(
                tenant_id=authorized_user.tenant_id,
                scope=MemoryScope.WORKFLOW,
                scope_id=graph.memory_scope,
                query="hello",
            )
        )
        assert not recalled.entries

    def test_the_entry_is_context_not_a_transcript(self, authorized_user):
        """A whole report in every future prompt would crowd out the task."""
        from orchestrator.runner import OUTCOME_EXCERPT, _condense_result

        condensed = _condense_result({"result": "word " * 2000})

        assert len(condensed) < OUTCOME_EXCERPT + 64

    def test_an_empty_result_is_not_worth_remembering(self):
        from orchestrator.runner import _condense_result

        assert _condense_result({"result": "   "}) == ""
        assert _condense_result(None) == ""


def _lesson(agent_id: str, text: str):
    from cwap_contracts.v4 import MemoryKind, MemoryWriteRequest

    return MemoryWriteRequest(
        tenant_id=TENANT,
        scope=MemoryScope.AGENT,
        scope_id=agent_id,
        kind=MemoryKind.LEARNING,
        text=text,
    )
