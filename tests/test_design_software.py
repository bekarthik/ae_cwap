"""Designing work the platform's authors did not anticipate.

Reported: asking for "a workflow that acts as an entire software development
organization … proper SDLC … merge the developed code to github" produced
Researcher → Writer → Reviewer, and offered to deliver an email. Two causes:

* **No blueprint covered building software**, and the nearest one won on a single
  keyword. A goal matching one word out of forty is a coincidence, not a reading.
* **Nothing connected a design to the tenant's MCP servers.** An agent whose job
  is to open a pull request cannot do it with a prompt.

And underneath both, the general problem the report points at: the deterministic
blueprint approach can only produce shapes somebody wrote down. Where no
blueprint fits, the model now proposes a team, validated against the same
contracts, with the blueprint as the fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from agents import registry as agent_registry
from cwap_contracts.v3 import DesignRequest, DesignStage, MCPServerConfig, MCPTransport, NodeType
from design import blueprints
from design import service as design_service
from llm_proxy.client import (
    LLMCompletion,
    LLMProxyError,
    ProviderCapabilities,
    reset_provider_cache,
)
from mcp_connect import registry as mcp_registry
from mcp_connect.client import reset_client

TENANT = "tenant-a"
FIXTURE = str(Path(__file__).parent / "mcp_fixture_server.py")

SDLC_GOAL = (
    "design a workflow that acts as an entire software development organization. "
    "Given a task or idea, the system should properly research, split into smaller "
    "development activities and fully develop the product. Ensure proper SDLC "
    "lifecycle is implemented. create necessary skills and connect with proper MCP "
    "agents to be able to merge the developed code to github"
)


class ScriptedProvider:
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

    def __init__(self, text: str = "", *, fail: bool = False) -> None:
        self._text = text
        self._fail = fail
        self.systems: list[str] = []
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, options=None):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        if self._fail:
            raise LLMProxyError("no model configured")
        return LLMCompletion(text=self._text, model="scripted-model")

    def converse(self, messages, *, system=None, tools=None, options=None):
        return self.complete("")


def designed(goal: str, **overrides):
    return design_service.design(
        DesignRequest(goal=goal, skip_questions=True, **overrides), TENANT
    )


def stored_agents(response):
    return [
        agent_registry.get(TENANT, node.agent_id)
        for node in response.graph.nodes
        if node.type is NodeType.AGENT
    ]


class TestSoftwareGoalsAreRecognised:
    def test_the_reported_goal_is_read_as_building_software(self):
        intent, confident = blueprints.classify_with_confidence(SDLC_GOAL)
        assert intent.key == "build_software"
        assert confident

    @pytest.mark.parametrize(
        "goal",
        [
            "Fix the failing tests in the payments module and open a PR",
            "Refactor the auth service and merge it to github",
            "Implement a REST API for invoices with tests",
            "Build a small library for parsing our log format",
        ],
    )
    def test_other_engineering_goals_land_there_too(self, goal):
        assert blueprints.classify_with_confidence(goal)[0].key == "build_software"

    def test_a_single_incidental_keyword_is_not_a_confident_reading(self):
        """A forty-word brief containing "research" once is not a research task,
        which is exactly how the reported goal became "write an email"."""
        _intent, confident = blueprints.classify_with_confidence(
            "Some long description of an unusual process that happens to mention "
            "research somewhere in the middle of a great deal of other text about "
            "an entirely different subject"
        )
        assert not confident

    def test_the_deliverable_options_match_the_work(self):
        """It used to offer "An email" for a request to build software."""
        response = design_service.design(DesignRequest(goal=SDLC_GOAL), TENANT)
        [question] = [q for q in response.questions if q.id == "deliverable"]

        assert any("code" in option.lower() for option in question.options)
        assert not any("email" in option.lower() for option in question.options)


class TestTheSdlcShape:
    def test_the_full_lifecycle_is_staffed(self):
        response = designed(SDLC_GOAL)
        assert [agent.name for agent in response.agents] == [
            "Analyst",
            "Architect",
            "Engineer",
            "Reviewer",
            "Integrator",
        ]

    def test_a_pipeline_longer_than_four_is_representable(self):
        """The old ceiling would have truncated this to specify-plan-build and
        dropped review and integration."""
        assert len(designed(SDLC_GOAL).agents) == 5
        assert design_service.MAX_AGENTS >= 5

    def test_agents_that_iterate_get_the_budget_to_do_it(self):
        """The graph is acyclic, so a review-and-rework cycle cannot be an edge.
        Iteration lives inside the agent, which means the budget is where the
        design expresses it."""
        by_name = {agent.name: agent for agent in stored_agents(designed(SDLC_GOAL))}

        assert by_name["Engineer"].max_iterations > by_name["Analyst"].max_iterations
        assert by_name["Reviewer"].max_iterations > 6

    def test_each_agent_explains_why_it_is_there(self):
        assert all(agent.rationale for agent in designed(SDLC_GOAL).agents)

    def test_it_still_runs_end_to_end(self, authorized_user):
        """A five-agent pipeline is only a design until it executes."""
        from cwap_common.db import read_only_session
        from cwap_common.models import Run
        from orchestrator.runner import RUN_SUCCEEDED, run_to_completion

        response = design_service.design(
            DesignRequest(goal=SDLC_GOAL, skip_questions=True), authorized_user.tenant_id
        )
        run_id = run_to_completion(
            graph=response.graph, job_context=authorized_user, inputs={}
        )

        with read_only_session() as session:
            run = session.get(Run, run_id)
            assert run.status == RUN_SUCCEEDED, run.error

    def test_an_objective_never_asks_for_a_value_twice(self):
        """These templates embed {{goal}} where it reads best, so appending it
        blindly would hand the model the same text twice."""
        graph = designed(SDLC_GOAL).graph
        for node in graph.nodes:
            if node.type is NodeType.AGENT:
                template = node.params["objective_template"]
                assert template.count("{{goal}}") <= 1
                assert template.count("{{previous}}") <= 1


class TestConnectedSystemsReachTheDesign:
    @pytest.fixture(autouse=True)
    def _drop_sessions(self):
        yield
        reset_client(None)

    @pytest.fixture
    def code_host(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", sys.executable)
        result = mcp_registry.connect(
            TENANT,
            name="repo",
            config=MCPServerConfig(
                transport=MCPTransport.STDIO, command=sys.executable, args=[FIXTURE]
            ),
            description="Read and write source code",
        )
        assert result.ok, result.message

    def test_an_agent_that_must_touch_a_repository_is_given_the_tools(self, code_host):
        """The point of connecting a server: the design uses it. Without this the
        Engineer would be asked to write code it cannot read or save."""
        by_name = {agent.name: agent for agent in stored_agents(designed(SDLC_GOAL))}
        engineer_skills = {
            skill.name
            for skill in __import__("skills.registry", fromlist=["x"]).get_many(
                TENANT, by_name["Engineer"].skill_ids
            )
        }

        assert any(name.startswith("repo_") for name in engineer_skills), engineer_skills

    def test_the_integrator_gets_the_tool_that_opens_a_pull_request(self, code_host):
        by_name = {agent.name: agent for agent in stored_agents(designed(SDLC_GOAL))}
        from skills import registry as skill_registry

        names = {
            skill.name
            for skill in skill_registry.get_many(TENANT, by_name["Integrator"].skill_ids)
        }
        assert "repo_create_pull_request" in names

    def test_an_agent_with_no_need_for_them_is_not_handed_them(self, code_host):
        """A tool list padded with irrelevant capabilities makes a model worse at
        choosing from it."""
        from skills import registry as skill_registry

        by_name = {agent.name: agent for agent in stored_agents(designed(SDLC_GOAL))}
        analyst = {
            skill.name for skill in skill_registry.get_many(TENANT, by_name["Analyst"].skill_ids)
        }
        assert not any(name.startswith("repo_") for name in analyst)

    def test_without_a_connection_the_gap_is_reported(self):
        """Rather than a workflow that looks complete and can reach nothing."""
        response = designed(SDLC_GOAL)
        blocked = [gap for gap in response.skill_gaps if gap.blocked_reason]

        assert blocked
        assert any("Connected systems" in gap.blocked_reason for gap in blocked)
        assert any(gap.needed_by == "Integrator" for gap in blocked)

    def test_the_gap_disappears_once_the_server_is_connected(self, code_host):
        response = designed(SDLC_GOAL)
        blocked = [gap for gap in response.skill_gaps if "not connected" in gap.blocked_reason]
        assert blocked == []


class TestAnUnrecognisedGoalIsPlannedByTheModel:
    def test_a_proposed_team_is_used(self):
        reset_provider_cache(
            ScriptedProvider(
                '{"agents": ['
                '{"name": "Surveyor", "role": "a surveyor of frobnication",'
                ' "objective": "Survey {{goal}}", "skills": ["survey_sites"],'
                ' "rationale": "Somebody has to look first."},'
                '{"name": "Frobnicator", "role": "a frobnicator",'
                ' "objective": "Frobnicate using {{previous}}", "skills": ["frobnicate"],'
                ' "rationale": "This is the actual work."}]}'
            )
        )
        response = designed("Xylophone quarterly frobnication across the northern sites")

        assert [agent.name for agent in response.agents] == ["Surveyor", "Frobnicator"]
        assert response.stage is DesignStage.DESIGNED

    def test_the_skills_it_asks_for_are_created(self):
        """It may name capabilities nobody built; that is what synthesis is for."""
        reset_provider_cache(
            ScriptedProvider(
                '{"agents": [{"name": "Surveyor", "role": "a surveyor",'
                ' "objective": "Survey {{goal}}", "skills": ["survey_sites"],'
                ' "rationale": "Someone must look."}]}'
            )
        )
        response = designed("Xylophone quarterly frobnication")

        from skills import registry as skill_registry

        assert skill_registry.find_by_name(TENANT, "survey_sites") is not None
        assert response.graph is not None

    def test_the_user_is_told_the_design_was_proposed(self):
        """It will not be identical next time, which is a different promise from
        the deterministic path and has to be stated."""
        reset_provider_cache(
            ScriptedProvider(
                '{"agents": [{"name": "Surveyor", "role": "a surveyor",'
                ' "objective": "Survey {{goal}}", "skills": ["summarise"],'
                ' "rationale": "Someone must look."}]}'
            )
        )
        response = designed("Xylophone quarterly frobnication")
        assert any("proposed" in note for note in response.notes)

    def test_a_confident_classification_is_a_hint_not_a_cage(self):
        """A blueprint that fits is worth telling the model about — and it is
        still free to decide otherwise. Fixing the structure in advance is what
        produced three writers for a software organisation."""
        provider = ScriptedProvider('{"agents": [{"name": "Different", "role": "r",'
                                    ' "objective": "Do {{goal}}", "skills": [],'
                                    ' "rationale": "x"}]}')
        reset_provider_cache(provider)
        response = designed("Research our competitors and write a short comparison")

        assert [agent.name for agent in response.agents] == ["Different"]
        # The blueprint was offered as context rather than imposed.
        assert "Researcher" in provider.prompts[0]

    def test_the_prompt_does_not_fix_the_agent_count(self):
        """The reported string: "exactly the same number of agents"."""
        from design.service import PLAN_SYSTEM

        assert "same number of agents" not in PLAN_SYSTEM
        assert "HOW MANY" in PLAN_SYSTEM

    @pytest.mark.parametrize(
        "text",
        [
            "I'm afraid I can't help with that.",
            '{"agents": []}',
            '{"agents": "not a list"}',
            '{"agents": [{"name": "", "role": "r", "objective": "o"}]}',
            '{"agents": [{"name": "A", "role": "r", "objective": "o"},'
            ' {"name": "A", "role": "r", "objective": "o"}]}',
            "{}",
        ],
        ids=["refusal", "empty", "wrong type", "missing name", "duplicate names", "no agents"],
    )
    def test_an_unusable_proposal_falls_back_to_a_blueprint(self, text):
        """Half a design is not a design. A goal the model cannot plan for still
        produces a working workflow."""
        reset_provider_cache(ScriptedProvider(text))
        response = designed("Xylophone quarterly frobnication")

        assert response.stage is DesignStage.DESIGNED
        assert response.agents
        assert response.graph is not None

    def test_it_works_with_no_model_at_all(self):
        reset_provider_cache(ScriptedProvider(fail=True))
        response = designed("Xylophone quarterly frobnication")

        assert response.stage is DesignStage.DESIGNED
        assert [agent.name for agent in response.agents] == ["Assistant"]

    def test_a_proposed_objective_always_references_its_inputs(self):
        """Otherwise the rendered prompt contains literal braces, or an agent is
        handed a predecessor's work it never mentions."""
        reset_provider_cache(
            ScriptedProvider(
                '{"agents": [{"name": "One", "role": "r", "objective": "Do the thing",'
                ' "skills": [], "rationale": "x"},'
                '{"name": "Two", "role": "r", "objective": "Do the other thing",'
                ' "skills": [], "rationale": "x"}]}'
            )
        )
        response = designed("Xylophone quarterly frobnication")

        first, second = response.agents
        assert "{{goal}}" in first.objective
        assert "{{previous}}" in second.objective

    def test_a_proposal_cannot_exceed_the_agent_ceiling(self):
        agents = ",".join(
            f'{{"name": "A{i}", "role": "r", "objective": "o", "skills": [], "rationale": "x"}}'
            for i in range(20)
        )
        reset_provider_cache(ScriptedProvider(f'{{"agents": [{agents}]}}'))
        response = designed("Xylophone quarterly frobnication")

        assert len(response.agents) <= design_service.MAX_AGENTS

    def test_a_proposed_workflow_runs(self, authorized_user):
        """The end of it: a shape nobody wrote down, executing."""
        from cwap_common.db import read_only_session
        from cwap_common.models import Run
        from orchestrator.runner import RUN_SUCCEEDED, run_to_completion

        reset_provider_cache(
            ScriptedProvider(
                '{"agents": [{"name": "Surveyor", "role": "a surveyor",'
                ' "objective": "Survey {{goal}}", "skills": ["summarise"],'
                ' "rationale": "Someone must look."},'
                '{"name": "Reporter", "role": "a reporter",'
                ' "objective": "Report on {{previous}}", "skills": ["draft_text"],'
                ' "rationale": "Someone must write it up."}]}'
            )
        )
        response = design_service.design(
            DesignRequest(goal="Xylophone quarterly frobnication", skip_questions=True),
            authorized_user.tenant_id,
        )
        run_id = run_to_completion(
            graph=response.graph, job_context=authorized_user, inputs={}
        )

        with read_only_session() as session:
            run = session.get(Run, run_id)
            assert run.status == RUN_SUCCEEDED, run.error


class TestPermissionIsReportedBeforeTheRun:
    """Found by driving the reported goal through the real UI.

    The design attaches a tool that can open a pull request, the run authoriser
    then refuses without the WRITE_EXTERNAL scope, and the user discovers that
    after placing five agents on a canvas and pressing Run. The design already
    knows; it should say so.
    """

    @pytest.fixture(autouse=True)
    def _drop_sessions(self):
        yield
        reset_client(None)

    @pytest.fixture
    def code_host(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", sys.executable)
        assert mcp_registry.connect(
            TENANT,
            name="repo",
            config=MCPServerConfig(
                transport=MCPTransport.STDIO, command=sys.executable, args=[FIXTURE]
            ),
        ).ok

    def test_a_design_that_will_need_write_scope_says_so(self, code_host):
        response = designed(SDLC_GOAL)
        blocked = [gap for gap in response.skill_gaps if "WRITE_EXTERNAL" in gap.blocked_reason]

        assert blocked, [gap.blocked_reason for gap in response.skill_gaps]
        assert "administrator" in blocked[0].blocked_reason

    def test_a_read_only_design_does_not_ask_for_it(self):
        """Demanding the scope for a workflow that only reasons would train
        people to grant it for everything."""
        response = designed("Research our competitors and write a short comparison")
        assert not any(
            "WRITE_EXTERNAL" in gap.blocked_reason for gap in response.skill_gaps
        )

    def test_the_run_authoriser_agrees_with_the_design(self, code_host, authorized_user):
        """The design's claim and the gate's decision must not drift apart."""
        from api_gateway.routers.runs import _needs_write_scope

        response = design_service.design(
            DesignRequest(goal=SDLC_GOAL, skip_questions=True), authorized_user.tenant_id
        )
        predicted = any(
            "WRITE_EXTERNAL" in gap.blocked_reason for gap in response.skill_gaps
        )
        assert predicted == _needs_write_scope(response.graph, authorized_user.tenant_id)


class TestConnectorsFollowLeastPrivilege:
    """Found by reading the design the UI produced: the Architect and Reviewer
    were handed `create_pull_request`. Neither writes anything, and giving them
    the ability also made the whole workflow demand a write scope on behalf of
    agents that never use it."""

    @pytest.fixture(autouse=True)
    def _drop_sessions(self):
        yield
        reset_client(None)

    @pytest.fixture
    def code_host(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", sys.executable)
        assert mcp_registry.connect(
            TENANT,
            name="repo",
            config=MCPServerConfig(
                transport=MCPTransport.STDIO, command=sys.executable, args=[FIXTURE]
            ),
        ).ok

    def skills_of(self, name: str):
        from skills import registry as skill_registry

        by_name = {agent.name: agent for agent in stored_agents(designed(SDLC_GOAL))}
        return {
            skill.name: skill
            for skill in skill_registry.get_many(TENANT, by_name[name].skill_ids)
        }

    @pytest.mark.parametrize("reader", ["Architect", "Reviewer"])
    def test_an_agent_that_only_reads_gets_only_read_tools(self, code_host, reader):
        for skill in self.skills_of(reader).values():
            if skill.origin.value == "mcp":
                assert skill.definition.get("read_only") is True, skill.name

    @pytest.mark.parametrize("writer", ["Engineer", "Integrator"])
    def test_an_agent_that_must_change_something_still_can(self, code_host, writer):
        assert "repo_create_pull_request" in self.skills_of(writer)

    def test_a_reader_can_still_read(self, code_host):
        assert "repo_read_file" in self.skills_of("Reviewer")


class TestAModelDesignedTeamReachesConnectedSystems:
    """A team the platform did not write templates for still has to be able to
    do the job it describes. What it needs is read from what the model said
    about each agent, because there is no template to consult."""

    @pytest.fixture(autouse=True)
    def _drop_sessions(self):
        yield
        reset_client(None)

    @pytest.fixture
    def code_host(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", sys.executable)
        assert mcp_registry.connect(
            TENANT,
            name="repo",
            config=MCPServerConfig(
                transport=MCPTransport.STDIO, command=sys.executable, args=[FIXTURE]
            ),
        ).ok

    def plan(self, *agents: str):
        reset_provider_cache(ScriptedProvider('{"agents": [' + ",".join(agents) + "]}"))
        return designed("Some entirely unanticipated kind of work")

    def skills_of(self, response, name: str):
        from skills import registry as skill_registry

        stored = {agent.name: agent for agent in stored_agents(response)}
        return {
            skill.name: skill
            for skill in skill_registry.get_many(TENANT, stored[name].skill_ids)
        }

    def test_an_agent_that_says_it_opens_a_pull_request_gets_the_tool(self, code_host):
        response = self.plan(
            '{"name": "Lander", "role": "an engineer who opens a pull request for '
            'finished work", "objective": "Open a pull request for {{goal}}",'
            ' "skills": ["draft_text"], "rationale": "x"}'
        )
        assert "repo_create_pull_request" in self.skills_of(response, "Lander")

    def test_an_agent_that_only_reads_is_offered_only_read_tools(self, code_host):
        response = self.plan(
            '{"name": "Inspector", "role": "a reviewer who reads the source and '
            'reports problems", "objective": "Review {{goal}}", "skills": ["critique"],'
            ' "rationale": "x"}'
        )
        for skill in self.skills_of(response, "Inspector").values():
            if skill.origin.value == "mcp":
                assert skill.definition.get("read_only") is True, skill.name

    def test_an_agent_with_no_external_work_gets_no_tools(self, code_host):
        """A tool list padded with irrelevant capabilities makes a model worse at
        choosing from it."""
        response = self.plan(
            '{"name": "Thinker", "role": "an analyst who reasons about tradeoffs",'
            ' "objective": "Weigh the options in {{goal}}", "skills": ["summarise"],'
            ' "rationale": "x"}'
        )
        assert not any(
            name.startswith("repo_") for name in self.skills_of(response, "Thinker")
        )


class TestTheCeilingIsAReviewBoundNotAStructure:
    def test_a_full_delivery_pipeline_fits(self):
        assert design_service.MAX_AGENTS >= 6

    def test_the_model_is_told_where_it_stops(self):
        from design.service import PLAN_SYSTEM

        assert f"1 and {design_service.MAX_AGENTS} agents" in PLAN_SYSTEM
