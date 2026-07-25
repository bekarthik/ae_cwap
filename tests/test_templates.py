"""Templates: a starting point that already works.

The third way into the canvas. Describing a goal is the most powerful route and a
blank canvas is the most direct, but both start from nothing, and "what can this
thing actually do?" is a new user's first question.

Two properties carry the weight:

* **A copy, not a reference.** Instantiating creates the tenant's own agents, so
  editing one changes your workflow and nobody else's.
* **Prerequisites are reported up front.** A template that quietly produces a
  workflow which fails on its third step is worse than one that says "upload
  something first".
"""

from __future__ import annotations

import pytest
from agents import registry as agent_registry
from cwap_contracts.v4 import NodeType
from skills import registry as skill_registry
from templates import service
from templates.catalogue import all_templates

TENANT = "tenant-a"


class TestTheCatalogue:
    def test_every_template_is_offered(self):
        assert {entry.key for entry in service.catalogue(TENANT)} == {
            template.key for template in all_templates()
        }

    def test_each_one_says_what_running_it_does(self):
        """A list of titles is a menu nobody can order from."""
        for entry in service.catalogue(TENANT):
            assert entry.summary and entry.detail
            assert entry.example_input, f"{entry.key} has no worked example"

    def test_each_agent_is_explained(self):
        for entry in service.catalogue(TENANT):
            assert entry.agents
            assert all(agent["rationale"] for agent in entry.agents)

    def test_templates_are_grouped(self):
        assert len({entry.category for entry in service.catalogue(TENANT)}) > 1


class TestPrerequisites:
    def test_a_document_template_says_to_upload_something_first(self):
        entry = next(e for e in service.catalogue(TENANT) if e.key == "document_qa")
        assert any("upload" in need.lower() for need in entry.requires)

    def test_the_requirement_disappears_once_it_is_met(self):
        from cwap_contracts.v4 import IngestRequest
        from knowledge.service import ingest

        ingest(
            IngestRequest(
                tenant_id=TENANT, title="Handbook", content="Expenses over 500 need approval."
            )
        )
        entry = next(e for e in service.catalogue(TENANT) if e.key == "document_qa")
        assert entry.requires == []

    def test_a_connector_template_says_what_to_connect(self):
        entry = next(e for e in service.catalogue(TENANT) if e.key == "code_review")
        assert entry.requires
        assert "connect" in entry.requires[0].lower()

    def test_a_template_with_no_prerequisites_reports_none(self):
        entry = next(e for e in service.catalogue(TENANT) if e.key == "research_brief")
        assert entry.requires == []


class TestInstantiating:
    def test_a_template_becomes_a_runnable_graph(self):
        result = service.instantiate("research_brief", TENANT)
        graph = result.graph

        assert graph.nodes[0].type is NodeType.INPUT
        assert graph.nodes[-1].type is NodeType.OUTPUT
        assert len(graph.edges) == len(graph.nodes) - 1

    def test_every_working_step_is_an_agent(self):
        """Same kind of object as a designed workflow — otherwise half the
        product's behaviour would depend on where a workflow came from."""
        graph = service.instantiate("research_brief", TENANT).graph
        working = [
            node for node in graph.nodes if node.type not in (NodeType.INPUT, NodeType.OUTPUT)
        ]

        assert working
        assert all(node.type is NodeType.AGENT and node.agent_id for node in working)

    def test_the_agents_are_created_for_this_tenant(self):
        service.instantiate("research_brief", TENANT)
        assert {agent.name for agent in agent_registry.list_all(TENANT)} >= {
            "Researcher",
            "Writer",
            "Reviewer",
        }

    def test_the_skills_each_agent_needs_are_resolved(self):
        graph = service.instantiate("research_brief", TENANT).graph
        node = next(n for n in graph.nodes if n.type is NodeType.AGENT)
        stored = agent_registry.get(TENANT, node.agent_id)

        assert stored.skill_ids
        assert skill_registry.get_many(TENANT, stored.skill_ids)

    def test_using_a_template_twice_gives_two_independent_copies(self):
        """Otherwise editing one workflow silently changes another."""
        first = service.instantiate("research_brief", TENANT).graph
        second = service.instantiate("research_brief", TENANT).graph

        assert first.id != second.id
        first_agents = {n.agent_id for n in first.nodes if n.type is NodeType.AGENT}
        second_agents = {n.agent_id for n in second.nodes if n.type is NodeType.AGENT}
        assert first_agents.isdisjoint(second_agents)

    def test_each_copy_gets_its_own_memory_scope(self):
        first = service.instantiate("research_brief", TENANT).graph
        second = service.instantiate("research_brief", TENANT).graph
        assert first.memory_scope_id != second.memory_scope_id

    def test_the_worked_example_becomes_the_default_input(self):
        """So a user can press Run immediately and see what it does."""
        graph = service.instantiate("research_brief", TENANT).graph
        assert graph.nodes[0].params["defaults"]["goal"]

    def test_an_unknown_template_is_a_typed_error(self):
        with pytest.raises(service.TemplateNotFound):
            service.instantiate("no-such-template", TENANT)

    @pytest.mark.parametrize("key", [t.key for t in all_templates()])
    def test_every_template_instantiates(self, key):
        result = service.instantiate(key, TENANT)
        assert result.graph.nodes
        assert result.agents
        assert result.notes

    def test_a_document_template_gets_a_search_skill_when_there_are_documents(self):
        from cwap_contracts.v4 import IngestRequest
        from knowledge.service import ingest

        ingest(IngestRequest(tenant_id=TENANT, title="Handbook", content="Some policy."))
        service.instantiate("document_qa", TENANT)

        names = {skill.name for skill in skill_registry.list_all(TENANT)}
        assert any(name.startswith("search_documents_") for name in names)


class TestTemplatesActuallyRun:
    @pytest.mark.parametrize("key", [t.key for t in all_templates()])
    def test_a_template_runs_end_to_end(self, key, authorized_user):
        """The claim a template makes is "this works". Verified for all of them,
        against the offline stub so it holds with no credentials at all."""
        from cwap_common.db import read_only_session
        from cwap_common.models import Run
        from orchestrator.runner import RUN_SUCCEEDED, run_to_completion

        result = service.instantiate(key, authorized_user.tenant_id)
        run_id = run_to_completion(
            graph=result.graph, job_context=authorized_user, inputs={}
        )

        with read_only_session() as session:
            run = session.get(Run, run_id)
            status, error, output = run.status, run.error, run.result

        assert status == RUN_SUCCEEDED, error
        assert output and output.get("result")


class TestTheApi:
    def test_templates_are_listed(self, client, auth):
        body = client.get("/api/templates", headers=auth).json()
        assert len(body) == len(all_templates())
        assert all(entry["summary"] for entry in body)

    def test_using_one_returns_a_graph_ready_for_the_canvas(self, client, auth):
        body = client.post("/api/templates/research_brief", headers=auth).json()

        assert body["graph"]["nodes"]
        assert body["agents"]
        assert body["notes"]

    def test_an_unknown_template_is_a_404(self, client, auth):
        assert client.post("/api/templates/nope", headers=auth).status_code == 404

    def test_templates_require_authentication(self, client):
        assert client.get("/api/templates").status_code in (401, 403)
