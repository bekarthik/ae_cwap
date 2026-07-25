"""The workspace as one graph.

Every other endpoint answers a question about one kind of thing. None of them
answered the question a person actually has when they open the product — what is
in here, and what is wired to what — even though the relationships all existed
already: an agent holds skills, a skill came from a connected server, a workflow
staffs its steps with agents, memory belongs to one of them.

This is a read-only projection, so the tests care about two things: that the
relationships are real, and that it never claims one that is not.
"""

from __future__ import annotations

from agents import registry as agent_registry
from conftest import make_linear_graph
from cwap_contracts.v4 import (
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    NodeType,
    ReviewConfig,
)
from memory import service as memory_service
from skills import registry as skill_registry


def workspace(client, auth) -> dict:
    response = client.get("/api/workspace/map", headers=auth)
    assert response.status_code == 200
    return response.json()


def ids(body: dict, kind: str) -> set[str]:
    return {node["id"] for node in body["nodes"] if node["kind"] == kind}


class TestItShowsWhatIsThere:
    def test_an_empty_workspace_is_empty_rather_than_an_error(self, client, auth):
        body = workspace(client, auth)

        assert body["nodes"] == []
        assert body["counts"]["agents"] == 0

    def test_agents_and_their_skills_are_linked(self, client, auth, isolated_platform):
        tenant = "tenant-a"
        skill_registry.ensure_builtins(tenant)
        skill = skill_registry.list_all(tenant)[0]
        agent = agent_registry.create(
            tenant, name="Researcher", role="You research.", skill_ids=[skill.id]
        )

        body = workspace(client, auth)

        assert agent.id in ids(body, "agent")
        assert skill.id in ids(body, "skill")
        assert {"source": agent.id, "target": skill.id, "kind": "holds"} in body["links"]

    def test_a_workflow_links_to_the_agents_that_staff_it(
        self, client, auth, isolated_platform
    ):
        tenant = "tenant-a"
        agent = agent_registry.create(tenant, name="Writer", role="You write.")
        graph = make_linear_graph()
        staffed = graph.model_copy(
            update={
                "nodes": [
                    node.model_copy(
                        update={"type": NodeType.AGENT, "agent_id": agent.id}
                    )
                    if node.id == "think"
                    else node
                    for node in graph.nodes
                ]
            }
        )
        client.put(
            f"/api/workflows/{staffed.id}",
            headers=auth,
            json={"graph": staffed.model_dump(mode="json")},
        )

        body = workspace(client, auth)

        assert {"source": staffed.id, "target": agent.id, "kind": "staffs"} in body["links"]

    def test_a_reviewer_is_shown_as_its_own_relationship(
        self, client, auth, isolated_platform
    ):
        """Who checks whose work is worth seeing at a glance."""
        tenant = "tenant-a"
        author = agent_registry.create(tenant, name="Author", role="You write.")
        reviewer = agent_registry.create(tenant, name="Reviewer", role="You review.")
        graph = make_linear_graph()
        reviewed = graph.model_copy(
            update={
                "nodes": [
                    node.model_copy(
                        update={
                            "type": NodeType.AGENT,
                            "agent_id": author.id,
                            "review": ReviewConfig(agent_id=reviewer.id),
                        }
                    )
                    if node.id == "think"
                    else node
                    for node in graph.nodes
                ]
            }
        )
        client.put(
            f"/api/workflows/{reviewed.id}",
            headers=auth,
            json={"graph": reviewed.model_dump(mode="json")},
        )

        body = workspace(client, auth)

        assert {
            "source": reviewed.id,
            "target": reviewer.id,
            "kind": "reviews",
        } in body["links"]

    def test_what_a_thing_has_learned_is_counted(self, client, auth, isolated_platform):
        """Drawn as glow: a neuron that has learned something is brighter."""
        tenant = "tenant-a"
        agent = agent_registry.create(tenant, name="Analyst", role="You analyse.")
        memory_service.remember(
            MemoryWriteRequest(
                tenant_id=tenant,
                scope=MemoryScope.AGENT,
                scope_id=agent.id,
                kind=MemoryKind.LEARNING,
                text="Check the primary source.",
            )
        )

        body = workspace(client, auth)
        node = next(item for item in body["nodes"] if item["id"] == agent.id)

        assert node["memories"] == 1
        assert body["counts"]["memories"] == 1

    def test_an_agent_on_its_own_backend_shows_the_backend(
        self, client, auth, isolated_platform
    ):
        tenant = "tenant-a"
        agent_registry.create(
            tenant, name="Triage", role="You sort.", model_provider="ollama"
        )

        body = workspace(client, auth)

        assert "provider:ollama" in ids(body, "provider")


class TestItNeverClaimsWhatIsNotThere:
    def test_a_link_to_something_missing_is_dropped(self, client, auth, isolated_platform):
        """A skill deleted while an agent still lists it would otherwise be an
        edge into empty space."""
        tenant = "tenant-a"
        agent_registry.create(
            tenant, name="Ghost", role="You hold a skill that is gone.", skill_ids=["sk_gone"]
        )

        body = workspace(client, auth)

        assert all(link["target"] != "sk_gone" for link in body["links"])

    def test_another_tenant_is_invisible(self, client, auth, isolated_platform):
        agent_registry.create("someone-else", name="Theirs", role="Not yours.")

        assert workspace(client, auth)["nodes"] == []

    def test_it_needs_a_session(self, client):
        assert client.get("/api/workspace/map").status_code == 401
