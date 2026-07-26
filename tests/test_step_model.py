"""Running one step on a different model, without changing the agent.

The agent editor answers "which model does this agent use", and that is the
wrong grain for a decision as local as "the synthesis step in *this* workflow
should use the big model". Changing the agent changes every workflow that uses
it. The only workaround was to clone the agent, which splits its memory in two
and makes both halves worse — the opposite of what a platform whose parts
improve across runs is for.

So a node may override the agent's model for itself. Three levels, each
inheriting from the next when blank:

    step  →  agent  →  workspace

The override lives in `node.params`, alongside `objective_template`, which
overrides the agent's objective in exactly the same way and for the same reason.
`params` is the executor-specific half of a node, and this is as
executor-specific as it gets.

What matters most here is the *negative* space: an untouched node behaves
exactly as it did before this existed, and an override never writes back to the
stored agent.
"""

from __future__ import annotations

import pytest
from agents import registry as agent_registry
from cwap_contracts.v4 import NodeType, WorkflowNode
from orchestrator import executors

TENANT = "tenant-step-model"


@pytest.fixture
def agent(isolated_platform):
    return agent_registry.create(
        TENANT,
        name="Analyst",
        role="You analyse things.",
        model_provider="ollama",
        model_override="llama3.1",
        thinking_effort="low",
    )


def node(**params) -> WorkflowNode:
    return WorkflowNode(
        id="agent_1", type=NodeType.AGENT, agent_id="agt_x", params=params
    )


class TestTheStepCanOverrideTheAgent:
    def test_a_named_provider_replaces_the_agents(self, agent):
        adjusted = executors._with_node_model(
            agent, node(model_provider="anthropic").params, "agent_1"
        )

        assert adjusted.model_provider == "anthropic"

    def test_a_named_model_replaces_the_agents(self, agent):
        adjusted = executors._with_node_model(
            agent,
            node(model_provider="anthropic", model_override="claude-opus-5").params,
            "agent_1",
        )

        assert adjusted.model_override == "claude-opus-5"

    def test_effort_can_be_raised_for_one_step(self, agent):
        """The case that motivates the whole thing: the same agent, thinking
        harder in the one place it matters."""
        adjusted = executors._with_node_model(
            agent, node(thinking_effort="high").params, "agent_1"
        )

        assert adjusted.thinking_effort == "high"

    def test_what_is_not_overridden_is_kept(self, agent):
        adjusted = executors._with_node_model(
            agent, node(thinking_effort="high").params, "agent_1"
        )

        assert adjusted.model_provider == "ollama"
        assert adjusted.model_override == "llama3.1"

    def test_everything_else_about_the_agent_is_untouched(self, agent):
        adjusted = executors._with_node_model(
            agent, node(model_provider="anthropic").params, "agent_1"
        )

        assert adjusted.id == agent.id
        assert adjusted.name == agent.name
        assert adjusted.role == agent.role
        assert adjusted.skill_ids == agent.skill_ids


class TestItChangesNothingItShouldNot:
    def test_an_untouched_node_returns_the_same_agent(self, agent):
        """Identity, not equality: the common path must not even copy."""
        assert executors._with_node_model(agent, {}, "agent_1") is agent

    def test_an_unrelated_param_changes_nothing(self, agent):
        params = node(objective_template="Summarise {{previous}}").params

        assert executors._with_node_model(agent, params, "agent_1") is agent

    def test_a_blank_override_means_inherit(self, agent):
        """An empty select is "use the agent's", not "use no provider"."""
        params = node(model_provider="", model_override="", thinking_effort="").params

        assert executors._with_node_model(agent, params, "agent_1") is agent

    def test_whitespace_is_not_a_choice(self, agent):
        assert executors._with_node_model(agent, node(model_provider="  ").params, "x") is agent

    def test_the_stored_agent_is_never_written_to(self, agent):
        executors._with_node_model(agent, node(model_provider="anthropic").params, "agent_1")
        reloaded = agent_registry.get(TENANT, agent.id)

        assert reloaded.model_provider == "ollama"

    def test_two_nodes_can_differ_from_each_other(self, agent):
        """One agent, two steps, two backends — which is the point."""
        cheap = executors._with_node_model(agent, node(model_provider="ollama").params, "a")
        strong = executors._with_node_model(agent, node(model_provider="anthropic").params, "b")

        assert cheap.model_provider != strong.model_provider


class TestBadInputIsCaughtAtTheNode:
    def test_an_unknown_effort_names_the_node(self, agent):
        """`params` is a free-form dict that no schema checks, so this is the
        only place it can be caught. Failing at the provider instead would name
        neither the node nor the field."""
        with pytest.raises(executors.NodeExecutionError) as raised:
            executors._with_node_model(agent, node(thinking_effort="enormous").params, "agent_7")

        assert "agent_7" in str(raised.value)

    def test_it_says_what_was_wrong(self, agent):
        with pytest.raises(executors.NodeExecutionError, match="unknown thinking effort"):
            executors._with_node_model(agent, node(thinking_effort="enormous").params, "agent_7")

    def test_a_number_is_accepted_as_text(self, agent):
        """Canvas params arrive as JSON, so a value may not be a string."""
        adjusted = executors._with_node_model(agent, {"model_override": 4}, "agent_1")

        assert adjusted.model_override == "4"


class TestItSurvivesTheGraph:
    def test_a_node_carrying_an_override_still_validates(self):
        """`params` is free-form, so the graph contract must not object."""
        built = node(model_provider="anthropic", model_override="claude-opus-5")

        assert built.params["model_provider"] == "anthropic"

    def test_the_api_round_trips_it(self, client, auth):
        created = client.post(
            "/api/agents", headers=auth, json={"name": "Analyst", "role": "You analyse."}
        ).json()

        saved = client.put(
            "/api/workflows/wf_step_model",
            headers=auth,
            json={
                "graph": {
                    "id": "wf_step_model",
                    "name": "Per-step model",
                    "nodes": [
                        {"id": "input", "type": "input", "params": {"fields": ["goal"]}},
                        {
                            "id": "agent_1",
                            "type": "agent",
                            "agent_id": created["id"],
                            "params": {
                                "objective_template": "Do {{goal}}",
                                "model_provider": "ollama",
                                "model_override": "llama3.1",
                                "thinking_effort": "high",
                            },
                        },
                        {"id": "output", "type": "output", "params": {}},
                    ],
                    "edges": [
                        {"id": "e1", "source": "input", "target": "agent_1"},
                        {"id": "e2", "source": "agent_1", "target": "output"},
                    ],
                },
            },
        )

        assert saved.status_code == 200, saved.text
        params = next(
            node["params"]
            for node in saved.json()["graph"]["nodes"]
            if node["id"] == "agent_1"
        )
        assert params["model_provider"] == "ollama"
        assert params["thinking_effort"] == "high"
