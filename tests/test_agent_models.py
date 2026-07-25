"""An agent runs on the model it was given, not the one the workspace picked.

`model_override` was on the contract, in the database and in the API response
for two versions, and nothing ever read it. That is worse than not offering it:
a field that is stored and ignored is a promise the product does not keep, and
"mix providers freely within one workflow" was that promise.

Three things have to hold for it to be true:

* an agent that names a backend runs on that backend, and one that names nothing
  still runs on the workspace's — an existing workflow cannot change behaviour;
* the credential comes from what the tenant saved *for that backend*, because a
  key entered for one provider must never be sent to another's endpoint;
* effort is per agent, since triage wants `low` and synthesis wants `high` and
  one deployment setting cannot be both.
"""

from __future__ import annotations

import pytest
from agents import registry as agent_registry
from agents import runtime
from cwap_contracts.v4 import AgentDefinition
from llm_proxy import client as client_module
from llm_proxy import store

TENANT = "tenant-models"


def an_agent(**changes) -> AgentDefinition:
    base = {
        "id": "agt_model",
        "tenant_id": TENANT,
        "name": "Analyst",
        "role": "You analyse things.",
    }
    return AgentDefinition(**{**base, **changes})


class TestTheAgentChoosesTheModel:
    def test_an_agent_that_names_nothing_uses_the_workspace_model(self, isolated_platform):
        chosen = runtime._provider_for(an_agent())

        assert chosen is client_module.get_provider()

    def test_an_agent_that_names_a_backend_gets_it(self, isolated_platform):
        client_module.reset_provider_cache(None)
        chosen = runtime._provider_for(
            an_agent(model_provider="ollama", model_override="llama3.1")
        )

        assert chosen.capabilities.provider == "ollama"
        assert chosen.capabilities.model == "llama3.1"

    def test_two_agents_in_one_workflow_can_differ(self, isolated_platform):
        """The whole point. A triage step and a synthesis step, one run."""
        client_module.reset_provider_cache(None)
        triage = runtime._provider_for(
            an_agent(model_provider="ollama", model_override="llama3.1")
        )
        synthesis = runtime._provider_for(
            an_agent(id="agt_two", model_provider="vllm", model_override="Qwen/Qwen2.5-7B")
        )

        assert triage.capabilities.provider != synthesis.capabilities.provider
        assert triage.capabilities.model != synthesis.capabilities.model

    def test_a_pinned_provider_still_wins(self, isolated_platform):
        """Tests and the dev runner must never be routed somewhere else."""
        pinned = client_module.StubProvider("stub-model")
        client_module.reset_provider_cache(pinned)
        try:
            assert runtime._provider_for(an_agent(model_provider="ollama")) is pinned
        finally:
            client_module.reset_provider_cache(None)


class TestWhereTheCredentialComesFrom:
    def test_a_key_saved_for_that_provider_is_used(self, isolated_platform):
        store.save(
            TENANT,
            kind=store.kind_for_provider("together"),
            provider="together",
            api_key="tok-together",
        )
        with store.acting_for(TENANT):
            found = store.credential_for(TENANT, "together")

        assert found is not None
        assert found.api_key == "tok-together"

    def test_the_workspace_key_counts_when_it_is_the_same_provider(self, isolated_platform):
        store.save(TENANT, provider="groq", model="llama-3.3-70b-versatile", api_key="tok-groq")

        assert store.credential_for(TENANT, "groq").api_key == "tok-groq"

    def test_a_key_for_one_provider_is_never_sent_to_another(self, isolated_platform):
        """The rule that makes storing several keys safe."""
        store.save(TENANT, provider="openai", model="gpt-4o", api_key="tok-openai")

        assert store.credential_for(TENANT, "together") is None

    def test_the_workspace_reports_which_backends_it_can_reach(self, isolated_platform):
        from llm_proxy import service

        store.save(TENANT, provider="openai", model="gpt-4o", api_key="tok-openai")
        store.save(
            TENANT,
            kind=store.kind_for_provider("together"),
            provider="together",
            api_key="tok-together",
        )

        assert service.configured_providers(TENANT) == ["openai", "together"]

    def test_a_local_backend_needs_no_key_to_count_as_reachable(self, isolated_platform):
        from llm_proxy import service

        store.save(TENANT, provider="ollama", model="llama3.1")

        assert "ollama" in service.configured_providers(TENANT)


class TestEffortIsPerAgent:
    def test_an_agent_asks_for_its_own_depth(self):
        assert runtime._options(an_agent(thinking_effort="low")).effort == "low"

    def test_blank_inherits_the_deployment_setting(self):
        assert runtime._options(an_agent()).effort is None

    def test_an_unknown_depth_is_refused_at_the_contract(self):
        with pytest.raises(ValueError, match="unknown thinking effort"):
            an_agent(thinking_effort="enormous")

    def test_it_reaches_the_model(self, isolated_platform, monkeypatch):
        seen: dict[str, object] = {}

        class Recorder:
            capabilities = client_module.StubProvider("stub-model").capabilities

            def converse(self, messages, *, system=None, tools=None, options=None, on_delta=None):
                seen["effort"] = options.effort if options else None
                return client_module.LLMCompletion(text="done", model="stub-model")

            def complete(self, prompt, *, system=None, options=None, on_delta=None):
                return client_module.LLMCompletion(text="done", model="stub-model")

        monkeypatch.setattr(runtime, "_provider_for", lambda agent: Recorder())
        agent = agent_registry.create(
            TENANT, name="Analyst", role="You analyse.", thinking_effort="high"
        )
        runtime.run_agent(agent, "Do the thing", runtime.AgentRunContext(tenant_id=TENANT))

        assert seen["effort"] == "high"


class TestItSurvivesStorage:
    def test_the_choice_round_trips(self, isolated_platform):
        agent = agent_registry.create(
            TENANT,
            name="Triage",
            role="You sort things quickly.",
            model_provider="ollama",
            model_override="llama3.1",
            thinking_effort="low",
        )
        loaded = agent_registry.get(TENANT, agent.id)

        assert (loaded.model_provider, loaded.model_override, loaded.thinking_effort) == (
            "ollama",
            "llama3.1",
            "low",
        )

    def test_the_api_accepts_and_returns_it(self, client, auth):
        created = client.post(
            "/api/agents",
            headers=auth,
            json={
                "name": "Synthesist",
                "role": "You write the final answer.",
                "model_provider": "anthropic",
                "model_override": "claude-opus-5",
                "thinking_effort": "high",
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["model_provider"] == "anthropic"
        assert body["thinking_effort"] == "high"

    def test_the_api_refuses_an_unknown_depth(self, client, auth):
        response = client.post(
            "/api/agents",
            headers=auth,
            json={"name": "X", "role": "Y", "thinking_effort": "enormous"},
        )
        assert response.status_code == 422
