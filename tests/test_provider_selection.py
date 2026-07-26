"""The workspace's chosen model is the one that runs.

A user configured LM Studio in the Models dialog, saw `LM Studio (local) ·
google/gemma-4-e4b` in the header, pressed Run, and every step came back
`[stub:...]`. The workspace setting was stored, displayed, and ignored.

The cause was one module global doing two jobs. `_provider` was both:

* **a pin** — what `reset_provider_cache` sets so tests and the dev runner are
  never routed somewhere they did not choose; and
* **a lazy cache of the deployment default** — assigned the first time anybody
  asked for a provider while no tenant configuration was in scope.

The second poisons the first. `/health` calls `describe_provider()`, which calls
`get_provider()` with no tenant in context, so a container healthcheck pins the
whole process to the deployment default within seconds of boot — and from then
on the very first line of `get_provider` returns that default and the tenant's
stored choice is never even looked up.

It is worth being precise about why this was invisible in testing: in the suite
a provider is *deliberately* pinned, so the same early-return is the correct
behaviour and every test passes. The bug only appears when nothing is pinned,
which is every real deployment and no test. These are the tests for the unpinned
case.
"""

from __future__ import annotations

import pytest
from llm_proxy import client as client_module
from llm_proxy import store

TENANT = "tenant-selection"


@pytest.fixture
def unpinned():
    """No pinned provider — the state every real deployment runs in."""
    client_module.reset_provider_cache(None)
    yield
    client_module.reset_provider_cache(None)


def a_workspace_on_lmstudio() -> None:
    store.save(
        TENANT,
        provider="lmstudio",
        model="google/gemma-4-e4b",
        base_url="http://localhost:1234/v1",
    )


class TestTheWorkspaceChoiceIsUsed:
    def test_a_stored_choice_is_what_runs(self, isolated_platform, unpinned):
        a_workspace_on_lmstudio()

        with store.acting_for(TENANT):
            chosen = client_module.get_provider()

        assert chosen.capabilities.provider == "lmstudio"
        assert chosen.capabilities.model == "google/gemma-4-e4b"

    def test_asking_without_a_tenant_first_does_not_poison_it(
        self, isolated_platform, unpinned
    ):
        """The actual bug. `/health` asks with no tenant in scope, and that must
        not decide what every later caller gets."""
        a_workspace_on_lmstudio()

        client_module.get_provider()  # what a container healthcheck does
        with store.acting_for(TENANT):
            chosen = client_module.get_provider()

        assert chosen.capabilities.provider == "lmstudio"

    def test_describe_provider_does_not_poison_it_either(
        self, isolated_platform, unpinned
    ):
        """`/health` reaches `get_provider` through this, so it is the real
        entry point rather than a hypothetical one."""
        a_workspace_on_lmstudio()

        client_module.describe_provider()
        with store.acting_for(TENANT):
            chosen = client_module.get_provider()

        assert chosen.capabilities.provider == "lmstudio"

    def test_two_tenants_do_not_share_a_provider(self, isolated_platform, unpinned):
        a_workspace_on_lmstudio()
        store.save("tenant-other", provider="ollama", model="llama3.1")

        with store.acting_for(TENANT):
            first = client_module.get_provider()
        with store.acting_for("tenant-other"):
            second = client_module.get_provider()

        assert first.capabilities.provider == "lmstudio"
        assert second.capabilities.provider == "ollama"

    def test_no_stored_choice_still_gets_the_deployment_default(
        self, isolated_platform, unpinned
    ):
        """Removing the poisoning must not remove the fallback."""
        with store.acting_for("tenant-with-nothing-saved"):
            chosen = client_module.get_provider()

        assert chosen is not None
        assert chosen.capabilities.provider


class TestAnAgentsOwnChoiceIsUsed:
    def test_an_agent_naming_a_backend_gets_it(self, isolated_platform, unpinned):
        """`provider_for_choice` short-circuited on the same poisoned global, so
        even an explicit per-agent backend was overridden by it."""
        client_module.get_provider()  # poison, under the old behaviour

        chosen = client_module.provider_for_choice("ollama", "llama3.1")

        assert chosen.capabilities.provider == "ollama"
        assert chosen.capabilities.model == "llama3.1"

    def test_an_agent_naming_nothing_falls_back_to_the_workspace(
        self, isolated_platform, unpinned
    ):
        a_workspace_on_lmstudio()
        client_module.get_provider()

        with store.acting_for(TENANT):
            chosen = client_module.provider_for_choice("", "")

        assert chosen.capabilities.provider == "lmstudio"


class TestPinningStillWins:
    """The half of the old behaviour that was correct, and has to survive.

    Tests and the dev runner pin a provider deliberately. If the fix for the
    above let a stored tenant choice override a pin, the suite would start
    reaching real endpoints — a much worse bug than the one being fixed.
    """

    def test_a_pin_beats_a_stored_workspace_choice(self, isolated_platform):
        a_workspace_on_lmstudio()
        pinned = client_module.StubProvider("stub-model")
        client_module.reset_provider_cache(pinned)
        try:
            with store.acting_for(TENANT):
                assert client_module.get_provider() is pinned
        finally:
            client_module.reset_provider_cache(None)

    def test_a_pin_beats_an_agents_own_choice(self, isolated_platform):
        pinned = client_module.StubProvider("stub-model")
        client_module.reset_provider_cache(pinned)
        try:
            assert client_module.provider_for_choice("ollama", "llama3.1") is pinned
        finally:
            client_module.reset_provider_cache(None)

    def test_unpinning_lets_the_workspace_choice_through_again(self, isolated_platform):
        a_workspace_on_lmstudio()
        client_module.reset_provider_cache(client_module.StubProvider("stub-model"))
        client_module.reset_provider_cache(None)
        try:
            with store.acting_for(TENANT):
                assert client_module.get_provider().capabilities.provider == "lmstudio"
        finally:
            client_module.reset_provider_cache(None)
