"""How long to wait for a model, and who gets to decide.

A run failed at 120 seconds against a local model that was still thinking. The
limit was a guess about somebody else's hardware, and the only way to change it
was an environment variable and a restart — on a deployment the person waiting
may not administer. Two things follow: the default has to suit the machine this
platform is most often pointed at (a laptop), and the number has to be settable
from the same screen where the model is chosen.
"""

from __future__ import annotations

import httpx
import pytest
from cwap_common.settings import get_settings, reset_settings_cache
from llm_proxy import store
from llm_proxy.client import OpenAICompatibleProvider, provider_for


@pytest.fixture
def settings_reset():
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestTheDefaultSuitsTheHardware:
    def test_the_deployment_default_allows_for_a_slow_local_model(self, settings_reset):
        """A reasoning model on a laptop can think for minutes before its first
        token. Two minutes was a limit that failed runs that were working."""
        assert get_settings().llm_timeout_seconds >= 600

    def test_a_provider_built_with_no_timeout_uses_that_default(self, settings_reset):
        provider = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        )
        assert provider._timeout == get_settings().llm_timeout_seconds

    def test_an_operator_can_still_set_one(self, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_TIMEOUT", "45")
        reset_settings_cache()
        try:
            assert get_settings().llm_timeout_seconds == 45
        finally:
            monkeypatch.delenv("CWAP_LLM_TIMEOUT")
            reset_settings_cache()


class TestATenantCanChangeItWithoutARestart:
    def test_a_stored_timeout_reaches_the_provider(self, settings_reset):
        config = store.StoredModelConfig(
            provider="ollama",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            timeout_seconds=900,
        )
        assert provider_for(config)._timeout == 900

    def test_zero_means_the_deployment_default(self, settings_reset):
        config = store.StoredModelConfig(
            provider="ollama", model="llama3.1", base_url="http://localhost:11434/v1"
        )
        assert provider_for(config)._timeout == get_settings().llm_timeout_seconds

    def test_it_survives_a_round_trip_through_the_database(self, isolated_platform):
        store.save(
            "tenant-a",
            provider="ollama",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            timeout_seconds=1200,
        )
        assert store.load("tenant-a").timeout_seconds == 1200

    def test_changing_only_the_timeout_still_rebuilds_the_client(self, isolated_platform):
        """The provider cache is keyed by configuration; a key that ignored the
        timeout would answer the next call with the old, short-lived client."""
        from llm_proxy import client as client_module

        store.save(
            "tenant-a",
            provider="ollama",
            model="llama3.1",
            base_url="http://localhost:11434/v1",
            timeout_seconds=300,
        )
        with store.acting_for("tenant-a"):
            client_module.reset_provider_cache(None)
            first = client_module.get_provider()

            store.save(
                "tenant-a",
                provider="ollama",
                model="llama3.1",
                base_url="http://localhost:11434/v1",
                timeout_seconds=900,
            )
            second = client_module.get_provider()

        assert first._timeout == 300
        assert second._timeout == 900

    def test_it_is_sent_when_testing_a_configuration_before_saving(self, isolated_platform):
        """Otherwise a slow local model fails the test that was meant to prove
        the setting works."""
        from llm_proxy import service

        seen: dict[str, object] = {}
        original = service.build_provider

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        service.build_provider = spy
        try:
            service.test_connection(
                "ollama",
                model="llama3.1",
                base_url="http://localhost:11434/v1",
                timeout_seconds=777,
            )
        finally:
            service.build_provider = original

        assert seen["timeout"] == 777


class TestTheApi:
    def test_the_field_is_saved_and_reported(self, client, auth):
        saved = client.put(
            "/api/models",
            headers=auth,
            json={
                "provider": "ollama",
                "model": "llama3.1",
                "base_url": "http://localhost:11434/v1",
                "timeout_seconds": 900,
            },
        )
        assert saved.status_code == 200

        body = client.get("/api/models", headers=auth).json()
        assert body["stored"]["timeout_seconds"] == 900

    def test_the_default_is_reported_so_the_field_can_show_it(self, client, auth):
        body = client.get("/api/models", headers=auth).json()
        assert body["default_timeout_seconds"] == get_settings().llm_timeout_seconds

    def test_an_unbounded_wait_is_refused(self, client, auth):
        """A request that can pin a worker forever is not a setting."""
        response = client.put(
            "/api/models",
            headers=auth,
            json={"provider": "ollama", "model": "llama3.1", "timeout_seconds": 99_999},
        )
        assert response.status_code == 422


class TestTheMessageSaysWhereToFixIt:
    def test_a_timeout_points_at_the_control_not_only_the_variable(self, monkeypatch):
        """The person reading this may not be able to set an environment
        variable, and does not need to."""
        provider = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        )

        def timeout(request):
            raise httpx.ReadTimeout("too slow", request=request)

        transport = httpx.MockTransport(timeout)
        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kwargs: real_client(
                transport=transport, **{k: v for k, v in kwargs.items() if k != "transport"}
            ),
        )

        with pytest.raises(Exception, match="How long to wait") as caught:
            provider.complete("hello")
        assert "CWAP_LLM_TIMEOUT" in str(caught.value)
