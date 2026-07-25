"""Changing model backend without restarting the server.

The claim being defended: a user picks a provider, sees what models that endpoint
actually has, tests the configuration, saves it, and the next run uses it. No
environment edit, no restart.

Three properties carry the weight:

* **A tenant's choice reaches a running step.** The provider is resolved at call
  time from a context the worker sets, not from a process-wide global.
* **One tenant's choice is not another's.** Including their credentials.
* **A stored credential goes in and never comes out.** Every response says
  whether a key is set; none says what it is.
"""

from __future__ import annotations

import httpx
import pytest
from cwap_common.db import read_only_session
from cwap_common.models import ModelSetting
from cwap_common.secrets import PREFIX, decrypt, encrypt
from cwap_common.settings import reset_settings_cache
from llm_proxy import service, store
from llm_proxy.client import get_provider, reset_provider_cache


@pytest.fixture
def patched_httpx(monkeypatch):
    def install(handler):
        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def factory(**kwargs):
            kwargs.pop("transport", None)
            return real_client(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", factory)

    return install


def models_response(*ids: str):
    return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})


class TestSecretsAtRest:
    def test_a_credential_round_trips(self):
        assert decrypt(encrypt("sk-secret-value")) == "sk-secret-value"

    def test_stored_form_is_not_the_plaintext(self):
        stored = encrypt("sk-secret-value")
        assert stored.startswith(PREFIX)
        assert "sk-secret-value" not in stored

    def test_an_empty_credential_stays_empty(self):
        assert encrypt("") == ""
        assert decrypt("") == ""

    def test_a_plaintext_value_is_tolerated(self):
        """So an operator can seed a row by hand, and a row written before
        encryption existed keeps working instead of failing opaquely."""
        assert decrypt("sk-written-by-hand") == "sk-written-by-hand"

    def test_a_rotated_secret_is_a_clear_error(self, monkeypatch):
        from cwap_common.secrets import SecretError

        stored = encrypt("sk-secret-value")
        monkeypatch.setenv("CWAP_SECRET_KEY", "a-completely-different-secret")
        reset_settings_cache()

        with pytest.raises(SecretError, match="CWAP_SECRET_KEY has changed"):
            decrypt(stored)


class TestStoringAChoice:
    def test_a_saved_choice_is_read_back(self):
        service.save_choice(
            "tenant-a", provider="ollama", model="llama3.1", base_url="http://box:11434/v1"
        )
        stored = store.load("tenant-a")
        assert (stored.provider, stored.model) == ("ollama", "llama3.1")

    def test_no_choice_means_the_deployment_default(self):
        assert store.load("tenant-a") is None

    def test_an_unknown_provider_is_refused(self):
        from llm_proxy.client import LLMConfigurationError

        with pytest.raises(LLMConfigurationError):
            service.save_choice("tenant-a", provider="not-a-real-provider")

    def test_the_credential_is_encrypted_in_the_database(self):
        """The property that matters for a leaked backup."""
        service.save_choice("tenant-a", provider="openai", model="gpt-4o", api_key="sk-live-key")

        with read_only_session() as session:
            row = session.query(ModelSetting).filter_by(tenant_id="tenant-a").one()

        assert row.api_key != "sk-live-key"
        assert row.api_key.startswith(PREFIX)
        assert store.load("tenant-a").api_key == "sk-live-key"

    def test_changing_model_keeps_the_stored_credential(self):
        """The API never returns a key, so a UI re-saving the form has nothing to
        send back. Without "None means unchanged" this would erase it."""
        service.save_choice("tenant-a", provider="openai", model="gpt-4o", api_key="sk-live-key")
        service.save_choice("tenant-a", provider="openai", model="gpt-4o-mini", api_key=None)

        stored = store.load("tenant-a")
        assert (stored.model, stored.api_key) == ("gpt-4o-mini", "sk-live-key")

    def test_a_credential_can_be_cleared_explicitly(self):
        service.save_choice("tenant-a", provider="openai", model="gpt-4o", api_key="sk-live-key")
        service.save_choice("tenant-a", provider="ollama", model="llama3.1", api_key="")
        assert store.load("tenant-a").api_key == ""

    def test_clearing_returns_to_the_deployment_default(self):
        service.save_choice("tenant-a", provider="ollama", model="llama3.1")
        assert service.clear_choice("tenant-a") is True
        assert store.load("tenant-a") is None


class TestResolutionAtCallTime:
    def test_a_tenants_choice_wins_over_the_deployment_default(self):
        service.save_choice("tenant-a", provider="ollama", model="llama3.1")
        reset_provider_cache(None)

        with store.acting_for("tenant-a"):
            assert get_provider().capabilities.model == "llama3.1"

    def test_a_tenant_with_no_choice_gets_the_deployment_default(self):
        service.save_choice("tenant-a", provider="ollama", model="llama3.1")
        reset_provider_cache(None)

        with store.acting_for("tenant-b"):
            assert get_provider().capabilities.provider == "stub"

    def test_one_tenants_choice_does_not_leak_into_another(self):
        """Two tenants, two endpoints, in the same process."""
        service.save_choice("tenant-a", provider="ollama", model="llama3.1")
        service.save_choice("tenant-b", provider="openai", model="gpt-4o", api_key="sk-b")
        reset_provider_cache(None)

        with store.acting_for("tenant-a"):
            a = get_provider().capabilities
        with store.acting_for("tenant-b"):
            b = get_provider().capabilities

        assert (a.provider, b.provider) == ("ollama", "openai")

    def test_the_context_does_not_outlive_its_block(self):
        """A fungible worker moving to another tenant's job must not inherit the
        previous tenant's endpoint or credential."""
        service.save_choice("tenant-a", provider="ollama", model="llama3.1")
        reset_provider_cache(None)

        with store.acting_for("tenant-a"):
            pass
        assert get_provider().capabilities.provider == "stub"

    def test_saving_takes_effect_without_a_restart(self):
        """The whole point. No process boundary between the save and the run."""
        reset_provider_cache(None)
        with store.acting_for("tenant-a"):
            assert get_provider().capabilities.provider == "stub"

            service.save_choice("tenant-a", provider="ollama", model="llama3.1")
            assert get_provider().capabilities.model == "llama3.1"

    def test_a_broken_configuration_store_falls_back_rather_than_failing(self, monkeypatch):
        """The proxy must work before the schema exists and in a process with no
        database — falling back to the environment, not failing the model call."""
        def explode(*args, **kwargs):
            raise RuntimeError("no such table: model_settings")

        monkeypatch.setattr(store, "load", explode)
        reset_provider_cache(None)

        with store.acting_for("tenant-a"):
            assert get_provider().capabilities.provider == "stub"


class TestDetectingModels:
    def test_an_endpoint_is_asked_what_it_serves(self, patched_httpx):
        """Rather than asking the user to type the id from memory."""
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return models_response("llama3.1", "qwen2.5", "nomic-embed-text")

        patched_httpx(handler)
        found = service.available_models("ollama")

        assert seen["url"].endswith("/v1/models")
        assert {model.id for model in found} == {"llama3.1", "qwen2.5", "nomic-embed-text"}

    def test_a_detected_model_carries_its_real_capabilities(self, patched_httpx):
        patched_httpx(lambda request: models_response("llama3.2-vision"))
        [found] = service.available_models("ollama")

        assert found.known is True
        assert (found.vision, found.tools) == (True, False)

    def test_a_model_the_catalogue_has_never_seen_still_appears(self, patched_httpx):
        """Hiding it would make a private fine-tune unusable from the picker."""
        patched_httpx(lambda request: models_response("acme/internal-finetune-v3"))
        [found] = service.available_models("vllm", base_url="http://localhost:8000/v1")

        assert found.id == "acme/internal-finetune-v3"
        assert found.known is False

    def test_catalogued_models_are_listed_first(self, patched_httpx):
        patched_httpx(lambda request: models_response("zzz-unknown", "llama3.1"))
        found = service.available_models("ollama")
        assert [model.id for model in found] == ["llama3.1", "zzz-unknown"]

    def test_a_bare_list_response_is_accepted(self, patched_httpx):
        """Not every OpenAI-compatible server wraps the list in `data`."""
        patched_httpx(lambda request: httpx.Response(200, json=[{"id": "llama3.1"}]))
        assert [m.id for m in service.available_models("ollama")] == ["llama3.1"]

    def test_a_server_that_is_not_running_says_so(self, patched_httpx):
        from llm_proxy.client import LLMProxyError

        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        patched_httpx(handler)
        with pytest.raises(LLMProxyError, match="is the server running"):
            service.available_models("ollama")

    def test_a_rejected_key_says_so(self, patched_httpx):
        from llm_proxy.client import LLMProxyError

        patched_httpx(lambda request: httpx.Response(401, json={"error": "bad key"}))
        with pytest.raises(LLMProxyError, match="rejected the API key"):
            service.available_models("openai", api_key="sk-wrong")

    def test_a_server_without_a_model_list_suggests_the_fallback(self, patched_httpx):
        from llm_proxy.client import LLMProxyError

        patched_httpx(lambda request: httpx.Response(404, text="not found"))
        with pytest.raises(LLMProxyError, match="enter the model id directly"):
            service.available_models("llamacpp")

    def test_detection_uses_the_stored_key_when_none_is_supplied(self, patched_httpx):
        """So "detect models" works after a save, without the browser holding a
        credential it was never given back."""
        service.save_choice("tenant-a", provider="openai", model="gpt-4o", api_key="sk-stored")
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return models_response("gpt-4o")

        patched_httpx(handler)
        service.available_models("openai", tenant_id="tenant-a")

        assert seen["auth"] == "Bearer sk-stored"

    def test_a_stored_key_is_not_sent_to_a_different_provider(self, patched_httpx):
        """Reusing an OpenAI key against a newly typed endpoint would send the
        credential somewhere it was never meant to go."""
        service.save_choice("tenant-a", provider="openai", model="gpt-4o", api_key="sk-stored")
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return models_response("llama3.1")

        patched_httpx(handler)
        service.available_models("ollama", tenant_id="tenant-a")

        assert seen["auth"] is None


class TestTestingAConnection:
    def test_a_working_configuration_reports_what_it_can_do(self, patched_httpx):
        patched_httpx(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "choices": [
                        {"message": {"content": "ready"}, "finish_reason": "stop"}
                    ],
                },
            )
        )
        result = service.test_connection("ollama", model="llama3.1")

        assert result.ok
        assert "ready" in result.message
        assert result.supports["tools"] is True

    def test_a_failure_is_returned_as_data_not_raised(self, patched_httpx):
        """Every outcome here is something the user has to read and act on; a 500
        would tell them less than the backend's own message."""

        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        patched_httpx(handler)
        result = service.test_connection("ollama", model="llama3.1")

        assert result.ok is False
        assert "is the server running" in result.message

    def test_a_missing_credential_is_reported_before_any_request(self):
        result = service.test_connection("openai", model="gpt-4o")
        assert result.ok is False
        assert "API key" in result.message


class TestTheApiNeverReturnsACredential:
    def test_the_configuration_endpoint_reports_only_that_a_key_is_set(self, client, auth):
        client.put(
            "/api/models",
            json={"provider": "openai", "model": "gpt-4o", "api_key": "sk-live-key"},
            headers=auth,
        )
        body = client.get("/api/models", headers=auth).json()

        assert body["stored"]["has_api_key"] is True
        assert "sk-live-key" not in str(body)

    def test_the_save_response_does_not_echo_the_key(self, client, auth):
        body = client.put(
            "/api/models",
            json={"provider": "openai", "model": "gpt-4o", "api_key": "sk-live-key"},
            headers=auth,
        ).json()
        assert "sk-live-key" not in str(body)

    def test_configuration_requires_authentication(self, client):
        assert client.get("/api/models").status_code in (401, 403)


class TestTheApiFlow:
    def test_the_picker_is_useful_before_anything_is_detected(self, client, auth):
        """A server that is not running yet should still offer real choices."""
        body = client.get("/api/models", headers=auth).json()
        ollama = next(p for p in body["providers"] if p["key"] == "ollama")

        assert body["source"] == "deployment"
        assert ollama["models"], "the curated catalogue should be offered up front"

    def test_detection_failure_is_reported_as_data(self, client, auth, patched_httpx):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        patched_httpx(handler)
        body = client.post(
            "/api/models/detect", json={"provider": "ollama"}, headers=auth
        ).json()

        assert body["ok"] is False
        assert body["models"] == []

    def test_saving_switches_what_the_tenant_runs_on(self, client, auth, patched_httpx):
        patched_httpx(lambda request: models_response("llama3.1"))

        client.put(
            "/api/models", json={"provider": "ollama", "model": "llama3.1"}, headers=auth
        )
        body = client.get("/api/models", headers=auth).json()

        assert body["source"] == "tenant"
        assert body["stored"]["model"] == "llama3.1"

    def test_a_tenant_can_go_back_to_the_deployment_default(self, client, auth):
        client.put(
            "/api/models", json={"provider": "ollama", "model": "llama3.1"}, headers=auth
        )
        client.delete("/api/models", headers=auth)

        assert client.get("/api/models", headers=auth).json()["source"] == "deployment"

    def test_custom_endpoints_can_be_switched_off(self, client, auth, monkeypatch):
        """A tenant-supplied base URL is a request this server will make, so a
        deployment that does not want that can keep the preset URLs only."""
        monkeypatch.setenv("CWAP_ALLOW_CUSTOM_MODEL_ENDPOINTS", "false")
        reset_settings_cache()

        response = client.put(
            "/api/models",
            json={"provider": "ollama", "base_url": "http://169.254.169.254/v1"},
            headers=auth,
        )
        assert response.status_code == 403

    def test_a_listed_provider_still_works_when_custom_endpoints_are_off(
        self, client, auth, monkeypatch
    ):
        monkeypatch.setenv("CWAP_ALLOW_CUSTOM_MODEL_ENDPOINTS", "false")
        reset_settings_cache()

        response = client.put(
            "/api/models", json={"provider": "ollama", "model": "llama3.1"}, headers=auth
        )
        assert response.status_code == 200


class TestItReachesARunningStep:
    def test_a_workflow_runs_on_the_tenants_chosen_backend(
        self, authorized_user, patched_httpx
    ):
        """The end of the chain. A choice saved through the API is what the
        worker actually calls, without the process restarting.
        """
        from conftest import make_linear_graph
        from orchestrator.runner import RUN_SUCCEEDED, run_to_completion

        calls: list[str] = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "choices": [
                        {
                            "message": {"content": "answered by the chosen backend"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        patched_httpx(handler)
        service.save_choice(
            authorized_user.tenant_id,
            provider="ollama",
            model="llama3.1",
            base_url="http://chosen-box:11434/v1",
        )
        reset_provider_cache(None)

        run_id = run_to_completion(
            graph=make_linear_graph(),
            job_context=authorized_user,
            inputs={"goal": "anything"},
        )

        from cwap_common.db import read_only_session as _session
        from cwap_common.models import Run

        with _session() as session:
            run = session.get(Run, run_id)
            status, result = run.status, run.result

        assert status == RUN_SUCCEEDED
        assert "answered by the chosen backend" in result["result"]
        assert any("chosen-box" in url for url in calls), calls

    def test_another_tenants_run_is_unaffected(self, authorized_user, patched_httpx):
        """Two tenants' jobs pass through the same worker process."""
        from conftest import make_linear_graph
        from orchestrator.runner import run_to_completion

        patched_httpx(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "choices": [{"message": {"content": "chosen"}, "finish_reason": "stop"}],
                },
            )
        )
        service.save_choice("some-other-tenant", provider="ollama", model="llama3.1")
        reset_provider_cache(None)

        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "anything"}
        )

        from cwap_common.db import read_only_session as _session
        from cwap_common.models import Run

        with _session() as session:
            result = session.get(Run, run_id).result

        # The stub is deterministic and prefixes its output; the other tenant's
        # endpoint was never consulted.
        assert "[stub:" in result["result"]
