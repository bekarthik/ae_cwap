"""Model portability: the platform must run on any backend, not just Claude.

These tests use `httpx.MockTransport`, so they exercise the real request
shaping and the real error mapping against a fake server — no network, no
model download, and no "it probably works against Ollama" hand-waving.
"""

from __future__ import annotations

import json

import httpx
import pytest
from cwap_common.settings import get_settings, reset_settings_cache
from knowledge.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    RemoteEmbedder,
    build_embedder,
    describe_embedder,
    set_embedder,
)
from llm_proxy.client import (
    GenerationOptions,
    LLMConfigurationError,
    LLMProxyError,
    OpenAICompatibleProvider,
    StubProvider,
    build_provider,
    describe_provider,
)
from llm_proxy.presets import PRESETS, known_providers, resolve


@pytest.fixture
def patched_httpx(monkeypatch):
    """Route every httpx.Client through a handler the test supplies."""

    def install(handler):
        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def factory(**kwargs):
            kwargs.pop("transport", None)
            return real_client(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", factory)

    return install


def chat_response(content: str, *, finish_reason: str = "stop", model: str = "llama3.1"):
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


class TestProviderSelection:
    def test_stub_is_the_default(self):
        provider = build_provider()
        assert provider.capabilities.provider == "stub"
        assert provider.capabilities.deterministic

    @pytest.mark.parametrize("key", ["ollama", "vllm", "lmstudio", "llamacpp", "litellm", "tgi"])
    def test_local_open_model_servers_need_no_credentials(self, key, monkeypatch):
        """Running an open-weight model on your own machine is a first-class
        deployment, not a fallback that demands a key anyway."""
        monkeypatch.setenv("CWAP_LLM_PROVIDER", key)
        monkeypatch.setenv("CWAP_LLM_MODEL", "some-open-model")
        reset_settings_cache()

        provider = build_provider()
        assert provider.capabilities.provider == key
        assert provider.capabilities.base_url

    @pytest.mark.parametrize("key", ["openai", "together", "groq", "openrouter", "mistral"])
    def test_hosted_providers_report_a_missing_key_clearly(self, key, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_PROVIDER", key)
        monkeypatch.delenv("CWAP_LLM_API_KEY", raising=False)
        reset_settings_cache()

        with pytest.raises(LLMConfigurationError, match="CWAP_LLM_API_KEY"):
            build_provider()

    def test_an_unknown_provider_lists_the_valid_ones(self, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_PROVIDER", "not-a-thing")
        reset_settings_cache()

        with pytest.raises(LLMConfigurationError) as excinfo:
            build_provider()
        assert "ollama" in str(excinfo.value)

    def test_a_bare_endpoint_needs_a_base_url(self, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("CWAP_LLM_MODEL", "whatever")
        reset_settings_cache()

        with pytest.raises(LLMConfigurationError, match="CWAP_LLM_BASE_URL"):
            build_provider()

    def test_a_server_with_no_model_configured_says_so(self, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_PROVIDER", "vllm")
        monkeypatch.delenv("CWAP_LLM_MODEL", raising=False)
        reset_settings_cache()

        with pytest.raises(LLMConfigurationError, match="CWAP_LLM_MODEL"):
            build_provider()

    def test_base_url_overrides_the_preset(self, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("CWAP_LLM_MODEL", "qwen2.5")
        monkeypatch.setenv("CWAP_LLM_BASE_URL", "http://gpu-box.internal:11434/v1")
        reset_settings_cache()

        assert build_provider().capabilities.base_url == "http://gpu-box.internal:11434/v1"

    def test_every_preset_is_reachable_by_name(self):
        for key in PRESETS:
            assert resolve(key) is not None
        assert "anthropic" in known_providers() and "ollama" in known_providers()


class TestOpenAICompatibleRequests:
    def test_the_request_matches_the_chat_completions_contract(self, patched_httpx):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return chat_response("the answer")

        patched_httpx(handler)
        provider = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        )
        completion = provider.complete(
            "What is 2+2?",
            system="Be terse.",
            options=GenerationOptions(max_tokens=128, temperature=0.2),
        )

        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        body = captured["body"]
        assert body["model"] == "llama3.1"
        assert body["max_tokens"] == 128
        assert body["temperature"] == 0.2
        assert body["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        assert completion.text == "the answer"
        assert completion.input_tokens == 11 and completion.output_tokens == 7

    def test_a_trailing_slash_in_the_base_url_does_not_double_up(self, patched_httpx):
        seen: list[str] = []
        patched_httpx(lambda request: (seen.append(str(request.url)), chat_response("ok"))[1])

        OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1/", model="llama3.1"
        ).complete("hi")

        assert seen == ["http://localhost:11434/v1/chat/completions"]

    def test_an_api_key_is_sent_as_a_bearer_token(self, patched_httpx):
        headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            headers.update(request.headers)
            return chat_response("ok")

        patched_httpx(handler)
        OpenAICompatibleProvider(
            provider="together",
            base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key="secret-key",
        ).complete("hi")

        assert headers["authorization"] == "Bearer secret-key"

    def test_no_authorization_header_when_running_locally(self, patched_httpx):
        headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            headers.update(request.headers)
            return chat_response("ok")

        patched_httpx(handler)
        OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        ).complete("hi")

        assert "authorization" not in headers

    def test_a_truncated_answer_is_flagged(self, patched_httpx):
        """`finish_reason: length` means the answer is cut off. Silently
        returning it would look like the model simply stopped early."""
        patched_httpx(lambda _r: chat_response("half an ans", finish_reason="length"))

        completion = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        ).complete("hi")

        assert completion.metadata["truncated"] is True
        assert completion.stop_reason == "length"

    def test_reasoning_models_that_leave_content_empty_still_return_text(self, patched_httpx):
        """Some reasoning models put their answer in `reasoning_content`."""
        patched_httpx(
            lambda _r: httpx.Response(
                200,
                json={
                    "model": "deepseek-reasoner",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": "", "reasoning_content": "the answer"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        completion = OpenAICompatibleProvider(
            provider="deepseek", base_url="https://api.deepseek.com/v1", model="deepseek-reasoner"
        ).complete("hi")

        assert completion.text == "the answer"

    def test_a_server_without_usage_still_works(self, patched_httpx):
        """Several local servers omit `usage` entirely."""
        patched_httpx(
            lambda _r: httpx.Response(
                200,
                json={
                    "model": "local",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                },
            )
        )

        completion = OpenAICompatibleProvider(
            provider="llamacpp", base_url="http://localhost:8080/v1", model="local"
        ).complete("hi")

        assert completion.text == "ok" and completion.input_tokens == 0


class TestOpenAICompatibleErrors:
    def test_a_server_that_is_not_running_says_so(self, patched_httpx):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        patched_httpx(handler)
        with pytest.raises(LLMProxyError, match="is the server running"):
            OpenAICompatibleProvider(
                provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
            ).complete("hi")

    def test_a_missing_model_points_at_the_setting(self, patched_httpx):
        patched_httpx(
            lambda _r: httpx.Response(404, json={"error": {"message": "model not found"}})
        )
        with pytest.raises(LLMProxyError, match="CWAP_LLM_MODEL"):
            OpenAICompatibleProvider(
                provider="ollama", base_url="http://localhost:11434/v1", model="missing-model"
            ).complete("hi")

    def test_a_rejected_key_points_at_the_setting(self, patched_httpx):
        patched_httpx(lambda _r: httpx.Response(401, json={"error": {"message": "bad key"}}))
        with pytest.raises(LLMProxyError, match="CWAP_LLM_API_KEY"):
            OpenAICompatibleProvider(
                provider="groq",
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
                api_key="nope",
            ).complete("hi")

    def test_a_timeout_suggests_the_fix(self, patched_httpx):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        patched_httpx(handler)
        with pytest.raises(LLMProxyError, match="CWAP_LLM_TIMEOUT"):
            OpenAICompatibleProvider(
                provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
            ).complete("hi")

    def test_a_garbled_response_is_reported_with_its_body(self, patched_httpx):
        patched_httpx(lambda _r: httpx.Response(200, text="not json at all"))
        with pytest.raises(LLMProxyError, match="unexpected response shape"):
            OpenAICompatibleProvider(
                provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
            ).complete("hi")


class TestCapabilityHonesty:
    """A knob a backend cannot honour must be reported, never silently dropped."""

    def test_open_models_take_temperature_and_ignore_effort(self, patched_httpx):
        patched_httpx(lambda _r: chat_response("ok"))
        completion = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        ).complete("hi", options=GenerationOptions(temperature=0.4, effort="xhigh"))

        assert completion.ignored_options == ("effort",)

    def test_claude_takes_effort_and_ignores_temperature(self):
        """Current Claude models reject sampling parameters with a 400, so the
        provider must not send one even when a node carries it."""
        from llm_proxy.client import AnthropicProvider, _filter_options

        capabilities = AnthropicProvider.__new__(AnthropicProvider)
        capabilities.capabilities = _anthropic_capabilities()
        accepted, ignored = _filter_options(
            GenerationOptions(temperature=0.7, effort="high"), capabilities.capabilities
        )

        assert "temperature" not in accepted
        assert ignored == ("temperature",)
        assert accepted["effort"] == "high"

    def test_a_workflow_stays_portable_across_backends(self, patched_httpx):
        """The same node params run on either backend — each honours what it
        can. That is what makes a saved workflow model-agnostic."""
        patched_httpx(lambda _r: chat_response("ok"))
        options = GenerationOptions(temperature=0.4, effort="high", max_tokens=256)

        open_model = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        ).complete("hi", options=options)
        stub = StubProvider().complete("hi", options=options)

        assert open_model.text and stub.text  # neither errors on the other's knob


def _anthropic_capabilities():
    from llm_proxy.client import ProviderCapabilities

    return ProviderCapabilities(
        provider="anthropic",
        label="Anthropic Claude",
        model="claude-opus-5",
        supports_effort=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_stop_sequences=True,
        supports_system_prompt=True,
    )


class TestDescribeProvider:
    def test_a_working_provider_reports_its_capabilities(self):
        info = describe_provider()
        assert info["configured"] is True
        assert info["supports"]["temperature"] is True

    def test_a_misconfigured_provider_reports_data_not_a_crash(self, monkeypatch):
        """A bad setting should show an actionable message in the UI, not a 500
        on an unrelated page."""
        from llm_proxy.client import reset_provider_cache

        monkeypatch.setenv("CWAP_LLM_PROVIDER", "openai")
        monkeypatch.delenv("CWAP_LLM_API_KEY", raising=False)
        reset_settings_cache()
        reset_provider_cache(None)

        info = describe_provider()
        assert info["configured"] is False
        assert "CWAP_LLM_API_KEY" in info["error"]


class TestRemoteEmbeddings:
    def test_a_batch_is_one_request(self, patched_httpx):
        """One HTTP round trip per chunk would make a modest upload take
        minutes against a local embedding server."""
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": [1.0, 0.0, 0.0]}
                        for i in range(len(body["input"]))
                    ]
                },
            )

        patched_httpx(handler)
        embedder = RemoteEmbedder(
            base_url="http://localhost:11434/v1", model="nomic-embed-text", provider="ollama"
        )
        vectors = embedder.embed_batch(["one", "two", "three"])

        assert len(calls) == 1 and len(vectors) == 3
        assert embedder.dimensions == 3

    def test_out_of_order_results_are_realigned(self, patched_httpx):
        """Some servers return embeddings out of order; `index` is authoritative."""
        patched_httpx(
            lambda _r: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        )
        vectors = RemoteEmbedder(
            base_url="http://localhost:11434/v1", model="nomic-embed-text"
        ).embed_batch(["first", "second"])

        assert vectors[0] == [1.0, 0.0]

    def test_a_missing_base_url_is_a_configuration_error(self):
        with pytest.raises(EmbeddingError, match="CWAP_EMBEDDING_BASE_URL"):
            RemoteEmbedder(base_url="", model="nomic-embed-text")

    def test_a_missing_model_is_a_configuration_error(self):
        with pytest.raises(EmbeddingError, match="CWAP_EMBEDDING_MODEL"):
            RemoteEmbedder(base_url="http://localhost:11434/v1", model="")

    def test_an_unreachable_server_says_so(self, patched_httpx):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        patched_httpx(handler)
        with pytest.raises(EmbeddingError, match="could not reach"):
            RemoteEmbedder(
                base_url="http://localhost:11434/v1", model="nomic-embed-text"
            ).embed_batch(["x"])

    def test_the_preset_supplies_a_default_embedding_model(self, monkeypatch):
        monkeypatch.setenv("CWAP_EMBEDDING_PROVIDER", "ollama")
        monkeypatch.delenv("CWAP_EMBEDDING_MODEL", raising=False)
        reset_settings_cache()

        embedder = build_embedder()
        assert embedder.identity == "ollama:nomic-embed-text"

    def test_hashing_stays_the_offline_default(self):
        assert build_embedder().identity.startswith("hashing:")
        assert describe_embedder()["semantic"] is False


class TestEmbeddingMigrationSafety:
    def test_switching_embedding_model_refuses_to_compare_old_vectors(self):
        """Cosine distance between two different embedding spaces looks
        plausible and means nothing. Refuse, and say what to do."""
        from cwap_contracts import IngestRequest, RetrievalRequest
        from knowledge.service import KnowledgeError, ingest, retrieve

        set_embedder(HashingEmbedder())
        handle = ingest(
            IngestRequest(tenant_id="tenant-a", title="Handbook", content="Rail travel is preferred.")
        )

        # Someone changes CWAP_EMBEDDING_MODEL and restarts.
        set_embedder(_FakeEmbedder("ollama:nomic-embed-text"))

        with pytest.raises(KnowledgeError, match="Re-upload"):
            retrieve(
                RetrievalRequest(
                    handle=handle.handle, tenant_id="tenant-a", query="travel", top_k=2
                )
            )


class _FakeEmbedder:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.dimensions = 4

    def embed(self, text: str) -> list[float]:  # pragma: no cover - never reached
        return [0.5, 0.5, 0.5, 0.5]


class TestRuntimeEndpoint:
    def test_the_canvas_can_discover_what_the_backend_supports(self):
        from api_gateway.app import create_app
        from fastapi.testclient import TestClient

        with TestClient(create_app()) as client:
            token = client.post(
                "/api/auth/register",
                json={
                    "email": "runtime@example.com",
                    "password": "a-sufficiently-long-password",
                    "tenant_id": "tenant-a",
                },
            ).json()["access_token"]
            body = client.get(
                "/api/runtime", headers={"Authorization": f"Bearer {token}"}
            ).json()

        assert body["llm"]["provider"] == "stub"
        assert "temperature" in body["llm"]["supports"]
        assert any(item["key"] == "ollama" for item in body["available_providers"])
        assert body["embeddings"]["configured"] is True

    def test_runtime_information_requires_authentication(self):
        from api_gateway.app import create_app
        from fastapi.testclient import TestClient

        with TestClient(create_app()) as client:
            assert client.get("/api/runtime").status_code == 401


def test_settings_reset(monkeypatch):
    """Guard the fixture itself: provider settings must not leak between tests."""
    reset_settings_cache()
    assert get_settings().llm_provider == "stub"
