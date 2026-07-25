"""Tool calling across backends, and the fallback for models that have none.

"Works with any model" is the platform's central claim, and agents are where it
is hardest to keep. Tool calling is the one capability an agent loop genuinely
needs, and plenty of capable open-weight models never learned it. So there are
two paths — native, and a prompted JSON protocol — and the interesting behaviour
is the transition between them: it must happen automatically, at runtime, without
the workflow author knowing which path they are on.

As in `test_providers.py`, these run against `httpx.MockTransport`, so the real
request shaping and the real error mapping are exercised without a network.
"""

from __future__ import annotations

import json

import httpx
import pytest
from llm_proxy.catalogue import CATALOGUE, capabilities_for, cards_for, find
from llm_proxy.client import (
    ChatMessage,
    OpenAICompatibleProvider,
    StubProvider,
    ToolCallRequest,
    _to_anthropic_messages,
    _to_openai_messages,
)
from llm_proxy.presets import PRESETS

SUMMARISE_TOOL = {
    "name": "summarise",
    "description": "Condense text into its key points.",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text."}},
        "required": ["text"],
    },
}


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


def provider(model: str = "llama3.1") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider="ollama", base_url="http://localhost:11434/v1", model=model
    )


def tool_call_response(name: str, arguments: dict):
    return httpx.Response(
        200,
        json={
            "model": "llama3.1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        },
    )


def text_response(content: str):
    return httpx.Response(
        200,
        json={
            "model": "llama3.1",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        },
    )


class TestNativeToolCalling:
    def test_tools_are_sent_in_the_openai_function_shape(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return text_response("done")

        patched_httpx(handler)
        provider().converse([ChatMessage(role="user", content="hi")], tools=[SUMMARISE_TOOL])

        [tool] = seen["tools"]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "summarise"
        assert tool["function"]["parameters"]["required"] == ["text"]
        assert seen["tool_choice"] == "auto"

    def test_a_requested_call_is_returned_provider_neutrally(self, patched_httpx):
        patched_httpx(lambda request: tool_call_response("summarise", {"text": "a doc"}))

        completion = provider().converse(
            [ChatMessage(role="user", content="summarise this")], tools=[SUMMARISE_TOOL]
        )

        [call] = completion.tool_calls
        assert (call.name, call.arguments) == ("summarise", {"text": "a doc"})
        assert completion.metadata["tool_mode"] == "native"

    def test_no_tools_requested_means_no_tools_key(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return text_response("done")

        patched_httpx(handler)
        provider().converse([ChatMessage(role="user", content="hi")])
        assert "tools" not in seen

    def test_malformed_arguments_do_not_crash_the_turn(self, patched_httpx):
        """Small models sometimes emit arguments that are not valid JSON. The raw
        text is handed on as a single argument, so the skill reports a missing
        argument back to the agent and the loop recovers. A parse crash does not
        recover."""
        patched_httpx(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {
                                            "name": "summarise",
                                            "arguments": "{not json",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        )
        completion = provider().converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )
        assert completion.tool_calls[0].arguments == {"input": "{not json"}


class TestAutomaticDowngrade:
    def test_a_server_that_rejects_tools_is_retried_on_the_prompted_protocol(
        self, patched_httpx
    ):
        """The transition that makes "any model" true. Tool support is not
        discoverable up front, so the only honest way to find out is to try."""
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append("tools" in payload)
            if "tools" in payload:
                return httpx.Response(
                    400, json={"error": {"message": "this model does not support tools"}}
                )
            return text_response("Denver is in Colorado.")

        patched_httpx(handler)
        client = provider()

        completion = client.converse(
            [ChatMessage(role="user", content="Where is Denver?")], tools=[SUMMARISE_TOOL]
        )

        assert attempts == [True, False]
        assert completion.text == "Denver is in Colorado."
        assert completion.metadata["tool_mode"] == "prompted"

    def test_the_downgrade_is_permanent_for_that_provider(self, patched_httpx):
        """Otherwise every single turn pays for a failing request first."""
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append("tools" in payload)
            if "tools" in payload:
                return httpx.Response(400, json={"error": {"message": "tools are unsupported"}})
            return text_response("ok")

        patched_httpx(handler)
        client = provider()

        client.converse([ChatMessage(role="user", content="one")], tools=[SUMMARISE_TOOL])
        client.converse([ChatMessage(role="user", content="two")], tools=[SUMMARISE_TOOL])

        assert attempts == [True, False, False]

    def test_the_downgrade_is_reported_in_capabilities(self, patched_httpx):
        """So the canvas stops claiming native tool use after it turned out to be
        untrue."""

        def handler(request):
            if "tools" in json.loads(request.content):
                return httpx.Response(400, json={"error": {"message": "tools not supported"}})
            return text_response("ok")

        patched_httpx(handler)
        client = provider()
        assert client.capabilities.supports_tools is True

        client.converse([ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL])
        assert client.capabilities.supports_tools is False

    def test_an_unrelated_error_is_not_mistaken_for_a_tool_rejection(self, patched_httpx):
        """A bad key must surface as a bad key, not silently degrade the loop."""
        from llm_proxy.client import LLMProxyError

        patched_httpx(
            lambda request: httpx.Response(401, json={"error": {"message": "invalid api key"}})
        )
        with pytest.raises(LLMProxyError, match="key"):
            provider().converse(
                [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
            )

    @pytest.mark.parametrize(
        "message",
        [
            "this model does not support tools",
            "tools are unsupported by this backend",
            "unknown field: tools",
            "Extra inputs are not permitted: tools",
            "function calling is not implemented",
            "tool_choice is not supported",
        ],
    )
    def test_the_many_ways_a_server_says_it_has_no_tools(self, patched_httpx, message):
        """Every backend phrases this differently, and a phrasing that is missed
        reports a perfectly capable open model as broken."""
        seen = []

        def handler(request):
            has_tools = "tools" in json.loads(request.content)
            seen.append(has_tools)
            if has_tools:
                return httpx.Response(400, json={"error": {"message": message}})
            return text_response("recovered")

        patched_httpx(handler)
        completion = provider().converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )
        assert completion.text == "recovered"

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            (401, "invalid api key"),
            (404, "model 'llama3.1' not found, try pulling it first"),
            (429, "rate limit exceeded"),
            (400, "messages: content is required"),
            (500, "internal server error"),
        ],
    )
    def test_unrelated_errors_are_never_mistaken_for_a_tool_rejection(
        self, patched_httpx, status, message
    ):
        """The cost of matching too broadly: a bad key would silently become a
        permanent downgrade, and the real problem would never be reported."""
        from llm_proxy.client import LLMProxyError

        patched_httpx(
            lambda request: httpx.Response(status, json={"error": {"message": message}})
        )
        client = provider()
        with pytest.raises(LLMProxyError):
            client.converse([ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL])
        assert client.capabilities.supports_tools is True

    def test_an_operator_can_force_the_prompted_protocol(self, patched_httpx, monkeypatch):
        from cwap_common.settings import reset_settings_cache

        monkeypatch.setenv("CWAP_LLM_TOOL_MODE", "prompted")
        reset_settings_cache()

        attempts = []

        def handler(request):
            attempts.append("tools" in json.loads(request.content))
            return text_response("ok")

        patched_httpx(handler)
        provider().converse([ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL])

        assert attempts == [False]


class TestPromptedProtocol:
    def forced(self, monkeypatch) -> OpenAICompatibleProvider:
        from cwap_common.settings import reset_settings_cache

        monkeypatch.setenv("CWAP_LLM_TOOL_MODE", "prompted")
        reset_settings_cache()
        return provider()

    def test_the_tools_are_described_in_the_system_prompt(self, patched_httpx, monkeypatch):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return text_response("ok")

        patched_httpx(handler)
        self.forced(monkeypatch).converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )

        system = seen["messages"][0]["content"]
        assert "summarise(text (string))" in system
        assert "JSON object" in system

    def test_a_json_reply_becomes_a_tool_call(self, patched_httpx, monkeypatch):
        patched_httpx(
            lambda request: text_response('{"tool": "summarise", "arguments": {"text": "a doc"}}')
        )
        completion = self.forced(monkeypatch).converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )

        [call] = completion.tool_calls
        assert (call.name, call.arguments) == ("summarise", {"text": "a doc"})
        assert completion.text == ""

    def test_a_fenced_json_reply_still_becomes_a_tool_call(self, patched_httpx, monkeypatch):
        patched_httpx(
            lambda request: text_response(
                'Sure:\n```json\n{"tool": "summarise", "arguments": {"text": "a doc"}}\n```'
            )
        )
        completion = self.forced(monkeypatch).converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )
        assert completion.tool_calls[0].name == "summarise"

    def test_prose_is_a_final_answer_not_a_failure(self, patched_httpx, monkeypatch):
        """A model that ignores the protocol degrades to a plain response."""
        patched_httpx(lambda request: text_response("Denver is in Colorado."))
        completion = self.forced(monkeypatch).converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )
        assert completion.tool_calls == ()
        assert completion.text == "Denver is in Colorado."

    def test_an_invented_tool_name_is_not_dispatched(self, patched_httpx, monkeypatch):
        """A hallucinated name in JSON must not become a call to something that
        does not exist; it is treated as prose instead."""
        patched_httpx(
            lambda request: text_response('{"tool": "teleport", "arguments": {"to": "Denver"}}')
        )
        completion = self.forced(monkeypatch).converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )
        assert completion.tool_calls == ()

    def test_a_json_object_that_is_not_a_call_is_prose(self, patched_httpx, monkeypatch):
        patched_httpx(lambda request: text_response('{"answer": "Colorado"}'))
        completion = self.forced(monkeypatch).converse(
            [ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL]
        )
        assert completion.tool_calls == ()
        assert "Colorado" in completion.text


class TestMessageTranslation:
    def conversation(self):
        return [
            ChatMessage(role="user", content="Summarise the report"),
            ChatMessage(
                role="assistant",
                content="Let me read it.",
                tool_calls=(ToolCallRequest(id="c1", name="summarise", arguments={"text": "x"}),),
            ),
            ChatMessage(role="tool", content="the summary", tool_call_id="c1", name="summarise"),
        ]

    def test_openai_carries_tool_results_as_their_own_role(self):
        translated = _to_openai_messages(self.conversation(), "You are a researcher.")

        assert translated[0] == {"role": "system", "content": "You are a researcher."}
        assert translated[2]["tool_calls"][0]["function"]["name"] == "summarise"
        assert translated[3] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "the summary",
        }

    def test_anthropic_carries_tool_results_inside_a_user_turn(self):
        translated = _to_anthropic_messages(self.conversation())

        assistant = translated[1]
        assert [block["type"] for block in assistant["content"]] == ["text", "tool_use"]

        result = translated[2]
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "c1"

    def test_consecutive_tool_results_merge_into_one_anthropic_turn(self):
        """The API rejects two user turns in a row, which is what a parallel tool
        call produces if each result becomes its own message."""
        messages = [
            ChatMessage(role="user", content="Read both"),
            ChatMessage(
                role="assistant",
                tool_calls=(
                    ToolCallRequest(id="c1", name="summarise"),
                    ToolCallRequest(id="c2", name="summarise"),
                ),
            ),
            ChatMessage(role="tool", content="first", tool_call_id="c1"),
            ChatMessage(role="tool", content="second", tool_call_id="c2"),
        ]
        translated = _to_anthropic_messages(messages)

        assert len(translated) == 3
        assert len(translated[2]["content"]) == 2

    def test_a_system_prompt_is_not_a_message_for_anthropic(self):
        """Anthropic takes `system` as a top-level parameter, so it must not be
        smuggled in as a message the way OpenAI expects."""
        translated = _to_anthropic_messages(self.conversation())
        assert all(message["role"] != "system" for message in translated)


class TestModelCatalogue:
    def test_a_catalogued_model_reports_its_real_capabilities(self):
        """Vision is curated here. Tool calling is not asserted away: this entry
        used to claim `tools=False`, which was true of the runtime in 2024 and
        stopped being true, and nothing would ever have corrected it."""
        tools, vision, thinking = capabilities_for("ollama", "llama3.2-vision")
        assert (vision, thinking) == (True, False)
        assert tools is True

    def test_a_reasoning_model_is_marked_as_thinking(self):
        _tools, _vision, thinking = capabilities_for("deepseek", "deepseek-reasoner")
        assert thinking is True

    def test_the_same_weights_under_a_different_provider_resolve(self):
        """The same Llama on Groq and on Together has the same capabilities."""
        assert find("groq", "meta-llama/Llama-3.3-70B-Instruct-Turbo") is not None

    def test_an_unlisted_vision_model_is_inferred_from_its_name(self):
        _tools, vision, _thinking = capabilities_for("ollama", "some-new-vision-7b")
        assert vision is True

    def test_an_unlisted_model_is_assumed_to_have_tools(self):
        """Guessing "no tools" would silently put every unlisted model on the
        slower prompted path; the runtime downgrades cleanly if the guess is
        wrong, which the reverse mistake does not."""
        tools, _vision, _thinking = capabilities_for("vllm", "internal-finetune-v3")
        assert tools is True

    def test_the_catalogue_covers_every_preset(self):
        """So the picker never shows a provider with no guidance at all — even if
        the guidance is 'you supply the model id'."""
        assert set(CATALOGUE) >= set(PRESETS)

    def test_every_catalogued_model_is_displayable(self):
        for provider_key in CATALOGUE:
            for card in cards_for(provider_key):
                rendered = card.as_dict()
                assert rendered["id"] and rendered["label"]
                assert set(rendered["supports"]) == {"tools", "vision", "thinking"}

    def test_the_configured_model_shapes_reported_capabilities(self):
        client = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.2-vision"
        )
        assert client.capabilities.supports_vision is True
        assert client.capabilities.tool_support == "catalogue"

    def test_the_stub_does_not_claim_tool_support(self):
        """It never asks for a tool, so an agent loop against it always ends on
        the first iteration. Claiming otherwise would promise agentic behaviour
        the default backend cannot deliver."""
        assert StubProvider().capabilities.supports_tools is False


class TestToolSupportIsNeverGuessedAway:
    """Reported from a real deployment: the canvas told a user their LM Studio
    model had no native tool calling, on the strength of a substring in its name.

    The asymmetry is the whole point. Assuming a model *has* tools and being
    wrong costs one rejected request, which the provider recovers from by
    downgrading permanently and retrying inside the same call. Assuming it does
    *not* routes a perfectly capable model onto the slower prompted protocol
    forever, and no later signal would ever correct it.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "gemma-3-12b-it",
            "phi-4",
            "llava-v1.6-mistral-7b",
            "qwen3-8b",
            "some-model-r1-variant",
            "internal-finetune-v3",
            "granite-3.3-8b-instruct",
        ],
    )
    def test_an_unlisted_model_is_assumed_capable(self, model):
        """Every one of these used to be guessed out of tool calling by a
        substring match."""
        tools, _vision, _thinking = capabilities_for("lmstudio", model)
        assert tools is True

    def test_the_guess_is_reported_as_a_guess(self):
        client = OpenAICompatibleProvider(
            provider="lmstudio", base_url="http://localhost:1234/v1", model="gemma-3-12b-it"
        )
        assert client.capabilities.supports_tools is True
        assert client.capabilities.tool_support == "assumed"

    def test_a_catalogued_model_is_reported_as_curated(self):
        client = OpenAICompatibleProvider(
            provider="ollama", base_url="http://localhost:11434/v1", model="llama3.1"
        )
        assert client.capabilities.tool_support == "catalogue"

    def test_vision_and_thinking_may_still_be_guessed(self):
        """Being wrong there changes which controls the canvas offers, not how a
        request is made — so the same caution does not apply."""
        _tools, vision, _thinking = capabilities_for("lmstudio", "some-vision-model")
        assert vision is True
        _tools, _vision, thinking = capabilities_for("lmstudio", "qwen3-8b-thinking")
        assert thinking is True

    def test_only_a_documented_limitation_may_assert_the_negative(self):
        """The bar an entry has to clear before the catalogue says "no tools":
        the provider documents it, not that the weights are rumoured to lack it."""
        from llm_proxy.catalogue import CATALOGUE

        asserted = [
            (provider, card.id)
            for provider, cards in CATALOGUE.items()
            for card in cards
            if not card.tools
        ]
        assert asserted == [("deepseek", "deepseek-reasoner"), ("stub", "stub-model")], (
            "a new tools=False entry needs a documented provider limitation "
            f"behind it, not a guess: {asserted}"
        )

    def test_a_real_rejection_outranks_the_assumption(self, patched_httpx):
        """And is recorded as observed, so the canvas can say the server told us
        rather than implying the platform knew in advance."""

        def handler(request):
            if "tools" in json.loads(request.content):
                return httpx.Response(
                    400, json={"error": {"message": "this model does not support tools"}}
                )
            return text_response("fine")

        patched_httpx(handler)
        client = provider("gemma-3-12b-it")
        assert client.capabilities.tool_support == "assumed"

        client.converse([ChatMessage(role="user", content="x")], tools=[SUMMARISE_TOOL])

        assert client.capabilities.supports_tools is False
        assert client.capabilities.tool_support == "observed"

    def test_forcing_the_prompted_protocol_is_reported_as_configured(self, monkeypatch):
        """An operator's decision is not the model's limitation, and the message
        a user reads should not confuse the two."""
        from cwap_common.settings import reset_settings_cache

        monkeypatch.setenv("CWAP_LLM_TOOL_MODE", "prompted")
        reset_settings_cache()

        client = provider("llama3.1")
        assert client.capabilities.supports_tools is False
        assert client.capabilities.tool_support == "configured"

    def test_the_runtime_endpoint_reports_the_provenance(self, client, auth):
        """The canvas needs it to decide whether it may state a limitation."""
        body = client.get("/api/runtime", headers=auth).json()
        assert "tool_support" in body["llm"]
