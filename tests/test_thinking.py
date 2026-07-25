"""Reasoning models: engaged, recovered from, and kept.

Detecting that a model can think and then never asking it to is worse than not
detecting it — the canvas says "thinking" while every request is a plain one.
Three things have to hold, and each is a separate failure the platform has
already made in one form or another:

* the request carries the parameter *that server* understands, since every
  backend invented its own and sending the wrong one is a 400;
* a server that does not take it costs one retried call, not a failed run —
  the same recovery shape as a rejected tools request;
* the reasoning that comes back is kept, because on a thinking model it is
  usually where the actual work is.

As in `test_tool_calling.py`, these run against `httpx.MockTransport`: real
request shaping, real error mapping, no network.
"""

from __future__ import annotations

import json

import httpx
import pytest
from agents import runtime as agent_runtime
from llm_proxy import reasoning
from llm_proxy.catalogue import capabilities_for
from llm_proxy.client import ChatMessage, OpenAICompatibleProvider


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


def thinker(provider: str = "ollama", model: str = "deepseek-r1") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider=provider, base_url="http://localhost:11434/v1", model=model
    )


def answer(content: str = "done", **message):
    return httpx.Response(
        200,
        json={
            "model": "deepseek-r1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content, **message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        },
    )


class TestTheModelIsAskedToThink:
    def test_a_thinking_model_is_actually_asked(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        thinker().complete("hello")

        assert seen["think"] is True

    def test_each_backend_gets_its_own_dialect(self):
        """The whole reason this is a table: no two servers agree."""
        assert reasoning.request_fields("ollama", "high") == {"think": True}
        assert reasoning.request_fields("lmstudio", "high") == {"reasoning_effort": "high"}
        assert reasoning.request_fields("openrouter", "low") == {"reasoning": {"effort": "low"}}
        assert reasoning.request_fields("vllm", "high") == {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    def test_effort_above_what_openai_accepts_is_mapped_down_not_sent_raw(self):
        """"max" is this platform's vocabulary; sending it verbatim is a 400."""
        assert reasoning.request_fields("openai", "max") == {"reasoning_effort": "high"}

    def test_an_unknown_backend_is_sent_nothing(self):
        assert reasoning.request_fields("something-new", "high") == {}
        assert not reasoning.knows("something-new")

    def test_a_model_that_always_reasons_is_never_asked_to(self):
        """DeepSeek's reasoner has no mode to switch on — asking would 400."""
        assert reasoning.request_fields("deepseek", "high") == {}
        assert not reasoning.knows("deepseek")

    def test_a_model_with_no_thinking_mode_is_left_alone(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        thinker(model="llama3.1").complete("hello")

        assert not any(key in seen for key in reasoning.PARAMS)

    def test_an_operator_can_turn_it_off_entirely(self, patched_httpx, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_THINKING_MODE", "off")
        from cwap_common.settings import reset_settings_cache

        reset_settings_cache()
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        try:
            thinker().complete("hello")
        finally:
            monkeypatch.delenv("CWAP_LLM_THINKING_MODE")
            reset_settings_cache()

        assert "think" not in seen

    def test_thinking_gets_room_to_think(self, patched_httpx, monkeypatch):
        """Reasoning is spent from the answer's budget, so a 4k ceiling truncates."""
        monkeypatch.setenv("CWAP_LLM_MAX_TOKENS", "512")
        from cwap_common.settings import reset_settings_cache

        reset_settings_cache()
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        try:
            thinker().complete("hello")
            budget = seen["max_tokens"]
        finally:
            monkeypatch.delenv("CWAP_LLM_MAX_TOKENS")
            reset_settings_cache()

        assert budget == reasoning.MIN_THINKING_TOKENS

    def test_a_thinking_backend_reports_effort_as_honoured(self):
        """The canvas may only show a depth control that reaches something."""
        assert thinker().capabilities.supports_effort is True
        assert thinker(model="llama3.1").capabilities.supports_effort is False


class TestARejectionCostsOneCallNotTheRun:
    def test_the_parameter_is_dropped_and_the_call_retried(self, patched_httpx):
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append(payload)
            if "think" in payload:
                return httpx.Response(
                    400, json={"error": {"message": "unknown parameter: think"}}
                )
            return answer("second time lucky")

        patched_httpx(handler)
        result = thinker().complete("hello")

        assert result.text == "second time lucky"
        assert len(attempts) == 2
        assert "think" not in attempts[1]

    def test_it_stops_asking_for_the_rest_of_the_process(self, patched_httpx):
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append(payload)
            if "think" in payload:
                return httpx.Response(400, json={"error": "think is not supported"})
            return answer()

        patched_httpx(handler)
        provider = thinker()
        provider.complete("first")
        provider.complete("second")

        # Rejected once, then never sent again — not re-learned per call.
        assert sum("think" in payload for payload in attempts) == 1
        assert provider.capabilities.supports_effort is False

    def test_a_rejection_does_not_claim_the_model_cannot_think(self, patched_httpx):
        """The server refused the *parameter*. The model may still reason."""

        def handler(request):
            payload = json.loads(request.content)
            if "think" in payload:
                return httpx.Response(400, json={"error": "unknown field think"})
            return answer()

        patched_httpx(handler)
        provider = thinker()
        provider.complete("hello")

        assert provider.capabilities.supports_thinking is True

    def test_an_unrelated_400_is_still_an_error(self, patched_httpx):
        """A rejected key must not be read as "does not take reasoning"."""

        def handler(request):
            return httpx.Response(401, json={"error": {"message": "invalid api key"}})

        patched_httpx(handler)
        with pytest.raises(Exception, match="rejected the credentials"):
            thinker().complete("hello")

    def test_the_matcher_needs_a_subject_and_a_negation(self):
        assert reasoning.looks_rejected("unknown parameter: reasoning_effort")
        assert reasoning.looks_rejected("this model does not support thinking")
        assert not reasoning.looks_rejected("rate limit exceeded")
        # About reasoning, but not a refusal of the parameter.
        assert not reasoning.looks_rejected("reasoning took too long")
        # A refusal, but of something else entirely.
        assert not reasoning.looks_rejected("unknown parameter: seed")


class TestTheReasoningIsKept:
    def test_reasoning_returned_beside_the_answer_is_not_discarded(self, patched_httpx):
        patched_httpx(lambda request: answer("42", reasoning_content="I counted twice."))
        result = thinker().complete("how many?")

        assert result.text == "42"
        assert result.reasoning == "I counted twice."

    def test_it_is_read_wherever_the_server_put_it(self, patched_httpx):
        patched_httpx(lambda request: answer("42", thinking="I counted twice."))
        assert thinker().complete("how many?").reasoning == "I counted twice."

    def test_structured_reasoning_blocks_are_flattened(self, patched_httpx):
        patched_httpx(
            lambda request: answer(
                "42", reasoning=[{"type": "text", "text": "first"}, {"text": "second"}]
            )
        )
        assert thinker().complete("how many?").reasoning == "first\nsecond"

    def test_an_empty_answer_falls_back_to_the_reasoning(self, patched_httpx):
        """A model that spent its budget thinking said *something*; show it."""
        patched_httpx(lambda request: answer("", reasoning_content="Halfway through…"))
        assert thinker().complete("how many?").text == "Halfway through…"

    def test_reasoning_is_carried_through_the_conversation_path_too(self, patched_httpx):
        patched_httpx(lambda request: answer("42", reasoning_content="Because."))
        result = thinker().converse([ChatMessage(role="user", content="how many?")])

        assert result.reasoning == "Because."

    def test_a_drafted_tool_call_inside_reasoning_is_never_invoked(self, patched_httpx):
        """A thinking model proposes and discards calls; only the answer counts."""
        patched_httpx(
            lambda request: answer(
                "I will just answer.",
                reasoning_content='Maybe {"tool": "summarise", "arguments": {"text": "x"}}?',
            )
        )
        provider = thinker()
        provider._tool_mode = "prompted"
        result = provider.converse(
            [ChatMessage(role="user", content="hi")],
            tools=[
                {
                    "name": "summarise",
                    "description": "Condense text.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

        assert result.tool_calls == ()
        assert result.text == "I will just answer."


class TestTheRunReportShowsTheThinking:
    def test_an_agent_records_what_the_model_worked_through(self, patched_httpx, monkeypatch):
        from cwap_contracts.v4 import AgentDefinition

        patched_httpx(lambda request: answer("The answer.", reasoning_content="Weighed both."))
        monkeypatch.setattr(agent_runtime, "get_provider", thinker)

        events = []
        agent = AgentDefinition(
            id="ag_thinker",
            tenant_id="t_1",
            name="Analyst",
            role="You weigh options.",
            max_iterations=2,
        )
        result = agent_runtime.run_agent(
            agent,
            "Decide something",
            agent_runtime.AgentRunContext(
                tenant_id="t_1",
                emit=lambda event, message="", data=None: events.append((event, data or {})),
            ),
        )

        assert result.reasoning == [{"iteration": 1, "text": "Weighed both."}]
        reasoned = [data for event, data in events if event == "agent.reasoned"]
        assert reasoned and reasoned[0]["reasoning"] == "Weighed both."


class TestWhichModelsAreRecognised:
    @pytest.mark.parametrize(
        "model",
        ["qwen3:14b", "gpt-oss:20b", "deepseek-r1:7b", "magistral-small", "o4-mini"],
    )
    def test_a_locally_tagged_reasoning_model_is_recognised(self, model):
        _tools, _vision, thinking = capabilities_for("ollama", model)
        assert thinking is True

    def test_a_plain_model_is_not_mistaken_for_one(self):
        _tools, _vision, thinking = capabilities_for("ollama", "llama3.1:8b")
        assert thinking is False


class TestTheAgentIsToldWhatItCanActuallyDo:
    def _agent(self):
        from cwap_contracts.v4 import AgentDefinition

        return AgentDefinition(
            id="ag_1", tenant_id="t_1", name="Analyst", role="You analyse things."
        )

    def test_an_agent_with_tools_is_told_to_check_rather_than_recall(self):
        from cwap_contracts.v4 import SkillDefinition, SkillKind

        skill = SkillDefinition(
            id="sk_1",
            tenant_id="t_1",
            name="search_the_web",
            description="Look things up.",
            kind=SkillKind.HTTP,
            definition={"url": "https://example.com", "method": "GET"},
        )
        prompt = agent_runtime._build_system_prompt(self._agent(), "", skills=[skill])

        assert "search_the_web" in prompt
        assert "Check rather than recall" in prompt

    def test_an_agent_with_no_tools_is_not_told_to_use_them(self):
        """Telling a model to use tools it has none of invents calls or apologies."""
        prompt = agent_runtime._build_system_prompt(self._agent(), "", skills=[])

        assert "You have no tools" in prompt
        assert "Check rather than recall" not in prompt
