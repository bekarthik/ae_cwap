"""Reading an answer as it is produced.

Three things have to hold, and the least visible one matters most.

The visible one: text reaches the run panel while the model is writing it.

The one that decides whether runs survive: **the timeout becomes a silence
detector.** Blocking reads made the deadline a ceiling on how long an answer
could take, so a local model producing a token a second was indistinguishable
from a dead one and got killed at the limit. Streaming applies it between
fragments instead — a slow model runs as long as it keeps talking, a wedged one
is caught in seconds.

The one that is easy to get wrong: a native tool call arrives as a name in one
frame and its arguments a character at a time across a dozen more. Dispatching
before the last fragment lands calls a skill with half its arguments.

As in `test_tool_calling.py`, these run against `httpx.MockTransport`: real
request shaping, real SSE parsing, no network.
"""

from __future__ import annotations

import json

import httpx
import pytest
from llm_proxy import streaming
from llm_proxy.client import ChatMessage, OpenAICompatibleProvider, StubProvider
from llm_proxy.live import LiveText

SUMMARISE_TOOL = {
    "name": "summarise",
    "description": "Condense text into its key points.",
    "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
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


def sse(*chunks: dict, done: bool = True) -> httpx.Response:
    """A server-sent-event response, as every OpenAI-compatible server sends it."""
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream; charset=utf-8"}
    )


def delta(content: str = "", **extra) -> dict:
    payload = {"content": content} if content else {}
    payload.update(extra)
    return {"model": "llama3.1", "choices": [{"index": 0, "delta": payload}]}


class TestTheAnswerArrivesWhileItIsWritten:
    def test_the_request_asks_for_a_stream(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return sse(delta("hello"))

        patched_httpx(handler)
        provider().complete("hi")

        assert seen["stream"] is True
        assert seen["stream_options"] == {"include_usage": True}

    def test_fragments_reach_the_caller_as_they_land(self, patched_httpx):
        patched_httpx(lambda request: sse(delta("The "), delta("answer"), delta(".")))
        seen: list[tuple[str, str]] = []

        result = provider().complete("hi", on_delta=lambda kind, text: seen.append((kind, text)))

        assert seen == [("content", "The "), ("content", "answer"), ("content", ".")]
        assert result.text == "The answer."

    def test_a_thinking_model_streams_its_thinking_separately(self, patched_httpx):
        patched_httpx(
            lambda request: sse(
                delta(reasoning_content="Let me "),
                delta(reasoning_content="check."),
                delta("42"),
            )
        )
        seen: list[tuple[str, str]] = []

        result = provider().complete("how many?", on_delta=lambda k, t: seen.append((k, t)))

        assert [kind for kind, _ in seen] == ["reasoning", "reasoning", "content"]
        assert result.reasoning == "Let me check."
        assert result.text == "42"

    def test_the_assembled_answer_is_shaped_like_a_normal_one(self):
        """The point of the design: nothing above the transport knows."""
        body = streaming.assemble(
            [
                'data: {"model":"m","choices":[{"delta":{"content":"a"}}]}',
                'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"stop"}]}',
                'data: {"usage":{"prompt_tokens":3,"completion_tokens":4}}',
                "data: [DONE]",
            ]
        )

        assert body["choices"][0]["message"]["content"] == "ab"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 4
        assert body["model"] == "m"

    def test_token_counts_survive_the_stream(self, patched_httpx):
        patched_httpx(
            lambda request: sse(
                delta("hi"),
                {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
            )
        )
        result = provider().complete("hi")

        assert (result.input_tokens, result.output_tokens) == (11, 7)

    def test_keepalives_and_blank_lines_are_not_answers(self):
        body = streaming.assemble(
            ["", ": ping", 'data: {"choices":[{"delta":{"content":"x"}}]}', "", "data: [DONE]"]
        )

        assert body["choices"][0]["message"]["content"] == "x"

    def test_the_offline_default_streams_too(self):
        """So the delta path is exercised without a model configured."""
        seen: list[str] = []
        result = StubProvider("stub-model").complete(
            "hello there", on_delta=lambda kind, text: seen.append(text)
        )

        assert len(seen) > 1
        assert "".join(seen) == result.text


class TestToolCallsAreReassembledBeforeUse:
    def test_a_call_split_across_frames_is_put_back_together(self, patched_httpx):
        patched_httpx(
            lambda request: sse(
                delta(
                    tool_calls=[
                        {"index": 0, "id": "call_1", "function": {"name": "summarise"}}
                    ]
                ),
                delta(tool_calls=[{"index": 0, "function": {"arguments": '{"te'}}]),
                delta(tool_calls=[{"index": 0, "function": {"arguments": 'xt": "hi"}'}}]),
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            )
        )

        result = provider().converse(
            [ChatMessage(role="user", content="go")], tools=[SUMMARISE_TOOL]
        )

        [call] = result.tool_calls
        assert call.id == "call_1"
        assert call.name == "summarise"
        assert call.arguments == {"text": "hi"}

    def test_two_calls_in_one_turn_stay_separate(self):
        body = streaming.assemble(
            [
                'data: {"choices":[{"delta":{"tool_calls":['
                '{"index":0,"id":"a","function":{"name":"one","arguments":"{}"}},'
                '{"index":1,"id":"b","function":{"name":"two","arguments":"{"}}]}}]}',
                'data: {"choices":[{"delta":{"tool_calls":'
                '[{"index":1,"function":{"arguments":"}"}}]}}]}',
                "data: [DONE]",
            ]
        )

        calls = body["choices"][0]["message"]["tool_calls"]
        assert [call["function"]["name"] for call in calls] == ["one", "two"]
        assert calls[1]["function"]["arguments"] == "{}"

    def test_a_prompted_call_is_not_shown_while_it_is_being_written(self, patched_httpx):
        """`{"tool": "summ…` on screen would read as the agent saying it."""
        patched_httpx(
            lambda request: sse(
                delta(reasoning_content="thinking"),
                delta('{"tool": "summarise", '),
                delta('"arguments": {"text": "hi"}}'),
            )
        )
        seen: list[tuple[str, str]] = []
        instance = provider()
        instance._tool_mode = "prompted"

        result = instance.converse(
            [ChatMessage(role="user", content="go")],
            tools=[SUMMARISE_TOOL],
            on_delta=lambda kind, text: seen.append((kind, text)),
        )

        assert result.tool_calls[0].name == "summarise"
        assert [kind for kind, _ in seen] == ["reasoning"]


class TestFailurePartWayThrough:
    def test_an_error_inside_the_stream_is_not_a_finished_answer(self, patched_httpx):
        """A partial returned as complete is worse than a failure."""
        patched_httpx(
            lambda request: sse(
                delta("Here is half an ans"),
                {"error": {"message": "the model ran out of memory"}},
            )
        )

        with pytest.raises(Exception, match="ran out of memory"):
            provider().complete("hi")

    def test_a_silent_timeout_says_the_model_never_answered(self, patched_httpx, monkeypatch):
        def handler(request):
            raise httpx.ReadTimeout("nothing", request=request)

        patched_httpx(handler)
        with pytest.raises(Exception, match="did not respond within"):
            provider().complete("hi")

    def test_a_timeout_after_output_says_the_model_stalled(self, patched_httpx):
        """A different failure from never answering, and a different fix."""

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_stall(),
            )

        patched_httpx(handler)
        with pytest.raises(Exception, match="went quiet") as caught:
            provider().complete("hi")
        assert "characters" in str(caught.value)

    def test_an_empty_stream_is_a_failure_not_an_empty_answer(self):
        with pytest.raises(streaming.StreamInterrupted):
            streaming.assemble(iter(()))

    def test_a_malformed_frame_does_not_lose_a_good_answer(self):
        body = streaming.assemble(
            [
                'data: {"choices":[{"delta":{"content":"good"}}]}',
                "data: {not json at all",
                "data: [DONE]",
            ]
        )

        assert body["choices"][0]["message"]["content"] == "good"


def _stall():
    """One fragment, then silence — a model that dies mid-sentence."""
    yield b'data: {"choices":[{"delta":{"content":"half an answer"}}]}\n\n'
    raise httpx.ReadTimeout("stalled")


class TestAServerThatWillNotStream:
    def test_a_rejection_drops_streaming_and_retries(self, patched_httpx):
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append(payload)
            if payload.get("stream"):
                return httpx.Response(400, json={"error": "stream is not supported"})
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}],
                },
            )

        patched_httpx(handler)
        instance = provider()
        result = instance.complete("hi")

        assert result.text == "fine"
        assert instance._streaming is False
        # Learned once, not re-learned per call.
        instance.complete("again")
        assert sum(bool(attempt.get("stream")) for attempt in attempts) == 1

    def test_refusing_the_token_count_costs_only_the_token_count(self, patched_httpx):
        """The live answer is worth more than the usage numbers for one call."""
        attempts = []

        def handler(request):
            payload = json.loads(request.content)
            attempts.append(payload)
            if "stream_options" in payload:
                return httpx.Response(
                    400, json={"error": "unknown parameter: stream_options"}
                )
            return sse(delta("still streaming"))

        patched_httpx(handler)
        instance = provider()
        result = instance.complete("hi")

        assert result.text == "still streaming"
        assert instance._streaming is True
        assert instance._stream_usage is False
        assert attempts[-1]["stream"] is True

    def test_a_server_that_ignores_the_flag_is_not_an_error(self, patched_httpx):
        """Some proxies accept `stream` and answer in one piece anyway."""
        patched_httpx(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "llama3.1",
                    "choices": [
                        {"message": {"content": "all at once"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                },
            )
        )

        result = provider().complete("hi")
        assert result.text == "all at once"
        assert result.output_tokens == 3

    def test_a_stream_without_the_right_content_type_is_still_read(self, patched_httpx):
        """The other half of the same problem: streaming, mislabelled."""
        patched_httpx(
            lambda request: httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"sneaky"}}]}\n\ndata: [DONE]\n\n',
                headers={"content-type": "application/json"},
            )
        )

        assert provider().complete("hi").text == "sneaky"

    def test_an_operator_can_turn_streaming_off(self, patched_httpx, monkeypatch):
        monkeypatch.setenv("CWAP_LLM_STREAM", "off")
        from cwap_common.settings import reset_settings_cache

        reset_settings_cache()
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )

        patched_httpx(handler)
        try:
            provider().complete("hi")
        finally:
            monkeypatch.delenv("CWAP_LLM_STREAM")
            reset_settings_cache()

        assert seen["stream"] is False
        assert "stream_options" not in seen

    def test_the_matcher_does_not_fire_on_an_unrelated_error(self):
        assert streaming.rejection("rate limit exceeded") is None
        assert streaming.rejection("stream is not supported") == "stream"
        assert streaming.rejection("unknown parameter: stream_options") == "stream_options"
        # About a stream, but not a refusal of the parameter.
        assert streaming.rejection("the stream ended early") is None


class TestTheBrowserGetsReadablePieces:
    def test_fragments_are_coalesced_rather_than_forwarded_one_by_one(self):
        """A websocket frame per token would spend a run's budget on "the"."""
        pushed: list[tuple[str, str]] = []
        now = [0.0]
        live = LiveText(
            lambda kind, text: pushed.append((kind, text)), clock=lambda: now[0]
        )

        for word in ["a ", "b ", "c "]:
            live("content", word)

        assert pushed == []
        live.flush()
        assert pushed == [("content", "a b c ")]

    def test_a_pause_pushes_what_is_waiting(self):
        pushed: list[str] = []
        now = [0.0]
        live = LiveText(lambda kind, text: pushed.append(text), clock=lambda: now[0])

        live("content", "first")
        now[0] = 10.0
        live("content", " second")

        assert pushed == ["first second"]

    def test_a_long_burst_does_not_wait_for_the_clock(self):
        pushed: list[str] = []
        live = LiveText(lambda kind, text: pushed.append(text), clock=lambda: 0.0, max_chars=8)

        live("content", "12345")
        live("content", "678")

        assert pushed == ["12345678"]

    def test_thinking_and_answer_are_never_spliced_together(self):
        """One string that is half reasoning and half answer is unreadable."""
        pushed: list[tuple[str, str]] = []
        live = LiveText(lambda kind, text: pushed.append((kind, text)), clock=lambda: 0.0)

        live("reasoning", "hmm")
        live("content", "answer")
        live.flush()

        assert pushed == [("reasoning", "hmm"), ("content", "answer")]


class TestARunShowsItselfBeingWritten:
    """The end of the path: a real run, the real log bus, real subscribers."""

    def test_a_run_publishes_the_answer_as_it_is_produced(self, authorized_user, client, auth):
        from conftest import make_linear_graph
        from cwap_common.logbus import log_bus
        from orchestrator.runner import Worker, start_run

        start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "speak"}
        )

        seen: list[object] = []
        original = log_bus.transient

        def spy(*args, **kwargs):
            seen.append((args, kwargs))
            return original(*args, **kwargs)

        log_bus.transient = spy
        try:
            Worker().drain()
        finally:
            log_bus.transient = original

        events = [args[1] for args, _ in seen]
        assert "llm.streaming" in events

    def test_the_fragments_are_never_written_to_the_run_report(
        self, authorized_user, client, auth
    ):
        """A row per fragment would bury the report under its own tokens — and
        the finished text is already on the step."""
        from conftest import make_linear_graph
        from cwap_common.db import read_only_session
        from cwap_common.models import RunLog
        from orchestrator.runner import Worker, start_run

        handle = start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "speak"}
        )
        Worker().drain()

        with read_only_session() as session:
            stored = {
                row.event for row in session.query(RunLog).filter_by(run_id=handle.run_id)
            }

        assert "llm.streaming" not in stored
        assert "step.completed" in stored

    def test_a_transient_event_does_not_consume_a_sequence_number(self):
        """A client reconnecting asks for "everything after N"; a gap there
        would be a durable event it never receives."""
        from cwap_common.logbus import LogBus

        bus = LogBus()
        first = bus.emit("run_seq", "step.started", message="one")
        bus.transient("run_seq", "llm.streaming", message="a fragment")
        second = bus.emit("run_seq", "step.completed", message="two")

        assert (first.seq, second.seq) == (1, 2)

    def test_a_live_fragment_reaches_a_subscriber(self):
        import asyncio

        from cwap_common.logbus import LogBus

        bus = LogBus()

        async def exercise():
            queue = bus.subscribe("run_live")
            bus.transient("run_live", "agent.streaming", message="half a sen")
            return await asyncio.wait_for(queue.get(), timeout=2)

        assert asyncio.run(exercise()).message == "half a sen"

    def test_a_live_fragment_is_not_dropped_by_the_replay_filter(
        self, authorized_user, client, auth
    ):
        """A subscriber discards anything at or below the sequence it already
        replayed. Transient events reuse that number, so without a marker every
        one of them is silently thrown away — which is exactly what happened the
        first time this was tried against a real streaming server."""
        from cwap_common.logbus import LogBus

        bus = LogBus()
        record = bus.transient("run_marked", "agent.streaming", message="a fragment")

        assert record.data["transient"] is True

    def test_the_spaces_between_fragments_survive_the_trip(
        self, authorized_user, client, auth
    ):
        """Every contract model strips whitespace from its string fields, so a
        fragment carried in `message` arrives with its edges shaved and the
        preview reads "Hereis the answer". The text goes in `data`, which is
        untyped and delivered as sent."""
        from cwap_common.logbus import LogBus

        bus = LogBus()
        record = bus.transient(
            "run_spaces", "agent.streaming", message="x is writing", data={"text": " is "}
        )

        assert record.data["text"] == " is "
