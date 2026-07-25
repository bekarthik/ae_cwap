"""Reading an answer as it is produced, instead of after it is finished.

Every model call used to be one blocking request. That is simpler, and it was
wrong for what this platform actually points at: a model on somebody's laptop
takes minutes, and for those minutes the run panel showed a spinner and the
timeout counted down against the whole generation rather than against silence.
A model producing a token a second for nine minutes was indistinguishable from
a model that had died — to the user, and to httpx.

Streaming fixes both, and the second one matters more than the visible one:

* **The timeout becomes a silence detector.** A read timeout applies per chunk,
  so a slow-but-working model is never cut off, while a wedged one is caught in
  seconds rather than after the full wait.
* **The answer appears while it is being written**, which is the difference
  between watching a workflow run and watching a progress bar.

The design decision worth stating: this module **reassembles the stream into the
same body a non-streaming call would have returned**. Everything above the
transport — tool-call parsing, reasoning extraction, truncation reporting, the
prompted protocol — keeps reading `body["choices"][0]["message"]` and does not
know the difference. Streaming is a property of the transport, not a second code
path through the provider, so there is no "streaming mode" for a bug to hide in.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from llm_proxy.reasoning import NEGATIONS, REASONING_KEYS

#: What the server sends instead of a final chunk. Not JSON, and not an error.
SENTINEL = "[DONE]"

#: Called with ("content" | "reasoning", text) for each fragment, as it lands.
DeltaSink = Callable[[str, str], None]


class StreamInterrupted(RuntimeError):
    """The stream carried an error, or stopped before the answer was complete.

    Distinct from a transport error because it arrives *inside* a 200: the
    server accepted the request, started answering, and then failed. The caller
    turns this into a normal provider error; the point of the type is that the
    partial answer must not be mistaken for a finished one.
    """


def assemble(lines: Iterable[str], *, on_delta: DeltaSink | None = None) -> dict[str, Any]:
    """Consume a server-sent-event stream and return one completed body.

    The shape returned is a non-streaming chat completion — same keys, same
    nesting — so callers are unchanged. Fragments are handed to `on_delta` as
    they arrive, which is the only thing that has to happen *during* the stream
    rather than after it.
    """
    content: list[str] = []
    thinking: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = ""
    model = ""
    usage: dict[str, Any] = {}
    saw_chunk = False

    for line in lines:
        chunk = _payload(line)
        if chunk is None:
            continue
        if chunk is _DONE:
            break

        _raise_if_error(chunk)
        saw_chunk = True
        model = chunk.get("model") or model
        # Sent in its own final chunk when `stream_options.include_usage` is on,
        # and by several servers unprompted. Either way, the last one wins.
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]

        for choice in chunk.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            # `delta` on a stream; `message` on servers that echo the whole
            # thing in one chunk, which is legal and which some proxies do.
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = choice.get("message")
            if not isinstance(delta, dict):
                continue

            text = delta.get("content")
            if isinstance(text, str) and text:
                content.append(text)
                if on_delta is not None:
                    on_delta("content", text)

            thought = _reasoning_delta(delta)
            if thought:
                thinking.append(thought)
                if on_delta is not None:
                    on_delta("reasoning", thought)

            _merge_tool_calls(calls, delta.get("tool_calls"))

    if not saw_chunk:
        raise StreamInterrupted("the server closed the stream without sending anything")

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if thinking:
        message["reasoning_content"] = "".join(thinking)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]

    return {
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason or "stop"}
        ],
        "usage": usage,
    }


class _Done:
    """Sentinel object; `[DONE]` is not JSON and is not a chunk."""


_DONE = _Done()


def _payload(line: str) -> dict[str, Any] | _Done | None:
    """One SSE line as a chunk, the end sentinel, or nothing to do.

    Blank lines separate events, `:` lines are comments (several servers send
    them as keepalives), and `event:` / `id:` fields are not used here.
    """
    line = (line or "").strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None

    body = line[len("data:") :].strip()
    if not body:
        return None
    if body == SENTINEL:
        return _DONE
    try:
        parsed = json.loads(body)
    except ValueError:
        # A malformed line is not worth failing an otherwise good answer over;
        # the alternative is losing a completed generation to one bad frame.
        return None
    return parsed if isinstance(parsed, dict) else None


def _raise_if_error(chunk: dict[str, Any]) -> None:
    """A 200 that turns into an error partway through still has to fail.

    Servers signal an out-of-memory, a cancelled generation or a context
    overflow this way. Returning the partial text as if it were the answer would
    hand a truncated result downstream as a finished one.
    """
    error = chunk.get("error")
    if not error:
        return
    if isinstance(error, dict):
        raise StreamInterrupted(str(error.get("message") or error))
    raise StreamInterrupted(str(error))


def _reasoning_delta(delta: dict[str, Any]) -> str:
    """The thinking fragment in this delta, under whichever key it came."""
    for key in REASONING_KEYS:
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            joined = "".join(
                str(block.get("text") or "") for block in value if isinstance(block, dict)
            )
            if joined:
                return joined
    return ""


def _merge_tool_calls(calls: dict[int, dict[str, Any]], fragments: Any) -> None:
    """Rebuild tool calls from the pieces they arrive in.

    This is the part of streaming that genuinely differs from a blocking read: a
    native tool call is delivered as a name in one frame and its arguments a few
    characters at a time across many, keyed by position in the array. Dispatching
    before the last fragment lands would call a tool with half its arguments.
    """
    if not isinstance(fragments, list):
        return

    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        index = fragment.get("index")
        index = index if isinstance(index, int) else len(calls)
        entry = calls.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )

        if fragment.get("id"):
            entry["id"] = fragment["id"]
        if fragment.get("type"):
            entry["type"] = fragment["type"]

        function = fragment.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name"):
            entry["function"]["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            entry["function"]["arguments"] += arguments


#: Asking for token counts on a stream is its own parameter, and not every
#: server takes it. Kept separate from `stream` itself so that a server which
#: dislikes it loses only the token counts, not the live answer.
USAGE_OPTION = {"include_usage": True}

_STREAM_SUBJECTS = ("stream",)
_OPTION_SUBJECTS = ("stream_options", "include_usage")


def rejection(body: str) -> str | None:
    """Which streaming parameter a 4xx is complaining about, if either.

    Same subject-plus-negation shape as the tools and reasoning matchers, and
    deliberately narrow: mistaking an unrelated 400 for "this server cannot
    stream" would quietly put every later call back on the blocking path.
    """
    lowered = (body or "").lower()
    if not any(negation in lowered for negation in NEGATIONS):
        return None
    if any(subject in lowered for subject in _OPTION_SUBJECTS):
        return "stream_options"
    if any(subject in lowered for subject in _STREAM_SUBJECTS):
        return "stream"
    return None
