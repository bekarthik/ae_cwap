"""Engaging a model's reasoning mode, whichever dialect its server speaks.

A thinking-capable model that is never asked to think is an expensive model used
badly. The platform detected the capability and did nothing with it: no request
ever carried a reasoning parameter, and any reasoning the model produced anyway
was discarded unless the answer happened to be empty.

The obstacle is that there is no agreed way to ask. The OpenAI wire format is
near-universal for *messages* and completely unstandardised for *reasoning* —
every server invented its own parameter, and sending the wrong one is a 400. So
each backend's dialect is written down here, and `client._post` treats a
rejection the same way it treats a rejected tools request: drop the parameter,
retry immediately, remember. A model that turns out not to accept it costs one
retried call, not a failed run.

`REASONING_FIELDS` is where a new backend is added. That is the whole extension
point — no new provider class, no branch in the request builder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Effort levels, cheapest first. The same vocabulary the Anthropic provider
#: uses, so a workflow that asks for "high" means one thing everywhere.
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")

#: What the OpenAI reasoning parameter accepts. Anything above "high" in this
#: platform's vocabulary maps onto it, since there is nothing higher to ask for.
_OPENAI_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def _openai_style(effort: str) -> dict[str, Any]:
    """`reasoning_effort`, as OpenAI's o-series and gpt-5 accept it."""
    return {"reasoning_effort": _OPENAI_EFFORT.get(effort, "medium")}


def _openrouter_style(effort: str) -> dict[str, Any]:
    """OpenRouter normalises across the models it fronts, under `reasoning`."""
    return {"reasoning": {"effort": _OPENAI_EFFORT.get(effort, "medium")}}


def _ollama_style(_effort: str) -> dict[str, Any]:
    """Ollama exposes thinking as a boolean; there is no depth to ask for."""
    return {"think": True}


def _template_style(_effort: str) -> dict[str, Any]:
    """vLLM and SGLang pass flags into the model's chat template.

    Qwen3 and similar read `enable_thinking` there. It is the server's documented
    way to reach a template variable, so it is the honest place to put it.
    """
    return {"chat_template_kwargs": {"enable_thinking": True}}


def _none(_effort: str) -> dict[str, Any]:
    """The model reasons unconditionally; asking would be a rejected parameter.

    DeepSeek's reasoner is the example: reasoning is the model, not a mode.
    """
    return {}


#: Provider key → how that server is asked to think. A provider absent from this
#: table is sent nothing, which is the safe default: an unknown server gets a
#: normal request rather than one it might reject.
REASONING_FIELDS: dict[str, Callable[[str], dict[str, Any]]] = {
    "openai": _openai_style,
    "openrouter": _openrouter_style,
    "ollama": _ollama_style,
    "vllm": _template_style,
    "lmstudio": _openai_style,
    "litellm": _openai_style,
    "groq": _openai_style,
    "fireworks": _openai_style,
    "deepseek": _none,
    "together": _openai_style,
    "mistral": _openai_style,
    "openai_compatible": _openai_style,
    "gemini": _openai_style,
    "xai": _openai_style,
    "llamacpp": _template_style,
    "tgi": _template_style,
}


def request_fields(provider: str, effort: str) -> dict[str, Any]:
    """The reasoning parameters to add for this backend, if any."""
    dialect = REASONING_FIELDS.get(provider.strip().lower())
    return dialect(effort or "medium") if dialect else {}


def knows(provider: str) -> bool:
    """Whether this platform has a way to ask this backend to think."""
    dialect = REASONING_FIELDS.get(provider.strip().lower())
    # A provider mapped to `_none` reasons on its own and takes no parameter;
    # there is nothing to ask for, so there is nothing to engage or to retry.
    return dialect is not None and dialect is not _none


#: Every key any dialect above can add. Used to take the request back to a plain
#: one after a server rejects it, without having to remember which dialect ran.
PARAMS = ("reasoning_effort", "reasoning", "think", "chat_template_kwargs")


def strip(payload: dict[str, Any]) -> dict[str, Any]:
    """The same request with no reasoning parameters — what the retry sends."""
    return {key: value for key, value in payload.items() if key not in PARAMS}


#: Where servers put the model's reasoning when they return it separately from
#: the answer. Checked in order; the first present wins.
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def extract(message: dict[str, Any]) -> str:
    """The model's reasoning, when the server returned it apart from the answer.

    Kept rather than dropped. It is the most direct evidence of *why* a step
    produced what it did, which is the whole promise of the run report — and on
    a reasoning model it is often the only place the real work is visible.
    """
    for key in REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        # OpenRouter returns structured reasoning blocks rather than a string.
        if isinstance(value, list):
            joined = "\n".join(
                str(block.get("text") or "")
                for block in value
                if isinstance(block, dict)
            ).strip()
            if joined:
                return joined
    return ""


#: Reasoning is spent from the same output budget as the answer, so a model that
#: thinks hard against a small ceiling truncates mid-sentence. Requests that
#: engage thinking get at least this much room.
MIN_THINKING_TOKENS = 8192


def token_budget(requested: int) -> int:
    """Output budget for a request that will spend part of it thinking."""
    return max(requested, MIN_THINKING_TOKENS)


#: Fragments a server uses when it does not accept a reasoning parameter. Same
#: subject-plus-negation shape as the tool-rejection matcher, and for the same
#: reason: every backend phrases it differently and an exact-sentence list would
#: need an entry per server and still miss the next one.
_SUBJECTS = ("reasoning", "think", "chat_template_kwargs", "enable_thinking")

#: The "it refused the parameter" half of the match, shared with `streaming`,
#: which needs exactly the same vocabulary for exactly the same reason.
NEGATIONS = (
    "not support",
    "unsupported",
    "not supported",
    "unknown field",
    "unknown parameter",
    "unrecognized",
    "unrecognised",
    "invalid parameter",
    "extra inputs are not permitted",
    "extra fields not permitted",
    "not implemented",
    "no such parameter",
    "does not accept",
)


def looks_rejected(body: str) -> bool:
    """Whether a 4xx means "I do not take a reasoning parameter"."""
    lowered = (body or "").lower()
    if not any(subject in lowered for subject in _SUBJECTS):
        return False
    return any(negation in lowered for negation in NEGATIONS)
