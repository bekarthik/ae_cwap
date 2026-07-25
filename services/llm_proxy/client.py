"""Provider implementations behind one narrow interface.

The platform is model-agnostic by construction: nothing outside this package
knows which model is running. A workflow authored against a local Llama runs
unchanged against Claude, and vice versa.

Three implementations cover everything:

* `StubProvider` — offline, deterministic, no credentials. The default.
* `AnthropicProvider` — the official Anthropic SDK.
* `OpenAICompatibleProvider` — one HTTP client for every other backend, because
  Ollama, vLLM, LM Studio, llama.cpp, TGI, LiteLLM, OpenAI, Together, Groq,
  OpenRouter, Fireworks, DeepSeek and Mistral all speak the same wire format.

The knobs different backends accept are genuinely incompatible — current Claude
models *reject* `temperature` with a 400, and open models have no notion of
`effort`. Rather than pretend otherwise, a provider declares its
`ProviderCapabilities`, silently-unsupported options are reported back in
`ignored_options`, and the canvas only shows controls the configured backend
actually honours.
"""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from cwap_common.settings import get_settings

from llm_proxy.catalogue import capabilities_for
from llm_proxy.presets import NATIVE_PROVIDERS, Preset, resolve

#: Effort levels the Anthropic API accepts, cheapest first.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class LLMProxyError(RuntimeError):
    """The provider could not produce a completion."""


class LLMRefusal(LLMProxyError):
    """The model's safety classifiers declined the request.

    This is a normal, successful HTTP response with `stop_reason == "refusal"`,
    not a transport error — so it must be checked before reading content, and it
    must not be retried with the same prompt.
    """

    def __init__(self, message: str, *, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class LLMConfigurationError(LLMProxyError):
    """The provider is misconfigured — a missing model id, base URL or key.

    Separated from transport failures because the fix is a setting, not a retry.
    """


@dataclass(frozen=True)
class GenerationOptions:
    """What a node asks for. Providers honour what they can and report the rest.

    Deliberately a superset of what any single backend supports: the node author
    should describe intent once, not maintain a variant per model.
    """

    max_tokens: int | None = None
    #: Anthropic-style reasoning depth.
    effort: str | None = None
    #: Sampling temperature, for backends that accept one.
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderCapabilities:
    """What the configured backend can actually do.

    Surfaced through the API so the canvas shows an effort selector against
    Claude and a temperature slider against Llama — instead of offering both and
    silently dropping one.
    """

    provider: str
    label: str
    model: str
    supports_effort: bool
    supports_temperature: bool
    supports_top_p: bool
    supports_stop_sequences: bool
    supports_system_prompt: bool
    #: Native tool/function calling. When false the agent loop falls back to a
    #: prompted JSON protocol, which is what makes small open models usable.
    supports_tools: bool = True
    #: Image input. Set from the model catalogue, since it is a property of the
    #: model rather than of the endpoint.
    supports_vision: bool = False
    #: Exposes a reasoning/thinking mode.
    supports_thinking: bool = False
    deterministic: bool = False
    base_url: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "label": self.label,
            "model": self.model,
            "base_url": self.base_url,
            "deterministic": self.deterministic,
            "notes": self.notes,
            "supports": {
                "effort": self.supports_effort,
                "temperature": self.supports_temperature,
                "top_p": self.supports_top_p,
                "stop_sequences": self.supports_stop_sequences,
                "system_prompt": self.supports_system_prompt,
                "tools": self.supports_tools,
                "vision": self.supports_vision,
                "thinking": self.supports_thinking,
            },
        }


@dataclass(frozen=True)
class ToolCallRequest:
    """A model asking for a skill to be run. Provider-neutral."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    """One turn of an agent conversation.

    `role` is one of user / assistant / tool. Providers translate this into
    whatever shape they need; nothing above this layer knows the difference
    between Anthropic content blocks and OpenAI tool_calls.
    """

    role: str
    content: str = ""
    tool_calls: tuple[ToolCallRequest, ...] = ()
    #: Set on tool results, matching the call they answer.
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class LLMCompletion:
    """What a node executor gets back. Deliberately provider-neutral."""

    text: str
    model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    #: Skills the model wants run before it can continue.
    tool_calls: tuple[ToolCallRequest, ...] = ()
    #: Options the caller asked for that this backend cannot honour. Recorded on
    #: the step so a run report never implies a knob took effect when it did not.
    ignored_options: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    capabilities: ProviderCapabilities

    def complete(
        self, prompt: str, *, system: str | None = None, options: GenerationOptions | None = None
    ) -> LLMCompletion: ...

    def converse(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: GenerationOptions | None = None,
    ) -> LLMCompletion: ...


def _filter_options(
    options: GenerationOptions, capabilities: ProviderCapabilities
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Split requested options into "this backend takes it" and "it does not"."""
    accepted: dict[str, Any] = {}
    ignored: list[str] = []

    checks = (
        ("effort", options.effort, capabilities.supports_effort),
        ("temperature", options.temperature, capabilities.supports_temperature),
        ("top_p", options.top_p, capabilities.supports_top_p),
        ("stop", options.stop or None, capabilities.supports_stop_sequences),
    )
    for name, value, supported in checks:
        if value is None:
            continue
        if supported:
            accepted[name] = value
        else:
            ignored.append(name)

    return accepted, tuple(ignored)


# ---------------------------------------------------------------------------
# stub
# ---------------------------------------------------------------------------


class StubProvider:
    """Deterministic offline provider.

    Not a mock bolted onto the tests — it is the default configuration, so the
    product demos and runs end to end before anyone provisions a key or pulls a
    model. Output is a hash-stable function of the prompt, which is what lets the
    execution tests assert on exact node outputs.
    """

    def __init__(self, model: str = "stub-model") -> None:
        self.capabilities = ProviderCapabilities(
            provider="stub",
            label="Deterministic stub",
            model=model,
            # The stub accepts every knob and honours none of them in a way that
            # changes output — that is the point of being deterministic.
            supports_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_stop_sequences=True,
            supports_system_prompt=True,
            # The stub never asks for a tool, so an agent loop always terminates
            # on its first iteration. Declaring otherwise would make the canvas
            # promise agentic behaviour the default backend cannot deliver.
            supports_tools=False,
            deterministic=True,
            notes="No network, no credentials. Output is a stable function of the prompt.",
        )

    def complete(
        self, prompt: str, *, system: str | None = None, options: GenerationOptions | None = None
    ) -> LLMCompletion:
        digest = hashlib.sha256(f"{system or ''}\x00{prompt}".encode()).hexdigest()[:12]
        summary = textwrap.shorten(" ".join(prompt.split()), width=240, placeholder="…")
        text = f"[stub:{digest}] {summary}"
        return LLMCompletion(
            text=text,
            model=self.capabilities.model,
            stop_reason="end_turn",
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            metadata={"provider": "stub", "deterministic": True},
        )

    def converse(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: GenerationOptions | None = None,
    ) -> LLMCompletion:
        """Never requests a tool, so an agent loop terminates in one iteration.

        That is what keeps the execution tests deterministic. Tests that need to
        exercise multi-iteration behaviour script a provider explicitly rather
        than relying on a stub guessing when to call something.
        """
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)
        return self.complete(transcript, system=system, options=options)


# ---------------------------------------------------------------------------
# anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Calls the Claude Messages API through the official SDK.

    Three details that are easy to get wrong on current models, handled here once
    rather than in every node executor:

    * **No sampling parameters.** `temperature` / `top_p` are rejected with a 400
      on current Claude models, so they are declared unsupported and reported as
      ignored instead of being sent.
    * **Refusals are 200s.** A safety decline returns `stop_reason="refusal"`
      with empty or partial content, so `stop_reason` is checked *before*
      content is read.
    * **Server-side fallback is opted into**, so a declined request is re-run on
      the recommended fallback within the same call rather than failing the run.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"
    DEFAULT_MODEL = "claude-opus-5"

    def __init__(self, model: str = "", *, api_key: str = "", timeout: int = 120) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise LLMConfigurationError(
                "the 'anthropic' package is required for CWAP_LLM_PROVIDER=anthropic; "
                "install it with: pip install -e '.[llm]'"
            ) from exc

        self._anthropic = anthropic
        # An empty api_key means "resolve from the environment or an `ant auth
        # login` profile", which is the SDK's own resolution order.
        self._client = (
            anthropic.Anthropic(api_key=api_key, timeout=timeout)
            if api_key
            else anthropic.Anthropic(timeout=timeout)
        )
        resolved = model or self.DEFAULT_MODEL
        tools, vision, thinking = capabilities_for("anthropic", resolved)
        self.capabilities = ProviderCapabilities(
            provider="anthropic",
            label="Anthropic Claude",
            model=resolved,
            supports_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_stop_sequences=True,
            supports_system_prompt=True,
            supports_tools=tools,
            supports_vision=vision,
            supports_thinking=thinking,
            notes="Current Claude models reject sampling parameters; use effort instead.",
        )

    def complete(
        self, prompt: str, *, system: str | None = None, options: GenerationOptions | None = None
    ) -> LLMCompletion:
        settings = get_settings()
        options = options or GenerationOptions()
        accepted, ignored = _filter_options(options, self.capabilities)

        effort = str(accepted.get("effort") or settings.llm_effort).lower()
        if effort not in EFFORT_LEVELS:
            raise LLMConfigurationError(
                f"unknown effort '{effort}'; expected one of {list(EFFORT_LEVELS)}"
            )

        request: dict[str, Any] = {
            "model": self.capabilities.model,
            "max_tokens": options.max_tokens or settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort},
            "betas": [self.FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        if "stop" in accepted:
            request["stop_sequences"] = list(accepted["stop"])

        try:
            response = self._client.beta.messages.create(**request)
        except TypeError:
            # An older SDK build may not type the `fallbacks` scalar form yet.
            # Losing the fallback is acceptable; losing the call is not.
            request.pop("fallbacks", None)
            request.pop("betas", None)
            response = self._client.messages.create(**request)
        except self._anthropic.APIStatusError as exc:
            raise LLMProxyError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMProxyError(f"could not reach the Claude API: {exc}") from exc

        # Check the stop reason before touching content: on a refusal, `content`
        # is empty or a discarded partial.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                "the model declined this request",
                category=getattr(details, "category", None),
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMCompletion(
            text=text,
            model=getattr(response, "model", self.capabilities.model),
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ignored_options=ignored,
            metadata={"provider": "anthropic", "effort": effort},
        )

    def converse(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: GenerationOptions | None = None,
    ) -> LLMCompletion:
        """Multi-turn with native tool use.

        Anthropic models carry tool calls as `tool_use` content blocks and expect
        results back as `tool_result` blocks inside a user turn — not as a
        separate role, which is where the OpenAI shape differs.
        """
        settings = get_settings()
        options = options or GenerationOptions()
        accepted, ignored = _filter_options(options, self.capabilities)
        effort = str(accepted.get("effort") or settings.llm_effort).lower()

        request: dict[str, Any] = {
            "model": self.capabilities.model,
            "max_tokens": options.max_tokens or settings.llm_max_tokens,
            "messages": _to_anthropic_messages(messages),
            "output_config": {"effort": effort},
            "betas": [self.FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in tools
            ]

        try:
            response = self._client.beta.messages.create(**request)
        except TypeError:
            request.pop("fallbacks", None)
            request.pop("betas", None)
            response = self._client.messages.create(**request)
        except self._anthropic.APIStatusError as exc:
            raise LLMProxyError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMProxyError(f"could not reach the Claude API: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                "the model declined this request", category=getattr(details, "category", None)
            )

        text_parts: list[str] = []
        calls: list[ToolCallRequest] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                calls.append(
                    ToolCallRequest(
                        id=block.id, name=block.name, arguments=dict(block.input or {})
                    )
                )

        usage = getattr(response, "usage", None)
        return LLMCompletion(
            text="".join(text_parts),
            model=getattr(response, "model", self.capabilities.model),
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            tool_calls=tuple(calls),
            ignored_options=ignored,
            metadata={"provider": "anthropic", "effort": effort},
        )


# ---------------------------------------------------------------------------
# openai-compatible (open models, local servers, and most hosted APIs)
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """One client for every backend that speaks `/chat/completions`.

    Implemented directly on httpx rather than a vendor SDK, deliberately. The
    servers this must work against — Ollama, llama.cpp, LM Studio, TGI, vLLM and
    a dozen hosted APIs — agree on the wire format but differ in the edges (some
    omit `usage`, some ignore `max_tokens`, some return a bare error string).
    Owning the request and the error mapping is what lets a local model produce
    the same clear failures as a hosted one, with no extra dependency to install
    for someone running entirely offline.
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 120,
        label: str = "",
        notes: str = "",
    ) -> None:
        if not base_url:
            raise LLMConfigurationError(
                f"provider '{provider}' needs a base URL; set CWAP_LLM_BASE_URL "
                "(for example http://localhost:11434/v1 for Ollama)"
            )
        if not model:
            raise LLMConfigurationError(
                f"provider '{provider}' needs a model id; set CWAP_LLM_MODEL "
                "to a model your server has loaded"
            )

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        # "auto" tries native tool calling once and downgrades permanently on a
        # rejection; an operator can force either mode.
        self._tool_mode = get_settings().llm_tool_mode
        tools, vision, thinking = capabilities_for(provider, model)
        self.capabilities = ProviderCapabilities(
            provider=provider,
            label=label or provider,
            model=model,
            supports_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_stop_sequences=True,
            supports_system_prompt=True,
            # A hint from the catalogue, not a promise: `_post` still downgrades
            # permanently if the server rejects a tools request at runtime.
            supports_tools=tools and self._tool_mode != "prompted",
            supports_vision=vision,
            supports_thinking=thinking,
            base_url=self._base_url,
            notes=notes,
        )

    def complete(
        self, prompt: str, *, system: str | None = None, options: GenerationOptions | None = None
    ) -> LLMCompletion:
        import httpx  # noqa: PLC0415 - already a dependency; imported at call site

        settings = get_settings()
        options = options or GenerationOptions()
        accepted, ignored = _filter_options(options, self.capabilities)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.capabilities.model,
            "messages": messages,
            "max_tokens": options.max_tokens or settings.llm_max_tokens,
            "temperature": accepted.get("temperature", settings.llm_temperature),
            "stream": False,
        }
        if "top_p" in accepted:
            payload["top_p"] = accepted["top_p"]
        if "stop" in accepted:
            payload["stop"] = list(accepted["stop"])

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions", json=payload, headers=headers
                )
        except httpx.ConnectError as exc:
            raise LLMProxyError(
                f"could not reach {self._base_url} — is the server running? ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProxyError(
                f"{self.capabilities.label} did not respond within {self._timeout}s. "
                "Local models on modest hardware are slow; raise CWAP_LLM_TIMEOUT."
            ) from exc

        if response.status_code >= 400:
            raise LLMProxyError(self._explain_error(response))

        try:
            body = response.json()
            choice = body["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            finish_reason = choice.get("finish_reason") or "stop"
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMProxyError(
                f"{self.capabilities.label} returned an unexpected response shape: "
                f"{response.text[:400]}"
            ) from exc

        # Some servers put reasoning in a separate field and leave content empty.
        if not text:
            text = (choice.get("message") or {}).get("reasoning_content") or ""

        usage = body.get("usage") or {}
        return LLMCompletion(
            text=text,
            model=body.get("model") or self.capabilities.model,
            stop_reason=finish_reason,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            ignored_options=ignored,
            metadata={
                "provider": self.capabilities.provider,
                "base_url": self._base_url,
                # Surfaced so a truncated answer is visible in the run report
                # rather than looking like the model simply stopped early.
                "truncated": finish_reason == "length",
            },
        )

    def converse(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: GenerationOptions | None = None,
    ) -> LLMCompletion:
        """Multi-turn, with native tool calling when the server supports it.

        Tool support is not discoverable up front: a hosted API advertises it,
        a local llama.cpp build may or may not have it, and the only honest way
        to find out is to try. So the first attempt sends `tools`; a rejection
        that names tools downgrades this provider to the prompted protocol for
        the rest of its life, and the call is retried immediately rather than
        failing the agent's turn.
        """
        if tools and self._tool_mode == "prompted":
            return self._converse_prompted(messages, system, tools, options)

        try:
            return self._converse_native(messages, system, tools, options)
        except _ToolsUnsupported:
            self._tool_mode = "prompted"
            self.capabilities = replace(self.capabilities, supports_tools=False)
            return self._converse_prompted(messages, system, tools or [], options)

    def _converse_native(
        self,
        messages: list[ChatMessage],
        system: str | None,
        tools: list[dict[str, Any]] | None,
        options: GenerationOptions | None,
    ) -> LLMCompletion:
        payload = self._base_payload(options)
        payload["messages"] = _to_openai_messages(messages, system)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"

        body = self._post(payload, tools_requested=bool(tools))
        choice = body["choices"][0]
        message = choice.get("message") or {}

        calls: list[ToolCallRequest] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            calls.append(
                ToolCallRequest(
                    id=raw.get("id") or f"call_{len(calls)}",
                    name=function.get("name", ""),
                    arguments=_loads_arguments(function.get("arguments")),
                )
            )

        usage = body.get("usage") or {}
        return LLMCompletion(
            text=message.get("content") or message.get("reasoning_content") or "",
            model=body.get("model") or self.capabilities.model,
            stop_reason=choice.get("finish_reason") or "stop",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            tool_calls=tuple(calls),
            metadata={
                "provider": self.capabilities.provider,
                "tool_mode": "native",
                "truncated": choice.get("finish_reason") == "length",
            },
        )

    def _converse_prompted(
        self,
        messages: list[ChatMessage],
        system: str | None,
        tools: list[dict[str, Any]],
        options: GenerationOptions | None,
    ) -> LLMCompletion:
        """Tool use for models that have none.

        Plenty of capable open-weight models never learned function calling. The
        alternative to this fallback is telling the user their model cannot run
        agents, which would make "works with any model" untrue. The model is
        asked to emit a single JSON object; anything that is not parseable as a
        call is treated as a final answer, so a model that ignores the protocol
        degrades to a plain one-shot response rather than erroring.
        """
        payload = self._base_payload(options)
        payload["messages"] = _to_openai_messages(
            messages, _prompted_system(system, tools) if tools else system
        )

        body = self._post(payload, tools_requested=False)
        choice = body["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""

        call = _parse_prompted_call(text, {tool["name"] for tool in tools}) if tools else None
        usage = body.get("usage") or {}
        return LLMCompletion(
            text="" if call else text,
            model=body.get("model") or self.capabilities.model,
            stop_reason=choice.get("finish_reason") or "stop",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            tool_calls=(call,) if call else (),
            metadata={
                "provider": self.capabilities.provider,
                "tool_mode": "prompted",
                "truncated": choice.get("finish_reason") == "length",
            },
        )

    def _base_payload(self, options: GenerationOptions | None) -> dict[str, Any]:
        settings = get_settings()
        options = options or GenerationOptions()
        accepted, _ignored = _filter_options(options, self.capabilities)
        payload: dict[str, Any] = {
            "model": self.capabilities.model,
            "max_tokens": options.max_tokens or settings.llm_max_tokens,
            "temperature": accepted.get("temperature", settings.llm_temperature),
            "stream": False,
        }
        if "top_p" in accepted:
            payload["top_p"] = accepted["top_p"]
        return payload

    def _post(self, payload: dict[str, Any], *, tools_requested: bool) -> dict[str, Any]:
        import httpx  # noqa: PLC0415

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions", json=payload, headers=headers
                )
        except httpx.ConnectError as exc:
            raise LLMProxyError(
                f"could not reach {self._base_url} — is the server running? ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProxyError(
                f"{self.capabilities.label} did not respond within {self._timeout}s. "
                "Local models on modest hardware are slow; raise CWAP_LLM_TIMEOUT."
            ) from exc

        if response.status_code >= 400:
            detail = self._explain_error(response)
            if tools_requested and _looks_like_tool_rejection(response):
                raise _ToolsUnsupported(detail)
            raise LLMProxyError(detail)

        try:
            body = response.json()
            body["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMProxyError(
                f"{self.capabilities.label} returned an unexpected response shape: "
                f"{response.text[:400]}"
            ) from exc
        return body

    def _explain_error(self, response: Any) -> str:
        """Turn a backend's error into something a workflow author can act on."""
        detail = response.text[:400]
        try:
            body = response.json()
            error = body.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or detail
            elif isinstance(error, str):
                detail = error
        except ValueError:
            pass

        if response.status_code == 404:
            return (
                f"{self.capabilities.label} does not have model "
                f"'{self.capabilities.model}' ({detail}). Check CWAP_LLM_MODEL, and that "
                "the model is pulled or loaded on the server."
            )
        if response.status_code in (401, 403):
            return (
                f"{self.capabilities.label} rejected the credentials ({detail}). "
                "Set CWAP_LLM_API_KEY."
            )
        if response.status_code == 429:
            return f"{self.capabilities.label} is rate limiting: {detail}"
        return f"{self.capabilities.label} returned {response.status_code}: {detail}"


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def build_provider(settings=None) -> LLMProvider:
    """Construct the provider named by configuration."""
    settings = settings or get_settings()
    name = settings.llm_provider.strip().lower()

    if name == "stub":
        return StubProvider(settings.llm_model or "stub-model")

    if name == "anthropic":
        return AnthropicProvider(
            settings.llm_model,
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    preset = resolve(name)
    if preset is None:
        from llm_proxy.presets import known_providers  # noqa: PLC0415

        raise LLMConfigurationError(
            f"unknown CWAP_LLM_PROVIDER '{settings.llm_provider}'. "
            f"Expected one of: {', '.join(known_providers())}"
        )

    return _from_preset(preset, settings)


def _from_preset(preset: Preset, settings) -> OpenAICompatibleProvider:
    base_url = settings.llm_base_url or preset.base_url
    model = settings.llm_model or preset.default_model
    api_key = settings.llm_api_key

    if preset.requires_key and not api_key:
        raise LLMConfigurationError(
            f"{preset.label} requires a key; set CWAP_LLM_API_KEY"
        )

    return OpenAICompatibleProvider(
        provider=preset.key,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=settings.llm_timeout_seconds,
        label=preset.label,
        notes=preset.notes,
    )


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Process-wide provider, chosen by configuration."""
    global _provider
    if _provider is None:
        _provider = build_provider()
    return _provider


def reset_provider_cache(provider: LLMProvider | None = None) -> None:
    """Swap the provider. Used by tests and by the dev runner."""
    global _provider
    _provider = provider


def describe_provider() -> dict[str, Any]:
    """Capabilities of the configured backend, for the API and the canvas.

    Reports a configuration error as data rather than raising, so a
    misconfigured deployment shows an actionable message in the UI instead of a
    500 on an unrelated page.
    """
    try:
        return {"configured": True, **get_provider().capabilities.as_dict()}
    except LLMConfigurationError as exc:
        settings = get_settings()
        return {
            "configured": False,
            "provider": settings.llm_provider,
            "error": str(exc),
            "supports": {
                "effort": False,
                "temperature": False,
                "top_p": False,
                "stop_sequences": False,
                "system_prompt": False,
            },
        }


__all__ = [
    "EFFORT_LEVELS",
    "NATIVE_PROVIDERS",
    "AnthropicProvider",
    "GenerationOptions",
    "LLMCompletion",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProxyError",
    "LLMRefusal",
    "OpenAICompatibleProvider",
    "ProviderCapabilities",
    "StubProvider",
    "build_provider",
    "describe_provider",
    "get_provider",
    "reset_provider_cache",
]


# ---------------------------------------------------------------------------
# message translation and the prompted tool protocol
# ---------------------------------------------------------------------------


class _ToolsUnsupported(LLMProxyError):
    """The server rejected the request *because* it was asked for tools.

    Internal: it triggers a permanent downgrade to the prompted protocol rather
    than surfacing to the caller.
    """


#: What the error has to be *about*. Without this, a rejected API key would
#: silently downgrade the whole provider to the prompted protocol.
_TOOL_SUBJECTS = ("tool", "function calling", "function_call")

#: How servers say "I do not have that". Every backend phrases it differently —
#: "does not support tools", "tools are unsupported", "unknown field: tools" —
#: so the match is subject-plus-negation rather than a list of exact sentences,
#: which would need an entry per server and still miss the next one.
_UNSUPPORTED_MARKERS = (
    "not support",
    "unsupported",
    "not supported",
    "unknown field",
    "unknown parameter",
    "unrecognized",
    "unrecognised",
    "invalid parameter",
    "extra inputs are not permitted",
    "not implemented",
    "not available",
    "no such parameter",
)


def _looks_like_tool_rejection(response: Any) -> bool:
    """Whether a 4xx means "I have no tools" rather than something else.

    Matching on text is unpleasant, but there is no status code that means it,
    and getting it wrong in either direction is costly: too narrow and a capable
    open model is reported as broken, too broad and a bad API key silently
    becomes a permanent downgrade.
    """
    body = (getattr(response, "text", "") or "").lower()
    if not any(subject in body for subject in _TOOL_SUBJECTS):
        return False
    return any(marker in body for marker in _UNSUPPORTED_MARKERS)


def _loads_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string* in the OpenAI shape, and small
    models sometimes emit something that is nearly JSON."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    import json  # noqa: PLC0415

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    except (ValueError, TypeError):
        return {"input": str(raw)}


def _to_openai_messages(
    messages: list[ChatMessage], system: str | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        if message.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
            continue

        entry: dict[str, Any] = {"role": message.role, "content": message.content or ""}
        if message.tool_calls:
            import json  # noqa: PLC0415

            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
        out.append(entry)
    return out


def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Anthropic carries tool results inside a *user* turn as content blocks,
    rather than as a distinct role. Consecutive results are merged into one
    turn, which the API requires."""
    out: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
            continue

        out.append({"role": message.role, "content": message.content or ""})

    return out


PROMPTED_TOOL_PROTOCOL = """\
You can use tools. To use one, reply with ONLY a JSON object and nothing else:

{"tool": "<tool_name>", "arguments": {"<param>": "<value>"}}

If you do not need a tool, reply with your answer as normal prose — no JSON.
Use one tool at a time and wait for its result before deciding what to do next.

Available tools:
"""


def _prompted_system(system: str | None, tools: list[dict[str, Any]]) -> str:
    """Describe the tools in the system prompt for models with no tool API."""
    described = []
    for tool in tools:
        properties = (tool.get("input_schema") or {}).get("properties") or {}
        params = ", ".join(
            f"{name} ({spec.get('type', 'string')})" for name, spec in properties.items()
        )
        described.append(f"- {tool['name']}({params}): {tool['description']}")

    block = PROMPTED_TOOL_PROTOCOL + "\n".join(described)
    return f"{system}\n\n{block}" if system else block


def _parse_prompted_call(text: str, known: set[str]) -> ToolCallRequest | None:
    """Read a tool call out of prose, or decide there isn't one.

    Returns None on anything ambiguous. A model that ignores the protocol should
    degrade to a plain answer, never to an invented tool call.
    """
    import json  # noqa: PLC0415
    import uuid as _uuid  # noqa: PLC0415

    candidate = text.strip()
    if "```" in candidate:
        blocks = [part for part in candidate.split("```") if "{" in part]
        if blocks:
            candidate = blocks[0]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]

    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(candidate[start : end + 1])
    except (ValueError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None
    name = parsed.get("tool") or parsed.get("name")
    if not isinstance(name, str) or name not in known:
        return None

    arguments = parsed.get("arguments") or parsed.get("input") or {}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}

    return ToolCallRequest(id=f"call_{_uuid.uuid4().hex[:12]}", name=name, arguments=arguments)
