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
from dataclasses import dataclass, field
from typing import Any, Protocol

from cwap_common.settings import get_settings

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
            },
        }


@dataclass(frozen=True)
class LLMCompletion:
    """What a node executor gets back. Deliberately provider-neutral."""

    text: str
    model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    #: Options the caller asked for that this backend cannot honour. Recorded on
    #: the step so a run report never implies a knob took effect when it did not.
    ignored_options: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    capabilities: ProviderCapabilities

    def complete(
        self, prompt: str, *, system: str | None = None, options: GenerationOptions | None = None
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
        self.capabilities = ProviderCapabilities(
            provider="anthropic",
            label="Anthropic Claude",
            model=model or self.DEFAULT_MODEL,
            supports_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_stop_sequences=True,
            supports_system_prompt=True,
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
        self.capabilities = ProviderCapabilities(
            provider=provider,
            label=label or provider,
            model=model,
            supports_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_stop_sequences=True,
            supports_system_prompt=True,
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
