"""Provider implementations behind one narrow interface."""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass, field
from typing import Any, Protocol

from cwap_common.settings import get_settings

#: Effort levels the Claude API accepts, cheapest first.
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


@dataclass(frozen=True)
class LLMCompletion:
    """What a node executor gets back. Deliberately provider-neutral."""

    text: str
    model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> LLMCompletion: ...


class StubProvider:
    """Deterministic offline provider.

    Not a mock bolted onto the tests — it is the default configuration, so the
    product demos and runs end-to-end before anyone provisions a key. Output is
    a hash-stable function of the prompt, which is what lets the execution tests
    assert on exact node outputs.
    """

    name = "stub"

    def __init__(self, model: str = "stub-model") -> None:
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> LLMCompletion:
        digest = hashlib.sha256(f"{system or ''}\x00{prompt}".encode()).hexdigest()[:12]
        summary = textwrap.shorten(" ".join(prompt.split()), width=240, placeholder="…")
        text = f"[stub:{digest}] {summary}"
        return LLMCompletion(
            text=text,
            model=self.model,
            stop_reason="end_turn",
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            metadata={"provider": self.name, "deterministic": True},
        )


class AnthropicProvider:
    """Calls the Claude Messages API through the official SDK.

    Three details that are easy to get wrong on current models and are handled
    here once, rather than in every node executor:

    * **No sampling parameters.** `temperature` / `top_p` / `top_k` are rejected
      with a 400 on Claude Opus 5, so an LLM node steers behaviour through the
      prompt and `effort`, never through a temperature knob.
    * **Refusals are 200s.** A safety decline returns `stop_reason="refusal"`
      with empty or partial content, so `stop_reason` is checked *before*
      content is read.
    * **Server-side fallback is opted into.** A declined request is re-run on
      Anthropic's recommended fallback within the same call instead of failing
      the workflow run outright.
    """

    name = "anthropic"

    #: Beta flag gating the `fallbacks: "default"` scalar form.
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, *, api_key: str = "", timeout: int = 120) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional at import time
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMProxyError(
                "the 'anthropic' package is required for CWAP_LLM_PROVIDER=anthropic"
            ) from exc

        self._anthropic = anthropic
        # An empty api_key means "resolve from the environment or an `ant auth
        # login` profile", which is the SDK's own default resolution order.
        self._client = (
            anthropic.Anthropic(api_key=api_key, timeout=timeout)
            if api_key
            else anthropic.Anthropic(timeout=timeout)
        )
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> LLMCompletion:
        settings = get_settings()
        resolved_effort = (effort or settings.llm_effort).lower()
        if resolved_effort not in EFFORT_LEVELS:
            raise LLMProxyError(
                f"unknown effort '{resolved_effort}'; expected one of {list(EFFORT_LEVELS)}"
            )

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": resolved_effort},
            "betas": [self.FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system

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
            model=getattr(response, "model", self.model),
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            metadata={"provider": self.name, "effort": resolved_effort},
        )


_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Process-wide provider, chosen by configuration."""
    global _provider
    if _provider is None:
        settings = get_settings()
        if settings.llm_provider == "anthropic":
            _provider = AnthropicProvider(
                settings.llm_model,
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_seconds,
            )
        else:
            _provider = StubProvider(settings.llm_model)
    return _provider


def reset_provider_cache(provider: LLMProvider | None = None) -> None:
    """Swap the provider. Used by tests and by the dev runner."""
    global _provider
    _provider = provider
