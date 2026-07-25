"""LLM Proxy Service — the single place the platform talks to a model.

Nothing else in the codebase imports a vendor SDK. That is deliberate: it keeps
API keys in one process, gives every model call one place to be logged and
redacted, and means swapping or adding a provider is a change to this package
alone.

Two providers ship:

* `StubProvider` (default) — deterministic, no network, no credentials. It is
  what makes `make dev` and the whole test suite work on a laptop with no keys,
  and it makes workflow-execution tests assert on exact outputs.
* `AnthropicProvider` — the real thing, via the official `anthropic` SDK.
"""

from llm_proxy.client import (
    LLMCompletion,
    LLMProvider,
    LLMProxyError,
    LLMRefusal,
    get_provider,
    reset_provider_cache,
)

__all__ = [
    "LLMCompletion",
    "LLMProvider",
    "LLMProxyError",
    "LLMRefusal",
    "get_provider",
    "reset_provider_cache",
]
