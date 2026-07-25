"""LLM Proxy Service — the single place the platform talks to a model.

Nothing else in the codebase imports a model client. That is what makes the
platform model-agnostic: a workflow is authored against capabilities, not
against a vendor, so the same graph runs on a local Llama, a self-hosted Qwen,
or a hosted Claude with no change to the workflow at all.

Configure with `CWAP_LLM_PROVIDER`:

    stub                            offline, deterministic (the default)
    anthropic                       Claude via the official SDK
    ollama | vllm | lmstudio |      open-weight models, local or self-hosted
      llamacpp | tgi | litellm
    openai | together | groq |      hosted OpenAI-compatible APIs
      openrouter | fireworks |
      deepseek | mistral
    openai_compatible               anything else, with CWAP_LLM_BASE_URL
"""

from llm_proxy.client import (
    EFFORT_LEVELS,
    AnthropicProvider,
    GenerationOptions,
    LLMCompletion,
    LLMConfigurationError,
    LLMProvider,
    LLMProxyError,
    LLMRefusal,
    OpenAICompatibleProvider,
    ProviderCapabilities,
    StubProvider,
    build_provider,
    describe_provider,
    get_provider,
    reset_provider_cache,
)
from llm_proxy.presets import PRESETS, Preset, known_providers, resolve

__all__ = [
    "EFFORT_LEVELS",
    "PRESETS",
    "AnthropicProvider",
    "GenerationOptions",
    "LLMCompletion",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProxyError",
    "LLMRefusal",
    "OpenAICompatibleProvider",
    "Preset",
    "ProviderCapabilities",
    "StubProvider",
    "build_provider",
    "describe_provider",
    "get_provider",
    "known_providers",
    "reset_provider_cache",
    "resolve",
]
