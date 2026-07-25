"""Known model backends.

Almost every model server in use today — local or hosted, open-weight or
proprietary — speaks the OpenAI chat-completions wire format. So the platform
needs exactly two client implementations: Anthropic's SDK, and one HTTP client
for everything else. The difference between "run Llama on my laptop" and "run
Qwen on Together" is then a base URL, which is what this table holds.

Adding a backend is an entry here, not a new provider class.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    base_url: str
    #: A sensible model to start with. Always overridable via CWAP_LLM_MODEL.
    default_model: str
    #: Hosted services need a key; something on localhost does not.
    requires_key: bool
    #: Default embedding model for the same endpoint, when it serves one.
    default_embedding_model: str = ""
    notes: str = ""


#: `local` presets need no credentials — the point is that an open-weight model
#: on your own machine is a first-class deployment, not a fallback.
PRESETS: dict[str, Preset] = {
    "ollama": Preset(
        key="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        default_model="llama3.1",
        requires_key=False,
        default_embedding_model="nomic-embed-text",
        notes="Ollama serves an OpenAI-compatible API on /v1. Pull a model first: `ollama pull llama3.1`.",
    ),
    "vllm": Preset(
        key="vllm",
        label="vLLM (self-hosted)",
        base_url="http://localhost:8000/v1",
        default_model="",
        requires_key=False,
        notes="Set CWAP_LLM_MODEL to the model id vLLM was started with.",
    ),
    "lmstudio": Preset(
        key="lmstudio",
        label="LM Studio (local)",
        base_url="http://localhost:1234/v1",
        default_model="",
        requires_key=False,
        notes="Start the local server from LM Studio's Developer tab.",
    ),
    "llamacpp": Preset(
        key="llamacpp",
        label="llama.cpp server (local)",
        base_url="http://localhost:8080/v1",
        default_model="",
        requires_key=False,
        notes="`llama-server` exposes an OpenAI-compatible API; the model id is ignored.",
    ),
    "tgi": Preset(
        key="tgi",
        label="Text Generation Inference (self-hosted)",
        base_url="http://localhost:8080/v1",
        default_model="",
        requires_key=False,
        notes="Hugging Face TGI serves /v1/chat/completions from version 1.4.",
    ),
    "litellm": Preset(
        key="litellm",
        label="LiteLLM proxy",
        base_url="http://localhost:4000/v1",
        default_model="",
        requires_key=False,
        notes="Fronts 100+ providers behind one OpenAI-compatible endpoint.",
    ),
    "openai": Preset(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        requires_key=True,
        default_embedding_model="text-embedding-3-small",
    ),
    "together": Preset(
        key="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        requires_key=True,
        default_embedding_model="BAAI/bge-base-en-v1.5",
    ),
    "groq": Preset(
        key="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        requires_key=True,
    ),
    "openrouter": Preset(
        key="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct",
        requires_key=True,
    ),
    "fireworks": Preset(
        key="fireworks",
        label="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        requires_key=True,
    ),
    "deepseek": Preset(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        requires_key=True,
    ),
    "mistral": Preset(
        key="mistral",
        label="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        requires_key=True,
        default_embedding_model="mistral-embed",
    ),
    # Generic escape hatch: anything else that speaks the same wire format.
    "openai_compatible": Preset(
        key="openai_compatible",
        label="OpenAI-compatible endpoint",
        base_url="",
        default_model="",
        requires_key=False,
        notes="Set CWAP_LLM_BASE_URL to any server exposing /chat/completions.",
    ),
}

#: Providers with their own client rather than the shared HTTP one.
NATIVE_PROVIDERS = frozenset({"stub", "anthropic"})


def resolve(key: str) -> Preset | None:
    return PRESETS.get(key.strip().lower())


def is_openai_compatible(key: str) -> bool:
    return key.strip().lower() in PRESETS


def known_providers() -> list[str]:
    return sorted(NATIVE_PROVIDERS | set(PRESETS))
