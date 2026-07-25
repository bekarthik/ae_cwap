"""A catalogue of common models, with what each one can actually do.

Two problems this solves.

**Discovery.** Asking a non-technical user to type a provider key and a model id
is asking them to already know the answer. The picker offers the backends people
actually run — Ollama on a laptop, vLLM in a cluster, Claude, an OpenAI-compatible
proxy — and the models each one commonly serves.

**Honesty about capability.** Tool calling, vision and a reasoning mode are
properties of the *model*, not of the endpoint: Together serves both a model with
native tool calling and one without, over the same URL. The agent loop needs to
know which, because a model without tool support has to be driven through the
prompted JSON protocol instead. So the flags live per model here, and the runtime
merges them into `ProviderCapabilities`.

The catalogue is a *hint*, never a gate. An unlisted model still runs — its
capabilities are inferred from its name (`_infer`), and anything the backend then
rejects is handled at runtime by the tool-support downgrade. Nobody is blocked
from using a model because this file has not heard of it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCard:
    id: str
    label: str
    #: Native function/tool calling. False sends the agent through the prompted
    #: protocol, which works everywhere but costs a little accuracy.
    tools: bool = True
    #: Image input.
    vision: bool = False
    #: An exposed reasoning/thinking mode.
    thinking: bool = False
    #: Approximate context window, for display only.
    context: int = 0
    notes: str = ""
    #: Open weights, i.e. runnable on your own hardware.
    open_weights: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "context": self.context,
            "notes": self.notes,
            "open_weights": self.open_weights,
            "supports": {"tools": self.tools, "vision": self.vision, "thinking": self.thinking},
        }


#: Keyed by preset key. Every entry is a suggestion the picker offers; the model
#: field stays free text underneath so a private fine-tune is still one keystroke
#: away.
CATALOGUE: dict[str, tuple[ModelCard, ...]] = {
    "anthropic": (
        ModelCard(
            "claude-opus-5", "Claude Opus 5",
            tools=True, vision=True, thinking=True, context=1_000_000,
            notes="Most capable. Adaptive thinking, vision, native tools.",
        ),
        ModelCard(
            "claude-sonnet-5", "Claude Sonnet 5",
            tools=True, vision=True, thinking=True, context=1_000_000,
            notes="Balanced cost and capability.",
        ),
        ModelCard(
            "claude-haiku-4-5", "Claude Haiku 4.5",
            tools=True, vision=True, thinking=False, context=200_000,
            notes="Fastest and cheapest; good for high-volume simple steps.",
        ),
    ),
    "openai": (
        ModelCard("gpt-4o", "GPT-4o", tools=True, vision=True, context=128_000),
        ModelCard("gpt-4o-mini", "GPT-4o mini", tools=True, vision=True, context=128_000),
        ModelCard(
            "o4-mini", "o4-mini",
            tools=True, vision=True, thinking=True, context=200_000,
            notes="Reasoning model; ignores temperature.",
        ),
    ),
    "ollama": (
        ModelCard(
            "llama3.1", "Llama 3.1 8B",
            tools=True, context=128_000, open_weights=True,
            notes="Good default. `ollama pull llama3.1`",
        ),
        ModelCard(
            "qwen2.5", "Qwen 2.5 7B",
            tools=True, context=128_000, open_weights=True,
            notes="Strong tool calling for its size.",
        ),
        ModelCard(
            "llama3.2-vision", "Llama 3.2 Vision 11B",
            tools=False, vision=True, context=128_000, open_weights=True,
            notes="Reads images. No native tool calling — the prompted protocol is used.",
        ),
        ModelCard(
            "deepseek-r1", "DeepSeek-R1 (distill)",
            tools=False, thinking=True, context=128_000, open_weights=True,
            notes="Reasons before answering; no native tool calling.",
        ),
        ModelCard(
            "mistral", "Mistral 7B",
            tools=True, context=32_000, open_weights=True,
        ),
        ModelCard(
            "gemma3", "Gemma 3",
            tools=False, vision=True, context=128_000, open_weights=True,
            notes="Vision-capable; driven through the prompted tool protocol.",
        ),
    ),
    "together": (
        ModelCard(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama 3.3 70B",
            tools=True, context=128_000, open_weights=True,
        ),
        ModelCard(
            "Qwen/Qwen2.5-72B-Instruct-Turbo", "Qwen 2.5 72B",
            tools=True, context=32_000, open_weights=True,
        ),
        ModelCard(
            "deepseek-ai/DeepSeek-R1", "DeepSeek-R1",
            tools=False, thinking=True, context=128_000, open_weights=True,
        ),
        ModelCard(
            "meta-llama/Llama-Vision-Free", "Llama 3.2 11B Vision",
            tools=False, vision=True, context=128_000, open_weights=True,
        ),
    ),
    "groq": (
        ModelCard(
            "llama-3.3-70b-versatile", "Llama 3.3 70B",
            tools=True, context=128_000, open_weights=True,
        ),
        ModelCard(
            "qwen/qwen3-32b", "Qwen 3 32B",
            tools=True, thinking=True, context=128_000, open_weights=True,
        ),
    ),
    "openrouter": (
        ModelCard(
            "meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B",
            tools=True, context=128_000, open_weights=True,
        ),
        ModelCard(
            "anthropic/claude-sonnet-5", "Claude Sonnet 5",
            tools=True, vision=True, thinking=True, context=1_000_000,
        ),
        ModelCard(
            "qwen/qwen2.5-vl-72b-instruct", "Qwen 2.5 VL 72B",
            tools=True, vision=True, context=128_000, open_weights=True,
        ),
    ),
    "fireworks": (
        ModelCard(
            "accounts/fireworks/models/llama-v3p3-70b-instruct", "Llama 3.3 70B",
            tools=True, context=128_000, open_weights=True,
        ),
        ModelCard(
            "accounts/fireworks/models/deepseek-r1", "DeepSeek-R1",
            tools=False, thinking=True, context=128_000, open_weights=True,
        ),
    ),
    "deepseek": (
        ModelCard("deepseek-chat", "DeepSeek Chat", tools=True, context=64_000),
        ModelCard(
            "deepseek-reasoner", "DeepSeek Reasoner",
            tools=False, thinking=True, context=64_000,
            notes="Reasoning mode; tool calling is not supported.",
        ),
    ),
    "mistral": (
        ModelCard("mistral-large-latest", "Mistral Large", tools=True, context=128_000),
        ModelCard(
            "pixtral-large-latest", "Pixtral Large",
            tools=True, vision=True, context=128_000,
        ),
    ),
    "vllm": (),
    "lmstudio": (),
    "llamacpp": (),
    "tgi": (),
    "litellm": (),
    "openai_compatible": (),
    "stub": (
        ModelCard(
            "stub-model", "Deterministic stub",
            tools=False, context=0,
            notes="No network. Output is a stable function of the prompt.",
        ),
    ),
}


#: Substring → (tools, vision, thinking), for models the catalogue has not heard
#: of. Order matters: the first match wins, so put the specific before the broad.
_HINTS: tuple[tuple[str, tuple[bool, bool, bool]], ...] = (
    ("vision", (False, True, False)),
    ("-vl", (True, True, False)),
    ("pixtral", (True, True, False)),
    ("llava", (False, True, False)),
    ("gemma", (False, True, False)),
    ("r1", (False, False, True)),
    ("reasoner", (False, False, True)),
    ("qwq", (False, False, True)),
    ("thinking", (True, False, True)),
    ("claude", (True, True, True)),
    ("gpt-4", (True, True, False)),
    ("gpt-5", (True, True, True)),
    ("qwen", (True, False, False)),
    ("llama", (True, False, False)),
    ("mistral", (True, False, False)),
    ("mixtral", (True, False, False)),
    ("phi", (False, False, False)),
)


def find(provider: str, model: str) -> ModelCard | None:
    """The catalogue entry for a model, if there is one."""
    if not model:
        return None
    for card in CATALOGUE.get(provider.strip().lower(), ()):
        if card.id == model:
            return card
    # A model listed under a different provider still describes the same weights
    # — the same Llama on Groq and on Together has the same capabilities.
    for cards in CATALOGUE.values():
        for card in cards:
            if card.id == model:
                return card
    return None


def capabilities_for(provider: str, model: str) -> tuple[bool, bool, bool]:
    """`(tools, vision, thinking)` for a model, catalogued or not.

    An unknown model gets an optimistic guess from its name and, failing that,
    `(True, False, False)` — assume tools, since most instruction-tuned models
    released since 2024 have them, and the runtime downgrades cleanly when the
    server says otherwise. Guessing "no tools" would silently put every unlisted
    model on the slower prompted path.
    """
    card = find(provider, model)
    if card is not None:
        return card.tools, card.vision, card.thinking

    lowered = model.lower()
    for needle, flags in _HINTS:
        if needle in lowered:
            return flags
    return True, False, False


def cards_for(provider: str) -> tuple[ModelCard, ...]:
    return CATALOGUE.get(provider.strip().lower(), ())
