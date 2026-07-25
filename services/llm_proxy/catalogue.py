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
            vision=True, context=128_000, open_weights=True,
            notes="Reads images.",
        ),
        ModelCard(
            "deepseek-r1", "DeepSeek-R1 (distill)",
            thinking=True, context=128_000, open_weights=True,
            notes="Reasons before answering.",
        ),
        ModelCard(
            "mistral", "Mistral 7B",
            context=32_000, open_weights=True,
        ),
        ModelCard(
            "gemma3", "Gemma 3",
            vision=True, context=128_000, open_weights=True,
            notes="Vision-capable.",
        ),
        ModelCard(
            "qwen3", "Qwen 3",
            thinking=True, context=128_000, open_weights=True,
            notes="Strong tool calling, with a thinking mode.",
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
            thinking=True, context=128_000, open_weights=True,
        ),
        ModelCard(
            "meta-llama/Llama-Vision-Free", "Llama 3.2 11B Vision",
            vision=True, context=128_000, open_weights=True,
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
            thinking=True, context=128_000, open_weights=True,
        ),
    ),
    "deepseek": (
        ModelCard("deepseek-chat", "DeepSeek Chat", tools=True, context=64_000),
        # The one remaining `tools=False`. DeepSeek documents that the reasoner
        # does not accept function calling, so this is a fact about the API
        # rather than a guess about the weights — which is the bar an entry has
        # to clear before it may assert the negative.
        ModelCard(
            "deepseek-reasoner", "DeepSeek Reasoner",
            tools=False, thinking=True, context=64_000,
            notes="DeepSeek's API does not accept tool calls on this model.",
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
    # LM Studio serves whatever the user has loaded, so a curated list would be
    # guesswork. Detection against the running server is the answer, and current
    # LM Studio builds support native tool calling for models that have it.
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


#: Substring → (vision, thinking), for models the catalogue has not heard of.
#:
#: Note what is *not* here: tool calling. A name is not evidence a model cannot
#: call tools, and guessing that it cannot is the expensive mistake — it silently
#: routes a capable model onto the slower prompted protocol and makes the canvas
#: state something untrue. The opposite guess costs one rejected request, which
#: the provider already recovers from by downgrading permanently and retrying.
#: So tool support is only ever asserted from an explicit catalogue entry or from
#: what the server actually did. See `capabilities_for`.
#:
#: Vision and thinking are guessable for the same reason tools are assumed rather
#: than denied: the guess is recoverable. A wrong thinking guess sends one
#: reasoning parameter the server does not take, and `client._send` drops it,
#: retries and stops asking. A wrong vision guess only changes which controls the
#: canvas offers. Neither can strand a capable model on a worse path.
#:
#: Order matters — the first match wins, so put the specific before the broad.
_HINTS: tuple[tuple[str, tuple[bool, bool]], ...] = (
    ("-vl", (True, False)),
    ("vision", (True, False)),
    ("pixtral", (True, False)),
    ("llava", (True, False)),
    ("gemma", (True, False)),
    ("reasoner", (False, True)),
    ("qwq", (False, True)),
    ("thinking", (False, True)),
    ("deepseek-r1", (False, True)),
    # Tagged local builds — `qwen3:14b`, `gpt-oss:20b` — never match a catalogue
    # id exactly, and these families all ship a reasoning mode.
    ("qwen3", (False, True)),
    ("gpt-oss", (False, True)),
    ("magistral", (False, True)),
    ("o3-", (False, True)),
    ("o4-mini", (False, True)),
    ("claude", (True, True)),
    ("gpt-4", (True, False)),
    ("gpt-5", (True, True)),
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

    Tool support is **never** inferred from a name. A catalogued model reports
    what the catalogue says; anything else is assumed capable, and the provider
    finds out for certain the first time it sends a tools request — a server that
    rejects it downgrades that provider permanently and retries within the same
    call, which is the only trustworthy answer available.

    That asymmetry is deliberate. Assuming tools and being wrong costs one
    rejected request. Assuming no tools and being wrong routes a perfectly
    capable model onto the slower prompted protocol *forever*, and there is no
    later signal that would ever correct it.
    """
    card = find(provider, model)
    if card is not None:
        return card.tools, card.vision, card.thinking

    vision, thinking = _infer(model)
    return True, vision, thinking


def _infer(model: str) -> tuple[bool, bool]:
    """`(vision, thinking)` guessed from a model's name."""
    lowered = model.lower()
    for needle, flags in _HINTS:
        if needle in lowered:
            return flags
    return False, False


def is_catalogued(provider: str, model: str) -> bool:
    """Whether the tool-support answer is curated rather than assumed.

    The UI needs this to decide whether it may state a capability as fact.
    """
    return find(provider, model) is not None


def cards_for(provider: str) -> tuple[ModelCard, ...]:
    return CATALOGUE.get(provider.strip().lower(), ())
