"""What model backend this deployment is running, and what else it could run.

The canvas needs this to be honest with the user: an effort selector means
nothing against a local Llama, a temperature slider means nothing against current
Claude models, which reject the parameter outright, and an agent node against a
model with no tool calling behaves differently from one that has it. Rather than
show every control and silently drop the ones that do not apply, the Inspector
renders what the configured backend declares it supports.

`available_providers` carries the model catalogue with it, so the picker can
offer "Ollama → Llama 3.2 Vision (reads images, no native tools)" instead of two
free-text boxes and a link to someone else's documentation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from knowledge.embeddings import describe_embedder
from llm_proxy.catalogue import cards_for
from llm_proxy.client import EFFORT_LEVELS, describe_provider
from llm_proxy.presets import PRESETS

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.get("")
def runtime_info(principal: Principal = Depends(current_principal)) -> dict[str, object]:
    """Model and embedding configuration.

    Authenticated because it names internal hosts. It never returns credentials.
    """
    return {
        "llm": describe_provider(),
        "embeddings": describe_embedder(),
        "effort_levels": list(EFFORT_LEVELS),
        "available_providers": [
            {
                "key": preset.key,
                "label": preset.label,
                "requires_key": preset.requires_key,
                "local": preset.base_url.startswith("http://localhost"),
                "base_url": preset.base_url,
                "default_model": preset.default_model,
                "notes": preset.notes,
                "models": [card.as_dict() for card in cards_for(preset.key)],
            }
            for preset in PRESETS.values()
        ],
    }
