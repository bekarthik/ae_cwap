"""What model backend this deployment is actually running.

The canvas needs this to be honest with the user: an effort selector means
nothing against a local Llama, and a temperature slider means nothing against
current Claude models, which reject the parameter outright. Rather than show
both and silently drop one, the Inspector renders the controls the configured
backend declares it supports.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from knowledge.embeddings import describe_embedder
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
                "notes": preset.notes,
            }
            for preset in PRESETS.values()
        ],
    }
