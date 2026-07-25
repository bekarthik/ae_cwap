"""Ask a backend what models it actually has.

The catalogue in `catalogue.py` is a curated guess — useful for showing what a
provider commonly serves, useless for knowing what *this* server has loaded.
Someone running Ollama has pulled three specific models; someone on OpenRouter
has access to hundreds. Making them type the id from memory is the thing the
picker exists to avoid.

Almost every backend answers `GET /v1/models`, so one probe covers all of them.
Anthropic has its own list endpoint through the SDK. The results are merged with
the catalogue so a model that is both detected *and* known shows its real
capabilities, while a detected-but-unknown model still appears — inferred from
its name — rather than being hidden because this file has not heard of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from cwap_common.settings import get_settings

from llm_proxy.catalogue import capabilities_for, find
from llm_proxy.client import LLMConfigurationError, LLMProxyError
from llm_proxy.presets import resolve


@dataclass(frozen=True)
class DetectedModel:
    id: str
    label: str
    tools: bool
    vision: bool
    thinking: bool
    #: True when the catalogue knows this model, so the flags are curated rather
    #: than guessed from its name.
    known: bool
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "known": self.known,
            "notes": self.notes,
            "supports": {"tools": self.tools, "vision": self.vision, "thinking": self.thinking},
        }


def detect(
    provider: str, *, base_url: str = "", api_key: str = "", timeout: int = 15
) -> list[DetectedModel]:
    """List the models a backend reports. Raises `LLMProxyError` if it cannot.

    Failure is reported rather than swallowed: "we could not reach your server"
    is the answer the user needs when they have mistyped a URL, and an empty list
    would read as "your server has no models".
    """
    key = provider.strip().lower()
    if key == "stub":
        return [_describe("stub-model", "stub")]
    if key == "anthropic":
        return _detect_anthropic(api_key=api_key, timeout=timeout)
    return _detect_openai_compatible(key, base_url=base_url, api_key=api_key, timeout=timeout)


def _detect_openai_compatible(
    provider: str, *, base_url: str, api_key: str, timeout: int
) -> list[DetectedModel]:
    import httpx  # noqa: PLC0415

    preset = resolve(provider)
    if preset is None:
        raise LLMConfigurationError(f"unknown provider '{provider}'")

    url = (base_url or preset.base_url).rstrip("/")
    if not url:
        raise LLMConfigurationError(
            f"{preset.label} needs an endpoint before its models can be listed"
        )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{url}/models", headers=headers)
    except httpx.ConnectError as exc:
        raise LLMProxyError(
            f"could not reach {url} — is the server running? ({exc})"
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMProxyError(f"{url} did not respond within {timeout}s") from exc

    if response.status_code == 401 or response.status_code == 403:
        raise LLMProxyError(f"{preset.label} rejected the API key")
    if response.status_code == 404:
        raise LLMProxyError(
            f"{url} has no /models endpoint. It may still work for completions — "
            "enter the model id directly."
        )
    if response.status_code >= 400:
        raise LLMProxyError(f"{url} returned {response.status_code}: {response.text[:200]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMProxyError(f"{url} returned something that is not JSON") from exc

    # OpenAI shape is `{"data": [{"id": ...}]}`; a few servers return a bare list.
    entries = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise LLMProxyError(f"{url} returned an unexpected model list shape")

    ids = [
        str(entry.get("id") or entry.get("name") or "")
        for entry in entries
        if isinstance(entry, dict)
    ]
    return _sorted([_describe(model_id, provider) for model_id in ids if model_id])


def _detect_anthropic(*, api_key: str, timeout: int) -> list[DetectedModel]:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise LLMConfigurationError(
            "the 'anthropic' package is required to list Claude models; "
            "install it with: pip install -e '.[llm]'"
        ) from exc

    settings = get_settings()
    resolved = api_key or settings.anthropic_api_key
    client = (
        anthropic.Anthropic(api_key=resolved, timeout=timeout)
        if resolved
        else anthropic.Anthropic(timeout=timeout)
    )
    try:
        listing = client.models.list(limit=100)
    except Exception as exc:  # noqa: BLE001 - SDK raises a family of errors
        raise LLMProxyError(f"could not list Claude models: {exc}") from exc

    return _sorted(
        [_describe(item.id, "anthropic", label=getattr(item, "display_name", "")) for item in listing.data]
    )


def _describe(model_id: str, provider: str, *, label: str = "") -> DetectedModel:
    card = find(provider, model_id)
    tools, vision, thinking = capabilities_for(provider, model_id)
    return DetectedModel(
        id=model_id,
        label=label or (card.label if card else model_id),
        tools=tools,
        vision=vision,
        thinking=thinking,
        known=card is not None,
        notes=card.notes if card else "",
    )


def _sorted(models: list[DetectedModel]) -> list[DetectedModel]:
    """Catalogued models first, then alphabetically.

    A server may report hundreds; the ones the platform can say something
    concrete about are the ones worth seeing first.
    """
    return sorted(models, key=lambda m: (not m.known, m.id))
