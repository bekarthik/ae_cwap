"""Choosing a model backend: detect, test, save.

The three things a person does when pointing the platform at a different model,
in the order they do them. `test` matters most: saving a configuration that
turns out to be wrong means every workflow fails at its first step with a
transport error, and the user has no way to tell a bad key from a wrong URL from
a model name their server does not have. So the configuration is exercised with
a real, tiny completion first, through the same construction path a run uses —
a configuration that tests green is exactly the one that will run.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_proxy import store
from llm_proxy.client import (
    GenerationOptions,
    LLMConfigurationError,
    LLMProxyError,
    build_provider,
    reset_provider_cache,
)
from llm_proxy.discovery import DetectedModel, detect
from llm_proxy.presets import resolve


@dataclass(frozen=True)
class ConnectionTest:
    ok: bool
    message: str
    model: str = ""
    #: What the backend turned out to support, which is often the useful part —
    #: a user picking a model for an agent needs to know it can call tools.
    supports: dict[str, bool] | None = None
    latency_ms: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "message": self.message,
            "model": self.model,
            "supports": self.supports or {},
            "latency_ms": self.latency_ms,
        }


def available_models(
    provider: str, *, base_url: str = "", api_key: str = "", tenant_id: str = ""
) -> list[DetectedModel]:
    """What this backend actually serves.

    A blank key falls back to the tenant's stored one, so "detect models" works
    after a save without the browser having to hold a credential it was never
    given back.
    """
    return detect(provider, base_url=base_url, api_key=_key_for(tenant_id, provider, api_key))


def test_connection(
    provider: str,
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    tenant_id: str = "",
) -> ConnectionTest:
    """Exercise a configuration with a real completion.

    Returns a result rather than raising: every outcome here is something the
    user needs to read and act on, and a 500 would tell them less than the
    backend's own message.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    try:
        client = build_provider(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=_key_for(tenant_id, provider, api_key),
        )
        completion = client.complete(
            "Reply with the single word: ready.",
            system="You are a connection check. Answer in one word.",
            options=GenerationOptions(max_tokens=16),
        )
    except LLMConfigurationError as exc:
        return ConnectionTest(ok=False, message=str(exc))
    except LLMProxyError as exc:
        return ConnectionTest(ok=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001 - a backend can fail in its own ways
        return ConnectionTest(ok=False, message=f"{type(exc).__name__}: {exc}")

    elapsed = int((time.perf_counter() - started) * 1000)
    capabilities = client.capabilities
    return ConnectionTest(
        ok=True,
        message=f"Connected. The model replied: {completion.text.strip()[:80] or '(nothing)'}",
        model=capabilities.model,
        supports={
            "tools": capabilities.supports_tools,
            "vision": capabilities.supports_vision,
            "thinking": capabilities.supports_thinking,
            "effort": capabilities.supports_effort,
            "temperature": capabilities.supports_temperature,
        },
        latency_ms=elapsed,
    )


def save_choice(
    tenant_id: str,
    *,
    provider: str,
    model: str = "",
    base_url: str = "",
    api_key: str | None = None,
    updated_by: str = "",
) -> store.StoredModelConfig:
    """Persist a tenant's choice and make it take effect immediately."""
    if provider.strip().lower() not in {"stub", "anthropic"} and resolve(provider) is None:
        raise LLMConfigurationError(f"unknown provider '{provider}'")

    saved = store.save(
        tenant_id,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        updated_by=updated_by,
    )
    # Providers are cached per configuration; a save that changed the model must
    # not be answered by a client built from the old one.
    reset_provider_cache(None)
    return saved


def clear_choice(tenant_id: str) -> bool:
    """Go back to whatever the deployment is configured with."""
    removed = store.clear(tenant_id)
    reset_provider_cache(None)
    return removed


def _key_for(tenant_id: str, provider: str, supplied: str) -> str:
    """A supplied key wins; otherwise the tenant's stored one for that provider.

    Scoped to the same provider deliberately — reusing an OpenAI key against a
    newly entered Together endpoint would send the credential somewhere it was
    never meant to go.
    """
    if supplied or not tenant_id:
        return supplied
    stored = store.load(tenant_id)
    if stored is None or stored.provider != provider:
        return ""
    return stored.api_key
