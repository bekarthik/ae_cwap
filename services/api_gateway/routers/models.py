"""Choosing a model backend from the browser.

Three verbs, in the order a person uses them: detect what an endpoint serves,
test that the configuration works, save it. Saving takes effect on the next step
executed — no restart, which is the point.

Credentials go in and never come out. Every response reports *whether* a key is
configured; none reports the key.
"""

from __future__ import annotations

from cwap_common.settings import get_settings
from fastapi import APIRouter, Depends, HTTPException
from llm_proxy import service, store
from llm_proxy.catalogue import cards_for
from llm_proxy.client import LLMConfigurationError, LLMProxyError, describe_provider
from llm_proxy.presets import PRESETS
from pydantic import BaseModel, ConfigDict, Field

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/models", tags=["models"])


class EndpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    base_url: str = Field(default="", max_length=512)
    #: Blank means "use the key already stored for this provider", so the browser
    #: never has to hold a credential it was not given.
    api_key: str = Field(default="", max_length=512)


class TestRequest(EndpointRequest):
    model: str = Field(default="", max_length=256)
    #: 0 means "use the deployment default". Bounded because a request that can
    #: be told to wait forever is a way to pin a worker.
    timeout_seconds: int = Field(default=0, ge=0, le=3600)


class SaveRequest(EndpointRequest):
    model: str = Field(default="", max_length=256)
    timeout_seconds: int = Field(default=0, ge=0, le=3600)
    #: `null` keeps the stored key; `""` clears it. The distinction is what lets
    #: a user change model without re-entering their credential.
    api_key: str | None = Field(default=None, max_length=512)


@router.get("")
def current_configuration(
    principal: Principal = Depends(current_principal),
) -> dict[str, object]:
    """What this tenant runs on now, and everything it could run on."""
    stored = store.load(principal.tenant_id)
    settings = get_settings()

    return {
        "active": describe_provider(),
        "source": "tenant" if stored else "deployment",
        "stored": stored.redacted() if stored else None,
        "allow_custom_endpoints": settings.allow_custom_model_endpoints,
        # The floor a tenant setting of 0 falls back to, so the field can show
        # what is actually in effect rather than an empty box.
        "default_timeout_seconds": settings.llm_timeout_seconds,
        "providers": [
            {
                "key": preset.key,
                "label": preset.label,
                "requires_key": preset.requires_key,
                "local": preset.base_url.startswith("http://localhost"),
                "base_url": preset.base_url,
                "default_model": preset.default_model,
                "notes": preset.notes,
                # The curated list, shown before anything is detected so the
                # picker is useful against a server that is not running yet.
                "models": [card.as_dict() for card in cards_for(preset.key)],
            }
            for preset in PRESETS.values()
        ],
    }


@router.post("/detect")
def detect_models(
    request: EndpointRequest, principal: Principal = Depends(current_principal)
) -> dict[str, object]:
    """Ask the endpoint what models it has, instead of asking the user."""
    _guard_custom_endpoint(request.base_url)
    try:
        found = service.available_models(
            request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            tenant_id=principal.tenant_id,
        )
    except LLMProxyError as exc:
        # Data, not a 500: "could not reach your server" is the answer, and the
        # UI shows it next to the field that caused it.
        return {"ok": False, "error": str(exc), "models": []}

    return {"ok": True, "models": [model.as_dict() for model in found]}


@router.post("/test")
def test_configuration(
    request: TestRequest, principal: Principal = Depends(current_principal)
) -> dict[str, object]:
    """Run one real completion through the configuration before it is saved."""
    _guard_custom_endpoint(request.base_url)
    return service.test_connection(
        request.provider,
        model=request.model,
        base_url=request.base_url,
        api_key=request.api_key,
        timeout_seconds=request.timeout_seconds,
        tenant_id=principal.tenant_id,
    ).as_dict()


@router.put("")
def save_configuration(
    request: SaveRequest, principal: Principal = Depends(current_principal)
) -> dict[str, object]:
    """Point this tenant at a model. Effective on the next step executed."""
    _guard_custom_endpoint(request.base_url)
    try:
        saved = service.save_choice(
            principal.tenant_id,
            provider=request.provider,
            model=request.model,
            base_url=request.base_url,
            api_key=request.api_key,
            timeout_seconds=request.timeout_seconds,
            updated_by=principal.user_id,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"stored": saved.redacted(), "active": describe_provider()}


@router.delete("", status_code=200)
def clear_configuration(
    principal: Principal = Depends(current_principal),
) -> dict[str, object]:
    """Go back to the deployment default."""
    return {"cleared": service.clear_choice(principal.tenant_id)}


def _guard_custom_endpoint(base_url: str) -> None:
    """A tenant-supplied base URL is a request this server will make.

    Usually that is the whole point — pointing at your own Ollama box. But it is
    also an SSRF surface, so a deployment that does not want tenants naming
    arbitrary endpoints can switch it off and keep the preset URLs only.
    """
    if base_url and not get_settings().allow_custom_model_endpoints:
        raise HTTPException(
            status_code=403,
            detail=(
                "custom model endpoints are disabled on this deployment; choose a "
                "listed provider, or ask an administrator to set "
                "CWAP_ALLOW_CUSTOM_MODEL_ENDPOINTS=true"
            ),
        )
