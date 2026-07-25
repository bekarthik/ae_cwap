"""Workflow templates.

Two endpoints: what is on offer, and make me a copy. Instantiation creates the
tenant's own agents and skills, so a copied template is editable without
changing anything for anyone else.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from templates import service

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/templates", tags=["templates"])


@router.get("", response_model=list[service.TemplateSummary])
def list_templates(
    principal: Principal = Depends(current_principal),
) -> list[service.TemplateSummary]:
    """Every template, with prerequisites checked against this workspace."""
    return service.catalogue(principal.tenant_id)


@router.post("/{key}", response_model=service.InstantiatedTemplate)
def use_template(
    key: str, principal: Principal = Depends(current_principal)
) -> service.InstantiatedTemplate:
    """Create this workspace's own copy of a template."""
    try:
        return service.instantiate(key, principal.tenant_id)
    except service.TemplateNotFound as exc:
        raise HTTPException(status_code=404, detail=f"no template '{key}'") from exc
