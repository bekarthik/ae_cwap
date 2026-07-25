"""Skill management, including synthesis on demand."""

from __future__ import annotations

from cwap_contracts.v2 import MemoryScope, SkillDefinition, SkillProposal
from fastapi import APIRouter, Depends, HTTPException
from knowledge.service import list_handles
from memory import service as memory_service
from pydantic import BaseModel, ConfigDict, Field
from skills import registry

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/skills", tags=["skills"])


class CapabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: str = Field(min_length=3, max_length=500)
    context: str = Field(default="", max_length=2000)


class EnsureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: SkillDefinition
    created: bool


@router.get("", response_model=list[SkillDefinition])
def list_skills(principal: Principal = Depends(current_principal)) -> list[SkillDefinition]:
    registry.ensure_builtins(principal.tenant_id)
    return registry.list_all(principal.tenant_id)


@router.post("/ensure", response_model=EnsureResponse)
def ensure_capability(
    request: CapabilityRequest, principal: Principal = Depends(current_principal)
) -> EnsureResponse:
    """Find a skill for a capability, creating one if none exists.

    The created skill is declarative — a prompt, a retrieval, a transform — never
    code, so synthesis cannot widen what the platform is able to do.
    """
    handles = [item.handle for item in list_handles(principal.tenant_id)]
    skill, created = registry.ensure_capability(
        principal.tenant_id,
        request.capability,
        context=request.context,
        knowledge_handles=handles,
    )
    return EnsureResponse(skill=skill, created=created)


@router.post("", response_model=SkillDefinition, status_code=201)
def create_skill(
    proposal: SkillProposal, principal: Principal = Depends(current_principal)
) -> SkillDefinition:
    try:
        return registry.create(principal.tenant_id, proposal)
    except registry.SkillError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/{skill_id}", status_code=204)
def delete_skill(skill_id: str, principal: Principal = Depends(current_principal)) -> None:
    if not registry.delete(principal.tenant_id, skill_id):
        raise HTTPException(status_code=404, detail="skill not found")
    memory_service.forget(principal.tenant_id, MemoryScope.SKILL, skill_id)
