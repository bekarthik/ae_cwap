"""Agent management.

Agents are stored objects, so they are listable, editable and inspectable. That
is the point: an agent you can only see inside one workflow cannot accumulate
memory or be improved once and used everywhere.
"""

from __future__ import annotations

from agents import registry
from cwap_contracts.v2 import AgentDefinition, AgentMemoryConfig, MemoryScope
from fastapi import APIRouter, Depends, HTTPException
from memory import service as memory_service
from pydantic import BaseModel, ConfigDict, Field

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=500)
    objective: str = Field(default="", max_length=2000)
    instructions: str = Field(default="", max_length=4000)
    skill_ids: list[str] = Field(default_factory=list, max_length=24)
    max_iterations: int = Field(default=6, ge=1, le=30)
    model_override: str | None = None
    memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)


@router.get("", response_model=list[AgentDefinition])
def list_agents(principal: Principal = Depends(current_principal)) -> list[AgentDefinition]:
    return registry.list_all(principal.tenant_id)


@router.post("", response_model=AgentDefinition, status_code=201)
def create_agent(
    request: AgentUpsert, principal: Principal = Depends(current_principal)
) -> AgentDefinition:
    return registry.create(principal.tenant_id, **request.model_dump())


@router.get("/{agent_id}", response_model=AgentDefinition)
def get_agent(
    agent_id: str, principal: Principal = Depends(current_principal)
) -> AgentDefinition:
    try:
        return registry.get(principal.tenant_id, agent_id)
    except registry.AgentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{agent_id}", response_model=AgentDefinition)
def update_agent(
    agent_id: str, request: AgentUpsert, principal: Principal = Depends(current_principal)
) -> AgentDefinition:
    try:
        return registry.update(principal.tenant_id, agent_id, **request.model_dump())
    except registry.AgentNotFound as exc:
        raise HTTPException(status_code=404, detail="agent not found") from exc


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: str, principal: Principal = Depends(current_principal)) -> None:
    if not registry.delete(principal.tenant_id, agent_id):
        raise HTTPException(status_code=404, detail="agent not found")
    # An agent's memory has no meaning without the agent.
    memory_service.forget(principal.tenant_id, MemoryScope.AGENT, agent_id)
