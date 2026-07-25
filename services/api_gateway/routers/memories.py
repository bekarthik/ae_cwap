"""Reading and correcting memory.

Memory that a user cannot see is memory they cannot trust. These routes expose
what each scope has learned, and let a wrong lesson be deleted rather than
quietly shaping every future run.
"""

from __future__ import annotations

from cwap_contracts.v3 import MemoryEntry, MemoryKind, MemoryScope, MemoryWriteRequest
from fastapi import APIRouter, Depends, HTTPException, Query
from memory import service as memory_service
from pydantic import BaseModel, ConfigDict, Field

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/memory", tags=["memory"])


class TeachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: MemoryScope
    scope_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)
    kind: MemoryKind = MemoryKind.PREFERENCE


@router.get("", response_model=list[MemoryEntry])
def list_scope(
    scope: MemoryScope = Query(...),
    scope_id: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
    principal: Principal = Depends(current_principal),
) -> list[MemoryEntry]:
    return memory_service.list_memories(principal.tenant_id, scope, scope_id, limit)


@router.post("", response_model=MemoryEntry, status_code=201)
def teach(
    request: TeachRequest, principal: Principal = Depends(current_principal)
) -> MemoryEntry:
    """Tell an agent, a skill or a workflow something directly."""
    return memory_service.remember(
        MemoryWriteRequest(
            tenant_id=principal.tenant_id,
            scope=request.scope,
            scope_id=request.scope_id,
            kind=request.kind,
            text=request.text,
        )
    )


@router.delete("/{entry_id}", status_code=204)
def forget_entry(entry_id: str, principal: Principal = Depends(current_principal)) -> None:
    if not memory_service.forget_entry(principal.tenant_id, entry_id):
        raise HTTPException(status_code=404, detail="memory not found")


@router.delete("", status_code=200)
def forget_scope(
    scope: MemoryScope = Query(...),
    scope_id: str = Query(...),
    principal: Principal = Depends(current_principal),
) -> dict[str, int]:
    """Make an agent, skill or workflow start over."""
    return {"forgotten": memory_service.forget(principal.tenant_id, scope, scope_id)}
