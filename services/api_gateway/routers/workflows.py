"""Workflow CRUD — the State Management DB behind the canvas."""

from __future__ import annotations

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import Workflow
from cwap_contracts.v4 import WorkflowGraph
from fastapi import APIRouter, Depends, HTTPException, status

from api_gateway.schemas import (
    SaveWorkflowRequest,
    WorkflowDetail,
    WorkflowSummary,
)
from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.get("", response_model=list[WorkflowSummary])
def list_workflows(principal: Principal = Depends(current_principal)) -> list[WorkflowSummary]:
    with read_only_session() as session:
        rows = (
            session.query(Workflow)
            .filter_by(tenant_id=principal.tenant_id)
            .order_by(Workflow.updated_at.desc())
            .all()
        )
        return [
            WorkflowSummary(
                id=row.id,
                name=row.name,
                version=row.version,
                node_count=len((row.graph or {}).get("nodes", [])),
                updated_at=row.updated_at,
            )
            for row in rows
        ]


@router.put("/{workflow_id}", response_model=WorkflowDetail)
def save_workflow(
    workflow_id: str,
    request: SaveWorkflowRequest,
    principal: Principal = Depends(current_principal),
) -> WorkflowDetail:
    """Create or update a workflow.

    The graph has already been structurally validated by the time it gets here —
    `WorkflowGraph` rejects cycles, dangling edges and malformed branches during
    request parsing, so an unexecutable graph fails to *save* rather than
    failing later in front of the user.
    """
    if request.graph.id != workflow_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"graph id '{request.graph.id}' does not match path id '{workflow_id}'",
        )

    with unit_of_work() as session:
        existing = session.get(Workflow, workflow_id)
        if existing is None:
            record = Workflow(
                id=workflow_id,
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                name=request.graph.name,
                version=1,
                graph=request.graph.model_dump(mode="json"),
            )
            session.add(record)
            session.flush()
        else:
            _assert_owned(existing, principal)
            existing.name = request.graph.name
            existing.version += 1
            existing.graph = request.graph.model_dump(mode="json")
            record = existing
            session.flush()

        return WorkflowDetail(
            id=record.id,
            name=record.name,
            version=record.version,
            graph=WorkflowGraph.model_validate(record.graph),
            updated_at=record.updated_at,
        )


@router.get("/{workflow_id}", response_model=WorkflowDetail)
def get_workflow(
    workflow_id: str, principal: Principal = Depends(current_principal)
) -> WorkflowDetail:
    with read_only_session() as session:
        record = session.get(Workflow, workflow_id)
        if record is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        _assert_owned(record, principal)
        return WorkflowDetail(
            id=record.id,
            name=record.name,
            version=record.version,
            graph=WorkflowGraph.model_validate(record.graph),
            updated_at=record.updated_at,
        )


@router.delete("/{workflow_id}", status_code=204)
def delete_workflow(
    workflow_id: str, principal: Principal = Depends(current_principal)
) -> None:
    with unit_of_work() as session:
        record = session.get(Workflow, workflow_id)
        if record is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        _assert_owned(record, principal)
        session.delete(record)


def _assert_owned(record: Workflow, principal: Principal) -> None:
    """404 rather than 403 on a cross-tenant read.

    Returning 403 would confirm the workflow exists, which leaks the id space
    across tenants.
    """
    if record.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=404, detail="workflow not found")
