"""Goal Intake Endpoint (Epic 1, User Story 2)."""

from __future__ import annotations

from cwap_contracts import GoalIntakeRequest
from fastapi import APIRouter, Depends
from knowledge.service import list_handles
from nlp.service import scaffold

from api_gateway.schemas import DiagnoseRequest, DiagnoseResponse
from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/diagnose", tags=["diagnosis"])


@router.post("", response_model=DiagnoseResponse)
def diagnose_goal(
    request: DiagnoseRequest, principal: Principal = Depends(current_principal)
) -> DiagnoseResponse:
    """Turn a plain-language goal into a reviewed-before-you-build draft.

    Any handles the caller did not pass explicitly are filled in from their own
    tenant's corpora, so a user who has already uploaded documents gets a
    grounded scaffold without having to name the corpus.
    """
    handles = list(request.knowledge_handles)
    if not handles:
        handles = [item.handle for item in list_handles(principal.tenant_id)]

    response = scaffold(
        GoalIntakeRequest(goal=request.goal, knowledge_handles=handles)
    )
    return DiagnoseResponse(diagnosis=response.diagnosis, graph=response.graph)
