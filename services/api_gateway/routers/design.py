"""The design conversation (Epic 1).

Two endpoints on one flow. `POST /api/design` runs a turn: with no answers it
returns the questions it needs; with answers it returns the finished design —
agents, their skills, and a runnable graph. `POST /api/design/direct` skips
straight to a design for callers that want the old one-shot behaviour.
"""

from __future__ import annotations

from cwap_contracts.v2 import DesignRequest
from design.service import design
from fastapi import APIRouter, Depends
from knowledge.service import list_handles

from api_gateway.schemas import DesignGoalRequest, DesignGoalResponse
from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/design", tags=["design"])


def _run(request: DesignGoalRequest, principal: Principal, *, force: bool) -> DesignGoalResponse:
    # Corpora the caller did not name explicitly are filled in from their own
    # tenant, so someone who has already uploaded documents gets a grounded
    # design without having to name the corpus.
    handles = request.knowledge_handles or [
        item.handle for item in list_handles(principal.tenant_id)
    ]

    response = design(
        DesignRequest(
            goal=request.goal,
            answers=request.answers,
            knowledge_handles=handles,
            skip_questions=request.skip_questions or force,
        ),
        principal.tenant_id,
    )
    return DesignGoalResponse(**response.model_dump())


@router.post("", response_model=DesignGoalResponse)
def design_workflow(
    request: DesignGoalRequest, principal: Principal = Depends(current_principal)
) -> DesignGoalResponse:
    """One turn: questions, or the finished design once they are answered."""
    return _run(request, principal, force=False)


@router.post("/direct", response_model=DesignGoalResponse)
def design_immediately(
    request: DesignGoalRequest, principal: Principal = Depends(current_principal)
) -> DesignGoalResponse:
    """Design from defaults without asking anything."""
    return _run(request, principal, force=True)
