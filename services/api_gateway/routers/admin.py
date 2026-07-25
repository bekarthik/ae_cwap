"""Operational surface for the Dead Letter Queue (mandate §2.C).

A DLQ that nobody can see is just a silent drop. These routes are what make it
"observable": list what was parked, read the precise reason, and resubmit once
the underlying contract or permission problem is fixed.
"""

from __future__ import annotations

from cwap_common.broker import get_broker
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import DeadLetter
from cwap_common.settings import get_settings
from fastapi import APIRouter, Depends, HTTPException, Query

from api_gateway.schemas import DeadLetterRecord
from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/dead-letters", response_model=list[DeadLetterRecord])
def list_dead_letters(
    principal: Principal = Depends(current_principal),
    reason_code: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[DeadLetterRecord]:
    with read_only_session() as session:
        query = session.query(DeadLetter)
        if reason_code:
            query = query.filter_by(reason_code=reason_code)
        rows = query.order_by(DeadLetter.created_at.desc()).limit(limit).all()
        return [
            DeadLetterRecord(
                id=row.id,
                queue=row.queue,
                reason_code=row.reason_code,
                reason=row.reason,
                run_id=row.run_id,
                detail=row.detail,
                created_at=row.created_at,
            )
            for row in rows
        ]


@router.post("/dead-letters/{record_id}/resubmit", status_code=202)
def resubmit(
    record_id: int, principal: Principal = Depends(current_principal)
) -> dict[str, str]:
    """Put a parked message back on the work queue.

    The payload goes back through the consumer wrapper on the next poll, so a
    message that is still invalid is simply dead-lettered again — resubmission
    can never bypass validation.
    """
    settings = get_settings()
    with unit_of_work() as session:
        record = session.get(DeadLetter, record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="dead letter not found")
        payload = record.payload
        record.resubmitted += 1

    get_broker().publish(settings.work_queue, payload)
    return {"status": "resubmitted", "queue": settings.work_queue}


@router.get("/queues")
def queue_depths(principal: Principal = Depends(current_principal)) -> dict[str, int]:
    settings = get_settings()
    broker = get_broker()
    return {
        settings.work_queue: broker.depth(settings.work_queue),
        settings.dead_letter_queue: broker.depth(settings.dead_letter_queue),
    }
