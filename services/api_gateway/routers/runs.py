"""Execution & Monitoring routes (Epic 4).

The "Run" button posts here and gets a run id back immediately; the browser then
opens the WebSocket and watches the run happen. Nothing about execution blocks
an HTTP request.
"""

from __future__ import annotations

import asyncio

from cwap_common.db import read_only_session
from cwap_common.logbus import log_bus
from cwap_common.models import Run, Workflow, WorkflowExecutionState
from cwap_contracts import NodeType, WorkflowGraph
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from orchestrator.runner import start_run

from api_gateway.schemas import (
    RunReport,
    RunSummary,
    StartRunRequest,
    StepReport,
)
from api_gateway.security import (
    WRITE_EXTERNAL,
    Principal,
    current_principal,
    principal_from_query_token,
)

router = APIRouter(tags=["runs"])

#: How long a WebSocket waits for an event before sending a keepalive. Without
#: this, an idle proxy will drop a connection watching a slow step.
KEEPALIVE_SECONDS = 20.0


def _needs_write_scope(graph: WorkflowGraph) -> bool:
    """A graph that can act on the outside world must declare a write scope."""
    return any(node.type is NodeType.HTTP_REQUEST for node in graph.nodes)


@router.post("/api/workflows/{workflow_id}/runs", response_model=RunSummary, status_code=202)
def create_run(
    workflow_id: str,
    request: StartRunRequest,
    principal: Principal = Depends(current_principal),
) -> RunSummary:
    graph = request.graph
    if graph is None:
        with read_only_session() as session:
            record = session.get(Workflow, workflow_id)
            if record is None or record.tenant_id != principal.tenant_id:
                raise HTTPException(status_code=404, detail="workflow not found")
            graph = WorkflowGraph.model_validate(record.graph)

    needs_write = _needs_write_scope(graph)
    if needs_write and WRITE_EXTERNAL not in principal.scopes:
        # Fail here with an explanation rather than letting the producer's
        # authorisation pre-check reject it with a generic 403.
        raise HTTPException(
            status_code=403,
            detail=(
                "this workflow contains a step that calls an external service, which "
                f"requires the '{WRITE_EXTERNAL}' scope. Ask an administrator to grant it."
            ),
        )

    handle = start_run(
        graph=graph,
        job_context=principal.job_context(needs_write=needs_write),
        inputs=request.inputs,
    )
    return _summary_for(handle.run_id, principal)


@router.get("/api/runs", response_model=list[RunSummary])
def list_runs(
    principal: Principal = Depends(current_principal),
    workflow_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[RunSummary]:
    with read_only_session() as session:
        query = session.query(Run).filter_by(tenant_id=principal.tenant_id)
        if workflow_id:
            query = query.filter_by(workflow_id=workflow_id)
        rows = query.order_by(Run.started_at.desc()).limit(limit).all()
        return [_to_summary(row) for row in rows]


@router.get("/api/runs/{run_id}", response_model=RunReport)
def get_run(run_id: str, principal: Principal = Depends(current_principal)) -> RunReport:
    """The transparent execution report: every step, its output, and the log."""
    with read_only_session() as session:
        run = session.get(Run, run_id)
        if run is None or run.tenant_id != principal.tenant_id:
            raise HTTPException(status_code=404, detail="run not found")
        summary = _to_summary(run)
        result = run.result
        steps = (
            session.query(WorkflowExecutionState)
            .filter_by(run_id=run_id)
            .order_by(WorkflowExecutionState.id)
            .all()
        )
        step_reports = [
            StepReport(
                node_id=step.node_id,
                source_service=step.source_service,
                resulting_state=step.resulting_state,
                output=step.output or {},
                derived_context=step.derived_context or {},
                duration_ms=step.duration_ms,
                completed_at=step.completed_at,
            )
            for step in steps
        ]

    logs = [event.model_dump(mode="json") for event in log_bus.history(run_id)]
    return RunReport(run=summary, result=result, steps=step_reports, logs=logs)


@router.websocket("/api/runs/{run_id}/logs")
async def stream_logs(
    websocket: WebSocket, run_id: str, token: str | None = Query(default=None)
) -> None:
    """Real-time log streaming.

    Replays history first, then tails live events. Because sequence numbers are
    monotonic per run, a client that reconnects can tell exactly where it left
    off and the feed has no gaps or duplicates.
    """
    try:
        principal = principal_from_query_token(token)
    except HTTPException:
        await websocket.close(code=4401, reason="unauthorized")
        return

    with read_only_session() as session:
        run = session.get(Run, run_id)
        if run is None or run.tenant_id != principal.tenant_id:
            await websocket.close(code=4404, reason="run not found")
            return
        terminal = run.status in {"SUCCEEDED", "FAILED"}

    await websocket.accept()
    # Subscribe before replaying history so an event emitted during replay is
    # buffered rather than lost.
    queue = log_bus.subscribe(run_id)
    try:
        last_seq = 0
        for event in log_bus.history(run_id):
            await websocket.send_json(event.model_dump(mode="json"))
            last_seq = max(last_seq, event.seq)

        if terminal:
            await websocket.close(code=1000, reason="run already finished")
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "keepalive", "run_id": run_id})
                continue

            if event.seq <= last_seq:
                continue  # already delivered during replay
            last_seq = event.seq
            await websocket.send_json(event.model_dump(mode="json"))
            if event.event in {"run.succeeded", "run.failed"}:
                await websocket.close(code=1000, reason="run finished")
                return
    except WebSocketDisconnect:
        return
    finally:
        log_bus.unsubscribe(run_id, queue)


def _summary_for(run_id: str, principal: Principal) -> RunSummary:
    with read_only_session() as session:
        run = session.get(Run, run_id)
        if run is None or run.tenant_id != principal.tenant_id:  # pragma: no cover
            raise HTTPException(status_code=404, detail="run not found")
        return _to_summary(run)


def _to_summary(run: Run) -> RunSummary:
    return RunSummary(
        run_id=run.id,
        workflow_id=run.workflow_id,
        status=run.status,
        steps_executed=run.steps_executed,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error=run.error,
    )
