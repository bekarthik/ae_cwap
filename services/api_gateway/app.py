"""API Gateway — the single entry point the browser talks to.

Responsibilities kept here and nowhere else: authentication, CORS, uniform error
translation, and routing to the service that owns each concern. It contains no
workflow logic of its own.

Error translation is the interesting part. A `ContractViolation` anywhere in the
stack becomes a `400 Bad Request: Contract Violation` with a machine-readable
`reason_code`, and an `AuthorizationFailure` becomes a `403` — exactly as the
mandate specifies, without every route having to remember to do it.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from cwap_common.db import init_db
from cwap_common.diagnostics import configure_logging
from cwap_common.logbus import build_relay, log_bus
from cwap_common.settings import get_settings
from cwap_contracts import AuthorizationFailure, ContractViolation, CwapContractError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llm_proxy.store import acting_for

from api_gateway.routers import ALL_ROUTERS
from api_gateway.security import bootstrap_demo_user

logger = logging.getLogger("cwap.gateway")

API_TITLE = "AI Cognitive Workflow Platform"
API_VERSION = "0.1.0"


#: Both spellings of the local dev frontend. A browser treats them as distinct
#: origins, and getting a CORS error because you typed 127.0.0.1 instead of
#: localhost is a pointless five minutes for anyone setting the project up.
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def _allowed_origins() -> list[str]:
    raw = os.environ.get("CWAP_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class InlineWorker:
    """Runs the execution worker on a background thread inside the API process.

    This is what lets the whole platform run as one command in development. In a
    Redis deployment the worker is a separate process and this stays disabled,
    which is why it is a setting rather than a hardcoded convenience.
    """

    def __init__(self, poll_seconds: float = 0.25) -> None:
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        from orchestrator.runner import Worker  # noqa: PLC0415 - avoid import cycle

        worker = Worker()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    worker.poll(timeout=self._poll_seconds)
                except Exception:  # pragma: no cover - a worker must not die
                    logger.exception("inline worker iteration failed")

        self._thread = threading.Thread(target=loop, name="cwap-inline-worker", daemon=True)
        self._thread.start()
        logger.info("inline worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before anything else, so the platform's own diagnostics reach the
    # container log. Without this, everything below WARNING was dropped — a
    # failing MCP connection wrote down exactly what it tried and what answered,
    # into a logger with no handler, while the operator debugged against
    # uvicorn's access lines.
    configure_logging()
    init_db()
    bootstrap_demo_user()

    # A build fingerprint as much as a status line. Whether the running image
    # actually contains the connector's diagnostic checks has been a genuine
    # question during debugging ("pulled the latest code" and a reused Docker
    # layer look identical from the browser), and an old image cannot print
    # this line.
    from mcp_connect import client as mcp_client  # noqa: PLC0415 - avoid cycle at import

    logger.info(
        "MCP connector ready — pre-flight on; reach %.0fs, handshake+listing %.0fs, "
        "tool call %.0fs",
        mcp_client.REACH_TIMEOUT,
        mcp_client.CONNECT_TIMEOUT,
        mcp_client.CALL_TIMEOUT,
    )

    # A browser watching a run holds its WebSocket to *this* process, while the
    # events are emitted wherever the worker runs. In a Redis deployment that is
    # another container, so without this the live feed shows nothing at all.
    log_bus.attach_relay(build_relay())

    worker: InlineWorker | None = None
    if get_settings().inline_worker:
        worker = InlineWorker()
        worker.start()

    try:
        yield
    finally:
        if worker is not None:
            worker.stop()
        log_bus.attach_relay(None)


def _tenant_of(request: Request) -> str:
    """The tenant on the request's bearer token, or "" if there isn't one."""
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return ""
    try:
        from api_gateway.security import decode_token  # noqa: PLC0415 - avoid a cycle

        return decode_token(header.split(" ", 1)[1]).tenant_id
    except Exception:  # noqa: BLE001 - an unauthenticated request is not an error here
        return ""


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "Visual, no-code builder for multi-step AI agent workflows. "
            "Every inter-service payload is governed by the versioned contract "
            "registry in `cwap_contracts`."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _scope_model_config_to_the_caller(request: Request, call_next):
        """Resolve the caller's chosen model backend for the whole request.

        Set here rather than in the auth dependency because a sync endpoint runs
        in a worker thread: a context variable set inside a dependency is not
        reliably visible to it, while middleware wraps the entire call. This does
        no authentication of its own — the routers' dependency remains the only
        thing that grants access — it only reads which tenant is asking, so a
        forged token buys nothing beyond a model configuration that will not
        decrypt into a usable key.
        """
        with acting_for(_tenant_of(request)):
            return await call_next(request)

    for router in ALL_ROUTERS:
        app.include_router(router)

    @app.exception_handler(ContractViolation)
    async def _contract_violation(_request: Request, exc: ContractViolation) -> JSONResponse:
        logger.warning("contract violation: %s", exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": "Contract Violation", **exc.to_dict()},
        )

    @app.exception_handler(AuthorizationFailure)
    async def _authorization_failure(
        _request: Request, exc: AuthorizationFailure
    ) -> JSONResponse:
        logger.warning("authorization failure: %s", exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": "Authorization Failure", **exc.to_dict()},
        )

    @app.exception_handler(CwapContractError)
    async def _contract_error(_request: Request, exc: CwapContractError) -> JSONResponse:
        logger.error("contract layer error: %s", exc.message)
        return JSONResponse(
            status_code=exc.http_status, content={"error": "Platform Error", **exc.to_dict()}
        )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        settings = get_settings()
        return {
            "status": "ok",
            "version": API_VERSION,
            "broker": settings.broker_backend,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_provider": settings.embedding_provider,
            "inline_worker": settings.inline_worker,
        }

    @app.get("/api/contracts", tags=["meta"])
    def contracts() -> dict[str, str]:
        """Expose the live contract fingerprints.

        Useful in an incident: it answers "which contract version is this
        deployment actually running?" without shelling into the container.
        """
        from cwap_contracts.registry import lock_snapshot  # noqa: PLC0415

        return lock_snapshot()

    return app


app = create_app()
