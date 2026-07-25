"""Celery entry point for the production deployment.

The platform's own transport abstraction (`cwap_common.broker`) is what the
worker actually reads from, so Celery is a *scheduler* here rather than a
serialization layer: a Celery task simply asks the worker to drain whatever the
gateway has validated. That keeps the dual-gateway guarantee intact — a payload
still cannot reach a worker without passing schema validation and the
authorisation runtime check.

Run in production with:

    celery -A orchestrator.celery_app worker --loglevel=info
    celery -A orchestrator.celery_app beat --loglevel=info
"""

from __future__ import annotations

from cwap_common.db import init_db
from cwap_common.settings import get_settings

try:  # pragma: no cover - Celery is an optional extra
    from celery import Celery
except ImportError:  # pragma: no cover
    Celery = None  # type: ignore[assignment]


def build_app():  # pragma: no cover - exercised only in a real deployment
    if Celery is None:
        raise RuntimeError(
            "celery is not installed; install the 'queue' extra: pip install -e '.[queue]'"
        )

    settings = get_settings()
    app = Celery(
        "cwap",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    app.conf.update(
        task_acks_late=True,  # a crashed worker's job is redelivered, not lost
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # long steps; do not hoard messages
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        beat_schedule={
            "drain-workflow-queue": {
                "task": "cwap.drain_workflow_queue",
                "schedule": 1.0,
            }
        },
    )

    @app.task(name="cwap.drain_workflow_queue")
    def drain_workflow_queue() -> int:
        from orchestrator.runner import Worker  # noqa: PLC0415 - avoid import cycle

        init_db()
        return Worker().drain()

    return app


app = build_app() if Celery is not None else None  # pragma: no cover
