"""A run must never look stuck.

Reported against a Docker deployment with a real LM Studio model: the first
agent ran — its output was visible in LM Studio — and the canvas showed nothing,
with the run stuck at PENDING.

Three separate defects produce that one symptom, and each is worth closing on its
own because each hides the others:

1. **An unexpected exception left the run PENDING.** The worker caught
   `NodeExecutionError` and `BindingError` and nothing else, so anything else
   propagated out of `process()`, killed the standalone worker's loop, and left a
   run that would never finish and never say why.
2. **The live log stream is in-process.** `docker-compose` runs the worker as its
   own container, so events emitted there could never reach a browser holding a
   WebSocket to the gateway. The headline "watch it run" feature was broken in
   the topology we ship.
3. **The canvas trusted the stream alone.** With no other source of truth, a run
   that finished perfectly still displayed as PENDING forever.
"""

from __future__ import annotations

import pytest
from conftest import make_linear_graph
from cwap_common.db import read_only_session
from cwap_common.models import Run
from cwap_contracts.v3 import NodeType
from orchestrator import executors
from orchestrator.runner import RUN_FAILED, RUN_SUCCEEDED, Worker, start_run


def run_status(run_id: str) -> tuple[str, str | None]:
    with read_only_session() as session:
        run = session.get(Run, run_id)
        return run.status, run.error


class TestAnUnexpectedFailureIsReported:
    @pytest.mark.parametrize(
        "boom",
        [
            RuntimeError("the model client blew up in a way nobody predicted"),
            ValueError("something returned a shape we did not expect"),
            KeyError("missing"),
            TypeError("NoneType is not subscriptable"),
        ],
        ids=["RuntimeError", "ValueError", "KeyError", "TypeError"],
    )
    def test_the_run_fails_with_the_reason_instead_of_hanging(
        self, authorized_user, monkeypatch, boom
    ):
        """A run that will never finish must say so. PENDING forever is the one
        outcome a user cannot act on."""

        def explode(request):
            raise boom

        monkeypatch.setitem(executors.EXECUTORS, NodeType.LLM, explode)

        handle = start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "x"}
        )
        Worker().drain()

        status, error = run_status(handle.run_id)
        assert status == RUN_FAILED
        assert error and type(boom).__name__ in error

    def test_the_worker_survives_to_handle_the_next_job(
        self, authorized_user, monkeypatch
    ):
        """The standalone worker had no guard around its loop, so one unexpected
        exception took the whole container down and every queued run with it."""
        calls = {"n": 0}
        original = executors.EXECUTORS[NodeType.LLM]

        def explode_once(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first job detonates")
            return original(request)

        monkeypatch.setitem(executors.EXECUTORS, NodeType.LLM, explode_once)

        doomed = start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "x"}
        )
        healthy = start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "y"}
        )

        worker = Worker()
        worker.drain()

        assert run_status(doomed.run_id)[0] == RUN_FAILED
        assert run_status(healthy.run_id)[0] == RUN_SUCCEEDED

    def test_the_failure_reaches_the_log_feed(self, authorized_user, monkeypatch):
        """So the reason is in the run report, not only in a container log."""

        def explode(request):
            raise RuntimeError("the model client blew up")

        monkeypatch.setitem(executors.EXECUTORS, NodeType.LLM, explode)

        handle = start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "x"}
        )
        Worker().drain()

        from cwap_common.logbus import log_bus

        events = log_bus.history(handle.run_id)
        assert any(event.event == "run.failed" for event in events)
        assert any("blew up" in event.message for event in events)


class TestTheStreamCrossesProcesses:
    """The worker and the gateway are separate containers in the shipped
    compose file, so an in-process fan-out reaches nobody."""

    def test_a_redis_deployment_gets_a_cross_process_bus(self, monkeypatch):
        from cwap_common.logbus import build_relay

        monkeypatch.setenv("CWAP_BROKER", "redis")
        from cwap_common.settings import reset_settings_cache

        reset_settings_cache()

        relay = build_relay()
        assert relay is not None, (
            "with a separate worker process, events must travel out of band or "
            "the browser never sees them"
        )

    def test_the_in_memory_deployment_needs_no_relay(self, monkeypatch):
        """One process, one bus — a relay would be pure overhead."""
        from cwap_common.logbus import build_relay

        monkeypatch.setenv("CWAP_BROKER", "memory")
        from cwap_common.settings import reset_settings_cache

        reset_settings_cache()
        assert build_relay() is None

    def test_events_published_elsewhere_reach_local_subscribers(self):
        """What a relay delivers: an event this process never emitted."""
        import asyncio

        from cwap_common.logbus import LogBus
        from cwap_contracts.v3 import LogEvent

        bus = LogBus()

        async def exercise():
            queue = bus.subscribe("run_1")
            bus.deliver(
                LogEvent(run_id="run_1", seq=7, event="step.completed", message="from afar")
            )
            return await asyncio.wait_for(queue.get(), timeout=2)

        received = asyncio.run(exercise())
        assert received.message == "from afar"

    def test_delivering_a_remote_event_does_not_write_it_again(self):
        """The process that emitted it already persisted it; writing again would
        double every line in the run report."""
        import asyncio

        from cwap_common.logbus import LogBus
        from cwap_common.models import RunLog
        from cwap_contracts.v3 import LogEvent

        bus = LogBus()

        async def exercise():
            bus.subscribe("run_1")
            bus.deliver(LogEvent(run_id="run_1", seq=1, event="x", message="already stored"))

        asyncio.run(exercise())

        with read_only_session() as session:
            assert session.query(RunLog).count() == 0


class TestTheRunReportIsTheSourceOfTruth:
    def test_a_finished_run_reports_its_status_without_the_stream(
        self, authorized_user, client, auth
    ):
        """The canvas can always ask. A run that completed must never read as
        PENDING just because a WebSocket did not deliver."""
        from orchestrator.runner import run_to_completion

        run_id = run_to_completion(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "x"}
        )
        report = client.get(f"/api/runs/{run_id}", headers=auth).json()

        assert report["run"]["status"] == RUN_SUCCEEDED
        assert report["result"]
